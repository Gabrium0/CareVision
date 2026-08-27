"""Unit tests for core/pipeline.py's Round-3 fast-path fixes:
1. Geometry staleness bound: `Pipeline._fast_hook` must stop feeding fast
   modules once the published face is older than `max_staleness`, and resume
   once a fresh face is republished (core/pipeline.py `_fast_hook`).
2. Independent motion energy: the fast-path motion_energy fed to modules must
   be computed fresh on the reader thread, not copied from a (possibly stale)
   heavy-loop reading.

No real camera or MediaPipe needed — a fake camera exposes the same
`register_fast_hook`/`current_fps` contract core.camera.Camera provides, and
a tiny fake module records exactly what `fast_update(ctx)` receives.

Run standalone:  python tests/pipeline_staleness_test.py
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FaceData, FrameContext
from core.pipeline import Pipeline
from core.fast_face_tracker import FastFaceTracker
from core.scheduler import Scheduler


class FakeCamera:
    """Mimics Camera's register_fast_hook()/current_fps contract."""
    def __init__(self):
        self._hooks = []
        self._capture_reset_hooks = []
        self._fps = 30.0

    @property
    def current_fps(self):
        return self._fps

    def register_fast_hook(self, hook):
        self._hooks.append(hook)

    def register_capture_reset_hook(self, hook):
        self._capture_reset_hooks.append(hook)

    def fire(self, frame, ts):
        for h in self._hooks:
            h(frame, ts)


class RecordingModule:
    """Fake scheduled module: records every fast_update() call. Also
    implements the normal DetectionModule.process() contract (a no-op here)
    since core.scheduler.Scheduler.tick() calls it every heavy-loop tick
    regardless of whether we care about that path in these tests."""
    name = "recorder"
    interval = 0.0
    requires = ()

    def __init__(self):
        self.calls: list[FrameContext] = []
        self.reset_count = 0

    def process(self, ctx: FrameContext):
        return None

    def fast_update(self, ctx: FrameContext) -> None:
        self.calls.append(ctx)

    def reset_capture(self) -> None:
        self.reset_count += 1


class DummyAggregator:
    def ingest(self, results):
        pass


def _frame(value: int = 100) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def _face(bbox=(10, 10, 30, 30)) -> FaceData:
    return FaceData(landmarks=np.zeros((478, 3), dtype=np.float32), bbox=bbox,
                    crop=_frame()[bbox[1]:bbox[3], bbox[0]:bbox[2]], has_iris=True)


def _build_pipeline(max_staleness: float):
    module = RecordingModule()
    scheduler = Scheduler([module])
    camera = FakeCamera()
    pipeline = Pipeline(camera, extractors=[], scheduler=scheduler,
                        aggregator=DummyAggregator(), max_staleness=max_staleness)
    assert pipeline._fast_modules == [module]
    return pipeline, camera, module


def _wait_for_calls(module, count: int, timeout: float = 1.0):
    deadline = time.time() + timeout
    while len(module.calls) < count and time.time() < deadline:
        time.sleep(.005)


def _capture_state(reason: str | None, ready: bool) -> dict:
    return {
        "stable": ready,
        "heart_rate_ready": ready,
        "block_reason": reason,
        "heart_rate_block_reason": reason,
        "guidance": "steady" if ready else "adjusting",
    }


def _gate_context(timestamp: float, reason: str | None, ready: bool) -> FrameContext:
    ctx = FrameContext(_frame(), timestamp, 0, 30.0)
    ctx.extras["showcase"] = _capture_state(reason, ready)
    return ctx


def _build_gated_pipeline():
    module = RecordingModule()
    camera = FakeCamera()
    gate = SimpleNamespace(capture_reset_grace_seconds=1.0, max_motion=12.0)
    pipeline = Pipeline(camera, [], Scheduler([module]), DummyAggregator(),
                        showcase_gate=gate)
    assert camera._capture_reset_hooks == [pipeline.reset_capture_state]
    return pipeline, module


def test_brief_face_loss_preserves_rppg_buffers():
    pipeline, module = _build_gated_pipeline()
    pipeline._update_capture_gate(_gate_context(10.0, None, True))
    pipeline._update_capture_gate(_gate_context(10.2, "no_face", False))
    pipeline._update_capture_gate(_gate_context(10.8, None, True))
    assert module.reset_count == 0


def test_prolonged_face_loss_resets_rppg_once_after_grace():
    pipeline, module = _build_gated_pipeline()
    pipeline._update_capture_gate(_gate_context(10.0, None, True))
    pipeline._update_capture_gate(_gate_context(10.2, "no_face", False))
    pipeline._update_capture_gate(_gate_context(11.1, "no_face", False))
    assert module.reset_count == 0
    pipeline._update_capture_gate(_gate_context(11.21, "no_face", False))
    pipeline._update_capture_gate(_gate_context(12.5, "no_face", False))
    assert module.reset_count == 1


def test_multiple_people_reset_rppg_immediately():
    pipeline, module = _build_gated_pipeline()
    pipeline._update_capture_gate(_gate_context(10.0, None, True))
    pipeline._update_capture_gate(_gate_context(10.01, "multiple", False))
    assert module.reset_count == 1


def _textured_frame(shift_x: int = 0) -> np.ndarray:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for y in range(15, 105, 10):
        for x in range(35, 125, 10):
            xx = x + shift_x
            if 1 <= xx < 159:
                frame[y - 2:y + 3, xx - 2:xx + 3] = (80 + x, 180, 120)
    return frame


def _tracked_face() -> FaceData:
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[:, 0] = 0.5
    landmarks[:, 1] = 0.5
    frame = _textured_frame()
    return FaceData(landmarks, (30, 10, 130, 110), frame[10:110, 30:130], True)


def test_fresh_geometry_feeds_module():
    """Firing the fast hook shortly after a face is published must feed."""
    pipeline, camera, module = _build_pipeline(max_staleness=0.25)
    ctx = FrameContext(frame=_frame(), timestamp=10.0, frame_index=0, fps=30.0, face=_face())
    pipeline.process_frame(ctx)          # publishes _latest_face at ts=10.0

    camera.fire(_frame(), 10.05)         # 50ms later: well within 250ms staleness
    _wait_for_calls(module, 1)
    assert len(module.calls) == 1, "fresh geometry should have fed the module"
    pipeline._stop_fast_sampler()
    print("[pipeline-staleness-test] fresh geometry feeds OK")


def test_stale_geometry_pauses_feeding():
    """Once published geometry exceeds max_staleness, feeding must stop —
    this is the exact bug that corrupted vitals: reusing a frozen bbox past
    the point a real person could have moved out from under it."""
    pipeline, camera, module = _build_pipeline(max_staleness=0.25)
    ctx = FrameContext(frame=_frame(), timestamp=10.0, frame_index=0, fps=30.0, face=_face())
    pipeline.process_frame(ctx)          # publishes _latest_face at ts=10.0

    camera.fire(_frame(), 10.05)         # fresh: feeds
    _wait_for_calls(module, 1)
    camera.fire(_frame(), 10.60)         # 600ms later: stale, must NOT feed
    time.sleep(.03)
    assert len(module.calls) == 1, (
        f"stale geometry should not have fed the module, got {len(module.calls)} calls")
    print("[pipeline-staleness-test] stale geometry pauses feeding OK")
    pipeline._stop_fast_sampler()


def test_refreshed_geometry_resumes_feeding():
    """A new heavy-loop detection must immediately re-enable feeding."""
    pipeline, camera, module = _build_pipeline(max_staleness=0.25)
    ctx1 = FrameContext(frame=_frame(), timestamp=10.0, frame_index=0, fps=30.0, face=_face())
    pipeline.process_frame(ctx1)
    camera.fire(_frame(), 10.60)         # stale, no feed
    time.sleep(.03)
    assert len(module.calls) == 0

    ctx2 = FrameContext(frame=_frame(), timestamp=10.60, frame_index=1, fps=30.0, face=_face())
    pipeline.process_frame(ctx2)         # heavy loop republishes a fresh face
    camera.fire(_frame(), 10.62)         # fresh again: must feed
    _wait_for_calls(module, 1)
    assert len(module.calls) == 1
    print("[pipeline-staleness-test] refreshed geometry resumes feeding OK")
    pipeline._stop_fast_sampler()


def test_motion_energy_is_computed_fresh_not_copied():
    """The fast-path motion_energy must come from Pipeline's own frame-diff
    on the reader thread, not from whatever the heavy loop last saw — the
    stale value must NOT leak through even though the heavy-loop ctx carries
    an unrelated (obviously-wrong) motion_energy."""
    pipeline, camera, module = _build_pipeline(max_staleness=1.0)
    heavy_ctx = FrameContext(frame=_frame(50), timestamp=10.0, frame_index=0, fps=30.0,
                             face=_face(), motion_energy=999.0)   # deliberately implausible
    pipeline.process_frame(heavy_ctx)

    camera.fire(_frame(50), 10.01)       # identical frame -> first diff is baseline (0.0)
    _wait_for_calls(module, 1)
    camera.fire(_frame(200), 10.02)      # very different frame -> real nonzero motion
    _wait_for_calls(module, 2)

    assert len(module.calls) == 2
    first_motion = module.calls[0].motion_energy
    second_motion = module.calls[1].motion_energy
    assert first_motion != 999.0, "must not copy the heavy loop's stale motion_energy"
    assert second_motion > first_motion, (
        f"motion energy should rise for a genuinely different frame: "
        f"{first_motion} -> {second_motion}")
    print(f"[pipeline-staleness-test] motion energy computed fresh OK "
          f"({first_motion:.1f} -> {second_motion:.1f}, heavy-loop stale was 999.0)")
    pipeline._stop_fast_sampler()


def test_extended_mode_bridges_slow_face_publication_but_rejects_motion():
    module = RecordingModule()
    camera = FakeCamera()
    pipeline = Pipeline(camera, [], Scheduler([module]), DummyAggregator(),
                        max_staleness=0.25, fast_path_mode="extended")
    ctx = FrameContext(_textured_frame(), 10.0, 0, 30.0, face=_tracked_face())
    pipeline.process_frame(ctx)
    camera.fire(_textured_frame(), 10.60)
    _wait_for_calls(module, 1)
    assert len(module.calls) == 1
    camera.fire(np.full_like(_textured_frame(), 255), 10.63)
    time.sleep(.03)
    assert len(module.calls) == 1
    pipeline._stop_fast_sampler()


def test_tracker_moves_roi_and_expires_without_anchor():
    tracker = FastFaceTracker(max_anchor_age=0.5)
    assert tracker.seed(_textured_frame(), _tracked_face(), 10.0)
    result = tracker.track(_textured_frame(shift_x=2), 10.03)
    assert result.face is not None
    assert result.face.bbox[0] >= 30
    expired = tracker.track(_textured_frame(shift_x=2), 10.6)
    assert expired.face is None
    assert expired.reason == "anchor expired"


def test_tracker_rejects_featureless_anchor():
    tracker = FastFaceTracker()
    assert not tracker.seed(np.zeros((120, 160, 3), dtype=np.uint8),
                            _tracked_face(), 10.0)
    assert tracker.diagnostics(10.0)["last_reason"] == "insufficient anchor features"


def test_tracked_mode_feeds_at_camera_rate_on_stable_frames():
    module = RecordingModule()
    camera = FakeCamera()
    pipeline = Pipeline(camera, [], Scheduler([module]), DummyAggregator(),
                        max_staleness=0.25, fast_path_mode="tracked")
    pipeline.process_frame(FrameContext(_textured_frame(), 10.0, 0, 30.0,
                                        face=_tracked_face()))
    for index in range(1, 31):
        camera.fire(_textured_frame(), 10.0 + index / 30.0)
        time.sleep(1 / 30.0)
    _wait_for_calls(module, 20)
    assert len(module.calls) >= 20
    diagnostics = pipeline.vitals_diagnostics([], now=11.0)
    assert diagnostics["fast_path"]["mode"] == "tracked"
    assert diagnostics["fast_path"]["tracker"]["tracked_sample_hz"] >= 20.0
    pipeline._stop_fast_sampler()


def main():
    """Run all pipeline staleness/motion-energy tests."""
    test_fresh_geometry_feeds_module()
    test_stale_geometry_pauses_feeding()
    test_refreshed_geometry_resumes_feeding()
    test_motion_energy_is_computed_fresh_not_copied()
    test_extended_mode_bridges_slow_face_publication_but_rejects_motion()
    test_tracker_moves_roi_and_expires_without_anchor()
    test_tracker_rejects_featureless_anchor()
    test_tracked_mode_feeds_at_camera_rate_on_stable_frames()
    print("[pipeline-staleness-test] OK")


if __name__ == "__main__":
    main()
