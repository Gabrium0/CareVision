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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FaceData, FrameContext
from core.pipeline import Pipeline
from core.scheduler import Scheduler


class FakeCamera:
    """Mimics Camera's register_fast_hook()/current_fps contract."""
    def __init__(self):
        self._hooks = []
        self._fps = 30.0

    @property
    def current_fps(self):
        return self._fps

    def register_fast_hook(self, hook):
        self._hooks.append(hook)

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

    def process(self, ctx: FrameContext):
        return None

    def fast_update(self, ctx: FrameContext) -> None:
        self.calls.append(ctx)


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


def test_fresh_geometry_feeds_module():
    """Firing the fast hook shortly after a face is published must feed."""
    pipeline, camera, module = _build_pipeline(max_staleness=0.25)
    ctx = FrameContext(frame=_frame(), timestamp=10.0, frame_index=0, fps=30.0, face=_face())
    pipeline.process_frame(ctx)          # publishes _latest_face at ts=10.0

    camera.fire(_frame(), 10.05)         # 50ms later: well within 250ms staleness
    assert len(module.calls) == 1, "fresh geometry should have fed the module"
    print("[pipeline-staleness-test] fresh geometry feeds OK")


def test_stale_geometry_pauses_feeding():
    """Once published geometry exceeds max_staleness, feeding must stop —
    this is the exact bug that corrupted vitals: reusing a frozen bbox past
    the point a real person could have moved out from under it."""
    pipeline, camera, module = _build_pipeline(max_staleness=0.25)
    ctx = FrameContext(frame=_frame(), timestamp=10.0, frame_index=0, fps=30.0, face=_face())
    pipeline.process_frame(ctx)          # publishes _latest_face at ts=10.0

    camera.fire(_frame(), 10.05)         # fresh: feeds
    camera.fire(_frame(), 10.60)         # 600ms later: stale, must NOT feed
    assert len(module.calls) == 1, (
        f"stale geometry should not have fed the module, got {len(module.calls)} calls")
    print("[pipeline-staleness-test] stale geometry pauses feeding OK")


def test_refreshed_geometry_resumes_feeding():
    """A new heavy-loop detection must immediately re-enable feeding."""
    pipeline, camera, module = _build_pipeline(max_staleness=0.25)
    ctx1 = FrameContext(frame=_frame(), timestamp=10.0, frame_index=0, fps=30.0, face=_face())
    pipeline.process_frame(ctx1)
    camera.fire(_frame(), 10.60)         # stale, no feed
    assert len(module.calls) == 0

    ctx2 = FrameContext(frame=_frame(), timestamp=10.60, frame_index=1, fps=30.0, face=_face())
    pipeline.process_frame(ctx2)         # heavy loop republishes a fresh face
    camera.fire(_frame(), 10.62)         # fresh again: must feed
    assert len(module.calls) == 1
    print("[pipeline-staleness-test] refreshed geometry resumes feeding OK")


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
    camera.fire(_frame(200), 10.02)      # very different frame -> real nonzero motion

    assert len(module.calls) == 2
    first_motion = module.calls[0].motion_energy
    second_motion = module.calls[1].motion_energy
    assert first_motion != 999.0, "must not copy the heavy loop's stale motion_energy"
    assert second_motion > first_motion, (
        f"motion energy should rise for a genuinely different frame: "
        f"{first_motion} -> {second_motion}")
    print(f"[pipeline-staleness-test] motion energy computed fresh OK "
          f"({first_motion:.1f} -> {second_motion:.1f}, heavy-loop stale was 999.0)")


def main():
    """Run all pipeline staleness/motion-energy tests."""
    test_fresh_geometry_feeds_module()
    test_stale_geometry_pauses_feeding()
    test_refreshed_geometry_resumes_feeding()
    test_motion_energy_is_computed_fresh_not_copied()
    print("[pipeline-staleness-test] OK")


if __name__ == "__main__":
    main()
