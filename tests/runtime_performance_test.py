"""Focused contracts for decoupled metrics, RealSense profiles, and debug state."""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FaceData, FrameContext
from core.events import Result, Visibility
from core.pipeline import Pipeline
from core.realsense_camera import RealSenseCamera
from core.runtime_metrics import RuntimeMetrics
from core.scheduler import Scheduler
from webui.debug_server import DebugServer, build_debug_payload, safe_json


def test_runtime_metrics_separate_capture_preview_and_analysis():
    metrics = RuntimeMetrics()
    base = 100.0
    for i in range(31):
        metrics.note_capture(30.0, base + i / 30.0)
        metrics.note_preview(base + i / 30.0)
        if i % 3 == 0:
            metrics.note_analysis(i, 55.0, base + i / 30.0)
    snap = metrics.snapshot(now=base + 1.0)
    assert snap["capture_fps"] == 30.0
    assert 29.0 <= snap["preview_fps"] <= 31.0
    assert 9.0 <= snap["analysis_fps"] <= 11.0
    assert snap["skipped_analysis_frames"] == 20
    metrics.note_fast_path("stale", now=base + 1.0)
    fast = metrics.snapshot(now=base + 1.25)["fast_path"]
    assert fast["latest_outcome"] == "stale"
    assert fast["latest_outcome_age_ms"] == 250.0


def test_blocking_detector_cannot_delay_critical_face_publication():
    class Camera:
        current_fps = 30.0

        def register_fast_hook(self, hook):
            self.fast_hook = hook

    class FaceExtractor:
        def extract(self, ctx):
            crop = ctx.frame[10:130, 10:130]
            ctx.face = FaceData(np.zeros((5, 3)), (10, 10, 130, 130), crop, False)
            ctx.person_present = True

    class Critical:
        name = "heart_rate"
        calls = 0

        def fast_update(self, ctx):
            pass

        def process(self, ctx):
            self.calls += 1

    class Blocking:
        name = "synthetic_blocker"

        def __init__(self):
            self.started = threading.Event()

        def process(self, ctx):
            self.started.set()
            time.sleep(0.3)

    class Aggregator:
        def ingest(self, results):
            pass

        def snapshot(self):
            return {}

    critical, blocker = Critical(), Blocking()
    pipeline = Pipeline(Camera(), [FaceExtractor()],
                        Scheduler([critical, blocker]), Aggregator(),
                        background_analysis=True)
    pipeline._start_background_worker()
    try:
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        pipeline.process_frame(FrameContext(frame, 1.0, 0, 30.0))
        assert blocker.started.wait(1.0)
        started = time.perf_counter()
        pipeline.process_frame(FrameContext(frame, 1.1, 1, 30.0))
        assert time.perf_counter() - started < 0.1
        assert critical.calls == 2
        assert pipeline._latest_face_ts == 1.1
    finally:
        pipeline._stop_background_worker()


def test_module_start_runs_before_background_frame_analysis():
    started = threading.Event()
    extracted = threading.Event()

    class Camera:
        current_fps = 30.0

        def frames(self):
            yield FrameContext(np.zeros((32, 32, 3), dtype=np.uint8), 1.0, 0, 30.0)

        def release(self):
            pass

    class WarmModule:
        name = "warm_module"
        interval = 0.0
        requires = ()

        def start(self):
            started.set()

        def process(self, ctx):
            return None

    class BackgroundExtractor:
        def extract(self, ctx):
            assert started.is_set()
            extracted.set()

    class Aggregator:
        def ingest(self, results):
            pass

        def snapshot(self):
            return {}

    pipeline = Pipeline(Camera(), [BackgroundExtractor()],
                        Scheduler([WarmModule()]), Aggregator(),
                        background_analysis=True)
    pipeline.run(on_frame=lambda ctx, results: extracted.wait(1.0), max_frames=1)

    assert started.is_set()
    assert extracted.is_set()


def test_shutdown_waits_for_background_extractor_before_closing_resources():
    entered = threading.Event()
    release = threading.Event()

    class Camera:
        current_fps = 30.0

        def frames(self):
            yield FrameContext(np.zeros((32, 32, 3), dtype=np.uint8), 1.0, 0, 30.0)

        def release(self):
            pass

    class BlockingExtractor:
        def __init__(self):
            self.closed = False

        def extract(self, ctx):
            entered.set()
            release.wait()
            assert not self.closed

        def close(self):
            self.closed = True

    class ShouldNotRun:
        name = "should_not_run"
        interval = 0.0
        requires = ()

        def __init__(self):
            self.calls = 0

        def process(self, ctx):
            self.calls += 1

    class Aggregator:
        def ingest(self, results):
            pass

        def snapshot(self):
            return {}

    extractor = BlockingExtractor()
    module = ShouldNotRun()
    pipeline = Pipeline(Camera(), [extractor], Scheduler([module]), Aggregator(),
                        background_analysis=True)
    runner = threading.Thread(
        target=lambda: pipeline.run(
            on_frame=lambda ctx, results: entered.wait(1.0) and False),
        daemon=True)
    runner.start()
    assert entered.wait(1.0)
    time.sleep(0.05)
    assert runner.is_alive()
    assert not extractor.closed

    release.set()
    runner.join(2.0)

    assert not runner.is_alive()
    assert extractor.closed
    assert module.calls == 0


def test_scheduler_stop_predicate_prevents_later_module_submission():
    stop = threading.Event()
    calls = []

    class First:
        name = "first"
        interval = 0.0
        requires = ()

        def process(self, ctx):
            calls.append("first")
            stop.set()

    class Second:
        name = "second"
        interval = 0.0
        requires = ()

        def process(self, ctx):
            calls.append("second")

    ctx = FrameContext(np.zeros((8, 8, 3), dtype=np.uint8), 1.0, 0, 30.0)
    Scheduler([First(), Second()]).tick(ctx, should_stop=stop.is_set)

    assert calls == ["first"]


def test_realsense_profile_ranking_prefers_depth_then_resolution():
    colors = {(1920, 1080, 30), (1280, 720, 30), (640, 480, 30)}
    depths = {(848, 480, 30), (640, 480, 30)}
    ranked = RealSenseCamera.rank_profile_candidates(
        colors, depths, (1920, 1080), 30, auto_resolution=True)
    assert ranked[0][:3] == ("color+depth", 1920, 1080)
    assert ranked[0][3:5] == (848, 480)
    assert [r[0] for r in ranked[:3]] == ["color+depth"] * 3
    assert ranked[3][:3] == ("color-only", 1920, 1080)


def test_realsense_explicit_resolution_does_not_expand_candidates():
    ranked = RealSenseCamera.rank_profile_candidates(
        {(1920, 1080, 30), (1280, 720, 30)}, {(640, 480, 30)},
        (1280, 720), 30, auto_resolution=False)
    assert ranked
    assert all(item[1:3] == (1280, 720) for item in ranked)


def test_private_debug_payload_includes_private_and_redacts_media():
    private = Result("agent", "hypothesis", np.float32(72.5),
                     visibility=Visibility.AGENT_ONLY,
                     conversation_tags=("private",))
    payload = build_debug_payload([private], {"capture_fps": 30.0},
                                  {"frame": np.zeros((2, 2)),
                                   "url": "data:image/jpeg;base64,AAAA"})
    encoded = json.dumps(payload)
    assert '"visibility": "agent_only"' in encoded
    assert '"value": 72.5' in encoded
    assert "AAAA" not in encoded
    assert encoded.count("<redacted-media>") == 2
    assert safe_json(float("nan")) is None


def test_private_debug_payload_includes_nvidia_skin_diagnostics():
    diagnostic = {
        "available": True, "status": "success",
        "last_attempt": {
            "stage": "preliminary", "raw_model_content": '{"finding_present":false}',
        },
    }
    payload = build_debug_payload([], {"capture_fps": 30.0},
                                  {"nvidia_skin": diagnostic})
    assert payload["system"]["nvidia_skin"] == diagnostic


def test_debug_server_binds_loopback_and_serves_readable_private_dashboard():
    server = DebugServer(lambda: {"ok": True}, port=0)
    server.start()
    try:
        host, port = server._httpd.server_address
        assert host == "127.0.0.1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/debug") as response:
            html = response.read().decode("utf-8")
            assert response.headers["Cache-Control"] == "no-store"
            assert "Care Monitor — Private Debug" in html
            assert "fetch('/debug/state'" in html
            assert "textContent" in html
            assert "renderVitals" in html
            assert "Vitals diagnostics unavailable" in html
            assert "NVIDIA skin VLM" in html
            assert "Latest private request diagnostics" in html
            assert "capturePanelState" in html
            assert "restorePanelState" in html
            assert "scrollTop" in html
            assert "selectionchange" in html
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/debug/state") as response:
            assert json.load(response) == {"ok": True}
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/data")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("private server must not serve companion routes")
    finally:
        server.stop()
