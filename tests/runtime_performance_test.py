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

from core.context import FaceData, FrameContext, Intrinsics, PoseData
from core.events import Result, Visibility
from core.pipeline import Pipeline
from core.realsense_camera import RealSenseCamera
from core.runtime_metrics import RuntimeMetrics
from core.runtime_resources import apply_loaded_limits
from core.scheduler import Scheduler
from modules._util import native_detail_context
from webui.debug_server import (DebugServer, build_audio_debug_state,
                                build_debug_payload, safe_json)


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
    metrics.note_fast_sampler(base + 1.0, 12.0)
    metrics.note_fast_sampler(base + 1.1, 20.0)
    metrics.note_fast_sampler_drop()
    sampler = metrics.snapshot(now=base + 1.25)["fast_sampler"]
    assert sampler["fps"] == 10.0
    assert sampler["queue_drops"] == 1
    metrics.note_fast_sampler_coalesced(2)
    assert metrics.snapshot()["fast_sampler"]["intentional_coalescing"] == 2
    assert sampler["latency_ms"]["max"] == 20.0
    metrics.set_face_worker_alive(True)
    metrics.note_face_worker(15.0, 480)
    metrics.note_face_worker_drop()
    face_worker = metrics.snapshot()["face_worker"]
    assert face_worker["alive"] is True
    assert face_worker["adaptive_width"] == 480
    assert face_worker["queue_replacements"] == 1


def test_camera_fast_hook_is_nonblocking_and_sampler_drops_oldest():
    class Camera:
        current_fps = 30.0
        def register_fast_hook(self, hook): self.fast_hook = hook

    class SlowVitals:
        name = "heart_rate"
        interval = 0.0
        requires = ()
        def __init__(self): self.timestamps = []
        def process(self, _ctx): return None
        def fast_update(self, ctx):
            self.timestamps.append(ctx.timestamp)
            time.sleep(.04)

    class Aggregator:
        def ingest(self, _results): pass

    camera, module = Camera(), SlowVitals()
    pipeline = Pipeline(camera, [], Scheduler([module]), Aggregator())
    frame = np.zeros((48, 64, 3), np.uint8)
    with pipeline._face_lock:
        pipeline._latest_face = FaceData(
            np.zeros((5, 3)), (5, 5, 40, 40), frame[5:40, 5:40], False)
        pipeline._latest_face_ts = 10.0
    started = time.perf_counter()
    for index in range(12):
        camera.fast_hook(frame, 10.0 + index / 100.0)
    enqueue_elapsed = time.perf_counter() - started
    deadline = time.time() + 2.0
    while time.time() < deadline and pipeline._fast_pending:
        time.sleep(.01)
    pipeline._stop_fast_sampler()
    assert enqueue_elapsed < .05
    assert module.timestamps == sorted(module.timestamps)
    assert pipeline.runtime_metrics.snapshot()["fast_sampler"]["queue_drops"] > 0


def test_background_submission_defers_pixel_scaling_to_worker():
    class Camera:
        current_fps = 30.0
        def register_fast_hook(self, _hook): pass
    frame = np.zeros((1080, 1920, 3), np.uint8)
    pipeline = Pipeline(Camera(), [], Scheduler([]), object(),
                        background_analysis=True, analysis_width=960)
    ctx = FrameContext(frame, 1.0, 0, 30.0)
    pipeline._submit_background(ctx)
    with pipeline._background_cv:
        pending = pipeline._background_pending
    assert pending is not None and pending.frame.shape == frame.shape
    pipeline._submit_background(FrameContext(frame, 1.1, 1, 30.0))
    state = pipeline.runtime_metrics.snapshot(now=time.time())
    assert state["background_coalesced_frames"] == 1
    assert state["background_queue_drops"] == 0


def test_background_context_scales_geometry_depth_and_intrinsics_only():
    class Camera:
        current_fps = 30.0
        def register_fast_hook(self, _hook): pass
    frame = np.zeros((1080, 1920, 3), np.uint8)
    depth = np.full((1080, 1920), 1000, np.uint16)
    landmarks = np.zeros((5, 3), np.float32)
    landmarks[:, :2] = (.5, .5)
    face = FaceData(landmarks, (480, 270, 1440, 810), frame[270:810, 480:1440], False)
    ctx = FrameContext(frame, 1.0, 0, 30.0, face=face, depth=depth,
                       intrinsics=Intrinsics(1000, 1000, 960, 540))
    pipeline = Pipeline(Camera(), [], Scheduler([]), object(),
                        background_analysis=True, analysis_width=960)
    scaled = pipeline._copy_for_background(ctx)
    assert ctx.frame.shape == (1080, 1920, 3)
    assert scaled.frame.shape == (540, 960, 3)
    assert scaled.depth.shape == (540, 960)
    assert scaled.face.bbox == (240, 135, 720, 405)
    assert scaled.intrinsics == Intrinsics(500, 500, 480, 270)
    assert np.array_equal(scaled.face.landmarks, landmarks)
    assert scaled.extras["_native_detail_context"] is ctx
    assert scaled.extras["quality_profile"] == "maximum"
    assert scaled.extras["detail_roi_cap"] == 640
    pose_landmarks = np.zeros((33, 4), np.float32)
    scaled.pose = PoseData(pose_landmarks, (100, 100, 800, 500))
    detail = native_detail_context(scaled)
    assert detail.frame.shape == frame.shape
    assert detail.pose.bbox == (200, 200, 1600, 1000)
    pipeline._stop_background_worker()


def test_scheduler_adaptively_throttles_passive_but_not_safety_modules():
    calls = []
    class Module:
        interval = 0.0
        requires = ()
        def __init__(self, name): self.name = name
        def process(self, _ctx): calls.append(self.name)
    scheduler = Scheduler([Module("rash"), Module("fall")])
    scheduler.set_load_factor(8.0)
    frame = np.zeros((8, 8, 3), np.uint8)
    scheduler.tick(FrameContext(frame, 1.0, 0, 30.0))
    scheduler.tick(FrameContext(frame, 1.1, 1, 30.0))
    assert calls.count("fall") == 2
    assert calls.count("rash") == 1
    assert scheduler.pop_throttled() >= 1


def test_scheduler_budget_defers_passive_but_always_runs_safety():
    calls = []
    class Module:
        interval = 0.0
        requires = ()
        def __init__(self, name): self.name = name
        def process(self, _ctx):
            calls.append(self.name)
            if self.name == "rash":
                time.sleep(.01)
    scheduler = Scheduler([Module("rash"), Module("bruise"), Module("fall")])
    ctx = FrameContext(np.zeros((4, 4, 3), np.uint8), 1.0, 0, 30.0)
    scheduler.tick(ctx, budget_ms=1.0)
    assert "fall" in calls
    assert scheduler.pop_throttled() >= 1


def test_maximum_quality_profile_applies_bounded_native_threads(monkeypatch):
    monkeypatch.delenv("APP_RESPECT_NATIVE_THREAD_ENV", raising=False)
    state = apply_loaded_limits("maximum")
    assert state["profile"] == "maximum"
    assert int(state["environment"]["OMP_NUM_THREADS"]) <= 2
    assert state["status"] == "healthy"


def test_runtime_health_reports_rates_percentiles_and_overload():
    metrics = RuntimeMetrics()
    for i in range(8):
        now = 100.0 + i * .2
        metrics.note_critical(600.0, now - .4, now=now)
        metrics.note_background_drop()
    state = metrics.snapshot(now=101.5)
    assert state["health"]["status"] == "degraded"
    assert "critical_latency_high" in state["health"]["reasons"]
    assert state["latency_distributions_ms"]["critical"]["p95"] == 600.0
    assert state["fast_path"]["acceptance_ratio"] == 0.0


def test_debug_health_marks_latched_cloud_authorization_as_degraded():
    payload = build_debug_payload([], {"health": {"status": "healthy", "reasons": [],
                                                   "actions": []}},
                                  {"moondream": {"available": True, "enabled": True,
                                                 "authorization_failed": True,
                                                 "circuit_state": "authorization_failed"}})
    assert payload["health"]["components"]["moondream"] == "degraded"
    assert "moondream_degraded" in payload["health"]["reasons"]


def test_optional_nvidia_error_degrades_but_does_not_fail_runtime():
    payload = build_debug_payload(
        [], {"health": {"status": "healthy", "reasons": [], "actions": []}},
        {"nvidia_skin": {"available": True, "consent": True,
                         "status": "error", "last_attempt": {"error": "TimeoutError"}},
         "history_writer": {"alive": True, "failures": 0, "dropped": 0}})
    assert payload["health"]["status"] == "degraded"
    assert payload["health"]["components"]["nvidia_skin"] == "degraded"
    assert "nvidia_skin_degraded" in payload["health"]["reasons"]


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
    pipeline._start_face_worker()
    try:
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        pipeline.process_frame(FrameContext(frame, 1.0, 0, 30.0))
        deadline = time.time() + 1.0
        while pipeline._face_result is None and time.time() < deadline:
            time.sleep(.01)
        assert blocker.started.wait(1.0)
        started = time.perf_counter()
        pipeline.process_frame(FrameContext(frame, 1.1, 1, 30.0))
        assert time.perf_counter() - started < 0.1
        assert critical.calls == 2
        assert pipeline._latest_face_ts == 1.0
    finally:
        pipeline._stop_face_worker()
        pipeline._stop_background_worker()


def test_blocking_advisor_runs_on_background_lane():
    advisor_started = threading.Event()
    advisor_release = threading.Event()

    class Camera:
        current_fps = 30.0

        def register_fast_hook(self, hook):
            self.fast_hook = hook

    class FaceExtractor:
        def extract(self, ctx):
            crop = ctx.frame[10:130, 10:130]
            ctx.face = FaceData(np.zeros((5, 3)), (10, 10, 130, 130), crop, False)

    class BackgroundModule:
        name = "background"
        interval = 0.0
        requires = ()

        def process(self, ctx):
            return Result("background", "value", 1.0, timestamp=ctx.timestamp)

    class Critical:
        name = "heart_rate"
        interval = 0.0
        requires = ()

        def fast_update(self, ctx):
            pass

        def process(self, ctx):
            return None

    class BlockingAdvisor:
        def evaluate(self, snapshot, now=None):
            advisor_started.set()
            advisor_release.wait(1.0)
            return [Result("advice", "value", True, timestamp=now)]

    class Aggregator:
        def ingest(self, results):
            pass

        def snapshot(self):
            return []

    pipeline = Pipeline(Camera(), [FaceExtractor()],
                        Scheduler([Critical(), BackgroundModule()]),
                        Aggregator(), advisor_engine=BlockingAdvisor(),
                        background_analysis=True)
    pipeline._start_background_worker()
    pipeline._start_face_worker()
    try:
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        pipeline.process_frame(FrameContext(frame, 1.0, 0, 30.0))
        deadline = time.time() + 1.0
        while pipeline._face_result is None and time.time() < deadline:
            time.sleep(.01)
        pipeline.process_frame(FrameContext(frame, 1.1, 1, 30.0))
        assert advisor_started.wait(1.0)
        started = time.perf_counter()
        pipeline.process_frame(FrameContext(frame, 1.2, 2, 30.0))
        assert time.perf_counter() - started < 0.1
        assert pipeline._latest_face_ts >= 1.0
        advisor_release.set()
        deadline = time.time() + 1.0
        while time.time() < deadline and not pipeline._background_done:
            time.sleep(.01)
        drained = pipeline._drain_background()
        assert any(result.module == "advice" for result in drained)
    finally:
        advisor_release.set()
        pipeline._stop_face_worker()
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
        "last_attempt": {"stage": "preliminary", "validation": None},
    }
    payload = build_debug_payload([], {"capture_fps": 30.0},
                                  {"nvidia_skin": diagnostic})
    assert payload["system"]["nvidia_skin"] == diagnostic


def test_audio_debug_state_reports_ready_disabled_and_unavailable():
    class Registry:
        def __init__(self, microphone, yamnet):
            self.items = {"microphone": microphone, "yamnet": yamnet}

        def get(self, name):
            return self.items.get(name)

    class Capability:
        def __init__(self, status, detail):
            self.status = status
            self.detail = detail
            self.updated_at = 123.0

    class Detector:
        def diagnostics(self):
            return {"available": True, "status": "ready", "mode": "cough_only",
                    "worker_alive": True, "windows_processed": 4,
                    "last_inference_at": 125.0,
                    "latest_cough_confidence": .42,
                    "peak_cough_confidence": .8, "threshold": .35,
                    "pending_cough_episode": True, "pending_burst_count": 1}

    ready_registry = Registry(Capability("ready", "16 kHz mono"),
                              Capability("ready", "cough-only"))
    ready = build_audio_debug_state(ready_registry, Detector(), True, "cough_only")
    assert ready["enabled"] is True
    assert ready["microphone"]["status"] == "ready"
    assert ready["detector"]["latest_cough_confidence"] == .42
    assert ready["detector"]["pending_cough_episode"] is True

    disabled_registry = Registry(Capability("unavailable", "disabled"),
                                 Capability("unavailable", "disabled"))
    disabled = build_audio_debug_state(
        disabled_registry, None, False, "broad_listening")
    assert disabled["detector"]["status"] == "disabled"

    unavailable = build_audio_debug_state(
        disabled_registry, None, True, "cough_only")
    assert unavailable["enabled"] is True
    assert unavailable["detector"]["status"] == "unavailable"

    replay = build_audio_debug_state(
        disabled_registry, None, True, "replay")
    assert replay["mode"] == "replay"
    assert replay["stt"]["status"] == "not_applicable"
    payload = build_debug_payload(
        [], {"health": {"status": "healthy", "reasons": [], "actions": []}},
        {"audio": replay})
    assert payload["health"]["components"]["audio"] == "healthy"


def test_audio_debug_payload_redacts_any_accidental_raw_samples():
    audio = {"detector": {"status": "ready",
                           "accidental_samples": np.ones(10, dtype=np.float32)}}
    payload = build_debug_payload([], {}, {"audio": audio})
    assert payload["system"]["audio"]["detector"]["accidental_samples"] == \
        "<redacted-media>"


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
            assert "Runtime health" in html
            assert "Vitals diagnostics unavailable" in html
            assert "Open-rPPG" in html
            assert "Classical" in html
            assert "backend-bpm" in html
            assert "measurement.rejection_reason" in html
            assert "Audio & cough detection" in html
            assert "renderAudio" in html
            assert "whisper_worker_alive" in html
            assert "latest_cough_confidence" in html
            assert "NVIDIA skin VLM" in html
            assert "History writer" in html
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
