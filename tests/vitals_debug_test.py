"""Private vitals diagnostics explain missing heart-rate values safely."""
from __future__ import annotations

import sys
import threading
import time
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import Result
from core.pipeline import Pipeline
from modules.rppg_backends.openrppg import OpenRPPGBackend, _cpu_allocation


class _Heart:
    name = "heart_rate"

    def __init__(self, backends, source="classical"):
        self.backends = backends
        self.source = source

    def diagnostics(self):
        return {"canonical_source": self.source,
                "backends": [dict(item) for item in self.backends]}


class _Metrics:
    def __init__(self, outcome="fed", age=10.0):
        self.value = {"fast_path": {
            "fed": 20, "stale": 0, "no_face": 0,
            "latest_outcome": outcome, "latest_outcome_age_ms": age}}

    def snapshot(self, now=None):
        return self.value


def _backend(name="classical", span=3.0, required=6.0, status=None,
             available=True, **extra):
    return {
        "name": name, "available": available, "samples": int(span * 10),
        "buffered_seconds": span, "required_seconds": required,
        "progress": min(1.0, span / required) if required else 0.0,
        "status": status or ("ready" if span >= required else
                              f"warming up {span:.0f}/{required:.0f}s"),
        "accepted": int(span * 10), "rejected": {},
        "inference_latency_ms": 0.0, **extra,
    }


def _pipeline(showcase, backends, outcome="fed"):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._vitals_lock = threading.Lock()
    pipeline._showcase_state = dict(showcase)
    pipeline.scheduler = SimpleNamespace(modules=[_Heart(backends)])
    pipeline.runtime_metrics = _Metrics(outcome)
    return pipeline


def test_blocked_capture_hides_even_fresh_bpm_and_uses_exact_guidance():
    pipeline = _pipeline({
        "capture_ready": False, "zone": "outside",
        "guidance": "Please move back slightly to the conversation marker."},
        [_backend(span=6.0)])
    current = Result("heart_rate", "bpm", 72.4, .88, ttl=8, timestamp=100.0)

    state = pipeline.vitals_diagnostics([current], now=101.0)

    assert state["state"] == "blocked"
    assert state["bpm"] is None
    assert state["confidence"] is None
    assert state["guidance"] == "Please move back slightly to the conversation marker."


def test_warmup_and_inference_states_report_backend_progress():
    showcase = {"capture_ready": True, "zone": "conversation", "guidance": "ready"}
    warming = _pipeline(showcase, [_backend(span=3.0)])
    state = warming.vitals_diagnostics([], now=100.0)
    assert state["state"] == "warming_up"
    assert "3.0/6.0s" in state["guidance"]
    assert state["backends"][0]["progress"] == .5
    assert state["backends"][0]["measurement"]["state"] == "warming_up"
    assert state["backends"][0]["measurement"]["bpm"] is None

    inferring = _pipeline(showcase, [
        _backend("open-rppg", 10.0, 10.0, "inferring", inference_pending=True)])
    state = inferring.vitals_diagnostics([], now=100.0)
    assert state["state"] == "inferring"
    assert "processing" in state["guidance"]
    assert state["backends"][0]["measurement"]["state"] == "inferring"


def test_ready_state_exposes_current_canonical_measurement_fields():
    backend = _backend(span=8.0, latest={"bpm": 72.4, "confidence": .88})
    pipeline = _pipeline(
        {"capture_ready": True, "zone": "conversation", "guidance": "ready"},
        [backend])
    current = Result("heart_rate", "bpm", 72.4, .88, ttl=8, timestamp=100.0,
                     source="local", quality=.91)

    state = pipeline.vitals_diagnostics([current], now=101.25)

    assert state["state"] == "ready"
    assert state["bpm"] == 72.4
    assert state["confidence"] == .88
    assert state["quality"] == .91
    assert state["source"] == "classical"
    assert state["measurement_age_seconds"] == 1.25


def test_debug_state_exposes_fresh_classical_and_rejected_openrppg_separately():
    backends = [
        _backend("classical", span=8.0,
                 latest={"bpm": 72.4, "confidence": .88}),
        _backend("open-rppg", span=10.0, status="low SQI 0.16<0.35",
                 latest={"raw_bpm": 136.0, "raw_confidence": .16,
                         "rejected_reason": "low SQI 0.16<0.35"}),
    ]
    pipeline = _pipeline(
        {"capture_ready": True, "zone": "conversation", "guidance": "ready"},
        backends)
    results = [
        Result("heart_rate", "bpm", 72.4, .88, ttl=8, timestamp=100.0),
        Result("heart_rate", "bpm_classical", 72.4, .88, ttl=8,
               timestamp=100.0, quality=.9),
        Result("heart_rate", "bpm_open_rppg", 136.0, .16, ttl=8,
               timestamp=100.0, quality=.2),
    ]

    state = pipeline.vitals_diagnostics(results, now=101.0)
    classical, openrppg = state["backends"]

    assert state["bpm"] == 72.4
    assert classical["measurement"] == {
        "bpm": 72.4, "confidence": .88, "quality": .9,
        "measurement_age_seconds": 1.0, "state": "accepted",
        "accepted": True, "rejection_reason": None}
    assert openrppg["measurement"]["bpm"] == 136.0
    assert openrppg["measurement"]["state"] == "rejected"
    assert openrppg["measurement"]["accepted"] is False
    assert "low SQI" in openrppg["measurement"]["rejection_reason"]
    assert "bpm" not in classical["latest"]
    assert "raw_bpm" not in openrppg["latest"]


def test_backend_measurements_hide_stale_and_positioning_blocked_values():
    backend = _backend("classical", span=8.0,
                       latest={"bpm": 72.4, "confidence": .88})
    reading = Result("heart_rate", "bpm_classical", 72.4, .88,
                     ttl=8, timestamp=90.0)
    ready = {"capture_ready": True, "zone": "conversation", "guidance": "ready"}
    stale = _pipeline(ready, [backend]).vitals_diagnostics([reading], now=101.0)
    assert stale["backends"][0]["measurement"]["bpm"] is None
    assert stale["backends"][0]["measurement"]["state"] == "stale"

    blocked = _pipeline({"capture_ready": False, "zone": "outside",
                         "guidance": "Move into position"}, [backend]) \
        .vitals_diagnostics([Result("heart_rate", "bpm_classical", 72.4, .88,
                                    ttl=8, timestamp=100.0)], now=101.0)
    assert blocked["backends"][0]["measurement"]["bpm"] is None
    assert blocked["backends"][0]["measurement"]["state"] == "blocked"


def test_expired_measurement_is_not_presented_as_current():
    pipeline = _pipeline(
        {"capture_ready": True, "zone": "conversation", "guidance": "ready"},
        [_backend(span=6.0)])
    expired = Result("heart_rate", "bpm", 72.4, .88, ttl=8, timestamp=90.0)
    state = pipeline.vitals_diagnostics([expired], now=101.0)
    assert state["state"] == "warming_up"
    assert state["bpm"] is None
    assert state["measurement_age_seconds"] is None


def test_unavailable_and_fast_path_starvation_are_readable():
    showcase = {"capture_ready": True, "zone": "conversation", "guidance": "ready"}
    unavailable = _pipeline(showcase, [
        _backend("open-rppg", 0, 10, "dependency unavailable", available=False)])
    unavailable_state = unavailable.vitals_diagnostics([], now=100)
    assert unavailable_state["state"] == "unavailable"
    assert unavailable_state["backends"][0]["measurement"]["state"] == "unavailable"

    no_face = _pipeline(showcase, [_backend()], outcome="no_face")
    assert "No current face" in no_face.vitals_diagnostics([], now=100)["guidance"]
    stale = _pipeline(showcase, [_backend()], outcome="stale")
    assert "stale" in stale.vitals_diagnostics([], now=100)["guidance"]


def test_backend_rejection_and_failure_statuses_survive_composition():
    showcase = {"capture_ready": True, "zone": "conversation", "guidance": "ready"}
    statuses = ["motion rejected (22>18)", "face jitter rejected",
                "warming up 4/10s (low light)", "inference failed: RuntimeError"]
    expected_states = ["rejected", "rejected", "warming_up", "failed"]
    for status, expected_state in zip(statuses, expected_states):
        pipeline = _pipeline(showcase, [_backend("open-rppg", status=status)])
        state = pipeline.vitals_diagnostics([], now=100)
        assert state["backends"][0]["status"] == status
        assert state["backends"][0]["measurement"]["state"] == expected_state


def test_openrppg_inference_failure_sets_readable_status():
    backend = OpenRPPGBackend.__new__(OpenRPPGBackend)
    backend._lock = threading.Lock()
    backend._cached = None
    backend._last_latency_ms = 0.0
    backend._status = "inferring"

    def fail(_tensor, _fps):
        raise RuntimeError("synthetic failure")

    backend._infer = fail
    assert backend._compute_tensor(object(), 30.0, 10.0) is None
    assert backend._status == "inference failed: RuntimeError"


def test_openrppg_auto_cpu_budget_and_cached_freshness():
    reserved, worker = _cpu_allocation("auto")
    assert reserved + worker >= 1
    if reserved + worker > 1:
        assert reserved >= 1 and worker >= 1

    backend = OpenRPPGBackend.__new__(OpenRPPGBackend)
    backend._cached = {"bpm": 72.0}
    backend.result_fresh_seconds = 10.0
    backend._last_completed_at = time.time() - 11.0
    assert backend._fresh_cached() is None
    backend._last_completed_at = time.time()
    assert backend._fresh_cached() == {"bpm": 72.0}


def test_openrppg_uniform_sampling_bounds_cpu_tensor_rate():
    timestamps = np.arange(0.0, 10.0, 1.0 / 30.0)
    indices = OpenRPPGBackend._uniform_indices(timestamps, 8.0)
    selected = timestamps[indices]
    assert 75 <= len(indices) <= 81
    assert len(indices) == len(set(indices.tolist()))
    effective_hz = (len(selected) - 1) / (selected[-1] - selected[0])
    assert 7.5 <= effective_hz <= 8.5


def test_openrppg_reacquires_after_three_stable_candidate_boxes():
    backend = OpenRPPGBackend.__new__(OpenRPPGBackend)
    backend.face_jitter_threshold = 0.1
    backend.face_reacquire_frames = 3
    backend._last_bbox = (0, 0, 100, 100)
    backend._candidate_bbox = None
    backend._candidate_count = 0
    moved = (40, 0, 140, 100)
    assert backend._stable_face(moved) is False
    assert backend._stable_face(moved) is False
    assert backend._stable_face(moved) is True
    assert backend._last_bbox == moved


def test_openrppg_reset_invalidates_but_keeps_running_generation_tracked():
    from collections import deque
    backend = OpenRPPGBackend.__new__(OpenRPPGBackend)
    backend._lock = threading.Lock()
    backend.ts = deque([1.0, 2.0])
    backend.crops = deque([object(), object()])
    backend._cached = {"bpm": 72}
    backend._status = "inferring"
    backend._low_light = False
    backend._last_bbox = (0, 0, 1, 1)
    backend._candidate_bbox = None
    backend._candidate_count = 0
    backend._bpm_history = deque([72.0])
    backend._generation = 4
    backend._job_generation = 4
    backend.reset()
    assert backend._generation == 5
    assert backend._job_generation == 4
    assert not backend.ts and not backend.crops


def test_openrppg_discards_obsolete_worker_result():
    backend = OpenRPPGBackend.__new__(OpenRPPGBackend)
    backend._proc = SimpleNamespace(is_alive=lambda: True)
    backend._out_q = queue.Queue()
    backend._out_q.put({"event": "result", "job_id": 7, "generation": 2,
                        "latency_ms": 123, "res": {"hr": 70}, "bvp": [],
                        "bts": [], "fps": 30, "span": 10})
    backend._active_job_id = 7
    backend._job_generation = 2
    backend._generation = 3
    backend._job_started_at = time.time()
    backend.inference_timeout_seconds = 90
    backend._worker_state = "ready"
    backend._last_latency_ms = 0
    backend._stale_discard_count = 0
    backend._cached = None
    backend._build_result = lambda *args: (_ for _ in ()).throw(
        AssertionError("stale result must not be built"))
    backend._drain_worker()
    assert backend._cached is None
    assert backend._stale_discard_count == 1
    assert backend._job_generation is None


def test_openrppg_restarts_after_timeout_or_unexpected_exit():
    for timed_out in (True, False):
        backend = OpenRPPGBackend.__new__(OpenRPPGBackend)
        backend._proc = SimpleNamespace(is_alive=lambda: timed_out)
        backend._out_q = queue.Queue()
        backend._job_generation = 1 if timed_out else None
        backend._job_started_at = 0
        backend.inference_timeout_seconds = 10
        backend._worker_state = "ready"
        reasons = []
        backend._restart_worker = reasons.append
        backend._drain_worker()
        assert reasons
        assert ("timed out" in reasons[0]) is timed_out
