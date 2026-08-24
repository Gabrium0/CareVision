"""Multi-person detection contracts.

Covers the pieces added for --enable-multi-person: matching anonymous tracks
to face/pose geometry (core/subjects.py:build_subject_views), per-subject
detector isolation (SubjectModulePool), the runtime pause/resume overlay
(ModuleGate) as consulted by Scheduler.tick, and the deterministic safety
rule that a secondary (non-primary) subject's ALERT-severity result never
escalates to the caregiver (alerts/manager.py).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from alerts.manager import AlertManager
from alerts.notifier import Channel
from core.context import FaceData, FrameContext, PoseData
from core.events import Result, Severity
from core.module_gate import ModuleGate
from core.pipeline import Pipeline
import core.registry as registry_module
from core.scheduler import Scheduler
from core.subjects import SubjectModulePool, build_subject_views


def _frame(w=640, h=480):
    return np.zeros((h, w, 3), np.uint8)


def _landmarks(n):
    return np.zeros((n, 3), np.float32)


def _ctx_with_faces_and_poses(faces: list[tuple], poses: list[tuple]) -> FrameContext:
    ctx = FrameContext(frame=_frame(), timestamp=1.0, frame_index=0, fps=30.0)
    ctx.extras["faces"] = [{"bbox": bbox, "landmarks": _landmarks(478)} for bbox in faces]
    ctx.extras["poses"] = [{"bbox": bbox, "landmarks": _landmarks(33)} for bbox in poses]
    return ctx


# --------------------------------------------------------------- subject views

def test_build_subject_views_matches_each_track_to_its_own_geometry():
    ctx = _ctx_with_faces_and_poses(
        faces=[(0, 0, 100, 100), (400, 0, 500, 100)],
        poses=[(0, 0, 120, 400), (400, 0, 520, 400)])
    tracks = [
        {"track_id": "track-1", "subject_id": "primary", "bbox": (0, 0, 120, 400),
         "primary": True, "ambiguous": False, "stable_frames": 5},
        {"track_id": "track-2", "subject_id": "track-2", "bbox": (400, 0, 520, 400),
         "primary": False, "ambiguous": False, "stable_frames": 2},
    ]
    views = build_subject_views(ctx, tracks)
    assert {v.subject_id for v in views} == {"primary", "track-2"}
    primary = next(v for v in views if v.subject_id == "primary")
    other = next(v for v in views if v.subject_id == "track-2")
    assert primary.face is not None and tuple(primary.face.bbox) == (0, 0, 100, 100)
    assert other.face is not None and tuple(other.face.bbox) == (400, 0, 500, 100)
    assert primary.primary is True and other.primary is False


def test_build_subject_views_leaves_geometry_none_without_overlap():
    ctx = _ctx_with_faces_and_poses(faces=[(400, 400, 500, 500)], poses=[])
    tracks = [{"track_id": "track-1", "subject_id": "primary", "bbox": (0, 0, 50, 50),
              "primary": True, "ambiguous": False, "stable_frames": 5}]
    views = build_subject_views(ctx, tracks)
    assert views[0].face is None and views[0].pose is None


# --------------------------------------------------------------- module gate

def test_module_gate_scopes_are_independent():
    gate = ModuleGate(primary_enabled={"fall"}, secondary_enabled=set())
    assert gate.enabled("fall", "primary") is True
    assert gate.enabled("fall", "secondary") is False
    gate.set("fall", True, scope="secondary")
    assert gate.enabled("fall", "secondary") is True
    assert gate.enabled("fall", "primary") is True   # untouched by the other scope
    gate.set("fall", False, scope="primary")
    assert gate.enabled("fall", "primary") is False
    assert gate.enabled("fall", "secondary") is True  # still untouched
    assert gate.snapshot() == {"primary": [], "secondary": ["fall"]}


def test_scheduler_skips_gated_off_module_and_resumes_warm():
    calls = []

    class Module:
        name = "probe"
        interval = 0.0
        requires = ()
        def process(self, _ctx):
            calls.append(1)
            return None

    gate = ModuleGate(primary_enabled=set())     # starts disabled
    scheduler = Scheduler([Module()], gate=gate, scope="primary")
    ctx = FrameContext(_frame(), 1.0, 0, 30.0)
    scheduler.tick(ctx)
    assert calls == []                            # gated off: never called
    gate.set("probe", True, scope="primary")
    ctx2 = FrameContext(_frame(), 1.1, 1, 30.0)
    scheduler.tick(ctx2)
    assert calls == [1]                           # resumes immediately, no reload needed


# --------------------------------------------------------------- subject pool

class _CounterModule:
    """Minimal DetectionModule stand-in with per-instance state, so isolation
    between two SubjectModulePool instances is directly observable."""
    name = "_test_counter_module"
    interval = 0.0
    requires = ()
    closed_instances: list[int] = []

    def __init__(self, **_params):
        self.count = 0
        self._id = id(self)

    def process(self, _ctx):
        self.count += 1
        return Result(module=self.name, key="count", value=self.count,
                      confidence=1.0, severity=Severity.INFO)

    def close(self):
        _CounterModule.closed_instances.append(self._id)


@pytest.fixture(autouse=True)
def _register_counter_module():
    """Register the fake module only for the duration of each test in this
    file -- core.registry._REGISTRY is a process-global singleton, and
    leaving a test-only entry in it would make every other test that scans
    all_registered() (e.g. tests/module_console_test.py's roster-completeness
    checks) see a module with no reliability tier, blurb, or config entry."""
    registry_module.register("_test_counter_module")(_CounterModule)
    yield
    registry_module._REGISTRY.pop("_test_counter_module", None)


def test_subject_pool_isolates_state_per_track_and_stamps_subject_id():
    _CounterModule.closed_instances.clear()
    gate = ModuleGate(secondary_enabled={"_test_counter_module"})
    pool = SubjectModulePool(["_test_counter_module"], {}, gate=gate, ttl=5.0)

    now = 100.0
    scheduler_a = pool.get("track-a", now)
    scheduler_b = pool.get("track-b", now)
    ctx = FrameContext(_frame(), now, 0, 30.0)

    results_a = scheduler_a.tick(ctx)
    results_a2 = scheduler_a.tick(FrameContext(_frame(), now + 1, 1, 30.0))
    results_b = scheduler_b.tick(ctx)

    # track-a's module accumulated two ticks; track-b's is still on its first
    # -- proof the two tracks never share module state.
    assert results_a[0].value == 1
    assert results_a2[0].value == 2
    assert results_b[0].value == 1
    assert pool.get("track-a", now) is scheduler_a   # same instance, not rebuilt


def test_subject_pool_evicts_and_closes_after_ttl():
    _CounterModule.closed_instances.clear()
    gate = ModuleGate(secondary_enabled={"_test_counter_module"})
    pool = SubjectModulePool(["_test_counter_module"], {}, gate=gate, ttl=2.0)
    scheduler = pool.get("track-a", now=100.0)
    module_id = scheduler.modules[0]._id
    pool.evict_stale(now=101.0)                  # within ttl: still alive
    assert pool.get("track-a", now=101.5) is scheduler
    pool.evict_stale(now=104.0)                   # 104 - 101.5 > ttl: evicted
    assert module_id in _CounterModule.closed_instances
    rebuilt = pool.get("track-a", now=105.0)
    assert rebuilt is not scheduler                # a fresh instance, not reused


# --------------------------------------------------------------- pipeline hook

def test_pipeline_ticks_secondary_subjects_and_caps_at_max_subjects():
    """Exercise Pipeline._tick_secondary_subjects directly -- it only reads
    self.subject_pool/self.max_subjects, so a full camera/extractor stack
    is not needed to prove the multi-person wiring."""
    gate = ModuleGate(secondary_enabled={"_test_counter_module"})
    pool = SubjectModulePool(["_test_counter_module"], {}, gate=gate)
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.subject_pool = pool
    pipeline.max_subjects = 2   # primary + 1 secondary

    ctx = FrameContext(_frame(), 10.0, 0, 30.0)
    from core.context import SubjectView
    ctx.subjects = [
        SubjectView(subject_id="primary", track_id="track-1", primary=True),
        SubjectView(subject_id="track-2", track_id="track-2", primary=False),
        SubjectView(subject_id="track-3", track_id="track-3", primary=False),
    ]
    results = pipeline._tick_secondary_subjects(ctx)
    subject_ids = {r.subject_id for r in results}
    assert subject_ids == {"track-2"}              # capped: track-3 excluded, primary excluded
    assert all(r.module == "_test_counter_module" for r in results)


# --------------------------------------------------------------- alert escalation

class _Recorder(Channel):
    name = "recorder"
    def __init__(self):
        self.sent = []
    def send(self, subject, body):
        self.sent.append(subject)
        return True


def _fall(subject_id: str) -> Result:
    return Result(module="fall", key="fall", value=True, confidence=0.9,
                 severity=Severity.ALERT, message="FALL DETECTED",
                 subject_id=subject_id)


def test_secondary_subject_alert_never_escalates():
    rec = _Recorder()
    mgr = AlertManager(channels=[rec], confirm_seconds=1.0,
                       cooldown_seconds=60.0, escalate_after=5.0)
    now = 0.0
    for step in range(1, 20):        # well past confirm_seconds and escalate_after
        now = float(step)
        mgr.evaluate([_fall("track-2")], now=now)
    assert rec.sent == []


def test_primary_subject_alert_still_escalates():
    rec = _Recorder()
    mgr = AlertManager(channels=[rec], confirm_seconds=1.0,
                       cooldown_seconds=60.0, escalate_after=5.0)
    now = 0.0
    for step in range(1, 20):
        now = float(step)
        mgr.evaluate([_fall("primary")], now=now)
    assert rec.sent, "primary-subject ALERT must still reach the caregiver"
