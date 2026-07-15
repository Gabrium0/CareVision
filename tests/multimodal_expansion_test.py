"""Contracts for multimodal metadata, privacy, workflows, replay, and sensors."""
from __future__ import annotations

import numpy as np
import pytest

from assessments import PROTOCOLS
from core.capabilities import CapabilityRegistry, CapabilityStatus
from core.events import PersistencePolicy, Result, Visibility
from core.replay import ReplayCamera
from core.tracking import AnonymousTracker
from core.workflows import WorkflowEngine, WorkflowStage
from output.aggregator import Aggregator
from output.dashboard import to_payload
from sensors.simulated import SimulatedSensor
from storage.event_store import EventStore
from modules.scene_vision import SceneVision, validate_scene
from core.context import FrameContext


def test_result_metadata_is_backward_compatible():
    """Old positional construction works while new metadata defaults safely."""
    result = Result("demo", "value", 3)
    assert result.subject_id == "primary"
    assert result.persistence == PersistencePolicy.NONE
    assert result.visibility == Visibility.PUBLIC


def test_event_store_refuses_private_and_raw_media(tmp_path):
    """Agent-only results and ndarray payloads never enter SQLite."""
    store = EventStore(tmp_path / "events.sqlite3")
    private = Result("vlm", "hypothesis", "condition", visibility=Visibility.AGENT_ONLY,
                     persistence=PersistencePolicy.EVENT)
    assert store.record_result(private) is None
    raw = Result("camera", "frame", np.zeros((2, 2)), persistence=PersistencePolicy.EVENT)
    with pytest.raises(TypeError):
        store.record_result(raw)
    with pytest.raises(TypeError):
        store.record("observation", {"value": "data:image/jpeg;base64,AAAA"})
    assert store.recent() == []
    store.close()


def test_dashboard_defensively_filters_agent_only_results():
    """Private hypotheses cannot leak even if a caller passes an internal snapshot."""
    private = Result("vlm", "hypothesis", "private-condition-name",
                     message="private-condition-name", visibility=Visibility.AGENT_ONLY)
    public = Result("demo", "quality", "usable", message="Usable image")
    payload = to_payload([public, private])
    serialized = str(payload)
    assert "private-condition-name" not in serialized
    assert "Usable image" in serialized


def test_subject_aggregation_isolated():
    """Visitor state cannot overwrite the primary subject's latest result."""
    agg = Aggregator()
    agg.ingest([Result("m", "k", 1), Result("m", "k", 9, subject_id="track-2")])
    assert agg.get("m", "k").value == 1
    assert agg.get("m", "k", subject_id="track-2").value == 9


def test_workflow_budgets_retry_and_questions(tmp_path):
    """The common engine enforces one retry and three follow-ups."""
    engine = WorkflowEngine(60, EventStore(tmp_path / "events.sqlite3"))
    session = engine.start("arm_drift", now=100)
    assert session is not None
    engine.transition(WorkflowStage.POSITIONING)
    assert engine.retry("reposition")
    assert not engine.retry("again")
    assert [engine.ask(str(i)) for i in range(4)] == [True, True, True, False]


def test_all_assessment_protocols_have_quality_contracts():
    """The full initial assessment library is registered declaratively."""
    assert set(PROTOCOLS) == {"sit_to_stand", "timed_up_and_go", "arm_drift",
                             "finger_tapping", "balance", "guided_gait",
                             "facial_movement", "read_aloud", "guided_breathing"}
    for protocol in PROTOCOLS.values():
        assert protocol.sampling_seconds > 0
        assert len(protocol.follow_up_topics) <= 3


def test_replay_is_camera_compatible():
    """Scripted scenarios yield production FrameContext objects deterministically."""
    camera = ReplayCamera("replay:healthy_greeting", request_fps=10,
                          request_size=(64, 48))
    frames = list(camera.frames())
    assert frames[0].frame.shape == (48, 64, 3)
    assert sum(len(f.extras["replay_events"]) for f in frames) == 1


def test_tracker_is_anonymous_and_marks_primary():
    """Tracking exposes only short-lived IDs and geometry ambiguity."""
    tracker = AnonymousTracker(primary_min_frames=1)
    tracks = tracker.update([(0, 0, 100, 100), (150, 0, 190, 40)], 200, 100, 1)
    assert tracks[0]["subject_id"] == "primary"
    assert tracks[1]["subject_id"].startswith("track-")
    assert "identity" not in tracks[0]
    # A visitor becoming larger on the next frame cannot take over primary.
    moved = tracker.update([(0, 0, 80, 80), (100, 0, 200, 100)], 200, 100, 2)
    assert next(t for t in moved if t["primary"])["track_id"] == tracks[0]["track_id"]


def test_capability_and_simulated_sensor():
    """Hardware-free showcase adapters report ready and emit trusted simulations."""
    sensor = SimulatedSensor("thermal_test", [{"key": "temperature_c", "value": 36.8,
                                                "unit": "C"}], interval=1)
    reading = sensor.poll(now=10)[0]
    assert reading.value == 36.8
    assert any(c["name"] == "thermal_test" and c["status"] == "ready"
               for c in CapabilityRegistry.instance().snapshot())


def test_scene_schema_and_consent_gate(monkeypatch):
    """Scene VLM remains idle without its independent consent flag."""
    scene = validate_scene({"objects": ["cup"], "activities": ["drinking"],
                            "locations": ["kitchen"], "visible_hazards": ["spill"],
                            "quality": .8, "confidence": .7,
                            "conversation_topics": ["hydration"]})
    assert scene.hazards == ("spill",)
    monkeypatch.setattr("modules.scene_vision.nvidia_api_key", lambda: "configured")
    module = SceneVision(consent=False, scan_interval=0)
    ctx = FrameContext(np.zeros((8, 8, 3), dtype=np.uint8), 1, 0, 10)
    assert module.process(ctx) is None
    assert module._pending is None
    module.close()
