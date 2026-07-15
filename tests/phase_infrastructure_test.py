"""Phase 1, 3, 4, 5, 6, and 7 completion contracts."""
from __future__ import annotations

import time
from concurrent.futures import Future

import numpy as np
import pytest

from agent.routines import RoutineReasoner
from agent.multimodal_reasoning import MultimodalReasoner
from agent.voice_agent import VoiceAgent
from audio.bus import AudioBus
from audio.intelligence import _canonical
from core.context import FrameContext, Intrinsics, PoseData
from core.events import PersistencePolicy, Result, Severity, Visibility
from core.replay import ReplayCamera
from core.workflows import WorkflowEngine, WorkflowStage
from modules.scene_vision import SceneVision, validate_scene
from sensors.adapters import BLEGattAdapter, RealSenseContextAdapter
from storage.event_store import EventStore
from storage.history_store import HistoryStore


def test_event_retention_chain_and_sensitive_rejection(tmp_path):
    """Typed causal chains retain safe summaries and purge by policy."""
    store = EventStore(tmp_path / "events.sqlite3")
    cid = "chain"
    store.record_assessment("started", "balance", correlation_id=cid, timestamp=10,
                            retention_seconds=5)
    store.record_question("symptoms", 1, correlation_id=cid, timestamp=11)
    store.record_answer("denied", correlation_id=cid, timestamp=12)
    store.record_recommendation("balance", "No diagnosis was made.", correlation_id=cid,
                                timestamp=13)
    assert [event["kind"] for event in store.chain(cid)] == [
        "assessment", "question", "answer", "recommendation"]
    with pytest.raises(TypeError):
        store.record("observation", {"biometric_embedding": [0.1, 0.2]})
    assert store.purge_expired(now=16) == 1


def test_workflow_timeout_concurrency_denial_and_rephrase(tmp_path):
    """Subjects isolate workflows and unclear/denied answers follow strict budgets."""
    engine = WorkflowEngine(event_store=EventStore(tmp_path / "events.sqlite3"))
    a = engine.start("arm_drift", "primary", now=10, timeout=5)
    b = engine.start("balance", "track-2", now=10, timeout=50)
    assert a and b and a.correlation_id != b.correlation_id
    engine.set_score("done", {}, .8, ("symptoms", "progression"), "track-2")
    assert engine.next_question("track-2") == "symptoms"
    assert engine.answer("unclear", "track-2") == "rephrase"
    assert engine.answer("unclear", "track-2") == "continue"
    assert engine.next_question("track-2") == "progression"
    assert engine.answer("denied", "track-2") == "suppressed"
    expired = engine.tick(now=16)
    assert expired[0].stage == WorkflowStage.TIMED_OUT


def test_assessment_concludes_neutrally_without_microphone(tmp_path, monkeypatch):
    """A speak-only installation still finishes a measured workflow safely."""
    store = EventStore(tmp_path / "events.sqlite3")
    engine = WorkflowEngine(event_store=store)
    monkeypatch.setattr(WorkflowEngine, "_instance", engine)
    monkeypatch.setattr(EventStore, "_instance", store)
    agent = VoiceAgent(speak=False, listener=None, min_gap=0)
    agent.gemini.available = False
    session = engine.start("balance", now=10)
    engine.set_score("Balance measurements captured.", {"body_sway": .01}, .9, ())
    text = agent.tick([], now=20)
    assert text == "Balance measurements captured."
    assert engine.active() is None
    assert any(event["kind"] == "recommendation" for event in store.chain(session.correlation_id))
    agent.close()


def test_replay_controls_and_synchronized_channels():
    """Replay exposes controls and routes sensor/answer events by channel."""
    camera = ReplayCamera("replay:wearable_thermal", request_fps=10,
                          request_size=(32, 24))
    assert camera.control("speed", 2)["speed"] == 2
    frames = list(camera.frames())
    sensors = [event for frame in frames
               for event in frame.extras["replay_channels"]["sensor"]]
    assert {event["key"] for event in sensors} == {"heart_rate_bpm", "skin_temperature_c"}
    assert camera.status()["scenario"] == "wearable_thermal"


def test_audio_bus_and_taxonomy():
    """One audio block reaches multiple consumers and labels normalize safely."""
    bus = AudioBus()
    one, two = bus.subscribe("one"), bus.subscribe("two")
    bus.publish(np.ones(160, dtype=np.float32), 1)
    assert len(one.get_nowait()[0]) == len(two.get_nowait()[0]) == 160
    assert _canonical("Smoke detector, smoke alarm") == "smoke_alarm"
    assert _canonical("Cough") == "cough"


def test_scene_schema_is_strict():
    """Loose or incomplete cloud scene responses are refused."""
    with pytest.raises(ValueError):
        validate_scene({"objects": []})


def test_scene_failure_backs_off_and_never_alerts(monkeypatch):
    """Cloud failures are sanitized, exponentially delayed, and cannot create alerts."""
    monkeypatch.setattr("modules.scene_vision.nvidia_api_key", lambda: "configured")
    scene = SceneVision(consent=True, scan_interval=0, window_frames=1,
                        backoff_base=7, backoff_max=60)
    failed = Future()
    failed.set_exception(TimeoutError("payload must not appear"))
    scene._pending = failed
    ctx = FrameContext(np.zeros((8, 8, 3), dtype=np.uint8), 100, 0, 10)
    result = scene.process(ctx)
    assert not result
    assert scene._failures == 1 and scene._next_allowed == 107
    assert scene._pending is None
    scene.close()


def test_routine_baselines_are_subject_isolated(tmp_path):
    """Visitor opportunities never enter the primary person's baseline."""
    reasoner = RoutineReasoner(interval=0)
    reasoner.store = HistoryStore(tmp_path / "history.sqlite3")
    now = time.time()
    primary = Result("scene_vision", "activities", ["drinking"], timestamp=now,
                     subject_id="primary")
    visitor = Result("scene_vision", "activities", ["eating"], timestamp=now,
                     subject_id="track-2")
    reasoner.evaluate([primary, visitor], now)
    assert reasoner.store.count_since("routine", "drink_opportunity", 60, "primary", now) == 1
    assert reasoner.store.count_since("routine", "meal_opportunity", 60, "primary", now) == 0
    assert reasoner.store.count_since("routine", "meal_opportunity", 60, "track-2", now) == 1


def test_cross_signal_fusion_never_combines_different_people():
    """A visitor's confirmation cannot corroborate the primary person's signals."""
    rows = [Result("sound_event", "cough", True, subject_id="primary"),
            Result("activity_level", "activity_drop", True, subject_id="primary"),
            Result("conversation", "symptoms_confirmed", True, subject_id="track-2")]
    assert MultimodalReasoner().evaluate(rows) == []
    rows[-1].subject_id = "primary"
    result = MultimodalReasoner().evaluate(rows)
    assert len(result) == 1 and result[0].subject_id == "primary"


def test_realsense_adapter_and_ble_parser():
    """Depth context and BLE payloads become metric readings off capture paths."""
    adapter = RealSenseContextAdapter(interval=0)
    pose = np.zeros((33,4), dtype=np.float32); pose[:,3] = 1; pose[:,0] = .5; pose[:,1] = .5
    pose[27,:2], pose[28,:2], pose[23,:2], pose[24,:2] = (.4,.9),(.6,.9),(.45,.55),(.55,.55)
    ctx = FrameContext(np.zeros((20,20,3),dtype=np.uint8), 1, 0, 10,
        pose=PoseData(pose,(0,0,20,20)), depth=np.full((20,20),1000,dtype=np.uint16),
        intrinsics=Intrinsics(100,100,10,10), ego_motion=.1)
    adapter.feed_context(ctx)
    keys = {reading.key for reading in adapter.poll()}
    assert "nearest_obstacle_m" in keys and "camera_motion_rad_s" in keys
    ble = BLEGattAdapter("test")
    ble._notification({"key":"heart_rate_bpm","format":"heart_rate","unit":"bpm"}, bytes([0,72]))
    assert ble.poll()[0].value == 72
