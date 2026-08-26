"""Synthetic contract tests for every guided assessment protocol."""
from __future__ import annotations

import numpy as np
import time

from assessments import PROTOCOLS
from core.replay import synthetic_face, synthetic_pose
from core.context import FrameContext, PoseData
from core.workflows import WorkflowEngine, WorkflowStage
from modules.guided_assessments import GuidedAssessments
from modules.near_fall import NearFall
from storage.event_store import EventStore


def _pose_samples(profile: str, seconds: float, fps: int = 10):
    return [{"timestamp": i/fps, "pose": synthetic_pose(profile, i/fps, seconds),
             "face": None, "support_used": False, "microphone_ready": True}
            for i in range(int(seconds*fps)+1)]


def test_pose_protocols_score_measured_fields():
    """Movement protocols derive their named outputs from landmark trajectories."""
    expected = {
        "sit_to_stand": ("repetitions", "failed_attempts", "support_used", "movement_consistency"),
        "timed_up_and_go": ("stand", "walk", "turn", "return", "sit", "total_time_seconds"),
        "arm_drift": ("left_downward_drift", "right_downward_drift", "symmetry", "compliance"),
        "finger_tapping": ("left_tapping_rate_hz", "right_tapping_rate_hz", "left_right_difference_hz"),
        "balance": ("body_sway", "corrective_steps", "support_used"),
        "guided_gait": ("cadence_spm", "step_symmetry", "turning_stability", "shuffling_indicator"),
        "guided_breathing": ("instruction_compliance", "observed_cycles", "respiration_consistency"),
    }
    profile = {"guided_gait": "guided_gait"}
    for name, keys in expected.items():
        protocol = PROTOCOLS[name]
        samples = _pose_samples(profile.get(name, name), protocol.sampling_seconds)
        ready, quality, _message = protocol.positioner(samples[0])
        assert ready and quality >= protocol.quality_gate
        score = protocol.scorer(samples)
        assert score.quality >= protocol.quality_gate
        assert all(key in score.measurements for key in keys)


def test_facial_and_read_aloud_protocols():
    """Face motion and speech timing tasks expose public-safe measurements."""
    face_samples = [{"timestamp": i/10, "face": synthetic_face("facial_movement", i/10)}
                    for i in range(101)]
    face_score = PROTOCOLS["facial_movement"].scorer(face_samples)
    assert "smile_symmetry" in face_score.measurements
    speech = [{"timestamp": 0, "microphone_ready": True},
              {"timestamp": 3.2, "microphone_ready": True,
               "speech_metrics": {"duration": 3.2, "words_per_minute": 150,
                                  "pauses": 1, "baseline_change": .1, "quality": .9}}]
    score = PROTOCOLS["read_aloud"].scorer(speech)
    assert score.measurements["completion"] is True
    assert score.measurements["words_per_minute"] == 150


def test_positioning_failure_is_neutral():
    """Missing landmarks fail quality without producing a score or diagnosis."""
    ready, quality, message = PROTOCOLS["arm_drift"].positioner({"pose": None})
    assert not ready and quality == 0
    assert "see" in message.lower()


def test_guided_module_complete_sampling_path(tmp_path):
    """The production module reaches scoring and questions on synthetic frames."""
    engine = WorkflowEngine(event_store=EventStore(tmp_path / "events.sqlite3"))
    module = GuidedAssessments()
    module.engine = engine
    session = engine.start("arm_drift", now=time.time(), timeout=60)
    engine.transition(WorkflowStage.POSITIONING,
                      message=PROTOCOLS["arm_drift"].instruction)
    result = None
    base = time.time()
    for index in range(30):
        ts = base + index*.5
        pose = synthetic_pose("arm_drift", index*.5, 10)
        ctx = FrameContext(np.zeros((48,64,3),dtype=np.uint8), ts, index, 2,
                           pose=PoseData(pose,(0,0,64,48)), person_present=True)
        out = module.process(ctx)
        if out is not None and getattr(out, "key", "") == "arm_drift":
            result = out
            break
    assert result is not None
    assert result.correlation_id == session.correlation_id
    assert engine.active().stage == WorkflowStage.QUESTIONS


def test_near_fall_replay_profile_drives_detector():
    """Near-fall replay uses pose movement rather than a pre-baked result."""
    module = NearFall()
    found = None
    for index in range(51):
        offset = index/10
        pose = synthetic_pose("near_fall", offset, 5)
        ctx = FrameContext(np.zeros((48,64,3),dtype=np.uint8), offset, index, 10,
                           pose=PoseData(pose,(0,0,64,48)), person_present=True)
        found = module.process(ctx) or found
    assert found is not None and found.key == "recovered"
