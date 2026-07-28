"""Pure unit tests for the stationary D435i showcase gate."""
import numpy as np
import pytest

from core.context import FaceData, FrameContext, Intrinsics, PoseData
from core.showcase import ShowcaseGate


INTR = Intrinsics(100.0, 100.0, 100.0, 100.0)


def _ctx(distance_m=1.2, *, face=True, motion=0.0, brightness=100,
         faces=1, poses=1):
    frame = np.full((200, 200, 3), brightness, dtype=np.uint8)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    face_data = (FaceData(landmarks, (45, 35, 155, 145), frame[35:145, 45:155])
                 if face else None)
    pose_lm = np.zeros((33, 4), dtype=np.float32)
    for idx, x, y in ((11, .4, .4), (12, .6, .4), (23, .42, .7), (24, .58, .7)):
        pose_lm[idx] = (x, y, 0, 1)
    pose = PoseData(pose_lm, (40, 20, 160, 190))
    return FrameContext(frame, 1.0, 1, 30.0, face=face_data, pose=pose,
                        person_present=True, motion_energy=motion,
                        depth=np.full((200, 200), int(distance_m * 1000), np.uint16),
                        depth_scale=.001, intrinsics=INTR,
                        extras={"face_count": faces, "pose_count": poses})


def test_conversation_zone_allows_stable_face_measurements():
    gate = ShowcaseGate()
    ctx = _ctx()
    results = gate.assess(ctx)
    assert ctx.extras["showcase"]["zone"] == "conversation"
    assert ctx.extras["showcase"]["stable"] is True
    assert gate.allow("heart_rate", ctx)
    assert not gate.allow("gait", ctx)
    assert {r.key for r in results} == {"zone", "capture_ready", "heart_rate_ready"}


def test_vlm_second_opinion_bypasses_the_conversation_stability_gate():
    """A routed cloud cue must not be judged by the label it is published under.

    skin_vision publishes each cue onto the detector that owns its subject, so
    a lip reading arrives labelled `dry_lips` -- a conversation module. Gating
    that on `stable` hid the reading whenever the subject was off the
    conversation marker, while the very same scan stayed visible under
    `skin_vision`, which the gate never touches.
    """
    gate = ShowcaseGate()
    ctx = _ctx(distance_m=2.8)  # movement zone: not "stable"
    gate.assess(ctx)
    assert ctx.extras["showcase"]["stable"] is False
    # The detector's own heuristic stays gated ...
    assert not gate.allow("dry_lips", ctx)
    assert not gate.allow("dry_lips", ctx, "local")
    # ... while the cloud second opinion published under it comes through.
    assert gate.allow("dry_lips", ctx, "nvidia_vlm")
    for module in ("facial_swelling", "skin_color", "sweating", "eye_redness",
                   "rash"):
        assert gate.allow(module, ctx, "nvidia_vlm"), module


def test_disabled_gate_still_allows_everything():
    gate = ShowcaseGate(enabled=False)
    ctx = _ctx(distance_m=2.8)
    assert gate.allow("dry_lips", ctx, "local")
    assert gate.allow("dry_lips", ctx, "nvidia_vlm")


def test_gate_rejects_multiple_people_and_bad_capture():
    gate = ShowcaseGate()
    ctx = _ctx(faces=2)
    gate.assess(ctx)
    assert not ctx.extras["showcase"]["stable"]
    assert "one at a time" in ctx.extras["showcase"]["guidance"]
    assert not gate.allow("heart_rate", ctx)
    ctx = _ctx(motion=20)
    gate.assess(ctx)
    assert not ctx.extras["showcase"]["stable"]
    assert "hold still" in ctx.extras["showcase"]["guidance"]


def test_movement_zone_only_allows_whole_body_features():
    gate = ShowcaseGate()
    ctx = _ctx(2.8)
    gate.assess(ctx)
    assert ctx.extras["showcase"]["zone"] == "movement"
    assert gate.allow("gait", ctx)
    assert gate.allow("heart_rate", ctx)
    assert ctx.extras["showcase"]["heart_rate_ready"] is True


def test_heart_rate_does_not_require_depth_or_conversation_distance():
    gate = ShowcaseGate()
    ctx = _ctx(2.8)
    ctx.depth = None
    ctx.intrinsics = None
    gate.assess(ctx)
    assert ctx.extras["showcase"]["zone"] == "outside"
    assert ctx.extras["showcase"]["stable"] is False
    assert ctx.extras["showcase"]["heart_rate_ready"] is True
    assert gate.allow("heart_rate", ctx)
    assert not gate.allow("facial_asymmetry", ctx)


def test_dashboard_payload_exposes_reasoning_card():
    pytest.importorskip("cv2")
    from output.dashboard import to_payload
    reasoning = {"observed": "hydration", "question": "Have you had water?",
                 "answer": "asked", "suggestion": None}
    assert to_payload([], reasoning=reasoning)["reasoning"] == reasoning
