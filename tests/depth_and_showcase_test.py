"""Tests for the D435i depth plumbing and the showcase interaction modules.

Covers, with synthetic in-memory data only (no camera, no pyrealsense2):
- FrameContext depth helpers (depth_m / deproject / mm_per_px) including
  hole handling;
- the scheduler's "depth" requires-token (depth modules skipped on RGB);
- graceful degrade: depth-aware modules run their unchanged 2D path when
  ctx.depth is None, and reset baselines when the modality flips;
- the attention / sneeze / face_touch stubbed-landmark behavior;
- ObservationMemory.context_text() confidence gating of the new signals;
- camera_factory backend selection and cross-type switch routing.

Run standalone:  python -m pytest tests/depth_and_showcase_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FaceData, FrameContext, Intrinsics, PoseData
from core.events import Result, Severity
from core.scheduler import Scheduler
from extractors import face_landmarks as FL

FRAME = 200
INTR = Intrinsics(fx=100.0, fy=100.0, ppx=100.0, ppy=100.0)


def _ctx(t=0.0, face=None, pose=None, depth=None):
    frame = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    return FrameContext(frame=frame, timestamp=t, frame_index=0, fps=30.0,
                        face=face, pose=pose, person_present=True,
                        depth=depth, intrinsics=INTR if depth is not None else None)


def _flat_depth(mm=1000):
    return np.full((FRAME, FRAME), mm, dtype=np.uint16)


# ---------------------------------------------------------------- helpers

def test_depth_helpers_on_flat_plane():
    ctx = _ctx(depth=_flat_depth(1000))
    assert abs(ctx.depth_m(100, 100) - 1.0) < 1e-6
    pt = ctx.deproject(100, 100)                 # principal point -> on axis
    assert np.allclose(pt, [0.0, 0.0, 1.0])
    pt = ctx.deproject(150, 100)                 # 50 px right at fx=100, z=1m
    assert np.allclose(pt, [0.5, 0.0, 1.0])
    assert abs(ctx.mm_per_px(100, 100) - 10.0) < 1e-6   # 1m / 100px-focal


def test_depth_helpers_handle_holes_and_absence():
    ctx = _ctx(depth=np.zeros((FRAME, FRAME), dtype=np.uint16))
    assert ctx.depth_m(100, 100) is None         # all holes
    assert ctx.deproject(100, 100) is None
    ctx = _ctx()                                 # RGB-only context
    assert ctx.depth_m(100, 100) is None
    assert ctx.mm_per_px(100, 100) is None


# -------------------------------------------------------------- scheduler

class _DepthProbe:
    name = "depth_probe"
    interval = 0.0
    requires = ("depth",)

    def __init__(self):
        self.calls = 0

    def process(self, ctx):
        self.calls += 1
        return None


def test_scheduler_skips_depth_modules_without_depth():
    probe = _DepthProbe()
    sched = Scheduler([probe])
    sched.tick(_ctx(t=1.0))
    assert probe.calls == 0
    sched.tick(_ctx(t=2.0, depth=_flat_depth()))
    assert probe.calls == 1


# ------------------------------------------------------------- respiration

def _pose(nose=(0.5, 0.3), wrists=((0.2, 0.9), (0.8, 0.9))):
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[0] = [nose[0], nose[1], 0.0, 1.0]
    lm[11] = [0.40, 0.45, 0.0, 1.0]
    lm[12] = [0.60, 0.45, 0.0, 1.0]
    lm[23] = [0.42, 0.70, 0.0, 1.0]
    lm[24] = [0.58, 0.70, 0.0, 1.0]
    lm[27] = [0.44, 0.95, 0.0, 1.0]
    lm[28] = [0.56, 0.95, 0.0, 1.0]
    lm[15] = [wrists[0][0], wrists[0][1], 0.0, 1.0]
    lm[16] = [wrists[1][0], wrists[1][1], 0.0, 1.0]
    lm[19] = [wrists[0][0], wrists[0][1], 0.0, 1.0]
    lm[20] = [wrists[1][0], wrists[1][1], 0.0, 1.0]
    return PoseData(landmarks=lm, bbox=(0, 0, FRAME - 1, FRAME - 1))


def test_respiration_samples_depth_and_resets_on_modality_flip():
    from modules.respiration import Respiration
    mod = Respiration()
    # RGB frame: buffer fills with shoulder-y (normalized units)
    mod.process(_ctx(t=0.0, pose=_pose()))
    assert not mod._depth_mode and len(mod.buf) == 1
    # depth frame: modality flips, buffer resets, sample is meters
    mod.process(_ctx(t=0.1, pose=_pose(), depth=_flat_depth(1500)))
    assert mod._depth_mode and len(mod.buf) == 1
    assert abs(mod.buf.v[-1] - 1.5) < 1e-6
    # back to RGB: resets again (graceful degrade, no unit mixing)
    mod.process(_ctx(t=0.2, pose=_pose()))
    assert not mod._depth_mode and len(mod.buf) == 1


# -------------------------------------------------------- facial asymmetry

def _face_landmarks(mouth_dy=0.0, iris_dx=0.0, nose_dx=0.0, eyes_open=True,
                    nose_dy=0.0):
    lm = np.zeros((478, 3), dtype=np.float32)
    lm[FL.NOSE_TIP] = [0.50 + nose_dx, 0.55 + nose_dy, 0.0]
    lm[FL.CHIN] = [0.50, 0.85, 0.0]
    lm[FL.FOREHEAD_TOP] = [0.50, 0.20, 0.0]
    lm[FL.LEFT_FACE_EDGE] = [0.20, 0.55, 0.0]
    lm[FL.RIGHT_FACE_EDGE] = [0.80, 0.55, 0.0]
    lm[FL.MOUTH_LEFT] = [0.40, 0.65, 0.0]
    lm[FL.MOUTH_RIGHT] = [0.60, 0.65 + mouth_dy, 0.0]
    gap = 0.010 if eyes_open else 0.0
    # left eye ring (EAR indices) around y=0.50
    lm[33] = [0.40, 0.50, 0.0]
    lm[133] = [0.46, 0.50, 0.0]
    lm[160] = [0.42, 0.50 - gap, 0.0]
    lm[158] = [0.44, 0.50 - gap, 0.0]
    lm[153] = [0.44, 0.50 + gap, 0.0]
    lm[144] = [0.42, 0.50 + gap, 0.0]
    lm[159] = [0.43, 0.50 - gap, 0.0]
    lm[145] = [0.43, 0.50 + gap, 0.0]
    # right eye ring
    lm[362] = [0.54, 0.50, 0.0]
    lm[263] = [0.60, 0.50, 0.0]
    lm[385] = [0.56, 0.50 - gap, 0.0]
    lm[387] = [0.58, 0.50 - gap, 0.0]
    lm[373] = [0.58, 0.50 + gap, 0.0]
    lm[380] = [0.56, 0.50 + gap, 0.0]
    lm[386] = [0.57, 0.50 - gap, 0.0]
    lm[374] = [0.57, 0.50 + gap, 0.0]
    # brows / cheeks
    lm[105] = [0.40, 0.42, 0.0]
    lm[334] = [0.60, 0.42, 0.0]
    lm[FL.LEFT_CHEEK] = [0.35, 0.60, 0.0]
    lm[FL.RIGHT_CHEEK] = [0.65, 0.60, 0.0]
    # iris centers (default: looking straight ahead)
    lm[FL.LEFT_IRIS[0]] = [0.43 + iris_dx, 0.50, 0.0]
    lm[FL.RIGHT_IRIS[0]] = [0.57 + iris_dx, 0.50, 0.0]
    return lm


def _face_ctx(t=0.0, depth=None, **kw):
    lm = _face_landmarks(**kw)
    face = FaceData(landmarks=lm, bbox=(0, 0, FRAME - 1, FRAME - 1),
                    crop=np.zeros((10, 10, 3), dtype=np.uint8), has_iris=True)
    frame = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    return FrameContext(frame=frame, timestamp=t, frame_index=0, fps=30.0,
                        face=face, person_present=True, depth=depth,
                        intrinsics=INTR if depth is not None else None)


def test_facial_asymmetry_3d_runs_on_flat_depth_and_resets_on_flip():
    from modules.facial_asymmetry import FacialAsymmetry
    mod = FacialAsymmetry(persist_history=False)
    # depth mode: symmetric face on a flat plane -> baseline learns, no findings
    assert mod.process(_face_ctx(t=0.0, depth=_flat_depth())) is None
    assert mod._depth_mode and mod.t0 is not None
    assert mod.process(_face_ctx(t=25.0, depth=_flat_depth())) is None
    # modality flip back to 2D resets the learned baseline
    mod.process(_face_ctx(t=26.0))
    assert not mod._depth_mode
    assert mod.t0 == 26.0                       # baseline restarted


def test_facial_asymmetry_skips_frame_on_depth_holes():
    from modules.facial_asymmetry import FacialAsymmetry
    mod = FacialAsymmetry(persist_history=False)
    holes = np.zeros((FRAME, FRAME), dtype=np.uint16)
    assert mod.process(_face_ctx(t=0.0, depth=holes)) is None
    assert mod.t0 is None                       # frame skipped, nothing learned


def test_facial_swelling_feature_vector_grows_with_depth():
    from modules.facial_swelling import FacialSwelling
    mod = FacialSwelling()
    assert len(mod._features(_face_ctx())) == 2
    assert len(mod._features(_face_ctx(depth=_flat_depth()))) == 3
    holes = np.zeros((FRAME, FRAME), dtype=np.uint16)
    assert mod._features(_face_ctx(depth=holes)) is None


# --------------------------------------------------------- height/distance

def test_height_distance_reads_metric_values():
    from modules.height_distance import HeightDistance
    mod = HeightDistance()
    out = mod.process(_ctx(t=0.0, pose=_pose(), depth=_flat_depth(1500)))
    by_key = {r.key: r for r in out}
    assert abs(float(by_key["distance_m"].value) - 1.5) < 0.05
    # nose y=0.3 -> 60px, ankles y=0.95 -> 190px; extent = 130px * 15mm/px = 1.95m
    assert "height_m" in by_key
    assert 1.9 < float(by_key["height_m"].value) < 2.3


# --------------------------------------------------------------- attention

def test_attention_eye_contact_true_when_facing_and_gazing():
    from modules.attention import Attention
    mod = Attention()
    out = mod.process(_face_ctx(t=0.0))
    vals = {r.key: r.value for r in out}
    assert vals["eye_contact"] is True


def test_attention_false_when_head_turned():
    from modules.attention import Attention
    mod = Attention()
    out = mod.process(_face_ctx(t=0.0, nose_dx=0.15))   # yaw offset > threshold
    vals = {r.key: r.value for r in out}
    assert vals["eye_contact"] is False
    assert vals["attention_seconds"] == 0.0


# ------------------------------------------------------------------ sneeze

def test_sneeze_fires_on_pitch_jerk_with_eye_closure():
    from modules.sneeze import Sneeze
    mod = Sneeze()
    # settle: open eyes, stable pitch
    for i in range(10):
        mod.process(_face_ctx(t=i * 0.05))
    # the jerk: head snaps down (nose drops) with eyes shut
    out = mod.process(_face_ctx(t=0.55, nose_dy=0.10, eyes_open=False))
    keys = {r.key: r for r in out}
    assert "sneeze" in keys and keys["sneeze"].severity == Severity.NOTICE
    assert keys["sneeze_count_10min"].value == 1
    # cooldown: an immediate second jerk does not double count
    out = mod.process(_face_ctx(t=0.6, nose_dy=0.10, eyes_open=False))
    assert all(r.key != "sneeze" for r in out)


def test_sneeze_ignores_slow_nod_with_open_eyes():
    from modules.sneeze import Sneeze
    mod = Sneeze()
    for i in range(10):
        mod.process(_face_ctx(t=i * 0.05))
    out = mod.process(_face_ctx(t=0.55, nose_dy=0.10))   # eyes stay open
    assert all(r.key != "sneeze" for r in out)


# -------------------------------------------------------------- face touch

def test_face_touch_counts_once_per_contact():
    from modules.face_touch import FaceTouch
    mod = FaceTouch()
    out = mod.process(_ctx(t=0.0, pose=_pose()))          # hands down
    assert all(r.key != "face_touch" for r in out)
    out = mod.process(_ctx(t=1.0, pose=_pose(wrists=((0.5, 0.31), (0.8, 0.9)))))
    keys = {r.key: r.value for r in out}
    assert keys.get("face_touch") is True
    # still touching: edge-triggered, no second event
    out = mod.process(_ctx(t=1.5, pose=_pose(wrists=((0.5, 0.31), (0.8, 0.9)))))
    assert all(r.key != "face_touch" for r in out)


# ------------------------------------------------------------ agent context

def _res(module, key, value, conf, message=""):
    return Result(module=module, key=key, value=value, confidence=conf,
                  severity=Severity.INFO, message=message, ttl=10.0)


def test_context_text_includes_new_signals_above_confidence_gates():
    from agent.state import ObservationMemory
    mem = ObservationMemory(name="Ada")
    mem.ingest([
        _res("respiration", "breaths_per_min", 14.0, 0.6),
        _res("age_estimation", "age_range", "(25-32)", 0.5),
        _res("drowsiness", "perclos", 0.05, 0.7),
        _res("attention", "eye_contact", True, 0.6),
        _res("height_distance", "height_m", 1.82, 0.6),
        _res("height_distance", "distance_m", 1.4, 0.8),
    ], now=100.0)
    text = mem.context_text()
    assert "Breathing about 14 per minute" in text
    assert "(25-32)" in text
    assert "Alertness: alert" in text
    assert "looking at you" in text
    assert "1.82 m" in text and "1.4 m away" in text


def test_context_text_omits_low_confidence_signals():
    from agent.state import ObservationMemory
    mem = ObservationMemory()
    mem.ingest([
        _res("respiration", "breaths_per_min", 14.0, 0.1),
        _res("height_distance", "height_m", 1.82, 0.1),
    ], now=100.0)
    text = mem.context_text()
    assert "Breathing" not in text and "1.82" not in text


# ---------------------------------------------------------- camera factory

def test_factory_selects_backend_and_routes_switches():
    from core.camera import Camera
    from core.camera_factory import SwitchableCamera, make_backend
    from core.ipad_camera import IPadCamera
    from core.realsense_camera import RealSenseCamera
    assert isinstance(make_backend("realsense"), RealSenseCamera)
    assert isinstance(make_backend("0"), Camera)
    # aiortc is NOT installed in this environment -- this must still succeed,
    # which is exactly what enforces core.ipad_camera's lazy-import split
    # (the WebRTC stack is only imported inside IPadCamera.open()/_build_link).
    assert isinstance(make_backend("ipad"), IPadCamera)
    cam = SwitchableCamera("0")
    cam.switch_to("1")                          # same type: delegates to inner
    assert cam.inner._pending is not None and cam._cross_pending is None
    cam.inner._pending = None
    cam.switch_to("realsense")                  # cross type: queued on facade
    assert cam._cross_pending is not None and cam.inner._pending is None

    # switching TO an iPad source is always a cross-type (rebuild) switch,
    # never delegated to an inner backend's own switch_to (see IPadCamera's
    # switch_to, which raises: re-pairing needs a fresh peer connection).
    cam2 = SwitchableCamera("0")
    cam2.switch_to("ipad")
    assert cam2._cross_pending is not None and cam2.inner._pending is None

    # constructing an iPad-sourced SwitchableCamera must not open any socket
    # (no aiortc installed, no relay reachable) -- open() is deferred until
    # the frames loop actually starts consuming.
    cam3 = SwitchableCamera("ipad")
    assert isinstance(cam3.inner, IPadCamera)
