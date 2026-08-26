"""Unit test for the jitter-robustification of modules/facial_swelling.py:
raw per-frame eye-opening/cheek-width features are pooled over a short
window (pooled_skin_sample, modules/_util.py) before the drift math, so a
single noisy landmark frame should not cross the swelling threshold the
way an unsmoothed single frame would, while a sustained change over
several frames still does.

Run standalone:  python tests/facial_swelling_pooling_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FaceData, FrameContext
from extractors import face_landmarks as FL
from modules.facial_swelling import FacialSwelling

FRAME_SIZE = 200
STEADY_EYE_GAP = 0.04     # normalized vertical eyelid gap (baseline "open" eye)


def _make_ctx(t, eye_gap=STEADY_EYE_GAP, cheek_half_span=0.15):
    """A face with symmetric eye-opening and cheek-width landmarks whose
    values are directly controllable (steady vs. a "puffy" eye gap)."""
    lm = np.zeros((478, 3), dtype=np.float32)
    lm[FL.LEFT_FACE_EDGE] = [0.20, 0.50, 0.0]
    lm[FL.RIGHT_FACE_EDGE] = [0.80, 0.50, 0.0]
    lm[159] = [0.42, 0.50 - eye_gap / 2, 0.0]   # left upper eyelid
    lm[145] = [0.42, 0.50 + eye_gap / 2, 0.0]   # left lower eyelid
    lm[386] = [0.58, 0.50 - eye_gap / 2, 0.0]   # right upper eyelid
    lm[374] = [0.58, 0.50 + eye_gap / 2, 0.0]   # right lower eyelid
    lm[FL.LEFT_CHEEK] = [0.50 - cheek_half_span, 0.55, 0.0]
    lm[FL.RIGHT_CHEEK] = [0.50 + cheek_half_span, 0.55, 0.0]
    face = FaceData(landmarks=lm, bbox=(0, 0, FRAME_SIZE - 1, FRAME_SIZE - 1),
                    crop=np.zeros((10, 10, 3), dtype=np.uint8), has_iris=True)
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    return FrameContext(frame=frame, timestamp=t, frame_index=0, fps=0.5, face=face)


def _learn_baseline(mod):
    """Steady frames through the 30s learning window; baseline settles at
    the steady eye/cheek ratios."""
    for t in (0.0, 10.0, 20.0, 29.0):
        assert mod.process(_make_ctx(t)) is None


def test_single_noisy_frame_does_not_trigger():
    mod = FacialSwelling()
    _learn_baseline(mod)
    # Exit learning, then a few steady frames to seed the pooling window.
    assert mod.process(_make_ctx(31.0)) is None
    assert mod.process(_make_ctx(33.0)) is None
    assert mod.process(_make_ctx(35.0)) is None
    # One single-frame eye-gap jitter (halved) -- would score ~0.25 unsmoothed
    # (well above the 0.08 threshold) but pooled with 3 prior steady samples
    # stays damped below it.
    r = mod.process(_make_ctx(37.0, eye_gap=STEADY_EYE_GAP / 2))
    assert r is None, f"a single noisy frame should not report swelling, got {r}"
    print("[facial-swelling-pooling-test] single noisy frame damped, no report OK")


def test_sustained_droop_still_triggers():
    mod = FacialSwelling()
    _learn_baseline(mod)
    mod.process(_make_ctx(31.0))
    mod.process(_make_ctx(33.0))
    mod.process(_make_ctx(35.0))
    r = None
    # Several consecutive frames with a halved eye gap -- once the pooling
    # window (6s) is no longer dominated by the earlier steady samples, the
    # drift should clear the threshold.
    for t in (37.0, 39.0, 41.0, 43.0):
        r = mod.process(_make_ctx(t, eye_gap=STEADY_EYE_GAP / 2))
    assert r is not None, "a sustained eye-opening drop should eventually report"
    assert r.key == "swelling"
    print("[facial-swelling-pooling-test] sustained droop still reports OK")


def main():
    """Run all facial-swelling pooling tests."""
    test_single_noisy_frame_does_not_trigger()
    test_sustained_droop_still_triggers()
    print("[facial-swelling-pooling-test] OK")


if __name__ == "__main__":
    main()
