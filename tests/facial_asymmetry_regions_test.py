"""Unit test for the regional-asymmetry upgrade to
modules/facial_asymmetry.py: symmetric landmark pairs are grouped into
mouth/eye/brow/cheek_edge regions, each with its own baseline and
sustained-hit counter, so a droop isolated to one region (e.g. mouth) is
reported under its own key without being diluted by -- or bleeding into --
unrelated regions, and only the FAST-relevant regions (mouth, eye) may
escalate to ALERT.

Builds a synthetic, exactly-mirror-symmetric face (so every region's
baseline settles near zero) and then perturbs one region's right-side point
to simulate a one-sided droop, verifying region isolation, the sustained
-hit gate, and the WARNING/ALERT severity split.

Run standalone:  python tests/facial_asymmetry_regions_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FaceData, FrameContext
from core.events import Severity
from extractors import face_landmarks as FL
from modules.facial_asymmetry import FacialAsymmetry

FRAME_SIZE = 200


def _make_landmarks(mouth_dy=0.0, brow_dy=0.0):
    """(478,3) normalized landmarks: an exactly mirror-symmetric face about
    nose.x=0.50, optionally perturbed on the RIGHT mouth or brow point only
    (a one-sided droop)."""
    lm = np.zeros((478, 3), dtype=np.float32)
    lm[FL.NOSE_TIP] = [0.50, 0.45, 0.0]
    lm[FL.CHIN] = [0.50, 0.85, 0.0]
    lm[FL.LEFT_FACE_EDGE] = [0.20, 0.55, 0.0]
    lm[FL.RIGHT_FACE_EDGE] = [0.80, 0.55, 0.0]
    lm[FL.MOUTH_LEFT] = [0.40, 0.65, 0.0]
    lm[FL.MOUTH_RIGHT] = [0.60, 0.65 + mouth_dy, 0.0]
    lm[159] = [0.42, 0.50, 0.0]           # left upper eyelid
    lm[386] = [0.58, 0.50, 0.0]           # right upper eyelid
    lm[145] = [0.42, 0.53, 0.0]           # left lower eyelid
    lm[374] = [0.58, 0.53, 0.0]           # right lower eyelid
    lm[105] = [0.40, 0.42, 0.0]           # left brow mid
    lm[334] = [0.60, 0.42 + brow_dy, 0.0]  # right brow mid
    lm[FL.LEFT_CHEEK] = [0.35, 0.60, 0.0]
    lm[FL.RIGHT_CHEEK] = [0.65, 0.60, 0.0]
    return lm


def _make_ctx(t, mouth_dy=0.0, brow_dy=0.0):
    lm = _make_landmarks(mouth_dy=mouth_dy, brow_dy=brow_dy)
    face = FaceData(landmarks=lm, bbox=(0, 0, FRAME_SIZE - 1, FRAME_SIZE - 1),
                    crop=np.zeros((10, 10, 3), dtype=np.uint8), has_iris=True)
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    return FrameContext(frame=frame, timestamp=t, frame_index=0, fps=2.0, face=face)


def _seed_baseline(mod):
    """First call seeds t0/baseline (returns None); second call (well past
    learning_seconds) exits the learning phase with a near-zero baseline for
    every region since the seed frame was exactly symmetric."""
    assert mod.process(_make_ctx(t=0.0)) is None
    assert mod.process(_make_ctx(t=25.0)) is None   # learning_seconds=20.0


def test_isolated_mouth_droop_reports_only_mouth_after_sustained_hits():
    mod = FacialAsymmetry(persist_history=False)
    _seed_baseline(mod)
    # dy=0.036 -> normalized change ~= 0.06 (>0.05 threshold, <0.08 alert_threshold)
    # for mouth only; all other regions stay exactly symmetric (change=0).
    r1 = mod.process(_make_ctx(t=25.5, mouth_dy=0.036))
    r2 = mod.process(_make_ctx(t=26.0, mouth_dy=0.036))
    assert r1 is None and r2 is None, "must not report before alert_after_count hits"

    r3 = mod.process(_make_ctx(t=26.5, mouth_dy=0.036))
    assert r3 is not None, "3rd sustained hit should report"
    keys = {res.key for res in r3}
    assert keys == {"asymmetry_mouth"}, f"expected only mouth key, got {keys}"
    mouth_res = r3[0]
    assert mouth_res.severity == Severity.WARNING, (
        f"moderate mouth change should be WARNING, got {mouth_res.severity}")
    print("[facial-asymmetry-regions-test] isolated mouth droop -> mouth-only, "
          "gated, WARNING OK")


def test_large_mouth_droop_escalates_to_alert():
    mod = FacialAsymmetry(persist_history=False)
    _seed_baseline(mod)
    # dy=0.06 -> normalized change ~= 0.10 (>= 0.08 alert_threshold).
    for t in (25.5, 26.0, 26.5):
        r = mod.process(_make_ctx(t=t, mouth_dy=0.06))
    assert r is not None
    assert r[0].key == "asymmetry_mouth"
    assert r[0].severity == Severity.ALERT, f"expected ALERT, got {r[0].severity}"
    print("[facial-asymmetry-regions-test] large mouth droop -> ALERT OK")


def test_brow_region_caps_at_warning_even_when_large():
    mod = FacialAsymmetry(persist_history=False)
    _seed_baseline(mod)
    # Large brow deviation (would be well above alert_threshold if it could
    # escalate) -- brow is a non-FAST region and must cap at WARNING.
    for t in (25.5, 26.0, 26.5):
        r = mod.process(_make_ctx(t=t, brow_dy=0.15))
    assert r is not None
    keys = {res.key for res in r}
    assert keys == {"asymmetry_brow"}, f"expected only brow key, got {keys}"
    assert r[0].severity == Severity.WARNING, (
        f"non-FAST region must cap at WARNING, got {r[0].severity}")
    print("[facial-asymmetry-regions-test] large brow deviation caps at WARNING OK")


def test_symmetric_face_never_reports():
    mod = FacialAsymmetry(persist_history=False)
    _seed_baseline(mod)
    for i, t in enumerate([25.5, 26.0, 26.5, 27.0, 27.5]):
        assert mod.process(_make_ctx(t=t)) is None, f"unexpected report at t={t}"
    print("[facial-asymmetry-regions-test] steady symmetric face -> never reports OK")


def main():
    """Run all facial-asymmetry regional tests."""
    test_isolated_mouth_droop_reports_only_mouth_after_sustained_hits()
    test_large_mouth_droop_escalates_to_alert()
    test_brow_region_caps_at_warning_even_when_large()
    test_symmetric_face_never_reports()
    print("[facial-asymmetry-regions-test] OK")


if __name__ == "__main__":
    main()
