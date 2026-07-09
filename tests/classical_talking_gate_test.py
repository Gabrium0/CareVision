"""Unit test for the talking-motion gate added to ClassicalBackend.update()
(modules/rppg_backends/classical.py): rapid mouth-aspect-ratio (MAR) changes
between consecutive frames (speech) should drop the cheek ROIs for that
sample and use forehead-only, since cheeks visibly move with speech and
would otherwise inject non-cardiac motion into the chrominance signal.

Spies on modules._util.roi_patch (as imported into the classical module) to
record which landmark indices get sampled each update() call, rather than
building a pixel-accurate synthetic face -- this directly tests the ROI
selection logic without depending on roi_patch's own pixel-cropping details
(already covered by other tests).

Run standalone:  python tests/classical_talking_gate_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.rppg_backends.classical as classical_mod
from core.context import FaceData, FrameContext
from extractors import face_landmarks as FL

FRAME_SIZE = 200


def _make_ctx(mar_gap: float, t: float) -> FrameContext:
    """FrameContext whose mouth landmarks yield MAR ~= 5 * mar_gap (see
    comment below), with forehead/cheek landmarks present but otherwise
    irrelevant here since roi_patch is spied/stubbed in these tests."""
    landmarks = np.zeros((300, 3), dtype=np.float32)
    landmarks[FL.MOUTH_LEFT] = [0.40, 0.70, 0.0]
    landmarks[FL.MOUTH_RIGHT] = [0.60, 0.70, 0.0]         # width = 0.20 (norm)
    landmarks[FL.MOUTH_TOP_INNER] = [0.50, 0.70 - mar_gap / 2, 0.0]
    landmarks[FL.MOUTH_BOTTOM_INNER] = [0.50, 0.70 + mar_gap / 2, 0.0]
    landmarks[FL.FOREHEAD_TOP] = [0.50, 0.20, 0.0]
    landmarks[FL.LEFT_CHEEK] = [0.35, 0.55, 0.0]
    landmarks[FL.RIGHT_CHEEK] = [0.65, 0.55, 0.0]
    face = FaceData(landmarks=landmarks, bbox=(0, 0, FRAME_SIZE - 1, FRAME_SIZE - 1),
                    crop=np.zeros((10, 10, 3), dtype=np.uint8), has_iris=True)
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    return FrameContext(frame=frame, timestamp=t, frame_index=0, fps=30.0, face=face)


def test_talking_gate_drops_and_restores_cheek_rois():
    be = classical_mod.ClassicalBackend(window_seconds=6.5, method="chrom",
                                        talk_delta_threshold=0.05)
    orig_roi_patch = classical_mod.roi_patch
    calls: list = []

    def spy(ctx, idx, radius_frac=0.10):
        calls.append(idx)
        return np.full((4, 4, 3), 100.0, dtype=np.float64)

    classical_mod.roi_patch = spy
    try:
        # First call: no prior MAR to diff against -> gate can't trigger yet
        # -> all three ROIs sampled.
        calls.clear()
        be.update(_make_ctx(mar_gap=0.02, t=0.0))
        assert set(calls) == {FL.FOREHEAD_TOP, FL.LEFT_CHEEK, FL.RIGHT_CHEEK}, calls

        # Second call: mouth abruptly opens (MAR jumps from ~0.10 to ~0.25,
        # delta ~0.15 > threshold 0.05) -> talking detected -> forehead only.
        calls.clear()
        be.update(_make_ctx(mar_gap=0.05, t=1 / 30.0))
        assert calls == [FL.FOREHEAD_TOP], f"expected forehead-only during talking, got {calls}"

        # Third call: mouth holds the SAME shape as the prior call (delta=0)
        # -> not talking anymore -> all three ROIs sampled again.
        calls.clear()
        be.update(_make_ctx(mar_gap=0.05, t=2 / 30.0))
        assert set(calls) == {FL.FOREHEAD_TOP, FL.LEFT_CHEEK, FL.RIGHT_CHEEK}, calls
    finally:
        classical_mod.roi_patch = orig_roi_patch
    print("[classical-talking-gate-test] cheeks dropped during talking, restored when steady OK")


def test_small_mouth_movement_does_not_trigger_gate():
    """A small, sub-threshold MAR delta (natural micro-movement, not speech)
    must not drop the cheeks -- only clearly-talking-sized deltas should."""
    be = classical_mod.ClassicalBackend(window_seconds=6.5, method="chrom",
                                        talk_delta_threshold=0.05)
    orig_roi_patch = classical_mod.roi_patch
    calls: list = []

    def spy(ctx, idx, radius_frac=0.10):
        calls.append(idx)
        return np.full((4, 4, 3), 100.0, dtype=np.float64)

    classical_mod.roi_patch = spy
    try:
        calls.clear()
        be.update(_make_ctx(mar_gap=0.02, t=0.0))
        calls.clear()
        # mar_gap 0.02 -> 0.021 is a tiny (~0.005) MAR delta, well under 0.05.
        be.update(_make_ctx(mar_gap=0.021, t=1 / 30.0))
        assert set(calls) == {FL.FOREHEAD_TOP, FL.LEFT_CHEEK, FL.RIGHT_CHEEK}, (
            f"small mouth movement should not have dropped cheeks, got {calls}")
    finally:
        classical_mod.roi_patch = orig_roi_patch
    print("[classical-talking-gate-test] sub-threshold movement leaves cheeks in OK")


def main():
    """Run all talking-gate tests."""
    test_talking_gate_drops_and_restores_cheek_rois()
    test_small_mouth_movement_does_not_trigger_gate()
    print("[classical-talking-gate-test] OK")


if __name__ == "__main__":
    main()
