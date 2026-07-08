"""Unit test for the median cross-call smoothing added to ClassicalBackend
(modules/rppg_backends/classical.py), mirroring OpenRPPGBackend's existing
`_bpm_history` pattern (modules/rppg_backends/openrppg.py).

Each compute() call independently FFT-picks a dominant frequency from
whatever's currently in the rolling buffer, so a single call landing on a
noisy/corrupted window can produce a wildly wrong raw estimate. This test
drives the backend through a real (green) beat -> corrupted beat -> real
beat sequence and asserts the *reported* (median-smoothed) bpm resists the
single bad pick, while a naive single-call reading would not.

Run standalone:  python tests/classical_smoothing_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.rppg_backends.classical import ClassicalBackend

FS = 30.0
GOOD_BPM = 72.0
BAD_BPM = 150.0
# Must exceed the hardcoded 6.0s minimum span compute() requires
# (see ClassicalBackend._snapshot), and each push must exceed window_seconds
# to fully evict older content so one compute() call sees a pure signal.
WINDOW_SECONDS = 6.5
PUSH_SECONDS = 7.0


def _push_pulse(be: ClassicalBackend, t0: float, bpm: float, seconds: float) -> float:
    """Push `seconds` of a synthetic clean pulse at `bpm` starting at t0;
    returns the next free timestamp."""
    n = int(FS * seconds)
    base = np.array([180.0, 150.0, 120.0])   # R,G,B skin-ish DC
    weight = np.array([0.3, 1.0, 0.2])       # green carries most of the pulse
    for i in range(n):
        t = t0 + i / FS
        pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
        rgb = base + 3.0 * weight * pulse
        be.buf.push(t, rgb)
    return t0 + seconds


def test_reported_bpm_resists_single_outlier_pick():
    be = ClassicalBackend(window_seconds=WINDOW_SECONDS, method="chrom", smoothing_window=5)
    t = 0.0

    # 1) Clean signal -> first raw pick should already be close to GOOD_BPM.
    t = _push_pulse(be, t, GOOD_BPM, PUSH_SECONDS)
    r1 = be.compute()
    assert r1 is not None, "expected a reading once the buffer has enough span"
    assert abs(r1["bpm"] - GOOD_BPM) < 3.0, f"first clean pick should be ~{GOOD_BPM}, got {r1['bpm']}"
    assert len(be._bpm_history) == 1

    # 2) Fully replace the buffer with a corrupted/wrong-frequency window
    #    (long enough to evict all the clean content) -> this call's raw pick
    #    should land on BAD_BPM and get appended to _bpm_history as an outlier.
    t = _push_pulse(be, t, BAD_BPM, PUSH_SECONDS)
    r2 = be.compute()
    assert r2 is not None
    raw_bad_pick = be._bpm_history[-1]
    assert abs(raw_bad_pick - BAD_BPM) < 5.0, (
        f"expected the corrupted window's raw pick to land near {BAD_BPM}, got {raw_bad_pick} "
        "-- test setup didn't actually produce an outlier")
    # Even with one bad raw pick in a 2-sample history, the median of
    # [72, 150] is their average (111) -- not yet "resisting" with only 2
    # points; the real resistance shows up once more good picks arrive below.

    # 3) Feed clean signal again for several more calls; the median-of-5
    #    should converge back to GOOD_BPM despite the one outlier still
    #    sitting in the smoothing window.
    last = None
    for _ in range(4):
        t = _push_pulse(be, t, GOOD_BPM, PUSH_SECONDS)
        last = be.compute()
        assert last is not None

    assert abs(last["bpm"] - GOOD_BPM) < 5.0, (
        f"median-smoothed bpm should have converged back to ~{GOOD_BPM} despite "
        f"one outlier pick, got {last['bpm']}")
    # The raw (unsmoothed) history still contains the outlier -- confirms the
    # test actually exercised resistance, not just a lucky clean run.
    assert any(abs(v - BAD_BPM) < 5.0 for v in be._bpm_history), (
        "outlier should still be present in the raw history the median is computed over")
    print(f"[classical-smoothing-test] reported bpm={last['bpm']} resisted outlier "
          f"pick of {raw_bad_pick:.1f} OK")


def test_smoothing_window_is_configurable():
    """smoothing_window=1 disables cross-call smoothing entirely (parity
    with how openrppg_smoothing_window=1 behaves)."""
    be = ClassicalBackend(window_seconds=WINDOW_SECONDS, method="chrom", smoothing_window=1)
    assert be._bpm_history.maxlen == 1
    print("[classical-smoothing-test] smoothing_window is configurable OK")


def main():
    """Run all classical-backend smoothing tests."""
    test_reported_bpm_resists_single_outlier_pick()
    test_smoothing_window_is_configurable()
    print("[classical-smoothing-test] OK")


if __name__ == "__main__":
    main()
