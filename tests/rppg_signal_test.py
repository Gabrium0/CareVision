"""Unit tests for the low-light + chrominance rPPG helpers in modules._util.

Runs standalone (no pytest needed):  python tests/rppg_signal_test.py

Covers:
- low_light_factor: monotonic, clamped to [0, 1], 0 when very dark, 1 when lit.
- chrom / pos: recover a known synthetic pulse (72 bpm) even under a strong
  brightness ramp, which is exactly the illumination robustness a raw
  green-channel mean lacks.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules._util import (low_light_factor, patch_brightness, chrom, pos,
                           bandpass, dominant_frequency)


def test_low_light_factor():
    """0 in the dark, 1 when lit, monotonic non-decreasing, clamped to [0,1]."""
    assert low_light_factor(0.0) == 0.0
    assert low_light_factor(-50.0) == 0.0        # clamped, never negative
    assert low_light_factor(25.0) == 0.0         # dark floor
    assert low_light_factor(90.0) == 1.0         # good level
    assert low_light_factor(255.0) == 1.0        # clamped, never > 1
    mid = low_light_factor((25.0 + 90.0) / 2.0)
    assert 0.4 < mid < 0.6                        # ~0.5 halfway up the ramp
    samples = [low_light_factor(b) for b in range(0, 260, 5)]
    assert all(b >= a - 1e-9 for a, b in zip(samples, samples[1:]))  # monotonic
    print("[rppg-test] low_light_factor OK")


def test_patch_brightness():
    """Luminance of a mid-grey BGR patch is ~128; a black patch is ~0."""
    grey = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert abs(patch_brightness(grey) - 128.0) < 1.0
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    assert patch_brightness(black) < 1.0
    print("[rppg-test] patch_brightness OK")


def _synthetic_rgb(fs=30.0, seconds=10.0, bpm=72.0):
    """(N,3) R,G,B means carrying a `bpm` pulse under a strong brightness ramp."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
    ramp = 1.0 + 0.6 * (t / t[-1])               # slow illumination drift x1.6
    base = np.array([180.0, 150.0, 120.0])       # R,G,B skin-ish DC
    weight = np.array([0.3, 1.0, 0.2])           # green carries most pulse
    rgb = ramp[:, None] * (base[None, :] + 3.0 * weight[None, :] * pulse[:, None])
    return rgb, fs


def _recovered_bpm(signal, fs):
    filt = bandpass(signal, fs, 0.7, 3.0)
    assert filt is not None
    dom = dominant_frequency(filt, fs, 0.7, 3.0)
    assert dom is not None
    return dom[0] * 60.0


def test_chrom_pos_recover_pulse():
    """CHROM and POS recover ~72 bpm despite a strong brightness ramp."""
    rgb, fs = _synthetic_rgb(bpm=72.0)
    for name, fn in (("chrom", chrom), ("pos", pos)):
        sig = fn(rgb)
        assert sig is not None, f"{name} returned None"
        bpm = _recovered_bpm(sig, fs)
        assert abs(bpm - 72.0) < 6.0, f"{name} recovered {bpm:.1f} bpm, expected ~72"
        print(f"[rppg-test] {name} recovered {bpm:.1f} bpm OK")


def main():
    """Run all helper tests."""
    test_low_light_factor()
    test_patch_brightness()
    test_chrom_pos_recover_pulse()
    print("[rppg-test] OK")


if __name__ == "__main__":
    main()
