"""Unit tests for the contactless rPPG SpO₂ module (modules/spo2.py).

Runs standalone (no pytest needed):  python tests/spo2_test.py

The synthetic generator injects a per-channel pulse of known fractional amplitude
on top of a slow brightness ramp, so after DC-normalized bandpassing each channel's
AC/DC ≈ its amplitude fraction and therefore ratio_of_ratios ≈ amp_num/amp_den. That
makes the ratio math directly assertable without a camera.

Covers:
- ratio_of_ratios: correct direction/monotonicity and its None guards.
- ratio_to_spo2: the linear map is decreasing in R.
- SpO2 module: gating below the sample/second floor, uncalibrated confidence
  suppression + label, display clamp, and a calibrated low-reading WARNING.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.spo2 import SpO2, channel_ac_dc, ratio_of_ratios, ratio_to_spo2

_AC_BAND = (0.7, 3.0)


def _pulsatile_rgb(fs=30.0, seconds=10.0, bpm=72.0,
                   amp=(0.04, 0.04, 0.02)):
    """(N,3) R,G,B means with a per-channel fractional pulse under a brightness ramp.

    Because bandpass strips the ramp/DC, each channel's std(AC)/mean(DC) ≈ its
    `amp` entry, so ratio_of_ratios(red, blue) ≈ amp_red / amp_blue.
    """
    n = int(fs * seconds)
    t = np.arange(n) / fs
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
    ramp = 1.0 + 0.6 * (t / t[-1])                 # slow illumination drift x1.6
    dc = np.array([180.0, 150.0, 120.0])           # R,G,B skin-ish DC
    amp = np.asarray(amp, dtype=np.float64)
    rgb = ramp[:, None] * dc[None, :] * (1.0 + amp[None, :] * pulse[:, None])
    return rgb, t


def _load_module_buffer(module, rgb, t, brightness=120.0):
    """Push synthetic samples straight into a module's buffer, bypassing the camera."""
    for row, ts in zip(rgb, t):
        module.buf.push(float(ts), np.asarray(row, dtype=np.float64))
        module._quality_events.append((float(ts), True))
    module._brightness = brightness
    module._fast_fed = True                        # skip _sample(ctx) in process()


def test_ratio_of_ratios_direction():
    """Higher red AC raises R; lower red AC lowers it (≈ amp_red/amp_blue)."""
    fs = 30.0
    high_red, _ = _pulsatile_rgb(fs=fs, amp=(0.06, 0.04, 0.02))
    low_red, _ = _pulsatile_rgb(fs=fs, amp=(0.02, 0.04, 0.02))
    r_high = ratio_of_ratios(high_red, fs, 0, 2, _AC_BAND)
    r_low = ratio_of_ratios(low_red, fs, 0, 2, _AC_BAND)
    assert r_high is not None and r_low is not None
    assert r_high > r_low                          # more red AC -> bigger ratio
    assert abs(r_high - 3.0) < 0.5                 # ≈ 0.06/0.02
    assert abs(r_low - 1.0) < 0.3                  # ≈ 0.02/0.02
    # SpO₂ = A − B·R decreases as R grows.
    assert ratio_to_spo2(r_high, 100.0, 5.0) < ratio_to_spo2(r_low, 100.0, 5.0)
    print("[spo2-test] ratio direction OK")


def test_ratio_none_guards():
    """Ratio helpers reject unusable channels (zero DC, too-short windows)."""
    fs = 30.0
    # A zero-DC channel is rejected by channel_ac_dc (division guard).
    zero_dc = np.zeros((300, 3))
    assert channel_ac_dc(zero_dc[:, 0], fs, _AC_BAND) is None
    assert ratio_of_ratios(np.ones((4, 3)), fs, 0, 2, _AC_BAND) is None   # too short
    assert ratio_of_ratios(np.ones((300, 2)), fs, 0, 2, _AC_BAND) is None  # wrong shape
    print("[spo2-test] ratio None guards OK")


def test_module_gates_below_sample_floor():
    """Too few seconds buffered -> process returns None (no reading)."""
    module = SpO2()
    rgb, t = _pulsatile_rgb(seconds=2.0)           # under required_seconds (6s)
    _load_module_buffer(module, rgb, t)
    assert module.process(None) is None
    module.close()
    print("[spo2-test] sample-floor gate OK")


def test_uncalibrated_is_low_confidence_and_labelled():
    """The shipped uncalibrated default emits a trend-only, near-zero-confidence read."""
    module = SpO2(window_seconds=10)
    assert module._calibrated is False
    rgb, t = _pulsatile_rgb(seconds=10.0)
    _load_module_buffer(module, rgb, t)
    results = module.process(None)
    assert results is not None
    spo2 = next(r for r in results if r.key == "spo2")
    assert spo2.confidence <= 0.1                  # placeholder coeffs never trusted
    assert "uncalibrated" in spo2.message
    assert module.display_min <= spo2.value <= module.display_max   # display clamp
    module.close()
    print("[spo2-test] uncalibrated suppression OK")


def test_calibrated_low_reading_warns():
    """With real coefficients tuned to a sub-92% reading + strong signal -> WARNING."""
    import tempfile

    fs = 30.0
    rgb, t = _pulsatile_rgb(fs=fs, seconds=10.0, amp=(0.06, 0.04, 0.02))
    ratio = ratio_of_ratios(rgb, fs, 0, 2, _AC_BAND)
    assert ratio is not None
    # Choose A so that SpO₂ = A − B·ratio lands at ~88 (< warn_below 92).
    b_coeff = 5.0
    a_coeff = 88.0 + b_coeff * ratio
    # Self-contained temp dir (pytest's tmp_path fixture is unreliable on this box).
    with tempfile.TemporaryDirectory() as d:
        cal = Path(d) / "cal.json"
        cal.write_text(json.dumps({"calibrated": True, "A": a_coeff, "B": b_coeff,
                                   "channels": ["red", "blue"]}), encoding="utf-8")
        module = SpO2(window_seconds=10, calibration_file=str(cal))
        assert module._calibrated is True
        _load_module_buffer(module, rgb, t, brightness=120.0)
        results = module.process(None)
    assert results is not None
    spo2 = next(r for r in results if r.key == "spo2")
    assert 87.0 <= spo2.value <= 89.0
    assert spo2.confidence >= module.warn_min_confidence
    assert spo2.severity.value == "warning"
    assert "low" in spo2.message.lower()
    module.close()
    print("[spo2-test] calibrated WARNING OK")


def test_debug_ratio_emitted_when_enabled():
    """debug_ratio=true adds a spo2_ratio result for calibration capture."""
    module = SpO2(window_seconds=10, debug_ratio=True)
    rgb, t = _pulsatile_rgb(seconds=10.0)
    _load_module_buffer(module, rgb, t)
    results = module.process(None)
    assert results is not None
    assert any(r.key == "spo2_ratio" for r in results)
    module.close()
    print("[spo2-test] debug_ratio emission OK")


if __name__ == "__main__":
    test_ratio_of_ratios_direction()
    test_ratio_none_guards()
    test_module_gates_below_sample_floor()
    test_uncalibrated_is_low_confidence_and_labelled()
    test_calibrated_low_reading_warns()
    test_debug_ratio_emitted_when_enabled()
    print("[spo2-test] all tests passed")
