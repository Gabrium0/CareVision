"""Tests for on-device rPPG: the native iPad app streams raw-pixel ROI colour
means as `rppg_samples`, and the backend feeds them straight into the classical
HR / SpO2 buffers instead of sampling the JPEG-decoded frame.

Covers the validator, the camera ingest/drain (clock-mapped, bounded), and the
device-first branch in both samplers — including that it bypasses roi_patch (so
it works with no face anchor / no pixels) and that a synthetic colour sinusoid
still yields the right BPM through the unchanged compute() path.

Run standalone:  python tests/ipad_rppg_samples_test.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext
from core.ipad_camera import IPadCamera, parse_rppg_row
from modules.rppg_backends.classical import ClassicalBackend
from modules.spo2 import SpO2


class _FakeLink:
    """Minimal link stub so IPadCamera.diagnostics() can call status()."""
    def status(self):
        return {"relay": "connected"}


def _ctx(samples):
    ctx = FrameContext(frame=np.zeros((4, 4, 3), np.uint8), timestamp=samples[0][0],
                       frame_index=0, fps=20.0)
    ctx.extras["rppg_samples"] = samples
    return ctx


def test_parse_rppg_row_accepts_valid_rejects_junk():
    assert parse_rppg_row([1.5, 10, 20, 30, 7]) == (1.5, (10.0, 20.0, 30.0), 7)
    assert parse_rppg_row([2.0, 0, 0, 0]) == (2.0, (0.0, 0.0, 0.0), 0)  # n optional
    # Rejections.
    assert parse_rppg_row([1.0, 10, 20]) is None            # too short
    assert parse_rppg_row("nope") is None
    assert parse_rppg_row([1.0, 10, 20, 300]) is None       # channel out of range
    assert parse_rppg_row([1.0, -1, 20, 30]) is None        # negative channel
    assert parse_rppg_row([float("inf"), 1, 2, 3]) is None  # non-finite time
    assert parse_rppg_row([1.0, "a", 2, 3]) is None         # non-numeric
    # A negative pixel count is clamped, not rejected.
    assert parse_rppg_row([1.0, 5, 6, 7, -3]) == (1.0, (5.0, 6.0, 7.0), 0)
    print("[ipad-rppg-test] parse_rppg_row validation OK")


def test_camera_ingest_and_drain_maps_and_bounds():
    cam = IPadCamera(link=_FakeLink())
    cam.ingest_rppg_samples({"s": [[100.0, 10, 20, 30, 5], [100.05, 11, 21, 31, 6]]})
    drained = cam.drain_rppg_samples()
    assert len(drained) == 2, drained
    for ts, rgb, n in drained:
        assert isinstance(ts, float) and math.isfinite(ts)
        assert len(rgb) == 3 and n >= 0
    assert cam.drain_rppg_samples() == []          # cleared after drain
    diag = cam.diagnostics()
    assert diag["ondevice_rppg"] is True and diag["ondevice_rppg_samples"] == 2

    # Junk rows are dropped; the pending buffer is bounded.
    cam2 = IPadCamera(link=object())
    cam2.ingest_rppg_samples({"s": [[1.0, 999, 0, 0]]})   # out-of-range -> dropped
    assert cam2.drain_rppg_samples() == []
    big = {"s": [[float(i), 1, 2, 3, 1] for i in range(2000)]}
    cam2.ingest_rppg_samples(big)                          # one message capped at 120
    assert len(cam2.drain_rppg_samples()) <= 120
    print("[ipad-rppg-test] camera ingest/drain map + bound OK")


def test_classical_device_branch_bypasses_pixels():
    backend = ClassicalBackend(method="green")
    ctx = _ctx([(50.0, (120.0, 130.0, 110.0), 400)])
    backend.update(ctx)
    assert len(backend.buf) == 1
    diag = backend.diagnostics()
    # roi_patch was never called: no face anchor would otherwise increment the
    # no-ROI rejection counter. Device samples skip pixels entirely.
    assert diag["accepted"] == 1
    assert diag["rejected"]["no_roi"] == 0
    print("[ipad-rppg-test] classical device branch bypasses roi_patch OK")


def test_classical_device_series_recovers_bpm():
    """A synthetic 72 bpm colour sinusoid still yields ~72 bpm through the
    unchanged compute() path — proving the device series is a drop-in."""
    backend = ClassicalBackend(method="green")
    fs, seconds, hz = 30.0, 12.0, 1.2      # 1.2 Hz == 72 bpm
    n = int(fs * seconds)
    for i in range(n):
        t = 50.0 + i / fs
        g = 130.0 + 8.0 * math.sin(2 * math.pi * hz * t)
        backend.update(_ctx([(t, (110.0, g, 120.0), 400)]))
    reading = backend.compute()
    assert reading is not None, "compute() should be ready after 12 s of samples"
    assert 60.0 <= reading["bpm"] <= 84.0, reading
    print(f"[ipad-rppg-test] classical device series -> {reading['bpm']} bpm OK")


def test_spo2_device_branch_populates_without_frame():
    spo2 = SpO2()
    for i in range(10):
        t = 70.0 + i * 0.1
        spo2._sample(_ctx([(t, (120.0, 100.0, 90.0), 300)]))
    assert len(spo2.buf) == 10
    assert spo2._source_mode == "compressed"   # no depth over the iPad link
    print("[ipad-rppg-test] spo2 device branch populates buffer OK")


def main():
    test_parse_rppg_row_accepts_valid_rejects_junk()
    test_camera_ingest_and_drain_maps_and_bounds()
    test_classical_device_branch_bypasses_pixels()
    test_classical_device_series_recovers_bpm()
    test_spo2_device_branch_populates_without_frame()
    print("[ipad-rppg-test] OK")


if __name__ == "__main__":
    main()
