"""Unit test for the temporal color-pooling helper (modules/_util.py::
pooled_skin_sample) and its use in modules/skin_color.py: averaging a
near-stationary subject's chroma sample over a short rolling window should
recover signal-to-noise lost to MJPEG 4:2:0 chroma subsampling, so a
pallor/flushing/cyanosis decision driven by the pooled value should not
flip on a single noisy frame the way one driven by the raw per-frame value
would.

Run standalone:  python tests/color_pooling_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules._util import TimedBuffer, pooled_skin_sample


def test_pooled_sample_is_mean_of_window():
    buf = TimedBuffer(seconds=4.0)
    samples = [np.array([0.30, 0.35, 0.35]),
               np.array([0.32, 0.34, 0.34]),
               np.array([0.28, 0.36, 0.36])]
    pooled = None
    for i, s in enumerate(samples):
        pooled = pooled_skin_sample(buf, s, timestamp=i * 1.0)
    expected = np.mean(samples, axis=0)
    assert np.allclose(pooled, expected), f"expected {expected}, got {pooled}"
    print("[color-pooling-test] pooled sample equals window mean OK")


def test_single_noisy_frame_is_damped_by_the_window():
    """A short run of stable frames followed by one noisy outlier should move
    the pooled value far less than the raw value would."""
    buf = TimedBuffer(seconds=4.0)
    stable = np.array([0.33, 0.33, 0.34])
    for i in range(5):
        pooled_skin_sample(buf, stable, timestamp=i * 0.5)
    outlier = np.array([0.10, 0.50, 0.40])   # a single MJPEG-noise-corrupted frame
    pooled = pooled_skin_sample(buf, outlier, timestamp=2.5)

    raw_jump = np.linalg.norm(outlier - stable)
    pooled_jump = np.linalg.norm(pooled - stable)
    assert pooled_jump < raw_jump * 0.5, (
        f"pooled value should be damped well below the raw jump "
        f"({pooled_jump:.3f} vs raw {raw_jump:.3f})")
    print("[color-pooling-test] single noisy frame damped by the pooling window OK")


def test_old_samples_age_out_of_the_window():
    buf = TimedBuffer(seconds=4.0)
    pooled_skin_sample(buf, np.array([1.0, 0.0, 0.0]), timestamp=0.0)
    # Jump far enough ahead that the first sample falls outside the window.
    pooled = pooled_skin_sample(buf, np.array([0.0, 1.0, 0.0]), timestamp=10.0)
    assert np.allclose(pooled, [0.0, 1.0, 0.0]), (
        f"expected only the fresh sample once the old one ages out, got {pooled}")
    print("[color-pooling-test] stale samples age out of the pooling window OK")


def main():
    """Run all color-pooling tests."""
    test_pooled_sample_is_mean_of_window()
    test_single_noisy_frame_is_damped_by_the_window()
    test_old_samples_age_out_of_the_window()
    print("[color-pooling-test] OK")


if __name__ == "__main__":
    main()
