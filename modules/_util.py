"""Shared helpers for detection modules: timed buffers, spectral analysis,
face-ROI sampling. Underscore prefix keeps the registry from importing it
as a module package.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


class TimedBuffer:
    """Rolling (timestamp, value) buffer with a fixed time horizon."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.t: deque[float] = deque()
        self.v: deque = deque()

    def push(self, timestamp: float, value) -> None:
        """Append one sample and discard anything older than the time horizon."""
        self.t.append(timestamp)
        self.v.append(value)
        while self.t and timestamp - self.t[0] > self.seconds:
            self.t.popleft()
            self.v.popleft()

    def __len__(self) -> int:
        return len(self.t)

    def span(self) -> float:
        """Time covered by the current buffer, in seconds."""
        return (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return timestamps and values as float arrays for signal processing."""
        return np.asarray(self.t, dtype=np.float64), np.asarray(self.v, dtype=np.float64)

    def resampled(self, fs: float) -> tuple[np.ndarray, float] | None:
        """Uniformly resample values at fs Hz; returns (signal, fs) or None."""
        if len(self.t) < 8 or self.span() < 2.0:
            return None
        t, v = self.arrays()
        t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
        if len(t_uniform) < 8:
            return None
        return np.interp(t_uniform, t, v), fs


def dominant_frequency(signal: np.ndarray, fs: float,
                       fmin: float, fmax: float) -> tuple[float, float] | None:
    """Strongest frequency in [fmin, fmax] Hz.

    Returns (frequency_hz, prominence 0..1) where prominence is peak power
    over total band power — a crude quality/confidence measure.
    """
    sig = signal - np.mean(signal)
    n = len(sig)
    if n < 16:
        return None
    # Windowing reduces FFT edge artifacts on short rolling buffers.
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(sig * window)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if band.sum() < 3 or spectrum[band].sum() <= 0:
        return None
    band_power = spectrum[band]
    peak_idx = int(np.argmax(band_power))
    # Treat one sharp spectral peak as higher quality than diffuse band energy.
    prominence = float(band_power[peak_idx] / band_power.sum())
    return float(freqs[band][peak_idx]), prominence


def bandpass(signal: np.ndarray, fs: float, fmin: float, fmax: float,
             order: int = 3) -> np.ndarray | None:
    """Butterworth band-pass filter a 1-D signal (or None if too short)."""
    nyq = fs / 2.0
    # filtfilt needs enough samples for padding; returning None avoids noisy startup values.
    if fmax >= nyq or len(signal) < 3 * (order + 1):
        return None
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, signal)


def peak_intervals(signal: np.ndarray, fs: float,
                   min_distance_s: float) -> np.ndarray:
    """Inter-peak intervals in seconds (for HRV from a pulse waveform)."""
    # min_distance_s encodes the fastest plausible beat rate and suppresses double counts.
    peaks, _ = find_peaks(signal, distance=max(1, int(min_distance_s * fs)))
    if len(peaks) < 3:
        return np.array([])
    return np.diff(peaks) / fs


def roi_patch(ctx, center_idx: int, radius_frac: float = 0.04) -> np.ndarray | None:
    """Square skin patch around a face landmark; radius relative to face width."""
    px = ctx.face_px()
    if px is None:
        return None
    x1, y1, x2, y2 = ctx.face.bbox
    r = max(2, int(radius_frac * (x2 - x1)))
    cx, cy = px[center_idx].astype(int)
    ax1, ay1 = max(0, cx - r), max(0, cy - r)
    ax2, ay2 = min(ctx.w, cx + r), min(ctx.h, cy + r)
    patch = ctx.frame[ay1:ay2, ax1:ax2]
    return patch if patch.size else None


def polygon_mask(shape_hw: tuple, points_px: np.ndarray) -> np.ndarray:
    """Binary mask (uint8) of the polygon given by points_px."""
    mask = np.zeros(shape_hw[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points_px.astype(np.int32)], 255)
    return mask


def face_skin_mask(ctx) -> np.ndarray | None:
    """Face-oval mask minus eyes and mouth — 'skin only' pixels."""
    from extractors import face_landmarks as FL
    px = ctx.face_px()
    if px is None:
        return None
    mask = polygon_mask(ctx.frame.shape, px[FL.FACE_OVAL])
    for hole in (FL.LEFT_EYE_RING, FL.RIGHT_EYE_RING, FL.OUTER_LIPS):
        cv2.fillPoly(mask, [px[hole].astype(np.int32)], 0)
    return mask
