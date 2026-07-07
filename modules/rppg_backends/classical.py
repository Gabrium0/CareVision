"""Classical rPPG backend: forehead green-channel bandpass + FFT.

This is the original, dependency-light implementation (numpy/scipy only):
sample the forehead skin patch, take the green channel mean, buffer it,
resample, bandpass 0.7-3 Hz, and read the dominant frequency. HRV RMSSD/
SDNN come from inter-beat intervals of the filtered waveform.

Reliability: medium; sensitive to lighting, motion, and skin tone.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from modules._util import (TimedBuffer, dominant_frequency, bandpass,
                           peak_intervals, roi_patch)
from extractors import face_landmarks as FL
from .base import RPPGBackend


class ClassicalBackend(RPPGBackend):
    label = "classical"
    available = True

    def __init__(self, window_seconds: float = 12.0):
        self.window_seconds = window_seconds
        self.buf = TimedBuffer(window_seconds)

    def update(self, ctx: FrameContext) -> None:
        patch = roi_patch(ctx, FL.FOREHEAD_TOP, radius_frac=0.10)
        if patch is not None and patch.size:
            self.buf.push(ctx.timestamp, float(patch[:, :, 1].mean()))  # green

    def compute(self) -> dict | None:
        rs = self.buf.resampled(fs=30.0)
        if rs is None or self.buf.span() < 6.0:
            return None
        signal, fs = rs
        filt = bandpass(signal, fs, 0.7, 3.0)
        if filt is None:
            return None
        dom = dominant_frequency(filt, fs, 0.7, 3.0)
        if dom is None:
            return None
        freq, prominence = dom
        bpm = freq * 60.0
        fill = min(1.0, self.buf.span() / self.window_seconds)
        conf = round(min(1.0, prominence * 3.0) * fill, 2)

        out = {"bpm": round(bpm, 1), "confidence": conf}
        rr = peak_intervals(filt, fs, min_distance_s=0.4)
        if len(rr) >= 4:
            rr_ms = rr * 1000.0
            out["hrv_rmssd_ms"] = round(float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))), 1)
            out["hrv_sdnn_ms"] = round(float(np.std(rr_ms)), 1)
        return out
