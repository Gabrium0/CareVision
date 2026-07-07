"""Remote photoplethysmography (rPPG) heart rate + HRV.

Method: sample the forehead skin patch each frame, take the green channel
mean (strongest pulsatile signal in RGB), buffer it, resample uniformly,
bandpass to 0.7-3 Hz (42-180 bpm) and find the dominant frequency. HRV is
estimated from inter-beat intervals of the filtered waveform.

Reliability: medium. Sensitive to lighting, motion, and skin tone. Emitted
confidence tracks spectral prominence and buffer fill; treat as a trend
indicator, not a medical measurement.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import (TimedBuffer, dominant_frequency, bandpass,
                           peak_intervals, roi_patch)
from extractors import face_landmarks as FL


@register("heart_rate")
class HeartRate(DetectionModule):
    interval = 0.0
    requires = ("face",)
    window_seconds = 12.0

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(self.window_seconds)

    def process(self, ctx: FrameContext):
        patch = roi_patch(ctx, FL.FOREHEAD_TOP, radius_frac=0.10)
        if patch is None or patch.size == 0:
            return None
        self.buf.push(ctx.timestamp, float(patch[:, :, 1].mean()))  # green

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

        results = []
        sev = Severity.INFO
        msg = f"Heart rate ~{bpm:.0f} bpm"
        if conf >= 0.35 and (bpm < 50 or bpm > 110):
            sev = Severity.WARNING
            msg = f"Heart rate ~{bpm:.0f} bpm (outside typical resting range)"
        results.append(self.result("bpm", round(bpm, 1), conf, sev, msg, ttl=8.0))

        rr = peak_intervals(filt, fs, min_distance_s=0.4)
        if len(rr) >= 4:
            rmssd = float(np.sqrt(np.mean(np.diff(rr * 1000.0) ** 2)))  # ms
            results.append(self.result(
                "hrv_rmssd_ms", round(rmssd, 1), round(conf * 0.8, 2),
                Severity.INFO, f"HRV (RMSSD) ~{rmssd:.0f} ms", ttl=8.0))
        return results
