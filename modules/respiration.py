"""Respiratory rate from shoulder vertical oscillation.

Method: track mean shoulder y-position over a long window, bandpass to
0.1-0.5 Hz (6-30 breaths/min), take dominant frequency. Falls back to
face-box vertical drift if pose shoulders are unavailable.

Reliability: medium; needs a fairly still subject and visible torso.
"""
from __future__ import annotations

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, dominant_frequency, bandpass
from extractors import pose as P


@register("respiration")
class Respiration(DetectionModule):
    interval = 0.0
    requires = ("pose",)
    window_seconds = 25.0

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(self.window_seconds)

    def process(self, ctx: FrameContext):
        lm = ctx.pose.landmarks
        if lm[P.L_SHOULDER, 3] < 0.5 or lm[P.R_SHOULDER, 3] < 0.5:
            return None
        shoulder_y = float((lm[P.L_SHOULDER, 1] + lm[P.R_SHOULDER, 1]) / 2.0)
        self.buf.push(ctx.timestamp, shoulder_y)

        rs = self.buf.resampled(fs=10.0)
        if rs is None or self.buf.span() < 15.0:
            return None
        signal, fs = rs
        filt = bandpass(signal, fs, 0.1, 0.5, order=2)
        if filt is None:
            return None
        dom = dominant_frequency(filt, fs, 0.1, 0.5)
        if dom is None:
            return None
        freq, prominence = dom
        brpm = freq * 60.0
        conf = round(min(1.0, prominence * 3.0) *
                     min(1.0, self.buf.span() / self.window_seconds), 2)
        sev = Severity.INFO
        msg = f"Respiration ~{brpm:.0f} breaths/min"
        if conf >= 0.35 and (brpm < 10 or brpm > 22):
            sev = Severity.NOTICE
            msg = f"Respiration ~{brpm:.0f} breaths/min (atypical)"
        return self.result("breaths_per_min", round(brpm, 1), conf, sev, msg, ttl=10.0)
