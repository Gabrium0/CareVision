"""Hand/limb tremor detection via wrist oscillation spectrum.

Method: track each wrist position (pose landmarks), remove slow drift,
and look for a dominant oscillation in 3-12 Hz — the band covering
physiological, essential, and Parkinsonian (4-6 Hz) tremor. Position is
normalized by shoulder width so distance to camera doesn't matter.

Reliability: MEDIUM when the hand is visible and reasonably still; camera
fps limits the top of the band (needs ~24+ fps for 12 Hz).
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, dominant_frequency, bandpass
from extractors import pose as P


@register("tremor")
class Tremor(DetectionModule):
    """Hand/limb tremor detection via wrist oscillation spectrum."""
    interval = 0.0
    requires = ("pose",)
    window_seconds = 5.0

    def __init__(self, **params):
        super().__init__(**params)
        self.bufs = {"left": TimedBuffer(self.window_seconds),
                     "right": TimedBuffer(self.window_seconds)}

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        sw = abs(lm[P.L_SHOULDER, 0] - lm[P.R_SHOULDER, 0]) + 1e-6
        results = []
        for side, wi in (("left", P.L_WRIST), ("right", P.R_WRIST)):
            if lm[wi, 3] < 0.6:
                continue
            self.bufs[side].push(ctx.timestamp, lm[wi, 0] / sw)
            rs = self.bufs[side].resampled(fs=30.0)
            if rs is None or self.bufs[side].span() < 3.0:
                continue
            signal, fs = rs
            filt = bandpass(signal, fs, 3.0, min(12.0, fs / 2 - 1))
            if filt is None:
                continue
            amp = float(np.std(filt))
            dom = dominant_frequency(filt, fs, 3.0, min(12.0, fs / 2 - 1))
            if dom is None or amp < 0.02:
                continue
            freq, prom = dom
            conf = float(min(0.8, prom * 3 * min(1.0, amp * 15)))
            if conf < 0.25:
                continue
            sev = Severity.NOTICE if freq < 4 or freq > 7 else Severity.WARNING
            results.append(self.result(
                f"tremor_{side}", round(freq, 1), conf, sev,
                f"{side.title()} hand tremor ~{freq:.1f} Hz detected", ttl=6.0))
        return results or None
