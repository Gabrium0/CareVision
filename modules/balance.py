"""Standing balance / postural sway.

Method: while standing (hips above knees, low vertical motion), track the
horizontal position of the body center (mid-hip). Excessive low-frequency
sway (0.1-1 Hz) relative to body width indicates instability / unsteadiness.

Reliability: MEDIUM.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, bandpass
from extractors import pose as P


@register("balance")
class Balance(DetectionModule):
    """Standing balance / postural sway."""
    interval = 0.0
    requires = ("pose",)
    window_seconds = 8.0

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(self.window_seconds)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        if lm[P.L_HIP, 3] < 0.5 or lm[P.R_HIP, 3] < 0.5:
            return None
        # crude standing check: knees below hips in image space
        if lm[P.L_KNEE, 3] > 0.5 and lm[P.L_KNEE, 1] < lm[P.L_HIP, 1]:
            return None
        shoulder_w = abs(lm[P.L_SHOULDER, 0] - lm[P.R_SHOULDER, 0]) + 1e-6
        mid_hip_x = (lm[P.L_HIP, 0] + lm[P.R_HIP, 0]) / 2.0
        self.buf.push(ctx.timestamp, mid_hip_x / shoulder_w)

        rs = self.buf.resampled(fs=15.0)
        if rs is None or self.buf.span() < 6.0:
            return None
        filt = bandpass(rs[0], 15.0, 0.1, 1.0, order=2)
        if filt is None:
            return None
        sway = float(np.std(filt))
        if sway < 0.06:
            return None
        conf = float(min(0.7, sway * 5))
        sev = Severity.WARNING if sway > 0.12 else Severity.NOTICE
        return self.result(
            "postural_sway", round(sway, 3), conf, sev,
            "Postural sway while standing (possible unsteadiness/fall risk)",
            ttl=8.0)
