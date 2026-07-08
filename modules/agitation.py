"""Agitation / restlessness screening.

Method: combine high, sustained upper-body motion with frequent posture
changes. We use the shared motion energy plus wrist-speed variance over a
short window; elevated, erratic movement without locomotion (distinct from
purposeful activity) reads as restlessness.

Reliability: LOW-MEDIUM behavioral indicator.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import pose as P


@register("agitation")
class Agitation(DetectionModule):
    """Agitation / restlessness screening."""
    interval = 0.0
    requires = ("pose",)
    window_seconds = 10.0

    def __init__(self, **params):
        super().__init__(**params)
        self.speed = TimedBuffer(self.window_seconds)
        self._prev = None

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        sw = abs(lm[P.L_SHOULDER, 0] - lm[P.R_SHOULDER, 0]) + 1e-6
        wrists = lm[[P.L_WRIST, P.R_WRIST], :2]
        if self._prev is not None:
            dt = max(ctx.timestamp - self._prev[1], 1e-3)
            v = float(np.mean(np.linalg.norm(wrists - self._prev[0], axis=1)) / sw / dt)
            self.speed.push(ctx.timestamp, v)
        self._prev = (wrists, ctx.timestamp)

        if self.speed.span() < 8.0:
            return None
        _, v = self.speed.arrays()
        mean_v, var_v = float(np.mean(v)), float(np.var(v))
        if mean_v > 0.4 and var_v > 0.15:
            conf = float(min(0.6, mean_v))
            return self.result(
                "agitation", round(mean_v, 2), conf, Severity.NOTICE,
                "Restless / agitated movement pattern", ttl=8.0)
        return None
