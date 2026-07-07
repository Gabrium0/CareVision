"""Bradykinesia (slowness of movement) screening.

Method: track overall body-landmark speed (median wrist/arm velocity,
normalized by shoulder width). Sustained low movement velocity during
active periods — combined with the motion extractor showing the person is
present and not simply resting — is reported as slowed movement.

Reliability: LOW-MEDIUM; a coarse behavioral indicator.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import pose as P


@register("bradykinesia")
class Bradykinesia(DetectionModule):
    interval = 0.0
    requires = ("pose",)
    window_seconds = 6.0

    def __init__(self, **params):
        super().__init__(**params)
        self.speeds = TimedBuffer(self.window_seconds)
        self._prev = None

    def process(self, ctx: FrameContext):
        lm = ctx.pose.landmarks
        sw = abs(lm[P.L_SHOULDER, 0] - lm[P.R_SHOULDER, 0]) + 1e-6
        pts = lm[[P.L_WRIST, P.R_WRIST, P.L_ELBOW, P.R_ELBOW], :2]
        if self._prev is not None:
            dt = max(ctx.timestamp - self._prev[1], 1e-3)
            speed = float(np.median(np.linalg.norm(pts - self._prev[0], axis=1)) / sw / dt)
            self.speeds.push(ctx.timestamp, speed)
        self._prev = (pts, ctx.timestamp)

        if self.speeds.span() < 5.0:
            return None
        _, v = self.speeds.arrays()
        # only meaningful when there is *some* activity in the scene
        if ctx.motion_energy < 0.6:
            return None
        median_speed = float(np.median(v))
        if median_speed > 0.15:
            return None
        conf = float(min(0.5, (0.15 - median_speed) * 3))
        return self.result(
            "movement_speed", round(median_speed, 3), conf, Severity.NOTICE,
            "Movements look slowed (screening for bradykinesia)", ttl=8.0)
