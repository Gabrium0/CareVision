"""Fall detection from body pose.

Method: track the torso orientation (shoulder-hip vector) and the vertical
position/height of the person's bounding box. A fall is a rapid transition
to a horizontal torso and/or a sudden large downward drop of the body
center, followed by the person remaining low. We require both a fast change
and a low-and-horizontal end state to reduce false positives from sitting.

Reliability: MEDIUM-HIGH for clear falls in view; the post-fall immobility
check is handled by the `unresponsive` module.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import pose as P


@register("fall")
class Fall(DetectionModule):
    interval = 0.0
    requires = ("pose",)

    def __init__(self, **params):
        super().__init__(**params)
        self.center_y = TimedBuffer(2.0)
        self._fallen_since = None

    def _torso_angle(self, lm):
        sh = (lm[P.L_SHOULDER, :2] + lm[P.R_SHOULDER, :2]) / 2.0
        hp = (lm[P.L_HIP, :2] + lm[P.R_HIP, :2]) / 2.0
        v = hp - sh
        # angle from vertical: 0 = upright, ~90 = lying down
        return float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))

    def process(self, ctx: FrameContext):
        lm = ctx.pose.landmarks
        for idx in (P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP):
            if lm[idx, 3] < 0.4:
                return None
        center_y = float((lm[P.L_HIP, 1] + lm[P.R_HIP, 1]) / 2.0)
        self.center_y.push(ctx.timestamp, center_y)
        angle = self._torso_angle(lm)

        # rapid downward motion of body center over ~0.5s
        rapid = False
        if self.center_y.span() > 0.4:
            _, ys = self.center_y.arrays()
            rapid = (ys[-1] - ys.min()) > 0.18       # dropped >18% of frame height

        horizontal = angle > 55.0
        low = center_y > 0.6                          # hips in lower part of frame

        if horizontal and low:
            if self._fallen_since is None:
                self._fallen_since = ctx.timestamp
            trigger = rapid or (ctx.timestamp - self._fallen_since > 1.0)
            if trigger:
                conf = 0.6 + (0.3 if rapid else 0.0)
                return self.result(
                    "fall", True, min(0.95, conf), Severity.ALERT,
                    "FALL DETECTED — person is down and horizontal", ttl=15.0)
        else:
            self._fallen_since = None
        return None
