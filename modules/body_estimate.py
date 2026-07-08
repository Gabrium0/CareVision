"""Rough body-build proxy (shoulder-to-hip / width-to-height).

Method: from pose landmarks, compute shoulder width and torso height and
their ratio to the overall standing height. This is only a coarse build
descriptor and explicitly NOT a medical BMI. Emitted at low confidence and
labeled as an estimate.

Reliability: LOW by nature (monocular, no calibration). Included for
completeness of the detectionList; use only as a very rough proxy.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import pose as P


@register("body_estimate")
class BodyEstimate(DetectionModule):
    """Rough body-build proxy (shoulder-to-hip / width-to-height)."""
    interval = 2.0
    requires = ("pose",)

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(20.0)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        for idx in (P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP, P.L_ANKLE):
            if lm[idx, 3] < 0.5:
                return None
        shoulder_w = abs(lm[P.L_SHOULDER, 0] - lm[P.R_SHOULDER, 0])
        hip_w = abs(lm[P.L_HIP, 0] - lm[P.R_HIP, 0])
        height = abs(lm[P.L_ANKLE, 1] - lm[P.L_SHOULDER, 1]) + 1e-6
        ratio = (0.5 * (shoulder_w + hip_w)) / height
        self.buf.push(ctx.timestamp, ratio)
        if self.buf.span() < 10.0:
            return None
        _, v = self.buf.arrays()
        r = float(np.median(v))
        build = "slim" if r < 0.22 else ("average" if r < 0.30 else "broad")
        return self.result(
            "build", build, 0.3, Severity.INFO,
            f"Body build estimate: {build} (rough proxy, not BMI)", ttl=60.0)
