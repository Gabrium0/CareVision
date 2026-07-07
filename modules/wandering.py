"""Wandering / pacing detection (disorientation, agitation).

Method: track the person's horizontal body-center position over a couple of
minutes. Repetitive back-and-forth traversal of the scene (many direction
reversals covering a wide horizontal range) indicates pacing/wandering.

Reliability: MEDIUM within a single camera view.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import pose as P


@register("wandering")
class Wandering(DetectionModule):
    interval = 0.5
    requires = ("pose",)

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(120.0)

    def process(self, ctx: FrameContext):
        lm = ctx.pose.landmarks
        if lm[P.L_HIP, 3] < 0.4 or lm[P.R_HIP, 3] < 0.4:
            return None
        cx = float((lm[P.L_HIP, 0] + lm[P.R_HIP, 0]) / 2.0)
        self.buf.push(ctx.timestamp, cx)
        if self.buf.span() < 60.0 or len(self.buf) < 30:
            return None
        _, xs = self.buf.arrays()
        rng = float(xs.max() - xs.min())
        # count direction reversals on a smoothed trajectory
        smooth = np.convolve(xs, np.ones(5) / 5, mode="valid")
        reversals = int(np.sum(np.diff(np.sign(np.diff(smooth))) != 0))
        if rng > 0.3 and reversals >= 6:
            conf = float(min(0.7, reversals / 20))
            return self.result(
                "pacing", reversals, conf, Severity.NOTICE,
                f"Repetitive pacing/wandering ({reversals} reversals)", ttl=20.0)
        return None
