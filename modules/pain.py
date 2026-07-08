"""Pain expression / grimacing screening (action-unit proxies).

Method: pain faces (PSPI-style) combine brow lowering, eye tightening
(orbital narrowing), nose wrinkling and raised upper lip. We approximate
these with landmark distances: low brow-eye gap, reduced eye aperture, and
raised/tightened upper lip, each normalized by face size and compared to a
short relaxed baseline. The summed score maps to a pain likelihood.

Reliability: LOW-MEDIUM; screening prompt only.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import face_landmarks as FL


@register("pain")
class Pain(DetectionModule):
    """Pain expression / grimacing screening (action-unit proxies)."""
    interval = 0.5
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.base = None
        self.t0 = None

    def _aus(self, ctx):
        px = ctx.face_px()
        fw = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        brow = (np.linalg.norm(px[FL.LEFT_BROW[2]] - px[159]) +
                np.linalg.norm(px[FL.RIGHT_BROW[2]] - px[386])) / 2.0 / fw
        eye = (np.linalg.norm(px[159] - px[145]) +
               np.linalg.norm(px[386] - px[374])) / 2.0 / fw
        upper_lip = np.linalg.norm(px[FL.NOSE_TIP] - px[FL.MOUTH_TOP_INNER]) / fw
        return np.array([brow, eye, upper_lip])

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        f = self._aus(ctx)
        if self.t0 is None:
            self.t0, self.base = ctx.timestamp, f
            return None
        if ctx.timestamp - self.t0 < 8.0:
            self.base = 0.9 * self.base + 0.1 * f
            return None
        brow_lower = max(0, (self.base[0] - f[0]) / self.base[0])
        eye_tight = max(0, (self.base[1] - f[1]) / self.base[1])
        lip_raise = max(0, (self.base[2] - f[2]) / self.base[2])
        score = brow_lower + eye_tight + lip_raise
        if score < 0.25:
            return None
        conf = float(min(0.7, score))
        return self.result(
            "pain", round(float(score), 2), conf, Severity.WARNING,
            "Possible pain/grimacing expression (check on person)", ttl=6.0)
