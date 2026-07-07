"""Facial / eyelid swelling (edema) screening via slow contour drift.

Method: track normalized eye-opening height and cheek fullness (cheek-to-
jaw width ratio) against a long rolling baseline. Puffiness reduces eye
aperture and increases lower-face fullness. Only slow, sustained changes
are reported (fast changes are expression, not edema).

Reliability: LOW; longitudinal drift indicator.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import face_landmarks as FL


@register("facial_swelling")
class FacialSwelling(DetectionModule):
    interval = 2.0
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.base = None
        self.t0 = None

    def _features(self, ctx):
        px = ctx.face_px()
        face_w = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        eye_l = np.linalg.norm(px[159] - px[145]) / face_w
        eye_r = np.linalg.norm(px[386] - px[374]) / face_w
        cheek_w = np.linalg.norm(px[FL.LEFT_CHEEK] - px[FL.RIGHT_CHEEK]) / face_w
        return np.array([(eye_l + eye_r) / 2.0, cheek_w])

    def process(self, ctx: FrameContext):
        f = self._features(ctx)
        if self.t0 is None:
            self.t0, self.base = ctx.timestamp, f
            return None
        if ctx.timestamp - self.t0 < 30.0:      # long learning window
            self.base = 0.9 * self.base + 0.1 * f
            return None
        self.base = 0.999 * self.base + 0.001 * f
        eye_drop = (self.base[0] - f[0]) / (self.base[0] + 1e-6)
        cheek_gain = (f[1] - self.base[1]) / (self.base[1] + 1e-6)
        score = 0.5 * max(0, eye_drop) + 0.5 * max(0, cheek_gain)
        if score < 0.08:
            return None
        conf = float(min(0.5, score * 3))
        return self.result(
            "swelling", round(float(score), 3), conf, Severity.NOTICE,
            "Possible facial/eyelid puffiness vs baseline (screening only)",
            ttl=30.0)
