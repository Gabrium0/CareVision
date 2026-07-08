"""Perspiration screening from forehead specular highlights.

Method: sweat makes skin glossy, producing bright, low-saturation specular
spots. On the forehead patch we measure the fraction of near-specular
pixels (high value, low saturation) relative to a rolling baseline.

Reliability: LOW; confounded by oily skin and direct light.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import roi_patch
from extractors import face_landmarks as FL


@register("sweating")
class Sweating(DetectionModule):
    """Perspiration screening from forehead specular highlights."""
    interval = 1.5
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.baseline = None

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        patch = roi_patch(ctx, FL.FOREHEAD_TOP, radius_frac=0.12)
        if patch is None or patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        spec = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 60)
        frac = float(spec.mean())
        if self.baseline is None:
            self.baseline = frac
            return None
        self.baseline = 0.98 * self.baseline + 0.02 * frac
        excess = frac - self.baseline
        if excess < 0.05:
            return None
        conf = float(min(0.6, excess * 4))
        return self.result(
            "sweat_gloss", round(excess, 3), conf, Severity.NOTICE,
            "Forehead looks glossy (possible sweating/fever — screening only)",
            ttl=12.0)
