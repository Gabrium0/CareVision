"""Eye redness / conjunctivitis screening from the sclera region.

Method: mask the eye rings, isolate the sclera (bright, low-saturation
pixels inside the eye) and measure its redness (LAB a-channel). Elevated
redness across both eyes is reported.

Reliability: LOW-MEDIUM; needs decent resolution on the eyes.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import polygon_mask
from extractors import face_landmarks as FL


@register("eye_redness")
class EyeRedness(DetectionModule):
    interval = 1.5
    requires = ("face",)

    def _sclera_redness(self, ctx, ring_idx):
        px = ctx.face_px()
        mask = polygon_mask(ctx.frame.shape, px[ring_idx])
        if mask.sum() < 40:
            return None
        hsv = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        sclera = (mask > 0) & (val > 90) & (sat < 110)  # whitish part
        if sclera.sum() < 20:
            return None
        lab = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2LAB)
        return float(lab[:, :, 1][sclera].mean())       # a-channel; 128=neutral

    def process(self, ctx: FrameContext):
        vals = [self._sclera_redness(ctx, r)
                for r in (FL.LEFT_EYE_RING, FL.RIGHT_EYE_RING)]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        redness = float(np.mean(vals)) - 128.0     # >0 means reddish sclera
        if redness < 6:
            return None
        conf = float(min(0.7, (redness - 6) / 15))
        sev = Severity.NOTICE if redness < 12 else Severity.WARNING
        return self.result(
            "sclera_redness", round(redness, 1), conf, sev,
            "Eye redness above neutral (possible irritation/conjunctivitis)",
            ttl=15.0)
