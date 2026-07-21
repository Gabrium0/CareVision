"""Dry / cracked lips screening (dehydration proxy).

Method: on the lip region, dryness raises local texture (cracks) and
reduces saturation/redness compared to healthy lips. We combine high
texture energy with low saturation inside the lip polygon.

Reliability: LOW; a soft hydration-reminder trigger, not diagnostic.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import face_color_plane, face_skin_region, polygon_mask
from extractors import face_landmarks as FL


@register("dry_lips")
class DryLips(DetectionModule):
    """Dry / cracked lips screening (dehydration proxy)."""
    interval = 3.0
    requires = ("face",)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        region = face_skin_region(ctx)
        if region is None:
            return None
        frame, _, (x1, y1, _, _) = region
        px = ctx.face_px() - np.array([x1, y1], dtype=np.float32)
        mask = polygon_mask(frame.shape, px[FL.OUTER_LIPS])
        if mask.sum() < 120:
            return None
        m = mask > 0
        gray = face_color_plane(ctx, "gray").astype(np.float32)
        hsv = face_color_plane(ctx, "hsv")
        blur = cv2.GaussianBlur(gray, (0, 0), 2)
        texture = np.abs(gray - blur)[m].mean()
        sat = float(hsv[:, :, 1][m].mean())
        # high texture + low saturation => dry/cracked
        dryness = texture / 12.0 + (60.0 - min(sat, 60.0)) / 60.0
        if dryness < 1.0:
            return None
        conf = float(min(0.55, (dryness - 1.0)))
        return self.result(
            "lip_dryness", round(float(dryness), 2), conf, Severity.NOTICE,
            "Lips look dry/cracked (consider a hydration reminder)", ttl=20.0)
