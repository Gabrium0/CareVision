"""Bruise / discoloration screening on facial skin.

Method: bruises are bluish-purple to yellow-green patches that differ from
surrounding skin hue. Within the skin mask we flag connected regions whose
hue deviates toward purple/blue or green-yellow and that are darker than
baseline skin. Reported as a screening prompt only.

Reliability: LOW.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import face_skin_mask


@register("bruise")
class Bruise(DetectionModule):
    """Bruise / discoloration screening on facial skin."""
    interval = 3.0
    requires = ("face",)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        mask = face_skin_mask(ctx)
        if mask is None or mask.sum() < 800:
            return None
        hsv = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)      # 0..179
        v = hsv[:, :, 2].astype(np.float32)
        skin = mask > 0

        # purple/blue hues ~ 110-160; also darker than typical skin value
        purple = (h > 110) & (h < 160)
        dark = v < (v[skin].mean() - 1.2 * v[skin].std())
        cand = (purple & dark & skin).astype(np.uint8)

        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(cand)
        skin_area = skin.sum()
        big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.004 * skin_area]
        if not big:
            return None
        area = sum(stats[i, cv2.CC_STAT_AREA] for i in big) / max(1, skin_area)
        conf = float(min(0.7, area * 10))
        return self.result(
            "bruise_fraction", round(float(area), 3), conf, Severity.NOTICE,
            f"Possible bruising/discoloration ({len(big)} region(s), screening only)",
            ttl=25.0)
