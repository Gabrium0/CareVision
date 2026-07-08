"""Rash / skin eruption screening on facial skin.

Method: within the face-skin mask, look for reddish, textured clusters.
Redness map = a-channel (LAB) high while overall not lip/eye; texture =
local standard deviation. Rash tends to be red AND locally patchy, unlike
uniform flushing. We report the fraction of skin area that is both red and
high-texture.

Reliability: LOW. A screening prompt only. Report as NOTICE/WARNING.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import face_skin_mask


@register("rash")
class Rash(DetectionModule):
    """Rash / skin eruption screening on facial skin."""
    interval = 2.0
    requires = ("face",)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        mask = face_skin_mask(ctx)
        if mask is None or mask.sum() < 800:
            return None
        lab = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32)         # green-red; high = red
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        skin = mask > 0
        a_skin = a[skin]
        a_thresh = a_skin.mean() + 1.5 * a_skin.std()
        redness = (a > a_thresh) & skin

        # local texture via high-pass energy
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        texture = np.abs(gray - blur)
        tex_thresh = texture[skin].mean() + 1.0 * texture[skin].std()
        patchy = (texture > tex_thresh) & redness

        frac = patchy.sum() / max(1, skin.sum())
        if frac < 0.02:
            return None
        conf = float(min(0.75, frac * 8))
        sev = Severity.WARNING if frac > 0.06 else Severity.NOTICE
        return self.result(
            "rash_fraction", round(float(frac), 3), conf, sev,
            f"Possible skin rash on ~{frac*100:.0f}% of face skin (screening only)",
            ttl=20.0)
