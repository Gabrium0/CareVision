"""Frame-difference motion energy, shared by activity/agitation/unresponsive."""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext


class MotionExtractor:
    def __init__(self):
        self._prev: np.ndarray | None = None

    def extract(self, ctx: FrameContext) -> None:
        small = cv2.resize(ctx.frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev is not None:
            ctx.motion_energy = float(
                np.mean(cv2.absdiff(gray, self._prev)))
        self._prev = gray
