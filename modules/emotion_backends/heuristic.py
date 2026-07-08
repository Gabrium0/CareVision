"""Landmark-geometry emotion heuristic (original, dependency-free).

Maps smile curvature, mouth open, brow furrow, and eye openness to a coarse
label. Low-medium reliability but fully offline; kept as the always-available
baseline shown next to the tested-model backends.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from modules.backends.base import Backend
from extractors import face_landmarks as FL


class HeuristicEmotionBackend(Backend):
    """Landmark-geometry emotion heuristic (offline baseline)."""
    label = "heuristic"
    available = True

    def __init__(self):
        self._ctx: FrameContext | None = None

    def update(self, ctx: FrameContext) -> None:
        """Feed one frame's data into the backend's rolling state."""
        self._ctx = ctx

    def compute(self) -> dict | None:
        """Return the backend's current reading dict, or None if not ready."""
        ctx = self._ctx
        if ctx is None or ctx.face is None:
            return None
        px = ctx.face_px()
        fw = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        mouth_w = np.linalg.norm(px[FL.MOUTH_LEFT] - px[FL.MOUTH_RIGHT]) / fw
        mouth_h = np.linalg.norm(px[FL.MOUTH_TOP_INNER] - px[FL.MOUTH_BOTTOM_INNER]) / fw
        corner_y = (px[FL.MOUTH_LEFT][1] + px[FL.MOUTH_RIGHT][1]) / 2.0
        center_y = (px[FL.MOUTH_TOP_INNER][1] + px[FL.MOUTH_BOTTOM_INNER][1]) / 2.0
        smile = (center_y - corner_y) / fw       # corners above center => smile
        brow_eye = np.linalg.norm(px[FL.LEFT_BROW[2]] - px[159]) / fw
        eye_open = np.linalg.norm(px[159] - px[145]) / fw

        label, conf = "neutral", 0.4
        if smile > 0.015 and mouth_w > 0.45:
            label, conf = "happy", min(0.8, 0.4 + smile * 15)
        elif mouth_h > 0.35 and eye_open > 0.10:
            label, conf = "surprise", 0.6
        elif brow_eye < 0.11 and smile < 0:
            label, conf = "angry", 0.5
        elif smile < -0.01:
            label, conf = "sad", 0.5
        return {"emotion": label, "confidence": conf}
