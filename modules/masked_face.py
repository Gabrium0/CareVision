"""Reduced facial expressiveness (masked / flat affect) over time.

Method: measure moment-to-moment variability of facial expression by
tracking the variance of key expression distances (brow height, eye
opening, mouth width/openness), normalized by face size, across a rolling
window. Persistently low variance = reduced expressiveness (relevant to
Parkinsonian masking and to depression/apathy flat affect).

Reliability: LOW-MEDIUM; longitudinal signal, best over minutes.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import face_landmarks as FL


@register("masked_face")
class MaskedFace(DetectionModule):
    interval = 0.3
    requires = ("face",)
    window_seconds = 20.0

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(self.window_seconds)

    def process(self, ctx: FrameContext):
        px = ctx.face_px()
        fw = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        brow = np.linalg.norm(px[FL.LEFT_BROW[2]] - px[159]) / fw
        eye = np.linalg.norm(px[159] - px[145]) / fw
        mouth_w = np.linalg.norm(px[FL.MOUTH_LEFT] - px[FL.MOUTH_RIGHT]) / fw
        mouth_h = np.linalg.norm(px[FL.MOUTH_TOP_INNER] - px[FL.MOUTH_BOTTOM_INNER]) / fw
        self.buf.push(ctx.timestamp, [brow, eye, mouth_w, mouth_h])

        if self.buf.span() < 15.0 or len(self.buf) < 20:
            return None
        _, v = self.buf.arrays()
        expressiveness = float(np.mean(np.std(np.vstack(v), axis=0)))
        if expressiveness > 0.010:
            return None
        conf = float(min(0.6, (0.010 - expressiveness) * 60))
        return self.result(
            "expressiveness", round(expressiveness, 4), conf, Severity.NOTICE,
            "Reduced facial expressiveness (flat affect / masking — screening)",
            ttl=20.0)
