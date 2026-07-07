"""Head nodding / drooping detection (drowsiness, loss of tone).

Method: estimate head pitch from the vertical offset of nose tip relative
to the eye line, normalized by face height. Repeated downward drops
followed by recovery (nodding) or a sustained downward droop are reported.

Reliability: MEDIUM.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import face_landmarks as FL


@register("head_nod")
class HeadNod(DetectionModule):
    interval = 0.0
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(6.0)
        self.baseline = None

    def process(self, ctx: FrameContext):
        px = ctx.face_px()
        eye_mid = (px[159] + px[386]) / 2.0
        face_h = np.linalg.norm(px[FL.FOREHEAD_TOP] - px[FL.CHIN]) + 1e-6
        pitch = (px[FL.NOSE_TIP][1] - eye_mid[1]) / face_h   # larger = head down
        self.buf.push(ctx.timestamp, pitch)
        if self.baseline is None:
            self.baseline = pitch
            return None
        self.baseline = 0.99 * self.baseline + 0.01 * pitch

        if self.buf.span() < 3.0:
            return None
        _, v = self.buf.arrays()
        drop = float(np.max(v) - np.min(v))
        sustained = pitch - self.baseline
        if sustained > 0.10:
            return self.result(
                "head_droop", round(float(sustained), 3), min(0.8, sustained * 5),
                Severity.WARNING, "Head drooping downward (drowsy/unresponsive?)",
                ttl=5.0)
        if drop > 0.06:
            # count zero-ish crossings to see if it's oscillatory (nodding)
            centered = v - np.mean(v)
            crossings = np.sum(np.diff(np.sign(centered)) != 0)
            if crossings >= 3:
                return self.result(
                    "nodding", round(drop, 3), min(0.7, drop * 6),
                    Severity.NOTICE, "Head nodding (dozing off?)", ttl=5.0)
        return None
