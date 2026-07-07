"""Yawning detection and frequency (fatigue indicator).

Method: mouth aspect ratio (MAR) from inner-lip landmarks. A wide, sustained
opening (> ~0.6 for > 1.5 s) counts as a yawn. Frequency is tracked over a
rolling 3-minute window.

Reliability: MEDIUM-HIGH for the gesture; distinguishing yawns from talking
uses the sustained-open duration.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import face_landmarks as FL


@register("yawn")
class Yawn(DetectionModule):
    interval = 0.0
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.events = TimedBuffer(180.0)
        self._open_since = None

    def process(self, ctx: FrameContext):
        px = ctx.face_px()
        w = np.linalg.norm(px[FL.MOUTH_LEFT] - px[FL.MOUTH_RIGHT]) + 1e-6
        h = np.linalg.norm(px[FL.MOUTH_TOP_INNER] - px[FL.MOUTH_BOTTOM_INNER])
        mar = h / w
        if mar > 0.6:
            if self._open_since is None:
                self._open_since = ctx.timestamp
            elif ctx.timestamp - self._open_since > 1.5:
                self.events.push(ctx.timestamp, 1.0)
                self._open_since = None
                rate = len(self.events)
                return self.result(
                    "yawn", rate, 0.7, Severity.NOTICE,
                    f"Yawn detected ({rate} in last 3 min — fatigue?)", ttl=8.0)
        else:
            self._open_since = None
        return None
