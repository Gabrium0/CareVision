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
    """Yawning detection and frequency (fatigue indicator)."""
    interval = 0.0
    requires = ("face",)
    mar_threshold = 0.60
    sustained_seconds = 1.5
    window_seconds = 180.0

    def __init__(self, **params):
        super().__init__(**params)
        self.events = TimedBuffer(self.window_seconds)
        self._open_since = None

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        px = ctx.face_px()
        w = np.linalg.norm(px[FL.MOUTH_LEFT] - px[FL.MOUTH_RIGHT]) + 1e-6
        h = np.linalg.norm(px[FL.MOUTH_TOP_INNER] - px[FL.MOUTH_BOTTOM_INNER])
        mar = h / w
        is_open = mar > self.mar_threshold
        open_duration = 0.0
        results = [
            self.result("mar", round(float(mar), 3), 0.7, Severity.INFO, "", ttl=4.0),
            self.result("mouth_open", bool(is_open), 0.7, Severity.INFO, "", ttl=4.0),
            self.result("yawn_count_3min", len(self.events), 0.6, Severity.INFO, "", ttl=8.0),
        ]
        if is_open:
            if self._open_since is None:
                self._open_since = ctx.timestamp
            open_duration = ctx.timestamp - self._open_since
            results.append(self.result("mouth_open_duration", round(float(open_duration), 1), 0.6,
                                       Severity.INFO, "", ttl=4.0))
            if open_duration > self.sustained_seconds:
                self.events.push(ctx.timestamp, 1.0)
                self._open_since = None
                rate = len(self.events)
                results.append(self.result(
                    "yawn", rate, 0.7, Severity.NOTICE,
                    f"Yawn detected ({rate} in last 3 min — fatigue?)", ttl=8.0))
        else:
            self._open_since = None
        yawn_rate = len(self.events) * (60.0 / max(self.window_seconds, 1e-6))
        results.append(self.result("yawn_rate_per_min", round(float(yawn_rate), 2), 0.6,
                                   Severity.INFO, "", ttl=8.0))
        return results
