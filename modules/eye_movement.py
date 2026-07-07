"""Gaze direction and involuntary eye oscillation (nystagmus) screening.

Method: with iris landmarks (refine_landmarks), compute iris center
relative to the eye corners to get a normalized gaze offset. Averaged
across both eyes gives gaze direction; a fast oscillation of that offset
(3-10 Hz) flags possible nystagmus.

Reliability: gaze direction MEDIUM; nystagmus LOW (fps-limited).
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, dominant_frequency, bandpass
from extractors import face_landmarks as FL


@register("eye_movement")
class EyeMovement(DetectionModule):
    interval = 0.0
    requires = ("face",)
    window_seconds = 4.0

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(self.window_seconds)

    def _gaze_offset(self, px, iris_c, corner_a, corner_b):
        eye_c = (px[corner_a] + px[corner_b]) / 2.0
        width = np.linalg.norm(px[corner_a] - px[corner_b]) + 1e-6
        return (px[iris_c] - eye_c) / width      # (dx, dy) normalized

    def process(self, ctx: FrameContext):
        if not ctx.face.has_iris:
            return None
        px = ctx.face_px()
        left = self._gaze_offset(px, FL.LEFT_IRIS[0], 33, 133)
        right = self._gaze_offset(px, FL.RIGHT_IRIS[0], 362, 263)
        gaze = (left + right) / 2.0
        self.buf.push(ctx.timestamp, float(gaze[0]))

        results = []
        direction = "center"
        if gaze[0] < -0.12:
            direction = "left"
        elif gaze[0] > 0.12:
            direction = "right"
        results.append(self.result(
            "gaze", direction, 0.5, Severity.INFO,
            f"Gaze toward {direction}", ttl=2.0))

        rs = self.buf.resampled(fs=30.0)
        if rs is not None and self.buf.span() > 2.5:
            filt = bandpass(rs[0], 30.0, 3.0, min(10.0, 30.0 / 2 - 1))
            if filt is not None and np.std(filt) > 0.02:
                dom = dominant_frequency(filt, 30.0, 3.0, 10.0)
                if dom and dom[1] > 0.25:
                    results.append(self.result(
                        "nystagmus", round(dom[0], 1), min(0.5, dom[1] * 2),
                        Severity.NOTICE,
                        f"Rapid eye oscillation ~{dom[0]:.1f} Hz (screening)", ttl=5.0))
        return results
