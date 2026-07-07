"""Drowsiness: eye-aspect-ratio, PERCLOS, blink rate, microsleep.

Method: EAR (eye aspect ratio) from eyelid landmarks. A per-person open-eye
baseline is learned; eyes are 'closed' below 60% of baseline. PERCLOS is
the fraction of time closed over a rolling 60 s window; long continuous
closures are flagged as microsleep. Blink rate is derived from closure
onsets.

Reliability: HIGH for eye closure/PERCLOS with a frontal face.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from extractors import face_landmarks as FL


def _ear(px, idx):
    p1, p2, p3, p4, p5, p6 = (px[i] for i in idx)
    v = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    h = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
    return v / h


@register("drowsiness")
class Drowsiness(DetectionModule):
    interval = 0.0
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.closed = TimedBuffer(60.0)     # (t, is_closed) for PERCLOS
        self.blinks = TimedBuffer(60.0)     # (t, 1) at each blink onset
        self.baseline = None
        self.t0 = None
        self._was_closed = False
        self._closed_since = None

    def process(self, ctx: FrameContext):
        px = ctx.face_px()
        ear = 0.5 * (_ear(px, FL.LEFT_EYE_EAR) + _ear(px, FL.RIGHT_EYE_EAR))
        if self.t0 is None:
            self.t0, self.baseline = ctx.timestamp, ear
        if ctx.timestamp - self.t0 < 3.0:
            self.baseline = max(self.baseline, ear)   # capture open-eye value
            return None

        is_closed = ear < 0.6 * self.baseline
        self.closed.push(ctx.timestamp, 1.0 if is_closed else 0.0)

        results = []
        # blink onset
        if is_closed and not self._was_closed:
            self.blinks.push(ctx.timestamp, 1.0)
            self._closed_since = ctx.timestamp
        if is_closed and self._closed_since is not None:
            dur = ctx.timestamp - self._closed_since
            if dur > 1.2:
                results.append(self.result(
                    "microsleep", round(dur, 1), min(0.9, dur / 3),
                    Severity.WARNING,
                    f"Eyes closed {dur:.1f}s (possible microsleep/unresponsive)",
                    ttl=4.0))
        if not is_closed:
            self._closed_since = None
        self._was_closed = is_closed

        if self.closed.span() > 20.0:
            _, c = self.closed.arrays()
            perclos = float(np.mean(c))
            blink_rate = len(self.blinks) * (60.0 / max(self.blinks.span(), 1e-6))
            if perclos > 0.15:
                results.append(self.result(
                    "perclos", round(perclos, 2), min(0.9, perclos * 3),
                    Severity.NOTICE if perclos < 0.3 else Severity.WARNING,
                    f"Drowsiness: eyes closed {perclos*100:.0f}% of the time",
                    ttl=10.0))
            results.append(self.result(
                "blink_rate", round(blink_rate, 0), 0.6, Severity.INFO,
                f"Blink rate ~{blink_rate:.0f}/min", ttl=10.0))
        return results or None
