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
from core.debug import log as debug_log
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
    perclos_window_seconds = 60.0
    baseline_seconds = 3.0
    closed_ear_ratio = 0.60
    perclos_notice = 0.15
    perclos_warning = 0.30
    microsleep_seconds = 1.2

    def __init__(self, **params):
        super().__init__(**params)
        self.closed = TimedBuffer(self.perclos_window_seconds)     # (t, is_closed) for PERCLOS
        self.blinks = TimedBuffer(self.perclos_window_seconds)     # (t, 1) at each blink onset
        self.baseline = None
        self.t0 = None
        self._was_closed = False
        self._closed_since = None

    def process(self, ctx: FrameContext):
        px = ctx.face_px()
        ear = 0.5 * (_ear(px, FL.LEFT_EYE_EAR) + _ear(px, FL.RIGHT_EYE_EAR))
        results = [self.result("ear", round(float(ear), 3), 0.7, Severity.INFO, "", ttl=4.0)]
        if self.t0 is None:
            self.t0, self.baseline = ctx.timestamp, ear
        if ctx.timestamp - self.t0 < self.baseline_seconds:
            self.baseline = max(self.baseline, ear)   # capture open-eye value
            results.append(self.result("perclos_status", "learning baseline", 0.0, Severity.INFO, "", ttl=4.0))
            return results

        closed_threshold = self.closed_ear_ratio * self.baseline
        is_closed = ear < closed_threshold
        self.closed.push(ctx.timestamp, 1.0 if is_closed else 0.0)

        results.extend([
            self.result("ear_baseline", round(float(self.baseline), 3), 0.7, Severity.INFO, "", ttl=4.0),
            self.result("eye_closed", bool(is_closed), 0.7, Severity.INFO, "", ttl=4.0),
            self.result("ear_closed_threshold", round(float(closed_threshold), 3), 0.7, Severity.INFO, "", ttl=4.0),
        ])
        # blink onset
        if is_closed and not self._was_closed:
            self.blinks.push(ctx.timestamp, 1.0)
            self._closed_since = ctx.timestamp
        if is_closed and self._closed_since is not None:
            dur = ctx.timestamp - self._closed_since
            results.append(self.result("microsleep_duration", round(float(dur), 1), 0.7, Severity.INFO, "", ttl=4.0))
            if dur > self.microsleep_seconds:
                results.append(self.result(
                    "microsleep", round(dur, 1), min(0.9, dur / 3),
                    Severity.WARNING,
                    f"Eyes closed {dur:.1f}s (possible microsleep/unresponsive)",
                    ttl=4.0))
        if not is_closed:
            self._closed_since = None
        self._was_closed = is_closed

        if self.closed.span() > max(5.0, self.perclos_window_seconds / 3.0):
            _, c = self.closed.arrays()
            perclos = float(np.mean(c))
            blink_rate = len(self.blinks) * (60.0 / max(self.blinks.span(), 1e-6))
            results.append(self.result("perclos", round(perclos, 2), 0.7, Severity.INFO, "", ttl=10.0))
            results.append(self.result("blink_rate", round(blink_rate, 0), 0.6, Severity.INFO, "", ttl=10.0))
            results.append(self.result("perclos_samples", len(c), 0.6, Severity.INFO, "", ttl=10.0))
            results.append(self.result("perclos_status", "ready", 0.0, Severity.INFO, "", ttl=10.0))
            debug_log("drowsiness", f"ear={ear:.3f} baseline={self.baseline:.3f} threshold={closed_threshold:.3f} "
                                     f"closed={is_closed} perclos={perclos:.2f} samples={len(c)} "
                                     f"blink_rate={blink_rate:.0f}")
            if perclos > self.perclos_notice:
                results.append(self.result(
                    "perclos", round(perclos, 2), min(0.9, perclos * 3),
                    Severity.NOTICE if perclos < self.perclos_warning else Severity.WARNING,
                    f"Drowsiness: eyes closed {perclos*100:.0f}% of the time",
                    ttl=10.0))
        else:
            results.append(self.result(
                "perclos_status", f"warming up {self.closed.span():.0f}/{self.perclos_window_seconds:.0f}s",
                0.0, Severity.INFO, "", ttl=4.0))
        return results or None
