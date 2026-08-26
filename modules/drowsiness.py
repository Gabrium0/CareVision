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
    """Drowsiness: eye-aspect-ratio, PERCLOS, blink rate, microsleep."""
    interval = 0.0
    requires = ("face",)
    perclos_window_seconds = 60.0
    baseline_seconds = 3.0
    closed_ear_ratio = 0.60
    perclos_notice = 0.15
    perclos_warning = 0.30
    microsleep_seconds = 1.2
    # Signal-quality gates. EAR is only trustworthy on a large, near-frontal
    # eye; on the OV2735 at 1-1.5 m the eye can be ~15 px across and the head is
    # often turned, so a fixed 0.7 confidence over-trusts junk frames and was a
    # major source of false "tired" prompts. `_signal_quality` maps the two
    # things that actually corrupt EAR -- eye pixel size and head yaw -- onto a
    # 0..1 factor the escalation confidences are scaled by, so low-quality
    # frames fall below the conversation layer's admission floor.
    min_eye_px = 10.0          # eye-corner span at/below this -> EAR unusable
    good_eye_px = 24.0         # at/above this -> full spatial confidence
    max_ear_yaw = 0.30         # |head-yaw proxy| at/above this -> EAR geometry invalid

    def __init__(self, **params):
        super().__init__(**params)
        self.closed = TimedBuffer(self.perclos_window_seconds)     # (t, is_closed) for PERCLOS
        self.blinks = TimedBuffer(self.perclos_window_seconds)     # (t, 1) at each blink onset
        self.baseline = None
        self.t0 = None
        self._was_closed = False
        self._closed_since = None
        self._blink_total = 0      # cumulative blinks since start; only ever rises

    def _signal_quality(self, px) -> float:
        """0..1 trust in this frame's EAR, from eye pixel size and head yaw.

        EAR degrades when the eye spans only a few pixels (its landmarks
        collapse into noise) or the head is turned (the vertical/horizontal
        ratio stops meaning eyelid closure). The emitted escalation
        confidences are multiplied by this so junk frames stay below the
        speech-admission floor instead of raising a false drowsiness prompt.
        """
        eye_w = 0.5 * (np.linalg.norm(px[33] - px[133])
                       + np.linalg.norm(px[362] - px[263]))
        span = self.good_eye_px - self.min_eye_px
        size_q = (eye_w - self.min_eye_px) / span if span > 0 else 1.0
        size_q = float(np.clip(size_q, 0.0, 1.0))

        edge_mid = (px[FL.LEFT_FACE_EDGE] + px[FL.RIGHT_FACE_EDGE]) / 2.0
        face_w = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        yaw = abs(float((px[FL.NOSE_TIP][0] - edge_mid[0]) / face_w))
        yaw_q = float(np.clip(1.0 - yaw / self.max_ear_yaw, 0.0, 1.0))
        return size_q * yaw_q

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        px = ctx.face_px()
        ear = 0.5 * (_ear(px, FL.LEFT_EYE_EAR) + _ear(px, FL.RIGHT_EYE_EAR))
        quality = self._signal_quality(px)
        results = [
            self.result("ear", round(float(ear), 3), 0.7, Severity.INFO, "", ttl=4.0),
            self.result("signal_quality", round(quality, 2), 0.0, Severity.INFO, "", ttl=4.0),
        ]
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
            self._blink_total += 1
            self._closed_since = ctx.timestamp
        # Monotonic lifetime counter, emitted every post-baseline frame so a demo
        # can blink and watch it tick up (rolling blink_rate is the medical signal;
        # this one is the easy-to-verify showcase counter).
        results.append(self.result("blink_count_total", self._blink_total, 0.6,
                                   Severity.INFO, "", ttl=8.0))
        if is_closed and self._closed_since is not None:
            dur = ctx.timestamp - self._closed_since
            results.append(self.result("microsleep_duration", round(float(dur), 1), 0.7, Severity.INFO, "", ttl=4.0))
            if dur > self.microsleep_seconds:
                results.append(self.result(
                    "microsleep", round(dur, 1), min(0.9, dur / 3) * quality,
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
                    "perclos", round(perclos, 2), min(0.9, perclos * 3) * quality,
                    Severity.NOTICE if perclos < self.perclos_warning else Severity.WARNING,
                    f"Drowsiness: eyes closed {perclos*100:.0f}% of the time",
                    ttl=10.0))
        else:
            results.append(self.result(
                "perclos_status", f"warming up {self.closed.span():.0f}/{self.perclos_window_seconds:.0f}s",
                0.0, Severity.INFO, "", ttl=4.0))
        return results or None
