"""Hand/limb tremor detection via wrist oscillation spectrum.

Method: track each wrist position (pose landmarks), remove slow drift,
and look for a dominant oscillation in 3-12 Hz — the band covering
physiological, essential, and Parkinsonian (4-6 Hz) tremor. Position is
normalized by shoulder width so distance to camera doesn't matter.

Reliability: MEDIUM when the hand is visible and reasonably still; camera
fps limits the top of the band (needs ~24+ fps for 12 Hz).

Scripted mode: during a `hold_still` elicitation window
(core/elicitation.py — the agent asks the person to hold a hand out flat
and keep still), buffers restart at window start and a single `tremor_test`
result is emitted when the window closes. Because the hand is deliberately
held still, any residual oscillation is signal rather than gesture noise,
so the reported confidence is boosted relative to passive watching — the
published video-vs-accelerometer validation (~0.98 correlation) is for
exactly this elicited protocol.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, dominant_frequency, bandpass
from extractors import pose as P
from core.elicitation import ElicitationState


@register("tremor")
class Tremor(DetectionModule):
    """Hand/limb tremor detection via wrist oscillation spectrum."""
    interval = 0.0
    requires = ("pose",)
    window_seconds = 5.0
    test_confidence_boost = 1.5   # elicited stillness: oscillation is signal

    def __init__(self, **params):
        super().__init__(**params)
        self.bufs = {"left": TimedBuffer(self.window_seconds),
                     "right": TimedBuffer(self.window_seconds)}
        self._test_id = None          # started-ts of the window being sampled
        self._test_peak = None        # (freq, conf) best finding in the window

    def _test_window(self, ctx, results) -> None:
        """Handle the hold-still elicitation window around normal sampling."""
        es = ElicitationState.instance()
        if es.active("hold_still", now=ctx.timestamp):
            if self._test_id != es.started:      # window just opened
                self._test_id = es.started
                self._test_peak = None
                for buf in self.bufs.values():   # gesture noise ends here
                    buf.t.clear(), buf.v.clear()
            for r in results:                    # track the strongest finding
                boosted = min(0.9, r.confidence * self.test_confidence_boost)
                if self._test_peak is None or boosted > self._test_peak[1]:
                    self._test_peak = (float(r.value), boosted)
        elif self._test_id is not None:          # window just closed: report once
            if self._test_peak is not None:
                freq, conf = self._test_peak
                results.append(self.result(
                    "tremor_test", round(freq, 1), conf, Severity.NOTICE,
                    f"Hold-still test: slight rhythmic movement ~{freq:.1f} Hz "
                    "(screening only)", ttl=15.0))
            else:
                results.append(self.result(
                    "tremor_test", "steady", 0.7, Severity.INFO,
                    "Hold-still test: hands looked nice and steady", ttl=15.0))
            self._test_id = None
            self._test_peak = None

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        sw = abs(lm[P.L_SHOULDER, 0] - lm[P.R_SHOULDER, 0]) + 1e-6
        results = []
        for side, wi in (("left", P.L_WRIST), ("right", P.R_WRIST)):
            if lm[wi, 3] < 0.6:
                continue
            self.bufs[side].push(ctx.timestamp, lm[wi, 0] / sw)
            rs = self.bufs[side].resampled(fs=30.0)
            if rs is None or self.bufs[side].span() < 3.0:
                continue
            signal, fs = rs
            filt = bandpass(signal, fs, 3.0, min(12.0, fs / 2 - 1))
            if filt is None:
                continue
            amp = float(np.std(filt))
            dom = dominant_frequency(filt, fs, 3.0, min(12.0, fs / 2 - 1))
            if dom is None or amp < 0.02:
                continue
            freq, prom = dom
            conf = float(min(0.8, prom * 3 * min(1.0, amp * 15)))
            if conf < 0.25:
                continue
            sev = Severity.NOTICE if freq < 4 or freq > 7 else Severity.WARNING
            results.append(self.result(
                f"tremor_{side}", round(freq, 1), conf, sev,
                f"{side.title()} hand tremor ~{freq:.1f} Hz detected", ttl=6.0))
        self._test_window(ctx, results)
        return results or None
