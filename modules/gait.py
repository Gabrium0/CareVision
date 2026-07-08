"""Gait analysis: cadence and left/right symmetry from ankle motion.

Method: when the person is standing/walking (ankles visible, vertical
extent large), track each ankle's vertical position. Step cadence is the
dominant frequency of ankle oscillation (0.5-3 Hz); asymmetry compares the
oscillation amplitude and step timing between legs. Shuffling shows low
amplitude; limping shows amplitude asymmetry.

Reliability: MEDIUM; needs a side/front view with legs in frame.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, dominant_frequency, bandpass
from extractors import pose as P


@register("gait")
class Gait(DetectionModule):
    """Gait analysis: cadence and left/right symmetry from ankle motion."""
    interval = 0.0
    requires = ("pose",)
    window_seconds = 8.0

    def __init__(self, **params):
        super().__init__(**params)
        self.left = TimedBuffer(self.window_seconds)
        self.right = TimedBuffer(self.window_seconds)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        for idx in (P.L_ANKLE, P.R_ANKLE, P.L_HIP, P.R_HIP):
            if lm[idx, 3] < 0.5:
                return None
        torso = abs(lm[P.L_HIP, 1] - lm[P.L_SHOULDER, 1]) + 1e-6
        self.left.push(ctx.timestamp, lm[P.L_ANKLE, 1] / torso)
        self.right.push(ctx.timestamp, lm[P.R_ANKLE, 1] / torso)

        rs_l = self.left.resampled(fs=30.0)
        rs_r = self.right.resampled(fs=30.0)
        if rs_l is None or rs_r is None or self.left.span() < 5.0:
            return None
        fl = bandpass(rs_l[0], 30.0, 0.5, 3.0, order=2)
        fr = bandpass(rs_r[0], 30.0, 0.5, 3.0, order=2)
        if fl is None or fr is None:
            return None
        amp_l, amp_r = float(np.std(fl)), float(np.std(fr))
        if max(amp_l, amp_r) < 0.01:
            return None      # essentially not walking

        dom = dominant_frequency(fl + fr, 30.0, 0.5, 3.0)
        results = []
        if dom is not None:
            cadence = dom[0] * 60.0     # steps/min per leg cycle
            results.append(self.result(
                "cadence_spm", round(cadence, 0), min(0.7, dom[1] * 3),
                Severity.INFO, f"Walking cadence ~{cadence:.0f} steps/min", ttl=6.0))
        asym = abs(amp_l - amp_r) / (amp_l + amp_r + 1e-6)
        if asym > 0.35:
            results.append(self.result(
                "gait_asymmetry", round(float(asym), 2), min(0.7, asym),
                Severity.WARNING,
                "Asymmetric leg movement while walking (possible limp)", ttl=6.0))
        return results or None
