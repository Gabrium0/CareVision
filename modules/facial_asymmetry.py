"""Facial asymmetry / droop screening (stroke-relevant, FAST 'F').

Method: using symmetric landmark pairs, mirror the face about its vertical
midline (defined by nose-tip to chin) and measure how far each left point
sits from its mirrored right counterpart, normalized by face width. Sudden
or sustained one-sided droop (especially mouth/eye) raises an alert.

We compare current asymmetry to a rolling personal baseline so that
naturally asymmetric faces don't false-positive; we flag a *change*.

Reliability: MEDIUM as a screen; NOT a diagnosis. Pairs well with a
prompt to smile/raise arms if triggered.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import face_landmarks as FL


@register("facial_asymmetry")
class FacialAsymmetry(DetectionModule):
    """Facial asymmetry / droop screening (stroke-relevant, FAST 'F')."""
    interval = 0.5
    requires = ("face",)
    learning_seconds = 20.0
    threshold = 0.05
    alert_threshold = 0.08
    alert_after_count = 3

    def __init__(self, **params):
        super().__init__(**params)
        self.baseline = None
        self.t0 = None
        self._hits = 0

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        px = ctx.face_px()
        nose = px[FL.NOSE_TIP]
        chin = px[FL.CHIN]
        axis = chin - nose
        n = np.linalg.norm(axis)
        if n < 1e-3:
            return None
        axis = axis / n
        normal = np.array([-axis[1], axis[0]])   # perpendicular (mirror normal)
        face_w = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6

        devs = []
        for li, ri in FL.SYMMETRY_PAIRS:
            lp, rp = px[li], px[ri]
            # reflect right point across the midline axis through nose
            v = rp - nose
            rp_mirror = nose + v - 2 * np.dot(v, normal) * normal
            devs.append(np.linalg.norm(lp - rp_mirror) / face_w)
        asym = float(np.mean(devs))

        if self.t0 is None:
            self.t0, self.baseline = ctx.timestamp, asym
            return None
        learning = (ctx.timestamp - self.t0) < self.learning_seconds
        a = 0.1 if learning else 0.01
        self.baseline = (1 - a) * self.baseline + a * asym
        if learning:
            return None

        change = asym - self.baseline
        if change < self.threshold:
            self._hits = 0
            return None
        self._hits += 1
        if self._hits < self.alert_after_count:
            return None
        conf = float(min(0.8, change * 10))
        sev = Severity.WARNING if change < self.alert_threshold else Severity.ALERT
        return self.result(
            "asymmetry_change", round(change, 3), conf, sev,
            "Facial asymmetry increased vs baseline (possible droop — "
            "ask person to smile; consider stroke check)", ttl=12.0)
