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
    """Head nodding / drooping detection (drowsiness, loss of tone)."""
    interval = 0.0
    requires = ("face",)
    window_seconds = 6.0
    droop_threshold = 0.10
    nod_drop_threshold = 0.06
    nod_crossings = 3

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(self.window_seconds)
        self.baseline = None
        self.nod_count = 0
        self._was_nodding = False

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        px = ctx.face_px()
        eye_mid = (px[159] + px[386]) / 2.0
        face_h = np.linalg.norm(px[FL.FOREHEAD_TOP] - px[FL.CHIN]) + 1e-6
        pitch = (px[FL.NOSE_TIP][1] - eye_mid[1]) / face_h   # larger = head down
        self.buf.push(ctx.timestamp, pitch)
        results = [self.result("head_pitch", round(float(pitch), 3), 0.6, Severity.INFO, "", ttl=4.0)]
        if self.baseline is None:
            self.baseline = pitch
            results.append(self.result("head_pitch_baseline", round(float(self.baseline), 3), 0.0,
                                       Severity.INFO, "", ttl=4.0))
            return results
        self.baseline = 0.99 * self.baseline + 0.01 * pitch
        sustained = pitch - self.baseline
        results.extend([
            self.result("head_pitch_baseline", round(float(self.baseline), 3), 0.6, Severity.INFO, "", ttl=4.0),
            self.result("head_drop", round(float(sustained), 3), 0.6, Severity.INFO, "", ttl=4.0),
            self.result("nod_count", self.nod_count, 0.6, Severity.INFO, "", ttl=8.0),
        ])

        if self.buf.span() < 3.0:
            results.append(self.result("nodding_score", 0.0, 0.0, Severity.INFO, "", ttl=4.0))
            return results
        _, v = self.buf.arrays()
        drop = float(np.max(v) - np.min(v))
        results.append(self.result("nodding_score", round(drop, 3), 0.6, Severity.INFO, "", ttl=4.0))
        if sustained > self.droop_threshold:
            results.append(self.result(
                "head_droop", round(float(sustained), 3), min(0.8, sustained * 5),
                Severity.WARNING, "Head drooping downward (drowsy/unresponsive?)",
                ttl=5.0))
        if drop > self.nod_drop_threshold:
            # count zero-ish crossings to see if it's oscillatory (nodding)
            centered = v - np.mean(v)
            crossings = np.sum(np.diff(np.sign(centered)) != 0)
            is_nodding = crossings >= self.nod_crossings
            if is_nodding and not self._was_nodding:
                self.nod_count += 1
                results.append(self.result("nod_count", self.nod_count, 0.7, Severity.INFO, "", ttl=8.0))
            self._was_nodding = bool(is_nodding)
            if is_nodding:
                results.append(self.result(
                    "nodding", round(drop, 3), min(0.7, drop * 6),
                    Severity.NOTICE, "Head nodding (dozing off?)", ttl=5.0))
        else:
            self._was_nodding = False
        return results or None
