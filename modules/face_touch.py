"""Nose-wipe / face-touch frequency (runny-nose behavioral proxy).

Method: a runny nose is not directly visible at 1-1.5 m, but the behavior
it drives is — repeated hand-to-nose contact. We watch the pose wrists and
index fingertips (MediaPipe pose landmarks 15/16/19/20) and count a touch
event whenever one enters a nose-centered radius, edge-triggered with a
cooldown so one lingering wipe counts once. The rolling 10-minute count
feeds the cold-symptom advisor (agent/advisor_engine.py) alongside
modules/sneeze.py; frequent face-touching is also a general illness/
discomfort signal in its own right.

Reliability: MEDIUM — pose wrists are tracked robustly at showcase
distance; fingertip occlusion of the face can drop a landmark exactly at
contact, so the radius is generous and the wrist is accepted as well.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer

_POSE_NOSE = 0
_HAND_POINTS = (15, 16, 19, 20)     # left/right wrist, left/right index tip
_L_SHOULDER, _R_SHOULDER = 11, 12


@register("face_touch")
class FaceTouch(DetectionModule):
    """Hand-to-nose contact frequency from pose landmarks."""
    interval = 0.0
    requires = ("pose",)
    touch_radius_ratio = 0.30    # of shoulder width, centered on the nose
    min_visibility = 0.5         # pose landmark visibility gate
    cooldown_seconds = 3.0       # one wipe = one event
    count_window_seconds = 600.0
    frequent_count = 4           # touches per window that read as "frequent"

    def __init__(self, **params):
        super().__init__(**params)
        self.events = TimedBuffer(self.count_window_seconds)
        self._touching = False
        self._last_event = -1e9

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        px = ctx.pose_px()
        if lm[_POSE_NOSE, 3] < self.min_visibility:
            return None
        shoulder_w = float(np.linalg.norm(px[_L_SHOULDER] - px[_R_SHOULDER])) + 1e-6
        radius = self.touch_radius_ratio * shoulder_w
        near = any(
            lm[i, 3] >= self.min_visibility
            and float(np.linalg.norm(px[i] - px[_POSE_NOSE])) < radius
            for i in _HAND_POINTS)

        results = [self.result("face_touch_count_10min", len(self.events), 0.6,
                               Severity.INFO, "", ttl=30.0)]
        if near and not self._touching and \
                ctx.timestamp - self._last_event >= self.cooldown_seconds:
            self._last_event = ctx.timestamp
            self.events.push(ctx.timestamp, 1.0)
            count = len(self.events)
            severity = (Severity.NOTICE if count >= self.frequent_count
                        else Severity.INFO)
            message = ("Frequent nose/face touching" if count >= self.frequent_count
                       else "")
            results.append(self.result("face_touch", True, 0.6, severity,
                                       message, ttl=8.0))
            results.append(self.result("face_touch_count_10min", count, 0.6,
                                       Severity.INFO, "", ttl=30.0))
        self._touching = bool(near)
        return results
