"""Eye contact / attention-to-robot detection.

Method: combine the normalized iris-vs-eye-corner gaze offset (the same
geometry modules/eye_movement.py uses) with a head-yaw proxy (nose tip
offset from the midpoint of the face edges): the person is "looking at the
robot" when both the eyes and the head point at the camera. Sustained
attention duration is tracked so the agent can tell a glance from genuine
engagement — the conversational trigger a showcase robot needs ("they're
looking at me, I should say something").

Reliability: MEDIUM — same iris-resolution limits as gaze in
modules/eye_movement.py, but the head-yaw term makes the combined signal
much more forgiving than gaze alone.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import face_landmarks as FL


@register("attention")
class Attention(DetectionModule):
    """Eye contact / attention-to-robot detection."""
    interval = 0.0
    requires = ("face",)
    gaze_threshold = 0.15        # |normalized gaze offset| below this = at camera
    yaw_threshold = 0.18         # |nose offset / face width| below this = facing camera
    sustain_seconds = 2.0        # eye contact this long counts as engaged

    def __init__(self, **params):
        super().__init__(**params)
        self._contact_since = None

    @staticmethod
    def _gaze_offset(px, iris_c, corner_a, corner_b):
        eye_c = (px[corner_a] + px[corner_b]) / 2.0
        width = np.linalg.norm(px[corner_a] - px[corner_b]) + 1e-6
        return (px[iris_c] - eye_c) / width

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        px = ctx.face_px()
        # Head yaw proxy: nose tip should sit midway between the face edges
        # when facing the camera; the offset grows roughly linearly with yaw.
        edge_mid = (px[FL.LEFT_FACE_EDGE] + px[FL.RIGHT_FACE_EDGE]) / 2.0
        face_w = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        yaw = float((px[FL.NOSE_TIP][0] - edge_mid[0]) / face_w)
        facing = abs(yaw) < self.yaw_threshold

        if ctx.face.has_iris:
            left = self._gaze_offset(px, FL.LEFT_IRIS[0], 33, 133)
            right = self._gaze_offset(px, FL.RIGHT_IRIS[0], 362, 263)
            gaze = (left + right) / 2.0
            gazing = float(np.hypot(*gaze)) < self.gaze_threshold
        else:
            gazing = facing            # no iris landmarks: head pose is all we have

        contact = facing and gazing
        if contact:
            if self._contact_since is None:
                self._contact_since = ctx.timestamp
            held = ctx.timestamp - self._contact_since
        else:
            self._contact_since = None
            held = 0.0

        results = [
            self.result("eye_contact", bool(contact), 0.6, Severity.INFO, "", ttl=2.0),
            self.result("attention_seconds", round(held, 1), 0.6, Severity.INFO, "", ttl=2.0),
        ]
        if held >= self.sustain_seconds:
            results.append(self.result(
                "engaged", True, 0.7, Severity.INFO,
                "Making sustained eye contact", ttl=3.0))
        return results
