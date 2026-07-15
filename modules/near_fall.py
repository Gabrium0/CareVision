"""Pose-only stumble/recovery context; never diagnoses and never alerts alone."""
from __future__ import annotations

from collections import deque

import numpy as np

from core.events import PersistencePolicy, Result, Severity
from core.registry import register
from modules.base import DetectionModule


@register("near_fall")
class NearFall(DetectionModule):
    """Detect rapid torso displacement followed by upright recovery."""
    name = "near_fall"
    requires = ("pose",)

    def __init__(self, **params):
        super().__init__(**params)
        self._history = deque(maxlen=30)
        self._stumble_at = None

    def process(self, ctx):
        """Emit a recovered stumble summary, never an urgent alert."""
        lm = ctx.pose.landmarks
        torso_y = float(np.mean(lm[[11, 12, 23, 24], 1]))
        self._history.append((ctx.timestamp, torso_y))
        if len(self._history) >= 3:
            dt = self._history[-1][0] - self._history[-3][0]
            dy = self._history[-1][1] - self._history[-3][1]
            if dt > 0 and dy / dt > 0.35:
                self._stumble_at = ctx.timestamp
        if self._stumble_at and ctx.timestamp - self._stumble_at <= 4 and torso_y < .65:
            self._stumble_at = None
            return Result(self.name, "recovered", True, .65, Severity.NOTICE,
                          "Stumble-like movement followed by upright recovery",
                          ttl=30, quality=.65, persistence=PersistencePolicy.EVENT)
        return None
