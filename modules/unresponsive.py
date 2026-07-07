"""Prolonged immobility / unresponsiveness detection.

Method: track scene motion energy (from the motion extractor) while a
person is present. If motion stays below a small threshold continuously for
`threshold_minutes`, raise an alert — this covers post-fall immobility,
collapse, or loss of consciousness. Resets as soon as meaningful motion
returns.

Reliability: MEDIUM; a person sitting very still (reading, sleeping) can
trigger it, so the greeting layer should confirm gently before escalating.
"""
from __future__ import annotations

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule


@register("unresponsive")
class Unresponsive(DetectionModule):
    interval = 1.0
    requires = ("person",)
    threshold_minutes = 3.0

    def __init__(self, **params):
        super().__init__(**params)
        self._still_since = None
        self._alerted = False

    def process(self, ctx: FrameContext):
        moving = ctx.motion_energy > 0.5
        if moving:
            self._still_since = None
            self._alerted = False
            return None
        if self._still_since is None:
            self._still_since = ctx.timestamp
            return None
        still_for = ctx.timestamp - self._still_since
        threshold = self.threshold_minutes * 60.0
        if still_for >= threshold and not self._alerted:
            self._alerted = True
            return self.result(
                "immobility", round(still_for / 60.0, 1), 0.7, Severity.ALERT,
                f"No movement for {still_for/60.0:.1f} min while present "
                "(check responsiveness)", ttl=30.0)
        if still_for >= threshold * 0.5:
            return self.result(
                "stillness", round(still_for / 60.0, 1), 0.4, Severity.NOTICE,
                f"Person unusually still for {still_for/60.0:.1f} min", ttl=15.0)
        return None
