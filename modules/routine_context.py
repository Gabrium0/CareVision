"""Meaningful routine opportunities derived from public scene summaries."""
from __future__ import annotations

from datetime import datetime

from core.events import PersistencePolicy, Result, Severity
from core.registry import register
from modules.base import DetectionModule


@register("routine")
class RoutineContext(DetectionModule):
    """Track occupancy/activity opportunities without claiming adherence."""
    name = "routine"
    interval = 30.0
    location = "unspecified"

    def __init__(self, **params):
        super().__init__(**params)
        self._last_present = None
        self._last_drink = None

    def process(self, ctx):
        """Emit sparse routine changes rather than raw per-frame state."""
        out = []
        if ctx.person_present != self._last_present:
            self._last_present = ctx.person_present
            out.append(Result(self.name, "occupancy", "occupied" if ctx.person_present else "empty",
                              .8, Severity.INFO, f"{self.location} is now " +
                              ("occupied" if ctx.person_present else "empty"), ttl=120,
                              location=self.location, persistence=PersistencePolicy.EVENT))
        if self._last_drink is not None:
            hours = max(0.0, (ctx.timestamp - self._last_drink) / 3600)
            out.append(Result(self.name, "hours_since_drink", round(hours, 1), .6,
                              Severity.INFO, "Interval since a recorded drink opportunity",
                              ttl=60, location=self.location))
        hour = datetime.fromtimestamp(ctx.timestamp).hour
        if ctx.person_present and (hour >= 23 or hour < 5):
            out.append(Result(self.name, "night_activity", True, .6, Severity.NOTICE,
                              "Activity is visible during the configured nighttime window",
                              ttl=60, location=self.location, persistence=PersistencePolicy.EVENT))
        return out or None
