"""Activity-level tracking and inactivity trend (longitudinal).

Method: log the shared motion energy to the persistent store each interval.
Compares the last 10 minutes of mean activity against the trailing hour's
mean to flag an unusual drop in activity (possible malaise/withdrawal) or
confirms normal activity for the greeting layer.

Reliability: MEDIUM as a relative trend; needs history to be meaningful.
"""
from __future__ import annotations

from collections import deque

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from storage.history_store import HistoryStore


@register("activity_level")
class ActivityLevel(DetectionModule):
    """Activity-level tracking and inactivity trend (longitudinal)."""
    interval = 5.0
    requires = ("person",)

    def __init__(self, **params):
        super().__init__(**params)
        self.store = HistoryStore.instance()
        self._rolling = getattr(self.store, "rolling_mean", None)
        self._samples = deque(maxlen=2048)
        if self._rolling is not None:
            # Registration is O(1); SQLite bootstrap happens on the aggregate worker.
            self._rolling("activity_level", "motion", 10 * 60)
            self._rolling("activity_level", "motion", 60 * 60)
        else:
            self._samples.extend(self.store.recent(
                "activity_level", "motion", 60 * 60))

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        self.store.add("activity_level", "motion", ctx.motion_energy, ctx.timestamp)
        if self._rolling is not None:
            recent = self._rolling("activity_level", "motion", 10 * 60,
                                   now=ctx.timestamp)
            baseline = self._rolling("activity_level", "motion", 60 * 60,
                                     now=ctx.timestamp)
            if recent is None or baseline is None or baseline < 1e-3:
                return None
            ratio = recent / baseline
            if ratio < 0.4:
                return self.result(
                    "activity_drop", round(ratio, 2), min(0.6, (0.4 - ratio) * 2),
                    Severity.NOTICE,
                    "Activity noticeably lower than usual today", ttl=60.0)
            return self.result("activity", round(recent, 2), 0.4, Severity.INFO,
                               "Activity level normal", ttl=60.0)
        self._samples.append((ctx.timestamp, float(ctx.motion_energy)))
        cutoff = ctx.timestamp - 60 * 60
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        baseline_values = [value for _, value in self._samples]
        recent_values = [value for ts, value in self._samples
                         if ts >= ctx.timestamp - 10 * 60]
        recent = (sum(recent_values) / len(recent_values)) if recent_values else None
        baseline = (sum(baseline_values) / len(baseline_values)) if baseline_values else None
        if recent is None or baseline is None or baseline < 1e-3:
            return None
        ratio = recent / baseline
        if ratio < 0.4:
            return self.result(
                "activity_drop", round(ratio, 2), min(0.6, (0.4 - ratio) * 2),
                Severity.NOTICE,
                "Activity noticeably lower than usual today", ttl=60.0)
        return self.result("activity", round(recent, 2), 0.4, Severity.INFO,
                           "Activity level normal", ttl=60.0)
