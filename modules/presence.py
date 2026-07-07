"""Presence tracking and greeting trigger.

Method: detect when a person appears after an absence and emits a NOTICE
that the greeting engine uses to trigger a fresh greeting (rather than
repeating every frame). Also logs presence to the persistent store for
time-in-view / routine analysis.

Reliability: HIGH for presence; the greeting logic lives in output/.
"""
from __future__ import annotations

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from storage.history_store import HistoryStore


@register("presence")
class Presence(DetectionModule):
    interval = 0.5
    absence_seconds = 15.0

    def __init__(self, **params):
        super().__init__(**params)
        self.store = HistoryStore.instance()
        self._present = False
        self._last_seen = None

    def process(self, ctx: FrameContext):
        now = ctx.timestamp
        if ctx.person_present:
            newly = (not self._present and
                     (self._last_seen is None or now - self._last_seen > self.absence_seconds))
            self._last_seen = now
            self._present = True
            self.store.add("presence", "present", 1.0, now)
            if newly:
                return self.result("arrival", True, 0.9, Severity.NOTICE,
                                   "Person arrived", ttl=5.0)
            return self.result("present", True, 0.9, Severity.INFO, "", ttl=2.0)
        else:
            if self._present and self._last_seen and now - self._last_seen > self.absence_seconds:
                self._present = False
                self.store.add("presence", "present", 0.0, now)
        return None
