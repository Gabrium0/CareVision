"""Aggregator: keeps the latest non-expired Result per (module, key).

Provides a single 'person state' snapshot that the greeting engine and
overlay consume, so neither special-cases individual detectors.

Numeric readings are smoothed with a short rolling median before display, so
the shown data (heart rate, HRV, cadence, valence, ...) is steady rather than
flickering per frame. Labels/booleans (emotion, fall) pass through unchanged.
"""
from __future__ import annotations

import statistics
from collections import deque

from core.events import Result, Severity


class Aggregator:
    def __init__(self, smooth_window: int = 5):
        self.state: dict[tuple[str, str], Result] = {}
        self.smooth_window = smooth_window
        self._hist: dict[tuple[str, str], deque] = {}

    def _smooth(self, r: Result) -> Result:
        # bool is an int subclass — exclude it (fall/present are events).
        if isinstance(r.value, bool) or not isinstance(r.value, (int, float)):
            return r
        key = (r.module, r.key)
        hist = self._hist.setdefault(key, deque(maxlen=self.smooth_window))
        hist.append(r.value)
        med = statistics.median(hist)
        r.value = round(med, 1) if isinstance(r.value, float) else int(round(med))
        return r

    def ingest(self, results: list[Result]) -> None:
        for r in results:
            self.state[(r.module, r.key)] = self._smooth(r)
        # drop expired, and forget history for keys no longer present
        self.state = {k: r for k, r in self.state.items() if not r.expired}
        for k in list(self._hist):
            if k not in self.state:
                self._hist.pop(k, None)

    def snapshot(self) -> list[Result]:
        return list(self.state.values())

    def by_severity(self, minimum: Severity) -> list[Result]:
        order = {Severity.INFO: 0, Severity.NOTICE: 1,
                 Severity.WARNING: 2, Severity.ALERT: 3}
        cutoff = order[minimum]
        return sorted([r for r in self.state.values() if order[r.severity] >= cutoff],
                      key=lambda r: (order[r.severity], r.confidence), reverse=True)

    def get(self, module: str, key: str) -> Result | None:
        r = self.state.get((module, key))
        return r if (r and not r.expired) else None
