"""Aggregator: keeps the latest non-expired Result per (module, key).

Provides a single 'person state' snapshot that the greeting engine and
overlay consume, so neither special-cases individual detectors.

Numeric readings are smoothed with a short rolling median before display, so
the shown data (heart rate, HRV, cadence, valence, ...) is steady rather than
flickering per frame. Labels/booleans (emotion, fall) pass through unchanged.
"""
from __future__ import annotations

import statistics
import threading
from collections import deque

from core.events import Result, Severity, Visibility


class Aggregator:
    """Keeps the latest non-expired result per (module, key) with rolling-median smoothing."""
    def __init__(self, smooth_window: int = 5):
        self.state: dict[tuple[str, str, str], Result] = {}
        self.smooth_window = smooth_window
        self._hist: dict[tuple[str, str, str], deque] = {}
        self._lock = threading.RLock()

    def _smooth(self, r: Result) -> Result:
        # bool is an int subclass — exclude it (fall/present are events).
        if isinstance(r.value, bool) or not isinstance(r.value, (int, float)):
            return r
        key = (r.subject_id, r.module, r.key)
        hist = self._hist.setdefault(key, deque(maxlen=self.smooth_window))
        hist.append(r.value)
        med = statistics.median(hist)
        r.value = round(med, 1) if isinstance(r.value, float) else int(round(med))
        return r

    def ingest(self, results: list[Result]) -> None:
        """Merge new results into the current person-state."""
        with self._lock:
            for r in results:
                self.state[(r.subject_id, r.module, r.key)] = self._smooth(r)
            # drop expired, and forget history for keys no longer present
            self.state = {k: r for k, r in self.state.items() if not r.expired}
            for k in list(self._hist):
                if k not in self.state:
                    self._hist.pop(k, None)

    def snapshot(self, include_agent_only: bool = False,
                 subject_id: str | None = None) -> list[Result]:
        """Return live public results, optionally including agent-only data."""
        with self._lock:
            values = [r for r in self.state.values()
                      if subject_id is None or r.subject_id == subject_id]
        if include_agent_only:
            return values
        return [r for r in values if r.visibility == Visibility.PUBLIC]

    def agent_snapshot(self) -> list[Result]:
        """Return the internal snapshot intended only for the voice agent."""
        return self.snapshot(include_agent_only=True)

    def by_severity(self, minimum: Severity) -> list[Result]:
        """Return live results at or above a severity, most-severe first."""
        order = {Severity.INFO: 0, Severity.NOTICE: 1,
                 Severity.WARNING: 2, Severity.ALERT: 3}
        cutoff = order[minimum]
        return sorted([r for r in self.snapshot() if order[r.severity] >= cutoff],
                      key=lambda r: (order[r.severity], r.confidence), reverse=True)

    def get(self, module: str, key: str,
            include_agent_only: bool = False, subject_id: str = "primary") -> Result | None:
        """Return the latest public result, or an internal one when requested."""
        with self._lock:
            r = self.state.get((subject_id, module, key))
        if not r or r.expired:
            return None
        if r.visibility == Visibility.AGENT_ONLY and not include_agent_only:
            return None
        return r
