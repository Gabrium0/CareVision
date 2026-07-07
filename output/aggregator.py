"""Aggregator: keeps the latest non-expired Result per (module, key).

Provides a single 'person state' snapshot that the greeting engine and
overlay consume, so neither special-cases individual detectors.
"""
from __future__ import annotations

from core.events import Result, Severity


class Aggregator:
    def __init__(self):
        self.state: dict[tuple[str, str], Result] = {}

    def ingest(self, results: list[Result]) -> None:
        for r in results:
            self.state[(r.module, r.key)] = r
        # drop expired
        self.state = {k: r for k, r in self.state.items() if not r.expired}

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
