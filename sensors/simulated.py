"""Deterministic optional sensor used by replay and hardware-free demos."""
from __future__ import annotations

import time

from core.capabilities import CapabilityRegistry, CapabilityStatus
from .base import SensorAdapter, SensorReading


class SimulatedSensor(SensorAdapter):
    """Cycle configured readings at a fixed deterministic interval."""
    available = True

    def __init__(self, name: str, readings: list[dict], interval: float = 5.0):
        self.name, self.readings, self.interval = name, list(readings), interval
        self._index, self._last = 0, -1e9
        CapabilityRegistry.instance().set(name, "sensor", CapabilityStatus.READY, "simulated")

    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Emit one configured reading when its interval elapses."""
        now = time.time() if now is None else now
        if not self.readings or now - self._last < self.interval:
            return []
        spec = self.readings[self._index % len(self.readings)]
        self._index += 1
        self._last = now
        return [SensorReading(str(spec["key"]), spec.get("value"), str(spec.get("unit", "")),
                              now, float(spec.get("quality", 1.0)),
                              str(spec.get("subject_id", "primary")), f"simulated:{self.name}")]
