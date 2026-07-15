"""Replay sensor adapter fed by synchronized scenario events."""
from __future__ import annotations

import queue

from core.capabilities import CapabilityRegistry, CapabilityStatus
from .base import SensorAdapter, SensorReading


class ReplaySensorAdapter(SensorAdapter):
    """Convert replay sensor events through the normal SensorAdapter contract."""
    name = "replay_sensors"
    available = True

    def __init__(self):
        self._pending: queue.Queue[SensorReading] = queue.Queue()
        CapabilityRegistry.instance().set(self.name, "sensor", CapabilityStatus.READY,
                                          "synchronized replay channel")

    def feed_context(self, ctx) -> None:
        """Queue readings synchronized with the current replay frame."""
        for event in ctx.extras.get("replay_channels", {}).get("sensor", []):
            self._pending.put(SensorReading(
                key=str(event["key"]), value=event.get("value"),
                unit=str(event.get("unit", "")), timestamp=ctx.timestamp,
                quality=float(event.get("quality", 1.0)),
                subject_id=str(event.get("subject_id", "primary")),
                source="replay_sensor"))

    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Drain due deterministic sensor readings."""
        out = []
        while True:
            try:
                out.append(self._pending.get_nowait())
            except queue.Empty:
                return out
