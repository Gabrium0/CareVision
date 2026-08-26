"""Common optional-sensor contract and non-blocking polling manager."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from core.capabilities import CapabilityRegistry, CapabilityStatus
from core.events import PersistencePolicy, Result, Severity
from storage.history_store import HistoryStore


@dataclass(frozen=True)
class SensorReading:
    """Timestamped trusted hardware or explicitly simulated measurement."""
    key: str
    value: Any
    unit: str
    timestamp: float
    quality: float = 1.0
    subject_id: str = "primary"
    source: str = "sensor"


class SensorAdapter(ABC):
    """Adapter interface for real and simulated optional hardware."""
    name = "sensor"
    available = False

    @abstractmethod
    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Return newly available readings without blocking."""

    def close(self) -> None:
        """Release optional resources."""


class UnavailableSensor(SensorAdapter):
    """Named placeholder that degrades gracefully when hardware is absent."""
    def __init__(self, name: str, reason: str = "not configured"):
        self.name, self.reason = name, reason
        CapabilityRegistry.instance().set(name, "sensor", CapabilityStatus.UNCONFIGURED, reason)

    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Unavailable adapters simply produce no data."""
        return []


class SensorManager:
    """Poll adapters and convert measurements into standard Results."""
    def __init__(self, adapters: list[SensorAdapter] | None = None):
        self.adapters = adapters or []
        self.history = HistoryStore.instance()
        registry = CapabilityRegistry.instance()
        for adapter in self.adapters:
            if registry.get(adapter.name) is None:
                registry.set(adapter.name, "sensor",
                             CapabilityStatus.READY if adapter.available else CapabilityStatus.UNCONFIGURED,
                             "connected" if adapter.available else "not connected")

    @classmethod
    def from_config(cls, config: dict | None, *, replay: bool = False) -> "SensorManager":
        """Build explicit simulations and safe placeholders for optional devices."""
        from .simulated import SimulatedSensor
        config = config or {}
        adapters: list[SensorAdapter] = []
        if replay:
            from .replay import ReplaySensorAdapter
            adapters.append(ReplaySensorAdapter())
        from .adapters import (BLEGattAdapter, GPIOBinaryAdapter,
                               RealSenseContextAdapter, SerialJSONSensorAdapter,
                               ThermalSensorAdapter)
        defaults = {
            "ble_heart_rate": [{"uuid": "00002a37-0000-1000-8000-00805f9b34fb",
                                "key": "heart_rate_bpm", "unit": "bpm", "format": "heart_rate"}],
            "pulse_oximeter": [{"uuid": "00002a5f-0000-1000-8000-00805f9b34fb",
                                "key": "spo2_pct", "unit": "%", "format": "sfloat_le", "offset": 1}],
            "smart_scale": [{"uuid": "00002a9d-0000-1000-8000-00805f9b34fb",
                             "key": "weight_kg", "unit": "kg", "format": "uint16_le",
                             "offset": 1, "scale": .005}],
            "ble_blood_pressure": [{"uuid": "00002a35-0000-1000-8000-00805f9b34fb",
                                    "key": "blood_pressure_systolic_mmhg", "unit": "mmHg",
                                    "format": "sfloat_le", "offset": 1},
                                   {"uuid": "00002a35-0000-1000-8000-00805f9b34fb",
                                    "key": "blood_pressure_diastolic_mmhg", "unit": "mmHg",
                                    "format": "sfloat_le", "offset": 3}],
        }
        for name in ("realsense_depth_imu", "ble_heart_rate", "pulse_oximeter",
                     "smart_scale", "ble_blood_pressure", "thermal", "environment", "door",
                     "bed_pressure", "appliance"):
            spec = config.get(name, {})
            if spec.get("simulated"):
                adapters.append(SimulatedSensor(name, spec.get("readings", []),
                                                interval=float(spec.get("interval", 5))))
            elif spec.get("enabled") and name == "realsense_depth_imu":
                adapters.append(RealSenseContextAdapter(**spec))
            elif spec.get("enabled") and name in defaults:
                ble_params = {k: v for k, v in spec.items()
                              if k not in ("enabled", "address", "characteristics")}
                adapters.append(BLEGattAdapter(name, address=spec.get("address"),
                    characteristics=spec.get("characteristics", defaults[name]), **ble_params))
            elif spec.get("enabled") and name == "thermal":
                adapters.append(ThermalSensorAdapter(**spec))
            elif spec.get("enabled") and name == "environment":
                adapters.append(SerialJSONSensorAdapter(name, **spec))
            elif spec.get("enabled") and name in ("door", "bed_pressure", "appliance"):
                adapters.append(GPIOBinaryAdapter(name, **spec))
            else:
                adapters.append(UnavailableSensor(name,
                    "disabled" if not spec.get("enabled") else "optional driver or hardware unavailable"))
        return cls(adapters)

    def feed_context(self, ctx) -> None:
        """Feed synchronized replay data to adapters that support it."""
        for adapter in self.adapters:
            feed = getattr(adapter, "feed_context", None)
            if feed is not None:
                feed(ctx)

    def poll(self, now: float | None = None) -> list[Result]:
        """Return only public measurement summaries, never device payloads."""
        out = []
        for adapter in self.adapters:
            for reading in adapter.poll(now):
                sensitive = reading.key.startswith(("blood_pressure", "spo2", "temperature",
                                                    "skin_temperature", "weight"))
                trusted = reading.source.startswith(("sensor:", "simulated:", "replay_sensor"))
                if sensitive and not trusted:
                    continue
                simulated = reading.source.startswith(("simulated:", "replay_sensor"))
                label = "Simulated " if simulated else ""
                severity = Severity.INFO
                qualifier = ""
                if isinstance(reading.value, (int, float)):
                    self.history.add("sensor", reading.key, float(reading.value),
                                     reading.timestamp, reading.subject_id)
                    if reading.key in ("temperature_c", "skin_temperature_c") \
                            and float(reading.value) >= 38:
                        severity, qualifier = Severity.NOTICE, " (elevated sensor measurement)"
                    elif reading.key == "spo2_pct" and float(reading.value) < 92:
                        severity, qualifier = Severity.WARNING, " (low sensor measurement)"
                out.append(Result("sensor", reading.key,
                                  {"value": reading.value, "unit": reading.unit},
                                  reading.quality, severity,
                                  (f"{label}{reading.key.replace('_', ' ')}: "
                                   f"{reading.value} {reading.unit}{qualifier}").strip(),
                                  ttl=30, subject_id=reading.subject_id,
                                  source=reading.source, quality=reading.quality,
                                  persistence=PersistencePolicy.BASELINE))
        return out

    def close(self) -> None:
        """Close all configured adapters."""
        for adapter in self.adapters:
            adapter.close()
