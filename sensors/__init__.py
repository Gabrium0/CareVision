"""Optional hardware and deterministic simulated sensor adapters."""

from .base import SensorAdapter, SensorManager, SensorReading
from .simulated import SimulatedSensor
from .replay import ReplaySensorAdapter
from .adapters import (BLEGattAdapter, GPIOBinaryAdapter, RealSenseContextAdapter,
                       SerialJSONSensorAdapter, ThermalSensorAdapter)

__all__ = ["SensorAdapter", "SensorManager", "SensorReading", "SimulatedSensor",
           "ReplaySensorAdapter"]
