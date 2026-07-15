"""Concrete optional hardware adapters with graceful dependency fallback."""
from __future__ import annotations

import asyncio
import json
import queue
import struct
import threading
import time
from pathlib import Path

import numpy as np

from core.capabilities import CapabilityRegistry, CapabilityStatus
from .base import SensorAdapter, SensorReading


class RealSenseContextAdapter(SensorAdapter):
    """Derive metric environment/mobility readings from aligned depth and IMU."""
    name = "realsense_depth_imu"
    available = False

    def __init__(self, interval: float = 1.0, **_params):
        self.interval, self._last = interval, -1e9
        self._pending: queue.Queue[SensorReading] = queue.Queue()
        CapabilityRegistry.instance().set(self.name, "sensor", CapabilityStatus.DEGRADED,
                                          "configured; waiting for aligned depth")

    def feed_context(self, ctx) -> None:
        """Summarize aligned depth without retaining a depth frame."""
        if ctx.depth is None or ctx.timestamp - self._last < self.interval:
            return
        if not self.available:
            self.available = True
            CapabilityRegistry.instance().set(self.name, "sensor", CapabilityStatus.READY,
                                              "aligned depth and IMU available")
        self._last = ctx.timestamp
        depth_m = ctx.depth.astype(np.float32) * ctx.depth_scale
        valid = depth_m[(depth_m > .15) & (depth_m < 10)]
        if valid.size:
            self._put("nearest_obstacle_m", float(np.percentile(valid, 2)), "m", ctx)
            floor_band = depth_m[int(ctx.h*.75):, :]
            floor = floor_band[(floor_band > .15) & (floor_band < 10)]
            if floor.size:
                self._put("floor_plane_distance_m", float(np.median(floor)), "m", ctx)
        self._put("camera_motion_rad_s", float(ctx.ego_motion), "rad/s", ctx)
        if ctx.pose is not None:
            px = ctx.pose_px()
            hip = np.mean(px[[23, 24]], axis=0)
            ankle = np.mean(px[[27, 28]], axis=0)
            hip3, ankle3 = ctx.deproject(*hip), ctx.deproject(*ankle)
            if hip3 is not None and ankle3 is not None:
                height = abs(float(ankle3[1]-hip3[1]))
                self._put("hip_height_above_floor_m", height, "m", ctx)
                shoulder = np.mean(px[[11,12]], axis=0)
                torso = hip-shoulder
                angle = float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1])+1e-6)))
                if angle > 55:
                    self._put("fall_height_m", height, "m", ctx)
            left, right = ctx.deproject(*px[27]), ctx.deproject(*px[28])
            if left is not None and right is not None:
                width = float(np.linalg.norm(left-right))
                self._put("stance_width_m", width, "m", ctx)
                self._put("metric_step_width_m", width, "m", ctx)

    def _put(self, key: str, value: float, unit: str, ctx) -> None:
        self._pending.put(SensorReading(key, round(value, 3), unit, ctx.timestamp,
                                        .9, "primary", "sensor:realsense_depth_imu"))

    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Drain derived metric readings without blocking."""
        out = []
        while True:
            try:
                out.append(self._pending.get_nowait())
            except queue.Empty:
                return out


class BLEGattAdapter(SensorAdapter):
    """Background BLE notification adapter with configurable characteristic parsers."""
    available = False

    def __init__(self, name: str, address: str | None = None,
                 characteristics: list[dict] | None = None, reconnect_seconds: float = 5,
                 **_params):
        self.name, self.address = name, address
        self.characteristics = characteristics or []
        self.reconnect_seconds = reconnect_seconds
        self._pending: queue.Queue[SensorReading] = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        if not address or not self.characteristics:
            self._status(CapabilityStatus.UNAVAILABLE, "BLE address/characteristics not configured")
            return
        try:
            import bleak  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self._status(CapabilityStatus.UNAVAILABLE, f"bleak unavailable: {type(exc).__name__}")
            return
        self.available = True
        self._status(CapabilityStatus.DEGRADED, "BLE worker starting")
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"ble-{name}")
        self._thread.start()

    def _status(self, status: CapabilityStatus, detail: str) -> None:
        CapabilityRegistry.instance().set(self.name, "sensor", status, detail)

    def _run(self) -> None:
        """Own a reconnecting asyncio BLE client outside capture threads."""
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        from bleak import BleakClient
        while not self._stop.is_set():
            try:
                async with BleakClient(self.address) as client:
                    self._status(CapabilityStatus.READY, "BLE connected")
                    for spec in self.characteristics:
                        await client.start_notify(spec["uuid"],
                            lambda _sender, data, s=spec: self._notification(s, bytes(data)))
                    while not self._stop.is_set() and client.is_connected:
                        await asyncio.sleep(.25)
            except Exception as exc:  # noqa: BLE001
                self._status(CapabilityStatus.DEGRADED,
                             f"BLE reconnecting after {type(exc).__name__}")
                await asyncio.sleep(self.reconnect_seconds)

    def _notification(self, spec: dict, data: bytes) -> None:
        try:
            parser = str(spec.get("format", "uint8"))
            offset = int(spec.get("offset", 0))
            if parser == "heart_rate":
                value = int.from_bytes(data[1:3], "little") if data[0] & 1 else data[1]
            elif parser == "uint16_le":
                value = int.from_bytes(data[offset:offset+2], "little")
            elif parser == "sint16_le":
                value = int.from_bytes(data[offset:offset+2], "little", signed=True)
            elif parser == "float32_le":
                value = struct.unpack_from("<f", data, offset)[0]
            elif parser == "sfloat_le":
                raw = int.from_bytes(data[offset:offset+2], "little")
                mantissa = raw & 0x0FFF
                if mantissa >= 0x0800:
                    mantissa -= 0x1000
                exponent = (raw >> 12) & 0x0F
                if exponent >= 8:
                    exponent -= 16
                value = mantissa * (10 ** exponent)
            else:
                value = data[offset]
            value = float(value) * float(spec.get("scale", 1))
            self._pending.put(SensorReading(str(spec["key"]), value,
                str(spec.get("unit", "")), time.time(), float(spec.get("quality", .98)),
                str(spec.get("subject_id", "primary")), f"sensor:{self.name}"))
        except (IndexError, KeyError, ValueError, struct.error):
            self._status(CapabilityStatus.DEGRADED, "invalid BLE measurement payload")

    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Drain decoded BLE measurements."""
        out = []
        while True:
            try:
                out.append(self._pending.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        """Stop the reconnecting BLE worker."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


class ThermalSensorAdapter(SensorAdapter):
    """MLX90640 adapter exposing only temperature summaries, never thermal frames."""
    name = "thermal"
    available = False

    def __init__(self, interval: float = 2.0, **_params):
        self.interval, self._last = interval, -1e9
        try:
            import board
            import busio
            import adafruit_mlx90640
            self._sensor = adafruit_mlx90640.MLX90640(busio.I2C(board.SCL, board.SDA))
            self.available = True
            CapabilityRegistry.instance().set(self.name, "sensor", CapabilityStatus.READY,
                                               "MLX90640 connected")
        except Exception as exc:  # noqa: BLE001
            CapabilityRegistry.instance().set(self.name, "sensor", CapabilityStatus.UNAVAILABLE,
                                               f"thermal unavailable: {type(exc).__name__}")

    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Read one thermal frame and immediately reduce it to safe summaries."""
        now = time.time() if now is None else now
        if not self.available or now - self._last < self.interval:
            return []
        frame = [0.0] * 768
        try:
            self._sensor.getFrame(frame)
        except Exception as exc:  # noqa: BLE001
            CapabilityRegistry.instance().set(self.name, "sensor", CapabilityStatus.DEGRADED,
                                               f"thermal read failed: {type(exc).__name__}")
            return []
        self._last = now
        values = np.asarray(frame)
        return [SensorReading("skin_temperature_c", round(float(np.percentile(values, 90)), 2),
                              "C", now, .8, source="sensor:thermal"),
                SensorReading("ambient_temperature_c", round(float(np.percentile(values, 10)), 2),
                              "C", now, .8, source="sensor:thermal")]


class SerialJSONSensorAdapter(SensorAdapter):
    """Non-blocking serial adapter for environment, air-quality, pollen, or scale hubs."""
    available = False

    def __init__(self, name: str, port: str | None = None, baudrate: int = 115200,
                 fields: dict | None = None, **_params):
        self.name, self.fields = name, fields or {}
        self._pending: queue.Queue[SensorReading] = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        if not port:
            CapabilityRegistry.instance().set(name, "sensor", CapabilityStatus.UNAVAILABLE,
                                               "serial port not configured")
            return
        try:
            import serial
            self._serial = serial.Serial(port, baudrate=baudrate, timeout=.25)
        except Exception as exc:  # noqa: BLE001
            CapabilityRegistry.instance().set(name, "sensor", CapabilityStatus.UNAVAILABLE,
                                               f"serial unavailable: {type(exc).__name__}")
            return
        self.available = True
        CapabilityRegistry.instance().set(name, "sensor", CapabilityStatus.READY,
                                           "serial hub connected")
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"serial-{name}")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = json.loads(self._serial.readline().decode("utf-8"))
                for incoming, spec in self.fields.items():
                    if incoming not in payload:
                        continue
                    if isinstance(spec, str):
                        spec = {"key": spec}
                    self._pending.put(SensorReading(str(spec.get("key", incoming)),
                        payload[incoming], str(spec.get("unit", "")), time.time(),
                        float(spec.get("quality", .95)), source=f"sensor:{self.name}"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue

    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Drain decoded serial summaries."""
        out = []
        while True:
            try:
                out.append(self._pending.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        """Stop the serial worker and close its port."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if getattr(self, "_serial", None):
            self._serial.close()


class GPIOBinaryAdapter(SensorAdapter):
    """Door, bed-pressure, or appliance state adapter using gpiozero."""
    available = False

    def __init__(self, name: str, pin: int | None = None, key: str = "active",
                 active_high: bool = True, **_params):
        self.name, self.key, self._last = name, key, None
        if pin is None:
            CapabilityRegistry.instance().set(name, "sensor", CapabilityStatus.UNAVAILABLE,
                                               "GPIO pin not configured")
            return
        try:
            from gpiozero import DigitalInputDevice
            self._device = DigitalInputDevice(pin, active_high=active_high)
            self.available = True
            CapabilityRegistry.instance().set(name, "sensor", CapabilityStatus.READY,
                                               "GPIO input connected")
        except Exception as exc:  # noqa: BLE001
            CapabilityRegistry.instance().set(name, "sensor", CapabilityStatus.UNAVAILABLE,
                                               f"GPIO unavailable: {type(exc).__name__}")

    def poll(self, now: float | None = None) -> list[SensorReading]:
        """Emit state changes only."""
        if not self.available:
            return []
        value = bool(self._device.value)
        if value == self._last:
            return []
        self._last = value
        return [SensorReading(self.key, value, "", time.time() if now is None else now,
                              .99, source=f"sensor:{self.name}")]

    def close(self) -> None:
        """Close the optional GPIO device."""
        if getattr(self, "_device", None):
            self._device.close()
