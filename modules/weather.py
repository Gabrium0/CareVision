"""Current weather from Open-Meteo for clothing recommendations.

No API key is required. Configure latitude/longitude in config/modules.yaml or
via WEATHER_LAT / WEATHER_LON environment variables.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

from core.context import FrameContext
from core.debug import log as debug_log
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule


@register("weather")
class Weather(DetectionModule):
    interval = 0.0
    latitude = None
    longitude = None
    refresh_seconds = 600.0
    timeout_seconds = 4.0

    def __init__(self, **params):
        super().__init__(**params)
        self._last_fetch = 0.0
        self._cached: dict | None = None
        self._status = "not configured"

    def _coords(self) -> tuple[float, float] | None:
        lat = self.latitude if self.latitude is not None else os.environ.get("WEATHER_LAT")
        lon = self.longitude if self.longitude is not None else os.environ.get("WEATHER_LON")
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None

    def _fetch(self) -> dict | None:
        coords = self._coords()
        if coords is None:
            self._status = "set latitude/longitude"
            return self._cached
        lat, lon = coords
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join([
                "temperature_2m", "apparent_temperature", "relative_humidity_2m",
                "precipitation", "rain", "wind_speed_10m", "uv_index",
            ]),
            "forecast_days": 1,
            "timezone": "auto",
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            self._status = f"weather fetch failed: {type(e).__name__}"
            return self._cached
        cur = payload.get("current") or {}
        self._cached = {
            "temperature_c": cur.get("temperature_2m"),
            "feels_like_c": cur.get("apparent_temperature"),
            "humidity_pct": cur.get("relative_humidity_2m"),
            "precipitation_mm": cur.get("precipitation"),
            "rain_mm": cur.get("rain"),
            "wind_kph": cur.get("wind_speed_10m"),
            "uv_index": cur.get("uv_index"),
            "status": "ready",
        }
        self._status = "ready"
        return self._cached

    def process(self, ctx: FrameContext):
        now = time.time()
        if self._cached is None or now - self._last_fetch >= self.refresh_seconds:
            self._last_fetch = now
            self._fetch()
        weather = dict(self._cached or {})
        weather.setdefault("status", self._status)
        ctx.extras["weather"] = weather
        debug_log("weather", f"status={weather.get('status')} temp={weather.get('temperature_c')} "
                             f"feels={weather.get('feels_like_c')} rain={weather.get('rain_mm')} "
                             f"wind={weather.get('wind_kph')}")

        results = [self.result("status", weather["status"], 0.0, Severity.INFO, "", ttl=20.0)]
        for key in ("temperature_c", "feels_like_c", "humidity_pct", "rain_mm", "wind_kph", "uv_index"):
            value = weather.get(key)
            if value is not None:
                results.append(self.result(key, round(float(value), 1), 0.8, Severity.INFO, "", ttl=20.0))
        return results
