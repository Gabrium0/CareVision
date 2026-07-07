"""Remote photoplethysmography (rPPG) heart rate + HRV.

Runs one or more pluggable backends and reports each one's numbers so they
can be compared side by side:
- "classical": forehead green-channel bandpass + FFT (numpy/scipy only).
- "openrppg":  neural models from the open-rppg toolbox (needs `rppg`+jax;
               degrades to nothing if unavailable).

Configure in config/modules.yaml, e.g.:
    heart_rate:
      backends: [classical, openrppg]

Each backend emits its own keys suffixed with the backend label
(bpm_classical, bpm_open_rppg, hrv_rmssd_ms_classical, ...), so the
dashboard shows both readings at once.

Reliability: medium. Sensitive to lighting, motion, and skin tone. Emitted
confidence reflects signal quality; treat as a trend indicator, not a
medical measurement.
"""
from __future__ import annotations

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules.rppg_backends.classical import ClassicalBackend
from modules.rppg_backends.openrppg import OpenRPPGBackend

_BACKENDS = {
    "classical": ClassicalBackend,
    "openrppg": OpenRPPGBackend,
}


@register("heart_rate")
class HeartRate(DetectionModule):
    interval = 0.0
    requires = ("face",)
    window_seconds = 12.0
    backends = ["classical"]          # overridden by config

    def __init__(self, **params):
        super().__init__(**params)
        self._backends = []
        for name in self.backends:
            cls = _BACKENDS.get(name)
            if cls is None:
                print(f"[heart_rate] unknown backend '{name}', skipping")
                continue
            inst = cls(window_seconds=self.window_seconds)
            if getattr(inst, "available", True):
                self._backends.append(inst)
        if not self._backends:
            # ensure at least the classical backend is present
            self._backends.append(ClassicalBackend(window_seconds=self.window_seconds))

    @staticmethod
    def _key(base: str, label: str) -> str:
        return f"{base}_{label.replace('-', '_')}"

    def process(self, ctx: FrameContext):
        results = []
        for be in self._backends:
            be.update(ctx)
            reading = be.compute()
            if not reading:
                continue
            label = be.label
            bpm = reading.get("bpm")
            conf = float(reading.get("confidence", 0.4))
            if bpm is not None:
                sev = Severity.INFO
                msg = f"HR ({label}) ~{bpm:.0f} bpm"
                if conf >= 0.35 and (bpm < 50 or bpm > 110):
                    sev = Severity.WARNING
                    msg = f"HR ({label}) ~{bpm:.0f} bpm (outside typical resting range)"
                results.append(self.result(self._key("bpm", label), bpm, conf,
                                           sev, msg, ttl=8.0))
            if "hrv_rmssd_ms" in reading:
                v = reading["hrv_rmssd_ms"]
                results.append(self.result(
                    self._key("hrv_rmssd_ms", label), v, round(conf * 0.8, 2),
                    Severity.INFO, f"HRV RMSSD ({label}) ~{v:.0f} ms", ttl=8.0))
            if "hrv_sdnn_ms" in reading:
                v = reading["hrv_sdnn_ms"]
                results.append(self.result(
                    self._key("hrv_sdnn_ms", label), v, round(conf * 0.8, 2),
                    Severity.INFO, f"HRV SDNN ({label}) ~{v:.0f} ms", ttl=8.0))
            if "breaths_per_min" in reading:
                v = reading["breaths_per_min"]
                results.append(self.result(
                    self._key("breaths_per_min", label), v, round(conf * 0.8, 2),
                    Severity.INFO, f"Respiration ({label}) ~{v:.0f} /min", ttl=8.0))
        return results or None

    def close(self):
        for be in self._backends:
            be.close()
