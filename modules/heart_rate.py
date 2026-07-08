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

Fast path: on a live webcam the heavy per-frame pipeline (32 modules +
MediaPipe) can run slower than the camera actually delivers frames, which
starves these frequency-domain backends of samples and aliases the FFT (see
core/pipeline.py). `fast_update()` lets core.pipeline.Pipeline feed backends
directly from the camera's reader thread at the full capture rate; once
that has happened at least once, `process()` stops re-feeding them itself
(to avoid double-counting) and only reads out the accumulated reading.
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
    """Remote photoplethysmography (rPPG) heart rate + HRV."""
    interval = 0.0
    requires = ("face",)
    window_seconds = 12.0
    backends = ["classical"]          # overridden by config
    classical_method = "chrom"        # chrom | pos | green (green = legacy A/B)
    classical_smoothing_window = 5    # median-smooth the last N raw FFT bpm picks
    openrppg_model = None             # None = open-rppg package default
    openrppg_brightness_normalize = True   # lift dark crops toward training range
    openrppg_infer_every = 5.0
    openrppg_min_seconds = 10.0
    openrppg_hrv_min_seconds = 30.0
    openrppg_min_confidence = 0.35
    openrppg_smoothing_window = 5
    openrppg_motion_threshold = 18.0
    openrppg_face_jitter_threshold = 0.12
    openrppg_async_inference = True

    def __init__(self, **params):
        super().__init__(**params)
        self._fast_fed = False    # True once fast_update() has fed a backend directly
        self._backends = []
        for name in self.backends:
            cls = _BACKENDS.get(name)
            if cls is None:
                print(f"[heart_rate] unknown backend '{name}', skipping")
                continue
            if name == "openrppg":
                inst = cls(
                    window_seconds=self.window_seconds,
                    model=self.openrppg_model,
                    infer_every=self.openrppg_infer_every,
                    min_seconds=self.openrppg_min_seconds,
                    hrv_min_seconds=self.openrppg_hrv_min_seconds,
                    min_confidence=self.openrppg_min_confidence,
                    smoothing_window=self.openrppg_smoothing_window,
                    motion_threshold=self.openrppg_motion_threshold,
                    face_jitter_threshold=self.openrppg_face_jitter_threshold,
                    async_inference=self.openrppg_async_inference,
                    brightness_normalize=self.openrppg_brightness_normalize,
                )
            else:
                inst = cls(window_seconds=self.window_seconds,
                           method=self.classical_method,
                           smoothing_window=self.classical_smoothing_window)
            if getattr(inst, "available", True):
                self._backends.append(inst)
        if not self._backends:
            # ensure at least the classical backend is present
            self._backends.append(ClassicalBackend(window_seconds=self.window_seconds))

    @staticmethod
    def _key(base: str, label: str) -> str:
        return f"{base}_{label.replace('-', '_')}"

    def _placeholder(self, metric: str, label: str):
        return self.result(self._key(metric, label), "...", 0.0, Severity.INFO, "", ttl=8.0)

    def _status_result(self, label: str, value: str):
        return self.result(self._key("status", label), value, 0.0, Severity.INFO, "", ttl=8.0)

    def fast_update(self, ctx: FrameContext) -> None:
        """Feed every backend from the camera's reader thread (see module
        docstring); called once per raw captured frame, independent of the
        heavy loop's cadence."""
        self._fast_fed = True
        for be in self._backends:
            be.update(ctx)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        results = []
        for be in self._backends:
            if not self._fast_fed:
                be.update(ctx)
            reading = be.compute()
            label = be.label
            if not reading:
                if label.startswith("open-rppg"):
                    for metric in ("bpm", "hrv_rmssd_ms", "hrv_sdnn_ms", "breaths_per_min"):
                        results.append(self._placeholder(metric, label))
                    status = getattr(be, "_status", "warming up")
                    results.append(self._status_result(label, status))
                continue
            if label.startswith("open-rppg"):
                results.append(self._status_result(label, str(reading.get("status", getattr(be, "_status", "ready")))))
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
            elif label.startswith("open-rppg"):
                results.append(self._placeholder("bpm", label))
            if "hrv_rmssd_ms" in reading:
                v = reading["hrv_rmssd_ms"]
                results.append(self.result(
                    self._key("hrv_rmssd_ms", label), v, round(conf * 0.8, 2),
                    Severity.INFO, f"HRV RMSSD ({label}) ~{v:.0f} ms", ttl=8.0))
            elif label.startswith("open-rppg"):
                results.append(self._placeholder("hrv_rmssd_ms", label))
            if "hrv_sdnn_ms" in reading:
                v = reading["hrv_sdnn_ms"]
                results.append(self.result(
                    self._key("hrv_sdnn_ms", label), v, round(conf * 0.8, 2),
                    Severity.INFO, f"HRV SDNN ({label}) ~{v:.0f} ms", ttl=8.0))
            elif label.startswith("open-rppg"):
                results.append(self._placeholder("hrv_sdnn_ms", label))
            if "breaths_per_min" in reading:
                v = reading["breaths_per_min"]
                results.append(self.result(
                    self._key("breaths_per_min", label), v, round(conf * 0.8, 2),
                    Severity.INFO, f"Respiration ({label}) ~{v:.0f} /min", ttl=8.0))
            elif label.startswith("open-rppg"):
                results.append(self._placeholder("breaths_per_min", label))
        return results or None

    def close(self):
        """Release any resources (models, threads, sockets) held here."""
        for be in self._backends:
            be.close()
