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

import threading

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
    classical_talk_delta_threshold = 0.05   # mouth-aspect-ratio delta that flags talking
    openrppg_model = None             # None = open-rppg package default
    openrppg_brightness_normalize = True   # lift dark crops toward training range
    openrppg_infer_every = 5.0
    openrppg_min_seconds = 10.0
    openrppg_hrv_min_seconds = 30.0
    openrppg_min_confidence = 0.35
    openrppg_smoothing_window = 5
    openrppg_motion_threshold = 18.0
    openrppg_face_jitter_threshold = 0.12
    openrppg_face_reacquire_frames = 3
    openrppg_inference_timeout_seconds = 90.0
    openrppg_cpu_reserved_cores = 2
    openrppg_async_inference = True
    debug_backend_values = False
    bpm_jump_threshold = 20.0
    bpm_jump_min_confidence = 0.65

    def __init__(self, **params):
        super().__init__(**params)
        self._fast_fed = False    # True once fast_update() has fed a backend directly
        self._backends = []
        self._last_bpm = None
        self._last_source: str | None = None
        self._diagnostic_lock = threading.Lock()
        self._latest_readings: dict[str, dict] = {}
        self._unavailable_backends: list[dict] = []
        for name in self.backends:
            cls = _BACKENDS.get(name)
            if cls is None:
                print(f"[heart_rate] unknown backend '{name}', skipping")
                self._unavailable_backends.append({
                    "name": name, "available": False, "status": "unknown backend"})
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
                    face_reacquire_frames=self.openrppg_face_reacquire_frames,
                    inference_timeout_seconds=self.openrppg_inference_timeout_seconds,
                    cpu_reserved_cores=self.openrppg_cpu_reserved_cores,
                    async_inference=self.openrppg_async_inference,
                    brightness_normalize=self.openrppg_brightness_normalize,
                )
            else:
                inst = cls(window_seconds=self.window_seconds,
                           method=self.classical_method,
                           smoothing_window=self.classical_smoothing_window,
                           talk_delta_threshold=self.classical_talk_delta_threshold)
            if getattr(inst, "available", True):
                self._backends.append(inst)
            else:
                self._unavailable_backends.append({
                    "name": getattr(inst, "label", name), "available": False,
                    "status": getattr(inst, "_status", "dependency or model unavailable")})
                inst.close()
        if not self._backends:
            # ensure at least the classical backend is present
            self._backends.append(ClassicalBackend(window_seconds=self.window_seconds))

    def diagnostics(self) -> dict:
        """Return JSON-safe backend progress without exposing raw samples/crops."""
        with self._diagnostic_lock:
            latest = {key: dict(value) for key, value in self._latest_readings.items()}
            source = self._last_source
        backends = []
        for backend in self._backends:
            snapshot = backend.diagnostics() if hasattr(backend, "diagnostics") else {
                "name": backend.label, "available": getattr(backend, "available", True),
                "status": "available",
            }
            reading = latest.get(backend.label)
            if reading is not None:
                snapshot["latest"] = reading
            backends.append(snapshot)
        backends.extend(dict(item) for item in self._unavailable_backends)
        return {"canonical_source": source, "backends": backends}

    @staticmethod
    def _key(base: str, label: str) -> str:
        return f"{base}_{label.replace('-', '_')}"

    def _placeholder(self, metric: str, label: str):
        return self.result(self._key(metric, label), "...", 0.0, Severity.INFO, "", ttl=8.0)

    def _status_result(self, label: str, value: str):
        return self.result(self._key("status", label), value, 0.0, Severity.INFO, "", ttl=8.0)

    @staticmethod
    def _numeric_bpm(reading: dict | None) -> float | None:
        if not reading or reading.get("bpm") is None:
            return None
        try:
            bpm = float(reading["bpm"])
        except (TypeError, ValueError):
            return None
        return bpm if 35.0 <= bpm <= 180.0 else None

    def _backend_results(self, label: str, reading: dict | None):
        """Verbose per-backend diagnostics, intended for testing/tuning mode."""
        results = []
        if not reading:
            if label.startswith("open-rppg"):
                for metric in ("bpm", "hrv_rmssd_ms", "hrv_sdnn_ms", "breaths_per_min"):
                    results.append(self._placeholder(metric, label))
                status = getattr(next((b for b in self._backends if b.label == label), None),
                                 "_status", "warming up")
                results.append(self._status_result(label, status))
            return results

        if label.startswith("open-rppg"):
            results.append(self._status_result(label, str(reading.get("status", "ready"))))
        bpm = reading.get("bpm", reading.get("raw_bpm"))
        conf = float(reading.get("confidence", reading.get("raw_confidence", 0.4)))
        if bpm is not None:
            sev = Severity.INFO
            msg = f"HR ({label}) ~{float(bpm):.0f} bpm"
            if reading.get("rejected_reason"):
                msg += f" ({reading['rejected_reason']})"
                conf = min(conf, 0.2)
            elif conf >= 0.35 and (float(bpm) < 50 or float(bpm) > 110):
                sev = Severity.WARNING
                msg = f"HR ({label}) ~{float(bpm):.0f} bpm (outside typical resting range)"
            results.append(self.result(self._key("bpm", label), float(bpm), conf,
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
        return results

    def _canonical_bpm_result(self, readings: list[tuple[str, dict]]):
        candidates = []
        for label, reading in readings:
            bpm = self._numeric_bpm(reading)
            if bpm is None:
                continue
            conf = float(reading.get("confidence", 0.0))
            if self._last_bpm is not None and abs(bpm - self._last_bpm) > self.bpm_jump_threshold:
                if conf < self.bpm_jump_min_confidence:
                    continue
            candidates.append((conf, label == "classical", label, bpm))
        if not candidates:
            return None
        conf, _, label, bpm = max(candidates, key=lambda item: (item[0], item[1]))
        self._last_bpm = bpm
        with self._diagnostic_lock:
            self._last_source = label
        sev = Severity.INFO
        msg = f"HR ~{bpm:.0f} bpm ({label})"
        if conf >= 0.35 and (bpm < 50 or bpm > 110):
            sev = Severity.WARNING
            msg = f"HR ~{bpm:.0f} bpm ({label}, outside typical resting range)"
        return self.result("bpm", round(bpm, 1), round(conf, 2), sev, msg, ttl=8.0)

    def fast_update(self, ctx: FrameContext) -> None:
        """Feed every backend from the camera's reader thread (see module
        docstring); called once per raw captured frame, independent of the
        heavy loop's cadence."""
        self._fast_fed = True
        for be in self._backends:
            be.update(ctx)

    def reset_capture(self) -> None:
        """Discard samples after a showcase framing/quality failure.

        A new stable-capture period must not calculate a pulse from frames
        gathered while the guest was moving, out of position, or competing
        with another person.  Backends intentionally expose simple rolling
        buffers, so clearing the shared buffer/crop contracts is enough to
        require a fresh window without rebuilding optional neural models.
        """
        self._last_bpm = None
        with self._diagnostic_lock:
            self._last_source = None
            self._latest_readings.clear()
        for be in self._backends:
            reset = getattr(be, "reset", None)
            if reset is not None:
                reset()

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        results = []
        readings = []
        for be in self._backends:
            if not self._fast_fed:
                be.update(ctx)
            reading = be.compute()
            label = be.label
            if reading:
                readings.append((label, reading))
                with self._diagnostic_lock:
                    self._latest_readings[label] = {
                        **dict(reading), "updated_at": ctx.timestamp}
            if self.debug_backend_values:
                results.extend(self._backend_results(label, reading))
        canonical = self._canonical_bpm_result(readings)
        if canonical is not None:
            results.insert(0, canonical)
        return results or None

    def close(self):
        """Release any resources (models, threads, sockets) held here."""
        for be in self._backends:
            be.close()
