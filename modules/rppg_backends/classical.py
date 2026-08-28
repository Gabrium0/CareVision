"""Classical rPPG backend: multi-ROI skin sampling + chrominance + FFT.

Dependency-light (numpy/scipy only). Samples skin patches from the forehead
and both cheeks, takes each patch's per-channel (R,G,B) mean, buffers them,
resamples, and projects to an illumination-robust pulse signal:

- method "chrom"/"pos": CHROM (de Haan 2013) / POS (Wang 2017) — divide each
  channel by its temporal mean so overall brightness/lighting cancels, then
  combine on the chrominance plane. Robust to poor/uneven light.
- method "green": legacy forehead green-channel mean, kept for A/B comparison.

The projected signal is bandpassed 0.7-3 Hz and its dominant frequency read as
the heart rate; HRV RMSSD/SDNN come from inter-beat intervals. Confidence is
scaled by a low-light quality factor so dark-room readings aren't over-trusted.

Each compute() call independently FFT-picks a peak from the current buffer, so
a single call can land on transient noise or a harmonic; reported bpm is the
median of the last `smoothing_window` raw picks (mirroring OpenRPPGBackend's
`_bpm_history` in rppg_backends/openrppg.py), and confidence is further
penalized when those recent picks disagree a lot (same jitter-penalty shape
as openrppg.py). This makes the two "side by side" backends' numbers directly
comparable instead of one being smoothed and the other raw.

Two of the three sampled ROIs are the cheeks, which move with speech (jaw/
mouth motion drags adjacent skin), injecting non-cardiac motion straight into
the chrominance signal. update() tracks the frame-to-frame mouth-aspect-ratio
delta (not an absolute open/closed threshold -- yawn.py's own docs note MAR
moves for both yawning and talking, and talking's swings are faster/smaller
than a sustained yawn) and falls back to forehead-only sampling for a frame
when that delta suggests active mouth movement.

Reliability: medium; sensitive to lighting, motion, and skin tone.
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

from core.context import FrameContext
from modules._util import (TimedBuffer, dominant_frequency, bandpass,
                           peak_intervals, roi_patch, patch_brightness,
                           low_light_factor, rppg_input_quality, chrom, pos,
                           mouth_aspect_ratio)
from extractors import face_landmarks as FL
from .base import RPPGBackend

# Skin ROIs sampled every frame; averaging more pixels raises SNR (~1/sqrt(N)).
_ROI_LANDMARKS = (FL.FOREHEAD_TOP, FL.LEFT_CHEEK, FL.RIGHT_CHEEK)


class ClassicalBackend(RPPGBackend):
    """Classical rPPG backend: multi-ROI skin chrominance + FFT."""
    label = "classical"
    available = True
    required_samples = 8

    def __init__(self, window_seconds: float = 12.0, method: str = "chrom",
                 smoothing_window: int = 5, talk_delta_threshold: float = 0.05):
        self.window_seconds = window_seconds
        self.method = method if method in ("chrom", "pos", "green") else "chrom"
        self.talk_delta_threshold = talk_delta_threshold
        self.buf = TimedBuffer(window_seconds)          # values are (R,G,B) means
        self._brightness = 128.0                        # EMA of ROI luminance
        # update() (writer) and compute() (reader) can run on different
        # threads when fed via the camera's fast path (see core/pipeline.py).
        self._lock = threading.Lock()
        self._accepted = 0
        self._rejected_no_roi = 0
        self._quality_events: deque[tuple[float, bool]] = deque()
        self._last_reading: dict | None = None
        # compute() only ever runs on the heavy-loop thread (see
        # modules/heart_rate.py), so this needs no lock of its own.
        self._bpm_history: deque[float] = deque(maxlen=max(1, int(smoothing_window)))
        # update() only ever runs on one thread at a time (the reader thread
        # once the fast path engages, else the heavy loop) -- same
        # single-writer assumption openrppg.py's _stable_face()/_last_bbox
        # already relies on, so this needs no lock either.
        self._last_mar: float | None = None

    def update(self, ctx: FrameContext) -> None:
        """Feed one frame's data into the backend's rolling state."""
        device = ctx.extras.get("rppg_samples")
        if device:
            self._ingest_device_samples(device)
            return
        roi_landmarks = _ROI_LANDMARKS
        mar = mouth_aspect_ratio(ctx)
        if mar is not None:
            if (self._last_mar is not None
                    and abs(mar - self._last_mar) > self.talk_delta_threshold):
                # Rapid mouth movement (talking) -- cheeks are moving, so
                # drop them for this sample rather than let non-cardiac
                # motion dilute the chrominance signal.
                roi_landmarks = (FL.FOREHEAD_TOP,)
            self._last_mar = mar
        pixels = []
        for idx in roi_landmarks:
            patch = roi_patch(ctx, idx, radius_frac=0.10)
            if patch is not None and patch.size:
                pixels.append(patch.reshape(-1, 3))
        if not pixels:
            with self._lock:
                self._rejected_no_roi += 1
                self._quality_events.append((ctx.timestamp, False))
                self._prune_quality_events(ctx.timestamp)
            return
        px = np.concatenate(pixels, axis=0).astype(np.float64)   # BGR
        # store as (R, G, B) so chrom/pos get channels in the expected order
        rgb_mean = px[:, ::-1].mean(axis=0)
        bright = patch_brightness(px)
        with self._lock:
            self.buf.push(ctx.timestamp, rgb_mean)
            self._brightness = 0.9 * self._brightness + 0.1 * bright
            self._accepted += 1
            self._quality_events.append((ctx.timestamp, True))
            self._prune_quality_events(ctx.timestamp)

    def _ingest_device_samples(self, samples) -> None:
        """Push iPad-computed ROI colour means straight into the buffer.

        The native app samples forehead/cheek skin colour from the *raw* camera
        pixels (no JPEG chroma loss) with the frame's precise PTS and applies its
        own mouth-motion talk-guard, handing us a ready ``(ts, (R,G,B))`` series.
        ``compute()`` is unchanged — it only reads ``self.buf`` — so this removes
        the two dominant rPPG error sources (JPEG 4:2:0 chroma + timestamp jitter)
        without touching the estimation math. See docs/IPAD_NATIVE_APP.md.
        """
        with self._lock:
            for ts, rgb, _n in samples:
                rgb = np.asarray(rgb, dtype=np.float64)
                self.buf.push(ts, rgb)
                bright = float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
                self._brightness = 0.9 * self._brightness + 0.1 * bright
                self._accepted += 1
                self._quality_events.append((ts, True))
                self._prune_quality_events(ts)

    def _prune_quality_events(self, now: float) -> None:
        while self._quality_events and now - self._quality_events[0][0] > self.window_seconds:
            self._quality_events.popleft()

    def diagnostics(self) -> dict:
        """Return a thread-safe snapshot without copying raw signal samples."""
        with self._lock:
            samples = len(self.buf)
            span = self.buf.span()
            brightness = self._brightness
            latest = dict(self._last_reading) if self._last_reading else None
            accepted = self._accepted
            rejected = self._rejected_no_roi
            rolling = list(self._quality_events)
        required = 6.0
        sample_progress = min(1.0, samples / self.required_samples)
        time_progress = min(1.0, span / required)
        ready = samples >= self.required_samples and span >= required
        sample_hz = ((samples - 1) / span if samples > 1 and span > 0 else 0.0)
        rolling_accepted = sum(accepted for _, accepted in rolling)
        rolling_total = len(rolling)
        return {
            "name": self.label, "available": True,
            "samples": samples, "buffered_seconds": round(span, 1),
            "required_seconds": required,
            "required_samples": self.required_samples,
            "sample_progress": round(sample_progress, 3),
            "effective_sample_hz": round(sample_hz, 2),
            "progress": round(min(time_progress, sample_progress), 3),
            "ready": ready,
            "status": ("ready" if ready else
                       f"warming up: {samples}/{self.required_samples} samples, "
                       f"{span:.0f}/{required:.0f}s"),
            "accepted": accepted, "rejected": {"no_roi": rejected},
            "rolling_acceptance_ratio": round(rolling_accepted / rolling_total, 3)
            if rolling_total else 0.0,
            "inference_latency_ms": 0.0, "latest": latest,
            "brightness": round(brightness, 1),
        }

    def reset(self) -> None:
        with self._lock:
            self.buf.t.clear()
            self.buf.v.clear()
            self._last_reading = None
            self._bpm_history.clear()
            self._last_mar = None
            self._quality_events.clear()

    def _snapshot(self):
        """Thread-safe copy of the buffer + brightness for compute() to use."""
        with self._lock:
            if len(self.buf) < 8 or self.buf.span() < 6.0:
                return None
            t, v = self.buf.arrays()
            span = self.buf.span()
            brightness = self._brightness
        return t, v, span, brightness

    def _signal(self, t: np.ndarray, v: np.ndarray, fs: float) -> np.ndarray | None:
        """Resample buffered RGB means to fs Hz and project to a pulse signal."""
        if v.ndim != 2 or v.shape[1] != 3:
            return None
        t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
        if len(t_uniform) < 16:
            return None
        rgb = np.stack([np.interp(t_uniform, t, v[:, c]) for c in range(3)], axis=1)
        if self.method == "green":
            return rgb[:, 1] - rgb[:, 1].mean()
        return pos(rgb, fs) if self.method == "pos" else chrom(rgb)

    def compute(self) -> dict | None:
        """Return the backend's current reading dict, or None if not ready."""
        snap = self._snapshot()
        if snap is None:
            return None
        t, v, span, brightness = snap
        fs = 30.0
        signal = self._signal(t, v, fs)
        if signal is None:
            return None
        filt = bandpass(signal, fs, 0.7, 3.0)
        if filt is None:
            return None
        dom = dominant_frequency(filt, fs, 0.7, 3.0)
        if dom is None:
            return None
        freq, prominence = dom
        raw_bpm = freq * 60.0
        self._bpm_history.append(raw_bpm)
        bpm = float(np.median(self._bpm_history))
        fill = min(1.0, span / self.window_seconds)
        # Photon-limited SNR: down-weight confidence when the ROI is too dark.
        light = low_light_factor(brightness)
        conf = min(1.0, prominence * 3.0) * fill * light
        if len(self._bpm_history) >= 3:
            jitter = float(np.std(np.asarray(self._bpm_history, dtype=np.float64)))
            # Mirrors openrppg.py's jitter penalty: if recent independent FFT
            # peak-picks disagree a lot, trust this reading less without fully
            # discarding it (same shape: half-weight floor at ~20bpm std).
            conf *= max(0.5, 1.0 - jitter / 20.0)
        conf = round(max(0.0, min(1.0, conf)), 2)

        intervals = np.diff(t)
        mean_interval = float(np.mean(intervals)) if len(intervals) else 0.0
        regularity = (max(0.0, 1.0 - float(np.std(intervals)) / mean_interval)
                      if mean_interval > 0 else 0.0)
        with self._lock:
            rolling = list(self._quality_events)
        acceptance = (sum(accepted for _, accepted in rolling) / len(rolling)
                      if rolling else 0.0)
        effective_hz = ((len(t) - 1) / span if len(t) > 1 and span > 0 else 0.0)
        quality = round(rppg_input_quality(brightness, effective_hz,
                                           acceptance, regularity), 2)

        out = {"bpm": round(bpm, 1), "confidence": conf, "quality": quality}
        rr = peak_intervals(filt, fs, min_distance_s=0.4)
        if len(rr) >= 4:
            rr_ms = rr * 1000.0
            out["hrv_rmssd_ms"] = round(float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))), 1)
            out["hrv_sdnn_ms"] = round(float(np.std(rr_ms)), 1)
        with self._lock:
            self._last_reading = dict(out)
        return out
