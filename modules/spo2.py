"""Contactless SpO₂ (blood-oxygen saturation) via remote photoplethysmography.

This is *remote pulse oximetry*: it reuses the same facial-skin rPPG signal the
heart-rate module extracts, but instead of reading the pulse *frequency* it reads
the **ratio-of-ratios** of the pulsatile (AC) and steady (DC) components across two
color channels:

    R    = (AC_red / DC_red) / (AC_blue / DC_blue)
    SpO₂ = A − B · R                               (A, B from a calibration file)

Why this is deliberately conservative:
- Consumer RGB channels are broad and overlapping, not the narrowband 660/940 nm a
  real oximeter uses, so R is only *weakly* tied to true SpO₂. The A/B coefficients
  are device- and subject-specific and MUST be fitted against a reference oximeter
  (see tools/spo2_calibrate.py); shipped uncalibrated, this reports a trend only.

Two acquisition modes, selected per-frame from `ctx.depth`, each with its own
channel pair and calibration coefficients (never mixed):
- **uncompressed** (RealSense, `ctx.depth is not None`): red/blue ratio, as
  designed originally. Behavior here is unchanged by the addition of the
  second mode below.
- **compressed** (any other source, e.g. the OV2735 USB2 webcam): red/green
  ratio. On MJPEG 4:2:0, luma (Y, carrying green's dominant contribution) is
  full-resolution while only chroma is subsampled, so red/green survives
  compression better than red/blue. This mode is opt-in
  (`enabled_on_compressed`, default off) because it has not been validated
  against a reference oximeter — DCT quantization noise in the chroma planes
  may still swamp the pulsatile amplitude even with the better channel pick.
  Ships uncalibrated by default like the RealSense path.

Switching sources mid-run (the 'c' hotkey, or RealSense hardware unplugged)
flushes the buffer so a single window never blends samples from the two
channel/coefficient bases.

Reliability: LOW / trend-only. Uncalibrated runs force near-zero confidence and a
"(uncalibrated, trend only)" label, and camera-only readings never escalate past a
WARNING — and only then when a real calibration file is loaded *and* confidence is
high. Treat as a screening trend, never a medical measurement.

Fast path: like modules/heart_rate.py, `fast_update()` is fed once per raw captured
frame by core/pipeline.py's dedicated vitals sampler, so the AC (pulsatile) component
isn't aliased by the heavy loop's slower cadence. `process()` only reads out.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import (TimedBuffer, bandpass, chrom, dominant_frequency,
                           low_light_factor, mouth_aspect_ratio, patch_brightness,
                           roi_patch, rppg_input_quality)
from extractors import face_landmarks as FL

# Same skin ROIs as the classical HR backend; more averaged pixels -> higher SNR.
_ROI_LANDMARKS = (FL.FOREHEAD_TOP, FL.LEFT_CHEEK, FL.RIGHT_CHEEK)
_CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2}


def channel_ac_dc(channel: np.ndarray, fs: float,
                  ac_band: tuple[float, float]) -> tuple[float, float] | None:
    """Pulsatile amplitude (AC) and steady level (DC) of one resampled channel.

    DC is the temporal mean; AC is the standard deviation of the cardiac-band
    (`ac_band` Hz) bandpassed signal. Returns None if the channel is too short
    to filter or its DC is ~0 (a division guard for the ratio-of-ratios).
    """
    dc = float(np.mean(channel))
    if not np.isfinite(dc) or abs(dc) < 1e-6:
        return None
    filt = bandpass(channel, fs, ac_band[0], ac_band[1])
    if filt is None:
        return None
    ac = float(np.std(filt))
    return ac, dc


def ratio_of_ratios(rgb: np.ndarray, fs: float, num_idx: int, den_idx: int,
                    ac_band: tuple[float, float]) -> float | None:
    """(AC/DC of `num_idx` channel) / (AC/DC of `den_idx` channel) on an (N,3) window.

    Returns None when either channel lacks a usable AC/DC (too short, flat DC, or a
    zero denominator) — the caller treats that as "no reading this window".
    """
    c = np.asarray(rgb, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        return None
    num = channel_ac_dc(c[:, num_idx], fs, ac_band)
    den = channel_ac_dc(c[:, den_idx], fs, ac_band)
    if num is None or den is None:
        return None
    ac_n, dc_n = num
    ac_d, dc_d = den
    perfusion_den = ac_d / dc_d
    if abs(perfusion_den) < 1e-9:
        return None
    return (ac_n / dc_n) / perfusion_den


def ratio_to_spo2(ratio: float, a_coeff: float, b_coeff: float) -> float:
    """Empirical linear map SpO₂ = A − B·R (coefficients from calibration)."""
    return a_coeff - b_coeff * ratio


@register("spo2")
class SpO2(DetectionModule):
    """Contactless rPPG blood-oxygen saturation (trend-only; RealSense mode
    calibrated identically to before, plain-webcam mode opt-in)."""
    interval = 0.5
    requires = ("face",)
    window_seconds = 20.0
    ac_band = (0.7, 3.0)             # cardiac band for the pulsatile AC component
    channels = ("red", "blue")       # uncompressed-source (RealSense) ratio pair
    compressed_channels = ("red", "green")  # compressed-source (webcam) ratio pair
    enabled_on_compressed = False    # opt-in: unvalidated against a reference oximeter
    calibration_file = "assets/spo2_calibration.json"
    min_pulse_prominence = 0.15      # require a clean concurrent pulse before reporting
    display_min = 70.0
    display_max = 100.0
    warn_below = 92.0
    warn_min_confidence = 0.5        # WARNING needs calibration AND this much confidence
    talk_delta_threshold = 0.05      # mouth-aspect-ratio delta that flags talking
    required_samples = 8
    required_seconds = 6.0
    resample_hz = 30.0
    debug_ratio = False              # emit spo2_ratio for tuning / calibration capture

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(self.window_seconds)     # values are (R,G,B) means
        self._brightness = 128.0                        # EMA of ROI luminance
        # fast_update (writer, sampler thread) and process (reader, heavy loop) can
        # run concurrently once the fast path engages -- same split as classical.py.
        self._lock = threading.Lock()
        self._fast_fed = False
        self._last_mar: float | None = None
        self._quality_events: deque[tuple[float, bool]] = deque()
        # Defaults to "uncompressed" until a real ctx is sampled (_sample()), so a
        # unit test that feeds the buffer directly and never calls _sample/fast_update
        # keeps exercising the original red/blue + A/B path unchanged.
        self._source_mode = "uncompressed"
        self._mode_seen = False
        self._channel_idx = {
            "uncompressed": (_CHANNEL_INDEX.get(str(self.channels[0]).lower(), 0),
                             _CHANNEL_INDEX.get(str(self.channels[1]).lower(), 2)),
            "compressed": (_CHANNEL_INDEX.get(str(self.compressed_channels[0]).lower(), 0),
                          _CHANNEL_INDEX.get(str(self.compressed_channels[1]).lower(), 1)),
        }
        self._calibration = self._load_calibration()

    def _load_calibration(self) -> dict[str, tuple[float, float, bool]]:
        """Load per-mode A/B from the calibration JSON, degrading to an
        uncalibrated default per mode independently.

        The default (A=100, B=5, calibrated=false) is a literature-ish placeholder,
        not a fitted curve: it lets a mode run and self-label while forcing
        confidence to near zero until tools/spo2_calibrate.py fits real coefficients
        for that mode. The two modes use different channel pairs, so their
        coefficients are never interchangeable -- each is read from its own keys
        (A/B/calibrated for uncompressed, A_rg/B_rg/calibrated_rg for compressed).
        """
        default = (100.0, 5.0, False)
        path = Path(self.calibration_file)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}

        def _coeffs(a_key: str, b_key: str, cal_key: str) -> tuple[float, float, bool]:
            try:
                a_coeff = float(data.get(a_key, default[0]))
                b_coeff = float(data.get(b_key, default[1]))
            except (TypeError, ValueError):
                return default
            return a_coeff, b_coeff, bool(data.get(cal_key, False))

        return {
            "uncompressed": _coeffs("A", "B", "calibrated"),
            "compressed": _coeffs("A_rg", "B_rg", "calibrated_rg"),
        }

    def _sample(self, ctx: FrameContext) -> None:
        """Sample the skin ROIs once and push their (R,G,B) mean into the buffer."""
        device = ctx.extras.get("rppg_samples")
        if device:
            # iPad-supplied raw-pixel ROI colour (see modules/rppg_backends/
            # classical.py). No depth over the iPad link, so this is the
            # compressed channel-pair regime; the device already talk-guarded it.
            self._source_mode = "compressed"
            self._mode_seen = True
            with self._lock:
                for ts, rgb, _n in device:
                    self.buf.push(ts, np.asarray(rgb, dtype=np.float64))
                    self._brightness = 0.9 * self._brightness + 0.1 * float(
                        0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
                    self._quality_events.append((ts, True))
                    self._prune_quality_events(ts)
            return
        mode = "uncompressed" if ctx.depth is not None else "compressed"
        if self._mode_seen and mode != self._source_mode:
            # Source changed mid-run ('c' hotkey, RealSense unplugged): a window
            # spanning two different channel pairs / coefficient bases would
            # silently blend incompatible SpO2 estimates.
            with self._lock:
                self.buf.t.clear()
                self.buf.v.clear()
                self._quality_events.clear()
                self._last_mar = None
        self._source_mode = mode
        self._mode_seen = True
        roi_landmarks = _ROI_LANDMARKS
        mar = mouth_aspect_ratio(ctx)
        if mar is not None:
            if (self._last_mar is not None
                    and abs(mar - self._last_mar) > self.talk_delta_threshold):
                # Talking drags the cheeks; drop them for this sample so non-cardiac
                # motion doesn't corrupt the AC amplitude (same guard as classical.py).
                roi_landmarks = (FL.FOREHEAD_TOP,)
            self._last_mar = mar
        pixels = []
        for idx in roi_landmarks:
            patch = roi_patch(ctx, idx, radius_frac=0.10)
            if patch is not None and patch.size:
                pixels.append(patch.reshape(-1, 3))
        if not pixels:
            with self._lock:
                self._quality_events.append((ctx.timestamp, False))
                self._prune_quality_events(ctx.timestamp)
            return
        px = np.concatenate(pixels, axis=0).astype(np.float64)   # BGR
        rgb_mean = px[:, ::-1].mean(axis=0)                      # store as (R,G,B)
        bright = patch_brightness(px)
        with self._lock:
            self.buf.push(ctx.timestamp, rgb_mean)
            self._brightness = 0.9 * self._brightness + 0.1 * bright
            self._quality_events.append((ctx.timestamp, True))
            self._prune_quality_events(ctx.timestamp)

    def _prune_quality_events(self, now: float) -> None:
        while (self._quality_events
               and now - self._quality_events[0][0] > self.window_seconds):
            self._quality_events.popleft()

    def fast_update(self, ctx: FrameContext) -> None:
        """Feed one raw captured frame from core.pipeline's vitals sampler."""
        self._fast_fed = True
        self._sample(ctx)

    def reset_capture(self) -> None:
        """Discard buffered samples after a showcase framing/quality failure.

        A fresh stable-capture period must not derive SpO₂ from frames gathered
        while the guest was moving or out of position (mirrors heart_rate.py).
        """
        with self._lock:
            self.buf.t.clear()
            self.buf.v.clear()
            self._quality_events.clear()
            self._last_mar = None

    def _snapshot(self):
        """Thread-safe copy of buffer + brightness + rolling acceptance for compute."""
        with self._lock:
            if len(self.buf) < self.required_samples or self.buf.span() < self.required_seconds:
                return None
            t, v = self.buf.arrays()
            span = self.buf.span()
            brightness = self._brightness
            rolling = list(self._quality_events)
        return t, v, span, brightness, rolling

    def _resample_rgb(self, t: np.ndarray, v: np.ndarray, fs: float) -> np.ndarray | None:
        """Uniformly resample the buffered (R,G,B) means to fs Hz."""
        if v.ndim != 2 or v.shape[1] != 3:
            return None
        t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
        if len(t_uniform) < 16:
            return None
        return np.stack([np.interp(t_uniform, t, v[:, c]) for c in range(3)], axis=1)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        if not self._fast_fed:
            # No dedicated fast sampler feeding us (e.g. replay/tests): sample here.
            self._sample(ctx)
        mode = self._source_mode
        if mode == "compressed" and not self.enabled_on_compressed:
            # Unvalidated against a reference oximeter -- silent no-op unless
            # explicitly opted in (config/modules.yaml: enabled_on_compressed).
            return None
        num_idx, den_idx = self._channel_idx[mode]
        a_coeff, b_coeff, calibrated = self._calibration[mode]
        snap = self._snapshot()
        if snap is None:
            return None
        t, v, span, brightness, rolling = snap
        fs = self.resample_hz
        rgb = self._resample_rgb(t, v, fs)
        if rgb is None:
            return None

        # Gate: SpO₂ from a ratio of noise is meaningless -- require a clean pulse.
        pulse = chrom(rgb)
        if pulse is None:
            return None
        filt = bandpass(pulse, fs, self.ac_band[0], self.ac_band[1])
        if filt is None:
            return None
        dom = dominant_frequency(filt, fs, self.ac_band[0], self.ac_band[1])
        if dom is None:
            return None
        _freq, prominence = dom
        if prominence < self.min_pulse_prominence:
            return None

        ratio = ratio_of_ratios(rgb, fs, num_idx, den_idx, self.ac_band)
        if ratio is None:
            return None
        raw = ratio_to_spo2(ratio, a_coeff, b_coeff)
        if not np.isfinite(raw):
            return None
        display = float(np.clip(raw, self.display_min, self.display_max))

        # Confidence: capture quality x pulse sharpness x calibration presence x fill.
        intervals = np.diff(t)
        mean_interval = float(np.mean(intervals)) if len(intervals) else 0.0
        regularity = (max(0.0, 1.0 - float(np.std(intervals)) / mean_interval)
                      if mean_interval > 0 else 0.0)
        acceptance = (sum(accepted for _, accepted in rolling) / len(rolling)
                      if rolling else 0.0)
        effective_hz = ((len(t) - 1) / span if len(t) > 1 and span > 0 else 0.0)
        quality = rppg_input_quality(brightness, effective_hz, acceptance, regularity)
        pulse_q = min(1.0, prominence * 3.0)
        fill = min(1.0, span / self.window_seconds)
        conf = quality * pulse_q * fill * low_light_factor(brightness)
        if not calibrated:
            conf = min(conf, 0.1)          # placeholder coefficients -> never trusted
        conf = round(max(0.0, min(1.0, conf)), 2)

        results = []
        sev = Severity.INFO
        mode_note = ", compressed color" if mode == "compressed" else ""
        msg = f"SpO₂ ~{display:.0f}%"
        if not calibrated:
            msg += f" (uncalibrated{mode_note}, trend only)"
        elif display < self.warn_below:
            # Even calibrated, a low-confidence dip is NOTICE, not WARNING.
            sev = (Severity.WARNING if conf >= self.warn_min_confidence
                   else Severity.NOTICE)
            msg = f"SpO₂ ~{display:.0f}% (low — check on person{mode_note})"
        results.append(self.result(
            "spo2", round(display, 1), conf, sev, msg, ttl=8.0,
            quality=round(quality, 2)))
        if self.debug_ratio:
            results.append(self.result(
                "spo2_ratio", round(float(ratio), 4), round(quality, 2),
                Severity.INFO, "", ttl=8.0))
        return results

    def close(self):
        """Release resources (none held beyond in-memory buffers)."""
        with self._lock:
            self.buf.t.clear()
            self.buf.v.clear()
            self._quality_events.clear()
