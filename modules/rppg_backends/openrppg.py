"""Neural rPPG backend using the open-rppg toolbox (KegangWangCCNU/open-rppg).

Reports the full set open-rppg can produce: heart rate, HRV (SDNN + RMSSD),
and breathing rate.

Design notes (from reading open-rppg's source + measured behaviour here):
- We keep our own rolling buffer of 128x128 face crops (cropped upstream by
  MediaPipe) and, on a throttle, run one inference pass over the window.
- The pass mirrors `Model.process_faces_tensor`: feed frames via update_face
  INSIDE a `with model:` block, then read `hr()` and `bvp()` AFTER the block
  exits (the worker only flushes on context exit — reading inside returns
  None). This one pass yields both the HR/SQI dict and the BVP waveform.
- open-rppg expects RGB frames; our crops are BGR (OpenCV), so we convert.
- HRV: open-rppg only fills its own hrv dict when SQI > 0.5. To surface HRV
  more readily we also run open-rppg's own `get_prv()` (HeartPy-based) on the
  neural BVP directly. Keys are HeartPy's: sdnn, rmssd (ms), breathingrate
  (Hz -> x60 for breaths/min).

Returns bpm + SQI (confidence) always; HRV/breathing whenever the BVP has
detectable beats. Degrades to unavailable if `rppg`/jax can't load.
"""
from __future__ import annotations

import time
import os
import sys
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from importlib import metadata
from statistics import median

import numpy as np

from core.context import FrameContext
from core.debug import log as debug_log
from .base import RPPGBackend


class OpenRPPGBackend(RPPGBackend):
    """Neural rPPG backend using the open-rppg toolbox (HR/HRV/breathing)."""
    label = "open-rppg"
    _diagnostics_logged = False

    def __init__(self, window_seconds: float = 12.0, model: str | None = None,
                 infer_every: float = 5.0, min_seconds: float = 10.0,
                 hrv_min_seconds: float = 30.0, min_confidence: float = 0.35,
                 smoothing_window: int = 5, motion_threshold: float | None = 35.0,
                 face_jitter_threshold: float | None = 0.25,
                 async_inference: bool = True):
        self.window_seconds = window_seconds
        self.infer_every = infer_every
        self.min_seconds = min_seconds
        self.hrv_min_seconds = hrv_min_seconds
        self.min_confidence = min_confidence
        self.smoothing_window = max(1, int(smoothing_window))
        self.motion_threshold = motion_threshold
        self.face_jitter_threshold = face_jitter_threshold
        self.async_inference = async_inference
        self.model_name = model or "default"
        self.label = "open-rppg" if model is None else f"open-rppg-{self.model_name}"
        self.available = False
        self.model = None
        self._cv2 = None
        self._get_prv = None
        self.ts: deque[float] = deque()
        self.crops: deque[np.ndarray] = deque()
        self._bpm_history: deque[float] = deque(maxlen=self.smoothing_window)
        self._last_bbox: tuple | None = None
        self._last_infer = 0.0
        self._cached: dict | None = None
        self._status = "starting"
        self._accepted = 0
        self._reject_motion = 0
        self._reject_jitter = 0
        self._reject_no_face = 0
        self._last_latency_ms = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="open-rppg")
        self._pending: Future | None = None

        try:
            self._prepare_runtime()
            import cv2
            import rppg
            from rppg.main import get_prv
            self._cv2 = cv2
            self._get_prv = get_prv
            self._log_runtime()
            t0 = time.time()
            self.model = rppg.Model() if not model else rppg.Model(model)
            print(f"[open-rppg] model '{self.model_name}' loaded in {time.time()-t0:.1f}s")
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[open-rppg] unavailable ({type(e).__name__}: {e}); "
                  "falling back to classical only")

    def update(self, ctx: FrameContext) -> None:
        """Feed one frame's data into the backend's rolling state."""
        if not self.available:
            return
        if ctx.face is None:
            self._reject_no_face += 1
            return
        if self.motion_threshold is not None and ctx.motion_energy > self.motion_threshold:
            self._reject_motion += 1
            self._status = f"motion rejected ({ctx.motion_energy:.0f}>{self.motion_threshold:.0f})"
            return
        if not self._stable_face(ctx.face.bbox):
            self._reject_jitter += 1
            self._status = "face jitter rejected"
            return
        crop = ctx.face.crop
        if crop is None or crop.size == 0:
            return
        face128 = self._cv2.resize(crop, (128, 128))   # BGR uint8
        self.ts.append(ctx.timestamp)
        self.crops.append(face128)
        self._accepted += 1
        while self.ts and ctx.timestamp - self.ts[0] > self.window_seconds:
            self.ts.popleft()
            self.crops.popleft()

    def _stable_face(self, bbox: tuple) -> bool:
        if self.face_jitter_threshold is None or self._last_bbox is None:
            self._last_bbox = bbox
            return True
        x1, y1, x2, y2 = bbox
        lx1, ly1, lx2, ly2 = self._last_bbox
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        lw, lh = max(1, lx2 - lx1), max(1, ly2 - ly1)
        center_delta = (((x1 + x2 - lx1 - lx2) / 2.0) ** 2 +
                        ((y1 + y2 - ly1 - ly2) / 2.0) ** 2) ** 0.5
        scale_delta = max(abs(w - lw) / lw, abs(h - lh) / lh)
        norm_delta = center_delta / max(w, h, 1)
        stable = norm_delta <= self.face_jitter_threshold and scale_delta <= self.face_jitter_threshold
        if stable:
            self._last_bbox = bbox
        return stable

    def _infer(self, tensor_rgb: np.ndarray, fps: float):
        """One inference pass -> (hr_dict, bvp_array, bvp_ts). Mirrors
        process_faces_tensor but also returns the BVP waveform."""
        with self.model:
            ts = 0.0
            for i in range(len(tensor_rgb)):
                self.model.update_face(tensor_rgb[i], ts)
                ts += 1.0 / fps
        res = self.model.hr(return_hrv=True)     # read after context exit
        bvp, bts = self.model.bvp()
        return res, np.asarray(bvp), bts

    def compute(self) -> dict | None:
        """Return the backend's current reading dict, or None if not ready."""
        if not self.available:
            return None
        now = time.time()
        span = (self.ts[-1] - self.ts[0]) if len(self.ts) > 1 else 0.0
        debug_log("openrppg", f"buffered={len(self.crops)} span={span:.1f}s accepted={self._accepted} "
                              f"reject_motion={self._reject_motion} reject_jitter={self._reject_jitter} "
                              f"reject_no_face={self._reject_no_face} pending={self._pending is not None} "
                              f"status={self._status} latency_ms={self._last_latency_ms:.0f}")
        if len(self.crops) < 16 or span < self.min_seconds:
            self._status = f"warming up {span:.0f}/{self.min_seconds:.0f}s"
            return self._cached

        if self._pending is not None:
            if not self._pending.done():
                self._status = "inferring"
                return self._cached
            try:
                self._cached = self._pending.result()
            except Exception as e:  # noqa: BLE001
                print(f"[open-rppg] inference failed: {e}")
            finally:
                self._pending = None

        if now - self._last_infer < self.infer_every:
            return self._cached
        self._last_infer = now
        self._status = "inferring"

        fps = len(self.ts) / max(span, 1e-6)
        tensor_bgr = np.stack(list(self.crops))
        tensor_rgb = np.ascontiguousarray(tensor_bgr[..., ::-1], dtype=np.uint8)
        if not self.async_inference:
            self._cached = self._compute_tensor(tensor_rgb, float(fps), span)
            return self._cached
        self._pending = self._executor.submit(self._compute_tensor, tensor_rgb, float(fps), span)
        return self._cached

    def _compute_tensor(self, tensor_rgb: np.ndarray, fps: float, span: float) -> dict | None:
        t0 = time.perf_counter()
        try:
            res, bvp, bts = self._infer(tensor_rgb, fps)
        except Exception as e:  # noqa: BLE001
            print(f"[open-rppg] inference failed: {e}")
            return self._cached
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        if not res or res.get("hr") is None or not np.isfinite(res["hr"]):
            return self._cached

        sqi = float(res.get("SQI") or 0.0)
        raw_bpm = float(res["hr"])
        self._bpm_history.append(raw_bpm)
        bpm = float(median(self._bpm_history))
        if len(self._bpm_history) >= 3:
            jitter = float(np.std(np.asarray(self._bpm_history, dtype=np.float64)))
            sqi *= max(0.5, 1.0 - jitter / 20.0)
        confidence = round(max(0.0, min(1.0, sqi)), 2)
        out = {"bpm": round(bpm, 1), "confidence": confidence}
        hrv_required = self.hrv_min_seconds * 0.95
        out["status"] = ("ready" if span >= hrv_required
                         else f"waiting for clean HRV window {span:.0f}/{self.hrv_min_seconds:.0f}s")

        # HRV: prefer open-rppg's gated dict; else run its own get_prv on the
        # neural BVP so SDNN/RMSSD/breathing show whenever beats are detectable.
        hrv = dict(res.get("hrv") or {})
        enough_hrv = span >= hrv_required and len(bvp) >= fps * hrv_required
        if enough_hrv and ("rmssd" not in hrv or "sdnn" not in hrv or "breathingrate" not in hrv):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    hrv = self._get_prv(bvp, bts, float(fps))
            except Exception:  # noqa: BLE001
                pass

        rmssd = _finite(hrv.get("rmssd"))
        sdnn = _finite(hrv.get("sdnn"))
        br_hz = _finite(hrv.get("breathingrate"))
        br_min = br_hz * 60.0 if (br_hz is not None and br_hz > 0) else None

        # Fallback: if HeartPy rejected beats (NaN), derive the missing metrics
        # straight from open-rppg's neural BVP by peak analysis.
        if enough_hrv and (rmssd is None or sdnn is None or br_min is None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fb = _hrv_from_bvp(bvp, float(fps))
            rmssd = rmssd if rmssd is not None else fb.get("rmssd")
            sdnn = sdnn if sdnn is not None else fb.get("sdnn")
            br_min = br_min if br_min is not None else fb.get("breaths_per_min")

        if rmssd is not None:
            out["hrv_rmssd_ms"] = round(rmssd, 1)
        if sdnn is not None:
            out["hrv_sdnn_ms"] = round(sdnn, 1)
        if br_min is not None and br_min > 0:
            out["breaths_per_min"] = round(br_min, 1)

        return out

    @staticmethod
    def _prepare_runtime() -> None:
        os.environ.setdefault("KERAS_BACKEND", "jax")
        imported = [name for name in ("keras", "tensorflow") if name in sys.modules]
        if imported:
            try:
                import keras
                backend = keras.backend.backend()
            except Exception:  # noqa: BLE001
                backend = "unknown"
            if backend != "jax":
                print("[open-rppg] WARNING: " + ", ".join(imported) +
                      f" already imported with Keras backend '{backend}'; "
                      "Open-RPPG may fail unless this is 'jax'")

    def _log_runtime(self) -> None:
        if OpenRPPGBackend._diagnostics_logged:
            return
        OpenRPPGBackend._diagnostics_logged = True
        print(f"[open-rppg] KERAS_BACKEND={os.environ.get('KERAS_BACKEND')}")
        versions = []
        for package in ("open-rppg", "jax", "jaxlib", "keras", "tensorflow"):
            try:
                versions.append(f"{package}={metadata.version(package)}")
            except metadata.PackageNotFoundError:
                versions.append(f"{package}=not-installed")
        print("[open-rppg] deps: " + ", ".join(versions))
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None:
            print(f"[open-rppg] CUDA_VISIBLE_DEVICES={cuda_visible}")
        try:
            import jax
            devices = jax.devices()
            desc = ", ".join(str(d) for d in devices)
            gpu = any(getattr(d, "platform", "").lower() == "gpu" for d in devices)
            print(f"[open-rppg] jax devices: {desc}")
            if not gpu:
                print("[open-rppg] WARNING: no JAX GPU device detected; inference is likely CPU-bound")
        except Exception as e:  # noqa: BLE001
            print(f"[open-rppg] could not inspect JAX devices ({type(e).__name__}: {e})")

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        if self._pending is not None:
            try:
                self._pending.result(timeout=10.0)
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._pending = None
        if self.model is not None:
            try:
                self.model.stop()
            except Exception:  # noqa: BLE001
                pass
        self._executor.shutdown(wait=True, cancel_futures=True)


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _hrv_from_bvp(bvp: np.ndarray, fps: float) -> dict:
    """Derive RMSSD/SDNN (ms) and breathing rate (/min) straight from the
    open-rppg neural BVP waveform, as a fallback when HeartPy returns NaN.
    RMSSD/SDNN come from beat-to-beat intervals; breathing from the slow
    respiratory modulation of the pulse (bandpass 0.1-0.5 Hz)."""
    from modules._util import peak_intervals, bandpass, dominant_frequency

    out: dict = {}
    sig = np.asarray(bvp, dtype=np.float64)
    rr = peak_intervals(sig, fps, min_distance_s=0.4)   # seconds between beats
    if len(rr) >= 4:
        rr_ms = rr * 1000.0
        out["rmssd"] = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        out["sdnn"] = float(np.std(rr_ms))
    filt = bandpass(sig, fps, 0.1, 0.5, order=2)
    if filt is not None:
        dom = dominant_frequency(filt, fps, 0.1, 0.5)
        if dom is not None:
            out["breaths_per_min"] = float(dom[0] * 60.0)
    return out
