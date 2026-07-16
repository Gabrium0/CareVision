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
import threading
import warnings
import queue
from multiprocessing import get_context
from collections import deque
from importlib import metadata
from statistics import median

import numpy as np

from core.context import FrameContext
from core.debug import log as debug_log
from modules._util import patch_brightness
from .base import RPPGBackend


def _limit_cpu_worker(reserved_cores: int) -> None:
    """Leave CPU capacity for MediaPipe; best-effort and Windows-safe."""
    count = max(1, os.cpu_count() or 1)
    worker_cores = max(1, count - max(0, int(reserved_cores)))
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
            kernel32.SetProcessAffinityMask(handle, (1 << worker_cores) - 1)
        elif hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(range(worker_cores)))
    except Exception:  # noqa: BLE001
        pass


def _openrppg_worker(in_q, out_q, model_name: str | None, reserved_cores: int) -> None:
    """Persistent model owner. The parent validates generation-tagged results."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    try:
        import rppg
        import jax
        devices = jax.devices()
        device_desc = ", ".join(str(device) for device in devices)
        gpu = any(getattr(device, "platform", "").lower() == "gpu" for device in devices)
        if not gpu:
            _limit_cpu_worker(reserved_cores)
        started = time.time()
        model = rppg.Model() if not model_name else rppg.Model(model_name)
        out_q.put({"event": "ready", "device": device_desc,
                   "resource_mode": "gpu" if gpu else "cpu_limited",
                   "load_latency_ms": (time.time() - started) * 1000.0,
                   "pid": os.getpid()})
    except Exception as exc:  # noqa: BLE001
        out_q.put({"event": "error", "error": f"load failed: {type(exc).__name__}: {exc}"})
        return
    while True:
        item = in_q.get()
        if item is None:
            return
        job_id, generation, tensor_rgb, fps, span = item
        started = time.perf_counter()
        try:
            with model:
                ts = 0.0
                for frame in tensor_rgb:
                    model.update_face(frame, ts)
                    ts += 1.0 / fps
            res = model.hr(return_hrv=True)
            bvp, bts = model.bvp()
            out_q.put({"event": "result", "job_id": job_id,
                       "generation": generation, "span": span, "fps": fps,
                       "res": res, "bvp": np.asarray(bvp), "bts": bts,
                       "latency_ms": (time.perf_counter() - started) * 1000.0})
        except Exception as exc:  # noqa: BLE001
            out_q.put({"event": "error", "job_id": job_id,
                       "generation": generation,
                       "error": f"inference failed: {type(exc).__name__}: {exc}"})


class OpenRPPGBackend(RPPGBackend):
    """Neural rPPG backend using the open-rppg toolbox (HR/HRV/breathing)."""
    label = "open-rppg"
    _diagnostics_logged = False

    def __init__(self, window_seconds: float = 12.0, model: str | None = None,
                 infer_every: float = 5.0, min_seconds: float = 10.0,
                 hrv_min_seconds: float = 30.0, min_confidence: float = 0.35,
                 smoothing_window: int = 5, motion_threshold: float | None = 35.0,
                 face_jitter_threshold: float | None = 0.25,
                 face_reacquire_frames: int = 3,
                 inference_timeout_seconds: float = 90.0,
                 cpu_reserved_cores: int = 2,
                 async_inference: bool = True,
                 brightness_normalize: bool = True,
                 brightness_target: float = 110.0):
        self.window_seconds = window_seconds
        self.brightness_normalize = brightness_normalize
        self.brightness_target = brightness_target
        self.infer_every = infer_every
        self.min_seconds = min_seconds
        self.hrv_min_seconds = hrv_min_seconds
        self.min_confidence = min_confidence
        self.smoothing_window = max(1, int(smoothing_window))
        self.motion_threshold = motion_threshold
        self.face_jitter_threshold = face_jitter_threshold
        self.face_reacquire_frames = max(1, int(face_reacquire_frames))
        self.inference_timeout_seconds = max(10.0, float(inference_timeout_seconds))
        self.cpu_reserved_cores = max(0, int(cpu_reserved_cores))
        self.async_inference = async_inference
        self.model_name = model or "default"
        self.label = "open-rppg" if model is None else f"open-rppg-{self.model_name}"
        self.available = False
        self._cv2 = None
        self._get_prv = None
        # update() (writer) and compute() (reader) can run on different
        # threads when fed via the camera's fast path (see core/pipeline.py).
        self._lock = threading.Lock()
        self.ts: deque[float] = deque()
        self.crops: deque[np.ndarray] = deque()
        self._bpm_history: deque[float] = deque(maxlen=self.smoothing_window)
        self._last_bbox: tuple | None = None
        self._candidate_bbox: tuple | None = None
        self._candidate_count = 0
        self._last_infer = 0.0
        self._cached: dict | None = None
        self._status = "starting"
        self._low_light = False
        self._accepted = 0
        self._reject_motion = 0
        self._reject_jitter = 0
        self._reject_no_face = 0
        self._last_latency_ms = 0.0
        self._ctx = get_context("spawn")
        self._in_q = None
        self._out_q = None
        self._proc = None
        self._worker_ready = False
        self._worker_state = "starting"
        self._worker_pid = None
        self._worker_device = None
        self._resource_mode = None
        self._load_latency_ms = 0.0
        self._job_id = 0
        self._active_job_id: int | None = None
        self._job_generation: int | None = None
        self._job_started_at = 0.0
        self._generation = 0
        self._restart_count = 0
        self._stale_discard_count = 0

        try:
            import cv2
            self._cv2 = cv2
            self._log_runtime()
            self.available = True
            self._start_worker()
        except Exception as e:  # noqa: BLE001
            print(f"[open-rppg] unavailable ({type(e).__name__}: {e}); "
                  "falling back to classical only")

    def _start_worker(self) -> None:
        self._in_q = self._ctx.Queue(maxsize=1)
        self._out_q = self._ctx.Queue()
        model = None if self.model_name == "default" else self.model_name
        self._proc = self._ctx.Process(
            target=_openrppg_worker,
            args=(self._in_q, self._out_q, model, self.cpu_reserved_cores),
            daemon=True)
        self._proc.start()
        self._worker_ready = False
        self._worker_state = "loading model"
        self._worker_pid = self._proc.pid

    def _stop_worker(self) -> None:
        proc = self._proc
        if proc is not None and proc.is_alive():
            try:
                self._in_q.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        for q in (self._in_q, self._out_q):
            try:
                q.close()
                q.cancel_join_thread()
            except Exception:  # noqa: BLE001
                pass
        self._proc = self._in_q = self._out_q = None
        self._worker_ready = False

    def _restart_worker(self, reason: str) -> None:
        self._stop_worker()
        self._restart_count += 1
        self._job_generation = None
        self._worker_state = reason
        self._start_worker()

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
        if self.brightness_normalize:
            face128 = self._boost_brightness(face128)
        with self._lock:
            self.ts.append(ctx.timestamp)
            self.crops.append(face128)
            self._accepted += 1
            while self.ts and ctx.timestamp - self.ts[0] > self.window_seconds:
                self.ts.popleft()
                self.crops.popleft()

    def diagnostics(self) -> dict:
        """Return a thread-safe snapshot without exposing face crops."""
        self._drain_worker()
        with self._lock:
            samples = len(self.ts)
            span = (self.ts[-1] - self.ts[0]) if samples > 1 else 0.0
            latest = dict(self._cached) if self._cached else None
            accepted = self._accepted
            rejected = {
                "motion": self._reject_motion,
                "jitter": self._reject_jitter,
                "no_face": self._reject_no_face,
            }
            status = self._status
            low_light = self._low_light
            latency = self._last_latency_ms
            pending = (self._job_generation is not None and
                       self._job_generation == self._generation)
            stale_pending = (self._job_generation is not None and
                             self._job_generation != self._generation)
        required_samples = 16
        sample_progress = min(1.0, samples / required_samples)
        time_progress = min(1.0, span / max(self.min_seconds, 1e-6))
        ready = samples >= required_samples and span >= self.min_seconds
        sample_hz = ((samples - 1) / span if samples > 1 and span > 0 else 0.0)
        # Derive the visible state from the same sample/generation snapshot.
        # This prevents a reset from leaving the global card on "inferring"
        # while the backend card correctly reports an empty warm-up buffer.
        if not ready:
            status = (f"warming up: {samples}/{required_samples} samples, "
                      f"{span:.0f}/{self.min_seconds:.0f}s")
            if low_light:
                status += " (low light)"
        elif pending:
            status = "inferring"
        elif stale_pending:
            status = "discarding stale inference"
        elif not self._worker_ready:
            status = self._worker_state
        return {
            "name": self.label, "available": bool(self.available),
            "samples": samples, "buffered_seconds": round(span, 1),
            "required_seconds": float(self.min_seconds),
            "required_samples": required_samples,
            "sample_progress": round(sample_progress, 3),
            "effective_sample_hz": round(sample_hz, 2),
            "progress": round(min(time_progress, sample_progress), 3),
            "ready": ready,
            "status": status, "accepted": accepted, "rejected": rejected,
            "inference_latency_ms": round(float(latency), 1),
            "inference_pending": pending, "low_light": low_light,
            "stale_inference_pending": stale_pending,
            "generation": self._generation,
            "job_generation": self._job_generation,
            "worker_state": self._worker_state,
            "worker_pid": self._worker_pid,
            "worker_device": self._worker_device,
            "resource_mode": self._resource_mode,
            "worker_load_latency_ms": round(self._load_latency_ms, 1),
            "worker_restarts": self._restart_count,
            "stale_results_discarded": self._stale_discard_count,
            "latest": latest,
        }

    def reset(self) -> None:
        """Discard samples and any pending result when capture becomes unsafe."""
        with self._lock:
            self.ts.clear()
            self.crops.clear()
            self._cached = None
            self._status = "waiting for stable capture"
            self._low_light = False
            self._last_bbox = None
            self._candidate_bbox = None
            self._candidate_count = 0
            self._bpm_history.clear()
            self._generation += 1

    def _bbox_close(self, bbox: tuple, reference: tuple) -> bool:
        """Whether two boxes describe the same stable face position."""
        if self.face_jitter_threshold is None:
            return True
        x1, y1, x2, y2 = bbox
        lx1, ly1, lx2, ly2 = reference
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        lw, lh = max(1, lx2 - lx1), max(1, ly2 - ly1)
        center_delta = (((x1 + x2 - lx1 - lx2) / 2.0) ** 2 +
                        ((y1 + y2 - ly1 - ly2) / 2.0) ** 2) ** 0.5
        scale_delta = max(abs(w - lw) / lw, abs(h - lh) / lh)
        norm_delta = center_delta / max(w, h, 1)
        return (norm_delta <= self.face_jitter_threshold and
                scale_delta <= self.face_jitter_threshold)

    def _stable_face(self, bbox: tuple) -> bool:
        if self.face_jitter_threshold is None or self._last_bbox is None:
            self._last_bbox = bbox
            self._candidate_bbox = None
            self._candidate_count = 0
            return True
        if self._bbox_close(bbox, self._last_bbox):
            self._last_bbox = bbox
            self._candidate_bbox = None
            self._candidate_count = 0
            return True

        # A single jump remains rejected, but a new box that stays stable for
        # several captured frames becomes the new reference.  Without this
        # candidate path one detector jump permanently anchors the backend to
        # an obsolete box and every later frame is rejected.
        if self._candidate_bbox is None or not self._bbox_close(bbox, self._candidate_bbox):
            self._candidate_bbox = bbox
            self._candidate_count = 1
            return False
        self._candidate_bbox = bbox
        self._candidate_count += 1
        if self._candidate_count < self.face_reacquire_frames:
            return False
        self._last_bbox = bbox
        self._candidate_bbox = None
        self._candidate_count = 0
        return True

    def _boost_brightness(self, crop: np.ndarray) -> np.ndarray:
        """Lift a dark face crop toward the model's expected brightness.

        Open-RPPG was trained on reasonably-lit faces; a dark crop is out of
        distribution and yields low SQI. A conservative, highlight-safe gain
        toward `brightness_target` is a distribution fix (not an SNR fix), so
        unlike the classical FFT path it can genuinely raise the neural SQI.
        """
        bright = patch_brightness(crop)
        self._low_light = bright < 25.0
        if bright < 1.0 or bright >= self.brightness_target:
            return crop
        gain = min(self.brightness_target / bright, 3.0)   # cap avoids noise blow-up
        if gain <= 1.02:
            return crop
        return self._cv2.convertScaleAbs(crop, alpha=gain, beta=0.0)

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
        self._drain_worker()
        now = time.time()
        with self._lock:
            n = len(self.ts)
            span = (self.ts[-1] - self.ts[0]) if n > 1 else 0.0
        debug_log("openrppg", f"buffered={n} span={span:.1f}s accepted={self._accepted} "
                              f"reject_motion={self._reject_motion} reject_jitter={self._reject_jitter} "
                              f"reject_no_face={self._reject_no_face} pending={self._job_generation is not None} "
                              f"status={self._status} latency_ms={self._last_latency_ms:.0f}")
        if n < 16 or span < self.min_seconds:
            self._status = (f"warming up: {n}/16 samples, "
                            f"{span:.0f}/{self.min_seconds:.0f}s")
            if self._low_light:
                self._status += " (low light)"
            return self._cached

        if self._job_generation is not None:
            self._status = ("inferring" if self._job_generation == self._generation
                            else "waiting for stale inference to finish")
            return self._cached
        if not self._worker_ready:
            self._status = self._worker_state
            return self._cached

        if now - self._last_infer < self.infer_every:
            return self._cached
        self._last_infer = now
        self._status = "inferring"

        # Re-snapshot right before building the inference tensor for the
        # freshest window; copying under the lock keeps the (slow) tensor
        # stack/inference off the writer thread's critical section.
        with self._lock:
            n = len(self.ts)
            span = (self.ts[-1] - self.ts[0]) if n > 1 else 0.0
            tensor_bgr = np.stack(list(self.crops))
        fps = n / max(span, 1e-6)
        tensor_rgb = np.ascontiguousarray(tensor_bgr[..., ::-1], dtype=np.uint8)
        if not self.async_inference:
            self._cached = self._compute_tensor(tensor_rgb, float(fps), span)
            return self._cached
        self._job_id += 1
        self._active_job_id = self._job_id
        self._job_generation = self._generation
        self._job_started_at = now
        try:
            self._in_q.put_nowait((self._job_id, self._generation, tensor_rgb,
                                   float(fps), span))
        except queue.Full:
            self._active_job_id = None
            self._job_generation = None
            self._status = "worker queue busy"
        return self._cached

    def _drain_worker(self) -> None:
        if self._proc is None:
            return
        # Drain terminal messages before inspecting process liveness. A worker
        # that reports a model-load error exits immediately; checking liveness
        # first would discard that useful error and restart forever.
        while True:
            try:
                msg = self._out_q.get_nowait()
            except queue.Empty:
                break
            event = msg.get("event")
            if event == "ready":
                self._worker_ready = True
                self._worker_state = "ready"
                self._worker_pid = msg.get("pid")
                self._worker_device = msg.get("device")
                self._resource_mode = msg.get("resource_mode")
                self._load_latency_ms = float(msg.get("load_latency_ms") or 0.0)
                print(f"[open-rppg] worker ready ({self._resource_mode}, {self._worker_device})")
            elif event == "result":
                if msg.get("job_id") != self._active_job_id:
                    continue
                self._last_latency_ms = float(msg.get("latency_ms") or 0.0)
                generation = int(msg.get("generation", -1))
                if generation == self._generation:
                    self._cached = self._build_result(
                        msg.get("res"), np.asarray(msg.get("bvp")), msg.get("bts"),
                        float(msg.get("fps")), float(msg.get("span")))
                else:
                    self._stale_discard_count += 1
                self._active_job_id = None
                self._job_generation = None
            elif event == "error":
                error = str(msg.get("error") or "worker error")
                if msg.get("job_id") == self._active_job_id:
                    self._active_job_id = None
                    self._job_generation = None
                self._status = error
                if msg.get("job_id") is None:
                    self._worker_state = "unavailable"
                    self.available = False
                print(f"[open-rppg] {error}")
        if (self._job_generation is not None and
                time.time() - self._job_started_at > self.inference_timeout_seconds):
            self._restart_worker("inference timed out; restarting")
            return
        if not self._proc.is_alive() and self._worker_state != "unavailable":
            self._restart_worker("worker exited; restarting")

    def _compute_tensor(self, tensor_rgb: np.ndarray, fps: float, span: float) -> dict | None:
        t0 = time.perf_counter()
        try:
            res, bvp, bts = self._infer(tensor_rgb, fps)
        except Exception as e:  # noqa: BLE001
            print(f"[open-rppg] inference failed: {e}")
            with self._lock:
                self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
                self._status = f"inference failed: {type(e).__name__}"
            return self._cached
        with self._lock:
            self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_result(res, bvp, bts, fps, span)

    def _build_result(self, res, bvp, bts, fps: float, span: float) -> dict | None:
        if not res or res.get("hr") is None or not np.isfinite(res["hr"]):
            with self._lock:
                self._status = "inference returned no valid heart rate"
            return self._cached

        sqi = float(res.get("SQI") or 0.0)
        raw_bpm = float(res["hr"])
        self._bpm_history.append(raw_bpm)
        bpm = float(median(self._bpm_history))
        if len(self._bpm_history) >= 3:
            jitter = float(np.std(np.asarray(self._bpm_history, dtype=np.float64)))
            # Penalize unstable recent HR estimates without fully discarding a usable SQI reading.
            sqi *= max(0.5, 1.0 - jitter / 20.0)
        confidence = round(max(0.0, min(1.0, sqi)), 2)
        out = {"raw_bpm": round(bpm, 1), "raw_confidence": confidence}
        hrv_required = self.hrv_min_seconds * 0.95
        out["status"] = ("ready" if span >= hrv_required
                          else f"waiting for clean HRV window {span:.0f}/{self.hrv_min_seconds:.0f}s")
        if confidence < self.min_confidence:
            out["rejected_reason"] = f"low SQI {confidence:.2f}<{self.min_confidence:.2f}"
            out["status"] = out["rejected_reason"]
            with self._lock:
                self._status = out["status"]
            return out

        out["bpm"] = round(bpm, 1)
        out["confidence"] = confidence

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

        with self._lock:
            self._status = out["status"]
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
        print("[open-rppg] JAX device inspection deferred to isolated worker")

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self._stop_worker()


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
