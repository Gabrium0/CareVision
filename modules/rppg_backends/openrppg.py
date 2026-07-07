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
from collections import deque

import numpy as np

from core.context import FrameContext
from .base import RPPGBackend


class OpenRPPGBackend(RPPGBackend):
    label = "open-rppg"

    def __init__(self, window_seconds: float = 12.0, model: str | None = None,
                 infer_every: float = 2.0, min_seconds: float = 6.0):
        self.window_seconds = window_seconds
        self.infer_every = infer_every
        self.min_seconds = min_seconds
        self.available = False
        self.model = None
        self._cv2 = None
        self._get_prv = None
        self.ts: deque[float] = deque()
        self.crops: deque[np.ndarray] = deque()
        self._last_infer = 0.0
        self._cached: dict | None = None

        try:
            import cv2
            import rppg
            from rppg.main import get_prv
            self._cv2 = cv2
            self._get_prv = get_prv
            t0 = time.time()
            self.model = rppg.Model() if not model else rppg.Model(model)
            print(f"[open-rppg] model loaded in {time.time()-t0:.1f}s")
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[open-rppg] unavailable ({type(e).__name__}: {e}); "
                  "falling back to classical only")

    def update(self, ctx: FrameContext) -> None:
        if not self.available or ctx.face is None:
            return
        crop = ctx.face.crop
        if crop is None or crop.size == 0:
            return
        face128 = self._cv2.resize(crop, (128, 128))   # BGR uint8
        self.ts.append(ctx.timestamp)
        self.crops.append(face128)
        while self.ts and ctx.timestamp - self.ts[0] > self.window_seconds:
            self.ts.popleft()
            self.crops.popleft()

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
        if not self.available:
            return None
        now = time.time()
        span = (self.ts[-1] - self.ts[0]) if len(self.ts) > 1 else 0.0
        if len(self.crops) < 16 or span < self.min_seconds:
            return self._cached
        if now - self._last_infer < self.infer_every:
            return self._cached
        self._last_infer = now

        fps = len(self.ts) / max(span, 1e-6)
        tensor_bgr = np.stack(list(self.crops))
        tensor_rgb = np.ascontiguousarray(tensor_bgr[..., ::-1], dtype=np.uint8)

        try:
            res, bvp, bts = self._infer(tensor_rgb, float(fps))
        except Exception as e:  # noqa: BLE001
            print(f"[open-rppg] inference failed: {e}")
            return self._cached
        if not res or res.get("hr") is None or not np.isfinite(res["hr"]):
            return self._cached

        sqi = float(res.get("SQI") or 0.0)
        out = {"bpm": round(float(res["hr"]), 1),
               "confidence": round(max(0.0, min(1.0, sqi)), 2)}

        # HRV: prefer open-rppg's gated dict; else run its own get_prv on the
        # neural BVP so SDNN/RMSSD/breathing show whenever beats are detectable.
        hrv = dict(res.get("hrv") or {})
        if ("rmssd" not in hrv or "sdnn" not in hrv) and len(bvp) > fps * 5:
            try:
                hrv = self._get_prv(bvp, bts, float(fps))
            except Exception:  # noqa: BLE001
                pass

        rmssd = _finite(hrv.get("rmssd"))
        sdnn = _finite(hrv.get("sdnn"))
        br_hz = _finite(hrv.get("breathingrate"))
        br_min = br_hz * 60.0 if (br_hz is not None and br_hz > 0) else None

        # Fallback: if HeartPy rejected beats (NaN), derive the missing metrics
        # straight from open-rppg's neural BVP by peak analysis.
        if (rmssd is None or sdnn is None or br_min is None) and len(bvp) > fps * 5:
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

        self._cached = out
        return out

    def close(self) -> None:
        if self.model is not None:
            try:
                self.model.stop()
            except Exception:  # noqa: BLE001
                pass


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
