"""Upper-body clothing detection for weather-aware recommendations.

Recognizes the garment on a pose-anchored torso crop and reasons explicitly
about sleeve length, so it works at normal desk distance (no standing back) and
tells a t-shirt from a long-sleeve shirt.

Design notes:
- Crop is anchored on the SHOULDERS and scaled by shoulder WIDTH (a distance-
  invariant reference), extended to include the upper arms, and needs NO hips —
  so it works up close (hips off-frame) as well as far. Falls back to a
  face-derived crop if pose shoulders aren't available.
- Recognizer defaults to FashionCLIP zero-shot image CLASSIFICATION
  (patrickjohncyh/fashion-clip) — fast (sub-second) and garment-aware. Falls
  back to general CLIP, then to OWLv2 detection (config `recognizer`).
- A pose-guided sleeve check compares the shoulder->elbow skin colour to the
  face skin tone (bare arm => short sleeve) and biases the final label. It is a
  heuristic, surfaced as a bias signal, not ground truth.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import cv2
import numpy as np

from core.context import FrameContext
from core.debug import log as debug_log
from core.events import Severity
from core.registry import register
from core.shared_signals import SharedSignals
from core.one_flight import DaemonOneFlight
from core.runtime_resources import apply_loaded_limits
from core.capabilities import CapabilityRegistry, CapabilityStatus
from modules.base import DetectionModule
from modules._util import roi_patch
from extractors import pose as P
from extractors import face_landmarks as FL


_DEFAULT_LABELS = [
    "tank top", "t-shirt", "long sleeve shirt", "hoodie",
    "sweater", "jacket", "coat", "raincoat",
]

_WARMTH = {
    "tank top": 0, "t-shirt": 1, "long sleeve shirt": 2, "hoodie": 3,
    "sweater": 3, "jacket": 4, "raincoat": 4, "coat": 5,
}

_SHORT_SLEEVE = {"tank top", "t-shirt"}
_LONG_SLEEVE = {"long sleeve shirt", "hoodie", "sweater", "jacket", "coat", "raincoat"}

_PROMPT = "a photo of a person wearing a {}"
_MODELS = {
    "clip": "patrickjohncyh/fashion-clip",
    "clip_fallback": "openai/clip-vit-base-patch32",
    "owlvit": "google/owlv2-base-patch16-ensemble",
    "owlvit_fallback": "google/owlvit-base-patch32",
}
_MIN_CROP_PX = 160          # upscale crops smaller than this on the short side


@register("clothing")
class Clothing(DetectionModule):
    """Upper-body clothing detection for weather-aware recommendations."""
    interval = 0.0
    requires = ("face",)
    backend = "owlvit"              # kept for back-compat; "none" disables
    recognizer = "clip"            # clip | owlvit
    model = None                    # override auto model choice if set
    labels = _DEFAULT_LABELS
    pipeline_threshold = 0.02       # owlvit detection threshold
    min_confidence = 0.15           # gate on the top (softmax/detection) score
    infer_every = 8.0
    load_timeout_seconds = 90.0     # report a stalled download/load honestly
    download_if_missing = True      # cache-first; only the primary may download

    def __init__(self, **params):
        super().__init__(**params)
        self._pipe = None
        self._task = None            # "zero-shot-image-classification" | "...-object-detection"
        self._prompts = [_PROMPT.format(l) for l in self.labels]
        self._prompt_to_label = {p: l for p, l in zip(self._prompts, self.labels)}
        self._last_infer = 0.0
        self._cached = {"clothing": "...", "confidence": 0.0, "status": "warming up"}
        self._load_attempted = False
        self._last_top = []
        self._last_sleeve = None
        self._last_latency_ms = 0.0
        self._lock = threading.Lock()
        self._executor = DaemonOneFlight("clothing-model")
        self._load_future: Future | None = None
        self._load_started: float | None = None
        self._inference_future: Future | None = None
        self._stopping = threading.Event()
        self._shared_signals = SharedSignals.instance()
        self._device = "uninitialized"
        self._model_name = None
        self._load_latency_ms = 0.0
        self._load_deadline_exceeded = False

    # ---- model loading ----------------------------------------------------
    @staticmethod
    def _exception_detail(exc: BaseException) -> str:
        """Return the deepest useful cause without leaking a full traceback."""
        deepest = exc
        seen: set[int] = set()
        while id(deepest) not in seen:
            seen.add(id(deepest))
            cause = deepest.__cause__ or deepest.__context__
            if cause is None:
                break
            deepest = cause
        message = str(deepest).strip()
        if "torchvision::nms" in message and "does not exist" in message:
            return "torchvision::nms missing"
        detail = f"{type(deepest).__name__}: {message}" if message else type(deepest).__name__
        return detail[:160]

    def _candidates(self) -> list[tuple[str, str]]:
        """Return the configured primary followed by cache-only fallbacks."""
        if self.recognizer == "owlvit":
            return [("zero-shot-object-detection", self.model or _MODELS["owlvit"]),
                    ("zero-shot-object-detection", _MODELS["owlvit_fallback"])]
        return [("zero-shot-image-classification", self.model or _MODELS["clip"]),
                ("zero-shot-image-classification", _MODELS["clip_fallback"]),
                ("zero-shot-object-detection", _MODELS["owlvit_fallback"])]

    def _set_loaded(self, pipe, task: str, name: str) -> bool:
        """Publish a loaded model unless shutdown began while it was loading."""
        if self._stopping.is_set():
            return False
        with self._lock:
            self._pipe = pipe
            self._task = task
            self._model_name = name
            self._cached["status"] = "clothing model ready; waiting for first frame"
        print(f"[clothing] loaded {task} model {name}")
        CapabilityRegistry.instance().set(
            self.name, "model", CapabilityStatus.READY,
            f"{task} model loaded on {self._device}")
        return True

    def _load(self) -> None:
        """Load cached candidates first, then optionally download only primary."""
        started = time.perf_counter()
        try:
            from transformers import pipeline
            import torch
            apply_loaded_limits("maximum")
            device = 0 if torch.cuda.is_available() else -1
            self._device = "cuda:0" if device == 0 else "cpu"
        except Exception as exc:  # noqa: BLE001
            detail = self._exception_detail(exc)
            self._load_latency_ms = (time.perf_counter() - started) * 1000.0
            print(f"[clothing] dependencies unavailable ({detail})")
            with self._lock:
                self._cached["status"] = f"clothing dependencies unavailable: {detail}"
            CapabilityRegistry.instance().set(
                self.name, "model", CapabilityStatus.UNCONFIGURED, detail)
            return
        candidates = self._candidates()
        last_detail = "unknown error"
        for task, name in candidates:
            if self._stopping.is_set():
                return
            try:
                pipe = pipeline(task, model=name, local_files_only=True, device=device)
                self._set_loaded(pipe, task, name)
                self._load_latency_ms = (time.perf_counter() - started) * 1000.0
                return
            except Exception as exc:  # noqa: BLE001
                last_detail = self._exception_detail(exc)
                print(f"[clothing] cached model {name} unavailable ({last_detail})")
        if self.download_if_missing and not self._stopping.is_set():
            task, name = candidates[0]
            with self._lock:
                self._cached["status"] = "downloading clothing model"
            try:
                pipe = pipeline(task, model=name, device=device)
                self._set_loaded(pipe, task, name)
                self._load_latency_ms = (time.perf_counter() - started) * 1000.0
                return
            except Exception as exc:  # noqa: BLE001
                last_detail = self._exception_detail(exc)
                print(f"[clothing] could not download/load {name} ({last_detail})")
        if self._stopping.is_set():
            return
        with self._lock:
            self._cached["status"] = f"clothing model unavailable: {last_detail}"
        CapabilityRegistry.instance().set(
            self.name, "model", CapabilityStatus.FAILED, last_detail)
        self._load_latency_ms = (time.perf_counter() - started) * 1000.0

    def start(self) -> None:
        """Begin model preload without waiting for a scheduled camera frame."""
        with self._lock:
            if (self.backend == "none" or self._stopping.is_set()
                    or self._load_attempted):
                return
            self._load_attempted = True
            self._cached["status"] = "loading clothing model"
            CapabilityRegistry.instance().set(
                self.name, "model", CapabilityStatus.LOADING, "loading model")
            self._load_started = time.monotonic()
            self._load_future = self._executor.submit(self._load)

    def _refresh_load_state(self) -> None:
        future = self._load_future
        if future is None:
            return
        if future.done():
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                detail = self._exception_detail(exc)
                with self._lock:
                    self._cached["status"] = f"clothing loader failed: {detail}"
            finally:
                self._load_future = None
                self._load_started = None
        elif (self._load_started is not None
              and time.monotonic() - self._load_started >= self.load_timeout_seconds):
            # Do not discard the future: a late successful load may recover.
            with self._lock:
                if self._pipe is None:
                    self._cached["status"] = "clothing model load timed out"
                    if not self._load_deadline_exceeded:
                        self._load_deadline_exceeded = True
                        CapabilityRegistry.instance().set(
                            self.name, "model", CapabilityStatus.DEGRADED,
                            "model load exceeded deadline; late recovery remains possible")

    # ---- geometry ---------------------------------------------------------
    def _torso_crop(self, ctx: FrameContext) -> np.ndarray | None:
        """Shoulder-anchored, shoulder-width-scaled torso crop. No hips needed;
        includes the upper arms so sleeves are visible. Falls back to a
        face-derived crop when pose shoulders are unavailable."""
        box = None
        if ctx.pose is not None:
            lm = ctx.pose.landmarks
            ls, rs = lm[P.L_SHOULDER], lm[P.R_SHOULDER]
            if ls[3] > 0.4 and rs[3] > 0.4:
                lx, ly = ls[0] * ctx.w, ls[1] * ctx.h
                rx, ry = rs[0] * ctx.w, rs[1] * ctx.h
                sw = abs(lx - rx)
                if sw < 8:                     # shoulders too collapsed to trust
                    sw = 0.25 * ctx.w
                sh_y = (ly + ry) / 2.0
                # Scale the crop from shoulder width so it stays stable at different camera distances.
                x1 = min(lx, rx) - 0.6 * sw
                x2 = max(lx, rx) + 0.6 * sw
                y1 = sh_y - 0.5 * sw           # up to collar / neck
                y2 = sh_y + 2.4 * sw           # down the torso
                # extend to include elbows when visible (captures sleeves)
                for ei in (P.L_ELBOW, P.R_ELBOW):
                    if lm[ei, 3] > 0.5:
                        ex, ey = lm[ei, 0] * ctx.w, lm[ei, 1] * ctx.h
                        x1, x2 = min(x1, ex - 0.2 * sw), max(x2, ex + 0.2 * sw)
                        y2 = max(y2, ey)
                box = (x1, y1, x2, y2)
        if box is None and ctx.face is not None:      # fall back to below-face
            fx1, fy1, fx2, fy2 = ctx.face.bbox
            fw, fh = fx2 - fx1, fy2 - fy1
            cx = (fx1 + fx2) // 2
            box = (cx - 1.6 * fw, fy2 - 0.2 * fh, cx + 1.6 * fw, fy2 + 3.0 * fh)
        if box is None:
            return None
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(ctx.w, int(box[2])); y2 = min(ctx.h, int(box[3]))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        crop = ctx.frame[y1:y2, x1:x2]
        short = min(crop.shape[:2])
        if short < _MIN_CROP_PX:
            scale = _MIN_CROP_PX / short
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def _resize_crop(crop: np.ndarray, max_dim: int = 336) -> np.ndarray:
        h, w = crop.shape[:2]
        m = max(h, w)
        if m <= max_dim:
            return crop
        s = max_dim / m
        return cv2.resize(crop, (int(w * s), int(h * s)),
                          interpolation=cv2.INTER_AREA)

    # ---- sleeve heuristic -------------------------------------------------
    @staticmethod
    def _chroma(bgr: np.ndarray) -> np.ndarray:
        px = bgr.reshape(-1, 3).astype(np.float32)
        b, g, r = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
        s = r + g + b + 1e-6
        return np.array([r / s, g / s, b / s])

    def _arm_patch(self, ctx: FrameContext, sh_i: int, el_i: int) -> np.ndarray | None:
        lm = ctx.pose.landmarks
        if lm[sh_i, 3] < 0.5 or lm[el_i, 3] < 0.5:
            return None
        sx, sy = lm[sh_i, 0] * ctx.w, lm[sh_i, 1] * ctx.h
        ex, ey = lm[el_i, 0] * ctx.w, lm[el_i, 1] * ctx.h
        # sample mid upper-arm (55% shoulder->elbow)
        px_, py_ = int(sx + 0.55 * (ex - sx)), int(sy + 0.55 * (ey - sy))
        r = max(4, int(0.08 * abs(lm[P.L_SHOULDER, 0] - lm[P.R_SHOULDER, 0]) * ctx.w))
        x1, y1 = max(0, px_ - r), max(0, py_ - r)
        x2, y2 = min(ctx.w, px_ + r), min(ctx.h, py_ + r)
        patch = ctx.frame[y1:y2, x1:x2]
        return patch if patch.size else None

    def _sleeve_state(self, ctx: FrameContext) -> str | None:
        """"bare" (short sleeve) | "covered" (long sleeve) | None if unknown.
        Compares upper-arm colour to the face skin tone."""
        if ctx.pose is None or ctx.face is None:
            return None
        face = roi_patch(ctx, FL.LEFT_CHEEK, radius_frac=0.06)
        if face is None or face.size == 0:
            return None
        skin = self._chroma(face)
        dists = []
        for sh_i, el_i in ((P.L_SHOULDER, P.L_ELBOW), (P.R_SHOULDER, P.R_ELBOW)):
            arm = self._arm_patch(ctx, sh_i, el_i)
            if arm is not None:
                dists.append(float(np.linalg.norm(self._chroma(arm) - skin)))
        if not dists:
            return None
        # Use the arm closest to face chroma; one bare arm is enough evidence for short sleeves.
        return "bare" if min(dists) < 0.05 else "covered"

    # ---- inference --------------------------------------------------------
    def _classify(self, image) -> list[tuple[str, float]]:
        if self._task == "zero-shot-image-classification":
            preds = self._pipe(image, candidate_labels=list(self.labels),
                               hypothesis_template=_PROMPT)
            return [(str(p["label"]).lower(), float(p["score"])) for p in preds]
        # OWLv2 detection path
        preds = self._pipe(image, candidate_labels=self._prompts,
                           threshold=self.pipeline_threshold)
        out = []
        for p in preds:
            pl = str(p.get("label", "")).lower()
            base = self._prompt_to_label.get(p.get("label"),
                                             next((b for b in self.labels if b in pl), pl))
            out.append((base, float(p.get("score", 0.0))))
        return sorted(out, key=lambda t: t[1], reverse=True)

    @staticmethod
    def _fuse_sleeve(ranked: list[tuple[str, float]], sleeve: str | None):
        if not sleeve:
            return ranked
        adj = []
        for label, score in ranked:
            # Sleeve evidence is only a bias; the model score still decides after re-ranking.
            if sleeve == "bare":
                score *= 1.6 if label in _SHORT_SLEEVE else 0.5
            else:  # covered
                score *= 1.3 if label in _LONG_SLEEVE else 0.4
            adj.append((label, score))
        return sorted(adj, key=lambda t: t[1], reverse=True)

    def _detect(self, crop_bgr: np.ndarray, sleeve: str | None) -> dict:
        if self._pipe is None:
            with self._lock:
                return dict(self._cached)
        try:
            from PIL import Image
            image = Image.fromarray(crop_bgr[:, :, ::-1])   # BGR -> RGB
            t0 = time.time()
            ranked = self._classify(image)
            self._last_latency_ms = (time.time() - t0) * 1000.0
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._cached["status"] = f"inference failed: {type(e).__name__}"
                return dict(self._cached)
        if not ranked:
            with self._lock:
                self._cached = {"clothing": "unknown", "confidence": 0.0,
                                "warmth_score": None, "status": "no clothing match"}
                return dict(self._cached)
        ranked = self._fuse_sleeve(ranked, sleeve)
        self._last_top = [(l, round(s, 3)) for l, s in ranked[:5]]
        label, score = ranked[0]
        # Sleeve fusion is a ranking bias and can raise an adjusted score
        # above one; never publish that adjustment as probability/confidence.
        score = max(0.0, min(1.0, float(score)))
        with self._lock:
            if score < self.min_confidence:
                self._cached = {"clothing": "unknown", "confidence": round(score, 2),
                                "warmth_score": None,
                                "status": f"low score: {label} {score:.2f}"}
            else:
                note = f" ({sleeve} arm)" if sleeve else ""
                self._cached = {"clothing": label, "confidence": round(score, 2),
                                "warmth_score": _WARMTH.get(label),
                                "status": "ready" + note}
            return dict(self._cached)

    # ---- per-frame --------------------------------------------------------
    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        self.start()  # direct/replay callers also get preload without Pipeline.run()
        self._refresh_load_state()
        crop = self._torso_crop(ctx)
        sleeve = self._sleeve_state(ctx)
        self._last_sleeve = sleeve
        now = time.time()
        if self._inference_future is not None and self._inference_future.done():
            try:
                with self._lock:
                    self._cached = self._inference_future.result()
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._cached["status"] = f"worker failed: {type(e).__name__}"
            finally:
                self._inference_future = None
        if crop is None:
            with self._lock:
                if (self._pipe is not None
                        and self._cached.get("clothing") in ("...", "unknown", None)):
                    self._cached = {"clothing": "unknown", "confidence": 0.0,
                                    "warmth_score": None,
                                    "status": "no person in frame"}
        elif (self._pipe is not None
              and now - self._last_infer >= self.infer_every
              and self._inference_future is None
              and not self._stopping.is_set()):
            self._last_infer = now
            # Keep the camera loop responsive by running model inference off-frame and reusing the cache.
            self._inference_future = self._executor.submit(
                self._detect, self._resize_crop(crop.copy()), sleeve)

        with self._lock:
            snapshot = dict(self._cached)
        if crop is not None:
            debug_log("clothing", f"crop={crop.shape[1]}x{crop.shape[0]} "
                                  f"sleeve={sleeve} latency_ms={self._last_latency_ms:.0f} "
                                  f"selected={snapshot.get('clothing')} "
                                  f"conf={snapshot.get('confidence')} "
                                  f"status={snapshot.get('status')} top={self._last_top}")
        ctx.extras["clothing"] = snapshot
        # Advice runs on a different cadence, so per-frame extras alone are
        # not a reliable handoff between the two modules.
        self._shared_signals.set("clothing", snapshot, ctx.timestamp)

        label = snapshot.get("clothing", "...")
        conf = max(0.0, min(1.0, float(snapshot.get("confidence", 0.0))))
        results = [
            self.result("upper_body", label, conf, Severity.INFO,
                        f"Clothing detected: {label}" if label not in ("...", "unknown") else "", ttl=12.0),
            self.result("status", snapshot.get("status", "unknown"), 0.0, Severity.INFO, "", ttl=12.0),
        ]
        warmth = snapshot.get("warmth_score")
        if warmth is not None:
            results.append(self.result("warmth_score", int(warmth), conf, Severity.INFO, "", ttl=12.0))
        return results

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self._stopping.set()
        for future in (self._load_future, self._inference_future):
            if future is not None:
                future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True, timeout=1.0)

    def diagnostics(self) -> dict:
        self._refresh_load_state()
        with self._lock:
            status = str(self._cached.get("status", "unknown"))
            ready = self._pipe is not None
        return {"status": "ready" if ready else (
                    "degraded" if self._load_deadline_exceeded else "loading"),
                "ready": ready, "device": self._device,
                "model": self._model_name, "task": self._task,
                "load_latency_ms": round(self._load_latency_ms, 1),
                "load_deadline_exceeded": self._load_deadline_exceeded,
                "inference_latency_ms": round(self._last_latency_ms, 1),
                "detail": status, "worker": self._executor.diagnostics()}
