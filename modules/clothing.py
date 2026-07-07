"""Upper-body clothing detection for weather-aware recommendations.

Uses OWL-ViT / OWLv2 zero-shot object detection via optional Transformers/
PyTorch dependencies. Without those packages the module stays active and
reports an actionable status instead of guessing clothing.

Detection notes (why earlier versions returned "no clothing match"):
- The HF zero-shot-object-detection pipeline defaults to threshold=0.1, but
  OWL score for clothing on a torso crop is ~0.03-0.12, so the default filtered
  everything. We pass a low `pipeline_threshold` and gate afterwards.
- OWL needs to actually see the torso: we detect on the pose person-box when
  available, fall back to a face-derived crop, clamp to the frame, and upscale
  tiny crops. If the torso isn't in frame we say so.
- Fewer, distinct labels wrapped in a prompt template ("a photo of a person
  wearing a t-shirt") score better than many near-synonyms.
- OWLv2 (owlv2-base-patch16-ensemble) is materially more accurate than
  owlvit-base-patch32; it is the default with a fallback.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from core.context import FrameContext
from core.debug import log as debug_log
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import pose as P


_DEFAULT_LABELS = [
    "tank top", "t-shirt", "long sleeve shirt", "hoodie",
    "sweater", "jacket", "coat", "raincoat",
]

_WARMTH = {
    "tank top": 0,
    "t-shirt": 1,
    "long sleeve shirt": 2,
    "hoodie": 3,
    "sweater": 3,
    "jacket": 4,
    "raincoat": 4,
    "coat": 5,
}

_PROMPT = "a photo of a person wearing a {}"
_FALLBACK_MODEL = "google/owlvit-base-patch32"
_MIN_CROP_PX = 160          # upscale crops smaller than this on the short side


@register("clothing")
class Clothing(DetectionModule):
    interval = 0.0
    requires = ("face",)
    backend = "owlvit"              # owlvit | none
    model = "google/owlv2-base-patch16-ensemble"   # OWLv2 default; falls back
    labels = _DEFAULT_LABELS
    pipeline_threshold = 0.02       # low, so the pipeline returns candidates
    min_confidence = 0.08           # post-filter gate
    infer_every = 8.0

    def __init__(self, **params):
        super().__init__(**params)
        self._pipe = None
        self._prompts = [_PROMPT.format(l) for l in self.labels]
        self._prompt_to_label = {p: l for p, l in zip(self._prompts, self.labels)}
        self._last_infer = 0.0
        self._cached = {"clothing": "...", "confidence": 0.0, "status": "warming up"}
        self._load_attempted = False
        self._last_top = []
        self._last_latency_ms = 0.0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clothing")
        self._pending: Future | None = None

    def _load(self) -> None:
        if self._load_attempted or self.backend == "none":
            return
        self._load_attempted = True
        from transformers import pipeline
        for name in (self.model, _FALLBACK_MODEL):
            try:
                self._pipe = pipeline("zero-shot-object-detection", model=name)
                self.model = name
                self._cached["status"] = "ready"
                print(f"[clothing] loaded zero-shot detector {name}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[clothing] could not load {name} ({type(e).__name__}: {e})")
        self._cached["status"] = "clothing model unavailable"

    def _label_of(self, pred_label: str) -> str:
        """Map a returned prompt (or raw label) back to a base clothing label."""
        pl = str(pred_label)
        if pl in self._prompt_to_label:
            return self._prompt_to_label[pl]
        low = pl.lower()
        for base in self.labels:                 # prompt may echo full sentence
            if base in low:
                return base
        return low

    def _person_region(self, ctx: FrameContext) -> np.ndarray | None:
        """Torso/person region for detection. Prefer the pose box (covers
        shoulders->hips); fall back to a face-derived upper-body crop. Clamp to
        the frame and upscale tiny crops so OWL has something to work with."""
        box = None
        if ctx.pose is not None:
            lm = ctx.pose.landmarks
            pts = []
            for idx in (P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP):
                if lm[idx, 3] > 0.4:
                    pts.append(lm[idx, :2] * np.array([ctx.w, ctx.h]))
            if len(pts) >= 2:
                pts = np.array(pts)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                mx = 0.35 * (x2 - x1) + 1
                my = 0.35 * (y2 - y1) + 1
                box = (x1 - mx, y1 - my, x2 + mx, y2 + my)
        if box is None and ctx.face is not None:      # fall back to below-face
            fx1, fy1, fx2, fy2 = ctx.face.bbox
            fw, fh = fx2 - fx1, fy2 - fy1
            cx = (fx1 + fx2) // 2
            box = (cx - 1.4 * fw, fy2, cx + 1.4 * fw, fy2 + 2.4 * fh)
        if box is None:
            return None
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(ctx.w, int(box[2])); y2 = min(ctx.h, int(box[3]))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        crop = ctx.frame[y1:y2, x1:x2]
        short = min(crop.shape[:2])
        if short < _MIN_CROP_PX:                       # upscale small crops
            import cv2
            scale = _MIN_CROP_PX / short
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        return crop

    def _detect(self, crop_bgr: np.ndarray) -> dict:
        self._load()
        if self._pipe is None:
            with self._lock:
                return dict(self._cached)
        try:
            from PIL import Image
            image = Image.fromarray(crop_bgr[:, :, ::-1])   # BGR -> RGB
            t0 = time.time()
            preds = self._pipe(image, candidate_labels=self._prompts,
                               threshold=self.pipeline_threshold)
            self._last_latency_ms = (time.time() - t0) * 1000.0
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._cached["status"] = f"inference failed: {type(e).__name__}"
                return dict(self._cached)
        if not preds:
            with self._lock:
                self._cached = {"clothing": "unknown", "confidence": 0.0,
                                "warmth_score": None, "status": "no clothing match"}
                return dict(self._cached)
        ranked = sorted(preds, key=lambda p: float(p.get("score", 0.0)), reverse=True)
        self._last_top = [(self._label_of(p.get("label", "unknown")),
                           round(float(p.get("score", 0.0)), 3)) for p in ranked[:5]]
        best = ranked[0]
        score = float(best.get("score", 0.0))
        label = self._label_of(best.get("label", "unknown"))
        with self._lock:
            if score < self.min_confidence:
                self._cached = {"clothing": "unknown", "confidence": round(score, 2),
                                "warmth_score": None,
                                "status": f"low score: {label} {score:.2f}"}
            else:
                self._cached = {"clothing": label, "confidence": round(score, 2),
                                "warmth_score": _WARMTH.get(label), "status": "ready"}
            return dict(self._cached)

    def process(self, ctx: FrameContext):
        crop = self._person_region(ctx)
        now = time.time()
        if self._pending is not None and self._pending.done():
            try:
                fresh = self._pending.result()
                with self._lock:
                    self._cached = fresh
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._cached["status"] = f"worker failed: {type(e).__name__}"
            finally:
                self._pending = None
        if crop is None:
            with self._lock:
                if self._cached.get("clothing") in ("...", "unknown", None):
                    self._cached = {"clothing": "unknown", "confidence": 0.0,
                                    "warmth_score": None,
                                    "status": "torso not in frame - sit back"}
        elif now - self._last_infer >= self.infer_every and self._pending is None:
            self._last_infer = now
            self._pending = self._executor.submit(self._detect, crop.copy())

        with self._lock:
            snapshot = dict(self._cached)
        if crop is not None:
            debug_log("clothing", f"crop={crop.shape[1]}x{crop.shape[0]} "
                                  f"latency_ms={self._last_latency_ms:.0f} "
                                  f"selected={snapshot.get('clothing')} "
                                  f"conf={snapshot.get('confidence')} "
                                  f"status={snapshot.get('status')} top={self._last_top}")
        ctx.extras["clothing"] = snapshot

        label = snapshot.get("clothing", "...")
        conf = float(snapshot.get("confidence", 0.0))
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
        self._executor.shutdown(wait=True, cancel_futures=True)
