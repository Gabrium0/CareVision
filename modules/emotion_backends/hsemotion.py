"""HSEmotion emotion+mood backend (tested; AffectNet-trained, ONNX).

Uses the `hsemotion-onnx` package (EfficientNet trained on AffectNet). Runs on
the MediaPipe face crop, returns an 8-class emotion plus a derived valence
("mood") from the class probabilities. ONNX, so it fits our existing
onnxruntime stack and stays CPU-light. Self-disables if the package is absent.

Throttled + cached: inference runs at most every `infer_every` seconds and the
last label is served between runs, so polling every frame stays cheap.
"""
from __future__ import annotations

import time

import numpy as np

from core.context import FrameContext
from modules.backends.base import Backend

# enet_b2_8 class order (AffectNet 8)
_LABELS = ["anger", "contempt", "disgust", "fear", "happiness",
           "neutral", "sadness", "surprise"]
_POS = {"happiness": 1.0, "surprise": 0.3}
_NEG = {"anger": 1.0, "sadness": 1.0, "fear": 1.0, "disgust": 1.0, "contempt": 0.5}


class HSEmotionBackend(Backend):
    """HSEmotion (AffectNet) emotion + valence backend."""
    label = "hsemotion"

    def __init__(self, model_name: str = "enet_b2_8", infer_every: float = 1.0):
        self.available = False
        self.recognizer = None
        self.infer_every = infer_every
        self._crop = None
        self._cv2 = None
        self._last = 0.0
        self._cached = None
        try:
            import cv2
            import urllib.request  # noqa: F401  hsemotion uses urllib.request
            #                       but only does `import urllib`; pre-import so
            #                       its first-run weight download resolves.
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self._cv2 = cv2
            self.recognizer = HSEmotionRecognizer(model_name=model_name)
            self.available = True
            print(f"[emotion/hsemotion] loaded {model_name}")
        except Exception as e:  # noqa: BLE001
            print(f"[emotion/hsemotion] unavailable ({type(e).__name__}: {e})")

    def update(self, ctx: FrameContext) -> None:
        """Feed one frame's data into the backend's rolling state."""
        if self.available and ctx.face is not None:
            crop = ctx.face.crop
            if crop is not None and crop.size:
                self._crop = crop

    def compute(self) -> dict | None:
        """Return the backend's current reading dict, or None if not ready."""
        if not self.available or self._crop is None:
            return self._cached
        now = time.time()
        if now - self._last < self.infer_every:
            return self._cached
        self._last = now
        rgb = self._cv2.cvtColor(self._crop, self._cv2.COLOR_BGR2RGB)
        try:
            label, scores = self.recognizer.predict_emotions(rgb, logits=False)
        except Exception as e:  # noqa: BLE001
            print(f"[emotion/hsemotion] inference failed: {e}")
            return self._cached
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if scores.sum() > 0:
            scores = scores / scores.sum()
        conf = float(scores.max()) if scores.size else 0.5
        # derived valence ("mood"): weighted positive minus negative mass, -1..1
        val = 0.0
        for i, lab in enumerate(_LABELS):
            if i < len(scores):
                val += _POS.get(lab, 0.0) * scores[i]
                val -= _NEG.get(lab, 0.0) * scores[i]
        self._cached = {"emotion": str(label).lower(), "confidence": round(conf, 2),
                        "valence": round(float(np.clip(val, -1, 1)), 2)}
        return self._cached
