"""DeepFace emotion (+age/gender) backend (tested; from research.md).

Uses the `deepface` package. Runs on the MediaPipe face crop with detection
skipped (we already have the face). Emotion is the headline; age/gender are
exposed too so the age module can reuse this backend. Self-disables if deepface
(TensorFlow) isn't installed.

DeepFace is comparatively heavy, so inference is throttled and cached.
"""
from __future__ import annotations

import time

from core.context import FrameContext
from modules.backends.base import Backend


class DeepFaceBackend(Backend):
    label = "deepface"

    def __init__(self, actions=("emotion",), infer_every: float = 1.5):
        self.available = False
        self._df = None
        self.actions = list(actions)
        self.infer_every = infer_every
        self._crop = None
        self._last = 0.0
        self._cached = None
        try:
            from deepface import DeepFace
            self._df = DeepFace
            self.available = True
            print(f"[emotion/deepface] loaded (actions={self.actions})")
        except Exception as e:  # noqa: BLE001
            print(f"[emotion/deepface] unavailable ({type(e).__name__}: {e})")

    def update(self, ctx: FrameContext) -> None:
        if self.available and ctx.face is not None:
            crop = ctx.face.crop
            if crop is not None and crop.size:
                self._crop = crop            # BGR, DeepFace's expected order

    def compute(self) -> dict | None:
        if not self.available or self._crop is None:
            return self._cached
        now = time.time()
        if now - self._last < self.infer_every:
            return self._cached
        self._last = now
        try:
            res = self._df.analyze(self._crop, actions=self.actions,
                                   enforce_detection=False, detector_backend="skip",
                                   silent=True)
        except Exception as e:  # noqa: BLE001
            print(f"[emotion/deepface] inference failed: {e}")
            return self._cached
        if isinstance(res, list):
            res = res[0] if res else {}
        out = {}
        if "dominant_emotion" in res:
            out["emotion"] = str(res["dominant_emotion"]).lower()
            scores = res.get("emotion", {})
            top = max(scores.values()) if scores else 100.0
            out["confidence"] = round(float(top) / 100.0, 2)
        if "age" in res:
            out["age"] = int(res["age"])
        if "dominant_gender" in res:
            out["gender"] = str(res["dominant_gender"]).lower()
        self._cached = out or self._cached
        return self._cached
