"""FER+ ONNX emotion backend (optional).

Uses models/emotion.onnx (FER+ 64x64 grayscale, 8 classes) if present; self-
disables otherwise. Preserves the original ONNX path from emotion.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.context import FrameContext
from modules.backends.base import Backend

_MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "emotion.onnx"
_FERPLUS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


class FerPlusBackend(Backend):
    label = "ferplus"

    def __init__(self):
        self.available = False
        self.session = None
        self.input_name = None
        self._crop = None
        if _MODEL.exists():
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    str(_MODEL), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.available = True
            except Exception as e:  # noqa: BLE001
                print(f"[emotion/ferplus] ONNX load failed ({e}); disabled")

    def update(self, ctx: FrameContext) -> None:
        if self.available and ctx.face is not None:
            x1, y1, x2, y2 = ctx.face.bbox
            self._crop = ctx.frame[y1:y2, x1:x2]

    def compute(self) -> dict | None:
        if not self.available or self._crop is None or self._crop.size == 0:
            return None
        gray = cv2.cvtColor(self._crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32)
        logits = self.session.run(None, {self.input_name: gray[None, None]})[0].ravel()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        i = int(np.argmax(probs))
        return {"emotion": _FERPLUS[i] if i < len(_FERPLUS) else str(i),
                "confidence": float(probs[i])}
