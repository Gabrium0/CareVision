"""Emotion recognition.

Two-tier: if an ONNX FER model is present at models/emotion.onnx (e.g. the
FER+ 64x64 grayscale model, 8 classes), it is used. Otherwise a transparent
facial-geometry heuristic runs from landmarks (smile curvature, mouth open,
brow raise/furrow, eye openness) mapping to happy/sad/surprise/angry/neutral.

Reliability: ONNX path MEDIUM-HIGH; heuristic path LOW-MEDIUM but fully
offline and dependency-free.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import face_landmarks as FL

_MODEL = Path(__file__).resolve().parent.parent / "models" / "emotion.onnx"
_FERPLUS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


@register("emotion")
class Emotion(DetectionModule):
    interval = 0.4
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.session = None
        self.input_name = None
        if _MODEL.exists():
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    str(_MODEL), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                print("[emotion] using ONNX model")
            except Exception as e:      # noqa: BLE001
                print(f"[emotion] ONNX load failed ({e}); using heuristic")

    def process(self, ctx: FrameContext):
        if self.session is not None:
            return self._onnx(ctx)
        return self._heuristic(ctx)

    def _onnx(self, ctx):
        x1, y1, x2, y2 = ctx.face.bbox
        crop = ctx.frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32)
        inp = gray[None, None, :, :]
        logits = self.session.run(None, {self.input_name: inp})[0].ravel()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        i = int(np.argmax(probs))
        label = _FERPLUS[i] if i < len(_FERPLUS) else str(i)
        return self.result("emotion", label, float(probs[i]),
                           Severity.INFO, f"Emotion: {label}", ttl=3.0)

    def _heuristic(self, ctx):
        px = ctx.face_px()
        fw = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        mouth_w = np.linalg.norm(px[FL.MOUTH_LEFT] - px[FL.MOUTH_RIGHT]) / fw
        mouth_h = np.linalg.norm(px[FL.MOUTH_TOP_INNER] - px[FL.MOUTH_BOTTOM_INNER]) / fw
        corner_y = (px[FL.MOUTH_LEFT][1] + px[FL.MOUTH_RIGHT][1]) / 2.0
        center_y = (px[FL.MOUTH_TOP_INNER][1] + px[FL.MOUTH_BOTTOM_INNER][1]) / 2.0
        smile = (center_y - corner_y) / (fw)      # corners above center => smile
        brow_eye = np.linalg.norm(px[FL.LEFT_BROW[2]] - px[159]) / fw
        eye_open = np.linalg.norm(px[159] - px[145]) / fw

        label, conf = "neutral", 0.4
        if smile > 0.015 and mouth_w > 0.45:
            label, conf = "happy", min(0.8, 0.4 + smile * 15)
        elif mouth_h > 0.35 and eye_open > 0.10:
            label, conf = "surprise", 0.6
        elif brow_eye < 0.11 and smile < 0:
            label, conf = "angry", 0.5
        elif smile < -0.01:
            label, conf = "sad", 0.5
        return self.result("emotion", label, conf, Severity.INFO,
                           f"Emotion: {label}", ttl=3.0)
