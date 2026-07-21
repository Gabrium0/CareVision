"""Age estimation (optional ONNX model).

Method: if models/age_googlenet.onnx (Levi-Hassner age net, 8 buckets) is
present, run it on the face crop. Otherwise the module disables itself
cleanly. Age is emitted once and cached (people don't change age mid-
session) to save compute.

Reliability: coarse buckets only; MEDIUM at best.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.capabilities import CapabilityStatus
from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule

_MODEL = Path(__file__).resolve().parent.parent / "models" / "age_googlenet.onnx"
_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


@register("age_estimation")
class AgeEstimation(DetectionModule):
    """Age estimation (optional ONNX model)."""
    interval = 5.0
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.session = None
        self._emitted = False
        if _MODEL.exists():
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    str(_MODEL), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:      # noqa: BLE001
                print(f"[age_estimation] ONNX load failed ({e}); disabled")
        else:
            print("[age_estimation] no model at models/age_googlenet.onnx; disabled")

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        if self.session is None or self._emitted:
            return None
        x1, y1, x2, y2 = ctx.face.bbox
        crop = ctx.frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        blob = cv2.dnn.blobFromImage(crop, 1.0, (224, 224), _MEAN, swapRB=False)
        probs = self.session.run(None, {self.input_name: blob.astype(np.float32)})[0].ravel()
        i = int(np.argmax(probs))
        self._emitted = True
        return self.result("age_range", _BUCKETS[i], float(probs[i]),
                           Severity.INFO, f"Estimated age {_BUCKETS[i]}", ttl=120.0)

    def capability_status(self):
        if self.session is None:
            return CapabilityStatus.UNCONFIGURED, "models/age_googlenet.onnx is not installed"
        return CapabilityStatus.READY, "ONNX age model loaded"
