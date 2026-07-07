"""Shared face extractor: MediaPipe Tasks FaceLandmarker (478 landmarks
including iris). Runs once per frame; every face-based module reads
ctx.face instead of running its own detector. Landmark map lives in
extractors/face_landmarks.py.

Uses the modern Tasks API (the legacy mp.solutions API is absent in
mediapipe >= 0.10.30). Requires models/face_landmarker.task.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from core.context import FaceData, FrameContext

_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


class FaceExtractor:
    def __init__(self):
        if not _MODEL.exists():
            raise FileNotFoundError(
                f"Missing {_MODEL}. Download face_landmarker.task from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task")
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def extract(self, ctx: FrameContext) -> None:
        rgb = np.ascontiguousarray(ctx.frame[:, :, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(ctx.timestamp * 1000)
        out = self.landmarker.detect_for_video(mp_img, ts_ms)
        if not out.face_landmarks:
            return
        lm = out.face_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

        px = pts[:, :2] * np.array([ctx.w, ctx.h])
        x1, y1 = px.min(axis=0).astype(int)
        x2, y2 = px.max(axis=0).astype(int)
        pad = int(0.05 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(ctx.w, x2 + pad), min(ctx.h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return

        ctx.face = FaceData(landmarks=pts, bbox=(x1, y1, x2, y2),
                            crop=ctx.frame[y1:y2, x1:x2],
                            has_iris=pts.shape[0] >= 478)
        ctx.person_present = True

    def close(self) -> None:
        self.landmarker.close()
