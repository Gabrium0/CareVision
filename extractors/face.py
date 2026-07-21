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
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from core.context import FaceData, FrameContext
from extractors.smoothing import OneEuroArray

_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


class FaceExtractor:
    """MediaPipe FaceLandmarker extractor; fills ctx.face once per frame."""
    def __init__(self, smooth: bool = True, input_width: int = 960):
        # One-Euro de-jitter on face landmarks steadies emotion/asymmetry/EAR;
        # face motion is low-frequency so this does not blur any measured signal.
        self._smooth_enabled = bool(smooth)
        self._smoother = (OneEuroArray(mincutoff=1.5, beta=0.05)
                          if self._smooth_enabled else None)
        self.input_width = max(320, int(input_width))
        if not _MODEL.exists():
            raise FileNotFoundError(
                f"Missing {_MODEL}. Download face_landmarker.task from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task")
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def extract(self, ctx: FrameContext) -> None:
        """Extract features from the frame and populate the shared context."""
        source = ctx.frame
        if ctx.w > self.input_width:
            scale = self.input_width / ctx.w
            source = cv2.resize(ctx.frame, (self.input_width, max(1, int(ctx.h * scale))))
        rgb = np.ascontiguousarray(source[:, :, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(ctx.timestamp * 1000)
        out = self.landmarker.detect_for_video(mp_img, ts_ms)
        if not out.face_landmarks:
            return
        ctx.extras["face_count"] = len(out.face_landmarks)
        ctx.extras["faces"] = []
        for raw in out.face_landmarks:
            raw_pts = np.array([[p.x, p.y, p.z] for p in raw], dtype=np.float32)
            raw_px = raw_pts[:, :2] * np.array([ctx.w, ctx.h])
            ax1, ay1 = raw_px.min(axis=0).astype(int)
            ax2, ay2 = raw_px.max(axis=0).astype(int)
            ctx.extras["faces"].append({"bbox": (max(0, ax1), max(0, ay1),
                                                    min(ctx.w, ax2), min(ctx.h, ay2)),
                                         "landmarks": raw_pts})
        # Select the closest-looking (largest image area) face, but preserve
        # the count so the showcase gate can reject competing guests.
        lm = max(out.face_landmarks,
                 key=lambda pts: (max(p.x for p in pts) - min(p.x for p in pts)) *
                                  (max(p.y for p in pts) - min(p.y for p in pts)))
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        if self._smoother is not None:
            pts = self._smoother(pts, ctx.timestamp).astype(np.float32)

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

    def reset(self) -> None:
        """Discard subject-bound smoothing after a camera/source switch."""
        self._smoother = (OneEuroArray(mincutoff=1.5, beta=0.05)
                          if self._smooth_enabled else None)

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self.landmarker.close()
