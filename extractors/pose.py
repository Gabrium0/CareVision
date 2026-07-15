"""Shared body-pose extractor: MediaPipe Tasks PoseLandmarker (33 landmarks).

Uses the modern Tasks API (legacy mp.solutions is absent in mediapipe
>= 0.10.30). Requires models/pose_landmarker_lite.task.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from core.context import FrameContext, PoseData

_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

# Landmark indices used by modules
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


class PoseExtractor:
    """MediaPipe PoseLandmarker extractor; fills ctx.pose once per frame."""
    def __init__(self, input_width: int = 960):
        self.input_width = max(320, int(input_width))
        if not _MODEL.exists():
            raise FileNotFoundError(
                f"Missing {_MODEL}. Download pose_landmarker_lite.task from "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

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
        if not out.pose_landmarks:
            return
        ctx.extras["pose_count"] = len(out.pose_landmarks)
        ctx.extras["poses"] = []
        for raw in out.pose_landmarks:
            raw_pts = np.array([[p.x, p.y, p.z, p.visibility] for p in raw], dtype=np.float32)
            visible = raw_pts[:, 3] > 0.5
            if visible.sum() >= 4:
                raw_px = raw_pts[visible, :2] * np.array([ctx.w, ctx.h])
                ax1, ay1 = raw_px.min(axis=0).astype(int)
                ax2, ay2 = raw_px.max(axis=0).astype(int)
                ctx.extras["poses"].append({"bbox": (max(0, ax1), max(0, ay1),
                                                       min(ctx.w, ax2), min(ctx.h, ay2)),
                                             "landmarks": raw_pts})
        lm = max(out.pose_landmarks,
                 key=lambda pts: (max(p.x for p in pts) - min(p.x for p in pts)) *
                                  (max(p.y for p in pts) - min(p.y for p in pts)))
        pts = np.array([[p.x, p.y, p.z, p.visibility] for p in lm],
                       dtype=np.float32)
        vis = pts[:, 3] > 0.5
        if vis.sum() < 4:
            return
        px = pts[vis, :2] * np.array([ctx.w, ctx.h])
        x1, y1 = px.min(axis=0).astype(int)
        x2, y2 = px.max(axis=0).astype(int)
        ctx.pose = PoseData(landmarks=pts, bbox=(max(0, x1), max(0, y1),
                                                 min(ctx.w, x2), min(ctx.h, y2)))
        ctx.person_present = True

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self.landmarker.close()
