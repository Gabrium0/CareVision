"""Per-frame shared state passed to every module.

Expensive extraction (face mesh, pose, motion) runs ONCE per frame in the
extractors and is cached here; modules only read from the context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FaceData:
    """Per-frame face landmarks, bounding box, and crop."""
    landmarks: np.ndarray          # (478, 3) normalized face-mesh landmarks
    bbox: tuple                    # (x1, y1, x2, y2) pixel coords
    crop: np.ndarray               # BGR face crop
    has_iris: bool = True          # refine_landmarks gives 478 points incl. iris


@dataclass
class PoseData:
    """Per-frame body-pose landmarks and bounding box."""
    landmarks: np.ndarray          # (33, 4) normalized pose landmarks (x, y, z, visibility)
    bbox: tuple                    # (x1, y1, x2, y2) pixel coords of person


@dataclass
class FrameContext:
    """Per-frame shared state passed to every module (frame + extractor outputs + scratch)."""
    frame: np.ndarray              # BGR frame
    timestamp: float               # capture time (time.time())
    frame_index: int
    fps: float                     # measured capture fps
    face: Optional[FaceData] = None
    pose: Optional[PoseData] = None
    motion_energy: float = 0.0     # mean abs frame diff, 0..255 scale
    person_present: bool = False
    extras: dict = field(default_factory=dict)  # scratch space for extractors

    @property
    def h(self) -> int:
        return self.frame.shape[0]

    @property
    def w(self) -> int:
        return self.frame.shape[1]

    def face_px(self) -> Optional[np.ndarray]:
        """Face landmarks in pixel coordinates, shape (478, 2)."""
        if self.face is None:
            return None
        return self.face.landmarks[:, :2] * np.array([self.w, self.h])

    def pose_px(self) -> Optional[np.ndarray]:
        """Pose landmarks in pixel coordinates, shape (33, 2)."""
        if self.pose is None:
            return None
        return self.pose.landmarks[:, :2] * np.array([self.w, self.h])
