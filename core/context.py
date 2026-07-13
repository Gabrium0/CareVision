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
class Intrinsics:
    """Pinhole camera intrinsics for metric depth math.

    A plain dataclass (not pyrealsense2's intrinsics type) so depth-aware
    modules and tests never need the RealSense SDK importable; the RealSense
    backend copies its calibrated values in here, rescaled to whatever frame
    size it actually delivers.
    """
    fx: float                      # focal length in pixels, x
    fy: float                      # focal length in pixels, y
    ppx: float                     # principal point x (pixels)
    ppy: float                     # principal point y (pixels)


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
    # Depth-capable sources (RealSense D435i) fill these; RGB-only sources
    # leave the defaults, and depth-dependent modules are skipped via the
    # scheduler's "depth" requires-token when `depth` is None.
    depth: Optional[np.ndarray] = None   # uint16 depth aligned to `frame`, same HxW
    depth_scale: float = 0.001           # meters per depth unit (D435i default 1mm)
    intrinsics: Optional[Intrinsics] = None  # color intrinsics at delivered size
    ego_motion: float = 0.0              # IMU motion magnitude (rad/s); 0 = static

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

    def depth_m(self, px: float, py: float, win: int = 5) -> Optional[float]:
        """Median depth in meters over a `win`x`win` window at pixel (px, py).

        The median over a small window rides out the D435i's per-pixel noise
        and the zero-valued holes stereo depth leaves on hair/edges; returns
        None when depth is absent or every pixel in the window is a hole.
        """
        if self.depth is None:
            return None
        x, y = int(px), int(py)
        r = max(win // 2, 0)
        patch = self.depth[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) * self.depth_scale

    def deproject(self, px: float, py: float, win: int = 5) -> Optional[np.ndarray]:
        """3D point (x, y, z) in meters, camera frame, at pixel (px, py).

        Plain pinhole math on our own `Intrinsics` (no distortion terms —
        negligible at the D435i color FOV center where subjects stand) so
        this works without pyrealsense2 installed.
        """
        if self.intrinsics is None:
            return None
        z = self.depth_m(px, py, win=win)
        if z is None:
            return None
        i = self.intrinsics
        return np.array([(px - i.ppx) / i.fx * z, (py - i.ppy) / i.fy * z, z])

    def mm_per_px(self, px: float, py: float) -> Optional[float]:
        """Millimeters spanned by one pixel at the subject's distance —
        lets ROI radii and drift thresholds be expressed in mm regardless
        of how far the person stands."""
        if self.intrinsics is None:
            return None
        z = self.depth_m(px, py)
        if z is None:
            return None
        return z / self.intrinsics.fx * 1000.0
