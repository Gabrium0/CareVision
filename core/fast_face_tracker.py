"""Lightweight, reader-thread face tracking for full-rate vitals sampling.

MediaPipe remains the authority: this tracker only bridges the short interval
between detections. It tracks anonymous image corners inside the current face
box and exposes no points or pixels through diagnostics.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .context import FaceData


@dataclass
class TrackResult:
    face: FaceData | None
    reason: str


class FastFaceTracker:
    """Bounded optical-flow bridge between authoritative face detections."""

    def __init__(self, width: int = 240, max_anchor_age: float = 1.5,
                 min_points: int = 12, min_inlier_ratio: float = 0.6):
        self.width = max(160, int(width))
        self.max_anchor_age = max(0.25, float(max_anchor_age))
        self.min_points = max(6, int(min_points))
        self.min_inlier_ratio = float(np.clip(min_inlier_ratio, 0.3, 1.0))
        self._lock = threading.Lock()
        self._gray: np.ndarray | None = None
        self._points: np.ndarray | None = None
        self._face: FaceData | None = None
        self._anchor_at = 0.0
        self._last_at = 0.0
        self._frame_shape: tuple[int, int] | None = None
        self._tracked = 0
        self._track_times: deque[float] = deque(maxlen=120)
        self._failures = 0
        self._last_reason = "waiting for MediaPipe anchor"

    def _small_gray(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.width / max(w, 1))
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), sw / w, sh / h

    def seed(self, frame: np.ndarray, face: FaceData, timestamp: float) -> bool:
        """Replace tracking state with a fresh authoritative MediaPipe face."""
        gray, sx, sy = self._small_gray(frame)
        x1, y1, x2, y2 = face.bbox
        mask = np.zeros_like(gray)
        ax1, ay1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        ax2, ay2 = min(gray.shape[1], int(x2 * sx)), min(gray.shape[0], int(y2 * sy))
        if ax2 <= ax1 or ay2 <= ay1:
            return False
        mask[ay1:ay2, ax1:ax2] = 255
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=48, qualityLevel=0.01, minDistance=5,
            blockSize=5, mask=mask)
        if points is None or len(points) < self.min_points:
            with self._lock:
                self._clear_locked("insufficient anchor features", failure=True)
            return False
        with self._lock:
            self._gray = gray
            self._points = points.astype(np.float32)
            self._face = face
            self._anchor_at = float(timestamp)
            self._last_at = float(timestamp)
            self._frame_shape = frame.shape[:2]
            self._last_reason = "anchored"
        return True

    def track(self, frame: np.ndarray, timestamp: float) -> TrackResult:
        """Track one current frame; return a fresh crop or a safe rejection."""
        with self._lock:
            if self._face is None or self._gray is None or self._points is None:
                return TrackResult(None, self._last_reason)
            if timestamp - self._anchor_at > self.max_anchor_age:
                self._clear_locked("anchor expired", failure=True)
                return TrackResult(None, "anchor expired")
            if self._frame_shape != frame.shape[:2]:
                self._clear_locked("frame size changed", failure=True)
                return TrackResult(None, "frame size changed")

            gray, sx, sy = self._small_gray(frame)
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self._gray, gray, self._points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            if new_points is None or status is None:
                self._clear_locked("optical flow failed", failure=True)
                return TrackResult(None, "optical flow failed")
            good = status.reshape(-1).astype(bool)
            if error is not None:
                good &= np.isfinite(error.reshape(-1)) & (error.reshape(-1) <= 20.0)
            old = self._points.reshape(-1, 2)[good]
            new = new_points.reshape(-1, 2)[good]
            if len(new) < self.min_points:
                self._clear_locked("tracked features lost", failure=True)
                return TrackResult(None, "tracked features lost")
            affine, inliers = cv2.estimateAffinePartial2D(
                old, new, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            inlier_ratio = (float(np.mean(inliers)) if inliers is not None and len(inliers)
                            else 0.0)
            if affine is None or inlier_ratio < self.min_inlier_ratio:
                self._clear_locked("unstable affine track", failure=True)
                return TrackResult(None, "unstable affine track")
            scale = float(np.hypot(affine[0, 0], affine[0, 1]))
            x1, y1, x2, y2 = self._face.bbox
            box_w_small = max(1.0, (x2 - x1) * sx)
            box_h_small = max(1.0, (y2 - y1) * sy)
            translation = float(np.hypot(affine[0, 2], affine[1, 2]))
            if not 0.85 <= scale <= 1.18 or translation > 0.25 * max(box_w_small, box_h_small):
                self._clear_locked("track transform out of bounds", failure=True)
                return TrackResult(None, "track transform out of bounds")

            h, w = frame.shape[:2]
            corners = np.array([[x1 * sx, y1 * sy], [x2 * sx, y1 * sy],
                                [x2 * sx, y2 * sy], [x1 * sx, y2 * sy]],
                               dtype=np.float32)
            moved = cv2.transform(corners[None, :, :], affine)[0]
            nx1 = max(0, int(np.floor(moved[:, 0].min() / sx)))
            ny1 = max(0, int(np.floor(moved[:, 1].min() / sy)))
            nx2 = min(w, int(np.ceil(moved[:, 0].max() / sx)))
            ny2 = min(h, int(np.ceil(moved[:, 1].max() / sy)))
            if nx2 <= nx1 or ny2 <= ny1:
                self._clear_locked("tracked box invalid", failure=True)
                return TrackResult(None, "tracked box invalid")

            landmarks = np.array(self._face.landmarks, copy=True)
            lm_small = landmarks[:, :2] * np.array([gray.shape[1], gray.shape[0]])
            moved_lm = cv2.transform(lm_small.astype(np.float32)[None, :, :], affine)[0]
            landmarks[:, :2] = moved_lm / np.array([gray.shape[1], gray.shape[0]])
            crop = frame[ny1:ny2, nx1:nx2]
            face = FaceData(landmarks=landmarks, bbox=(nx1, ny1, nx2, ny2),
                            crop=crop, has_iris=self._face.has_iris)
            self._gray = gray
            self._points = new.reshape(-1, 1, 2).astype(np.float32)
            self._face = face
            self._last_at = float(timestamp)
            self._tracked += 1
            self._track_times.append(float(timestamp))
            self._last_reason = "tracked"
            return TrackResult(face, "tracked")

    def reset(self, reason: str = "reset") -> None:
        with self._lock:
            self._clear_locked(reason, failure=False)

    def _clear_locked(self, reason: str, failure: bool) -> None:
        self._gray = self._points = self._face = None
        self._anchor_at = self._last_at = 0.0
        self._frame_shape = None
        self._last_reason = reason
        if failure:
            self._failures += 1

    def diagnostics(self, now: float) -> dict:
        with self._lock:
            rate = 0.0
            if len(self._track_times) > 1:
                span = self._track_times[-1] - self._track_times[0]
                rate = (len(self._track_times) - 1) / span if span > 0 else 0.0
            return {
                "active": self._face is not None,
                "anchor_age_seconds": (round(max(0.0, now - self._anchor_at), 3)
                                       if self._anchor_at else None),
                "tracked_frames": self._tracked,
                "tracked_sample_hz": round(rate, 2),
                "failures": self._failures,
                "last_reason": self._last_reason,
                "max_anchor_age_seconds": self.max_anchor_age,
            }
