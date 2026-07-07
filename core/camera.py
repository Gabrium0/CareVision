"""Frame source abstraction: webcam index, video file, or RTSP URL."""
from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
import numpy as np

from .context import FrameContext


class Camera:
    def __init__(self, source: int | str = 0, target_width: int = 960):
        self.source = source
        self.target_width = target_width
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps_smooth = 30.0
        self._last_t: Optional[float] = None

    def open(self) -> None:
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        backend = cv2.CAP_DSHOW if isinstance(self.source, int) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.source, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")

    def frames(self) -> Iterator[FrameContext]:
        if self.cap is None:
            self.open()
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if frame.shape[1] > self.target_width:
                scale = self.target_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            now = time.time()
            if self._last_t is not None:
                dt = max(now - self._last_t, 1e-3)
                self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * (1.0 / dt)
            self._last_t = now
            yield FrameContext(frame=frame, timestamp=now, frame_index=idx,
                               fps=self._fps_smooth)
            idx += 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
