"""Frame source abstraction: webcam index, video file, or RTSP URL.

For a live webcam we lock exposure / white-balance / gain by default. Auto
exposure and auto white-balance continuously rescale pixel values in
response to the scene, which is the single biggest accuracy killer for
rPPG (heart rate) and every skin-color signal (pallor/jaundice/cyanosis):
the personal color baseline drifts under the algorithm's feet. Locking them
trades a one-time manual setup for stable, comparable pixels over time.

We also request a fixed capture fps and warn if the delivered rate drifts,
because the frequency-domain modules (heart rate, tremor, respiration)
resample assuming ~30 fps and will alias if the camera runs slower.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
import numpy as np

from .context import FrameContext


class Camera:
    """Frame source abstraction over a webcam, video file, or stream."""
    def __init__(self, source: int | str = 0, target_width: int = 960,
                 lock: bool = True, exposure: float | None = None,
                 request_fps: float = 30.0, request_size: tuple = (1280, 720),
                 settle_seconds: float = 1.5):
        self.source = source
        self.target_width = target_width
        self.lock = lock                     # lock exposure/WB/gain (webcam only)
        self.exposure = exposure             # manual exposure; None = keep current
        self.request_fps = request_fps
        self.request_size = request_size
        self.settle_seconds = settle_seconds
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps_smooth = float(request_fps)
        self._last_t: Optional[float] = None
        self._fps_warned = False

    def _is_webcam(self) -> bool:
        return isinstance(self.source, int)

    def _configure(self) -> None:
        """Request resolution/fps and (optionally) lock auto controls."""
        cap = self.cap
        w, h = self.request_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, self.request_fps)
        if not (self._is_webcam() and self.lock):
            return
        # Let auto-exposure settle to a sane level first, then freeze it.
        deadline = time.time() + self.settle_seconds
        while time.time() < deadline:
            cap.read()
        # 0.25 = manual mode for most DirectShow/UVC cameras (0.75 = auto).
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:  # noqa: BLE001
            pass
        if self.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        # Freeze white balance at its current (settled) temperature if readable.
        wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        if wb and wb > 0:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        print(f"[camera] locked auto-exposure/WB (exposure="
              f"{cap.get(cv2.CAP_PROP_EXPOSURE):.0f}, wb={wb:.0f}); "
              f"requested {w}x{h}@{self.request_fps:.0f}fps")

    def open(self) -> None:
        """Open the underlying capture source."""
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        backend = cv2.CAP_DSHOW if self._is_webcam() else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.source, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")
        self._configure()

    def _check_fps(self) -> None:
        if self._fps_warned or self._is_webcam() is False:
            return
        # after a few seconds of streaming, warn once if far from requested fps
        if self._last_t and self._fps_smooth < 0.8 * self.request_fps:
            print(f"[camera] WARNING: delivering ~{self._fps_smooth:.1f} fps "
                  f"(< {self.request_fps:.0f}); frequency-domain vitals/tremor "
                  "may be inaccurate. Improve lighting or lower resolution.")
            self._fps_warned = True

    def frames(self) -> Iterator[FrameContext]:
        """Yield a FrameContext per captured frame."""
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
                if idx == 90:                 # ~3s in at 30fps
                    self._check_fps()
            self._last_t = now
            yield FrameContext(frame=frame, timestamp=now, frame_index=idx,
                               fps=self._fps_smooth)
            idx += 1

    def release(self) -> None:
        """Release the capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
