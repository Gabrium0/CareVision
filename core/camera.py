"""Frame source abstraction: webcam index, video file, or RTSP URL.

For a live webcam we lock exposure / white-balance / gain by default. Auto
exposure and auto white-balance continuously rescale pixel values in
response to the scene, which is the single biggest accuracy killer for
rPPG (heart rate) and every skin-color signal (pallor/jaundice/cyanosis):
the personal color baseline drifts under the algorithm's feet. Locking them
trades a one-time manual setup for stable, comparable pixels over time.

We also request a fixed capture fps and warn if the delivered rate drifts,
because the frequency-domain modules (heart rate, tremor, respiration)
resample assuming ~30 fps and will alias if the camera runs slower. Frame
rate is prioritized over brightness when tuning exposure/gain: a long
exposure (e.g. -3 => ~125ms integration => an ~8fps ceiling on many UVC
cameras) starves the FFT-based vitals of samples far worse than a dim image
starves them of signal-to-noise, so we shrink exposure first and use sensor
gain to hold brightness, only lengthening exposure again if fps has room.

On a live webcam a background reader thread owns the device so vitals
modules can sample every captured frame via `register_fast_hook()`, even
when the heavier per-frame detection pipeline can't keep up; the main loop
consumes whatever frame is latest, naturally dropping any it can't process.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .context import FrameContext


class Camera:
    """Frame source abstraction over a webcam, video file, or stream."""
    def __init__(self, source: int | str = 0, target_width: int = 960,
                 lock: bool = True, exposure: float | None = None,
                 request_fps: float = 30.0, request_size: tuple = (1280, 720),
                 settle_seconds: float = 1.5, target_brightness: float = 90.0,
                 allow_gain_boost: bool = True):
        self.source = source
        self.target_width = target_width
        self.lock = lock                     # lock exposure/WB/gain (webcam only)
        self.exposure = exposure             # manual exposure; None = keep current
        self.request_fps = request_fps
        self.request_size = request_size
        self.settle_seconds = settle_seconds
        self.target_brightness = target_brightness   # mean luminance to reach before locking
        self.allow_gain_boost = allow_gain_boost     # raise gain if exposure alone is too dark
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps_smooth = float(request_fps)
        self._last_t: Optional[float] = None
        self._fps_warned = False
        # Reader-thread state (webcam only; see frames()/_reader_loop()).
        self._fast_hooks: list[Callable[[np.ndarray, float], None]] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: Optional[float] = None
        self._latest_index = -1

    def _is_webcam(self) -> bool:
        return isinstance(self.source, int)

    @property
    def current_fps(self) -> float:
        """Smoothed delivered fps, readable from other threads/modules."""
        return self._fps_smooth

    def register_fast_hook(self, hook: Callable[[np.ndarray, float], None]) -> None:
        """Register `hook(frame, timestamp)` to run on the reader thread for
        every captured frame (webcam sources only), ahead of the slower
        per-frame detection pipeline. Keep hooks light — no MediaPipe/heavy
        CV — since they run once per raw camera frame, not once per
        processed frame."""
        self._fast_hooks.append(hook)

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
        else:
            # In a dark room, locking now would freeze a dark/slow image and
            # starve rPPG of both signal and samples. Tune exposure/gain
            # (fps first, then brightness) BEFORE locking.
            self._tune_exposure_and_gain()
        # Freeze white balance at its current (settled) temperature if readable.
        wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        if wb and wb > 0:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        bright = self._measure_brightness()
        fps = self._measure_fps(n=10)
        print(f"[camera] locked auto-exposure/WB (exposure="
              f"{cap.get(cv2.CAP_PROP_EXPOSURE):.0f}, gain={cap.get(cv2.CAP_PROP_GAIN):.0f}, "
              f"wb={wb:.0f}, brightness={bright:.0f}, measured_fps={fps:.1f}); "
              f"requested {w}x{h}@{self.request_fps:.0f}fps")
        if fps < 0.9 * self.request_fps:
            print(f"[camera] WARNING: sensor still only delivers ~{fps:.1f} fps at this "
                  f"exposure; frequency-domain vitals will alias. Try more light "
                  "(shorter exposure needs fewer photons per frame) or a lower --fps.")
        if bright < self.target_brightness * 0.6:
            print(f"[camera] WARNING: scene is dark (brightness ~{bright:.0f} < "
                  f"{self.target_brightness:.0f}); rPPG heart rate and skin-color "
                  "signals will be unreliable. Add light for accurate vitals.")

    def _measure_brightness(self) -> float:
        """Mean luminance (0-255) of a freshly grabbed frame; 0 if unavailable."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    def _measure_fps(self, n: int = 15) -> float:
        """Delivered fps measured by timing n consecutive reads (best-effort)."""
        cap = self.cap
        t0 = time.time()
        count = 0
        for _ in range(n):
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        dt = max(time.time() - t0, 1e-6)
        return count / dt if count else 0.0

    def _tune_exposure_and_gain(self) -> None:
        """Fps first, brightness second: a long exposure caps frame rate far
        more damagingly for the FFT-based vitals than a dim image caps SNR
        (no amount of post-processing recovers frames the sensor never
        captured). So: if delivered fps is short, shrink exposure (raising
        gain to hold brightness) until fps recovers or an exposure floor is
        hit; only then, if brightness is still low and fps has headroom, grow
        exposure back up toward `target_brightness`, using gain once exposure
        reaches the fps-safe ceiling. Best-effort/bounded: camera property
        scales are driver-specific, so we nudge and re-measure rather than
        compute exact steps.
        """
        cap = self.cap
        fps = self._measure_fps(n=10)
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = cap.get(cv2.CAP_PROP_GAIN)
        fps_floor = 0.9 * self.request_fps

        if fps < fps_floor:
            for _ in range(10):
                exposure -= 1.0                       # halves integration time (log2 scale)
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                if self.allow_gain_boost:
                    gain += 8.0
                    cap.set(cv2.CAP_PROP_GAIN, gain)
                for _ in range(2):
                    cap.read()                         # let the new setting take effect
                fps = self._measure_fps(n=8)
                if fps >= fps_floor:
                    break

        bright = self._measure_brightness()
        if bright >= self.target_brightness:
            return
        # exposure (log2 seconds) that still guarantees the requested fps.
        exposure_ceiling = (math.log2(1.0 / self.request_fps) if self.request_fps > 0
                            else exposure)
        for _ in range(8):
            if bright >= self.target_brightness:
                return
            if exposure < exposure_ceiling - 0.5:
                exposure += 1.0
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            elif self.allow_gain_boost:
                gain += 8.0
                if not cap.set(cv2.CAP_PROP_GAIN, gain):
                    return                              # camera has no writable gain
            else:
                return
            for _ in range(2):
                cap.read()
            bright = self._measure_brightness()

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

    def _prep_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] > self.target_width:
            scale = self.target_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return frame

    def _note_fps(self, now: float, idx: int) -> None:
        if self._last_t is not None:
            dt = max(now - self._last_t, 1e-3)
            self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * (1.0 / dt)
            if idx == 90:                 # ~3s in at 30fps
                self._check_fps()
        self._last_t = now

    def _reader_loop(self) -> None:
        """Continuously reads the device and publishes the latest frame plus
        fires fast hooks, decoupling device capture rate from however fast
        the main (heavy) loop can consume frames."""
        idx = 0
        while not self._reader_stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = self._prep_frame(frame)
            now = time.time()
            with self._latest_lock:
                self._latest_frame = frame
                self._latest_ts = now
                self._latest_index = idx
            for hook in self._fast_hooks:
                try:
                    hook(frame, now)
                except Exception:  # noqa: BLE001
                    pass
            self._note_fps(now, idx)
            idx += 1

    def _frames_threaded(self) -> Iterator[FrameContext]:
        """Webcam path: a reader thread owns the device; yield whatever frame
        is latest, dropping any the main loop couldn't keep up with."""
        if self._reader_thread is None:
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True, name="camera-reader")
            self._reader_thread.start()
        last_seen = -1
        out_idx = 0
        while True:
            while True:
                with self._latest_lock:
                    idx, frame, ts = self._latest_index, self._latest_frame, self._latest_ts
                if idx != last_seen and frame is not None:
                    last_seen = idx
                    break
                if not self._reader_thread.is_alive():
                    return
                time.sleep(0.001)
            yield FrameContext(frame=frame, timestamp=ts, frame_index=out_idx,
                               fps=self._fps_smooth)
            out_idx += 1

    def _frames_sync(self) -> Iterator[FrameContext]:
        """File/URL path: read exactly once per yielded frame (deterministic,
        no dropped frames — matters for reproducible clip processing)."""
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = self._prep_frame(frame)
            now = time.time()
            self._note_fps(now, idx)
            yield FrameContext(frame=frame, timestamp=now, frame_index=idx,
                               fps=self._fps_smooth)
            idx += 1

    def frames(self) -> Iterator[FrameContext]:
        """Yield a FrameContext per captured frame."""
        if self.cap is None:
            self.open()
        if self._is_webcam():
            yield from self._frames_threaded()
        else:
            yield from self._frames_sync()

    def release(self) -> None:
        """Release the capture device."""
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
