"""RealSense D435i frame source: aligned color + depth + IMU ego-motion.

Mirrors the `core.camera.Camera` interface the pipeline relies on
(`frames()`, `register_fast_hook()`, `switch_to()`, `current_fps`,
`release()`) so `Pipeline` and `main.py` treat both interchangeably —
see `core.camera_factory.make_camera` for how a source string picks the
backend. Differences from the UVC path that matter:

- **Depth is aligned to color** (`rs.align`), so `FrameContext.depth[y, x]`
  is the distance of the *color* pixel (y, x). Depth-aware modules index it
  directly with the same landmark pixel coordinates they already use.
- **Intrinsics are copied into the SDK-free `core.context.Intrinsics`**,
  rescaled to the delivered (possibly downsized) frame, so metric helpers
  (`depth_m`/`deproject`/`mm_per_px`) and their tests never import
  pyrealsense2.
- **No exposure/WB locking logic**: USB 3 uncompressed RGB doesn't fight
  the MJPEG/auto-exposure battles the OV2735 needs (`core/camera.py`'s
  docstring); the SDK's defaults are stable enough, and rPPG's per-backend
  gates handle residual drift.
- **IMU**: a second, callback-driven pipeline streams the gyro and keeps a
  smoothed angular-speed magnitude in `FrameContext.ego_motion` (rad/s).
  On a stationary G1 this stays ~0; when the robot moves it lets modules
  (or a future shared confidence factor) deweight frequency-domain vitals.
- **IR emitter flag**: the D435i's projector speckle can contaminate RGB
  skin pixels in some modes; `emitter=False` trades depth quality for
  clean color capture.

`pyrealsense2` is imported lazily (optional extra, see
requirements-realsense.txt) so every RGB-only install keeps working.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .context import FrameContext, Intrinsics

REALSENSE_SOURCES = ("realsense", "rs", "d435i")


def is_realsense_source(source) -> bool:
    """True if a --source/--alt-source value names the RealSense backend."""
    return isinstance(source, str) and source.strip().lower() in REALSENSE_SOURCES


class RealSenseCamera:
    """D435i frame source yielding color frames with aligned depth + IMU."""

    def __init__(self, source: str = "realsense", target_width: int = 960,
                 request_fps: float = 30.0, request_size: tuple = (1280, 720),
                 emitter: bool = True, **_unused):
        # **_unused swallows UVC-only camera_opts (lock/exposure/gain...) so
        # main.py can pass one opts dict to whichever backend gets built.
        self.source = source
        self.target_width = target_width
        self.request_fps = request_fps
        self.request_size = request_size
        self.emitter = emitter
        self._rs = None                      # pyrealsense2 module, set in open()
        self._pipe = None                    # color+depth rs.pipeline
        self._align = None
        self._motion_pipe = None             # callback-driven IMU rs.pipeline
        self._depth_scale = 0.001
        self._depth_available = True
        self._intrinsics: Optional[Intrinsics] = None
        self._scale = 1.0                    # delivered-size / native-size factor
        self._ego_motion = 0.0               # EMA of |gyro| rad/s, IMU-thread written
        self._fps_smooth = float(request_fps)
        self._last_t: Optional[float] = None
        # Reader-thread state, mirroring core.camera.Camera.
        self._fast_hooks: list[Callable[[np.ndarray, float], None]] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest: Optional[tuple] = None   # (frame, depth, ts, index)
        self._pending: Optional[tuple] = None

    @property
    def current_fps(self) -> float:
        """Smoothed delivered fps, readable from other threads/modules."""
        return self._fps_smooth

    def register_fast_hook(self, hook: Callable[[np.ndarray, float], None]) -> None:
        """Register `hook(frame, timestamp)` to run on the reader thread for
        every captured color frame — same contract as `Camera`, so the
        vitals fast path works unchanged on the D435i."""
        self._fast_hooks.append(hook)

    def switch_to(self, source, opts: dict | None = None) -> None:
        """Queue a source switch; realsense->realsense reopens in place,
        anything else is handled a level up by `SwitchableCamera`, which
        polls `take_pending()` and swaps the whole backend."""
        with self._latest_lock:
            self._pending = (source, opts or {})

    def take_pending(self) -> Optional[tuple]:
        """Atomically fetch and clear any queued switch request."""
        with self._latest_lock:
            pend, self._pending = self._pending, None
        return pend

    def open(self) -> None:
        """Start the color+depth pipeline (and best-effort IMU pipeline).

        Falls back through progressively lower resolutions if the requested
        (or higher) combinations fail, so the D435i connects even on USB
        bandwidth-limited hubs or with firmware quirks.  Depth is dropped
        last — only if no color+depth pair works at all.
        """
        import pyrealsense2 as rs  # lazy: optional dependency
        self._rs = rs

        requested_w, requested_h = self.request_size
        requested_fps = int(self.request_fps)

        # Build a fallback chain: query the device for what it actually
        # supports, then walk from highest resolution downward.
        dev = rs.context().query_devices()
        sensor = (dev[0].first_depth_sensor() if len(dev) else None)
        if sensor is None:
            raise RuntimeError("No RealSense device found")

        color_profiles = set()
        depth_profiles = set()
        for p in sensor.get_stream_profiles():
            if not p.is_video_stream_profile():
                continue
            vp = p.as_video_stream_profile()
            key = (vp.width(), vp.height(), vp.fps())
            if p.stream_type() == rs.stream.color and p.format() == rs.format.bgr8:
                color_profiles.add(key)
            elif p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
                depth_profiles.add(key)

        # Also query the RGB sensor for color profiles (D435i has separate sensors)
        rgb_sensor = None
        for s in dev[0].sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                rgb_sensor = s
                break
        if rgb_sensor is not None:
            for p in rgb_sensor.get_stream_profiles():
                if not p.is_video_stream_profile():
                    continue
                vp = p.as_video_stream_profile()
                if p.stream_type() == rs.stream.color and p.format() == rs.format.bgr8:
                    color_profiles.add((vp.width(), vp.height(), vp.fps()))

        # Candidates: best resolution first, prefer requested fps, then 30, then 60
        all_res = {(w, h) for w, h, _ in color_profiles | depth_profiles}
        target_resolutions = []
        # Always try the requested resolution first
        target_resolutions.append((requested_w, requested_h))
        # Then step down through common D435i resolutions
        for res in sorted(all_res, key=lambda r: r[0] * r[1], reverse=True):
            if res != (requested_w, requested_h) and res not in target_resolutions:
                target_resolutions.append(res)

        fps_preference = [requested_fps, 30, 60, 90, 15, 6]

        # Build candidate (color, depth) pairs
        candidates = []
        for w, h in target_resolutions:
            for fps in fps_preference:
                color_ok = (w, h, fps) in color_profiles
                depth_ok = (w, h, fps) in depth_profiles
                if color_ok and depth_ok:
                    candidates.append(("color+depth", w, h, fps))
                elif color_ok:
                    candidates.append(("color-only", w, h, fps))

        if not candidates:
            raise RuntimeError(
                "RealSense device has no compatible color stream profiles")

        self._pipe = rs.pipeline()
        last_err = None
        for mode, w, h, fps in candidates:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            if mode == "color+depth":
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            try:
                profile = self._pipe.start(cfg)
                self._depth_available = (mode == "color+depth")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        else:
            self._pipe = None
            raise RuntimeError(
                f"Cannot open RealSense device: none of the "
                f"{len(candidates)} tried configurations worked "
                f"(last error: {last_err})")

        self._align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        if not self.emitter:
            try:
                depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
                print("[realsense] IR emitter disabled (clean RGB, weaker depth)")
            except Exception:  # noqa: BLE001
                pass
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self._scale = (self.target_width / intr.width
                       if intr.width > self.target_width else 1.0)
        s = self._scale
        self._intrinsics = Intrinsics(fx=intr.fx * s, fy=intr.fy * s,
                                       ppx=intr.ppx * s, ppy=intr.ppy * s)
        self._start_motion_pipe(rs)
        depth_tag = "color+depth" if self._depth_available else "color-only"
        print(f"[realsense] {depth_tag} {intr.width}x{intr.height}@{fps}fps, "
              f"depth_scale={self._depth_scale:.4f} m/unit, "
              f"delivered_width={int(intr.width * s)}")

    def _start_motion_pipe(self, rs) -> None:
        """Best-effort gyro stream -> smoothed `ego_motion` magnitude. The
        D435i's motion streams live on a separate sensor, so a second
        callback-driven pipeline keeps the frame loop simple; devices or
        firmwares without an IMU just leave ego_motion at 0.0."""
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.gyro)
            self._motion_pipe = rs.pipeline()
            self._motion_pipe.start(cfg, self._on_motion_frame)
        except Exception as e:  # noqa: BLE001
            self._motion_pipe = None
            print(f"[realsense] IMU unavailable ({e}); ego_motion stays 0")

    def _on_motion_frame(self, frame) -> None:
        """IMU callback thread: EMA the gyro magnitude. A single smoothed
        float is enough for gating — modules only ask 'is the camera
        moving', not 'where did it go'."""
        try:
            m = frame.as_motion_frame()
            if not m:
                return
            g = m.get_motion_data()
            mag = float(np.sqrt(g.x * g.x + g.y * g.y + g.z * g.z))
            self._ego_motion = 0.9 * self._ego_motion + 0.1 * mag
        except Exception:  # noqa: BLE001
            pass

    def _note_fps(self, now: float) -> None:
        if self._last_t is not None:
            dt = max(now - self._last_t, 1e-3)
            self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * (1.0 / dt)
        self._last_t = now

    def _grab(self) -> Optional[tuple]:
        """Blocking read of one aligned (color, depth) pair, resized to
        `target_width` together (nearest-neighbor for depth: interpolating
        millimeter values across object boundaries invents phantom
        surfaces).  When depth is unavailable, returns ``None`` for the
        depth array."""
        frames = self._pipe.wait_for_frames(timeout_ms=5000)
        frames = self._align.process(frames)
        color = frames.get_color_frame()
        if not color:
            return None
        depth = frames.get_depth_frame() if self._depth_available else None
        frame = np.asanyarray(color.get_data())
        dep = np.asanyarray(depth.get_data()) if depth is not None else None
        if self._scale < 1.0:
            frame = cv2.resize(frame, None, fx=self._scale, fy=self._scale)
            if dep is not None:
                dep = cv2.resize(dep, (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        return frame, dep

    def _start_reader(self) -> None:
        """Start the background reader thread that owns the device."""
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="realsense-reader")
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """Reader thread: owns the device, publishes the latest pair, fires
        fast hooks — the same decoupling `Camera._reader_loop` does so the
        heavy detection loop can't starve the vitals fast path."""
        idx = 0
        while not self._reader_stop.is_set():
            try:
                pair = self._grab()
            except Exception:  # noqa: BLE001
                break
            if pair is None:
                continue
            frame, dep = pair
            now = time.time()
            with self._latest_lock:
                self._latest = (frame, dep, now, idx)
            for hook in self._fast_hooks:
                try:
                    hook(frame, now)
                except Exception:  # noqa: BLE001
                    pass
            self._note_fps(now)
            idx += 1

    def frames(self) -> Iterator[FrameContext]:
        """Yield the latest FrameContext (with depth/intrinsics/ego_motion),
        dropping frames the consumer can't keep up with."""
        if self._pipe is None:
            self.open()
        if self._reader_thread is None:
            self._start_reader()
        last_seen = -1
        out_idx = 0
        while True:
            with self._latest_lock:
                latest, pending = self._latest, self._pending
            if pending is not None:
                # realsense->realsense: reopen here; cross-backend switches
                # are SwitchableCamera's job — return so it can swap us out.
                if is_realsense_source(pending[0]):
                    self.take_pending()
                    self.release()
                    self.open()
                    self._start_reader()
                    last_seen = -1
                    continue
                return
            if latest is None or latest[3] == last_seen:
                if not self._reader_thread.is_alive():
                    return
                time.sleep(0.001)
                continue
            frame, dep, ts, last_seen = latest
            yield FrameContext(frame=frame, timestamp=ts, frame_index=out_idx,
                               fps=self._fps_smooth, depth=dep,
                               depth_scale=self._depth_scale,
                               intrinsics=self._intrinsics,
                               ego_motion=self._ego_motion)
            out_idx += 1

    def release(self) -> None:
        """Stop threads and both pipelines."""
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        for pipe_attr in ("_pipe", "_motion_pipe"):
            pipe = getattr(self, pipe_attr)
            if pipe is not None:
                try:
                    pipe.stop()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, pipe_attr, None)
        with self._latest_lock:
            self._latest = None
