"""Pipeline: capture -> shared extractors -> scheduled modules -> aggregator.

Vitals fast path: on a live webcam, the reader thread in `core.camera.Camera`
can deliver frames faster than the heavy per-frame detection loop below can
process them (32 modules + MediaPipe). Frequency-domain vitals (heart_rate)
need every captured frame, not just the ones the heavy loop gets to, so any
scheduled module exposing `fast_update(ctx)` is fed directly from the
camera's reader thread via a registered fast hook — reusing the most
recently detected face geometry (landmarks/bbox are normalized/frame-size
coordinates, so they still apply to a newer same-size frame) with a crop
freshly cut from the new frame's pixels. The heavy loop still calls the
module's normal `process()`, which then only reads out the accumulated
state instead of re-feeding it (see modules/heart_rate.py).

Staleness guard: the heavy loop can be slow (many `interval=0.0` modules run
every tick, including MediaPipe face+pose), so the published face geometry
can go stale for many consecutive fast-path frames. If a real person moves
during that window, reusing a frozen bbox silently samples the wrong ROI —
and the wrong ROI does NOT get caught by the backends' own motion/jitter
gates, because those gates compare the reused (frozen) bbox to itself and
would always call it "stable" even while the underlying scene has moved.
So `_fast_hook` (a) refuses to feed modules once the published geometry is
older than `max_staleness`, degrading gracefully back to heavy-loop-only
feeding instead of corrupting the buffers, and (b) computes TRUE current
motion energy itself every fast-path frame (mirroring extractors/motion.py)
instead of reusing the heavy loop's own possibly-stale motion reading, so
motion-based gates in the backends see real, current motion.

Diagnostics: whether the staleness guard above is actually starving the
vitals buffers (heavy loop too slow to republish fresh geometry often
enough) or not is otherwise invisible from the outside. Run with
`--debug-modules pipeline` (or `pipeline,openrppg` to also see backend-level
accept/reject counts) to get a periodic summary of how often the heavy loop
republishes a fresh face vs. how many fast-path frames were fed/rejected.
"""
from __future__ import annotations

import threading
import traceback

import cv2
import numpy as np

from .camera import Camera
from .context import FaceData, FrameContext
from .debug import enabled as debug_enabled, log as debug_log
from .events import Result
from .scheduler import Scheduler


class Pipeline:
    """Orchestrates capture -> extractors -> scheduler -> aggregator -> advisor each frame."""
    def __init__(self, camera: Camera, extractors: list, scheduler: Scheduler,
                 aggregator, advisor_engine=None, max_staleness: float = 0.25):
        self.camera = camera
        self.extractors = extractors
        self.scheduler = scheduler
        self.aggregator = aggregator
        self.advisor_engine = advisor_engine
        self.max_staleness = max_staleness   # seconds; see module docstring
        self._face_lock = threading.Lock()
        self._latest_face: FaceData | None = None
        self._latest_face_ts = 0.0
        self._motion_prev: np.ndarray | None = None   # reader-thread-only state
        # Diagnostic counters (only accumulated when --debug-modules pipeline
        # is set, see _maybe_log_diag); reader-thread-only, no lock needed.
        self._diag_fed = 0
        self._diag_stale = 0
        self._diag_no_face = 0
        self._diag_heavy_publishes = 0
        self._diag_window_start = 0.0
        fast_modules = [m for m in scheduler.modules if hasattr(m, "fast_update")]
        register = getattr(camera, "register_fast_hook", None)
        if fast_modules and register is not None:
            self._fast_modules = fast_modules
            register(self._fast_hook)
        else:
            self._fast_modules = []

    def _motion_energy(self, frame: np.ndarray) -> float:
        """True current frame-difference motion energy, computed on the
        reader thread itself (same algorithm as extractors/motion.py) with
        its own `_motion_prev` state — never touched by any other thread, so
        it needs no lock — instead of reusing the heavy loop's possibly-stale
        reading."""
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        energy = 0.0
        if self._motion_prev is not None:
            energy = float(np.mean(cv2.absdiff(gray, self._motion_prev)))
        self._motion_prev = gray
        return energy

    def _maybe_log_diag(self, now: float) -> None:
        """Periodically summarize how the staleness guard is behaving —
        reveals whether it's starving the vitals buffers of samples (heavy
        loop republishing too rarely) vs. feeding them plentifully. Reader-
        thread-only; `_diag_heavy_publishes` is a plain int bumped from the
        main thread too, so it can occasionally lose an increment under the
        GIL, which is fine for a coarse diagnostic rate."""
        if self._diag_window_start == 0.0:
            self._diag_window_start = now
            return
        elapsed = now - self._diag_window_start
        if elapsed < 5.0:
            return
        total = self._diag_fed + self._diag_stale + self._diag_no_face
        debug_log("pipeline", (
            f"fast_hz={total/elapsed:.1f} fed={self._diag_fed} "
            f"stale_rejected={self._diag_stale} no_face={self._diag_no_face} "
            f"heavy_face_publish_hz={self._diag_heavy_publishes/elapsed:.2f} "
            f"max_staleness={self.max_staleness:.2f}s"))
        self._diag_fed = self._diag_stale = self._diag_no_face = 0
        self._diag_heavy_publishes = 0
        self._diag_window_start = now

    def _fast_hook(self, frame, ts: float) -> None:
        """Runs on the camera's reader thread for every raw captured frame.
        Kept light: no MediaPipe, just re-cut the last known face bbox from
        the new frame's pixels so vitals sample real, current color data at
        the camera's full rate — but only while that geometry is still fresh
        enough to trust (see module docstring)."""
        motion = self._motion_energy(frame)
        with self._face_lock:
            face, face_ts = self._latest_face, self._latest_face_ts
        diag = debug_enabled("pipeline")
        if face is None:
            if diag:
                self._diag_no_face += 1
                self._maybe_log_diag(ts)
            return
        if (ts - face_ts) > self.max_staleness:
            if diag:
                self._diag_stale += 1
                self._maybe_log_diag(ts)
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face.bbox
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        fresh_face = FaceData(landmarks=face.landmarks, bbox=(x1, y1, x2, y2),
                              crop=crop, has_iris=face.has_iris)
        fast_ctx = FrameContext(frame=frame, timestamp=ts, frame_index=-1,
                                fps=self.camera.current_fps, face=fresh_face,
                                motion_energy=motion)
        if diag:
            self._diag_fed += 1
            self._maybe_log_diag(ts)
        for module in self._fast_modules:
            try:
                module.fast_update(fast_ctx)
            except Exception:  # noqa: BLE001
                print(f"[pipeline] module '{module.name}' fast_update raised:")
                traceback.print_exc()

    def process_frame(self, ctx: FrameContext) -> list[Result]:
        """Run extractors, modules, and the advisor for one frame."""
        for ex in self.extractors:
            ex.extract(ctx)
        if self._fast_modules and ctx.face is not None:
            with self._face_lock:
                self._latest_face = ctx.face
                self._latest_face_ts = ctx.timestamp
            if debug_enabled("pipeline"):
                self._diag_heavy_publishes += 1
        results = self.scheduler.tick(ctx)
        self.aggregator.ingest(results)
        if self.advisor_engine is not None:
            advice = self.advisor_engine.evaluate(self.aggregator.snapshot())
            if advice:
                self.aggregator.ingest(advice)
                results.extend(advice)
        return results

    def run(self, on_frame=None, max_frames: int | None = None) -> None:
        """on_frame(ctx, results) -> bool; return False to stop."""
        try:
            for ctx in self.camera.frames():
                results = self.process_frame(ctx)
                if on_frame is not None and on_frame(ctx, results) is False:
                    break
                if max_frames is not None and ctx.frame_index + 1 >= max_frames:
                    break
        finally:
            self.camera.release()
            for ex in self.extractors:
                close = getattr(ex, "close", None)
                if close:
                    close()
            for module in self.scheduler.modules:
                close = getattr(module, "close", None)
                if close:
                    close()
