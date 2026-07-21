"""Pipeline: capture -> shared extractors -> scheduled modules -> aggregator.

Vitals fast path: on a live webcam, the reader thread in `core.camera.Camera`
can deliver frames faster than the heavy per-frame detection loop below can
process them (32 modules + MediaPipe). The reader only publishes references
to a bounded sampler queue; optical flow and `fast_update(ctx)` run on a
dedicated thread so native/model work can never stall physical capture. Any
scheduled module exposing `fast_update(ctx)` is fed there — reusing the most
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
import time
import traceback
import uuid
import copy
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from .camera import Camera
from .context import FaceData, FrameContext, Intrinsics, PoseData
from .debug import enabled as debug_enabled, log as debug_log
from .events import Result
from .fast_face_tracker import FastFaceTracker
from .runtime_metrics import RuntimeMetrics
from .scheduler import Scheduler
from .showcase import ShowcaseGate
from .tracking import AnonymousTracker
from storage.event_store import EventStore
from storage.history_store import HistoryStore


class Pipeline:
    """Orchestrates capture -> extractors -> scheduler -> aggregator -> advisor each frame."""
    def __init__(self, camera: Camera, extractors: list, scheduler: Scheduler,
                 aggregator, advisor_engine=None, max_staleness: float = 0.25,
                 showcase_gate: ShowcaseGate | None = None,
                 camera_location: str | None = None,
                 tracking_enabled: bool = False,
                 runtime_metrics: RuntimeMetrics | None = None,
                 background_analysis: bool = False,
                  fast_path_mode: str = "strict",
                  extended_staleness: float = 0.75,
                  tracker_max_anchor_age: float = 1.5,
                  analysis_width: int = 960,
                  quality_profile: str = "maximum",
                  runtime_config: dict | None = None):
        self.camera = camera
        self.extractors = extractors
        self.scheduler = scheduler
        self.aggregator = aggregator
        self.advisor_engine = advisor_engine
        self.max_staleness = max_staleness   # seconds; see module docstring
        self.showcase_gate = showcase_gate
        self.event_store = EventStore.instance()
        self.tracker = AnonymousTracker()
        self.camera_location = camera_location
        self.tracking_enabled = tracking_enabled
        self.runtime_metrics = runtime_metrics or RuntimeMetrics()
        self.background_analysis = bool(background_analysis)
        self.analysis_width = max(160, int(analysis_width))
        quality = str(quality_profile or "maximum").strip().lower()
        if quality not in {"maximum", "balanced", "realtime"}:
            raise ValueError(f"unknown quality profile: {quality_profile!r}")
        self.quality_profile = quality
        runtime_config = runtime_config or {}
        default_hz = {"maximum": 12.0, "balanced": 10.0, "realtime": 8.0}[quality]
        self._fast_target_hz = max(6.5, float(
            runtime_config.get("fast_sampler_target_hz", default_hz)))
        caps = runtime_config.get("detail_roi_caps") or {}
        self.detail_roi_cap = int(caps.get(quality, {
            "maximum": 640, "balanced": 480, "realtime": 320}[quality]))
        budgets = runtime_config.get("background_budgets_ms") or {}
        self._background_budget_ms = float(budgets.get(quality, {
            "maximum": 350.0, "balanced": 225.0, "realtime": 150.0}[quality]))
        self._fast_last_processed_ts = 0.0
        self.runtime_metrics.set_quality_state(
            quality, quality, self.detail_roi_cap)
        mode = str(fast_path_mode).strip().lower()
        if mode not in ("strict", "extended", "tracked"):
            raise ValueError(f"unknown vitals fast-path mode: {fast_path_mode!r}")
        self.fast_path_mode = mode
        self.extended_staleness = max(self.max_staleness, float(extended_staleness))
        self._fast_motion_threshold = float(
            getattr(showcase_gate, "max_motion", 12.0))
        self._face_tracker = FastFaceTracker(max_anchor_age=tracker_max_anchor_age)
        self._fast_rejection_reason: str | None = None
        self._face_lock = threading.Lock()
        self._vitals_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._latest_face: FaceData | None = None
        self._latest_face_ts = 0.0
        self._motion_prev: np.ndarray | None = None   # sampler-thread-only state
        self._latest_fast_motion = 0.0
        self._latest_fast_motion_at = 0.0
        self._fast_cv = threading.Condition()
        self._fast_pending: deque[tuple[np.ndarray, float]] = deque(maxlen=2)
        self._fast_stop = False
        self._fast_thread: threading.Thread | None = None
        self._pending_face_anchor: tuple[np.ndarray, FaceData, float] | None = None
        self._fast_capture_ready = showcase_gate is None
        self._capture_was_ready = showcase_gate is None
        self._capture_blocked_since: float | None = None
        self._capture_reset_done = False
        self._showcase_state = {
            "enabled": showcase_gate is not None,
            "capture_ready": showcase_gate is None,
            "zone": None,
            "guidance": ("Waiting for positioning assessment"
                         if showcase_gate is not None else "Capture gate disabled"),
        }
        # Diagnostic counters (only accumulated when --debug-modules pipeline
        # is set, see _maybe_log_diag); sampler-thread-only, no lock needed.
        self._diag_fed = 0
        self._diag_stale = 0
        self._diag_no_face = 0
        self._diag_heavy_publishes = 0
        self._diag_window_start = 0.0
        fast_modules = [m for m in scheduler.modules if hasattr(m, "fast_update")]
        self._critical_scheduler = Scheduler(fast_modules)
        self._background_scheduler = Scheduler(
            [m for m in scheduler.modules if m not in fast_modules])
        self._critical_extractors = [
            ex for ex in extractors
            if ex.__class__.__name__ in ("FaceExtractor", "MotionExtractor")]
        self._face_extractor = next(
            (ex for ex in self._critical_extractors
             if ex.__class__.__name__ == "FaceExtractor"), None)
        self._face_configured_width = int(
            getattr(self._face_extractor, "input_width", 640)) if self._face_extractor else 640
        if self.background_analysis and self._face_extractor is not None:
            self._critical_extractors = [ex for ex in self._critical_extractors
                                         if ex is not self._face_extractor]
        self._face_cv = threading.Condition()
        self._face_pending: tuple[int, FrameContext] | None = None
        self._face_generation = 0
        self._face_result: tuple[float, FaceData | None, list, bool, np.ndarray] | None = None
        self._face_stop = False
        self._face_thread: threading.Thread | None = None
        self._face_worker_detached = False
        self._face_last_consumed = 0.0
        self._background_extractors = [
            ex for ex in extractors
            if ex not in self._critical_extractors and ex is not self._face_extractor]
        self._background_cv = threading.Condition()
        self._background_pending: FrameContext | None = None
        self._background_busy = False
        self._background_done: deque[list[Result]] = deque()
        self._background_stop = False
        self._background_thread: threading.Thread | None = None
        self._adaptive_level = 0
        self._adaptive_overloads = 0
        self._adaptive_recovery = 0
        self._advisor_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="advisor") if advisor_engine is not None else None
        self._advisor_pending: Future | None = None
        self._advisor_last_submit = -1e9
        self._modules_started = False
        register = getattr(camera, "register_fast_hook", None)
        if fast_modules and register is not None:
            self._fast_modules = fast_modules
            register(self._enqueue_fast_frame)
        else:
            self._fast_modules = []

    def _start_fast_sampler(self) -> None:
        """Start the bounded full-resolution vitals sampler once."""
        if not self._fast_modules:
            return
        with self._fast_cv:
            if self._fast_thread is not None and self._fast_thread.is_alive():
                return
            self._fast_stop = False
            self._fast_thread = threading.Thread(
                target=self._fast_sampler_loop, daemon=True, name="vitals-sampler")
            self._fast_thread.start()

    def _enqueue_fast_frame(self, frame: np.ndarray, ts: float) -> None:
        """Camera-thread callback: publish immediately and drop oldest on load."""
        self._start_fast_sampler()
        with self._fast_cv:
            if self._fast_stop:
                return
            if len(self._fast_pending) == self._fast_pending.maxlen:
                self._fast_pending.popleft()
                self.runtime_metrics.note_fast_sampler_drop()
            self._fast_pending.append((frame, float(ts)))
            self._fast_cv.notify()

    def _fast_sampler_loop(self) -> None:
        while True:
            with self._fast_cv:
                while not self._fast_pending and not self._fast_stop:
                    self._fast_cv.wait()
                if self._fast_stop:
                    return
                frame, ts = self._fast_pending.popleft()
                if self._fast_pending:
                    skipped = len(self._fast_pending)
                    frame, ts = self._fast_pending.pop()
                    self._fast_pending.clear()
                    self.runtime_metrics.note_fast_sampler_coalesced(skipped)
                anchor = self._pending_face_anchor
                self._pending_face_anchor = None
            if (self.background_analysis and self._fast_last_processed_ts and
                    ts - self._fast_last_processed_ts < 1.0 / self._fast_target_hz):
                self.runtime_metrics.note_fast_sampler_coalesced()
                continue
            self._fast_last_processed_ts = ts
            started = time.perf_counter()
            if anchor is not None and self.fast_path_mode == "tracked":
                anchor_frame, face, anchor_ts = anchor
                self._face_tracker.seed(anchor_frame, face, anchor_ts)
            self._fast_hook(frame, ts)
            self.runtime_metrics.note_fast_sampler(
                ts, (time.perf_counter() - started) * 1000.0)

    def _stop_fast_sampler(self) -> None:
        with self._fast_cv:
            self._fast_stop = True
            self._fast_pending.clear()
            self._pending_face_anchor = None
            self._fast_cv.notify_all()
        thread = self._fast_thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                print("[pipeline] vitals sampler did not stop within 3s; continuing shutdown")
        self._fast_thread = None

    def _start_face_worker(self) -> None:
        if not self.background_analysis or self._face_extractor is None:
            return
        with self._face_cv:
            if self._face_thread is not None and self._face_thread.is_alive():
                return
            self._face_stop = False
            self._face_worker_detached = False
            self._face_thread = threading.Thread(
                target=self._face_worker_loop, daemon=True, name="face-authority")
            self._face_thread.start()
            self.runtime_metrics.set_face_worker_alive(True)

    def _face_worker_loop(self) -> None:
        while True:
            with self._face_cv:
                while self._face_pending is None and not self._face_stop:
                    self._face_cv.wait()
                if self._face_stop:
                    return
                pending = self._face_pending
                self._face_pending = None
            if pending is None:
                continue
            generation, work = pending
            started = time.perf_counter()
            try:
                fallback_level = {"maximum": 3, "balanced": 2,
                                  "realtime": 1}[self.quality_profile]
                self._face_extractor.input_width = (
                    min(self._face_configured_width, 480)
                    if self._adaptive_level >= fallback_level
                    else self._face_configured_width)
                self._face_extractor.extract(work)
                result = (work.timestamp, work.face,
                          list(work.extras.get("faces", [])),
                          bool(work.person_present), work.frame)
                with self._face_cv:
                    if generation == self._face_generation:
                        self._face_result = result
            except Exception:  # noqa: BLE001
                print("[pipeline] authoritative face worker raised:")
                traceback.print_exc()
            finally:
                self.runtime_metrics.note_face_worker(
                    (time.perf_counter() - started) * 1000.0,
                    int(getattr(self._face_extractor, "input_width", 0)))

    def _poll_and_submit_face(self, ctx: FrameContext) -> None:
        """Apply the newest bounded face result, then publish the current frame."""
        with self._face_cv:
            result = self._face_result
            if result is not None:
                ts, face, faces, present, _frame = result
                age = max(0.0, ctx.timestamp - ts)
                if age <= max(1.5, self.extended_staleness):
                    if face is not None:
                        x1, y1, x2, y2 = face.bbox
                        current_face = FaceData(
                            np.array(face.landmarks, copy=True), face.bbox,
                            ctx.frame[y1:y2, x1:x2], face.has_iris)
                        ctx.face = current_face
                    ctx.extras["faces"] = faces
                    ctx.person_present = bool(present or face is not None)
                    ctx.extras["face_authoritative_timestamp"] = ts
                    if ts > self._face_last_consumed and face is not None:
                        self._face_last_consumed = ts
                        with self._face_lock:
                            self._latest_face = face
                            self._latest_face_ts = ts
                        if self.fast_path_mode == "tracked":
                            with self._fast_cv:
                                self._pending_face_anchor = (_frame, face, ts)
            if self._face_pending is not None:
                self.runtime_metrics.note_face_worker_drop()
            pending = copy.copy(ctx)
            pending.extras = dict(ctx.extras)
            pending.frame = np.array(ctx.frame, copy=True)
            pending.depth = None
            pending.face = None
            self._face_pending = (self._face_generation, pending)
            self._face_cv.notify()

    def _stop_face_worker(self) -> None:
        with self._face_cv:
            self._face_stop = True
            self._face_pending = None
            self._face_cv.notify_all()
        thread = self._face_thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                self._face_worker_detached = True
                print("[pipeline] face worker did not stop within 2s; continuing shutdown")
        self.runtime_metrics.set_face_worker_alive(
            bool(thread is not None and thread.is_alive()))
        self._face_thread = None

    def _start_background_worker(self) -> None:
        if not self.background_analysis or self._background_thread is not None:
            return
        self._background_stop = False
        self._background_thread = threading.Thread(
            target=self._background_loop, daemon=True, name="detector-analysis")
        self._background_thread.start()

    def _start_modules(self) -> None:
        """Start optional asynchronous module warmups before frame analysis."""
        if self._modules_started:
            return
        self._modules_started = True
        ordered = sorted(self.scheduler.modules,
                         key=lambda module: (0 if module.name == "clothing" else
                                             1 if module.name == "heart_rate" else 2))
        for index, module in enumerate(ordered):
            start = getattr(module, "start", None)
            if start is not None:
                start()
                if index == 0 and module.name == "clothing":
                    # Let PyTorch claim the GPU/import lock before CPU-native
                    # model workers begin their own imports.
                    time.sleep(0.5)

    def preload_modules(self) -> None:
        """Begin idempotent asynchronous model warmups before camera startup."""
        self._start_modules()

    def _stop_background_worker(self) -> None:
        with self._background_cv:
            self._background_stop = True
            self._background_pending = None
            self._background_cv.notify_all()
        if self._background_thread is not None:
            # Extractors and modules own native/threaded resources.  Do not
            # close those underneath an analysis call that is still unwinding.
            self._background_thread.join(timeout=3.0)
            if self._background_thread.is_alive():
                print("[pipeline] background worker did not stop within 3s; continuing shutdown")
            else:
                self._background_thread = None
        if self._advisor_pending is not None:
            self._advisor_pending.cancel()
            self._advisor_pending = None
        if self._advisor_executor is not None:
            self._advisor_executor.shutdown(wait=False, cancel_futures=True)
            self._advisor_executor = None

    def _poll_or_submit_advisor(self, ctx: FrameContext,
                                results: list[Result]) -> list[Result]:
        """Keep history/advice queries off the serial detector worker."""
        advice: list[Result] = []
        pending = self._advisor_pending
        if pending is not None and pending.done():
            # Completion callback already delivered the result batch.
            self._advisor_pending = None
        if (self._advisor_executor is not None and self._advisor_pending is None
                and ctx.timestamp - self._advisor_last_submit >= 1.0):
            snapshot = list(self.aggregator.snapshot())
            current = {(r.subject_id, r.module, r.key): r for r in snapshot}
            current.update({(r.subject_id, r.module, r.key): r for r in results})
            self._advisor_pending = self._advisor_executor.submit(
                self.advisor_engine.evaluate, list(current.values()), ctx.timestamp)
            def deliver(future):
                try:
                    completed = list(future.result() or [])
                except Exception as exc:  # noqa: BLE001
                    print(f"[advisor] background evaluation failed ({type(exc).__name__})")
                    completed = []
                if completed and not self._background_stop:
                    with self._background_cv:
                        self._background_done.append(completed)
            self._advisor_pending.add_done_callback(deliver)
            self._advisor_last_submit = ctx.timestamp
        return advice

    def _background_loop(self) -> None:
        """Run pose and non-vitals modules serially without blocking geometry."""
        while True:
            with self._background_cv:
                while self._background_pending is None and not self._background_stop:
                    self._background_cv.wait()
                if self._background_stop:
                    return
                ctx = self._background_pending
                self._background_pending = None
                self._background_busy = True
            if ctx is None:
                continue
            started = time.perf_counter()
            timings: dict[str, float] = {}
            scaling_started = time.perf_counter()
            ctx = self._copy_for_background(ctx)
            timings["coordinator:background_scaling"] = round(
                (time.perf_counter() - scaling_started) * 1000.0, 2)
            for ex in self._background_extractors:
                if self._background_stop:
                    return
                t0 = time.perf_counter()
                ex.extract(ctx)
                timings[f"extractor:{ex.__class__.__name__}"] = round(
                    (time.perf_counter() - t0) * 1000.0, 2)
            if self._background_stop:
                return
            results = self._background_scheduler.tick(
                ctx, timings=timings, should_stop=lambda: self._background_stop,
                budget_ms=self._background_budget_ms)
            if self._background_stop:
                return
            if self.showcase_gate is not None:
                results = [r for r in results if self.showcase_gate.allow(r.module, ctx)]
            if self.advisor_engine is not None:
                results.extend(self._poll_or_submit_advisor(ctx, results))
            throttled = self._background_scheduler.pop_throttled()
            if throttled:
                self.runtime_metrics.note_background_throttled(throttled)
            latency_ms = (time.perf_counter() - started) * 1000.0
            source_index = int(ctx.extras.get("capture_index", ctx.frame_index))
            self.runtime_metrics.note_analysis(source_index, latency_ms)
            self.runtime_metrics.note_stage_timings(timings)
            with self._background_cv:
                self._background_busy = False
                self._background_done.append(results)

    def _copy_for_background(self, ctx: FrameContext) -> FrameContext:
        """Build a synchronized low-resolution context for passive analysis."""
        cloned = copy.copy(ctx)
        cloned.extras = dict(ctx.extras)
        # Geometry is inferred on bounded frames, but detail detectors can crop
        # high-resolution ROIs from this immutable capture context.
        cloned.extras["_native_detail_context"] = ctx
        cloned.extras["quality_profile"] = self.quality_profile
        cloned.extras["detail_roi_cap"] = self.detail_roi_cap
        h, w = ctx.frame.shape[:2]
        if w <= self.analysis_width:
            return cloned
        scale = self.analysis_width / float(w)
        new_w = self.analysis_width
        new_h = max(1, int(round(h * scale)))
        cloned.frame = cv2.resize(ctx.frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if ctx.depth is not None:
            cloned.depth = cv2.resize(ctx.depth, (new_w, new_h),
                                      interpolation=cv2.INTER_NEAREST)
        if ctx.intrinsics is not None:
            cloned.intrinsics = Intrinsics(
                fx=ctx.intrinsics.fx * scale, fy=ctx.intrinsics.fy * scale,
                ppx=ctx.intrinsics.ppx * scale, ppy=ctx.intrinsics.ppy * scale)
        def scaled_box(box):
            return tuple(int(round(float(value) * scale)) for value in box)
        if ctx.face is not None:
            bbox = scaled_box(ctx.face.bbox)
            x1, y1, x2, y2 = bbox
            cloned.face = FaceData(np.array(ctx.face.landmarks, copy=True), bbox,
                                   cloned.frame[y1:y2, x1:x2], ctx.face.has_iris)
        if ctx.pose is not None:
            cloned.pose = PoseData(np.array(ctx.pose.landmarks, copy=True),
                                   scaled_box(ctx.pose.bbox))
        for key in ("faces", "poses"):
            if isinstance(ctx.extras.get(key), list):
                cloned.extras[key] = [{**item, "bbox": scaled_box(item["bbox"])}
                                      for item in ctx.extras[key] if "bbox" in item]
        cloned.extras["analysis_scale"] = scale
        cloned.extras["capture_shape"] = (h, w)
        for key in tuple(cloned.extras):
            if key.startswith("_arm_skin_cache") or key.startswith("_frame_color_cache"):
                cloned.extras.pop(key, None)
        return cloned

    def _set_adaptive_level(self, overloaded: bool) -> None:
        if overloaded:
            self._adaptive_overloads += 1
            self._adaptive_recovery = 0
            if self._adaptive_overloads >= 3 and self._adaptive_level < 3:
                self._adaptive_level += 1
                self._adaptive_overloads = 0
        else:
            self._adaptive_recovery += 1
            self._adaptive_overloads = 0
            if self._adaptive_recovery >= 30 and self._adaptive_level > 0:
                self._adaptive_level -= 1
                self._adaptive_recovery = 0
        factor = float(2 ** self._adaptive_level)
        self._background_scheduler.set_load_factor(factor)
        state = "normal" if self._adaptive_level == 0 else "throttling"
        self.runtime_metrics.set_scheduler_state(self._adaptive_level, factor, state)
        fallback_level = {"maximum": 3, "balanced": 2,
                          "realtime": 1}[self.quality_profile]
        effective = ("reduced_geometry" if self._adaptive_level >= fallback_level
                     else self.quality_profile)
        self.runtime_metrics.set_quality_state(
            self.quality_profile, effective, self.detail_roi_cap)

    def _submit_background(self, ctx: FrameContext) -> None:
        with self._background_cv:
            if self._background_stop:
                return
            if self._background_pending is not None or self._background_busy:
                self.runtime_metrics.note_background_coalesced()
                self._set_adaptive_level(True)
                return
            self._set_adaptive_level(False)
            # Only copy the mutable context shell here. Pixel/depth resizing is
            # intentionally owned by the background worker.
            pending = copy.copy(ctx)
            pending.extras = dict(ctx.extras)
            self._background_pending = pending
            self.runtime_metrics.note_background_submission()
            self._background_cv.notify()

    def _drain_background(self) -> list[Result]:
        with self._background_cv:
            batches = list(self._background_done)
            self._background_done.clear()
        return [result for batch in batches for result in batch]

    def _reset_fast_modules(self) -> None:
        self._face_tracker.reset("capture reset")
        with self._fast_cv:
            self._fast_pending.clear()
            self._pending_face_anchor = None
        for module in self._fast_modules:
            reset = getattr(module, "reset_capture", None)
            if reset is not None:
                reset()

    def reset_capture_state(self) -> None:
        """Reset subject-bound sampling after an explicit camera/source change."""
        self._reset_fast_modules()
        with self._face_lock:
            self._latest_face = None
            self._latest_face_ts = 0.0
        with self._face_cv:
            self._face_generation += 1
            self._face_pending = None
            self._face_result = None
            self._face_last_consumed = 0.0
        reset_face = getattr(self._face_extractor, "reset", None)
        if reset_face is not None:
            reset_face()
        self._capture_blocked_since = None
        self._capture_reset_done = True

    def _update_capture_gate(self, ctx: FrameContext) -> None:
        state = ctx.extras["showcase"]
        ready = bool(state.get("heart_rate_ready", state["stable"]))
        self._fast_capture_ready = ready
        with self._vitals_lock:
            self._showcase_state = {
                **dict(state), "enabled": True, "capture_ready": ready,
                "guidance": state.get("heart_rate_guidance", state.get("guidance"))}
        if ready:
            self._capture_blocked_since = None
            self._capture_reset_done = False
        else:
            reason = state.get("heart_rate_block_reason", state.get("block_reason"))
            transient = reason in ("motion", "lighting")
            if self._capture_was_ready and self._capture_blocked_since is None:
                self._capture_blocked_since = ctx.timestamp
            should_reset = False
            if not transient:
                should_reset = self._capture_was_ready or self._capture_blocked_since is not None
            elif self._capture_blocked_since is not None:
                grace = float(getattr(self.showcase_gate,
                                      "capture_reset_grace_seconds", 1.0))
                should_reset = ctx.timestamp - self._capture_blocked_since >= grace
            if should_reset and not self._capture_reset_done:
                self._reset_fast_modules()
                self._capture_reset_done = True
        self._capture_was_ready = ready

    def _motion_energy(self, frame: np.ndarray) -> float:
        """True current frame-difference motion energy, computed on the
        sampler thread itself (same algorithm as extractors/motion.py) with
        its own `_motion_prev` state — never touched by any other thread, so
        it needs no lock — instead of reusing the heavy loop's possibly-stale
        reading."""
        h, w = frame.shape[:2]
        # Strided sampling avoids a full-HD interpolation pass on the sampler.
        small = frame[::max(1, h // 120), ::max(1, w // 160)][:120, :160]
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
        """Runs on the dedicated sampler thread for queued captured frames.
        Kept light: no MediaPipe, just re-cut the last known face bbox from
        the new frame's pixels so vitals sample real, current color data at
        the camera's full rate — but only while that geometry is still fresh
        enough to trust (see module docstring)."""
        stage_started = time.perf_counter()
        motion = self._motion_energy(frame)
        self.runtime_metrics.note_stage_timings({
            "fast:motion": (time.perf_counter() - stage_started) * 1000.0})
        with self._vitals_lock:
            self._latest_fast_motion = motion
            self._latest_fast_motion_at = ts
        if not self._fast_capture_ready:
            self._fast_rejection_reason = "capture gate blocked"
            return
        with self._face_lock:
            face, face_ts = self._latest_face, self._latest_face_ts
        diag = debug_enabled("pipeline")
        if face is None:
            self._fast_rejection_reason = "no face anchor"
            self.runtime_metrics.set_tracking_state(False, False)
            self.runtime_metrics.note_fast_path("no_face", ts)
            if diag:
                self._diag_no_face += 1
                self._maybe_log_diag(ts)
            return
        if self.fast_path_mode in ("extended", "tracked") and motion > self._fast_motion_threshold:
            self._fast_rejection_reason = (
                f"motion rejected ({motion:.1f}>{self._fast_motion_threshold:.1f})")
            if self.fast_path_mode == "tracked":
                self._face_tracker.reset("motion rejected")
            self.runtime_metrics.note_fast_path("stale", ts)
            return

        selected_face = face
        if self.fast_path_mode == "tracked":
            stage_started = time.perf_counter()
            tracked = self._face_tracker.track(frame, ts)
            self.runtime_metrics.note_stage_timings({
                "fast:optical_flow": (time.perf_counter() - stage_started) * 1000.0})
            self.runtime_metrics.set_tracking_state(
                tracked.face is not None, face is not None)
            if tracked.face is not None:
                selected_face = tracked.face
            elif (ts - face_ts) > self.max_staleness:
                self._fast_rejection_reason = tracked.reason
                self.runtime_metrics.note_fast_path("stale", ts)
                return
        lease = (self.extended_staleness if self.fast_path_mode == "extended"
                 else self.max_staleness)
        if self.fast_path_mode != "tracked" and (ts - face_ts) > lease:
            self._fast_rejection_reason = "geometry lease expired"
            self.runtime_metrics.note_fast_path("stale", ts)
            if diag:
                self._diag_stale += 1
                self._maybe_log_diag(ts)
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = selected_face.bbox
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        fresh_face = FaceData(landmarks=selected_face.landmarks, bbox=(x1, y1, x2, y2),
                              crop=crop, has_iris=selected_face.has_iris)
        fast_ctx = FrameContext(frame=frame, timestamp=ts, frame_index=-1,
                                fps=self.camera.current_fps, face=fresh_face,
                                motion_energy=motion)
        if diag:
            self._diag_fed += 1
            self._maybe_log_diag(ts)
        for module in self._fast_modules:
            try:
                stage_started = time.perf_counter()
                module.fast_update(fast_ctx)
                self.runtime_metrics.note_stage_timings({
                    f"fast:module:{module.name}":
                    (time.perf_counter() - stage_started) * 1000.0})
            except Exception:  # noqa: BLE001
                print(f"[pipeline] module '{module.name}' fast_update raised:")
                traceback.print_exc()
        self.runtime_metrics.note_fast_path("fed", ts)
        self._fast_rejection_reason = None

    def vitals_diagnostics(self, results: list[Result], now: float | None = None,
                           performance: dict | None = None) -> dict:
        """Compose a private, JSON-safe explanation of heart-rate readiness."""
        now = float(now or time.time())
        with self._vitals_lock:
            showcase = dict(self._showcase_state)
        heart = next((module for module in self.scheduler.modules
                      if module.name == "heart_rate"), None)
        heart_diag = heart.diagnostics() if heart is not None and hasattr(heart, "diagnostics") \
            else {"canonical_source": None, "backends": []}
        backends = heart_diag.get("backends", [])
        available = [item for item in backends if item.get("available")]
        performance = performance or self.runtime_metrics.snapshot(now=now)
        tracker = getattr(self, "_face_tracker", None)
        fast_path = {
            **performance.get("fast_path", {}),
            "mode": getattr(self, "fast_path_mode", "strict"),
            "strict_staleness_seconds": getattr(self, "max_staleness", 0.25),
            "extended_staleness_seconds": getattr(self, "extended_staleness", 0.75),
            "rejection_reason": getattr(self, "_fast_rejection_reason", None),
            "tracker": (tracker.diagnostics(now) if tracker is not None else {
                "active": False, "anchor_age_seconds": None,
                "tracked_frames": 0, "tracked_sample_hz": 0.0,
                "failures": 0, "last_reason": "disabled",
                "max_anchor_age_seconds": 1.5}),
        }

        canonical = max((result for result in results
                         if result.module == "heart_rate" and result.key == "bpm"
                         and result.subject_id == "primary"),
                        key=lambda result: result.timestamp, default=None)
        measurement_age = (max(0.0, now - canonical.timestamp)
                           if canonical is not None else None)
        capture_ready = bool(showcase.get("capture_ready"))
        fresh = bool(canonical is not None and measurement_age is not None
                     and measurement_age <= canonical.ttl and capture_ready)
        bpm = None
        if fresh:
            try:
                candidate = float(canonical.value)
                bpm = candidate if 35.0 <= candidate <= 180.0 else None
            except (TypeError, ValueError):
                bpm = None
            fresh = bpm is not None

        if not capture_ready:
            state = "blocked"
            guidance = str(showcase.get("guidance") or "Capture quality is not ready")
        elif not available:
            state = "unavailable"
            guidance = "No heart-rate backend is available"
        elif fresh:
            state = "ready"
            guidance = "Heart-rate estimate is current"
        elif any("inferr" in str(item.get("status", "")).lower()
                 or item.get("inference_pending") for item in available):
            state = "inferring"
            guidance = "A heart-rate backend is processing the clean sample window"
        else:
            state = "warming_up"
            recent_outcome = fast_path.get("latest_outcome")
            recent_age = fast_path.get("latest_outcome_age_ms")
            if recent_age is not None and recent_age <= 2000 and recent_outcome == "no_face":
                guidance = "No current face geometry; face the camera clearly"
            elif recent_age is not None and recent_age <= 2000 and recent_outcome == "stale":
                guidance = "Face geometry is stale; analysis is not refreshing quickly enough"
            else:
                leader = max(available, key=lambda item: float(item.get("progress", 0.0)))
                buffered = float(leader.get("buffered_seconds", 0.0))
                required = float(leader.get("required_seconds", 0.0))
                samples = int(leader.get("samples", 0))
                required_samples = int(leader.get("required_samples", 0))
                if required > 0 and required_samples > 0:
                    guidance = (f"Collecting clean samples: {samples}/{required_samples} "
                                f"samples, {buffered:.1f}/{required:.1f}s")
                else:
                    guidance = (f"Collecting clean samples: {buffered:.1f}/{required:.1f}s"
                                if required > 0 else "Waiting for a heart-rate result")

        # Backend-specific BPM values are private comparison telemetry. Build
        # them from short-lived Results rather than indefinitely cached model
        # diagnostics so blocked or stale readings cannot look current.
        for backend in backends:
            name = str(backend.get("name") or "backend")
            result_key = "bpm_" + name.replace("-", "_")
            candidate = max((result for result in results
                             if result.module == "heart_rate"
                             and result.key == result_key
                             and result.subject_id == "primary"),
                            key=lambda result: result.timestamp, default=None)
            age = max(0.0, now - candidate.timestamp) if candidate is not None else None
            within_ttl = bool(candidate is not None and age is not None
                              and age <= candidate.ttl)
            candidate_fresh = bool(within_ttl and capture_ready)
            candidate_bpm = None
            if candidate_fresh:
                try:
                    value = float(candidate.value)
                    candidate_bpm = value if 35.0 <= value <= 180.0 else None
                except (TypeError, ValueError):
                    candidate_bpm = None
            candidate_fresh = bool(candidate_fresh and candidate_bpm is not None)

            latest = dict(backend.get("latest") or {})
            rejection = latest.get("rejected_reason")
            status_text = str(backend.get("status") or latest.get("status") or "")
            if not rejection and any(token in status_text.lower() for token in
                                     ("rejected", "low sqi", "failed")):
                rejection = status_text
            if not capture_ready:
                measurement_state = "blocked"
            elif candidate is not None and not within_ttl:
                measurement_state = "stale"
            elif candidate_fresh and rejection:
                measurement_state = "rejected"
            elif candidate_fresh:
                measurement_state = "accepted"
            elif not backend.get("available"):
                measurement_state = "unavailable"
            elif "failed" in status_text.lower():
                measurement_state = "failed"
            elif "inferr" in status_text.lower() or backend.get("inference_pending"):
                measurement_state = "inferring"
            elif rejection:
                measurement_state = "rejected"
            else:
                measurement_state = "warming_up"

            backend["measurement"] = {
                "bpm": round(candidate_bpm, 1) if candidate_fresh else None,
                "confidence": candidate.confidence if candidate_fresh else None,
                "quality": candidate.quality if candidate_fresh else None,
                "measurement_age_seconds": round(age, 2) if candidate_fresh else None,
                "state": measurement_state,
                "accepted": bool(candidate_fresh and not rejection),
                "rejection_reason": str(rejection)[:240] if rejection else None,
            }
            # Cached diagnostics remain useful for progress and error details,
            # but their BPM can outlive the Result TTL. Keep BPM solely in the
            # freshness-checked measurement object above.
            if latest:
                latest.pop("bpm", None)
                latest.pop("raw_bpm", None)
                backend["latest"] = latest

        return {
            "state": state,
            "bpm": round(bpm, 1) if bpm is not None else None,
            "confidence": canonical.confidence if fresh else None,
            "quality": canonical.quality if fresh else None,
            "source": ((heart_diag.get("canonical_source") or canonical.source)
                       if fresh else None),
            "measurement_age_seconds": (round(measurement_age, 2) if fresh else None),
            "capture_ready": capture_ready,
            "zone": showcase.get("zone"),
            "guidance": guidance,
            "fast_path": fast_path,
            "backends": backends,
        }

    def process_frame(self, ctx: FrameContext) -> list[Result]:
        """Run extractors, modules, and the advisor for one frame."""
        started = time.perf_counter()
        timings: dict[str, float] = {}
        if self.background_analysis and self._face_extractor is not None:
            face_started = time.perf_counter()
            self._poll_and_submit_face(ctx)
            timings["coordinator:face_worker_exchange"] = round(
                (time.perf_counter() - face_started) * 1000.0, 2)
        active_extractors = (self._critical_extractors
                             if self.background_analysis else self.extractors)
        for ex in active_extractors:
            if (self.background_analysis and self._fast_modules
                    and ex.__class__.__name__ == "MotionExtractor"):
                with self._vitals_lock:
                    motion, motion_at = (self._latest_fast_motion,
                                         self._latest_fast_motion_at)
                if motion_at and ctx.timestamp - motion_at <= 0.5:
                    ctx.motion_energy = motion
                    timings["extractor:MotionExtractor"] = 0.0
                    continue
            t0 = time.perf_counter()
            ex.extract(ctx)
            timings[f"extractor:{ex.__class__.__name__}"] = round(
                (time.perf_counter() - t0) * 1000.0, 2)
        tracking_started = time.perf_counter()
        boxes = [p["bbox"] for p in ctx.extras.get("poses", [])]
        if not boxes:
            boxes = [f["bbox"] for f in ctx.extras.get("faces", [])]
        ctx.extras["tracks"] = self.tracker.update(boxes, ctx.w, ctx.h, ctx.timestamp)
        if self.tracking_enabled and ctx.extras["tracks"]:
            primary = next((t for t in ctx.extras["tracks"] if t["primary"]), None)
            if primary is None:
                # Do not let an anonymous visitor contaminate primary state while
                # the configured primary is absent or assignment is ambiguous.
                ctx.pose = None
                ctx.face = None
                ctx.person_present = False
            else:
                def overlap(a, b):
                    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                    return ix * iy
                pose_item = max(ctx.extras.get("poses", []),
                                key=lambda p: overlap(primary["bbox"], p["bbox"]), default=None)
                face_item = max(ctx.extras.get("faces", []),
                                key=lambda f: overlap(primary["bbox"], f["bbox"]), default=None)
                if pose_item is not None:
                    ctx.pose = PoseData(pose_item["landmarks"], pose_item["bbox"])
                if face_item is not None:
                    x1, y1, x2, y2 = face_item["bbox"]
                    ctx.face = FaceData(face_item["landmarks"], face_item["bbox"],
                                        ctx.frame[y1:y2, x1:x2],
                                        face_item["landmarks"].shape[0] >= 478)
                ctx.person_present = ctx.pose is not None or ctx.face is not None
        timings["coordinator:anonymous_tracking"] = round(
            (time.perf_counter() - tracking_started) * 1000.0, 2)
        showcase_started = time.perf_counter()
        showcase_results = self.showcase_gate.assess(ctx) if self.showcase_gate else []
        if self.showcase_gate is not None:
            self._update_capture_gate(ctx)
        timings["coordinator:showcase_gate"] = round(
            (time.perf_counter() - showcase_started) * 1000.0, 2)
        publication_started = time.perf_counter()
        if self._fast_modules and ctx.face is not None:
            authoritative_ts = float(
                ctx.extras.get("face_authoritative_timestamp", ctx.timestamp))
            with self._face_lock:
                self._latest_face = ctx.face
                self._latest_face_ts = authoritative_ts
            if self.fast_path_mode == "tracked" and not self.background_analysis:
                with self._fast_cv:
                    self._pending_face_anchor = (ctx.frame, ctx.face, authoritative_ts)
            if debug_enabled("pipeline"):
                self._diag_heavy_publishes += 1
        timings["coordinator:face_publication"] = round(
            (time.perf_counter() - publication_started) * 1000.0, 2)
        if self.background_analysis:
            results = self._critical_scheduler.tick(ctx, timings=timings)
            if self.showcase_gate is not None:
                results = [r for r in results if self.showcase_gate.allow(r.module, ctx)]
            drain_started = time.perf_counter()
            results.extend(self._drain_background())
            timings["coordinator:background_drain"] = round(
                (time.perf_counter() - drain_started) * 1000.0, 2)
            submit_started = time.perf_counter()
            self._submit_background(ctx)
            timings["coordinator:background_submission"] = round(
                (time.perf_counter() - submit_started) * 1000.0, 2)
        else:
            results = self.scheduler.tick(ctx, timings=timings)
            if self.showcase_gate is not None:
                results = [r for r in results if self.showcase_gate.allow(r.module, ctx)]
        results = showcase_results + results
        routing_started = time.perf_counter()
        if self.camera_location:
            for result in results:
                if result.location is None:
                    result.location = self.camera_location
        for result in results:
            if result.persistence.value != "none" and result.correlation_id is None:
                result.correlation_id = uuid.uuid4().hex
        timings["coordinator:result_routing"] = round(
            (time.perf_counter() - routing_started) * 1000.0, 2)
        persistence_started = time.perf_counter()
        for result in results:
            try:
                self.event_store.record_result(result)
            except (TypeError, ValueError) as exc:
                print(f"[events] refused unsafe {result.module}.{result.key}: {exc}")
        timings["coordinator:persistence"] = round(
            (time.perf_counter() - persistence_started) * 1000.0, 2)
        aggregate_started = time.perf_counter()
        self.aggregator.ingest(results)
        timings["coordinator:aggregation"] = round(
            (time.perf_counter() - aggregate_started) * 1000.0, 2)
        if self.advisor_engine is not None and not self.background_analysis:
            advisor_started = time.perf_counter()
            advice = self.advisor_engine.evaluate(self.aggregator.snapshot())
            if advice:
                for result in advice:
                    self.event_store.record_result(result)
                self.aggregator.ingest(advice)
                results.extend(advice)
            timings["coordinator:advisor"] = round(
                (time.perf_counter() - advisor_started) * 1000.0, 2)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.runtime_metrics.note_stage_timings(timings)
        if self.background_analysis:
            geometry_ts = (float(ctx.extras.get(
                "face_authoritative_timestamp", ctx.timestamp))
                if ctx.face is not None else self._latest_face_ts)
            self.runtime_metrics.note_critical(latency_ms, geometry_ts)
        else:
            source_index = int(ctx.extras.get("capture_index", ctx.frame_index))
            self.runtime_metrics.note_analysis(source_index, latency_ms)
        return results

    def run(self, on_frame=None, max_frames: int | None = None) -> None:
        """on_frame(ctx, results) -> bool; return False to stop."""
        self._stop_requested.clear()
        try:
            self._start_modules()
            self._start_background_worker()
            self._start_fast_sampler()
            self._start_face_worker()
            for ctx in self.camera.frames():
                if self._stop_requested.is_set():
                    break
                results = self.process_frame(ctx)
                if on_frame is not None:
                    callback_started = time.perf_counter()
                    keep_running = on_frame(ctx, results)
                    self.runtime_metrics.note_stage_timings({
                        "coordinator:on_frame": round(
                            (time.perf_counter() - callback_started) * 1000.0, 2)})
                    if keep_running is False:
                        break
                if max_frames is not None and ctx.frame_index + 1 >= max_frames:
                    break
        finally:
            self.camera.release()
            self._stop_fast_sampler()
            self._stop_face_worker()
            self._stop_background_worker()
            self.event_store.flush()
            HistoryStore.instance().flush()
            for ex in self.extractors:
                if ex is self._face_extractor and self._face_worker_detached:
                    continue
                close = getattr(ex, "close", None)
                if close:
                    close()
            for module in self.scheduler.modules:
                close = getattr(module, "close", None)
                if close:
                    close()

    def request_stop(self) -> None:
        """Ask an asynchronously running pipeline loop to stop cleanly."""
        self._stop_requested.set()
        with self._background_cv:
            self._background_stop = True
            self._background_pending = None
            self._background_cv.notify_all()
        with self._fast_cv:
            self._fast_stop = True
            self._fast_pending.clear()
            self._fast_cv.notify_all()
        with self._face_cv:
            self._face_stop = True
            self._face_pending = None
            self._face_cv.notify_all()
