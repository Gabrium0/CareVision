"""Thread-safe live performance counters shared by runtime and debug output."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any


class RuntimeMetrics:
    """Rolling JSON-safe metrics plus an actionable runtime health summary."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._capture_fps = 0.0
        self._preview_times: deque[float] = deque(maxlen=120)
        self._overlay_ages_ms: deque[float] = deque(maxlen=120)
        self._analysis_times: deque[float] = deque(maxlen=120)
        self._analysis_latency_ms = 0.0
        self._analysis_latencies: deque[float] = deque(maxlen=240)
        self._critical_times: deque[float] = deque(maxlen=120)
        self._critical_latency_ms = 0.0
        self._critical_latencies: deque[float] = deque(maxlen=240)
        self._geometry_timestamp = 0.0
        self._geometry_ages_ms: deque[float] = deque(maxlen=240)
        self._background_queue_drops = 0
        self._background_submissions = 0
        self._background_throttled = 0
        self._background_coalesced = 0
        self._background_events: deque[tuple[float, str]] = deque(maxlen=600)
        self._stage_latency_ms: dict[str, float] = {}
        self._stage_samples: dict[str, deque[float]] = {}
        self._scheduler: dict[str, Any] = {"level": 0, "factor": 1.0,
                                           "state": "normal"}
        self._latest_frame_ts = 0.0
        self._last_analysis_source_index: int | None = None
        self._skipped_analysis_frames = 0
        self._fast_fed = 0
        self._fast_stale = 0
        self._fast_no_face = 0
        self._fast_sampler_times: deque[float] = deque(maxlen=240)
        self._fast_sampler_latencies: deque[float] = deque(maxlen=240)
        self._fast_sampler_drops = 0
        self._fast_sampler_coalesced = 0
        self._fast_sampler_events: deque[tuple[float, str]] = deque(maxlen=1200)
        self._quality: dict[str, Any] = {"configured": "maximum",
                                        "effective": "maximum"}
        self._last_fast_outcome: str | None = None
        self._last_fast_outcome_at = 0.0
        self._camera: dict[str, Any] = {}
        self._tracking_active = False
        self._face_anchor_present = False
        self._face_worker_latencies: deque[float] = deque(maxlen=120)
        self._face_worker_drops = 0
        self._face_worker_width: int | None = None
        self._face_worker_alive = False
        self._face_worker_last_at = 0.0

    def note_capture(self, fps: float, timestamp: float,
                     camera: dict | None = None) -> None:
        with self._lock:
            self._capture_fps = float(fps)
            self._latest_frame_ts = float(timestamp)
            if self._geometry_timestamp:
                self._geometry_ages_ms.append(
                    max(0.0, float(timestamp) - self._geometry_timestamp) * 1000.0)
            if camera:
                self._camera = dict(camera)

    def note_preview(self, now: float | None = None,
                     overlay_timestamp: float | None = None) -> None:
        with self._lock:
            at = float(now or time.time())
            self._preview_times.append(at)
            if overlay_timestamp is not None:
                self._overlay_ages_ms.append(
                    max(0.0, at - float(overlay_timestamp)) * 1000.0)

    def note_analysis(self, source_index: int, latency_ms: float,
                      now: float | None = None) -> None:
        with self._lock:
            if self._last_analysis_source_index is not None:
                self._skipped_analysis_frames += max(
                    0, int(source_index) - self._last_analysis_source_index - 1)
            self._last_analysis_source_index = int(source_index)
            self._analysis_times.append(float(now or time.time()))
            value = float(latency_ms)
            self._analysis_latencies.append(value)
            self._analysis_latency_ms = (value if self._analysis_latency_ms == 0.0
                                         else 0.85 * self._analysis_latency_ms + 0.15 * value)

    def note_critical(self, latency_ms: float, geometry_timestamp: float,
                      now: float | None = None) -> None:
        with self._lock:
            at = float(now or time.time())
            self._critical_times.append(at)
            value = float(latency_ms)
            self._critical_latencies.append(value)
            self._critical_latency_ms = (value if self._critical_latency_ms == 0.0
                                         else 0.85 * self._critical_latency_ms + 0.15 * value)
            self._geometry_timestamp = float(geometry_timestamp)

    def note_background_drop(self) -> None:
        with self._lock:
            self._background_queue_drops += 1
            self._background_events.append((time.time(), "drop"))

    def note_background_submission(self) -> None:
        with self._lock:
            self._background_submissions += 1
            self._background_events.append((time.time(), "submit"))

    def note_background_throttled(self, count: int = 1) -> None:
        with self._lock:
            self._background_throttled += max(0, int(count))
            self._background_events.append((time.time(), "throttle"))

    def note_background_coalesced(self) -> None:
        with self._lock:
            self._background_coalesced += 1
            self._background_events.append((time.time(), "coalesced"))

    def note_face_worker(self, latency_ms: float, width: int | None = None) -> None:
        with self._lock:
            self._face_worker_latencies.append(float(latency_ms))
            if width is not None:
                self._face_worker_width = int(width)
            self._face_worker_alive = True
            self._face_worker_last_at = time.time()

    def note_face_worker_drop(self) -> None:
        with self._lock:
            self._face_worker_drops += 1

    def set_face_worker_alive(self, alive: bool) -> None:
        with self._lock:
            self._face_worker_alive = bool(alive)

    def set_scheduler_state(self, level: int, factor: float, state: str) -> None:
        with self._lock:
            self._scheduler = {"level": int(level), "factor": float(factor),
                               "state": str(state)}

    def note_stage_timings(self, timings: dict[str, float]) -> None:
        with self._lock:
            for name, raw in timings.items():
                value = float(raw)
                previous = self._stage_latency_ms.get(name)
                self._stage_latency_ms[name] = (value if previous is None
                                                else 0.85 * previous + 0.15 * value)
                self._stage_samples.setdefault(name, deque(maxlen=120)).append(value)

    def note_fast_path(self, outcome: str, now: float | None = None) -> None:
        with self._lock:
            if outcome == "fed":
                self._fast_fed += 1
            elif outcome == "stale":
                self._fast_stale += 1
            elif outcome == "no_face":
                self._fast_no_face += 1
            else:
                return
            self._last_fast_outcome = outcome
            self._last_fast_outcome_at = float(now or time.time())

    def note_fast_sampler(self, capture_timestamp: float, latency_ms: float) -> None:
        """Record one completed sampler iteration without conflating capture FPS."""
        with self._lock:
            self._fast_sampler_times.append(float(capture_timestamp))
            self._fast_sampler_latencies.append(float(latency_ms))
            self._fast_sampler_events.append((time.time(), "complete"))

    def note_fast_sampler_drop(self) -> None:
        with self._lock:
            self._fast_sampler_drops += 1
            self._fast_sampler_events.append((time.time(), "drop"))

    def note_fast_sampler_coalesced(self, count: int = 1) -> None:
        with self._lock:
            self._fast_sampler_coalesced += max(0, int(count))
            self._fast_sampler_events.append((time.time(), "coalesced"))

    def set_quality_state(self, configured: str, effective: str,
                          detail_roi_cap: int | None = None) -> None:
        with self._lock:
            self._quality = {"configured": str(configured),
                             "effective": str(effective),
                             "detail_roi_cap": detail_roi_cap}

    def set_tracking_state(self, active: bool, face_present: bool) -> None:
        with self._lock:
            self._tracking_active = bool(active)
            self._face_anchor_present = bool(face_present)

    @staticmethod
    def _rate(times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    @staticmethod
    def _summary(values) -> dict[str, float | None]:
        ordered = sorted(float(v) for v in values if math.isfinite(float(v)))
        if not ordered:
            return {"p50": None, "p95": None, "max": None}
        def percentile(q: float) -> float:
            index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
            return round(ordered[index], 1)
        return {"p50": percentile(0.5), "p95": percentile(0.95),
                "max": round(ordered[-1], 1)}

    def snapshot(self, now: float | None = None) -> dict:
        now = float(now or time.time())
        with self._lock:
            recent = [kind for at, kind in self._background_events if now - at <= 30.0]
            submits, drops = recent.count("submit"), recent.count("drop")
            drop_rate = drops / max(1, submits + drops)
            critical_fps = self._rate(self._critical_times)
            geometry_age = (max(0.0, now - self._geometry_timestamp) * 1000.0
                            if self._geometry_timestamp else None)
            critical_stats = self._summary(self._critical_latencies)
            fast_fps = self._rate(self._fast_sampler_times)
            fast_recent = [kind for at, kind in self._fast_sampler_events
                           if now - at <= 30.0]
            fast_drop_rate = (fast_recent.count("drop") /
                              max(1, fast_recent.count("drop") +
                                  fast_recent.count("complete")))
            reasons = []
            if len(self._critical_times) >= 5 and critical_fps < 10.0:
                reasons.append("critical_fps_low")
            if geometry_age is not None and geometry_age > 250.0:
                if self._tracking_active:
                    reasons.append("authoritative_geometry_slow")
                elif not self._face_anchor_present:
                    reasons.append("face_missing")
                else:
                    reasons.append("geometry_stale")
            if drop_rate >= 0.10 and len(recent) >= 5:
                reasons.append("background_overloaded")
            if critical_stats["p95"] is not None and critical_stats["p95"] > 500.0:
                reasons.append("critical_latency_high")
            if len(self._fast_sampler_times) >= 8 and fast_fps < 8.0:
                reasons.append("fast_sampler_slow")
            if len(fast_recent) >= 10 and fast_drop_rate >= 0.10:
                reasons.append("fast_sampler_overloaded")
            if (self._face_worker_alive and self._face_worker_last_at
                    and now - self._face_worker_last_at > 2.0):
                reasons.append("face_worker_stalled")
            actions = {
                "critical_fps_low": "reduce background detector load",
                "geometry_stale": "face the camera or inspect face extraction",
                "face_missing": "no face anchor is available for vitals tracking",
                "authoritative_geometry_slow": (
                    "optimize authoritative face refresh; the tracked vitals bridge is active"),
                "background_overloaded": "adaptive scheduling is throttling passive analysis",
                "critical_latency_high": "inspect the slowest critical stage",
                "face_worker_stalled": "inspect authoritative face worker latency",
                "fast_sampler_slow": "inspect fast sampler sub-stage latency and CPU contention",
                "fast_sampler_overloaded": "reduce native CPU oversubscription or sampler work",
            }
            total_fast = self._fast_fed + self._fast_stale + self._fast_no_face
            return {
                "health": {
                    "status": "degraded" if reasons else "healthy",
                    "reasons": reasons,
                    "actions": [actions[r] for r in reasons],
                    "targets": {"critical_fps_min": 10.0,
                                "geometry_age_p95_ms_max": 250.0,
                                "background_drop_rate_max": 0.10,
                                "critical_operation_ms_max": 500.0},
                },
                "capture_fps": round(self._capture_fps, 1),
                "preview_fps": round(self._rate(self._preview_times), 1),
                "overlay_age_ms": self._summary(self._overlay_ages_ms),
                "analysis_fps": round(self._rate(self._analysis_times), 1),
                "analysis_latency_ms": round(self._analysis_latency_ms, 1),
                "critical_fps": round(critical_fps, 1),
                "critical_latency_ms": round(self._critical_latency_ms, 1),
                "geometry_age_ms": round(geometry_age, 1) if geometry_age is not None else None,
                "background_queue_drops": self._background_queue_drops,
                "background_queue_drop_rate_30s": round(drop_rate, 3),
                "background_submissions": self._background_submissions,
                "background_throttled_modules": self._background_throttled,
                "background_coalesced_frames": self._background_coalesced,
                "scheduler": dict(self._scheduler),
                "latency_distributions_ms": {
                    "analysis": self._summary(self._analysis_latencies),
                    "critical": critical_stats,
                    "geometry_age": self._summary(self._geometry_ages_ms),
                    "stages": {name: self._summary(values)
                               for name, values in self._stage_samples.items()},
                },
                "slow_stages_ms": dict(sorted(
                    ((name, round(value, 1)) for name, value in self._stage_latency_ms.items()),
                    key=lambda item: item[1], reverse=True)[:8]),
                "latest_frame_age_ms": (round(max(0.0, now - self._latest_frame_ts) * 1000.0, 1)
                                        if self._latest_frame_ts else None),
                "skipped_analysis_frames": self._skipped_analysis_frames,
                "fast_path": {
                    "fed": self._fast_fed, "stale": self._fast_stale,
                    "no_face": self._fast_no_face,
                    "acceptance_ratio": round(self._fast_fed / max(1, total_fast), 3),
                    "latest_outcome": self._last_fast_outcome,
                    "latest_outcome_age_ms": (
                        round(max(0.0, now - self._last_fast_outcome_at) * 1000.0, 1)
                        if self._last_fast_outcome_at else None),
                },
                "fast_sampler": {
                    "fps": round(fast_fps, 1),
                    "queue_drops": self._fast_sampler_drops,
                    "queue_drop_rate_30s": round(fast_drop_rate, 3),
                    "intentional_coalescing": self._fast_sampler_coalesced,
                    "latency_ms": self._summary(self._fast_sampler_latencies),
                    "tracking_active": self._tracking_active,
                    "face_anchor_present": self._face_anchor_present,
                },
                "face_worker": {
                    "alive": self._face_worker_alive,
                    "latency_ms": self._summary(self._face_worker_latencies),
                    "queue_replacements": self._face_worker_drops,
                    "adaptive_width": self._face_worker_width,
                    "last_completion_age_ms": (round(max(
                        0.0, now - self._face_worker_last_at) * 1000.0, 1)
                        if self._face_worker_last_at else None),
                },
                "camera": dict(self._camera),
                "quality_profile": dict(self._quality),
            }
