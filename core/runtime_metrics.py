"""Thread-safe live performance counters shared by runtime and debug output."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class RuntimeMetrics:
    """Small rolling metrics store; snapshots are JSON-safe plain values."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._capture_fps = 0.0
        self._preview_times: deque[float] = deque(maxlen=120)
        self._analysis_times: deque[float] = deque(maxlen=120)
        self._analysis_latency_ms = 0.0
        self._critical_times: deque[float] = deque(maxlen=120)
        self._critical_latency_ms = 0.0
        self._geometry_timestamp = 0.0
        self._background_queue_drops = 0
        self._stage_latency_ms: dict[str, float] = {}
        self._latest_frame_ts = 0.0
        self._last_analysis_source_index: int | None = None
        self._skipped_analysis_frames = 0
        self._fast_fed = 0
        self._fast_stale = 0
        self._fast_no_face = 0
        self._last_fast_outcome: str | None = None
        self._last_fast_outcome_at = 0.0
        self._camera: dict[str, Any] = {}

    def note_capture(self, fps: float, timestamp: float, camera: dict | None = None) -> None:
        with self._lock:
            self._capture_fps = float(fps)
            self._latest_frame_ts = float(timestamp)
            if camera:
                self._camera = dict(camera)

    def note_preview(self, now: float | None = None) -> None:
        with self._lock:
            self._preview_times.append(float(now or time.time()))

    def note_analysis(self, source_index: int, latency_ms: float,
                      now: float | None = None) -> None:
        with self._lock:
            if self._last_analysis_source_index is not None:
                self._skipped_analysis_frames += max(
                    0, int(source_index) - self._last_analysis_source_index - 1)
            self._last_analysis_source_index = int(source_index)
            self._analysis_times.append(float(now or time.time()))
            latency_ms = float(latency_ms)
            self._analysis_latency_ms = (latency_ms if self._analysis_latency_ms == 0.0
                                         else 0.85 * self._analysis_latency_ms + 0.15 * latency_ms)

    def note_critical(self, latency_ms: float, geometry_timestamp: float,
                      now: float | None = None) -> None:
        """Record the face/vitals coordinator lane independently of slow analysis."""
        with self._lock:
            at = float(now or time.time())
            self._critical_times.append(at)
            latency_ms = float(latency_ms)
            self._critical_latency_ms = (
                latency_ms if self._critical_latency_ms == 0.0
                else 0.85 * self._critical_latency_ms + 0.15 * latency_ms)
            self._geometry_timestamp = float(geometry_timestamp)

    def note_background_drop(self) -> None:
        with self._lock:
            self._background_queue_drops += 1

    def note_stage_timings(self, timings: dict[str, float]) -> None:
        with self._lock:
            for name, value in timings.items():
                value = float(value)
                previous = self._stage_latency_ms.get(name)
                self._stage_latency_ms[name] = (value if previous is None
                                                else 0.85 * previous + 0.15 * value)

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

    @staticmethod
    def _rate(times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def snapshot(self, now: float | None = None) -> dict:
        now = float(now or time.time())
        with self._lock:
            return {
                "capture_fps": round(self._capture_fps, 1),
                "preview_fps": round(self._rate(self._preview_times), 1),
                "analysis_fps": round(self._rate(self._analysis_times), 1),
                "analysis_latency_ms": round(self._analysis_latency_ms, 1),
                "critical_fps": round(self._rate(self._critical_times), 1),
                "critical_latency_ms": round(self._critical_latency_ms, 1),
                "geometry_age_ms": (round(max(0.0, now - self._geometry_timestamp) * 1000.0, 1)
                                    if self._geometry_timestamp else None),
                "background_queue_drops": self._background_queue_drops,
                "slow_stages_ms": dict(sorted(
                    ((name, round(value, 1)) for name, value in self._stage_latency_ms.items()),
                    key=lambda item: item[1], reverse=True)[:8]),
                "latest_frame_age_ms": (round(max(0.0, now - self._latest_frame_ts) * 1000.0, 1)
                                        if self._latest_frame_ts else None),
                "skipped_analysis_frames": self._skipped_analysis_frames,
                "fast_path": {
                    "fed": self._fast_fed,
                    "stale": self._fast_stale,
                    "no_face": self._fast_no_face,
                    "latest_outcome": self._last_fast_outcome,
                    "latest_outcome_age_ms": (
                        round(max(0.0, now - self._last_fast_outcome_at) * 1000.0, 1)
                        if self._last_fast_outcome_at else None),
                },
                "camera": dict(self._camera),
            }
