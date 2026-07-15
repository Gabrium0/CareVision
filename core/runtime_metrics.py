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
        self._latest_frame_ts = 0.0
        self._last_analysis_source_index: int | None = None
        self._skipped_analysis_frames = 0
        self._fast_fed = 0
        self._fast_stale = 0
        self._fast_no_face = 0
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

    def note_fast_path(self, outcome: str) -> None:
        with self._lock:
            if outcome == "fed":
                self._fast_fed += 1
            elif outcome == "stale":
                self._fast_stale += 1
            elif outcome == "no_face":
                self._fast_no_face += 1

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
                "latest_frame_age_ms": (round(max(0.0, now - self._latest_frame_ts) * 1000.0, 1)
                                        if self._latest_frame_ts else None),
                "skipped_analysis_frames": self._skipped_analysis_frames,
                "fast_path": {
                    "fed": self._fast_fed,
                    "stale": self._fast_stale,
                    "no_face": self._fast_no_face,
                },
                "camera": dict(self._camera),
            }
