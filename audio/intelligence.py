"""Optional YAMNet sound events consuming the shared AudioBus."""
from __future__ import annotations

import csv
import queue
import threading
import time
from collections import deque

import numpy as np

from core.capabilities import CapabilityRegistry, CapabilityStatus
from core.events import PersistencePolicy, Result, Severity


_INTEREST = {"cough", "sneeze", "throat clearing", "laughter", "crying",
             "speech", "smoke detector, smoke alarm", "glass", "water", "door",
             "running", "breaking", "crash", "impact", "help"}


def _canonical(label: str) -> str | None:
    """Map YAMNet display names into the bounded showcase event taxonomy."""
    rules = (("smoke", "smoke_alarm"), ("alarm", "alarm"),
             ("cough", "cough"), ("sneeze", "sneeze"),
             ("throat", "throat_clearing"), ("laugh", "laughter"),
             ("cry", "crying"), ("glass", "glass_breaking"),
             ("water", "running_water"), ("door", "door_sound"),
             ("impact", "impact"), ("crash", "impact"),
             ("breaking", "glass_breaking"), ("help", "call_for_help"))
    low = label.lower()
    return next((name for needle, name in rules if needle in low), None)


class SoundEventDetector:
    """Background YAMNet classifier with counted, corroborated cough episodes."""

    def __init__(self, bus, enabled: bool = True, threshold: float = 0.35,
                 allowed_events: set[str] | None = None, model=None, labels=None,
                 sample_rate: int = 16000, hop_seconds: float = 0.1,
                 confirm_windows: int = 2, merge_seconds: float = 2.0,
                 quiet_seconds: float = 3.0, strong_threshold: float = 0.8,
                 start_worker: bool = True):
        self.available = False
        self.enabled = enabled
        self.threshold = threshold
        self.allowed_events = set(allowed_events) if allowed_events is not None else None
        self.mode = "cough_only" if self.allowed_events == {"cough"} else "broad_listening"
        self.sample_rate = sample_rate
        self.window_samples = sample_rate
        self.hop_samples = max(1, int(sample_rate * hop_seconds))
        self.confirm_windows = max(1, confirm_windows)
        self.merge_seconds = merge_seconds
        self.quiet_seconds = quiet_seconds
        self.strong_threshold = strong_threshold
        self._bus, self._in = bus, bus.subscribe("sound-events")
        self._out: queue.Queue[Result] = queue.Queue()
        self._stop = threading.Event()
        self._recent: deque[tuple[float, str]] = deque(maxlen=64)
        self._thread = None
        self._audio = np.empty(0, dtype=np.float32)
        self._audio_start: float | None = None
        self._positive_run: list[float] = []
        self._last_cough_confidence = 0.0
        self._burst_active = False
        self._episode: dict | None = None
        self._max_scores: dict[str, float] = {}
        self._diagnostic_lock = threading.RLock()
        self._windows_processed = 0
        self._last_inference_at: float | None = None
        registry = CapabilityRegistry.instance()
        if not enabled:
            registry.set("yamnet", "model", CapabilityStatus.UNAVAILABLE, "disabled")
            return
        try:
            if model is None:
                import tensorflow_hub as hub
                import tensorflow as tf  # noqa: F401
                model = hub.load("https://tfhub.dev/google/yamnet/1")
                class_map = model.class_map_path().numpy().decode()
                with open(class_map, encoding="utf-8") as source:
                    labels = [row[2] for row in csv.reader(source)][1:]
            if labels is None:
                raise ValueError("YAMNet labels are required")
            self._model = model
            self._labels = list(labels)
        except Exception as exc:  # noqa: BLE001 - optional model boundary
            registry.set("yamnet", "model", CapabilityStatus.UNAVAILABLE,
                         f"optional backend unavailable: {type(exc).__name__}")
            return
        self.available = True
        detail = "cough-only shared microphone" if self.allowed_events == {"cough"} \
            else "shared microphone bus"
        registry.set("yamnet", "model", CapabilityStatus.READY, detail)
        if start_worker:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="yamnet-events")
            self._thread.start()

    def _loop(self) -> None:
        """Classify overlapping one-second windows away from capture threads."""
        while not self._stop.is_set():
            try:
                block, timestamp = self._in.get(timeout=0.5)
            except queue.Empty:
                continue
            self.feed(block, timestamp)

    def feed(self, block: np.ndarray, timestamp: float) -> None:
        """Add one timestamped block and synchronously process complete windows."""
        block = np.asarray(block, dtype=np.float32).ravel()
        if not len(block):
            return
        if self._audio_start is None:
            self._audio_start = float(timestamp)
        self._audio = np.concatenate((self._audio, block))
        while len(self._audio) >= self.window_samples:
            audio = self._audio[:self.window_samples]
            window_end = self._audio_start + self.window_samples / self.sample_rate
            self._classify_window(audio, window_end)
            self._audio = self._audio[self.hop_samples:]
            self._audio_start += self.hop_samples / self.sample_rate

    def _classify_window(self, audio: np.ndarray, timestamp: float) -> None:
        scores, _embeddings, _spectrogram = self._model(audio)
        # YAMNet emits short internal frames. Peak pooling preserves transient
        # coughs; the separate multi-window confirmation rejects one-frame noise.
        means = np.asarray(scores).max(axis=0)
        best: dict[str, float] = {}
        for index in np.argsort(means)[-10:][::-1]:
            label = _canonical(self._labels[int(index)])
            if label is None or (self.allowed_events is not None
                                 and label not in self.allowed_events):
                continue
            best[label] = max(best.get(label, 0.0), float(means[index]))
        with self._diagnostic_lock:
            self._windows_processed += 1
            self._last_inference_at = time.time()
            for label, confidence in best.items():
                self._max_scores[label] = max(self._max_scores.get(label, 0.0), confidence)
            self._observe_cough(best.get("cough", 0.0), timestamp)
            for label, confidence in best.items():
                if label != "cough" and confidence >= self.threshold:
                    self._observe_general(label, confidence, timestamp)

    def _observe_cough(self, confidence: float, timestamp: float) -> None:
        """Confirm consecutive windows, count bursts, and close quiet episodes."""
        with self._diagnostic_lock:
            prior = self._last_cough_confidence
            self._last_cough_confidence = confidence
            strong_pair = (confidence >= self.strong_threshold
                           and prior >= self.threshold / 2)
            supporting_pair = (bool(self._positive_run)
                               and max(self._positive_run) >= self.strong_threshold
                               and confidence >= self.threshold / 2)
            if confidence >= self.threshold or strong_pair or supporting_pair:
                if strong_pair and not self._positive_run:
                    self._positive_run.append(prior)
                self._positive_run.append(confidence)
                if self._burst_active and self._episode is not None:
                    self._episode["ended_at"] = timestamp
                    self._episode["last_positive"] = timestamp
                    self._episode["confidences"].append(confidence)
                if not self._burst_active and len(self._positive_run) >= self.confirm_windows:
                    self._record_burst(timestamp, self._positive_run)
                    self._burst_active = True
                return
            self._positive_run.clear()
            self._burst_active = False
            if self._episode is not None and \
                    timestamp - self._episode["last_positive"] >= self.quiet_seconds:
                self._emit_episode()

    def _record_burst(self, timestamp: float, confidences: list[float]) -> None:
        if self._episode is not None and \
                timestamp - self._episode["last_burst"] > self.merge_seconds:
            self._emit_episode()
        if self._episode is None:
            started_at = timestamp - ((self.confirm_windows - 1)
                                      * self.hop_samples / self.sample_rate)
            self._episode = {"count": 0, "started_at": started_at,
                             "ended_at": timestamp, "last_positive": timestamp,
                             "last_burst": timestamp, "confidences": []}
        self._episode["count"] += 1
        self._episode["last_burst"] = timestamp
        self._episode["ended_at"] = timestamp
        self._episode["last_positive"] = timestamp
        self._episode["confidences"].extend(confidences)

    def _emit_episode(self) -> None:
        episode, self._episode = self._episode, None
        if episode is None:
            return
        confidence = float(np.mean(episode["confidences"]))
        started_at, ended_at = episode["started_at"], episode["ended_at"]
        count = int(episode["count"])
        value = {"count": count, "started_at": started_at, "ended_at": ended_at}
        noun = "burst" if count == 1 else "bursts"
        self._out.put(Result(
            "sound_event", "cough_episode", value, confidence, Severity.NOTICE,
            f"Detected {count} cough-like {noun}", ttl=20, source="microphone",
            quality=confidence, evidence_window=(started_at, ended_at),
            persistence=PersistencePolicy.EVENT, timestamp=ended_at))

    def _observe_general(self, label: str, confidence: float, timestamp: float) -> None:
        self._recent.append((timestamp, label))
        repeated = sum(1 for when, name in self._recent
                       if timestamp - when <= 30 and name == label) >= 2
        urgent = label in ("smoke_alarm", "call_for_help")
        if not repeated and not urgent:
            return
        severity = Severity.ALERT if urgent else Severity.NOTICE
        self._out.put(Result(
            "sound_event", label, True, confidence, severity,
            f"Repeated {label} sound" if repeated else f"Detected {label}",
            ttl=20, source="microphone", quality=confidence,
            persistence=PersistencePolicy.EVENT, timestamp=timestamp))

    def flush(self, timestamp: float | None = None) -> None:
        """Close a pending cough episode after a known end-of-stream boundary."""
        with self._diagnostic_lock:
            if self._episode is None:
                return
            if timestamp is None or timestamp - self._episode["last_positive"] >= self.quiet_seconds:
                self._emit_episode()

    def pop_results(self) -> list[Result]:
        """Drain public-safe event summaries, never audio samples."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def diagnostics(self) -> dict:
        """Return bounded operational telemetry without audio or model tensors."""
        with self._diagnostic_lock:
            status = "ready" if self.available else (
                "disabled" if not self.enabled else "unavailable")
            return {
                "available": self.available,
                "status": status,
                "mode": self.mode,
                "worker_alive": bool(self._thread and self._thread.is_alive()),
                "windows_processed": self._windows_processed,
                "last_inference_at": self._last_inference_at,
                "latest_cough_confidence": self._last_cough_confidence,
                "peak_cough_confidence": self._max_scores.get("cough", 0.0),
                "max_scores": dict(self._max_scores),
                "threshold": self.threshold,
                "pending_cough_episode": self._episode is not None,
                "pending_burst_count": int(self._episode["count"])
                if self._episode is not None else 0,
            }

    def close(self) -> None:
        """Stop the worker and release its bus subscription."""
        self._stop.set()
        self._bus.unsubscribe("sound-events")
        if self._thread:
            self._thread.join(timeout=2)
        self.flush()
