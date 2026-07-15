"""Optional YAMNet sound events consuming the shared AudioBus."""
from __future__ import annotations

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
    """Background YAMNet classifier with repeated-event corroboration."""
    def __init__(self, bus, enabled: bool = True, threshold: float = 0.35):
        self.available = False
        self.threshold = threshold
        self._bus, self._in = bus, bus.subscribe("sound-events")
        self._out: queue.Queue[Result] = queue.Queue()
        self._stop = threading.Event()
        self._recent: deque[tuple[float, str]] = deque(maxlen=64)
        self._thread = None
        registry = CapabilityRegistry.instance()
        if not enabled:
            registry.set("yamnet", "model", CapabilityStatus.UNAVAILABLE, "disabled")
            return
        try:
            import tensorflow_hub as hub
            import tensorflow as tf  # noqa: F401
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            class_map = self._model.class_map_path().numpy().decode()
            import csv
            self._labels = [row[2] for row in csv.reader(open(class_map, encoding="utf-8"))][1:]
        except Exception as exc:  # noqa: BLE001
            registry.set("yamnet", "model", CapabilityStatus.UNAVAILABLE,
                         f"optional backend unavailable: {type(exc).__name__}")
            return
        self.available = True
        registry.set("yamnet", "model", CapabilityStatus.READY, "shared microphone bus")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="yamnet-events")
        self._thread.start()

    def _loop(self) -> None:
        """Classify rolling ~1 second windows away from capture threads."""
        blocks, samples = [], 0
        while not self._stop.is_set():
            try:
                block, ts = self._in.get(timeout=0.5)
            except queue.Empty:
                continue
            blocks.append(block)
            samples += len(block)
            if samples < 16000:
                continue
            audio = np.concatenate(blocks)[-16000:]
            blocks, samples = [], 0
            scores, _embeddings, _spectrogram = self._model(audio)
            means = np.asarray(scores).mean(axis=0)
            for index in np.argsort(means)[-5:][::-1]:
                raw_label = self._labels[int(index)].lower()
                confidence = float(means[index])
                label = _canonical(raw_label)
                if confidence < self.threshold or label is None:
                    continue
                self._recent.append((ts, label))
                repeated = sum(1 for when, name in self._recent
                               if ts - when <= 30 and name == label) >= 2
                urgent = label in ("smoke_alarm", "call_for_help")
                if not repeated and not urgent:
                    continue
                severity = Severity.ALERT if urgent else Severity.NOTICE
                self._out.put(Result("sound_event", label, True, confidence, severity,
                                     f"Repeated {label} sound" if repeated else f"Detected {label}",
                                     ttl=20, source="microphone", quality=confidence,
                                     persistence=PersistencePolicy.EVENT))

    def pop_results(self) -> list[Result]:
        """Drain public-safe event summaries, never audio samples."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        """Stop the worker and release its bus subscription."""
        self._stop.set()
        self._bus.unsubscribe("sound-events")
        if self._thread:
            self._thread.join(timeout=2)
