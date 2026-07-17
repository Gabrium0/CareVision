"""In-memory audio fan-out so one microphone stream serves every consumer."""
from __future__ import annotations

import queue
import threading
import time

import numpy as np


class AudioBus:
    """Bounded pub/sub bus; samples are never persisted or logged."""
    def __init__(self):
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.RLock()

    def subscribe(self, name: str, max_blocks: int = 64) -> queue.Queue:
        """Create or replace a named bounded subscription."""
        with self._lock:
            q = queue.Queue(maxsize=max_blocks)
            self._queues[name] = q
            return q

    def unsubscribe(self, name: str) -> None:
        """Remove a consumer and allow queued audio to be collected."""
        with self._lock:
            self._queues.pop(name, None)

    def publish(self, samples: np.ndarray, timestamp: float | None = None) -> None:
        """Fan a copied block out without ever blocking the capture thread."""
        item = (np.asarray(samples, dtype=np.float32).ravel().copy(),
                time.time() if timestamp is None else timestamp)
        with self._lock:
            queues = list(self._queues.values())
        for q in queues:
            queued_item = (item[0].copy(), item[1])
            try:
                q.put_nowait(queued_item)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(queued_item)
                except (queue.Empty, queue.Full):
                    pass
