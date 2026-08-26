"""Small thread-safe exchange for non-media runtime summaries."""
from __future__ import annotations

import threading
import time


class SharedSignals:
    """Publish short-lived JSON-safe summaries across capture/audio workers."""
    _instance = None

    def __init__(self):
        self._values: dict[str, tuple[object, float]] = {}
        self._lock = threading.RLock()

    @classmethod
    def instance(cls) -> "SharedSignals":
        """Return the process-wide exchange."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set(self, key: str, value, now: float | None = None) -> None:
        """Publish a summary value and timestamp."""
        with self._lock:
            self._values[key] = (value, time.time() if now is None else now)

    def get(self, key: str, default=None, max_age: float | None = None,
            now: float | None = None):
        """Read a value, optionally requiring it to be fresh."""
        with self._lock:
            item = self._values.get(key)
        if item is None:
            return default
        value, timestamp = item
        now = time.time() if now is None else now
        if max_age is not None and now - timestamp > max_age:
            return default
        return value
