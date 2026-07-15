"""Runtime registry for hardware, model, cloud, and sensor readiness."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum


class CapabilityStatus(Enum):
    """Operational state exposed to the showcase dashboard."""
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class Capability:
    """Public-safe state of one optional or required capability."""
    name: str
    category: str
    status: CapabilityStatus
    detail: str = ""
    updated_at: float = 0.0


class CapabilityRegistry:
    """Thread-safe singleton registry that never stores credentials."""
    _instance = None

    def __init__(self):
        self._items: dict[str, Capability] = {}
        self._lock = threading.RLock()

    @classmethod
    def instance(cls) -> "CapabilityRegistry":
        """Return the shared registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set(self, name: str, category: str, status: CapabilityStatus,
            detail: str = "") -> None:
        """Create or update a capability without exposing secret values."""
        with self._lock:
            self._items[name] = Capability(name, category, status, detail, time.time())

    def snapshot(self) -> list[dict]:
        """Return a stable JSON-safe dashboard snapshot."""
        with self._lock:
            out = []
            for item in sorted(self._items.values(), key=lambda x: (x.category, x.name)):
                data = asdict(item)
                data["status"] = item.status.value
                out.append(data)
            return out

    def get(self, name: str) -> Capability | None:
        """Return one capability record without exposing mutable registry state."""
        with self._lock:
            return self._items.get(name)
