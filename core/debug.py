"""Small opt-in debug logger for noisy live modules."""
from __future__ import annotations

import os
import time

_last: dict[str, float] = {}


def enabled(name: str) -> bool:
    """Enabled."""
    raw = os.environ.get("APP_DEBUG_MODULES", "")
    items = {p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()}
    return "all" in items or name.lower() in items


def log(name: str, message: str, interval: float = 5.0) -> None:
    """Print a debug line for a module when its debug flag is enabled."""
    if not enabled(name):
        return
    now = time.time()
    key = name.lower()
    if now - _last.get(key, 0.0) < interval:
        return
    _last[key] = now
    print(f"[{name}/debug] {message}")
