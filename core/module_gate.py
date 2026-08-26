"""Runtime pause/resume overlay for already-instantiated detector modules.

Toggling a module here never loads, unloads, or reconstructs it -- the
scheduler simply skips a gated-off module's `process()` call, so a heavy
model (FashionCLIP, deepface, rPPG backends) stays warm and re-enabling it
costs nothing. This is deliberately narrower than "add a module that was
never started": a module absent from `config/modules.yaml` (enabled: false)
is never instantiated and cannot be toggled on without a restart -- the
/modules console explains that distinction rather than offering a dead
switch.

State lives only in this process (never written back to modules.yaml), so a
restart always returns to the config file's declared set. Reads swap in a
frozenset reference (atomic under the GIL) so the scheduler's hot per-frame
path never blocks on a lock; writes take a short lock to keep concurrent
toggles from racing each other.
"""
from __future__ import annotations

import threading


class ModuleGate:
    """Two independently toggleable scopes: `primary` (the elected subject)
    and `secondary` (every other anonymously-tracked person)."""

    def __init__(self, primary_enabled: set[str] | None = None,
                 secondary_enabled: set[str] | None = None):
        self._lock = threading.Lock()
        self._primary: frozenset[str] = frozenset(primary_enabled or ())
        self._secondary: frozenset[str] = frozenset(secondary_enabled or ())

    def enabled(self, name: str, scope: str = "primary") -> bool:
        """Lock-free read for the scheduler's per-frame hot path."""
        return name in (self._primary if scope == "primary" else self._secondary)

    def set(self, name: str, enabled: bool, scope: str = "primary") -> None:
        with self._lock:
            current = self._primary if scope == "primary" else self._secondary
            updated = (current | {name}) if enabled else (current - {name})
            if scope == "primary":
                self._primary = frozenset(updated)
            else:
                self._secondary = frozenset(updated)

    def snapshot(self) -> dict:
        """Public state for the /modules console; never exposes the lock."""
        return {"primary": sorted(self._primary), "secondary": sorted(self._secondary)}


def seed_start_paused(gate: ModuleGate, names, loaded) -> tuple[list[str], list[str]]:
    """Config-declared boot state: listed modules start gated off in both
    scopes while staying warm (their instances exist; the scheduler just
    skips them). Unknown or not-loaded names are returned rather than
    applied, so a stale config entry can never silently disable anything.
    Returns (applied, ignored); from here on only the /modules console owns
    toggle state."""
    wanted = [str(n) for n in (names or [])]
    applied = [name for name in wanted if name in loaded]
    for name in applied:
        gate.set(name, False, "primary")
        gate.set(name, False, "secondary")
    return applied, [name for name in wanted if name not in applied]
