"""Shared state for agent-scripted active tests (elicitation windows).

The feedback behind the tremor module is explicit: elicited measurement
(ask the person to hold a hand out and keep still) validates far better
than passive observation. That needs a tiny contract between the agent
layer (which speaks the instruction and starts the window) and detection
modules (which sample differently inside it). A process-wide singleton —
the same pattern as storage.history_store.HistoryStore.instance() — keeps
that contract out of the frame pipeline entirely: no FrameContext changes,
no scheduler changes.

Lives in core/ (not agent/) because modules/ may only depend on core and
extractors; the agent layer already depends on core.
"""
from __future__ import annotations

import time


class ElicitationState:
    """Process-wide record of the currently active scripted test window."""
    _instance = None

    def __init__(self):
        self.test: str | None = None     # e.g. "hold_still"
        self.started: float = 0.0
        self.until: float = 0.0

    @classmethod
    def instance(cls) -> "ElicitationState":
        """Return the process-wide singleton, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def begin(self, test: str, duration: float, now: float | None = None) -> None:
        """Open a test window (called by the agent when it speaks the ask)."""
        now = time.time() if now is None else now
        self.test = test
        self.started = now
        self.until = now + duration

    def active(self, test: str | None = None, now: float | None = None) -> bool:
        """True while a (matching) test window is open."""
        now = time.time() if now is None else now
        if self.test is None or now >= self.until:
            return False
        return test is None or self.test == test

    def clear(self) -> None:
        """Close the window early."""
        self.test = None
        self.until = 0.0
