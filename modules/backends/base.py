"""Common interface for a detector backend.

A backend is fed one frame at a time via update(ctx) and asked for a reading
via compute(); compute() returns a dict of named values (or None if not
ready). A multi-backend module runs several of these and emits one Result per
backend, so the heuristic and any tested-model backends all show at once.

    label:     short name shown in messages / result keys ("heuristic", "deepface")
    available: False if the backend's deps/weights are missing (module skips it)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.context import FrameContext


class Backend(ABC):
    label: str = "backend"
    available: bool = True

    @abstractmethod
    def update(self, ctx: FrameContext) -> None:
        """Feed one frame (cheap: buffer / stash what compute() needs)."""

    @abstractmethod
    def compute(self) -> dict | None:
        """Return a reading dict, or None if not ready this call."""

    def close(self) -> None:        # optional cleanup (release models)
        pass
