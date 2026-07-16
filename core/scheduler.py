"""Cadence-aware scheduler.

Each module declares:
  interval: minimum seconds between runs (0.0 = every frame)
  requires: which context features it needs ("face", "pose", "person", "depth")

The scheduler only calls a module when its interval elapsed AND its
requirements are present in the current frame, so a heavy module can't
accidentally run at full framerate and modules never see missing inputs.
"""
from __future__ import annotations

import traceback
import time

from .context import FrameContext
from .events import Result


class Scheduler:
    """Runs each module at its declared cadence when its required inputs are present."""
    def __init__(self, modules: list):
        self.modules = modules
        self._last_run: dict[str, float] = {}

    @staticmethod
    def _requirements_met(module, ctx: FrameContext) -> bool:
        for req in getattr(module, "requires", ()):
            if req == "face" and ctx.face is None:
                return False
            if req == "pose" and ctx.pose is None:
                return False
            if req == "person" and not ctx.person_present:
                return False
            if req == "depth" and ctx.depth is None:
                return False
        return True

    def tick(self, ctx: FrameContext, timings: dict[str, float] | None = None) -> list[Result]:
        """Advance one step: update state and act if warranted."""
        results: list[Result] = []
        for module in self.modules:
            interval = getattr(module, "interval", 0.0)
            last = self._last_run.get(module.name, -1e9)
            if ctx.timestamp - last < interval:
                continue
            if not self._requirements_met(module, ctx):
                continue
            self._last_run[module.name] = ctx.timestamp
            started = time.perf_counter()
            try:
                out = module.process(ctx)
            except Exception:
                print(f"[scheduler] module '{module.name}' raised:")
                traceback.print_exc()
                continue
            finally:
                if timings is not None:
                    timings[f"module:{module.name}"] = round(
                        (time.perf_counter() - started) * 1000.0, 2)
            if out is None:
                continue
            results.extend(out if isinstance(out, list) else [out])
        return results
