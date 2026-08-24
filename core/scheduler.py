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
from collections.abc import Callable

from .context import FrameContext
from .events import Result


class Scheduler:
    """Runs each module at its declared cadence when its required inputs are present."""
    def __init__(self, modules: list, gate=None, scope: str = "primary"):
        self.modules = modules
        self._last_run: dict[str, float] = {}
        self._load_factor = 1.0
        self._throttled = 0
        self._budget_cursor = 0
        # Optional runtime pause/resume overlay (core/module_gate.py). `gate`
        # is None for call sites that predate toggling -- every module then
        # runs exactly as before (gate-less scheduling is unaffected).
        self.gate = gate
        self.scope = scope

    _NEVER_THROTTLE = {"heart_rate", "respiration", "fall", "unresponsive",
                       "near_fall", "guided_assessments"}
    _LOW_PRIORITY = {"rash", "bruise", "eye_redness", "dry_lips", "skin_color",
                     "arm_skin", "skin_vision", "scene_vision", "clothing",
                     "clothing_advice", "grooming", "age_estimation", "body_estimate"}

    def set_load_factor(self, factor: float) -> None:
        """Slow passive work during overload without disabling any module."""
        self._load_factor = max(1.0, min(8.0, float(factor)))

    def pop_throttled(self) -> int:
        value, self._throttled = self._throttled, 0
        return value

    def _effective_interval(self, module, ctx: FrameContext) -> float:
        base = max(0.0, float(getattr(module, "interval", 0.0)))
        active_test = False
        if module.name in {"arm_skin", "skin_vision", "tremor"}:
            from .elicitation import ElicitationState
            active_test = ElicitationState.instance().active(now=ctx.timestamp)
        if (self._load_factor <= 1.0 or module.name in self._NEVER_THROTTLE
                or active_test):
            return base
        factor = (self._load_factor if module.name in self._LOW_PRIORITY
                  else self._load_factor ** 0.5)
        return max(base, 0.25) * factor

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

    def tick(self, ctx: FrameContext, timings: dict[str, float] | None = None,
             should_stop: Callable[[], bool] | None = None,
             budget_ms: float | None = None) -> list[Result]:
        """Advance one step: update state and act if warranted."""
        results: list[Result] = []
        modules = self.modules
        if budget_ms is not None and modules:
            exempt = [m for m in modules if m.name in self._NEVER_THROTTLE]
            passive = [m for m in modules if m.name not in self._NEVER_THROTTLE]
            if passive:
                cursor = self._budget_cursor % len(passive)
                passive = passive[cursor:] + passive[:cursor]
                self._budget_cursor = (cursor + 1) % len(passive)
            modules = exempt + passive
        tick_started = time.perf_counter()
        for module in modules:
            if should_stop is not None and should_stop():
                break
            if self.gate is not None and not self.gate.enabled(module.name, self.scope):
                continue
            interval = self._effective_interval(module, ctx)
            last = self._last_run.get(module.name, -1e9)
            if ctx.timestamp - last < interval:
                if interval > float(getattr(module, "interval", 0.0)):
                    self._throttled += 1
                continue
            if not self._requirements_met(module, ctx):
                continue
            elapsed_ms = (time.perf_counter() - tick_started) * 1000.0
            active_test = False
            if module.name in {"arm_skin", "skin_vision", "tremor"}:
                from .elicitation import ElicitationState
                active_test = ElicitationState.instance().active(now=ctx.timestamp)
            if (budget_ms is not None and elapsed_ms >= float(budget_ms)
                    and module.name not in self._NEVER_THROTTLE and not active_test):
                self._throttled += 1
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
