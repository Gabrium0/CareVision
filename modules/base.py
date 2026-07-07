"""Base class every detection module implements.

A module is a drop-in file under modules/:

    from core.registry import register
    from modules.base import DetectionModule

    @register("my_detector")
    class MyDetector(DetectionModule):
        interval = 1.0            # run at most once a second
        requires = ("face",)      # skipped when no face in frame

        def process(self, ctx):
            return self.result("my_key", 0.7, message="…")

Modules keep their own rolling buffers as instance attributes; the
scheduler guarantees requirements are met before process() is called.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from core.context import FrameContext
from core.events import Result, Severity


class DetectionModule(ABC):
    name: str = "unnamed"          # set by @register
    interval: float = 0.0          # min seconds between runs; 0 = every frame
    requires: tuple = ()           # subset of ("face", "pose", "person")

    def __init__(self, **params: Any):
        # unknown yaml params land here so configs never crash a module
        for k, v in params.items():
            setattr(self, k, v)

    @abstractmethod
    def process(self, ctx: FrameContext) -> Optional[Result | list[Result]]:
        ...

    def result(self, key: str, value: Any, confidence: float = 0.5,
               severity: Severity = Severity.INFO, message: str = "",
               ttl: float = 10.0) -> Result:
        return Result(module=self.name, key=key, value=value,
                      confidence=confidence, severity=severity,
                      message=message, ttl=ttl)
