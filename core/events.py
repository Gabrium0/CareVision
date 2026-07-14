"""Unified result schema emitted by every detection module."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(Enum):
    """Severity levels that control how the aggregator, greeting, and alerting treat a result."""
    INFO = "info"          # normal observation (emotion, heart rate in range)
    NOTICE = "notice"      # worth mentioning in a greeting/recommendation
    WARNING = "warning"    # possible health indicator, suggest attention
    ALERT = "alert"        # urgent (fall, unresponsiveness, stroke signs)


class Visibility(Enum):
    """Controls which consumers may receive a result."""
    PUBLIC = "public"
    AGENT_ONLY = "agent_only"


@dataclass
class Result:
    """A single detection outcome.

    module:     registered module name (e.g. "heart_rate")
    key:        what was measured (a module may emit several keys)
    value:      measurement or label (float, str, bool, dict)
    confidence: 0..1 self-assessed reliability of this reading
    severity:   how the aggregator/greeting engine should treat it
    message:    human-readable one-liner for overlays and logs
    ttl:        seconds this result stays valid in the aggregator
    visibility: public by default; agent_only never reaches UI/alerts/history
    """
    module: str
    key: str
    value: Any
    confidence: float = 0.5
    severity: Severity = Severity.INFO
    message: str = ""
    ttl: float = 10.0
    visibility: Visibility = Visibility.PUBLIC
    timestamp: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl
