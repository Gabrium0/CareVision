"""Deterministic ranking and interruption budget for candidate questions."""
from __future__ import annotations

import math
import time


class AttentionPlanner:
    """Rank intent candidates without allowing a generative model to choose alerts."""
    def __init__(self, health_prompt_gap: float = 60.0):
        self.health_prompt_gap = health_prompt_gap
        self._last_health_prompt: dict[str, float] = {}
        self._denied_until: dict[str, float] = {}

    def deny(self, topic: str, seconds: float = 600, now: float | None = None) -> None:
        """Suppress a declined topic for the configured cooldown."""
        now = time.time() if now is None else now
        self._denied_until[topic] = now + seconds

    @staticmethod
    def _bounded(value, default: float, low: float, high: float) -> float:
        """Return finite bounded metadata, treating missing values as unknown.

        Result.quality is intentionally optional.  Missing quality is neutral
        rather than zero-quality; malformed metadata must not take down the
        voice-agent runtime either.
        """
        if value is None:
            return default
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        if not math.isfinite(number):
            return default
        return max(low, min(high, number))

    def choose(self, candidates: list, now: float | None = None):
        """Choose by severity/novelty/quality metadata and prompt budget."""
        now = time.time() if now is None else now
        allowed = []
        for intent in candidates:
            topic = getattr(intent, "topic", None) or intent.signature.split(":", 1)[0]
            if self._denied_until.get(topic, 0) > now:
                continue
            explicit_health = getattr(intent, "health_prompt", None)
            health = (intent.kind in ("observation", "question", "follow_up")
                      if explicit_health is None else bool(explicit_health))
            if health and now - self._last_health_prompt.get("primary", -1e9) < self.health_prompt_gap:
                continue
            quality = self._bounded(getattr(intent, "quality", None), 1.0, 0.0, 1.0)
            confidence = self._bounded(getattr(intent, "confidence", None), 1.0, 0.0, 1.0)
            novelty = self._bounded(getattr(intent, "novelty", None), 1.0, 0.0, 1.0)
            severity = self._bounded(getattr(intent, "severity_score", None), 0.0, 0.0, 3.0)
            score = (intent.priority + 8*severity) * \
                (0.35 + 0.65 * confidence * quality) * novelty
            allowed.append((score, intent, health))
        if not allowed:
            return None
        _score, choice, health = max(allowed, key=lambda item: item[0])
        if health:
            self._last_health_prompt["primary"] = now
        return choice
