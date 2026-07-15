"""Deterministic ranking and interruption budget for candidate questions."""
from __future__ import annotations

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
            quality = float(getattr(intent, "quality", 1.0))
            confidence = float(getattr(intent, "confidence", 1.0))
            novelty = float(getattr(intent, "novelty", 1.0))
            severity = max(0.0, min(3.0, float(getattr(intent, "severity_score", 0.0))))
            score = (intent.priority + 8*severity) * \
                (0.35 + 0.65 * confidence * quality) * novelty
            allowed.append((score, intent, health))
        if not allowed:
            return None
        _score, choice, health = max(allowed, key=lambda item: item[0])
        if health:
            self._last_health_prompt["primary"] = now
        return choice
