"""Deterministic ranking and interruption budget for candidate questions."""
from __future__ import annotations

import math
import time
from collections import deque

#: Minimum seconds between two spoken candidates of the same category.
#: "general" is deliberately absent: no entry means no gap, so every
#: pre-existing candidate (all of which are "general") is unaffected.
DEFAULT_CATEGORY_GAPS: dict[str, float] = {
    "vitals": 300.0,
    "skin": 600.0,
    "movement": 300.0,
    "routine": 900.0,
    "environment": 180.0,
    "mood": 600.0,
    "social": 45.0,
}

_HEALTH_WINDOW_SECONDS = 3600.0


class AttentionPlanner:
    """Rank intent candidates without allowing a generative model to choose alerts."""
    def __init__(self, health_prompt_gap: float = 60.0,
                 category_gaps: dict[str, float] | None = None,
                 health_prompts_per_hour: int = 6):
        self.health_prompt_gap = health_prompt_gap
        self.category_gaps = dict(DEFAULT_CATEGORY_GAPS if category_gaps is None
                                  else category_gaps)
        self.health_prompts_per_hour = int(health_prompts_per_hour)
        self._last_health_prompt: dict[str, float] = {}
        self._denied_until: dict[str, float] = {}
        self._last_category: dict[str, float] = {}
        self._health_history: deque[float] = deque()

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
        """Choose by severity/novelty/quality metadata and prompt budget.

        Three independent budgets gate a candidate before it can be ranked:
        a per-topic denial cooldown, a per-category gap (categories absent from
        `category_gaps` — notably "general" — are never gated), and, for health
        prompts, both the minimum gap and a rolling hourly cap.
        """
        now = time.time() if now is None else now
        while self._health_history and now - self._health_history[0] > _HEALTH_WINDOW_SECONDS:
            self._health_history.popleft()
        allowed = []
        for intent in candidates:
            topic = getattr(intent, "topic", None) or intent.signature.split(":", 1)[0]
            if self._denied_until.get(topic, 0) > now:
                continue
            # getattr with a default matches the defensive style above: test
            # candidates may be plain SimpleNamespaces without this field.
            category = getattr(intent, "category", "general") or "general"
            category_gap = self.category_gaps.get(category)
            if category_gap is not None and \
                    now - self._last_category.get(category, -1e9) < category_gap:
                continue
            explicit_health = getattr(intent, "health_prompt", None)
            health = (intent.kind in ("observation", "question", "follow_up")
                      if explicit_health is None else bool(explicit_health))
            if health:
                if now - self._last_health_prompt.get("primary", -1e9) < self.health_prompt_gap:
                    continue
                if len(self._health_history) >= self.health_prompts_per_hour:
                    continue
            quality = self._bounded(getattr(intent, "quality", None), 1.0, 0.0, 1.0)
            confidence = self._bounded(getattr(intent, "confidence", None), 1.0, 0.0, 1.0)
            novelty = self._bounded(getattr(intent, "novelty", None), 1.0, 0.0, 1.0)
            severity = self._bounded(getattr(intent, "severity_score", None), 0.0, 0.0, 3.0)
            score = (intent.priority + 8*severity) * \
                (0.35 + 0.65 * confidence * quality) * novelty
            allowed.append((score, intent, health, category))
        if not allowed:
            return None
        _score, choice, health, category = max(allowed, key=lambda item: item[0])
        if health:
            self._last_health_prompt["primary"] = now
            self._health_history.append(now)
        self._last_category[category] = now
        return choice
