"""Conversation policy: decide WHEN and WHAT the agent says.

Event-driven + timed. On arrival it greets (and can small-talk to "stall"); as
the modules surface salient things (clothing not right for the weather, low
mood, tiredness, discomfort, atypical vitals) it raises the highest-priority
one that it hasn't already mentioned. Rate-limited so it doesn't chatter, and it
remembers what it said so it won't repeat a topic until the situation changes.

Safety-critical ALERTs are NOT handled here — those go through the deterministic
AlertManager. The policy only produces friendly, conversational lines.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from core.events import Severity
from agent.state import ObservationMemory
from agent.attention import AttentionPlanner

_ORDER = {Severity.INFO: 0, Severity.NOTICE: 1, Severity.WARNING: 2, Severity.ALERT: 3}


@dataclass
class Intent:
    """One thing the agent could say now, with priority and a templated fallback."""
    kind: str            # greeting | observation | small_talk
    signature: str       # for no-repeat bookkeeping
    llm_intent: str      # instruction to the LLM
    detail: str          # facts for the LLM
    fallback: str        # templated line if no LLM
    priority: int
    confidence: float = 1.0
    quality: float = 1.0
    novelty: float = 1.0
    health_prompt: bool | None = None
    topic: str | None = None
    severity_score: float = 0.0


_SMALL_TALK = [
    "Make light, friendly small talk and ask how their day is going.",
    "Share a warm, general pleasantry and invite them to chat.",
    "Gently check in and let them know you're here if they need anything.",
]


class Policy:
    """Decides when and what the voice agent says (event-driven + timed, no-repeat)."""
    def __init__(self, min_gap: float = 8.0, small_talk_interval: float = 45.0,
                 repeat_cooldown: float = 600.0):
        self.min_gap = min_gap
        self.small_talk_interval = small_talk_interval
        self.repeat_cooldown = repeat_cooldown
        self._last_spoken = -1e9        # so the first utterance isn't gated
        self._spoken: dict[str, float] = {}
        self._greeted_for: float | None = None
        self._small_talk_idx = 0
        self.attention = AttentionPlanner(health_prompt_gap=60.0)

    def _fresh(self, sig: str, now: float) -> bool:
        last = self._spoken.get(sig)
        return last is None or now - last > self.repeat_cooldown

    def _candidates(self, mem: ObservationMemory, now: float) -> list[Intent]:
        cands: list[Intent] = []
        tod = mem.time_of_day()

        # greeting on a new arrival
        if mem.arrived_at is not None and self._greeted_for != mem.arrived_at:
            cands.append(Intent(
                "greeting", f"greeting:{mem.arrived_at}",
                "Greet the person by name for the time of day and make brief, "
                "warm small talk.", f"It is {tod}.",
                f"Good {tod}, {mem.name}! Lovely to see you.", 100))

        # clothing not right for the weather (the driving example)
        adv = mem.get("clothing_advice", "recommendation")
        if adv is not None and _ORDER[adv.severity] >= _ORDER[Severity.NOTICE]:
            sig = f"clothing:{adv.value}"
            if self._fresh(sig, now):
                cands.append(Intent(
                    "observation", sig,
                    "Gently mention what you noticed about their clothing "
                    "versus the weather and offer a suggestion.",
                    str(adv.value), str(adv.value), 60,
                    confidence=adv.confidence, quality=adv.quality,
                    severity_score=float(_ORDER[adv.severity])))

        vitals = mem.get("vitals_advice", "recommendation")
        if vitals is not None and _ORDER[vitals.severity] >= _ORDER[Severity.NOTICE]:
            sig = f"vitals:{vitals.value}"
            if self._fresh(sig, now):
                cands.append(Intent(
                    "observation", sig,
                    "Gently mention the health observation without diagnosing, "
                    "and suggest a calm check-in or rest.",
                    str(vitals.value), str(vitals.value), 65,
                    confidence=vitals.confidence, quality=vitals.quality,
                    severity_score=float(_ORDER[vitals.severity])))

        # discomfort / pain
        pain = mem.get("pain", "pain")
        if pain is not None and pain.severity == Severity.WARNING and self._fresh("pain", now):
            cands.append(Intent(
                "observation", "pain",
                "Gently ask if they are comfortable or in any discomfort.",
                str(pain.message), "You look a little uncomfortable — are you okay?", 70,
                confidence=pain.confidence, quality=pain.quality,
                severity_score=float(_ORDER[pain.severity])))

        # tiredness
        per = mem.get("drowsiness", "perclos")
        if per is not None and _ORDER[per.severity] >= _ORDER[Severity.NOTICE] and self._fresh("tired", now):
            cands.append(Intent(
                "observation", "tired",
                "Kindly note they seem tired and suggest a rest if they'd like.",
                str(per.message), "You seem a little tired — a short rest might feel good.", 50,
                confidence=per.confidence, quality=per.quality,
                severity_score=float(_ORDER[per.severity])))

        # low mood
        if mem.mood() == "low" and self._fresh("mood", now):
            cands.append(Intent(
                "observation", "mood",
                "Warmly acknowledge they seem a bit down and offer company.",
                "Apparent mood is low.",
                f"You seem a little down, {mem.name}. I'm right here if you'd like to talk.", 55))

        # idle small talk
        if now - self._last_spoken >= self.small_talk_interval:
            cands.append(Intent(
                "small_talk", f"smalltalk:{int(now // self.small_talk_interval)}",
                _SMALL_TALK[self._small_talk_idx % len(_SMALL_TALK)], "",
                "How has your day been so far?", 10))
        return cands

    def next_intent(self, mem: ObservationMemory, now: float | None = None,
                    extra: list[Intent] | None = None,
                    suppress_routine: bool = False) -> Intent | None:
        """Pick the highest-priority thing to say now, or None.

        `extra` lets the corroboration/elicitation layers inject their own
        candidates (follow-up questions, conclusions, replies) while this
        policy stays the single place that rate-limits and de-duplicates.
        """
        now = time.time() if now is None else now
        if now - self._last_spoken < self.min_gap:
            # Being spoken to beats the anti-chatter gap: replies and
            # confirmed conclusions may interject; everything else waits.
            cands = [c for c in (extra or [])
                     if c.kind in ("reply", "conclusion")
                     and self._fresh(c.signature, now)]
        else:
            routine = [] if suppress_routine else self._candidates(mem, now)
            cands = [c for c in routine + list(extra or [])
                     if c.kind == "greeting" or self._fresh(c.signature, now)]
        if not cands:
            return None
        return self.attention.choose(cands, now)

    def mark_spoken(self, intent: Intent, now: float | None = None) -> None:
        """Record that an intent was spoken (for no-repeat/cadence)."""
        now = time.time() if now is None else now
        self._last_spoken = now
        self._spoken[intent.signature] = now
        if intent.kind == "greeting":
            self._greeted_for = float(intent.signature.split(":")[1])
        if intent.kind == "small_talk":
            self._small_talk_idx += 1
