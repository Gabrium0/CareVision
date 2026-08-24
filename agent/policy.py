"""Conversation policy: decide WHEN and WHAT the agent says.

Event-driven + timed. On arrival it greets (and can small-talk to "stall"); as
the modules surface salient things (clothing not right for the weather, low
mood, tiredness, discomfort, atypical vitals) it raises the highest-priority
one that it hasn't already mentioned. Rate-limited so it doesn't chatter, and it
remembers what it said so it won't repeat a topic until the situation changes.

Observation topics live in the declarative table in `agent/topics.py`; this
module keeps the derived candidates (greeting, mood, small talk), the
no-repeat/cadence bookkeeping, and the hand-off to `agent/attention.py`.

Safety-critical ALERTs are NOT handled here — those go through the deterministic
AlertManager. The policy only produces friendly, conversational lines.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from agent.state import ObservationMemory
from agent.attention import AttentionPlanner
from agent.topics import TOPICS, apply_config, build_intent


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
    quality: float | None = 1.0
    novelty: float = 1.0
    health_prompt: bool | None = None
    topic: str | None = None
    severity_score: float = 0.0
    # NOTE: `category` must stay LAST. voice_agent.py constructs Intent
    # positionally in ~20 places, so new fields may only be appended.
    category: str = "general"


# A confirmed emergency is the sole interruption while a person is thinking.
# Conversation replies may still bypass the ordinary anti-chatter gap after the
# person has taken their turn, but they must never cut into an answer window.
_GAP_INTERJECTABLE = ("reply", "conclusion", "urgent_alert")
_ANSWER_WINDOW_INTERJECTABLE = ("urgent_alert",)

# Hand-written rotation so repeat visits don't hear an identical opening line.
# The first entry of each row is the original greeting; the variant is picked
# deterministically from the arrival timestamp (varies across visits, stable
# within one, trivially testable). Same warmth rules as every other line:
# warm, brief, no assertions about the person beyond seeing them.
_GREETINGS = {
    "morning": (
        "Good morning, {name}! Lovely to see you.",
        "Morning, {name}! Good to see you up and about.",
        "Hello {name}, and a very good morning to you."),
    "afternoon": (
        "Good afternoon, {name}! Lovely to see you.",
        "Afternoon, {name}! It's nice to see you.",
        "Hello {name}, I hope your day is going well."),
    "evening": (
        "Good evening, {name}! Lovely to see you.",
        "Evening, {name}! A pleasure to see you.",
        "Hello {name}, I hope you've had a pleasant day."),
}

_MOOD_LINES = (
    "You seem a little down, {name}. I'm right here if you'd like to talk.",
    "If today feels heavy, {name}, I'm happy just to keep you company.",
    "You sound a bit low, {name}. Would it help to talk for a bit?",
)

# (LLM instruction, spoken fallback) pairs rotated by _small_talk_idx. The
# first fallback is load-bearing: tests pin "How has your day been so far?".
_SMALL_TALK = [
    ("Make light, friendly small talk and ask how their day is going.",
     "How has your day been so far?"),
    ("Share a warm, general pleasantry and invite them to chat.",
     "It's nice to have some company — anything pleasant on your agenda today?"),
    ("Gently check in and let them know you're here if they need anything.",
     "How are you keeping today?"),
    ("Invite a small memory or story; listen warmly.",
     "Seen anything interesting lately, or anyone stop by?"),
    ("Ask after a simple daily pleasure and offer shared enthusiasm.",
     "Have you had a nice cup of tea or coffee yet today?"),
]


class Policy:
    """Decides when and what the voice agent says (event-driven + timed, no-repeat)."""
    def __init__(self, min_gap: float = 8.0, small_talk_interval: float = 45.0,
                 repeat_cooldown: float = 600.0, conversation=None):
        self.min_gap = min_gap
        self.small_talk_interval = small_talk_interval
        self.repeat_cooldown = repeat_cooldown
        self._last_spoken = -1e9        # so the first utterance isn't gated
        self._spoken: dict[str, float] = {}
        self._greeted_for: float | None = None
        self._small_talk_idx = 0
        # The `conversation:` section of config/modules.yaml tunes numbers and
        # enable flags only; prose (llm_intent/fallback) is never overridable,
        # because a fallback is safety-critical speech reviewed in diffs.
        cfg = dict(conversation or {})
        self.topics = apply_config(cfg.get("topics"), TOPICS)
        self.attention = AttentionPlanner(
            health_prompt_gap=float(cfg.get("health_prompt_gap", 60.0)),
            category_gaps=cfg.get("category_gaps"),
            health_prompts_per_hour=int(cfg.get("health_prompts_per_hour", 6)))

    def _fresh(self, sig: str, now: float) -> bool:
        last = self._spoken.get(sig)
        return last is None or now - last > self.repeat_cooldown

    def _candidates(self, mem: ObservationMemory, now: float, *,
                    corroboration=None) -> list[Intent]:
        """Routine candidates for this tick: derived, then table, then filler.

        `corroboration` is keyword-only and defaults to None, which means
        "admit every spec" — callers that only want the raw table (tests, the
        dashboard) keep the two-positional-argument call signature.
        """
        cands: list[Intent] = []
        tod = mem.time_of_day()

        # --- Derived candidates. These are NOT (module,key)-sourced and so
        # cannot live in agent/topics.py:
        #   greeting  reads mem.arrived_at, and mark_spoken() parses the
        #             "greeting:{ts}" signature back into _greeted_for below.
        #   mood      is computed by mem.mood() across several emotion/valence
        #             backends, not read from one result.
        #   small_talk is a rotating filler line driven by _small_talk_idx and
        #             the speaking cadence, with no observation behind it.
        if mem.arrived_at is not None and self._greeted_for != mem.arrived_at:
            variants = _GREETINGS.get(tod, _GREETINGS["morning"])
            greeting = variants[int(mem.arrived_at) % len(variants)]
            cands.append(Intent(
                "greeting", f"greeting:{mem.arrived_at}",
                "Greet the person by name for the time of day and make brief, "
                "warm small talk.", f"It is {tod}.",
                greeting.format(name=mem.name), 100))

        if mem.mood() == "low" and self._fresh("mood", now):
            mood_line = _MOOD_LINES[int(now) % len(_MOOD_LINES)]
            cands.append(Intent(
                "observation", "mood",
                "Warmly acknowledge they seem a bit down and offer company.",
                "Apparent mood is low.",
                mood_line.format(name=mem.name), 55))

        # --- Declarative observation topics (agent/topics.py).
        for spec in self.topics:
            intent = build_intent(spec, mem, now, corroboration=corroboration)
            if intent is not None and self._fresh(intent.signature, now):
                cands.append(intent)

        # idle small talk
        if now - self._last_spoken >= self.small_talk_interval:
            instruction, fallback = _SMALL_TALK[
                self._small_talk_idx % len(_SMALL_TALK)]
            cands.append(Intent(
                "small_talk", f"smalltalk:{int(now // self.small_talk_interval)}",
                instruction, "", fallback, 10))
        return cands

    def next_intent(self, mem: ObservationMemory, now: float | None = None,
                    extra: list[Intent] | None = None,
                    suppress_routine: bool = False,
                    corroboration=None,
                    awaiting_answer: bool = False) -> Intent | None:
        """Pick the highest-priority thing to say now, or None.

        `extra` lets the corroboration/elicitation layers inject their own
        candidates (follow-up questions, conclusions, replies) while this
        policy stays the single place that rate-limits and de-duplicates.
        `corroboration` is forwarded to the topic table so a signal is either
        asked about or mentioned, never both.
        `awaiting_answer` says a question the agent already asked is still
        inside its answer window; it defaults to False so a caller that does
        no turn-taking bookkeeping behaves exactly as before.
        """
        now = time.time() if now is None else now
        if awaiting_answer:
            # A question owns the floor until the person answers or its window
            # lapses. Only a deterministic confirmed emergency may interrupt.
            cands = [c for c in (extra or [])
                     if c.kind in _ANSWER_WINDOW_INTERJECTABLE
                     and self._fresh(c.signature, now)]
        elif now - self._last_spoken < self.min_gap:
            # Once the person has taken their turn, a reply/conclusion can be
            # prompt despite the normal anti-chatter gap.
            cands = [c for c in (extra or [])
                     if c.kind in _GAP_INTERJECTABLE
                     and self._fresh(c.signature, now)]
        else:
            routine = ([] if suppress_routine else
                       self._candidates(mem, now, corroboration=corroboration))
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
