"""Corroboration engine: low-confidence flags -> follow-up questions -> gated
conclusions.

This is the clinical-decision-support pattern the whole system is built
around: a vision cue alone is a *prior*, not a finding. When a detector
raises something low-confidence (a possible rash, pallor, cold symptoms),
the agent doesn't announce it — it asks a natural follow-up question
("have you noticed any skin changes lately?"). Only when the person's
answer corroborates the cue does the agent gently surface a conclusion;
a denial suppresses the topic for hours instead of arguing with the human.

State machine per topic:  flagged -> asked -> confirmed | denied | unclear
(unclear allows one re-ask; denied/confirmed enter a long cooldown).

Answer interpretation prefers Gemini (a one-word classification call via
`GeminiClient.classify_answer`) and falls back to keyword matching so the
loop still works fully offline.

The engine produces *data* (which question to ask, which conclusion is
ready); `agent/voice_agent.py` turns those into `agent/policy.py` Intents
so all rate-limiting/no-repeat behavior stays in one place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.events import Result, Severity

_ORDER = {Severity.INFO: 0, Severity.NOTICE: 1, Severity.WARNING: 2, Severity.ALERT: 3}

# Words that (dis)confirm a health follow-up in casual speech. Checked
# negations first: "no, not really" must not confirm via "really".
_DENY = ("no", "nope", "not really", "nothing", "haven't", "hasn't", "don't",
         "doesn't", "i'm fine", "im fine", "i am fine", "all good", "never")
_CONFIRM = ("yes", "yeah", "yep", "a bit", "a little", "i have", "i do",
            "actually", "lately", "sometimes", "now that you mention",
            "i guess so", "kind of", "sort of")


def interpret_answer_keywords(text: str) -> str:
    """Offline yes/no/unclear classification of a spoken reply."""
    # Strip punctuation and pad so every match is whole-word bounded
    # ("no" must not fire inside "noticed").
    t = " " + re.sub(r"[^a-z' ]+", " ", text.lower()) + " "
    for w in _DENY:
        if f" {w} " in t:
            return "denied"
    for w in _CONFIRM:
        if f" {w} " in t:
            return "confirmed"
    return "unclear"


@dataclass
class FollowUpRule:
    """One low-confidence signal worth a conversational follow-up."""
    topic: str                   # stable id, e.g. "skin_changes"
    module: str                  # triggering Result.module
    key: str                     # triggering Result.key (prefix match)
    question: str                # what the agent asks
    conclusion: str              # what it gently says if confirmed
    max_confidence: float = 0.6  # above this the signal doesn't need asking
    min_severity: Severity = Severity.NOTICE


# The feedback's named low-confidence signals, phrased as check-ins.
DEFAULT_RULES = [
    FollowUpRule(
        "skin_changes", "rash", "rash",
        "Have you noticed any skin changes or irritation lately?",
        "It might be worth keeping an eye on that skin spot, and mentioning "
        "it next time you see a doctor."),
    FollowUpRule(
        "tiredness_pallor", "skin_color", "pallor",
        "Have you been feeling more tired or run-down than usual lately?",
        "Since you're feeling run down, some rest and a good meal could "
        "help — and mention it to a doctor if it keeps up."),
    FollowUpRule(
        "hydration", "dry_lips", "dry_lips",
        "Have you had enough to drink today?",
        "A glass of water sounds like a good idea then."),
    FollowUpRule(
        "feeling_warm", "sweating", "sweating",
        "Are you feeling warm or a bit overheated?",
        "Maybe cool down for a moment and have some water."),
    FollowUpRule(
        "cold_symptoms", "wellness_advice", "cold_symptoms",
        "Are you feeling a bit under the weather today?",
        "Taking it easy and drinking something warm might be just the "
        "thing today."),
    FollowUpRule(
        "low_mood", "expressivity", "expressivity_low",
        "You seem a little quieter than usual — how are you feeling today?",
        "Thanks for sharing that with me. I'm always here if you'd like "
        "some company."),
]


@dataclass
class TopicState:
    """Where one topic currently sits in the flagged->asked->answered flow."""
    status: str = "flagged"          # flagged | asked | confirmed | denied | unclear
    flagged_at: float = 0.0
    asked_at: float = 0.0
    answered_at: float = 0.0
    asks: int = 0
    concluded: bool = False


class CorroborationEngine:
    """Tracks topic state; the VoiceAgent turns its output into Intents."""

    def __init__(self, rules: list[FollowUpRule] | None = None,
                 gemini=None, answer_window: float = 30.0,
                 denied_cooldown: float = 4 * 3600.0,
                 concluded_cooldown: float = 4 * 3600.0,
                 max_asks: int = 2):
        self.rules = {r.topic: r for r in (rules if rules is not None
                                           else DEFAULT_RULES)}
        self.gemini = gemini             # GeminiClient or None (keyword fallback)
        self.answer_window = answer_window
        self.denied_cooldown = denied_cooldown
        self.concluded_cooldown = concluded_cooldown
        self.max_asks = max_asks
        self.topics: dict[str, TopicState] = {}

    # ------------------------------------------------------------ observe

    def _in_cooldown(self, st: TopicState, now: float) -> bool:
        if st.status == "denied":
            return now - st.answered_at < self.denied_cooldown
        if st.status == "confirmed":
            return now - st.answered_at < self.concluded_cooldown
        return False

    def observe(self, snapshot: list[Result], now: float) -> None:
        """Flag topics whose low-confidence trigger appears in the snapshot."""
        for rule in self.rules.values():
            st = self.topics.get(rule.topic)
            if st is not None and (st.status in ("flagged", "asked")
                                   or self._in_cooldown(st, now)):
                continue
            hit = next(
                (r for r in snapshot
                 if r.module == rule.module and r.key.startswith(rule.key)
                 and r.confidence <= rule.max_confidence
                 and _ORDER[r.severity] >= _ORDER[rule.min_severity]),
                None)
            if hit is not None:
                self.topics[rule.topic] = TopicState(status="flagged",
                                                     flagged_at=now)

    # ---------------------------------------------------------- questions

    def next_question(self, now: float):
        """Oldest flagged topic's (topic, rule), or None."""
        flagged = [(t, st) for t, st in self.topics.items()
                   if st.status == "flagged"]
        if not flagged:
            return None
        topic, _st = min(flagged, key=lambda x: x[1].flagged_at)
        return topic, self.rules[topic]

    def mark_asked(self, topic: str, now: float) -> None:
        """Record that the agent just voiced this topic's question."""
        st = self.topics[topic]
        st.status = "asked"
        st.asked_at = now
        st.asks += 1

    # ------------------------------------------------------------- answers

    def _interpret(self, rule: FollowUpRule, text: str) -> str:
        if self.gemini is not None and getattr(self.gemini, "available", False):
            verdict = self.gemini.classify_answer(rule.question, text)
            if verdict in ("confirmed", "denied", "unclear"):
                return verdict
        return interpret_answer_keywords(text)

    def hear(self, text: str, now: float):
        """Route a heard utterance to the topic awaiting an answer.

        Returns (topic, verdict) when it answered a pending question, else
        None (free conversation — the voice agent replies contextually).
        """
        pending = [(t, st) for t, st in self.topics.items()
                   if st.status == "asked"
                   and now - st.asked_at <= self.answer_window]
        if not pending:
            return None
        topic, st = max(pending, key=lambda x: x[1].asked_at)  # most recent ask
        verdict = self._interpret(self.rules[topic], text)
        if verdict == "unclear" and st.asks < self.max_asks:
            st.status = "flagged"            # allow one gentle re-ask
        else:
            st.status = verdict
            st.answered_at = now
        return topic, verdict

    # ---------------------------------------------------------- conclusions

    def pending_conclusions(self):
        """[(topic, rule)] for confirmed topics not yet concluded aloud."""
        return [(t, self.rules[t]) for t, st in self.topics.items()
                if st.status == "confirmed" and not st.concluded]

    def mark_concluded(self, topic: str) -> None:
        """Record that the conclusion line was spoken."""
        self.topics[topic].concluded = True

    def suppress(self, topic: str, now: float) -> None:
        """Dismiss a topic while a richer flow handles the same observation."""
        state = self.topics.get(topic)
        if state is None or state.status not in ("denied", "confirmed"):
            self.topics[topic] = TopicState(status="denied", answered_at=now)

    def status(self, topic: str) -> str | None:
        """Current state of a topic (for dashboards/tests)."""
        st = self.topics.get(topic)
        return st.status if st else None
