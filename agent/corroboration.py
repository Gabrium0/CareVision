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

Answer interpretation prefers the configured language model (a one-word
classification call) and falls back to keyword matching so the
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


# Language that must never reach the person in a check-in. An LLM-phrased line
# may gently *ask* how someone feels, but must never assert a finding, name a
# condition, or accuse. This is the airlock for the "LLM proposes, deterministic
# disposes" pattern: on any hit we discard the generation and speak the
# hand-authored rule text, which is safe by construction and works offline.
_UNSAFE_CHECK_IN = (
    "diagnos", "disease", "stroke", "cancer", "tumor", "tumour", "infection",
    "dementia", "alzheimer", "parkinson", "symptom of", "medical condition",
    "you have ", "you are showing", "you're showing", "youre showing",
    "signs of", "this looks like", "it looks like you", "appears to be",
    "you seem sick", "you look ill", "you look unwell", "you are unwell",
)


def safe_check_in(generated: str | None, fallback: str) -> str:
    """Validate an LLM-generated health check-in line before it is spoken.

    Mirrors SkinDialogue.safe_speech: a generated line may gently ask, but must
    never assert a finding, name a condition, or accuse. Anything that trips the
    blocklist (or is empty / over-long) falls back to the deterministic
    hand-authored text, which is safe by construction. Pure and offline.
    """
    if not generated:
        return fallback
    text = " ".join(str(generated).split())
    if not text or len(text) > 240:
        return fallback
    lowered = text.lower()
    if any(bad in lowered for bad in _UNSAFE_CHECK_IN):
        return fallback
    return text


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
        "hydration", "dry_lips", "lip_dryness",
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
    # --- broadened coverage; keys verified against each detector's emit calls.
    FollowUpRule(
        "discomfort", "pain", "pain",
        "Are you feeling any aches or discomfort at the moment?",
        "Sorry to hear that — resting comfortably might help, and it's worth "
        "mentioning to a doctor if it keeps up."),
    FollowUpRule(
        "puffiness", "facial_swelling", "swelling",
        "Have you noticed any puffiness or swelling lately?",
        "It might be worth keeping an eye on that and mentioning it next time "
        "you see a doctor."),
    FollowUpRule(
        "recent_injury", "bruise", "bruise_fraction",
        "Have you bumped or knocked yourself anywhere recently?",
        "Take care of that spot — and let someone know if it stays sore or "
        "isn't healing."),
    # drowsiness emits an INFO "perclos" every tick and re-emits it at
    # NOTICE/WARNING only when sustained; min_severity NOTICE catches just the
    # noteworthy one. (Yawn is folded in here rather than a second fatigue ask.)
    FollowUpRule(
        "tiredness", "drowsiness", "perclos",
        "You seem a little tired — did you manage to rest well?",
        "Some rest sounds like it would do you good today."),
    FollowUpRule(
        "restlessness", "agitation", "agitation",
        "You seem a bit restless — is anything on your mind?",
        "Thanks for sharing. I'm here if you'd like to talk, or just some "
        "company."),
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
                 language_model=None, answer_window: float = 30.0,
                 denied_cooldown: float = 4 * 3600.0,
                 concluded_cooldown: float = 4 * 3600.0,
                 max_asks: int = 2):
        self.rules = {r.topic: r for r in (rules if rules is not None
                                           else DEFAULT_RULES)}
        self.language_model = language_model
        self.answer_window = answer_window
        self.denied_cooldown = denied_cooldown
        self.concluded_cooldown = concluded_cooldown
        self.max_asks = max_asks
        self.topics: dict[str, TopicState] = {}
        # Memoized LLM topic choice, keyed by the current flagged-topic set, so
        # the selector runs once per distinct set rather than every tick.
        self._steer_cache: tuple[tuple[str, ...], str | None] | None = None

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

    def flagged_topics(self, now: float):
        """All topics currently awaiting a question, oldest-flagged first.

        This is the neutral candidate set offered to an LLM selector. It can
        only ever contain topics the engine already flagged from a real
        detector hit — the selector reorders, it never creates a topic."""
        flagged = [(t, self.rules[t]) for t, st in self.topics.items()
                   if st.status == "flagged"]
        flagged.sort(key=lambda tr: self.topics[tr[0]].flagged_at)
        return flagged

    def next_question_steered(self, now: float, selector=None):
        """Like next_question, but an optional LLM `selector` may choose which
        flagged topic to raise.

        `selector(candidates)` receives the `[(topic, rule)]` flagged set and
        returns a topic id (or None). The choice is membership-checked against
        the flagged set; anything invalid, empty, offline, or raising falls back
        to the deterministic oldest-flagged topic. The result is memoized per
        flagged set so the selector isn't re-invoked every tick.
        """
        candidates = self.flagged_topics(now)
        if not candidates:
            self._steer_cache = None
            return None
        if selector is not None:
            signature = tuple(t for t, _ in candidates)
            if self._steer_cache is not None and self._steer_cache[0] == signature:
                choice = self._steer_cache[1]
            else:
                try:
                    choice = selector(candidates)
                except Exception:  # noqa: BLE001 - a bad selector never blocks the ask
                    choice = None
                self._steer_cache = (signature, choice)
            if choice in {t for t, _ in candidates}:
                return choice, self.rules[choice]
        return candidates[0]

    def mark_asked(self, topic: str, now: float) -> None:
        """Record that the agent just voiced this topic's question."""
        st = self.topics[topic]
        st.status = "asked"
        st.asked_at = now
        st.asks += 1

    # ------------------------------------------------------------- answers

    def _interpret(self, rule: FollowUpRule, text: str) -> str:
        if (self.language_model is not None
                and getattr(self.language_model, "available", False)):
            verdict = self.language_model.classify_answer(rule.question, text)
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

    def funnel(self) -> dict:
        """Research telemetry: the flagged -> asked -> answered funnel.

        A privacy-safe count only (topic ids + states, never raw cues or
        answers), so it can be exposed on /debug/state to study which visual
        priors actually lead to corroboration versus denial."""
        by_status = {"flagged": 0, "asked": 0, "confirmed": 0,
                     "denied": 0, "unclear": 0}
        for st in self.topics.values():
            by_status[st.status] = by_status.get(st.status, 0) + 1
        return {"topics_seen": len(self.topics),
                "by_status": by_status,
                "concluded": sum(1 for st in self.topics.values() if st.concluded)}
