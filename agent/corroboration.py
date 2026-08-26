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

from agent.answers import interpret_answer
from core.events import Result, Severity

_ORDER = {Severity.INFO: 0, Severity.NOTICE: 1, Severity.WARNING: 2, Severity.ALERT: 3}


def interpret_answer_keywords(text: str) -> str:
    """Offline yes/no/unclear classification (confirmed vocabulary).

    A thin alias over agent.answers.interpret_answer: the corroboration state
    machine and SkinDialogue label an affirmation "confirmed", so the shared
    classifier's "affirmed" is mapped back here.
    """
    verdict = interpret_answer(text)
    return "confirmed" if verdict == "affirmed" else verdict


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


# A check-in must be spoken TO the person, in the second person. An LLM
# generation sometimes drifts into a third-person report ("They've agreed to a
# check-in", "The person seems tired") — talking about the person to a caregiver
# rather than to them. That is a quality failure, not a safety one, so it sits in
# its own deterministic guard: any third-person reference to the person discards
# the generation for the hand-authored rule text (which is second-person by
# construction). Over-triggering is safe — it only ever falls back to vetted text.
_THIRD_PERSON = re.compile(
    r"\b(?:they|them|their|theirs|themselves|she|he|her|hers|him|his)\b"
    r"|\bthe\s+(?:person|patient|resident|user|elderly|senior|individual)\b",
    re.IGNORECASE)


def enforce_second_person(generated: str | None, fallback: str) -> str:
    """Keep an LLM-phrased check-in addressed to the person ('you').

    Returns the generation only if it never refers to the person in the third
    person; otherwise the deterministic hand-authored fallback. Pure and offline;
    mirrors the safe_check_in airlock but guards address, not safety.
    """
    if not generated:
        return fallback
    text = " ".join(str(generated).split())
    if not text or _THIRD_PERSON.search(text):
        return fallback
    return text


# Corroboration is "ask, don't announce the prior": the low-confidence visual
# cue lives in the observation context the LLM sees, but the person must never
# hear it named ("I noticed a possible skin change in the living room ..."). A
# generation that prepends such a disclosure has its announcing sentence removed;
# the genuine question/line is kept. If nothing safe remains, the hand-authored
# rule text stands in. Deterministic and offline, like the other airlocks.
_PRIOR_DISCLOSURE = re.compile(
    r"\bi(?:'ve| have)?\s+(?:noticed|notice|see|saw|observed?|detected?|spotted|"
    r"can see|could see)\b"
    r"|\bmy (?:camera|sensor|reading)s?\b|\bthe camera\b|\bsensors?\b"
    r"|\bin (?:the|your) (?:living\s?room|bedroom|kitchen|bathroom|hallway|"
    r"dining\s?room|room)\b",
    re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    # Split on sentence punctuation AND on dashes/semicolons/colons, so an
    # announcement joined to the real question by an em-dash ("I noticed a rash
    # — have you felt itchy?") can have just its announcing clause removed.
    parts = re.split(r"(?<=[.!?])\s+|\s*[—–]\s*|\s+-\s+|\s*[;:]\s+",
                     text.strip())
    return [s.strip() for s in parts if s and s.strip()]


def strip_prior_disclosure(generated: str | None, fallback: str) -> str:
    """Remove any sentence that announces the visual prior or a location.

    Keeps the genuine spoken line (the question or gentle conclusion) so the
    LLM's natural phrasing survives; falls back to the hand-authored rule text
    only when nothing disclosure-free remains. Pure and offline.
    """
    if not generated:
        return fallback
    text = " ".join(str(generated).split())
    kept = [s for s in _split_sentences(text) if not _PRIOR_DISCLOSURE.search(s)]
    cleaned = " ".join(kept).strip()
    if not cleaned or _PRIOR_DISCLOSURE.search(cleaned):
        return fallback
    return cleaned


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
    # modules/sweating.py emits key "sweat_gloss"; matching is by key prefix,
    # so the old "sweating" key could never match and this rule never fired.
    FollowUpRule(
        "feeling_warm", "sweating", "sweat_gloss",
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

    def cache_selection(self, signature, choice: str | None) -> None:
        """Write an asynchronously resolved steer choice into the memo cache.

        The cache is keyed on the flagged-topic tuple, so a choice that arrives
        several ticks after it was requested lands on the set it was asked
        about; if the flagged set has moved on, the stale entry is simply never
        read. The membership check in `next_question_steered` stays the
        authority — a cached choice is still only honored if it is a member of
        the set currently being offered.
        """
        self._steer_cache = (tuple(signature), choice)

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

    def pending_question(self, now: float):
        """(topic, rule) of the question currently awaiting an answer, or None.

        Split out of `hear` so a caller that wants to defer the verdict (the
        voice agent, while a model refines an ambiguous reply) can identify the
        question without moving the state machine.
        """
        pending = [(t, st) for t, st in self.topics.items()
                   if st.status == "asked"
                   and now - st.asked_at <= self.answer_window]
        if not pending:
            return None
        topic, _st = max(pending, key=lambda x: x[1].asked_at)  # most recent ask
        return topic, self.rules[topic]

    def classify(self, topic: str, text: str) -> str:
        """Interpret `text` as an answer to `topic`'s question."""
        return self._interpret(self.rules[topic], text)

    def apply_verdict(self, topic: str, verdict: str, now: float):
        """Advance one topic's state machine by an already-decided verdict.

        The single place the flagged->asked->answered transition happens, so a
        verdict that arrives asynchronously moves the identical accounting a
        synchronous keyword hit would have — including the one gentle re-ask an
        `unclear` buys while `asks` is still under `max_asks`.
        """
        st = self.topics.get(topic)
        if st is None:
            return None
        if verdict == "unclear" and st.asks < self.max_asks:
            st.status = "flagged"            # allow one gentle re-ask
        else:
            st.status = verdict
            st.answered_at = now
        return topic, verdict

    def hear(self, text: str, now: float):
        """Route a heard utterance to the topic awaiting an answer.

        Returns (topic, verdict) when it answered a pending question, else
        None (free conversation — the voice agent replies contextually).
        Synchronous and offline by construction.
        """
        pending = self.pending_question(now)
        if pending is None:
            return None
        topic, _rule = pending
        return self.apply_verdict(topic, self.classify(topic, text), now)

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

    def rule_state(self, topic: str) -> tuple[FollowUpRule | None, str | None]:
        """Return (rule, current status) for a topic id, both None-safe.

        `agent/topics.py` uses this for the ask-versus-mention handshake: it
        needs the rule's `max_confidence` threshold (below which this engine
        owns the signal and will ASK) together with whether a question is
        currently pending, without reaching into the engine's internals.
        """
        return self.rules.get(topic), self.status(topic)

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
