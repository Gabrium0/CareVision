"""Declarative topic table: which detector signals may become conversation.

`agent/policy.py` used to carry a hand-written if-chain, one branch per topic.
This module replaces that with data: a `TopicSpec` names a `(module, key)`
pair, the admission gates it must clear, and the hand-authored wording used to
raise it. Adding a topic becomes a table entry, not a new code path.

Privacy invariant (why no spec can leak a private hypothesis)
-------------------------------------------------------------
The table reads observations *exclusively* through `ObservationMemory.get` /
`ObservationMemory.get_val`. `ObservationMemory.ingest` (agent/state.py:44-46)
`continue`s on `Visibility.AGENT_ONLY` **before** writing to `self.latest`, so
an agent-only Result never lands in the memory the table queries. A spec that
names an agent-only `(module, key)` therefore resolves to `None` and produces
no candidate at all — regardless of what that spec declares. Private
hypotheses stay on the dedicated agent-only paths (the corroboration question
flow and `agent/conversation.py`'s bounded context), which is the only place
they are allowed to influence speech, and only as a gentle question.

Authority invariant
-------------------
Nothing here consults a language model. `build_intent` is a pure, deterministic
function of memory contents plus the spec; the LLM only ever phrases an intent
that this table (and `agent/attention.py`) already selected.

Formatting invariant
--------------------
Every `fallback` template in `TOPICS` is a hand-authored constant *in this
file*. `render_fallback` calls `str.format_map` with a plain dict of
pre-stringified values, so detector data is only ever a format *argument*,
never part of the format string. No attribute traversal (`{value.__class__}`)
or index escape is reachable from observed data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol, runtime_checkable

from agent.corroboration import DEFAULT_RULES
from agent.state import ObservationMemory
from core.events import Result, Severity

_ORDER = {Severity.INFO: 0, Severity.NOTICE: 1,
          Severity.WARNING: 2, Severity.ALERT: 3}

#: Placeholders a `TopicSpec.fallback` template may use.
ALLOWED_PLACEHOLDERS = frozenset({"value", "message", "name", "tod", "detail"})

#: Signature styles a `TopicSpec.signature` may name.
SIGNATURE_STYLES = frozenset({"topic", "topic:value"})

#: Spec fields `TopicSpec.apply_overrides` will take from `config/modules.yaml`.
#: Deliberately numeric/boolean only. Every prose field (`llm_intent`,
#: `fallback`, `detail`) is absent, so no config file can ever change what the
#: agent SAYS — only whether, and how eagerly, an already-authored line is
#: raised. `topic`/`module`/`key`/`category` are absent too: retargeting a spec
#: at a different observation from yaml would silently break the ownership and
#: corroboration invariants the tests pin.
OVERRIDABLE_FIELDS = frozenset({"enabled", "priority", "min_confidence",
                                "min_quality", "min_severity"})

#: Gate attributes an override may retune, under a nested `gate:` mapping.
#: Bounds only — never `unit`/`decimals` (rendering, i.e. prose) and never
#: `names`/`field`/`container` (which observation is read).
OVERRIDABLE_GATE_BOUNDS = frozenset({"minimum", "maximum", "max_items"})

#: Categories that `agent/attention.py` knows how to rate-limit. "general"
#: is deliberately absent from its gap table (no gap => never category-gated).
KNOWN_CATEGORIES = frozenset({"general", "vitals", "skin", "movement",
                              "routine", "environment", "mood", "social"})

# The two trailing sentences that follow a "Supporting visible appearance
# cues:" list. They differ by call site and are asserted on character-for-
# character by tests/corroboration_and_elicitation_test.py, so they stay
# parameterised rather than unified into one wording.
SUPPORT_TRAILING_OBSERVATION = (
    "Treat them only as corroboration, not as a cause or diagnosis.")
SUPPORT_TRAILING_CHECK_IN = (
    "Use them only to phrase the check-in; do not state a cause or diagnosis.")


# --------------------------------------------------------------- primitives

def _number(value: Any) -> float | None:
    """Return a finite float for `value`, or None if it is not a real number.

    Guarded so a malformed detector value can never raise inside ranking:
    booleans (which float() would silently accept), None, and the codebase's
    "not measured yet" placeholders are rejected, as are NaN/inf.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip() in ("", "...", "unknown"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _short(text: Any, limit: int = 120) -> str:
    """Collapse whitespace, strip, and truncate `text` to `limit` characters."""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip()


# -------------------------------------------------------------------- gates

@runtime_checkable
class Gate(Protocol):
    """An admission test plus a bounded rendering for one observed value."""

    def passes(self, value: Any) -> bool:
        """True if `value` is shaped and ranged such that the topic may fire."""

    def render(self, value: Any) -> str:
        """Return a short, bounded string for this value (never unbounded)."""


@dataclass(frozen=True)
class Threshold:
    """Numeric gate: admit a finite number inside [minimum, maximum]."""
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    decimals: int = 1

    def passes(self, value: Any) -> bool:
        """True if `value` is a finite number within the configured band."""
        number = _number(value)
        if number is None:
            return False
        if self.minimum is not None and number < self.minimum:
            return False
        if self.maximum is not None and number > self.maximum:
            return False
        return True

    def render(self, value: Any) -> str:
        """Return the number at the configured precision, plus any unit."""
        number = _number(value)
        if number is None:
            return ""
        return f"{number:.{self.decimals}f}{self.unit}"


@dataclass(frozen=True)
class IsTrue:
    """Boolean gate: admit only the literal `True`, never a truthy value."""

    def passes(self, value: Any) -> bool:
        """True only when `value is True` (1, "yes" and [] all fail)."""
        return value is True

    def render(self, value: Any) -> str:
        """Return a fixed word; a boolean carries no other detail."""
        return "yes" if value is True else "no"


@dataclass(frozen=True)
class OneOf:
    """String gate: admit a value drawn from a closed vocabulary."""
    values: tuple[str, ...] = ()

    def passes(self, value: Any) -> bool:
        """True if `value` is a string listed in `values`."""
        return isinstance(value, str) and value.strip() in self.values

    def render(self, value: Any) -> str:
        """Return the label itself, bounded."""
        return _short(value, 60)


@dataclass(frozen=True)
class DictField:
    """Dict gate: apply `inner` to one named field of a dict value."""
    field: str
    inner: Any

    def passes(self, value: Any) -> bool:
        """True if `value` is a dict whose `field` clears the inner gate."""
        if not isinstance(value, dict) or self.field not in value:
            return False
        return bool(self.inner.passes(value[self.field]))

    def render(self, value: Any) -> str:
        """Return the inner gate's rendering of the named field."""
        if not isinstance(value, dict) or self.field not in value:
            return ""
        return _short(self.inner.render(value[self.field]))


@dataclass(frozen=True)
class DictCues:
    """Dict gate for the `{"cues": {name: label}}` appearance-cue shape."""
    container: str = "cues"
    names: tuple[str, ...] = ()

    def _cues(self, value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        cues = value.get(self.container)
        return cues if isinstance(cues, dict) else {}

    def passes(self, value: Any) -> bool:
        """True if at least one of `names` (or any cue, if empty) is present."""
        cues = self._cues(value)
        if not cues:
            return False
        if not self.names:
            return True
        return any(name in cues for name in self.names)

    def render(self, value: Any) -> str:
        """Return "name label" pairs for the matching cues, bounded."""
        cues = self._cues(value)
        names = self.names or tuple(cues)
        parts = [f"{name.replace('_', ' ')} {_short(cues[name], 24)}"
                 for name in names if name in cues]
        return _short(", ".join(parts))


@dataclass(frozen=True)
class NonEmptyList:
    """Sequence gate: admit a list/tuple holding at least `minimum` items."""
    minimum: int = 1
    max_items: int = 3

    def passes(self, value: Any) -> bool:
        """True if `value` is a list/tuple of at least `minimum` entries."""
        return isinstance(value, (list, tuple)) and len(value) >= self.minimum

    def render(self, value: Any) -> str:
        """Return the first `max_items` entries, comma-joined and bounded."""
        if not isinstance(value, (list, tuple)):
            return ""
        return _short(", ".join(_short(item, 40)
                                for item in value[:self.max_items]))


# ---------------------------------------------------------------- topic spec

@dataclass(frozen=True)
class TopicSpec:
    """One conversational topic, sourced from a single detector `(module, key)`.

    Field order is part of the contract: entries in `TOPICS` are written
    keyword-first, but the ordering keeps positional construction stable for
    the small number of tests that use it.

    topic/module/key   the stable topic id and the observation it reads
    llm_intent         instruction handed to the phrasing model
    fallback           hand-authored template spoken when no model is available
    priority           base rank handed to `AttentionPlanner`
    category           rate-limit bucket ("general" is never category-gated)
    kind               Intent kind (drives the health-prompt budget)
    min_severity/min_confidence/min_quality   admission floors
    gate               optional value-shape/range test (also bounds rendering)
    detail             "message", "value", or a literal string for Intent.detail
    signature          "topic" or "topic:value" (no-repeat bookkeeping key)
    health_prompt      None defers to `kind`; True/False forces the budget
    support_cues       appearance-cue names appended to `llm_intent` when seen
    corroborated_by    FollowUpRule.topic this spec hands off to when uncertain
    enabled            False removes the spec from the table without deleting it
    """
    topic: str
    module: str
    key: str
    llm_intent: str
    fallback: str
    priority: int
    category: str = "general"
    kind: str = "observation"
    min_severity: Severity = Severity.NOTICE
    min_confidence: float = 0.35
    min_quality: float = 0.0
    gate: Any = None
    detail: str = "message"
    signature: str = "topic"
    health_prompt: bool | None = None
    support_cues: tuple[str, ...] = ()
    corroborated_by: str | None = None
    enabled: bool = True

    def apply_overrides(self, cfg: Any) -> "TopicSpec":
        """Return a copy of this spec with `cfg`'s numeric/boolean tweaks applied.

        `cfg` is one `conversation.topics.<topic>` mapping from
        `config/modules.yaml`. Only `OVERRIDABLE_FIELDS` and, under a nested
        `gate:` key, `OVERRIDABLE_GATE_BOUNDS` are accepted; **any** other key
        raises `ValueError` at load time rather than being ignored, so a typo
        (`min_confidance`, or an attempt to reword `fallback`) is loud instead
        of silently leaving the deployed spec on its defaults.

        Deployment can therefore retune *how eagerly* a topic is raised, but
        never *what is said* — the prose stays hand-authored in this file,
        which is what makes the offline fallback and the safety airlocks
        trustworthy.
        """
        if cfg is None:
            return self
        if not isinstance(cfg, dict):
            raise ValueError(
                f"topic {self.topic!r}: override must be a mapping, "
                f"got {type(cfg).__name__}")

        changes: dict[str, Any] = {}
        for name, raw in cfg.items():
            key = str(name)
            if key == "gate":
                changes["gate"] = self._override_gate(raw)
                continue
            if key not in OVERRIDABLE_FIELDS:
                raise ValueError(
                    f"topic {self.topic!r}: {key!r} is not overridable; "
                    f"allowed keys are "
                    f"{sorted(OVERRIDABLE_FIELDS | {'gate'})}")
            changes[key] = _coerce_override(self.topic, key, raw)
        return replace(self, **changes)

    def _override_gate(self, cfg: Any) -> Any:
        """Return this spec's gate with `cfg`'s numeric bounds replaced."""
        if not isinstance(cfg, dict):
            raise ValueError(
                f"topic {self.topic!r}: 'gate' override must be a mapping, "
                f"got {type(cfg).__name__}")
        if self.gate is None:
            raise ValueError(
                f"topic {self.topic!r}: has no gate to override")
        available = {f.name for f in fields(self.gate)} & OVERRIDABLE_GATE_BOUNDS
        changes: dict[str, Any] = {}
        for name, raw in cfg.items():
            key = str(name)
            if key not in available:
                raise ValueError(
                    f"topic {self.topic!r}: gate bound {key!r} is not "
                    f"overridable on {type(self.gate).__name__}; allowed "
                    f"bounds are {sorted(available)}")
            number = _number(raw)
            if number is None:
                raise ValueError(
                    f"topic {self.topic!r}: gate bound {key!r} must be a "
                    f"finite number, got {raw!r}")
            changes[key] = int(number) if key == "max_items" else number
        return replace(self.gate, **changes)


def _coerce_override(topic: str, key: str, raw: Any) -> Any:
    """Validate and convert one override value for `TopicSpec.apply_overrides`."""
    if key == "enabled":
        if not isinstance(raw, bool):
            raise ValueError(
                f"topic {topic!r}: 'enabled' must be true or false, got {raw!r}")
        return raw
    if key == "min_severity":
        if isinstance(raw, Severity):
            return raw
        try:
            return Severity(str(raw).strip().lower())
        except ValueError:
            raise ValueError(
                f"topic {topic!r}: 'min_severity' must be one of "
                f"{[s.value for s in Severity]}, got {raw!r}") from None
    number = _number(raw)
    if number is None:
        raise ValueError(
            f"topic {topic!r}: {key!r} must be a finite number, got {raw!r}")
    if key == "priority":
        return int(number)
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"topic {topic!r}: {key!r} must be within 0..1, got {raw!r}")
    return number


def apply_config(cfg: Any, topics: tuple[TopicSpec, ...] = ()) -> tuple[TopicSpec, ...]:
    """Return `topics` (default: `TOPICS`) with a `conversation.topics` overlay.

    Raises `ValueError` for a topic id that no spec declares, for the same
    reason `apply_overrides` rejects unknown fields: a stale or misspelled id
    in deployment config must not look like a working override.
    """
    table = topics or TOPICS
    if not cfg:
        return table
    if not isinstance(cfg, dict):
        raise ValueError(f"conversation.topics must be a mapping, "
                         f"got {type(cfg).__name__}")
    known = {spec.topic for spec in table}
    unknown = sorted(str(name) for name in cfg if str(name) not in known)
    if unknown:
        raise ValueError(f"conversation.topics names unknown topics: {unknown}")
    return tuple(spec.apply_overrides(cfg.get(spec.topic)) for spec in table)


# ------------------------------------------------------------------ the table

TOPICS: tuple[TopicSpec, ...] = (
    TopicSpec(
        topic="clothing", module="clothing_advice", key="recommendation",
        llm_intent="Gently mention what you noticed about their clothing "
                   "versus the weather and offer a suggestion.",
        fallback="{value}", priority=60,
        detail="value", signature="topic:value"),
    TopicSpec(
        topic="vitals", module="vitals_advice", key="recommendation",
        llm_intent="Gently mention the health observation without diagnosing, "
                   "and suggest a calm check-in or rest.",
        fallback="{value}", priority=65,
        detail="value", signature="topic:value"),
    TopicSpec(
        topic="pain", module="pain", key="pain",
        llm_intent="Gently ask if they are comfortable or in any discomfort.",
        fallback="You look a little uncomfortable — are you okay?", priority=70,
        min_severity=Severity.WARNING, corroborated_by="discomfort"),
    TopicSpec(
        topic="tired", module="drowsiness", key="perclos",
        llm_intent="Kindly ask how they are feeling and suggest a rest if "
                   "they'd like.",
        fallback="You seem a little tired — how are you feeling? A short rest "
                 "might feel good.",
        priority=50,
        # category "mood" (600s gap) rate-limits the prompt instead of leaving it
        # on ungated "general"; min_confidence 0.6 rejects a barely-tripped
        # perclos=0.25 reading, so only a well-supported drowsiness signal speaks.
        category="mood", min_confidence=0.6,
        support_cues=("under_eye_darkness", "under_eye_puffiness"),
        corroborated_by="tiredness"),

    # ----------------------------------------------------------- movement
    # Every gate bound below mirrors the emitting detector's own reporting
    # threshold, so a spec can only ever narrow what the module already chose
    # to publish — it never re-derives a finding from a raw number.
    TopicSpec(
        topic="gait_asymmetry", module="gait", key="gait_asymmetry",
        llm_intent="Gently offer a steadying hand or a place to sit while they "
                   "are moving about. Do not name a cause.",
        fallback="You're moving a little unevenly just now — would you like a "
                 "hand, or somewhere to sit for a moment?",
        priority=55, category="movement",
        # modules/gait.py only emits above 0.35; 0.15 is a floor, not a finding.
        gate=Threshold(minimum=0.15, decimals=2)),
    TopicSpec(
        topic="postural_sway", module="balance", key="postural_sway",
        llm_intent="Warmly offer something steady to hold and suggest taking "
                   "their time. Never mention falling or a cause.",
        fallback="Take your time getting steady — I'm right here if you'd like "
                 "something to hold on to.",
        priority=52, category="movement",
        gate=Threshold(minimum=0.06, decimals=3)),
    # One spec per hand: modules/tremor.py emits f"tremor_{side}", and the two
    # sides are independent readings that deserve independent no-repeat and
    # denial bookkeeping.
    TopicSpec(
        topic="tremor_hand_left", module="tremor", key="tremor_left",
        llm_intent="Kindly offer to help steady or rest their left hand. Do "
                   "not name a condition or read out a frequency.",
        fallback="Your left hand looks a little shaky just now — would resting "
                 "it somewhere steady feel better?",
        priority=50, category="movement",
        gate=Threshold(minimum=3.0, unit=" Hz")),
    TopicSpec(
        topic="tremor_hand_right", module="tremor", key="tremor_right",
        llm_intent="Kindly offer to help steady or rest their right hand. Do "
                   "not name a condition or read out a frequency.",
        fallback="Your right hand looks a little shaky just now — would "
                 "resting it somewhere steady feel better?",
        priority=50, category="movement",
        gate=Threshold(minimum=3.0, unit=" Hz")),
    TopicSpec(
        topic="movement_slowed", module="bradykinesia", key="movement_speed",
        llm_intent="Warmly offer to slow the pace down and keep them company. "
                   "Never suggest a cause.",
        fallback="You seem to be taking things a little slower today — shall "
                 "we go at an easy pace?",
        priority=48, category="movement",
        gate=Threshold(maximum=0.15, decimals=3)),
    TopicSpec(
        topic="near_fall", module="near_fall", key="recovered",
        llm_intent="Gently check they are alright after a wobble and offer to "
                   "stay close by. Do not call it a fall.",
        fallback="That looked like a bit of a wobble — are you alright? I can "
                 "stay close by if you'd like.",
        priority=75, category="movement",
        # An unsteady moment is the one thing here worth spending the health
        # budget on even though its Intent kind would already qualify; the
        # explicit True keeps that true if the kind is ever retuned.
        gate=IsTrue(), health_prompt=True),
    TopicSpec(
        topic="pacing", module="wandering", key="pacing",
        llm_intent="Warmly invite them to sit down for a moment or offer "
                   "company. Never call it wandering or agitation.",
        fallback="You've been up and about a fair bit — would you like to sit "
                 "down with me for a minute?",
        priority=45, category="movement",
        gate=Threshold(minimum=3, decimals=0)),
    TopicSpec(
        topic="stability_check", module="multimodal_reasoning",
        key="stability_check",
        llm_intent="Gently ask how they are feeling on their feet and offer to "
                   "walk along with them. State no cause.",
        fallback="How are you feeling on your feet today? I'm happy to walk "
                 "along with you if that helps.",
        priority=66, category="movement"),
    TopicSpec(
        topic="standing_change", module="multimodal_reasoning",
        key="standing_change",
        llm_intent="Kindly offer practical help with getting up — a steadier "
                   "chair, or a hand. Do not suggest a diagnosis.",
        fallback="Getting up looks like it's taking a bit more out of you "
                 "lately — would a steadier chair or a hand help?",
        priority=58, category="movement"),

    # ------------------------------------------------------------- vitals
    # modules/respiration.py `breaths_per_min` and modules/spo2.py `spo2` are
    # DELIBERATELY not specd. agent/advisor_engine.py::VitalsAdvisor already
    # folds both into its single `vitals_advice.recommendation` line (see the
    # "vitals" spec at the top of this table), so a person hears ONE gentle
    # vitals voice rather than three competing ones about the same breath.
    TopicSpec(
        topic="cough_activity", module="multimodal_reasoning",
        key="cough_activity",
        llm_intent="Gently ask how they are feeling and offer something warm "
                   "to drink. Never name an illness.",
        fallback="You've had a bit of a cough and a quiet stretch — how are "
                 "you feeling? Something warm to drink might be nice.",
        priority=64, category="vitals"),

    # --------------------------------------------------------------- skin
    TopicSpec(
        topic="facial_appearance", module="skin_vision", key="facial_appearance",
        llm_intent="Warmly ask how they are feeling today and offer a quiet "
                   "moment. Never describe their face back to them.",
        fallback="How are you feeling today? A quiet moment and something warm "
                 "to drink might be just the thing.",
        priority=40, category="skin",
        # modules/skin_vision.py publishes this at Severity.INFO (a bounded VLM
        # scan every `scan_interval`, only when a cue is positive and ungated),
        # so the spec has to admit INFO to reach it at all. See _INFO_JUSTIFIED
        # in tests/topic_table_test.py.
        min_severity=Severity.INFO,
        gate=DictCues(names=("nose_redness", "cheek_redness", "lip_dryness")),
        corroborated_by="cold_symptoms"),
    TopicSpec(
        topic="eye_redness", module="eye_redness", key="sclera_redness",
        llm_intent="Kindly suggest resting their eyes for a bit. Never name "
                   "irritation, infection, or any condition.",
        fallback="Your eyes look a little tired — would resting them for a bit "
                 "feel good?",
        priority=42, category="skin",
        gate=Threshold(minimum=6.0)),
    TopicSpec(
        topic="flat_affect", module="masked_face", key="expressiveness",
        llm_intent="Warmly ask how they are doing and offer company. Never "
                   "comment on their expression or name a condition.",
        fallback="You've been quiet for a little while — how are you doing? "
                 "I'm happy to just keep you company.",
        priority=44, category="skin",
        # NOTE: modules/masked_face.py caps its confidence at exactly 0.6, which
        # is also the low_mood FollowUpRule's max_confidence, so this handshake
        # currently always resolves to "let the corroboration engine ask".
        # That is the safe direction (a question, not a remark), but it means
        # the mention branch is unreachable until that rule's threshold moves.
        gate=Threshold(maximum=0.010, decimals=4),
        corroborated_by="low_mood"),
    TopicSpec(
        topic="nystagmus", module="eye_movement", key="nystagmus",
        llm_intent="Gently ask whether anything feels off and offer to pause "
                   "for a moment. Never name a condition.",
        fallback="How are you feeling just now — anything feeling off? We can "
                 "take a moment if you'd like.",
        priority=46, category="skin",
        gate=Threshold(minimum=3.0, unit=" Hz")),
    # modules/grooming.py:82-84 emits f"{key}_change" for exactly two tracked
    # keys. Enumerated explicitly rather than prefix-matched: a spec owns ONE
    # (module, key) pair, and the uniqueness invariant plus covered_keys() both
    # depend on that being literal.
    TopicSpec(
        topic="grooming_change_hair", module="grooming", key="hair_texture_change",
        llm_intent="Warmly offer practical help with getting ready for the "
                   "day. Never comment on how they look.",
        fallback="Would you like a hand with anything today — brushing your "
                 "hair, or getting sorted for the morning?",
        priority=38, category="skin",
        gate=Threshold(minimum=1.0, decimals=2)),
    TopicSpec(
        topic="grooming_change_jaw", module="grooming", key="jaw_texture_change",
        llm_intent="Warmly offer practical help with freshening up. Never "
                   "comment on how they look.",
        fallback="Would you like a hand with anything this morning — a shave, "
                 "or a bit of a freshen-up?",
        priority=38, category="skin",
        gate=Threshold(minimum=1.0, decimals=2)),

    # ------------------------------------------------------------ routine
    TopicSpec(
        topic="activity_drop", module="activity_level", key="activity_drop",
        llm_intent="Warmly offer something easy to do together. Never frame "
                   "the quiet stretch as a problem.",
        fallback="It's been a quiet stretch — would you like to stretch your "
                 "legs, or shall I put something on?",
        priority=47, category="routine",
        gate=Threshold(maximum=0.4, decimals=2)),
    TopicSpec(
        topic="hydration_window", module="routine", key="hours_since_drink",
        llm_intent="Gently offer a glass of water. Keep it light — this is a "
                   "kind reminder, not a concern.",
        fallback="It's been a little while since your last drink — would a "
                 "glass of water sound good about now?",
        priority=36, category="routine",
        # agent/routines.py publishes the elapsed-hours counter at
        # Severity.INFO because the number itself is not noteworthy; the gate
        # below is what makes it worth a word. See _INFO_JUSTIFIED in
        # tests/topic_table_test.py.
        min_severity=Severity.INFO,
        gate=Threshold(minimum=4.0, unit=" hours"),
        corroborated_by="hydration"),
    TopicSpec(
        topic="hydration_opportunity", module="multimodal_reasoning",
        key="hydration_opportunity",
        llm_intent="Warmly point out that a drink is close by and invite them "
                   "to have some.",
        fallback="There's a drink close by — would now be a good moment for a "
                 "sip?",
        priority=44, category="routine"),
    TopicSpec(
        topic="night_activity", module="routine", key="unusual_night_activity",
        llm_intent="Gently check everything is alright and offer help. Never "
                   "imply they should be asleep.",
        fallback="You're up a little outside your usual hours — is everything "
                 "alright? I'm here if you need anything.",
        priority=54, category="routine", gate=IsTrue()),
    TopicSpec(
        topic="missed_meal", module="routine", key="missed_meal_opportunity",
        llm_intent="Warmly offer something to eat. Never say they missed a "
                   "meal or imply they forgot.",
        fallback="It's been a while since your usual mealtime — would you like "
                 "something to eat?",
        priority=42, category="routine", gate=IsTrue()),
    TopicSpec(
        topic="missed_drink", module="routine", key="missed_drink_opportunity",
        llm_intent="Warmly offer a drink. Never say they missed one or imply "
                   "they forgot.",
        fallback="It's been a while since your usual cup — would a cup of tea "
                 "sound good?",
        priority=42, category="routine", gate=IsTrue()),
    TopicSpec(
        topic="daily_summary", module="routine", key="daily_summary",
        llm_intent="Offer a short, warm round-up of the day if they'd like "
                   "one. Nothing clinical, nothing alarming.",
        fallback="Would you like a little round-up of how today went?",
        priority=30, category="routine",
        # agent/routines.py publishes the end-of-day roll-up at Severity.INFO
        # (it is a summary, not a new finding) and at most once per day. See
        # _INFO_JUSTIFIED in tests/topic_table_test.py.
        min_severity=Severity.INFO,
        gate=NonEmptyList(max_items=3)),

    # -------------------------------------------------------- environment
    # modules/scene_vision.py `objects` and `activities` are DELIBERATELY not
    # specd: both are Severity.INFO room inventories, and agent/routines.py
    # already consumes them into the `routine.*` meal/drink/medication
    # opportunities specd above. A spec here would just narrate the furniture.
    TopicSpec(
        topic="scene_hazard", module="scene_vision", key="hazards",
        llm_intent="Warmly offer to help with something in the room. This is "
                   "about the room, never about the person's health.",
        fallback="I noticed something worth a look over there — {value}. Would "
                 "you like a hand with it?",
        priority=72, category="environment",
        # Only the confirmed list (seen twice in ten minutes) is published at
        # NOTICE; a one-off VLM guess stays INFO and is filtered out here.
        gate=NonEmptyList(),
        detail="value",
        # A tidy-up offer is not a health check-in, so it must not consume the
        # hourly health-prompt budget that real check-ins depend on.
        health_prompt=False),
)


# ----------------------------------------------------------------- rendering

def render_value(spec: TopicSpec, result: Result) -> str:
    """Return the spec's string form of an observed value.

    A spec with a `gate` gets the gate's bounded rendering. A spec without one
    keeps the raw `str(value)` the hand-written policy used, so migrated topics
    produce byte-identical signatures and fallbacks.
    """
    if spec.gate is not None:
        return spec.gate.render(result.value)
    return str(result.value)


def detail_text(spec: TopicSpec, result: Result, rendered_value: str) -> str:
    """Return the facts handed to the phrasing model for this spec."""
    if spec.detail == "message":
        return str(result.message)
    if spec.detail == "value":
        return rendered_value
    return spec.detail


def signature_for(spec: TopicSpec, rendered_value: str) -> str:
    """Return the no-repeat bookkeeping signature for this candidate."""
    if spec.signature == "topic:value":
        return f"{spec.topic}:{rendered_value}"
    return spec.topic


def render_fallback(spec: TopicSpec, result: Result, mem: ObservationMemory,
                    rendered_value: str) -> str:
    """Fill a spec's hand-authored fallback template with bounded strings.

    The format string is always a constant declared in this module, so observed
    data only ever arrives as a format *argument*. Every substitution is
    pre-stringified into a plain dict, which makes attribute traversal such as
    `{value.__class__}` unreachable and turns an unknown placeholder into a
    loud KeyError rather than a silent leak.
    """
    fields = {
        "value": str(rendered_value),
        "message": str(result.message),
        "name": str(mem.name),
        "tod": str(mem.time_of_day()),
        "detail": str(detail_text(spec, result, rendered_value)),
    }
    return spec.fallback.format_map(fields)


def support_clause(mem: ObservationMemory, keys, trailing: str) -> str:
    """Return the " Supporting visible appearance cues: ..." clause, or "".

    `keys` are appearance-cue names in the order they should be listed; only
    those currently present in `ObservationMemory.facial_cues()` are named.
    `trailing` is the closing sentence, which differs per call site (see
    SUPPORT_TRAILING_OBSERVATION / SUPPORT_TRAILING_CHECK_IN) and is preserved
    verbatim rather than unified.
    """
    if not keys:
        return ""
    cues = mem.facial_cues()
    present = [str(key).replace("_", " ") for key in keys if key in cues]
    if not present:
        return ""
    return (" Supporting visible appearance cues: " + ", ".join(present) +
            ". " + trailing)


# ------------------------------------------------------------- de-confliction

def _corroboration_admits(spec: TopicSpec, result: Result,
                          corroboration) -> bool:
    """Decide the ask-versus-mention handshake for one spec.

    A spec that names `corroborated_by` shares its Result with a
    `FollowUpRule`. Below the rule's `max_confidence` the corroboration engine
    owns the signal and the agent ASKS; above it the engine declined the cue as
    confident enough not to need asking and the agent MENTIONS. Either way,
    never both — and never while a question for that rule is still pending.

    `corroboration=None` means "no engine to consult", which admits everything
    so callers that only want the raw table keep working.
    """
    if spec.corroborated_by is None or corroboration is None:
        return True
    rule, status = corroboration.rule_state(spec.corroborated_by)
    if rule is None:
        return True
    confidence = _number(result.confidence)
    if confidence is None or confidence <= rule.max_confidence:
        return False                       # the corroboration engine may ask
    return status not in ("flagged", "asked")   # no question pending


def covered_keys() -> frozenset[tuple[str, str]]:
    """Return the `(module, key)` pairs this layer already speaks for.

    Used by `agent/conversation.py::TopicQueue` so a signal owned by the table
    (or by a corroboration follow-up rule) is not *also* promoted as an
    unauthored "By the way, ..." proactive line in a second voice.
    """
    keys = {(spec.module, spec.key) for spec in TOPICS if spec.enabled}
    keys.update((rule.module, rule.key) for rule in DEFAULT_RULES)
    return frozenset(keys)


# ------------------------------------------------------------------- building

def build_intent(spec: TopicSpec, mem: ObservationMemory, now: float,
                 corroboration=None):
    """Return the Intent this spec would raise now, or None.

    Gate chain, in order: enabled -> the observation exists -> it is not
    expired -> severity floor -> confidence floor -> quality floor -> value
    gate -> corroboration handshake. Intent metadata mirrors the hand-written
    policy exactly: `quality` is passed through unmodified (bounding lives in
    `AttentionPlanner._bounded`) and `novelty` keeps its 1.0 default.
    """
    # Imported here: agent/policy.py imports this module, so a module-level
    # import would close the cycle.
    from agent.policy import Intent

    if not spec.enabled:
        return None
    result = mem.get(spec.module, spec.key)
    if result is None or result.expired:
        return None
    if _ORDER[result.severity] < _ORDER[spec.min_severity]:
        return None
    confidence = _number(result.confidence)
    if confidence is None or confidence < spec.min_confidence:
        return None
    if result.quality is not None:
        quality = _number(result.quality)
        if quality is None or quality < spec.min_quality:
            return None
    if spec.gate is not None and not spec.gate.passes(result.value):
        return None
    if not _corroboration_admits(spec, result, corroboration):
        return None

    rendered = render_value(spec, result)
    llm_intent = spec.llm_intent
    if spec.support_cues:
        llm_intent += support_clause(mem, spec.support_cues,
                                     SUPPORT_TRAILING_OBSERVATION)
    return Intent(
        spec.kind, signature_for(spec, rendered), llm_intent,
        detail_text(spec, result, rendered),
        render_fallback(spec, result, mem, rendered), spec.priority,
        confidence=result.confidence, quality=result.quality,
        health_prompt=spec.health_prompt, topic=spec.topic,
        severity_score=float(_ORDER[result.severity]),
        category=spec.category)
