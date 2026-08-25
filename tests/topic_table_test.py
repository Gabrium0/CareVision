"""Invariants and golden output for the declarative topic table.

`agent/topics.py` turns detector observations into conversation candidates from
data rather than a hand-written if-chain. These tests pin the properties that
make that safe: no duplicate ownership, no INFO-severity chatter, exactly one
of ask-or-mention per shared signal, bounded format templates, and structural
unreachability of agent-only results. The golden test pins the four migrated
candidates character-for-character against the wording the if-chain produced.

Everything is offline and synthetic; no camera, model, or network.
"""
from __future__ import annotations

import string
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.corroboration import CorroborationEngine, DEFAULT_RULES
from agent.policy import Policy
from agent.state import ObservationMemory
from agent.topics import (
    ALLOWED_PLACEHOLDERS, DictCues, DictField, IsTrue, KNOWN_CATEGORIES,
    NonEmptyList, OneOf, SIGNATURE_STYLES, SUPPORT_TRAILING_CHECK_IN,
    SUPPORT_TRAILING_OBSERVATION, TOPICS, Threshold, TopicSpec, apply_config,
    build_intent, covered_keys, support_clause,
)
from core.events import Result, Severity, Visibility

# Specs allowed to fire on Severity.INFO, each with the reason it cannot become
# chatter. The bar is deliberately high: modules/presence.py publishes an INFO
# `presence.present` every single frame, and tests/agent_alert_test.py asserts
# the agent stays silent while that is streaming, so an unjustified INFO spec
# would turn a per-frame status row into conversation.
_INFO_JUSTIFIED: frozenset[tuple[str, str]] = frozenset({
    # agent/routines.py:129 - a "hours since the last drink" counter is INFO
    # because the number is not itself noteworthy. It is emitted at most once
    # per RoutineReasoner interval (30s), the spec's Threshold(minimum=4.0)
    # admits it only after a four-hour gap, and the "routine" category gap
    # (900s) rate-limits it further.
    ("routine", "hours_since_drink"),
    # agent/routines.py:181 - the end-of-day roll-up is INFO because it
    # summarises changes already reported, not a new finding. Guarded by
    # `_last_summary_day`, so it is emitted at most once per subject per day.
    ("routine", "daily_summary"),
    # modules/skin_vision.py:2535 - facial appearance cues are INFO because a
    # VLM cue is a prior, never a finding. Emitted at most once per
    # `scan_interval` (60s), only when a cue is positive and ungated, and the
    # spec additionally defers to the cold_symptoms follow-up rule.
    ("skin_vision", "facial_appearance"),
})


def _res(module, key, value, conf=0.8, severity=Severity.NOTICE, message="msg",
         **kwargs):
    return Result(module=module, key=key, value=value, confidence=conf,
                  severity=severity, message=message, ttl=60.0, **kwargs)


# ------------------------------------------------------------- invariants

def test_module_key_pairs_are_unique_across_the_table():
    pairs = [(spec.module, spec.key) for spec in TOPICS]
    assert len(pairs) == len(set(pairs)), f"duplicate ownership: {pairs}"


def test_no_spec_fires_on_info_severity_without_justification():
    offenders = [(spec.module, spec.key) for spec in TOPICS
                 if spec.min_severity == Severity.INFO
                 and (spec.module, spec.key) not in _INFO_JUSTIFIED]
    assert offenders == [], (
        "INFO-severity specs turn per-frame status rows into conversation; "
        f"add a justification to _INFO_JUSTIFIED or raise min_severity: {offenders}")


def test_every_corroborated_by_names_a_real_follow_up_rule():
    topics = {rule.topic for rule in DEFAULT_RULES}
    for spec in TOPICS:
        if spec.corroborated_by is not None:
            assert spec.corroborated_by in topics, spec.topic


def test_every_signal_shared_with_a_follow_up_rule_declares_the_handshake():
    for spec in TOPICS:
        match = next((rule for rule in DEFAULT_RULES
                      if rule.module == spec.module
                      and spec.key.startswith(rule.key)), None)
        if match is None:
            continue
        assert spec.corroborated_by == match.topic, (
            f"{spec.topic} shares {spec.module}.{spec.key} with rule "
            f"{match.topic!r} but does not declare corroborated_by")


def test_fallback_placeholders_stay_inside_the_allowed_set():
    parser = string.Formatter()
    for spec in TOPICS:
        names = [field for _text, field, _spec, _conv
                 in parser.parse(spec.fallback) if field is not None]
        assert set(names) <= ALLOWED_PLACEHOLDERS, (spec.topic, names)
        # Exact membership also rules out attribute/index traversal such as
        # "{value.__class__}" or "{value[0]}", which would not compare equal.
        for name in names:
            assert name in ALLOWED_PLACEHOLDERS


def test_every_category_and_signature_style_is_known():
    for spec in TOPICS:
        assert spec.category in KNOWN_CATEGORIES, spec.topic
        assert spec.signature in SIGNATURE_STYLES, spec.topic


def test_covered_keys_spans_the_table_and_the_follow_up_rules():
    covered = covered_keys()
    for spec in TOPICS:
        assert (spec.module, spec.key) in covered
    for rule in DEFAULT_RULES:
        assert (rule.module, rule.key) in covered


def test_topic_ids_are_unique_across_the_table():
    # `topic` keys the AttentionPlanner's per-topic denial cooldown and the
    # policy's no-repeat bookkeeping, so two specs sharing one id would make a
    # single "not now" silence both.
    ids = [spec.topic for spec in TOPICS]
    assert len(ids) == len(set(ids)), f"duplicate topic id: {ids}"


def test_every_gate_admits_the_value_shape_its_module_publishes():
    # A gate mismatched to the emitted value type is the silent failure this
    # table is most exposed to (the sweating rule went months without firing
    # for the same class of reason), so each gate is exercised against one
    # representative value of the shape its detector actually publishes.
    samples = {
        "gait_asymmetry": 0.42, "postural_sway": 0.081,
        "tremor_hand_left": 5.2, "tremor_hand_right": 5.2,
        "movement_slowed": 0.031, "near_fall": True, "pacing": 9,
        "eye_redness": 9.4, "flat_affect": 0.0062, "nystagmus": 4.8,
        "grooming_change_hair": 1.31, "grooming_change_jaw": 1.31,
        "activity_drop": 0.22, "hydration_window": 5.5,
        "night_activity": True, "missed_meal": True, "missed_drink": True,
        "daily_summary": ["Activity outside the usual window."],
        "scene_hazard": ["a rug edge lifting"],
        "facial_appearance": {"cues": {"cheek_redness": "mild"},
                              "confidence": 0.72},
    }
    for spec in TOPICS:
        if spec.gate is None:
            continue
        assert spec.topic in samples, f"add a sample value for {spec.topic}"
        assert spec.gate.passes(samples[spec.topic]), spec.topic
        assert spec.gate.render(samples[spec.topic]) != "", spec.topic


def test_repaired_sweating_rule_matches_the_emitted_key():
    # modules/sweating.py emits "sweat_gloss"; the rule used to name "sweating".
    rule = next(r for r in DEFAULT_RULES if r.topic == "feeling_warm")
    assert rule.key == "sweat_gloss"
    engine = CorroborationEngine()
    engine.observe([_res("sweating", "sweat_gloss", 0.09, conf=0.3)], now=100.0)
    assert engine.status("feeling_warm") == "flagged"


# ------------------------------------------------------------------ gates

def test_threshold_gate_admits_range_and_rejects_non_numbers():
    gate = Threshold(minimum=1.0, maximum=10.0, unit=" bpm", decimals=1)
    assert gate.passes(5) and gate.passes(1.0) and gate.passes(10.0)
    assert not gate.passes(0.5) and not gate.passes(11)
    assert not gate.passes(True)          # bool is never a measurement
    assert not gate.passes(None) and not gate.passes("...") and not gate.passes("x")
    assert not gate.passes(float("nan"))
    assert gate.render(5) == "5.0 bpm"


def test_is_true_gate_requires_the_literal_true():
    gate = IsTrue()
    assert gate.passes(True)
    assert not gate.passes(1) and not gate.passes("yes") and not gate.passes([1])
    assert not gate.passes(False) and not gate.passes(None)
    assert gate.render(True) == "yes"


def test_one_of_gate_is_a_closed_vocabulary():
    gate = OneOf(("mild", "marked"))
    assert gate.passes("mild") and gate.passes(" marked ")
    assert not gate.passes("severe") and not gate.passes(None) and not gate.passes(1)
    assert gate.render("mild") == "mild"


def test_dict_field_gate_delegates_to_its_inner_gate():
    gate = DictField("score", Threshold(minimum=0.5, decimals=2))
    assert gate.passes({"score": 0.8})
    assert not gate.passes({"score": 0.1})
    assert not gate.passes({"other": 0.8}) and not gate.passes("nope")
    assert gate.render({"score": 0.8}) == "0.80"


def test_dict_cues_gate_matches_named_appearance_cues():
    gate = DictCues(names=("under_eye_darkness",))
    value = {"cues": {"under_eye_darkness": "mild", "lip_dryness": "marked"}}
    assert gate.passes(value)
    assert not gate.passes({"cues": {"lip_dryness": "marked"}})
    assert not gate.passes({"cues": "not-a-dict"}) and not gate.passes(None)
    assert gate.render(value) == "under eye darkness mild"


def test_non_empty_list_gate_bounds_its_rendering():
    gate = NonEmptyList(minimum=2, max_items=2)
    assert gate.passes(["a", "b", "c"])
    assert not gate.passes(["a"]) and not gate.passes("ab") and not gate.passes(None)
    assert gate.render(["a", "b", "c"]) == "a, b"


def test_support_clause_preserves_each_call_sites_trailing_sentence():
    mem = ObservationMemory()
    mem.ingest([_res("skin_vision", "facial_appearance",
                     {"cues": {"under_eye_darkness": "mild"}},
                     conf=0.8, severity=Severity.INFO)], now=100.0)
    observation = support_clause(mem, ("under_eye_darkness", "under_eye_puffiness"),
                                 SUPPORT_TRAILING_OBSERVATION)
    assert observation == (
        " Supporting visible appearance cues: under eye darkness. "
        "Treat them only as corroboration, not as a cause or diagnosis.")
    check_in = support_clause(mem, ("under_eye_darkness",), SUPPORT_TRAILING_CHECK_IN)
    assert check_in == (
        " Supporting visible appearance cues: under eye darkness. "
        "Use them only to phrase the check-in; do not state a cause or diagnosis.")
    assert support_clause(mem, (), SUPPORT_TRAILING_CHECK_IN) == ""
    assert support_clause(mem, ("not_a_cue",), SUPPORT_TRAILING_CHECK_IN) == ""


# ---------------------------------------------------------- config overlay

def _spec(topic: str) -> TopicSpec:
    return next(s for s in TOPICS if s.topic == topic)


def test_overrides_retune_the_numeric_admission_gates():
    tuned = _spec("gait_asymmetry").apply_overrides(
        {"enabled": True, "min_confidence": 0.45, "priority": 52,
         "min_quality": 0.2, "min_severity": "notice"})
    assert (tuned.min_confidence, tuned.priority) == (0.45, 52)
    assert tuned.min_quality == 0.2 and tuned.min_severity == Severity.NOTICE
    assert _spec("gait_asymmetry").min_confidence == 0.35   # original untouched


def test_overrides_can_retune_a_gates_numeric_bounds_only():
    tuned = _spec("hydration_window").apply_overrides({"gate": {"minimum": 6.0}})
    assert tuned.gate.minimum == 6.0
    assert tuned.gate.unit == " hours"          # rendering is not config
    for bad in ({"gate": {"unit": " h"}}, {"gate": {"decimals": 3}},
                {"gate": {"names": ("x",)}}):
        with pytest.raises(ValueError):
            _spec("hydration_window").apply_overrides(bad)
    with pytest.raises(ValueError):             # spec has no gate at all
        _spec("pain").apply_overrides({"gate": {"minimum": 1.0}})


def test_prose_and_routing_fields_are_never_overridable():
    # The whole point of the overlay: deployment may retune WHETHER a topic is
    # raised, never WHAT is said, nor which observation it is said about. The
    # fallback text is what the safety airlocks fall back TO.
    for field in ("fallback", "llm_intent", "detail", "topic", "module", "key",
                  "category", "signature", "kind", "corroborated_by",
                  "support_cues", "health_prompt"):
        with pytest.raises(ValueError) as excinfo:
            _spec("gait_asymmetry").apply_overrides({field: "anything"})
        assert field in str(excinfo.value)


def test_a_typo_raises_instead_of_being_silently_ignored():
    with pytest.raises(ValueError) as excinfo:
        _spec("gait_asymmetry").apply_overrides({"min_confidance": 0.5})
    assert "min_confidance" in str(excinfo.value)
    for bad in ({"enabled": "yes"}, {"priority": "high"},
                {"min_confidence": 1.5}, {"min_confidence": float("nan")},
                {"min_severity": "urgent"}, {"min_quality": None}):
        with pytest.raises(ValueError):
            _spec("gait_asymmetry").apply_overrides(bad)
    with pytest.raises(ValueError):
        _spec("gait_asymmetry").apply_overrides(["not", "a", "mapping"])


def test_apply_config_rejects_an_unknown_topic_id():
    with pytest.raises(ValueError) as excinfo:
        apply_config({"gate_asymmetry": {"enabled": False}})
    assert "gate_asymmetry" in str(excinfo.value)
    assert apply_config(None) == TOPICS
    assert apply_config({}) == TOPICS


def test_apply_config_disables_a_topic_without_deleting_it():
    table = apply_config({"pacing": {"enabled": False}})
    assert [s.topic for s in table] == [s.topic for s in TOPICS]
    assert next(s for s in table if s.topic == "pacing").enabled is False
    assert covered_keys() >= {(s.module, s.key) for s in TOPICS if s.enabled}


def test_the_shipped_config_overlay_loads_cleanly():
    # config/modules.yaml is deployment truth; a stale topic id or a typo there
    # must fail here rather than at runtime on a robot.
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config" / "modules.yaml")
        .read_text(encoding="utf-8"))
    conversation = cfg.get("conversation") or {}
    table = apply_config(conversation.get("topics"))
    assert len(table) == len(TOPICS)
    for category in (conversation.get("category_gaps") or {}):
        assert category in KNOWN_CATEGORIES, category


# ------------------------------------------------------- privacy structure

def test_agent_only_result_for_a_specd_key_yields_no_candidate():
    spec = next(s for s in TOPICS if s.topic == "pain")
    mem = ObservationMemory()
    mem.ingest([_res(spec.module, spec.key, 0.9, conf=0.9,
                     severity=Severity.WARNING,
                     visibility=Visibility.AGENT_ONLY)], now=100.0)
    # ObservationMemory.ingest skips AGENT_ONLY before writing to `latest`, so
    # the table cannot see it at all — no gate has to be trusted for this.
    assert mem.get(spec.module, spec.key) is None
    assert build_intent(spec, mem, 100.0) is None
    policy = Policy(small_talk_interval=1e9)
    assert all(intent.signature != "pain"
               for intent in policy._candidates(mem, 100.0))


# ---------------------------------------------------------- the handshake

def _pain_memory(confidence: float) -> ObservationMemory:
    mem = ObservationMemory()
    mem.ingest([_res("pain", "pain", 0.8, conf=confidence,
                     severity=Severity.WARNING,
                     message="Possible pain/grimacing expression")], now=100.0)
    return mem


def _pain_spec() -> TopicSpec:
    return next(s for s in TOPICS if s.topic == "pain")


def test_confident_signal_with_an_idle_rule_is_mentioned():
    engine = CorroborationEngine()
    rule, status = engine.rule_state("discomfort")
    assert rule is not None and status is None
    mem = _pain_memory(rule.max_confidence + 0.1)
    intent = build_intent(_pain_spec(), mem, 100.0, corroboration=engine)
    assert intent is not None and intent.signature == "pain"


def test_low_confidence_signal_is_left_to_the_corroboration_ask():
    engine = CorroborationEngine()
    rule, _status = engine.rule_state("discomfort")
    mem = _pain_memory(rule.max_confidence)      # at the threshold: engine asks
    assert build_intent(_pain_spec(), mem, 100.0, corroboration=engine) is None


def test_flagged_or_asked_rule_suppresses_the_mention():
    for status in ("flagged", "asked"):
        engine = CorroborationEngine()
        engine.observe([_res("pain", "pain", 0.4, conf=0.3,
                             severity=Severity.WARNING)], now=100.0)
        if status == "asked":
            engine.mark_asked("discomfort", 101.0)
        assert engine.status("discomfort") == status
        rule, _ = engine.rule_state("discomfort")
        mem = _pain_memory(rule.max_confidence + 0.2)
        assert build_intent(_pain_spec(), mem, 102.0, corroboration=engine) is None, status


def test_no_engine_admits_every_spec():
    # tests/corroboration_and_elicitation_test.py calls _candidates(mem, now)
    # positionally; the keyword-only None default must keep that working.
    mem = _pain_memory(0.4)
    assert build_intent(_pain_spec(), mem, 100.0) is not None


def test_drowsiness_double_fire_is_resolved_in_exactly_one_direction():
    spec = next(s for s in TOPICS if s.topic == "tired")
    rule = next(r for r in DEFAULT_RULES if r.topic == "tiredness")
    for confidence, expect_ask in ((0.45, True), (0.75, False)):
        engine = CorroborationEngine()
        snapshot = [_res("drowsiness", "perclos", 0.25, conf=confidence,
                         message="Drowsiness: eyes closed 25% of the time")]
        engine.observe(snapshot, now=100.0)
        mem = ObservationMemory()
        mem.ingest(snapshot, now=100.0)
        asked = engine.status("tiredness") == "flagged"
        mentioned = build_intent(spec, mem, 100.0, corroboration=engine) is not None
        assert asked is expect_ask
        assert asked != mentioned, (confidence, rule.max_confidence)


# ------------------------------------------------------ golden regression

def _golden_memory() -> ObservationMemory:
    mem = ObservationMemory(name="Ada")
    mem.ingest([
        _res("clothing_advice", "recommendation", "Bring a layer.", conf=0.6),
        _res("vitals_advice", "recommendation", "Take it easy.", conf=0.7),
        _res("pain", "pain", 0.8, conf=0.9, severity=Severity.WARNING,
             message="Possible pain/grimacing expression (check on person)"),
        _res("drowsiness", "perclos", 0.25, conf=0.75,
             message="Drowsiness: eyes closed 25% of the time"),
    ], now=100.0)
    return mem


GOLDEN = {
    "clothing": {
        "signature": "clothing:Bring a layer.",
        "priority": 60,
        "category": "general",
        "fallback": "Bring a layer.",
        "llm_intent": "Gently mention what you noticed about their clothing "
                      "versus the weather and offer a suggestion.",
    },
    "vitals": {
        "signature": "vitals:Take it easy.",
        "priority": 65,
        "category": "general",
        "fallback": "Take it easy.",
        "llm_intent": "Gently mention the health observation without diagnosing, "
                      "and suggest a calm check-in or rest.",
    },
    "pain": {
        "signature": "pain",
        "priority": 70,
        "category": "general",
        "fallback": "You look a little uncomfortable — are you okay?",
        "llm_intent": "Gently ask if they are comfortable or in any discomfort.",
    },
    "tired": {
        "signature": "tired",
        "priority": 50,
        # "mood" (600s gap) rate-limits the drowsiness prompt; it was "general"
        # (never category-gated) before the false-positive tuning.
        "category": "mood",
        "fallback": "You seem a little tired — how are you feeling? "
                    "A short rest might feel good.",
        "llm_intent": "Kindly ask how they are feeling and suggest a rest if "
                      "they'd like.",
    },
}


def test_migrated_candidates_match_the_hand_written_wording_exactly():
    mem = _golden_memory()
    built = {spec.topic: build_intent(spec, mem, 100.0) for spec in TOPICS}
    assert set(GOLDEN) <= set(built)
    # Memory holding only the four legacy observations must produce only the
    # four legacy candidates: no later spec may fire off another spec's data.
    assert [topic for topic, intent in built.items()
            if intent is not None] == list(GOLDEN)
    for topic, expected in GOLDEN.items():
        intent = built[topic]
        assert intent is not None, topic
        assert intent.signature == expected["signature"], topic
        assert intent.priority == expected["priority"], topic
        assert intent.fallback == expected["fallback"], topic
        assert intent.llm_intent == expected["llm_intent"], topic
        assert intent.kind == "observation", topic
        assert intent.category == expected["category"], topic


def test_migrated_candidates_carry_the_same_ranking_metadata():
    mem = _golden_memory()
    by_topic = {spec.topic: build_intent(spec, mem, 100.0) for spec in TOPICS
                if spec.topic in GOLDEN}
    assert by_topic["clothing"].confidence == 0.6
    assert by_topic["clothing"].severity_score == 1.0     # NOTICE
    assert by_topic["pain"].severity_score == 2.0         # WARNING
    assert by_topic["clothing"].detail == "Bring a layer."
    assert by_topic["pain"].detail == (
        "Possible pain/grimacing expression (check on person)")
    # quality is passed through unmodified; bounding lives in the planner.
    assert all(intent.quality is None for intent in by_topic.values())
    assert all(intent.novelty == 1.0 for intent in by_topic.values())
    assert all(intent.health_prompt is None for intent in by_topic.values())


def test_policy_emits_the_table_candidates_in_declared_order():
    mem = _golden_memory()
    policy = Policy(small_talk_interval=1e9)
    table = {spec.topic for spec in TOPICS}
    signatures = [intent.signature for intent in policy._candidates(mem, 100.0)
                  if intent.topic in table]
    assert signatures == ["clothing:Bring a layer.", "vitals:Take it easy.",
                          "pain", "tired"]
