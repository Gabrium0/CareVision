"""LLM-steered topic selection + broadened rule coverage (agent/corroboration.py).

Steering keeps the "LLM proposes, deterministic disposes" invariant: an LLM
selector may reorder which *already-flagged* topic is raised, but its choice is
membership-checked and any invalid/None/raising selector falls back to the
deterministic oldest-flagged topic. Coverage tests confirm the new rules flag on
the exact detector keys the investigation verified. All offline — no network.
"""
from __future__ import annotations

from agent.corroboration import CorroborationEngine, DEFAULT_RULES
from agent.moondream_client import MoondreamClient
from core.events import Result, Severity


def _engine():
    return CorroborationEngine(rules=list(DEFAULT_RULES))


def _hit(module, key, severity=Severity.NOTICE, confidence=0.3):
    return Result(module=module, key=key, value=1, message="cue",
                  severity=severity, confidence=confidence)


def _flag_two(engine):
    """Flag skin_changes (older) then tiredness (newer)."""
    engine.observe([_hit("rash", "rash")], now=100.0)
    engine.observe([_hit("drowsiness", "perclos")], now=200.0)


# --------------------------------------------------------------- flagged_topics

def test_flagged_topics_oldest_first():
    engine = _engine()
    _flag_two(engine)
    ids = [t for t, _ in engine.flagged_topics(300.0)]
    assert ids == ["skin_changes", "tiredness"]


# ---------------------------------------------------------- steering behavior

def test_valid_selector_choice_is_honored_over_oldest():
    engine = _engine()
    _flag_two(engine)
    topic, _rule = engine.next_question_steered(300.0, lambda cands: "tiredness")
    assert topic == "tiredness"  # not the oldest (skin_changes)


def test_invalid_choice_falls_back_to_oldest():
    engine = _engine()
    _flag_two(engine)
    topic, _rule = engine.next_question_steered(300.0, lambda cands: "not_a_topic")
    assert topic == "skin_changes"


def test_none_and_raising_selectors_fall_back_to_deterministic():
    engine = _engine()
    _flag_two(engine)
    assert engine.next_question_steered(300.0, lambda c: None)[0] == "skin_changes"

    def boom(_cands):
        raise RuntimeError("selector exploded")
    assert engine.next_question_steered(300.0, boom)[0] == "skin_changes"


def test_no_selector_matches_next_question():
    engine = _engine()
    _flag_two(engine)
    assert engine.next_question_steered(300.0, None) == engine.next_question(300.0)


def test_selector_is_memoized_per_flagged_set():
    engine = _engine()
    _flag_two(engine)
    calls = []

    def selector(cands):
        calls.append(tuple(t for t, _ in cands))
        return cands[-1][0]

    engine.next_question_steered(300.0, selector)
    engine.next_question_steered(301.0, selector)
    assert len(calls) == 1  # same flagged set -> selector runs once

    # A new flagged topic changes the set -> selector re-runs.
    engine.observe([_hit("agitation", "agitation")], now=400.0)
    engine.next_question_steered(401.0, selector)
    assert len(calls) == 2


def test_no_flagged_topics_returns_none_and_clears_cache():
    engine = _engine()
    assert engine.next_question_steered(100.0, lambda c: "x") is None
    assert engine._steer_cache is None


# ------------------------------------------------- MoondreamClient.select_topic

def _bare_client():
    return MoondreamClient.__new__(MoondreamClient)  # skip network/key __init__


def test_select_topic_returns_valid_offered_id():
    client = _bare_client()
    client._complete = lambda prompt, kind: "tiredness"
    cands = [("hydration", "Had enough to drink?"), ("tiredness", "Rested well?")]
    assert client.select_topic(cands, "ctx") == "tiredness"


def test_select_topic_rejects_hallucinated_or_none():
    client = _bare_client()
    cands = [("hydration", "Q1"), ("tiredness", "Q2")]
    client._complete = lambda p, k: "banana"
    assert client.select_topic(cands, "ctx") is None
    client._complete = lambda p, k: "none"
    assert client.select_topic(cands, "ctx") is None
    client._complete = lambda p, k: None          # offline / circuit open
    assert client.select_topic(cands, "ctx") is None


def test_select_topic_empty_candidates():
    assert _bare_client().select_topic([], "ctx") is None


# -------------------------------------------------------- broadened coverage

def test_new_rules_present():
    engine = _engine()
    for topic in ("discomfort", "puffiness", "recent_injury", "tiredness",
                  "restlessness"):
        assert topic in engine.rules


def test_new_rules_flag_on_verified_keys():
    cases = [
        ("discomfort", "pain", "pain", Severity.WARNING),
        ("puffiness", "facial_swelling", "swelling", Severity.NOTICE),
        ("recent_injury", "bruise", "bruise_fraction", Severity.NOTICE),
        ("tiredness", "drowsiness", "perclos", Severity.NOTICE),
        ("restlessness", "agitation", "agitation", Severity.NOTICE),
    ]
    for topic, module, key, severity in cases:
        engine = _engine()
        engine.observe([_hit(module, key, severity, confidence=0.3)], now=100.0)
        assert engine.status(topic) == "flagged", f"{topic} did not flag"


def test_high_confidence_hit_does_not_flag_a_gentle_question():
    # A strong cue (above max_confidence) is left to the alert path, not a
    # tentative check-in.
    engine = _engine()
    engine.observe([_hit("pain", "pain", Severity.WARNING, confidence=0.9)],
                   now=100.0)
    assert engine.status("discomfort") is None
