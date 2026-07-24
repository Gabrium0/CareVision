"""Safety airlock + research telemetry for the visual-prior -> gentle-question
corroboration loop (agent/corroboration.py).

The conversation planner lets an LLM phrase the check-in question from a
low-confidence visual prior, but a generated line must never assert a finding,
name a condition, or accuse. safe_check_in() is the airlock: unsafe or empty
generations fall back to the deterministic, hand-authored rule text. These
tests pin that contract without any network/LLM.
"""
from __future__ import annotations

import pytest

from agent.corroboration import (CorroborationEngine, DEFAULT_RULES,
                                 safe_check_in)

_FALLBACK = "Have you noticed any skin changes or irritation lately?"


def test_gentle_generated_question_passes_through():
    good = "How have you been feeling today — anything on your mind?"
    assert safe_check_in(good, _FALLBACK) == good


@pytest.mark.parametrize("unsafe", [
    "It looks like you have a rash — you should see a doctor.",
    "You're showing signs of a stroke.",
    "This looks like an infection to me.",
    "You look unwell today.",
    "That could be a symptom of dementia.",
])
def test_unsafe_generation_falls_back_to_hand_authored_text(unsafe):
    assert safe_check_in(unsafe, _FALLBACK) == _FALLBACK


def test_empty_or_none_generation_falls_back():
    assert safe_check_in(None, _FALLBACK) == _FALLBACK
    assert safe_check_in("", _FALLBACK) == _FALLBACK
    assert safe_check_in("   ", _FALLBACK) == _FALLBACK


def test_overlong_generation_falls_back():
    assert safe_check_in("word " * 100, _FALLBACK) == _FALLBACK


def test_whitespace_is_normalized_when_safe():
    assert safe_check_in("  How   are\n you? ", _FALLBACK) == "How are you?"


def test_blocklist_is_case_insensitive():
    assert safe_check_in("You Have A Fever", _FALLBACK) == _FALLBACK


def test_funnel_counts_topic_states():
    engine = CorroborationEngine(rules=list(DEFAULT_RULES))
    # Nothing observed yet.
    empty = engine.funnel()
    assert empty["topics_seen"] == 0
    assert empty["by_status"] == {"flagged": 0, "asked": 0, "confirmed": 0,
                                  "denied": 0, "unclear": 0}

    # Drive one topic through flagged -> asked -> confirmed.
    from core.events import Result, Severity
    topic = "skin_changes"
    rule = engine.rules[topic]
    hit = Result(module=rule.module, key=rule.key, value=1,
                 message="possible skin change", severity=Severity.NOTICE,
                 confidence=0.3)
    engine.observe([hit], now=100.0)
    assert engine.funnel()["by_status"]["flagged"] == 1

    engine.mark_asked(topic, now=101.0)
    assert engine.funnel()["by_status"]["asked"] == 1

    engine.hear("yeah, a little", now=102.0)
    funnel = engine.funnel()
    assert funnel["by_status"]["confirmed"] == 1
    assert funnel["topics_seen"] == 1
