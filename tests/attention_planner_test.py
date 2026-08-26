"""Regression tests for deterministic attention metadata ranking."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.attention import AttentionPlanner
from agent.policy import Intent, Policy
from agent.state import ObservationMemory
from core.events import Result, Severity


def _intent(signature: str, priority: int = 50, **metadata) -> Intent:
    return Intent("observation", signature, "test", "", signature, priority,
                  health_prompt=False, **metadata)


def test_none_quality_is_neutral_and_does_not_crash():
    intent = _intent("missing-quality", quality=None)
    assert AttentionPlanner().choose([intent], now=1.0) is intent


def test_explicit_zero_quality_remains_penalized():
    zero = _intent("zero", quality=0.0)
    neutral = _intent("neutral", quality=None)
    assert AttentionPlanner().choose([zero, neutral], now=1.0) is neutral


def test_missing_metadata_uses_neutral_defaults():
    candidate = SimpleNamespace(kind="small_talk", signature="plain",
                                priority=10, health_prompt=False)
    assert AttentionPlanner().choose([candidate], now=1.0) is candidate


def test_invalid_and_non_finite_metadata_cannot_crash_ranking():
    candidate = _intent("invalid", confidence="not-a-number", quality=math.nan,
                        novelty=math.inf, severity_score=None)
    assert AttentionPlanner().choose([candidate], now=1.0) is candidate


def test_valid_candidate_ordering_is_unchanged():
    lower = _intent("lower", priority=40, confidence=0.8, quality=0.8)
    higher = _intent("higher", priority=50, confidence=0.8, quality=0.8)
    assert AttentionPlanner().choose([lower, higher], now=1.0) is higher


def test_policy_accepts_result_with_optional_quality():
    memory = ObservationMemory()
    memory.ingest([Result("clothing_advice", "recommendation", "Bring a layer.",
                          confidence=0.6, quality=None, severity=Severity.NOTICE,
                          message="Bring a layer.")], now=100.0)
    policy = Policy(min_gap=0.0)
    intent = policy.next_intent(memory, now=100.0)
    assert intent is not None
    assert intent.signature == "clothing:Bring a layer."
    assert intent.quality is None
