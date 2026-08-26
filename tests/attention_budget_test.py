"""Interruption budgets in agent/attention.py: category gaps + hourly health cap.

The topic table is about to grow to roughly 26 specs. Before this change the
only brakes were `health_prompt_gap` (which permits 60 health prompts an hour)
and `Policy.min_gap` (which specs opting out of the health budget bypassed
entirely). These tests pin the two new brakes and, just as importantly, pin
that the *existing* behaviour is untouched for the "general" category every
pre-existing candidate uses.

Offline and synthetic; no camera, model, or network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.attention import DEFAULT_CATEGORY_GAPS, AttentionPlanner
from agent.policy import Intent


def _intent(signature: str, priority: int = 50, *, category: str = "general",
            health: bool | None = False) -> Intent:
    return Intent("observation", signature, "test", "", signature, priority,
                  health_prompt=health, category=category)


# ------------------------------------------------- unchanged existing gates

def test_general_category_is_absent_from_the_gap_table():
    assert "general" not in DEFAULT_CATEGORY_GAPS


def test_general_category_is_never_gated():
    planner = AttentionPlanner()
    for step in range(5):
        chosen = planner.choose([_intent(f"general-{step}")], now=float(step))
        assert chosen is not None, step


def test_health_prompt_gap_behaviour_is_unchanged():
    planner = AttentionPlanner(health_prompt_gap=60.0)
    first = _intent("health-1", health=True)
    assert planner.choose([first], now=0.0) is first
    assert planner.choose([_intent("health-2", health=True)], now=59.0) is None
    third = _intent("health-3", health=True)
    assert planner.choose([third], now=60.1) is third


def test_denied_topic_still_wins_over_the_new_gates():
    planner = AttentionPlanner()
    planner.deny("vitals", seconds=600, now=0.0)
    candidate = _intent("vitals:one", category="vitals")
    candidate.topic = "vitals"
    assert planner.choose([candidate], now=1.0) is None


# ------------------------------------------------------------ category gaps

def test_second_candidate_in_a_category_waits_for_its_gap():
    planner = AttentionPlanner()
    first = _intent("vitals:one", category="vitals")
    assert planner.choose([first], now=0.0) is first
    assert planner.choose([_intent("vitals:two", category="vitals")],
                          now=299.0) is None
    later = _intent("vitals:three", category="vitals")
    assert planner.choose([later], now=301.0) is later


def test_a_different_category_is_admitted_inside_another_categorys_gap():
    planner = AttentionPlanner()
    assert planner.choose([_intent("vitals:one", category="vitals")], now=0.0)
    social = _intent("social:one", category="social")
    assert planner.choose([social], now=100.0) is social


def test_category_gap_skips_the_blocked_candidate_not_the_whole_turn():
    planner = AttentionPlanner()
    assert planner.choose([_intent("vitals:one", priority=90,
                                   category="vitals")], now=0.0)
    blocked = _intent("vitals:two", priority=90, category="vitals")
    runner_up = _intent("skin:one", priority=10, category="skin")
    assert planner.choose([blocked, runner_up], now=10.0) is runner_up


def test_custom_category_gaps_replace_the_defaults():
    planner = AttentionPlanner(category_gaps={"vitals": 5.0})
    assert planner.choose([_intent("vitals:one", category="vitals")], now=0.0)
    assert planner.choose([_intent("vitals:two", category="vitals")], now=4.0) is None
    assert planner.choose([_intent("vitals:three", category="vitals")], now=6.0)
    # "skin" is no longer in the table, so it is no longer gated at all.
    assert planner.choose([_intent("skin:one", category="skin")], now=6.1)
    assert planner.choose([_intent("skin:two", category="skin")], now=6.2)


# ------------------------------------------------------- hourly health cap

def _fill_health_budget(planner: AttentionPlanner, count: int, start: float = 0.0):
    for step in range(count):
        now = start + step
        chosen = planner.choose([_intent(f"health-{now}", health=True)], now=now)
        assert chosen is not None, now


def test_seventh_health_prompt_in_an_hour_is_blocked_then_released():
    # health_prompt_gap=0 isolates the rolling hourly cap from the min gap.
    planner = AttentionPlanner(health_prompt_gap=0.0, health_prompts_per_hour=6)
    _fill_health_budget(planner, 6, start=0.0)
    assert planner.choose([_intent("health-7", health=True)], now=6.0) is None
    # Still blocked just before the oldest entry leaves the rolling hour.
    assert planner.choose([_intent("health-7", health=True)], now=3600.0) is None
    released = _intent("health-7", health=True)
    assert planner.choose([released], now=3600.5) is released


def test_hourly_cap_does_not_touch_non_health_candidates():
    planner = AttentionPlanner(health_prompt_gap=0.0, health_prompts_per_hour=2)
    _fill_health_budget(planner, 2, start=0.0)
    assert planner.choose([_intent("health-3", health=True)], now=2.0) is None
    chatty = _intent("small-talk", health=False)
    assert planner.choose([chatty], now=2.1) is chatty


def test_health_kinds_are_still_inferred_when_not_declared():
    planner = AttentionPlanner(health_prompt_gap=0.0, health_prompts_per_hour=1)
    inferred = Intent("follow_up", "ask:x", "test", "", "test", 50)
    assert planner.choose([inferred], now=0.0) is inferred
    assert planner.choose([Intent("question", "ask:y", "t", "", "t", 50)],
                          now=1.0) is None
