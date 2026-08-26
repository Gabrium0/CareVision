"""Tests for the single shared answer classifier (agent/answers.py).

Three divergent keyword classifiers were unified here; these tests pin both
the widened vocabulary and every phrase the pre-existing suites already
depend on, so the merge can only ever have added coverage.

Run standalone:  python -m pytest tests/answer_classifier_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.answers import VERDICTS, interpret_answer
from agent.corroboration import interpret_answer_keywords


# ------------------------------------------------------------- new vocabulary

@pytest.mark.parametrize("phrase", [
    "hmm, I'm not really sure", "not sure", "I'm not too sure", "unsure",
    "not certain", "hard to say", "I can't tell", "no idea",
    "I don't know", "dont know", "who knows really",
])
def test_uncertainty_is_unclear_not_denied(phrase):
    # "not really sure" / "don't know" must buy a gentle re-ask, not a denial
    # cooldown; the uncertainty vocabulary is checked before the denial tokens
    # ("not really", "don't") it would otherwise trip.
    assert interpret_answer(phrase) == "unclear"


@pytest.mark.parametrize("phrase", [
    "not now", "maybe later", "rather not", "i'd rather not", "id rather not",
    "no thanks", "no thank you", "not today", "leave it", "another time",
    "cancel", "stop", "dont", "none",
])
def test_new_denials(phrase):
    assert interpret_answer(phrase) == "denied"


@pytest.mark.parametrize("phrase", [
    "okay", "ok", "sure", "go ahead", "why not", "please do", "sounds good",
    "alright", "of course", "definitely", "that would be nice",
    "i would like",
])
def test_new_confirmations(phrase):
    assert interpret_answer(phrase) == "affirmed"


@pytest.mark.parametrize("phrase", [
    "Maybe later, thanks.", "I'd rather not right now.",
    "No thank you, another time.", "Not today please.",
])
def test_new_denials_in_a_sentence(phrase):
    assert interpret_answer(phrase) == "denied"


@pytest.mark.parametrize("phrase", [
    "Sure, go ahead.", "Alright, that would be nice.", "Of course, why not.",
])
def test_new_confirmations_in_a_sentence(phrase):
    assert interpret_answer(phrase) == "affirmed"


def test_verdict_vocabulary():
    assert VERDICTS == ("affirmed", "denied", "unclear")
    for phrase in ("no", "yes", "the weather is nice"):
        assert interpret_answer(phrase) in VERDICTS


# ------------------------------------------- the "please" short/imperative rule

@pytest.mark.parametrize("phrase", [
    "please", "yes please", "please do", "please do that", "Please!",
])
def test_short_please_affirms(phrase):
    assert interpret_answer(phrase) == "affirmed"


@pytest.mark.parametrize("phrase", [
    "please check my arm instead",
    "Please check my tremor",
    "please take a look at the camera",
    "could you please have a look at this spot",
])
def test_imperative_please_is_not_an_affirmation(phrase):
    assert interpret_answer(phrase) == "unclear"


# ------------------------------------ phrases pinned by pre-existing test suites

@pytest.mark.parametrize("phrase,expected", [
    # tests/corroboration_and_elicitation_test.py::test_keyword_interpretation
    ("No, not really.", "denied"),
    ("Yes, a little actually", "affirmed"),
    ("The weather is nice", "unclear"),
    ("I'm fine, thanks", "denied"),
    # other utterances driven through the agent in that suite
    ("yes, I noticed a spot lately", "affirmed"),
    ("no, nothing like that", "denied"),
    ("yes, a bit itchy actually", "affirmed"),
    ("what a lovely day", "unclear"),
    ("hmm the birds are singing", "unclear"),
    ("hello robot", "unclear"),
    # tests/conversation_agent_test.py:128,151
    ("yes, please", "affirmed"),
    ("No, cancel that", "denied"),
    ("Please check my tremor", "unclear"),
    ("Check my hand tremor", "unclear"),
    ("What is my heart rate?", "unclear"),
])
def test_existing_phrases_keep_their_verdict(phrase, expected):
    assert interpret_answer(phrase) == expected


# "no" must stay whole-word bounded: the original bug this classifier fixes.
def test_no_does_not_fire_inside_another_word():
    assert interpret_answer("I noticed nothing unusual") == "denied"  # "nothing"
    assert interpret_answer("I have noticed a change") == "affirmed"  # "i have"
    assert interpret_answer("that is a novelty") == "unclear"


def test_denials_are_checked_before_confirmations():
    assert interpret_answer("no, not really, I guess so") == "denied"


# --------------------------------------------------- scripted replay scenarios

def _replay_answers():
    path = Path(__file__).resolve().parent.parent / "config" / "replay_scenarios.json"
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    seen = []
    for scenario in scenarios.values():
        for event in scenario.get("events", []):
            if event.get("type") == "answer":
                seen.append(event["text"])
    return seen


@pytest.mark.parametrize("text,expected", [
    ("No discomfort", "denied"),
    ("No recent change", "denied"),
    ("No warning signs", "denied"),
    ("Yes I feel unwell", "affirmed"),
    # Bare "fine" is deliberately NOT a denial -- only "i'm fine"/"i am fine".
    ("Feeling fine, thanks", "unclear"),
])
def test_scripted_replay_answers(text, expected):
    assert text in _replay_answers(), f"{text!r} no longer in replay_scenarios.json"
    assert interpret_answer(text) == expected


# --------------------------------------------------------- corroboration alias

@pytest.mark.parametrize("phrase,expected", [
    ("Yes, a little actually", "confirmed"),
    ("okay", "confirmed"),
    ("please", "confirmed"),
    ("No, not really.", "denied"),
    ("maybe later", "denied"),
    ("The weather is nice", "unclear"),
    ("please check my arm instead", "unclear"),
])
def test_alias_returns_the_confirmed_vocabulary(phrase, expected):
    assert interpret_answer_keywords(phrase) == expected


def test_alias_never_leaks_the_affirmed_label():
    for phrase in ("yes", "sure", "definitely", "no", "stop", "the sky"):
        assert interpret_answer_keywords(phrase) != "affirmed"
