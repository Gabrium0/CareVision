"""Guardrails for what the agent is allowed to say aloud.

`VoiceAgent._clean_spoken_line` is the deterministic "disposes" half of the
LLM-proposes/deterministic-disposes pattern: it rejects a generation that echoes
the controller/context, lists stage directions, names raw events, or speaks raw
sensor readings — falling back to a hand-authored caring line instead.
"""
from __future__ import annotations

from agent.voice_agent import VoiceAgent as V


def _rejected(s: str) -> bool:
    return V._clean_spoken_line(s) == ""


def test_meta_and_event_and_directive_lines_are_rejected():
    for bad in [
        "The conversation controller instruction (not spoken verbatim): be warm.",
        "The audio controller asked about your day.",
        "By the way, Person arrived would you like to talk about that?",
        "By the way, face touching would you like to talk about that?",
        "Ask about local weather. Describe a recent event. Offer advice.",
        "Make small talk and ask how their day is going.",
    ]:
        assert _rejected(bad), bad


def test_raw_sensor_readings_are_rejected():
    for bad in [
        "Your breathing is about 9.6 breaths per minute.",
        "I noticed a jaundice tint of 0.0 today.",
        "Your heart rate is 62 right now.",
        "Your oxygen saturation is 95%.",
    ]:
        assert _rejected(bad), bad


def test_natural_lines_including_ordinary_numbers_pass():
    for good in [
        "Good morning, how did you sleep?",
        "Your heart rate looks steady today — how are you feeling?",
        "Your breathing seems calm and easy.",
        "See you at 3 o'clock, take care.",
        "Take 3 deep breaths with me.",
        "You are such a lovely person to chat with.",
    ]:
        assert V._clean_spoken_line(good) == " ".join(good.split()), good
