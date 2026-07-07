"""Conversational voice agent.

A modular, stateful layer on top of the detection pipeline: it accumulates what
it learns about the person over time and speaks proactively — greeting on
arrival, making light small talk, and raising salient observations (e.g. that
their clothing is too light for the weather) as the modules surface them. Uses
Gemini for natural phrasing (key from .env) with an offline templated fallback,
and speaks via offline TTS.
"""
