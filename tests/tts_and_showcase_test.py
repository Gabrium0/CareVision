"""Tests for the polished talking-agent surface: neural TTS chain, conversational
rotation, denied-answer acknowledgment, and the narrated showcase tour.

Covers, with synthetic data and no audio hardware required:
- utterance chunking (sentence split, long-clause cap);
- Speaker engine selection: piper -> pyttsx3 -> print, synchronously;
- a live piper warm-up + non-blocking say(), skipped when no voice is bundled;
- public_line(): agent-authored caption fields only;
- greeting/small-talk rotation (first entries are the pinned originals);
- the warm ack queued when the person denies a check-in;
- start_showcase(): intro -> circuit -> single closing wrap-up.

Run standalone:  python -m pytest tests/tts_and_showcase_test.py
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.policy import Policy, _GREETINGS, _SMALL_TALK
from agent.state import ObservationMemory
from agent.voice_agent import VoiceAgent, _DEMO_CIRCUIT
from audio.tts import Speaker, _split_chunks
from core.workflows import WorkflowEngine
from storage.event_store import EventStore


def _isolate_event_store(monkeypatch, tmp_path):
    """Point the process-wide singleton at this test's own sqlite file."""
    store = EventStore(tmp_path / "events.sqlite3")
    monkeypatch.setattr(EventStore, "_instance", store)
    return store


# ------------------------------------------------------------------- chunks

def test_split_chunks_breaks_on_sentences():
    assert _split_chunks("Hello! How are you? Fine.") == \
        ["Hello!", "How are you?", "Fine."]


def test_split_chunks_caps_long_sentences_at_clauses():
    sentence = ", ".join(f"point number {i}" for i in range(40)) + "."
    chunks = _split_chunks(sentence)
    assert len(chunks) > 1
    assert all(len(chunk) <= 280 for chunk in chunks)
    joined = " ".join(chunks)
    assert joined.startswith("point number 0") and joined.endswith(".")


def test_split_chunks_handles_blank_and_whitespace():
    assert _split_chunks("   ") == []
    assert _split_chunks("") == []


# ------------------------------------------------------- engine selection

def test_disabled_speaker_is_print_only():
    speaker = Speaker(enabled=False)
    assert speaker.enabled is False
    status = speaker.status()
    assert status["engine"] == "print" and not status["ready"]


def test_selection_prefers_piper_when_available(monkeypatch):
    from audio import tts_piper
    speaker = Speaker(enabled=False)
    monkeypatch.setattr(tts_piper, "dependency_available", lambda: True)
    monkeypatch.setattr(tts_piper, "resolve_voice",
                        lambda name: Path("assets/tts/fake-medium.onnx"))
    speaker._engine_pref = "auto"
    backend = speaker._select_backend()
    assert speaker.engine_name == "piper"
    assert speaker.model == "fake-medium"
    assert backend is not None


def test_selection_falls_back_to_fake_pyttsx3(monkeypatch):
    from audio import tts_piper

    calls = {"rate": None}

    fake_engine = types.SimpleNamespace(
        setProperty=lambda key, value: calls.update({key: value}),
        say=lambda text: None, runAndWait=lambda: None)
    fake_module = types.ModuleType("pyttsx3")
    fake_module.init = lambda: fake_engine
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_module)
    monkeypatch.setattr(tts_piper, "dependency_available", lambda: False)

    speaker = Speaker(enabled=False)
    speaker._engine_pref = "auto"
    speaker._select_backend()
    assert speaker.engine_name == "pyttsx3"
    assert speaker.model == "system"
    assert calls["rate"] == speaker.rate


def _voice_bundled() -> bool:
    from audio import tts_piper
    return tts_piper.dependency_available() and bool(tts_piper.available_voices())


def test_live_piper_warms_up_and_say_is_nonblocking():
    """Integration: exercises the real bundled voice when present."""
    if not _voice_bundled():
        import pytest
        pytest.skip("no piper voice bundled in assets/tts")
    speaker = Speaker()
    try:
        deadline = time.monotonic() + 60.0
        while not speaker.ready and time.monotonic() < deadline:
            time.sleep(0.05)
        assert speaker.ready, speaker.status()
        t0 = time.monotonic()
        speaker.say("One short line.")
        assert time.monotonic() - t0 < 0.05          # queue, never synthesize
    finally:
        speaker.close()


# ------------------------------------------------------------ captions api

def test_public_line_carries_agent_fields_only(monkeypatch, tmp_path):
    _isolate_event_store(monkeypatch, tmp_path)
    agent = VoiceAgent(speak=False, moondream_enabled=False)
    try:
        agent.last_utterance = "Good afternoon, Margaret!"
        line = agent.public_line()
        assert set(line) == {"text", "speaking", "listening"}
        assert line["text"] == "Good afternoon, Margaret!"
        # No listener attached -> the agent reports no ears, never a transcript.
        assert line["listening"] is False
    finally:
        agent.close()


# ------------------------------------------------------ conversational polish

def _memory(arrived_at: float | None) -> ObservationMemory:
    mem = ObservationMemory(name="Ada")
    mem.arrived_at = arrived_at
    return mem


def test_greeting_rotates_deterministically_per_arrival():
    policy = Policy(small_talk_interval=1e9)
    first = policy._candidates(_memory(1000.0), 1000.0)[0]
    second = policy._candidates(_memory(2001.0), 2001.0)[0]
    assert first.kind == "greeting" and second.kind == "greeting"
    assert first.fallback != second.fallback            # variety across visits
    # The wall-clock bucket still reaches the LLM instruction verbatim.
    tod = ObservationMemory.time_of_day()
    assert tod in first.detail
    for key, row in _GREETINGS.items():
        assert row[0].startswith("Good ")               # original kept as [0]
        assert "{name}" in row[0]


def test_small_talk_rotation_keeps_the_pinned_first_line():
    assert _SMALL_TALK[0][1] == "How has your day been so far?"
    assert len(_SMALL_TALK) >= 4                        # real rotation available
    policy = Policy(min_gap=0.0, small_talk_interval=0.5, repeat_cooldown=0.0)
    mem = _memory(None)
    seen = set()
    now = 1000.0
    for _slot in range(len(_SMALL_TALK)):
        intent = policy.next_intent(mem, now=now)
        assert intent is not None and intent.kind == "small_talk"
        seen.add(intent.fallback)
        policy.mark_spoken(intent, now=now)             # advance the rotation
        now += 0.5
    assert len(seen) >= 3                               # fallbacks actually rotate


def test_denied_check_in_queues_no_pleasantry(monkeypatch, tmp_path):
    """A "no" gets no interjected ack: the next tailored check-in must own the
    very next spoken turn (the adaptive-question contract)."""
    _isolate_event_store(monkeypatch, tmp_path)
    agent = VoiceAgent(speak=False, moondream_enabled=False)
    try:
        agent._record_corroboration_answer("hydration", "denied")
        assert agent._action_ack is None
        extra = agent._extra_intents(now=1000.0, heard=[], handled=False)
        assert not [i for i in extra if i.fallback.startswith("Okay")]
    finally:
        agent.close()


def test_confirmed_check_in_queues_no_ack(monkeypatch, tmp_path):
    _isolate_event_store(monkeypatch, tmp_path)
    agent = VoiceAgent(speak=False, moondream_enabled=False)
    try:
        agent._record_corroboration_answer("hydration", "confirmed")
        assert agent._action_ack is None                # conclusion speaks instead
    finally:
        agent.close()


# ---------------------------------------------------------- showcase tour

def _fresh_agent(monkeypatch, tmp_path) -> VoiceAgent:
    _isolate_event_store(monkeypatch, tmp_path)
    agent = VoiceAgent(speak=False, moondream_enabled=False)
    agent.workflows = WorkflowEngine(event_store=EventStore(tmp_path / "workflows.sqlite3"))
    return agent


def test_showcase_speaks_intro_then_runs_the_circuit(monkeypatch, tmp_path):
    agent = _fresh_agent(monkeypatch, tmp_path)
    assert agent.start_showcase() is True
    assert agent._showcase_pending_close is True
    intro = agent.last_utterance.lower()
    assert "three quick checks" in intro and "balance" in intro
    session = agent.workflows.active("primary")
    assert session is not None and session.protocol == _DEMO_CIRCUIT[0]
    assert len(agent._demo_queue) == len(_DEMO_CIRCUIT) - 1


def test_showcase_closes_exactly_once_after_the_last_step(monkeypatch, tmp_path):
    agent = _fresh_agent(monkeypatch, tmp_path)
    agent.start_showcase()
    closing_seen = 0
    for _step in range(len(_DEMO_CIRCUIT)):
        agent.workflows.conclude("primary")
        agent._advance_demo_circuit()
        if "little tour" in agent.last_utterance.lower():
            closing_seen += 1
    assert agent.workflows.active("primary") is None
    assert agent._demo_queue == []
    assert closing_seen == 1
    assert agent._showcase_pending_close is False
    agent._advance_demo_circuit()                       # idempotent afterwards
    assert closing_seen == 1


def test_showcase_refused_while_a_workflow_is_active(monkeypatch, tmp_path):
    agent = _fresh_agent(monkeypatch, tmp_path)
    agent.workflows.start("sit_to_stand")
    assert agent.start_showcase() is False
    assert agent.last_utterance == ""                   # nothing spoken over it
