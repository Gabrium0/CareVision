"""The agent adapts and asks questions based on the data it observes.

Three layers, from fastest to most faithful:

1. Deterministic (no network, always runs): detector data selects a matching
   tailored follow-up question; data without a question trigger stays silent;
   context selection adapts to the question and the available data; and a
   person's answer changes what happens next (one gentle re-ask, a cooldown
   after denial, or a conclusion after confirmation).
2. Stub-backed Moondream-on pipeline (no network, always runs): proves the
   full enabled path — intent -> structured context -> submit_response ->
   poll_response -> speech guard -> spoken line.
3. Live Moondream integration (skips without a key): runs the real model and
   shows ONE agent asking a different natural-language question as the data
   stream changes, then a confirmation closing the loop. The test gives the
   provider extra time (raises `agent._response_deadline` and the client HTTP
   timeout) so the model's own phrasing is spoken rather than the 4 s templated
   fallback. Run with
   `python -m pytest tests/adaptive_question_test.py -m integration -s`
   after setting `X-Moondream-Auth` (or `MOONDREAM_API_KEY`) in .env.
"""
from __future__ import annotations

import time

import pytest

from agent.corroboration import CorroborationEngine
from agent.conversation import (
    AgentContextBroker, AgentResponse, guard_agent_only_speech,
)
from core.events import Result, Severity, Visibility


def _res(module, key, value, confidence, severity=Severity.NOTICE,
         message="x", timestamp=100.0, visibility=Visibility.PUBLIC):
    return Result(module=module, key=key, value=value, confidence=confidence,
                  severity=severity, message=message, ttl=10.0,
                  timestamp=timestamp, visibility=visibility)


class _FakeListener:
    """Stand-in for the microphone; the agent drains it each tick."""

    available = True

    def __init__(self):
        self.queue = []

    def pop_utterances(self):
        out, self.queue = self.queue, []
        return out

    def close(self):
        pass


def _agent(**policy_kwargs):
    """A speak-free VoiceAgent with the Moondream transport enabled."""
    from agent.voice_agent import VoiceAgent
    return VoiceAgent(name="Ada", speak=False, listener=_FakeListener(),
                      moondream_enabled=True, **policy_kwargs)


def _drain_until_spoken(agent, snapshot, now, label="",
                        deadline_s=30.0, sleep_s=0.1):
    """Tick until the agent speaks, handling the async submit -> poll flow."""
    started = time.monotonic()
    attempts = 0
    while time.monotonic() - started < deadline_s:
        text = agent.tick(snapshot, now=now)
        if text:
            return text
        attempts += 1
        if attempts % 20 == 0 and attempts > 0:
            print(f"      ... {label}: still waiting for the agent "
                  f"({attempts * sleep_s:.1f}s)")
        time.sleep(sleep_s)
        now += sleep_s
    return None


# --------------------------------------------------------------------------
# Layer 1: deterministic adaptation (no network, always runs)
# --------------------------------------------------------------------------

def test_detector_data_selects_a_matching_tailored_question():
    """Different detector data yields a different, matching question."""
    cases = [
        ("rash", "rash", "skin_changes", "skin"),
        ("drowsiness", "perclos", "tiredness", "tired"),
        ("dry_lips", "lip_dryness", "hydration", "drink"),
        ("sweating", "sweat_gloss", "feeling_warm", "warm"),
        ("pain", "pain", "discomfort", "aches"),
    ]
    for module, key, expected_topic, keyword in cases:
        engine = CorroborationEngine()
        engine.observe([_res(module, key, 1, 0.4)], now=100.0)
        got = engine.next_question(100.0)
        assert got is not None, f"{module}/{key} produced no question"
        topic, rule = got
        assert topic == expected_topic, f"{module}/{key} flagged {topic!r}"
        assert keyword in rule.question.lower(), (
            f"{module}/{key} question does not match its data: {rule.question!r}")


def test_data_without_a_question_trigger_is_not_asked_about():
    """High-confidence, INFO-only, or absent data never prompts a question."""
    # A strong cue is too certain to need a gentle check-in (alert path owns it).
    engine = CorroborationEngine()
    engine.observe([_res("rash", "rash", 1, 0.9)], now=100.0)
    assert engine.next_question(100.0) is None
    # INFO severity is routine, not worth asking about.
    engine = CorroborationEngine()
    engine.observe([_res("rash", "rash", 1, 0.4, Severity.INFO)], now=100.0)
    assert engine.next_question(100.0) is None
    # No data at all -> nothing to ask about.
    assert CorroborationEngine().next_question(100.0) is None


def test_context_selection_adapts_to_the_question_and_the_data():
    """The broker ranks data relevant to the current question, and adapts
    when the available data changes."""
    broker = AgentContextBroker()
    broker.ingest([
        _res("heart_rate", "bpm", 74, 0.9, message="Heart rate 74 bpm"),
        _res("rash", "rash", 1, 0.4, message="Possible rash signal"),
    ], now=100.0)
    context = broker.build("What is my heart rate?", [])
    heart = next(item for item in context.items if item.module == "heart_rate")
    rash = next(item for item in context.items if item.module == "rash")
    assert context.items.index(heart) < context.items.index(rash)

    broker2 = AgentContextBroker()
    broker2.ingest([_res("sweating", "sweat_gloss", 1, 0.4,
                         message="Sweat gloss detected")], now=100.0)
    context2 = broker2.build("Are you feeling warm?", [])
    assert context2.items and context2.items[0].module == "sweating"

    # A private hypothesis stays queryable but is never spoken as a fact.
    broker3 = AgentContextBroker()
    broker3.ingest([_res("skin_vision", "hypothesis", "eczema", 0.5,
                         message="Possible eczema hypothesis",
                         visibility=Visibility.AGENT_ONLY)], now=101.0)
    context3 = broker3.build("skin", [])
    assert any(item.module == "skin_vision" for item in context3.items)
    text, reason = guard_agent_only_speech(
        "It looks like you have eczema.", broker3.items(),
        "I have an uncertain observation; would you like to check it together?")
    assert "eczema" not in text.lower()
    assert reason == "agent_only_assertion_blocked"


def test_the_answer_changes_future_questions():
    """Unclear buys one gentle re-ask; a denial silences the topic; a
    confirmation produces a conclusion — and in every case the same data
    no longer re-asks afterwards."""
    snap = [_res("rash", "rash", 1, 0.4)]

    # unclear -> exactly one gentle re-ask, then terminal.
    engine = CorroborationEngine()
    engine.observe(snap, now=100.0)
    engine.mark_asked("skin_changes", 101.0)
    assert engine.hear("what a lovely day", 105.0) == ("skin_changes",
                                                       "unclear")
    assert engine.status("skin_changes") == "flagged"   # may be re-asked once
    engine.mark_asked("skin_changes", 110.0)
    assert engine.hear("hmm, the birds are singing", 112.0) == (
        "skin_changes", "unclear")
    assert engine.status("skin_changes") == "unclear"   # terminal
    assert engine.next_question(112.0) is None

    # denied -> cooldown: the same data no longer re-asks.
    engine = CorroborationEngine()
    engine.observe(snap, now=100.0)
    engine.mark_asked("skin_changes", 101.0)
    assert engine.hear("no, nothing like that", 105.0) == ("skin_changes",
                                                           "denied")
    assert engine.pending_conclusions() == []
    engine.observe(snap, now=200.0)
    assert engine.next_question(200.0) is None

    # confirmed -> a conclusion, and the topic then goes quiet.
    engine = CorroborationEngine()
    engine.observe(snap, now=100.0)
    engine.mark_asked("skin_changes", 101.0)
    assert engine.hear("yes, a bit itchy actually", 105.0) == ("skin_changes",
                                                               "confirmed")
    assert [t for t, _ in engine.pending_conclusions()] == ["skin_changes"]
    engine.mark_concluded("skin_changes")
    engine.observe(snap, now=200.0)
    assert engine.next_question(200.0) is None


# --------------------------------------------------------------------------
# Layer 2: Moondream-on pipeline, offline (stub provider, always runs)
# --------------------------------------------------------------------------

class _CannedProvider:
    """In-process stand-in for the Moondream client with a fixed reply."""

    def __init__(self, reply):
        self.reply = reply
        self.request = None
        self._polls = 0

    def status(self):
        return {"active": True}

    def submit_response(self, messages, context_items, image=None):
        self.request = (list(messages), list(context_items), image)
        return "stub-request"

    def poll_response(self, _request_id):
        self._polls += 1
        if self._polls < 2:
            return False, None
        return True, AgentResponse(self.reply, provider_status="ready")

    def close(self):
        pass


def test_moondream_enabled_pipeline_uses_structured_response_and_guard():
    """With the transport enabled, the agent phrases its data-driven question
    through the structured submit -> poll path and lands on the spoken line."""
    agent = _agent(min_gap=0, small_talk_interval=1e9)
    agent.moondream.close()
    provider = _CannedProvider("I noticed some skin irritation lately — "
                               "have you been feeling itchy?")
    agent.moondream = provider
    agent.elicitation.clear()
    try:
        snapshot = [_res("rash", "rash", 1, 0.4, timestamp=1000.0)]
        text = _drain_until_spoken(agent, snapshot, 1000.0,
                                   label="stub pipeline")
        assert text and "itchy" in text
        assert agent.corroboration.status("skin_changes") == "asked"
        messages, items, image = provider.request
        assert image is None
        assert any(item.module == "rash" for item in items)
        assert any("Conversation controller instruction" in m["content"]
                   for m in messages)
    finally:
        agent.elicitation.clear()
        agent.close()


# --------------------------------------------------------------------------
# Layer 3: live Moondream, real model (integration; skips without a key)
# --------------------------------------------------------------------------

@pytest.mark.integration
def test_live_moondream_agent_adapts_questions_to_data():
    """One real agent, one changing data stream: the model phrases a
    different tailored question for each cue, never leaks a private
    hypothesis, and turns a confirmation into a conclusion."""
    import agent.moondream_client as moondream_module

    if not moondream_module.moondream_api_key():
        pytest.skip("set X-Moondream-Auth (or MOONDREAM_API_KEY) in .env to "
                    "run the live Moondream presentation demo")

    agent = _agent(min_gap=0, small_talk_interval=1e9)
    # Presentation-friendly: back-to-back questions, no 60 s health gap.
    agent.policy.attention.health_prompt_gap = 0.0
    agent.policy.attention.health_prompts_per_hour = 12
    # Give the real model room to respond (the client's HTTP timeout and the
    # agent's response deadline both need to be comfortably above the 4 s
    # production default so the LLM phrasing is spoken, not the fallback).
    agent.moondream.timeout = 15.0
    agent._response_deadline = 12.0
    agent.elicitation.clear()
    try:
        base = time.time()
        steps = [
            ("rash", "rash", "skin_changes"),
            ("drowsiness", "perclos", "tiredness"),
            ("sweating", "sweat_gloss", "feeling_warm"),
        ]
        print("\n== Adaptive questions demo (real Moondream) ==")
        spoken = []
        for index, (module, key, topic) in enumerate(steps):
            at = base + index * 2
            snapshot = [_res(module, key, 1, 0.4,
                             message=f"{module} cue", timestamp=at)]
            text = _drain_until_spoken(agent, snapshot, at, label=topic)
            assert text, f"step {index + 1}: the agent did not speak"
            spoken.append(text)
            assert agent.corroboration.status(topic) == "asked"
            print(f"  [data] {module}/{key} -> agent asked: {text}")
            if index < len(steps) - 1:
                # The agent now waits for the answer to the question it just
                # asked (agent/voice_agent.py::_awaiting_answer), so this demo
                # must take its turn before the next cue arrives. A denial
                # closes the topic without producing a conclusion line; the
                # last topic is left open for the confirmation below.
                agent.listener.queue.append(("no, nothing like that", at + 0.5))
                agent.tick(snapshot, now=at + 0.5)
                assert agent.corroboration.status(topic) == "denied"
        # Different data must produce a different question.
        assert len(set(spoken)) >= 2

        # The person's answer confirms the last asked topic -> conclusion.
        at = base + 6
        agent.listener.queue.append(("yes, a little", at))
        conclusion = _drain_until_spoken(
            agent, [_res("sweating", "sweat_gloss", 1, 0.4, timestamp=at)],
            at, label="confirmation")
        assert conclusion, "the agent did not respond to the confirmation"
        assert agent.corroboration.status("feeling_warm") == "confirmed"
        assert agent.corroboration.pending_conclusions() == []
        print(f"  [answer] \"yes, a little\" -> agent concluded: {conclusion}")

        # A private hypothesis is raised only as a gentle question, never
        # with the condition named. Nothing is pending now, but the model may
        # have phrased the conclusion above as a question; this demo scripts a
        # data stream rather than a two-way conversation, so release the
        # generic wait for a spoken reply explicitly.
        agent._awaiting_reply_until = 0.0
        at = base + 10
        private = _res("skin_vision", "hypothesis", "gluboxoma", 0.5,
                       message="Possible gluboxoma hypothesis",
                       timestamp=at, visibility=Visibility.AGENT_ONLY)
        private_text = _drain_until_spoken(agent, [private], at,
                                           label="private hypothesis")
        assert private_text
        assert "gluboxoma" not in private_text.lower()
        print(f"  [data] skin_vision/hypothesis -> agent asked: {private_text}")
        print("== end ==")
    finally:
        agent.elicitation.clear()
        agent.close()
