"""Turn-taking: the agent waits for the answer to a question it just asked.

Before this gate the agent talked AT the person — it asked a question and kept
speaking, repeating the same small-talk line three times in one replay. Now,
whenever a question the agent asked is still inside its answer window, only a
reply or a confirmed conclusion may interject.

The load-bearing safety property is the mute guard: an installation with no
microphone (`--no-voice`, `listener=None`) can never receive an answer, so it
must never wait for one. Everything runs offline: Moondream is disabled and a
fake listener stands in for the microphone, following
tests/conversation_agent_test.py.

Run standalone:  python -m pytest tests/turn_taking_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.policy import Intent, Policy
from agent.state import ObservationMemory
from core.events import Result, Severity

SMALL_TALK = "How has your day been so far?"


def _result(module, key, value, confidence, severity=Severity.NOTICE,
            message="x", ttl=10.0):
    return Result(module=module, key=key, value=value, confidence=confidence,
                  severity=severity, message=message, ttl=ttl)


class _Listener:
    """Stand-in for the microphone; the agent drains it each tick."""

    available = True

    def __init__(self):
        self.queue = []

    def pop_utterances(self):
        out, self.queue = self.queue, []
        return out

    def close(self):
        pass


class _DeafListener(_Listener):
    """A listener whose ASR backend failed at runtime (audio/stt.py flips this)."""

    available = False


def _agent(listener=_Listener, *, small_talk_interval=45.0):
    """A speak-free, offline VoiceAgent with the anti-chatter gap removed.

    `min_gap` is zeroed so every silence in these tests is attributable to the
    turn-taking gate rather than to the pre-existing cadence timer.
    """
    from agent.voice_agent import VoiceAgent
    agent = VoiceAgent(name="Ada", speak=False, moondream_enabled=False,
                       listener=None if listener is None else listener(),
                       min_gap=0, small_talk_interval=small_talk_interval)
    return agent


def _rash():
    return [_result("rash", "rash", 0.3, 0.4)]


# ------------------------------------------------------------------- the gate

def test_corroboration_ask_waits_for_the_answer_then_speaks_again():
    """A voiced check-in holds the floor until its answer window lapses."""
    agent = _agent()
    try:
        asked = agent.tick(_rash(), now=1000.0)
        assert asked is not None and "skin" in asked.lower()
        assert agent.corroboration.status("skin_changes") == "asked"
        # Silence from the person: the agent does not fill it.
        for now in (1002.0, 1010.0, 1029.0):
            assert agent.tick(_rash(), now=now) is None, f"spoke again at {now}"
            assert agent._awaiting_answer(now) is True
        # The 30 s window has lapsed; the agent may take the floor again.
        assert agent._awaiting_answer(1050.0) is False
        assert agent.tick(_rash(), now=1050.0) is not None
    finally:
        agent.close()


def test_status_asked_alone_never_mutes_the_agent():
    """The gate reads the answer window, not the never-expiring topic status."""
    agent = _agent()
    try:
        assert agent.tick(_rash(), now=1000.0) is not None
        # Nothing in the engine ever moves an unanswered topic off "asked".
        assert agent.corroboration.status("skin_changes") == "asked"
        assert agent._awaiting_answer(1000.0 + 3600.0) is False
    finally:
        agent.close()


def test_only_urgent_alerts_interject_while_awaiting_an_answer():
    """A thinking pause admits confirmed emergencies and nothing conversational."""
    policy = Policy(min_gap=0)
    memory = ObservationMemory(name="Ada")
    question = Intent("question", "confirm-action:arm_check:None:1000", "", "",
                      "Would you like me to start the arm check now?", 120)
    follow_up = Intent("follow_up", "ask:skin_changes:1", "", "",
                       "Have you noticed any skin changes lately?", 45)
    reply = Intent("reply", "reply:1", "", "", "I'm glad you told me that.", 80)
    conclusion = Intent("conclusion", "conclude:skin_changes", "", "",
                        "Please keep an eye on that spot.", 68)
    urgent = Intent("urgent_alert", "urgent:fall:1", "", "",
                    "I detected a fall. Please call for help now.", 1000)

    chosen = policy.next_intent(memory, now=1000.0, awaiting_answer=True,
                                extra=[question, follow_up, reply, conclusion])
    assert chosen is None
    chosen = policy.next_intent(memory, now=1001.0, awaiting_answer=True,
                                extra=[question, follow_up, urgent])
    assert chosen is not None and chosen.kind == "urgent_alert"
    # The flag defaults to False, so every pre-existing caller is unaffected.
    chosen = policy.next_intent(memory, now=1003.0, suppress_routine=True,
                                extra=[question, follow_up])
    assert chosen is not None and chosen.kind == "question"


def test_conclusion_still_reaches_the_person_while_the_agent_waits():
    """Ordinary conclusions wait until the person has had time to answer."""
    agent = _agent()
    try:
        assert agent.tick([], now=1000.0) == SMALL_TALK
        assert agent._awaiting_answer(1005.0) is True
        assert agent.tick([], now=1005.0) is None
        finding = _result("tremor", "tremor_test", "steady", 0.8,
                          severity=Severity.INFO,
                          message="Your hand looked nice and steady.", ttl=30.0)
        assert agent.tick([finding], now=1006.0) is None
    finally:
        agent.close()


# ------------------------------------------------------- the mute regression

def test_agent_without_a_listener_is_never_gated():
    """MUTE GUARD: with no ears an answer can never arrive, so never wait.

    Ticked side by side against an identical agent that can hear, so the test
    fails if the deaf agent ever starts behaving like the gated one.
    """
    deaf = _agent(listener=None, small_talk_interval=10.0)
    hearing = _agent(small_talk_interval=10.0)
    try:
        assert deaf.tick(_rash(), now=1000.0) is not None      # both ask
        assert hearing.tick(_rash(), now=1000.0) is not None
        assert deaf._awaiting_answer(1010.0) is False
        assert hearing._awaiting_answer(1010.0) is True
        # Inside the answer window the deaf agent keeps talking, exactly as
        # before this change; the hearing one holds the floor for the person.
        assert deaf.tick(_rash(), now=1010.0) == SMALL_TALK
        assert hearing.tick(_rash(), now=1010.0) is None
        assert deaf.tick(_rash(), now=1021.0) is not None
        assert hearing.tick(_rash(), now=1021.0) is None
        # A question the deaf agent asks never arms the generic wait either.
        assert deaf._awaiting_reply_until == 0.0
        # ... and once the window lapses the hearing agent recovers its voice.
        assert hearing.tick(_rash(), now=1040.0) is not None
    finally:
        deaf.close()
        hearing.close()


def test_listener_that_cannot_hear_behaves_like_no_listener():
    """`available = False` (a failed ASR backend) disables waiting too."""
    agent = _agent(listener=_DeafListener, small_talk_interval=10.0)
    try:
        assert agent.tick(_rash(), now=1000.0) is not None
        assert agent._can_hear() is False
        assert agent._awaiting_answer(1010.0) is False
        assert agent.tick(_rash(), now=1010.0) == SMALL_TALK
    finally:
        agent.close()


# ----------------------------------------- the untracked conversational ask

def test_small_talk_does_not_repeat_while_its_own_question_is_outstanding():
    """The observed bug: "How has your day been so far?" three times over."""
    agent = _agent(small_talk_interval=3.0)
    try:
        assert agent.tick([], now=1000.0) == SMALL_TALK
        assert agent._awaiting_reply_until == 1030.0
        for now in (1004.0, 1012.0, 1029.9):
            assert agent.tick([], now=now) is None, f"repeated small talk at {now}"
        assert agent.tick([], now=1031.0) is not None
    finally:
        agent.close()


def test_a_question_arms_the_generic_wait_and_a_statement_does_not():
    """The final spoken text decides — a fallback can turn a question into prose."""
    agent = _agent()
    try:
        asking = Intent("small_talk", "smalltalk:1", "", "", SMALL_TALK, 10)
        agent._speak_intent(asking, SMALL_TALK, 1000.0, action=None)
        assert agent._awaiting_reply_until == 1030.0

        agent._awaiting_reply_until = 0.0
        telling = Intent("reply", "reply:2", "", "",
                         "I'm glad you told me that.", 80)
        agent._speak_intent(telling, "I'm glad you told me that.", 1010.0,
                            action=None)
        assert agent._awaiting_reply_until == 0.0
    finally:
        agent.close()


def test_reply_window_is_configurable_from_the_conversation_section():
    """`conversation.reply_window` reaches the agent on the existing config path."""
    from agent.voice_agent import VoiceAgent
    agent = VoiceAgent(name="Ada", speak=False, moondream_enabled=False,
                       listener=_Listener(), min_gap=0,
                       conversation={"reply_window": 5.0})
    try:
        assert agent.reply_window == 5.0
        assert agent.tick([], now=1000.0) == SMALL_TALK
        assert agent._awaiting_reply_until == 1005.0
        assert agent._awaiting_answer(1004.0) is True
        assert agent._awaiting_answer(1006.0) is False
    finally:
        agent.close()


def test_any_heard_utterance_even_off_topic_clears_the_wait():
    """A substantive off-topic remark still counts as taking a turn."""
    agent = _agent(small_talk_interval=3.0)
    try:
        assert agent.tick([], now=1000.0) == SMALL_TALK
        assert agent.tick([], now=1004.0) is None
        agent.listener.queue.append(("the weather is nice today", 1005.0))
        said = agent.tick([], now=1006.0)
        assert agent._awaiting_reply_until == 0.0
        assert said is not None            # a warm reply, not more silence
    finally:
        agent.close()


def test_a_short_substantive_remark_is_a_turn_not_a_thinking_sound():
    """Brevity alone never disqualifies an utterance from counting as a turn.

    A blanket "three words or fewer is filler" rule silently discarded
    "hello robot" and, worse, "my back hurts" — the person kept the floor they
    had already given up and the agent answered nothing.
    """
    for remark in ("hello robot", "my back hurts", "I feel dizzy"):
        agent = _agent()
        try:
            assert agent.tick([], now=1000.0) == SMALL_TALK
            agent.listener.queue.append((remark, 1001.0))
            said = agent.tick([], now=1002.0)
            assert agent._awaiting_reply_until == 0.0, remark
            assert said is not None, remark
        finally:
            agent.close()


def test_hesitation_preserves_the_answer_window_without_triggering_a_reask():
    """Short thinking sounds are not consumed as unclear health answers."""
    agent = _agent()
    try:
        assert agent.tick(_rash(), now=1000.0) is not None
        agent.listener.queue.append(("okay then", 1004.0))
        assert agent.tick(_rash(), now=1005.0) is None
        assert agent.corroboration.status("skin_changes") == "asked"
        assert agent._pending_classification is None
        assert agent._awaiting_answer(1005.0) is True
    finally:
        agent.close()


def test_confirmed_fall_interrupts_once_but_other_alerts_do_not():
    """Only the narrow confirmed-fall path may interrupt a thinking pause."""
    agent = _agent()
    fall = _result("fall", "fall", True, 0.9, severity=Severity.ALERT,
                   message="FALL DETECTED")
    other_alert = _result("other", "other", True, 0.9, severity=Severity.ALERT,
                          message="Other alert")
    try:
        assert agent.tick([], now=1000.0) == SMALL_TALK
        assert agent.tick([other_alert], now=1005.0) is None
        said = agent.tick([fall], now=1006.0)
        assert said == "I detected a fall. Please call for help now."
        assert agent.tick([fall], now=1007.0) is None
    finally:
        agent.close()


def test_explicit_help_interrupts_once_during_a_thinking_pause():
    """A person's deterministic help request receives immediate spoken guidance."""
    agent = _agent()
    try:
        assert agent.tick([], now=1000.0) == SMALL_TALK
        agent.listener.queue.append(("help me", 1004.0))
        assert agent.tick([], now=1005.0) == (
            "I heard you ask for help. Please call for help now.")
        assert agent.tick([], now=1006.0) is None
    finally:
        agent.close()
