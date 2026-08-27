"""Offline reply variety and provider-retirement guardrails.

Covers three live-showcase failure modes: an unreachable language model must
never repeat one canned acknowledgement, a retired hosted model id (HTTP 410)
must latch off requests with one clear announcement instead of a failed-request
line per utterance, and a generation that reads like a status report rather than
speech must never be spoken at all.
"""
from __future__ import annotations

import threading
import urllib.error

from agent.moondream_client import MoondreamClient
from agent.policy import Intent
from agent.voice_agent import VoiceAgent, _REPLY_FALLBACKS


def _bare_client(*, retired: bool = False) -> MoondreamClient:
    client = MoondreamClient.__new__(MoondreamClient)  # skip network/key init
    client._lock = threading.RLock()
    client._closed = False
    client._enabled = True
    client.available = True
    client._key = "test-key"
    client._generation_requests = 0
    client._classification_requests = 0
    client._last_request_at = None
    client._failures = 0
    client._successes = 0
    client._last_error = None
    client._last_http_status = None
    client._consecutive_failures = 0
    client._circuit_open_until = 0.0
    client._authorization_failed = False
    client._model_retired = retired
    client._retryable = not retired
    client._lifecycle = "model_retired" if retired else "configured"
    client._last_announced_label = None
    return client


def _bare_agent() -> VoiceAgent:
    agent = object.__new__(VoiceAgent)
    agent.last_utterance = ""
    agent._reply_fallback_idx = 0
    return agent


def test_every_reply_fallback_reads_like_speech():
    for line in _REPLY_FALLBACKS:
        assert VoiceAgent._clean_spoken_line(line) == line, line


# --------------------------------------------------------- spoken-line airlock

def test_announcing_a_gap_in_the_data_is_never_spoken():
    """A companion answers what it can; it does not read out its coverage."""
    for line in ("I don't have that information.",
                 "I don't have any information about Instagram.",
                 "I'm sorry, I have no information about the news today.",
                 "I don't have access to that right now."):
        assert VoiceAgent._clean_spoken_line(line) == "", line


def test_quoting_the_person_or_citing_the_data_is_never_spoken():
    """Narrating the inputs is not conversation."""
    for line in ("You said you're going to end the video.",
                 "You mentioned you're not going home.",
                 "It said your skin colour is looking very normal.",
                 "According to the data, your breathing seems calm."):
        assert VoiceAgent._clean_spoken_line(line) == "", line


def test_invented_claims_and_leading_tag_questions_are_never_spoken():
    """An intent that may only ASK must not come back as an assertion.

    "You enjoy a good cup of tea, don't you?" was generated from a small-talk
    intent whose whole content was to ask whether they had had a cup of tea:
    the preference, and the agreement it invites, are both fabricated.
    """
    for line in ("You enjoy a good cup of tea, don't you?",
                 "You always take a walk after lunch.",
                 "You usually rest at this time.",
                 "You're feeling much better, aren't you?"):
        assert VoiceAgent._clean_spoken_line(line) == "", line


def test_ordinary_offers_and_similes_still_speak():
    """The guards are narrow: normal warm phrasing must survive them."""
    for line in ("Would you like a cup of tea?",
                 "It's lovely chatting with you like this.",
                 "I'd like to hear about your morning, if you'd like to share.",
                 "Have you had a nice cup of tea or coffee yet today?",
                 "That sounds lovely, doesn't it?"):
        assert VoiceAgent._clean_spoken_line(line) == line, line


# --------------------------------------------- the busy-provider reply lane

def _reply_agent() -> VoiceAgent:
    """A speak-free offline agent, as tests/turn_taking_test.py builds one."""
    return VoiceAgent(name="Ada", speak=False, moondream_enabled=False,
                      min_gap=0)


def test_busy_provider_replies_rotate_instead_of_repeating():
    """The reply lane must not speak fallback zero on every single turn.

    There is one in-flight generation slot, so `tick` frequently hands this
    lane its own `intent.fallback` as the "generated" candidate. That string
    reads like flawless speech, so treating it as a generation spoke
    _REPLY_FALLBACKS[0] forever — the "I'm glad you told me that." loop from
    the live run. An unchanged fallback means nothing was generated.
    """
    agent = _reply_agent()
    try:
        spoken = []
        for index in range(4):
            intent = Intent("reply", f"reply:{index}", "", "",
                            _REPLY_FALLBACKS[0], 80)
            # 10s apart: well clear of the reply cadence gate, so any repetition
            # here is the lane's choice rather than a suppressed turn.
            spoken.append(agent._speak_intent(
                intent, intent.fallback, 1000.0 + index * 10.0, action=None))
        assert all(line for line in spoken)
        assert len(set(spoken)) == len(spoken), spoken
        assert set(spoken) <= set(_REPLY_FALLBACKS)
    finally:
        agent.close()


def test_a_real_generation_still_wins_over_the_rotation():
    agent = _reply_agent()
    try:
        intent = Intent("reply", "reply:9", "", "", _REPLY_FALLBACKS[0], 80)
        assert agent._speak_intent(
            intent, "That sounds like a lovely morning.", 1000.0,
            action=None) == "That sounds like a lovely morning."
    finally:
        agent.close()


def test_steer_stays_silent_rather_than_repeating_one_generic_line():
    """A rejected "by the way" is silence, not the same sentence again.

    Every topic shares one generic steer fallback, so speaking it whenever the
    generation is missing or unspeakable turns a demo into that sentence on
    repeat. The topic is already marked raised by the caller, so silence costs
    nothing.
    """
    agent = _reply_agent()
    try:
        generic = "By the way, how are you feeling right now — is anything bothering you?"
        intent = Intent("observation", "steer:primary:yawn:yawn", "", "",
                        generic, 42)
        assert agent._speak_intent(intent, generic, 1000.0, action=None) == ""
        assert agent._speak_intent(
            intent, "It said you yawned.", 1100.0, action=None) == ""
        # A real, speakable line does go out.
        assert agent._speak_intent(
            intent, "Are you feeling a little tired?", 1200.0,
            action=None) == "Are you feeling a little tired?"
    finally:
        agent.close()


def test_rotation_never_repeats_the_prior_line():
    agent = _bare_agent()
    picks = [agent._next_reply_fallback()
             for _ in range(len(_REPLY_FALLBACKS) * 3)]
    for previous, current in zip(picks, picks[1:]):
        assert previous != current, (previous, current)
    assert set(picks) <= set(_REPLY_FALLBACKS)
    # A full cycle covers every variant before any repeats.
    first_cycle = picks[:len(_REPLY_FALLBACKS)]
    assert len(set(first_cycle)) == len(_REPLY_FALLBACKS)


def test_http_410_latches_model_retired_and_throttles_logs():
    client = _bare_client()

    gone = urllib.error.HTTPError("https://example.test", 410, "Gone", {}, None)

    first = client._record_failure(gone, "generation")
    assert first is not None and "410" in first and "--voice-model" in first
    assert client._lifecycle == "model_retired"
    assert client._model_retired is True
    assert client._retryable is False
    # The retirement latch stops further requests without a circuit timer.
    assert client._begin_request("generation") is None

    # Same-shape failures stay quiet until the tenth consecutive one.
    repeat_notes = [client._record_failure(gone, "generation") for _ in range(9)]
    assert all(note is None for note in repeat_notes[:8])
    assert repeat_notes[8] is not None


def test_explicit_toggle_rearms_after_retirement():
    client = _bare_client(retired=True)

    client.set_enabled(False)
    client.set_enabled(True)
    assert client._model_retired is False
    assert client._begin_request("generation") == "test-key"
