"""Echo-loop guardrails: the agent must never hear and answer itself.

Device-routed speech plays centimetres from the paired microphone. These
tests pin the four deterministic defenses — audible-end turn-taking, whisper
hallucination rejection, self-echo transcript filtering, and reply cadence —
plus the silence-over-canned choice when a generated reply lands too late.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from audio import tts as tts_mod
from audio.stt import (
    _MAX_AGENT_MUTE_SECONDS, Listener, _segment_is_trustworthy,
)
from agent.voice_agent import (
    _ANSWER_EXEMPT_MAX_WORDS, _ECHO_LOOKBACK_SECONDS, _REPLY_MIN_GAP_SECONDS,
    _REPLY_WINDOW_MAX, VoiceAgent, _estimate_spoken_seconds,
    _is_degenerate_repetition, _is_likely_echo, _repeats_agent_phrase,
)


# ---------------------------------------------------------------- tts pacing

class _StubBackend:
    """Synthesizes known-length silence so pacing math is exact."""

    def __init__(self, seconds_per_chunk: float, rate: int = 16000):
        self.seconds = seconds_per_chunk
        self.rate = rate
        self.engine = object()

    def load(self):
        return self.engine

    def synthesize(self, engine, text):
        n = int(self.seconds * self.rate)
        yield np.zeros(n, dtype=np.float32), self.rate


def _bare_speaker(backend: _StubBackend) -> tts_mod.Speaker:
    speaker = object.__new__(tts_mod.Speaker)
    speaker.enabled = True
    speaker.speaking = False
    speaker.engine_name = "piper"
    speaker._backend = backend
    speaker._engine = backend.engine
    speaker.remote_sink = None
    speaker.local_playback = True
    speaker.on_speech_state = None
    return speaker


class _Sink:
    def __init__(self):
        self.chunks = []

    def __call__(self, samples, rate):
        self.chunks.append((len(samples), rate))


def test_device_routed_speech_holds_speaking_for_audible_duration(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: sleeps.append(s))
    # Focus on real-chunk pacing; the lead pad has its own test below.
    monkeypatch.setattr(tts_mod, "_REMOTE_LEAD_SECONDS", 0.0)
    speaker = _bare_speaker(_StubBackend(seconds_per_chunk=2.0))
    sink = _Sink()
    speaker.set_remote_sink(sink, local_playback=False)
    # Mirror the worker thread's contract around _emit.
    speaker.speaking = True
    speaker._emit("One sentence. Two sentences.", None)
    speaker.speaking = False
    assert len(sink.chunks) == 2
    # Each chunk is held for its real duration, plus the remote tail once.
    assert sleeps == [2.0, 2.0, tts_mod._REMOTE_TAIL_SECONDS]


def test_device_route_prepends_silent_lead_pad(monkeypatch):
    """A silent pad precedes the first word so the iPad's Bluetooth (A2DP) route
    has time to settle after the mic is released — otherwise iOS clips or briefly
    plays the opening on the built-in speaker."""
    sleeps: list[float] = []
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(tts_mod, "_REMOTE_LEAD_SECONDS", 0.5)
    speaker = _bare_speaker(_StubBackend(seconds_per_chunk=2.0))
    sink = _Sink()
    speaker.set_remote_sink(sink, local_playback=False)
    speaker.speaking = True
    speaker._emit("One sentence.", None)
    speaker.speaking = False
    # First routed block is the silent pad; then the one real chunk.
    assert len(sink.chunks) == 2
    pad_len, pad_rate = sink.chunks[0]
    assert pad_len == int(0.5 * pad_rate)                 # 0.5s of silence
    # The pad is held for its duration, then the real chunk, then the tail.
    assert sleeps == [0.5, 2.0, tts_mod._REMOTE_TAIL_SECONDS]


def test_lead_pad_is_skipped_on_the_local_playback_path(monkeypatch):
    """The pad is a device-route affordance; local laptop audio must not get it."""
    sleeps: list[float] = []
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(tts_mod, "_REMOTE_LEAD_SECONDS", 0.5)
    speaker = _bare_speaker(_StubBackend(seconds_per_chunk=1.0))

    class _Player:
        def play(self, samples, rate):
            pass

        def wait(self):
            pass

    speaker._emit("Hello.", _Player())      # local_playback stays True
    assert sleeps == []


class _StateRecorder:
    def __init__(self):
        self.events: list[bool] = []

    def __call__(self, active):
        self.events.append(active)


def test_speech_state_callback_fires_on_speaking_edges(monkeypatch):
    """The True edge must precede audio (mic release first) and the False edge
    must follow it (mic re-acquire after the line drains)."""
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(tts_mod, "_REMOTE_LEAD_SECONDS", 0.0)
    speaker = _bare_speaker(_StubBackend(seconds_per_chunk=0.1))
    order: list[str] = []
    speaker.set_remote_sink(lambda samples, rate: order.append("audio"),
                            local_playback=False)

    def _record(active):
        order.append("speak" if active else "listen")
        assert speaker.speaking is active     # flag is exact at the edge

    speaker.set_speech_state_callback(_record)
    speaker._speak_one("One sentence.", None)
    assert order[0] == "speak"                # released before any audio
    assert order[-1] == "listen"              # re-acquired after audio
    assert "audio" in order[1:-1]
    assert not speaker.speaking


def test_speech_state_callback_errors_never_mute_the_agent(monkeypatch):
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(tts_mod, "_REMOTE_LEAD_SECONDS", 0.0)
    speaker = _bare_speaker(_StubBackend(seconds_per_chunk=0.1))
    speaker.set_remote_sink(_Sink(), local_playback=False)
    speaker.set_speech_state_callback(lambda active: (_ for _ in ()).throw(RuntimeError("boom")))
    speaker._speak_one("Hello.", None)        # must not raise
    assert not speaker.speaking


def test_agent_audio_control_message_shape(monkeypatch):
    """Mirrors main.py's wiring: each speaking edge produces an `agent_audio`
    control message the iPad turns into a mic release/acquire."""
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(tts_mod, "_REMOTE_LEAD_SECONDS", 0.0)
    speaker = _bare_speaker(_StubBackend(seconds_per_chunk=0.1))
    speaker.set_remote_sink(_Sink(), local_playback=False)
    sent: list[dict] = []
    speaker.set_speech_state_callback(lambda active: sent.append(
        {"type": "agent_audio", "phase": "speaking" if active else "listening"}))
    speaker._speak_one("Hello there.", None)
    assert sent == [
        {"type": "agent_audio", "phase": "speaking"},
        {"type": "agent_audio", "phase": "listening"},
    ]


def test_local_playback_path_is_unchanged(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: sleeps.append(s))
    speaker = _bare_speaker(_StubBackend(seconds_per_chunk=1.0))

    class _Player:
        def play(self, samples, rate):
            pass

        def wait(self):
            pass

    speaker._emit("Hello.", _Player())
    # Only the inter-chunk pause path exists locally (single chunk: none).
    assert sleeps == []


# ------------------------------------------------- whisper trustworthiness

def test_low_confidence_segments_are_rejected():
    assert not _segment_is_trustworthy(
        "Thanks for watching!", avg_logprob=-0.5, no_speech_prob=0.7)
    assert not _segment_is_trustworthy(
        "I'll see you in the next video", avg_logprob=-1.4, no_speech_prob=0.2)
    assert not _segment_is_trustworthy(
        "Thanks for watching and subscribing", avg_logprob=-0.9,
        no_speech_prob=0.45)


def test_normal_confident_segments_pass():
    assert _segment_is_trustworthy(
        "Thank you, dear.", avg_logprob=-0.4, no_speech_prob=0.1)
    assert _segment_is_trustworthy(
        "See you in the next video", avg_logprob=-0.3, no_speech_prob=0.05)


# ------------------------------------------------------------ echo filter

def test_garbled_self_utterance_is_detected_as_echo():
    recent = ["It's lovely chatting with you like this."]
    assert _is_likely_echo("lovely chatting with you like this", recent)
    # Word-order-preserving phrase runs survive garbling and are caught even
    # when overall similarity is moderate.
    assert _is_likely_echo("i'm gonna keep you tall to meet the rest of you",
                           ["I'm glad to have a chat with the rest of you."])


def test_short_answers_to_agent_questions_are_never_dropped():
    agent = object.__new__(VoiceAgent)
    from agent.state import ObservationMemory
    agent.memory = ObservationMemory()
    now = time.time()
    agent.memory.agent_said("How has your day been so far?", now - 2.0)
    agent._echo_drops = 0
    kept = agent._drop_echoes(
        [("Good so far", now), ("It was good.", now), ("Yes please", now)], now)
    assert [text for text, _ts in kept] == \
        ["Good so far", "It was good.", "Yes please"]
    assert agent._echo_drops == 0


def test_person_question_and_help_requests_are_exempt():
    agent = object.__new__(VoiceAgent)
    from agent.state import ObservationMemory
    agent.memory = ObservationMemory()
    now = time.time()
    agent.memory.agent_said("I'm right here whenever you need me.", now - 2.0)
    agent._echo_drops = 0
    kept = agent._drop_echoes(
        [("are you right here?", now), ("help me stand up", now),
         ("I need help reaching the shelf", now)], now)
    assert len(kept) == 3


def test_asr_repetition_loops_are_dropped_but_emphatic_kept():
    assert _is_degenerate_repetition(
        "I'm gonna go home, I'm gonna go home, "
        "I'm gonna go home, I'm gonna go home.")
    assert not _is_degenerate_repetition(
        "I want to go home now please and then I would like to rest a bit.")
    agent = object.__new__(VoiceAgent)
    from agent.state import ObservationMemory
    agent.memory = ObservationMemory()
    now = time.time()
    agent._echo_drops = 0
    kept = agent._drop_echoes([("no no no no", now)], now)
    assert len(kept) == 1          # short emphatic repeats stay conversational


def test_new_person_speech_is_never_echo():
    recent = ["How has your day been so far?",
              "I'm glad you told me that."]
    assert not _is_likely_echo("I need help reaching the shelf", recent)
    assert not _is_likely_echo("", recent)


def test_short_agent_line_cannot_swallow_long_remark():
    assert not _is_likely_echo(
        "please bring me my blanket and my reading glasses from the table",
        ["Okay."])


def test_drop_echoes_uses_recent_dialogue_window():
    agent = object.__new__(VoiceAgent)
    from agent.state import ObservationMemory
    agent.memory = ObservationMemory()
    now = time.time()
    agent.memory.agent_said("I'm glad you told me that.", now - 2.0)
    agent.memory.agent_said("How has your day been?", now - 400.0)
    agent._echo_drops = 0
    heard = [("glad you told me that", now), ("what's for lunch", now)]
    kept = agent._drop_echoes(heard, now)
    assert [text for text, _ts in kept] == ["what's for lunch"]
    assert agent._echo_drops == 1


def _echo_agent(*agent_lines, now: float) -> VoiceAgent:
    """A bare agent whose dialogue holds the given recent agent utterances."""
    from agent.state import ObservationMemory
    agent = object.__new__(VoiceAgent)
    agent.memory = ObservationMemory()
    for line in agent_lines:
        agent.memory.agent_said(line, now - 2.0)
    agent._echo_drops = 0
    return agent


def test_echo_carrying_answer_vocabulary_is_still_dropped():
    """The answer exemption must not be a hole the agent's own voice fits through.

    Our voice comes back full of the words `interpret_answer` is built from
    ("don't", "okay", "actually"), so a garbled self-transcription reads as a
    confident yes or no. Long echoes that repeat a phrase we just said are
    dropped regardless of how answer-shaped they look.
    """
    now = time.time()
    agent = _echo_agent("It can feel like we're wrapping up the day, can't it?",
                        now=now)
    heard = [("You don't like it? We're wrapping up the day already", now)]
    assert agent._drop_echoes(heard, now) == []
    assert agent._echo_drops == 1

    # Same shape, this time exempted by a stray confirmation token.
    agent = _echo_agent("I'm glad to have a chat with the rest of you.", now=now)
    assert agent._drop_echoes(
        [("okay i'm glad to have a chat with the rest of you", now)], now) == []


def test_short_yes_no_answers_stay_exempt_even_after_a_similar_agent_line():
    """Brevity is what makes the exemption safe: a real answer is short."""
    now = time.time()
    agent = _echo_agent("Would you like me to start the arm check now?", now=now)
    kept = agent._drop_echoes([("No thanks", now), ("Yes please", now),
                               ("okay", now)], now)
    assert [text for text, _ts in kept] == ["No thanks", "Yes please", "okay"]
    assert agent._echo_drops == 0
    assert _ANSWER_EXEMPT_MAX_WORDS <= 4      # "answer", not "sentence"


def test_help_requests_are_exempt_unconditionally():
    """A missed call for help costs more than a spurious one."""
    now = time.time()
    agent = _echo_agent("Please call for help if you need to, I'm right here.",
                        now=now)
    kept = agent._drop_echoes([("please call for help if you need to", now)], now)
    assert len(kept) == 1


def test_repeats_agent_phrase_ignores_short_acknowledgements():
    assert not _repeats_agent_phrase("i would like some tea please",
                                     ["Okay.", "Of course."])
    # Punctuation-blind on both sides: ASR emits none, and the comma after
    # "day" in the spoken line must not break the phrase run.
    assert _repeats_agent_phrase(
        "something something wrapping up the day yes",
        ["It can feel like we're wrapping up the day, can't it?"])


def test_echo_lookback_expires():
    agent = object.__new__(VoiceAgent)
    recent_window = _ECHO_LOOKBACK_SECONDS
    assert recent_window < 60      # bounded; stale agent lines never filter


# ----------------------------------------------------------- reply cadence

def _agent_for_cadence() -> VoiceAgent:
    agent = object.__new__(VoiceAgent)
    agent._reply_spoken_at = deque()
    agent._last_fallback_reason = None
    return agent


def test_reply_rapid_fire_is_blocked_then_allows_gap():
    agent = _agent_for_cadence()
    t0 = 1000.0
    assert agent._replies_allowed(t0)          # first reply
    assert not agent._replies_allowed(t0 + 1)  # echo-bait: too soon
    assert not agent._replies_allowed(t0 + _REPLY_MIN_GAP_SECONDS - 0.5)
    assert agent._replies_allowed(t0 + _REPLY_MIN_GAP_SECONDS + 0.1)


def test_reply_window_cap():
    agent = _agent_for_cadence()
    for i in range(_REPLY_WINDOW_MAX):
        assert agent._replies_allowed(1000.0 + i * (_REPLY_MIN_GAP_SECONDS + 1))
    assert not agent._replies_allowed(
        1000.0 + (_REPLY_WINDOW_MAX) * (_REPLY_MIN_GAP_SECONDS + 1))


# --------------------------------------------- listener mute window state

def test_mark_agent_spoke_extends_mute_by_estimate():
    listener = object.__new__(Listener)
    listener.speaker = None
    listener.speech_tail_seconds = 0.5
    listener._last_agent_speech = -1e9
    listener._agent_speech_until = -1e9
    start = time.time()
    listener.mark_agent_spoke(start, estimated_seconds=_MAX_AGENT_MUTE_SECONDS)
    assert listener._agent_speech_until >= start + _MAX_AGENT_MUTE_SECONDS - 0.01
    # Without an estimate only the short tail applies.
    listener2 = object.__new__(Listener)
    listener2.speaker = None
    listener2.speech_tail_seconds = 0.5
    listener2._last_agent_speech = -1e9
    listener2._agent_speech_until = -1e9
    listener2.mark_agent_spoke(start)
    assert listener2._agent_speech_until <= start + 0.6


def test_estimate_spoken_seconds_bounds():
    assert _estimate_spoken_seconds("") == 1.0
    long_line = " ".join(["word"] * 200)
    assert _estimate_spoken_seconds(long_line) == 10.0
