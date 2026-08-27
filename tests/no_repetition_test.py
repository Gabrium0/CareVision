"""The deterministic no-repeat guard in VoiceAgent._speak_intent.

A failed or rate-limited generation falls back to a fixed template, so the same
line can be selected again on a later turn. The guard must drop that repeat
(stay silent, still advancing cadence) while letting a genuinely new line
through, and it must never suppress a line with a bound action or a conclusion.
All offline: no model, no network, no audio.
"""
from __future__ import annotations

import pytest

from agent.policy import Intent
from agent.voice_agent import VoiceAgent, is_near_duplicate, _normalize_spoken


@pytest.fixture
def agent(monkeypatch, tmp_path):
    import agent.moondream_client as moondream_module
    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: None)
    # Isolate the singleton stores to a temp DB so this test never touches (or
    # locks against) a running app's data/events.sqlite3.
    from storage.event_store import EventStore
    from storage.history_store import HistoryStore
    monkeypatch.setattr(EventStore, "_instance",
                        EventStore(path=tmp_path / "events.sqlite3"), raising=False)
    monkeypatch.setattr(HistoryStore, "_instance",
                        HistoryStore(path=tmp_path / "history.db"), raising=False)
    a = VoiceAgent(speak=False, moondream_enabled=False, min_gap=0,
                   small_talk_interval=1e12)
    a.elicitation.clear()
    yield a
    a.close()


def _line(text, sig):
    """A plain conversational intent whose fallback is `text` (no bound action)."""
    return Intent("small_talk", sig, "", "", text, 10)


def test_near_duplicate_helper():
    recent = [_normalize_spoken("I'm glad you told me that.")]
    assert is_near_duplicate("I'm glad you told me that.", recent)
    assert is_near_duplicate("I am really glad you told me that.", recent)
    assert not is_near_duplicate("How are you feeling this evening?", recent)
    assert not is_near_duplicate("", recent)


def test_repeated_line_is_suppressed_but_a_new_one_is_not(agent):
    first = agent._speak_intent(_line("I'm glad you told me that.", "reply:a"), None,
                                now=100.0)
    assert first == "I'm glad you told me that."
    assert agent.suppressed_repeats == 0

    # Same line, different intent/turn: must be dropped, not spoken again.
    second = agent._speak_intent(_line("I'm glad you told me that.", "reply:b"), None,
                                 now=200.0)
    assert second is None
    assert agent.suppressed_repeats == 1
    assert agent.conversation_diagnostics()["suppressed_repeats"] == 1

    # A genuinely different line still gets spoken.
    third = agent._speak_intent(_line("How are you feeling this evening?", "reply:c"),
                                None, now=300.0)
    assert third == "How are you feeling this evening?"
    assert agent.suppressed_repeats == 1


def test_conclusions_and_actions_are_never_suppressed(agent):
    agent._speak_intent(_line("The same important note.", "reply:x"), None, now=100.0)
    # A conclusion repeating the text is exempt (safety/clinical lines must land).
    concl = Intent("conclusion", "conclude:y", "", "", "The same important note.", 90)
    assert agent._speak_intent(concl, None, now=200.0) == "The same important note."
    assert agent.suppressed_repeats == 0
