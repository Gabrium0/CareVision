"""The showcase demo-injection scenarios actually move the agent.

`webui/inject.py` lets a live showcase feed the agent a synthetic reading (high
heart rate, frequent yawns, low mood, a possible skin change) so an audience can
watch it react instead of having to make a real vital change happen on cue. The
injected signals are ordinary ``Result`` objects that travel the same path as
real detector output, so these tests assert the *reaction*, not just the plumbing:
each scenario reaches the agent's view and drives the response the plan promises.

The heart-rate path is spoken by the pipeline's ``VitalsAdvisor`` (not inside
``VoiceAgent.tick``), so it is exercised through that advisor directly; the
corroboration and mood paths run inside ``tick`` and are asserted end to end.

Everything is offline: no model, no network, no audio, and the singleton stores
are redirected to a temp DB so the suite never touches or locks a running app's
data/events.sqlite3.

Run standalone:  python -m pytest tests/demo_inject_test.py
"""
from __future__ import annotations

import pytest

from core.events import Result, Severity
from webui.inject import DemoInjector, SCENARIOS


def _results(scenario_id: str, now: float = 1000.0) -> list[Result]:
    return SCENARIOS[scenario_id][2](now)


# ------------------------------------------------------------- the injector

def test_catalog_matches_scenarios():
    catalog = DemoInjector.catalog()
    assert [c["id"] for c in catalog] == list(SCENARIOS)
    assert [c["label"] for c in catalog] == [v[0] for v in SCENARIOS.values()]


def test_every_scenario_is_stamped_synthetic_and_wellformed():
    for sid in SCENARIOS:
        results = _results(sid)
        assert results, f"{sid} produced no signals"
        for r in results:
            assert r.source == "demo_inject", f"{sid} not marked synthetic"
            assert r.module and r.key
            assert r.ttl > 0


def test_trigger_holds_then_expires():
    di = DemoInjector()
    di.trigger("high_heart_rate", now=1000.0)
    held = di.drain(now=1002.0)   # inside the 5 s hold
    assert any(r.module == "heart_rate" and r.key == "bpm" for r in held)
    assert di.drain(now=1006.0) == []   # hold lapsed, nothing re-emitted


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        DemoInjector().trigger("bogus", now=1000.0)


# ------------------------------------------------------------- the reactions

@pytest.fixture
def agent(monkeypatch, tmp_path):
    import agent.moondream_client as moondream_module
    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: None)
    from storage.event_store import EventStore
    from storage.history_store import HistoryStore
    monkeypatch.setattr(EventStore, "_instance",
                        EventStore(path=tmp_path / "events.sqlite3"), raising=False)
    monkeypatch.setattr(HistoryStore, "_instance",
                        HistoryStore(path=tmp_path / "history.db"), raising=False)
    from agent.voice_agent import VoiceAgent
    a = VoiceAgent(name="Ada", speak=False, moondream_enabled=False,
                   listener=None, min_gap=0, small_talk_interval=1e12)
    yield a
    a.close()


def test_possible_rash_makes_the_agent_ask(agent):
    """The low-confidence skin cue becomes a gentle spoken question."""
    said = agent.tick(_results("possible_rash"), now=1000.0)
    assert said is not None and "skin" in said.lower()
    assert agent.corroboration.status("skin_changes") == "asked"


def test_low_mood_reaches_the_agents_view_and_voice(agent):
    """Negative valence flips the agent's mood read and flags the mood follow-up."""
    agent.tick(_results("low_mood"), now=1000.0)
    assert agent.memory.mood() == "low"
    assert agent.corroboration.status("low_mood") == "flagged"


def test_frequent_yawns_reach_the_agents_view(agent):
    """The fatigue cue and the visible yawn count both land in the agent's memory."""
    agent.tick(_results("frequent_yawns"), now=1000.0)
    perclos = agent.memory.get_val("drowsiness", "perclos")
    assert perclos is not None and perclos == pytest.approx(0.34, abs=0.05)
    assert agent.memory.get_val("yawn", "yawn_count_3min") == 7


def test_high_heart_rate_drives_a_vitals_recommendation():
    """The heart-rate path is spoken by the pipeline advisor, so exercise it there."""
    from agent.advisor_engine import VitalsAdvisor
    advice = VitalsAdvisor().evaluate(_results("high_heart_rate"), now=1000.0)
    assert advice, "no vitals recommendation produced for an elevated heart rate"
    rec = advice[0]
    assert rec.module == "vitals_advice" and rec.key == "recommendation"
    assert "heart rate" in str(rec.value).lower()
    assert rec.severity in (Severity.NOTICE, Severity.WARNING)
