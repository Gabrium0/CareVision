"""Unit tests for the guest/client demo circuit ('d' hotkey / --demo).

Exercises VoiceAgent.start_demo_circuit()/_advance_demo_circuit() against a
private WorkflowEngine (not the process-wide singleton) so tests stay
isolated, mirroring the pattern in assessment_library_test.py.
"""
from __future__ import annotations

from agent.voice_agent import VoiceAgent, _DEMO_CIRCUIT
from core.workflows import WorkflowEngine
from storage.event_store import EventStore


def _fresh_agent(tmp_path):
    agent = VoiceAgent(speak=False, moondream_enabled=False)
    agent.workflows = WorkflowEngine(event_store=EventStore(tmp_path / "events.sqlite3"))
    return agent


def test_start_demo_circuit_starts_first_protocol(tmp_path):
    agent = _fresh_agent(tmp_path)
    assert agent.start_demo_circuit() is True
    session = agent.workflows.active("primary")
    assert session is not None
    assert session.protocol == _DEMO_CIRCUIT[0]
    assert list(agent._demo_queue) == list(_DEMO_CIRCUIT[1:])


def test_demo_circuit_waits_for_the_active_step_to_conclude(tmp_path):
    agent = _fresh_agent(tmp_path)
    agent.start_demo_circuit()
    first = agent.workflows.active("primary").protocol
    agent._advance_demo_circuit()   # still running -> must not skip ahead
    assert agent.workflows.active("primary").protocol == first
    assert len(agent._demo_queue) == len(_DEMO_CIRCUIT) - 1

    agent.workflows.conclude("primary")
    agent._advance_demo_circuit()
    second = agent.workflows.active("primary")
    assert second is not None and second.protocol == _DEMO_CIRCUIT[1]
    assert len(agent._demo_queue) == len(_DEMO_CIRCUIT) - 2


def test_demo_circuit_runs_to_completion_and_stops(tmp_path):
    agent = _fresh_agent(tmp_path)
    agent.start_demo_circuit()
    for expected in _DEMO_CIRCUIT[1:]:
        agent.workflows.conclude("primary")
        agent._advance_demo_circuit()
        assert agent.workflows.active("primary").protocol == expected
    agent.workflows.conclude("primary")
    agent._advance_demo_circuit()
    assert agent.workflows.active("primary") is None
    assert agent._demo_queue == []


def test_start_demo_circuit_ignored_while_already_queued(tmp_path):
    agent = _fresh_agent(tmp_path)
    agent.start_demo_circuit()
    assert agent.start_demo_circuit() is False


def test_start_demo_circuit_ignored_when_subject_has_a_workflow(tmp_path):
    agent = _fresh_agent(tmp_path)
    agent.workflows.start("sit_to_stand")
    assert agent.start_demo_circuit() is False
    assert agent._demo_queue == []


def _advance_to_arm_drift(agent):
    """Run the circuit forward until arm_drift (the middle step) is active."""
    agent.start_demo_circuit()
    agent.workflows.conclude("primary")   # facial_movement scored -> success
    agent._advance_demo_circuit()
    assert agent.workflows.active("primary").protocol == "arm_drift"


def test_failed_step_is_announced_and_retried_with_guidance(tmp_path):
    agent = _fresh_agent(tmp_path)
    _advance_to_arm_drift(agent)
    pending = len(agent._demo_queue)       # balance still queued behind arm_drift

    agent.workflows.cancel("primary")      # arm_drift never captured
    agent._advance_demo_circuit()

    active = agent.workflows.active("primary")
    assert active is not None and active.protocol == "arm_drift"   # retried, not skipped
    said = agent.last_utterance.lower()
    assert "arm check" in said and "try again" in said            # honest + actionable
    assert len(agent._demo_queue) == pending                      # balance still pending


def test_failed_step_skipped_after_retry_budget_exhausted(tmp_path):
    agent = _fresh_agent(tmp_path)
    _advance_to_arm_drift(agent)

    agent.workflows.cancel("primary")      # first miss -> retry
    agent._advance_demo_circuit()
    assert agent.workflows.active("primary").protocol == "arm_drift"

    agent.workflows.cancel("primary")      # second miss -> honest skip (announced alone)
    agent._advance_demo_circuit()
    assert "skipping the arm check" in agent.last_utterance.lower()
    assert agent.workflows.active("primary") is None

    agent._advance_demo_circuit()          # next tick moves on to the final step
    active = agent.workflows.active("primary")
    assert active is not None and active.protocol == "balance"


def test_concluded_step_produces_no_skip_line(tmp_path):
    agent = _fresh_agent(tmp_path)
    _advance_to_arm_drift(agent)
    said = agent.last_utterance.lower()
    assert "skip" not in said and "couldn't capture" not in said


def test_timed_out_step_is_announced_and_retried(tmp_path):
    agent = _fresh_agent(tmp_path)
    _advance_to_arm_drift(agent)
    session = agent.workflows.active("primary")

    agent.workflows.tick(now=session.deadline + 1)     # deadline lapses -> TIMED_OUT
    assert agent.workflows.active("primary") is None
    agent._advance_demo_circuit()

    assert "try again" in agent.last_utterance.lower()
    assert agent.workflows.active("primary").protocol == "arm_drift"


def test_failure_on_final_step_is_announced(tmp_path):
    agent = _fresh_agent(tmp_path)
    _advance_to_arm_drift(agent)
    agent.workflows.conclude("primary")    # arm_drift scored -> balance (final step)
    agent._advance_demo_circuit()
    assert agent.workflows.active("primary").protocol == "balance"

    agent.workflows.cancel("primary")      # final step miss -> retry
    agent._advance_demo_circuit()
    assert agent.workflows.active("primary").protocol == "balance"

    agent.workflows.cancel("primary")      # miss again -> honest skip, nothing after
    agent._advance_demo_circuit()
    assert "skipping the balance check" in agent.last_utterance.lower()
    assert agent.workflows.active("primary") is None
    assert agent._demo_queue == []
