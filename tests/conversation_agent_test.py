"""Connected companion context, steering, confirmation, and vision contracts."""
from __future__ import annotations

import json
import time

import numpy as np

from agent.conversation import (
    AgentContextBroker, AgentResponse, ConversationTurn, TopicQueue, VisionCadence,
    guard_agent_only_speech,
)
from core.events import Result, Severity, Visibility


def _result(module="heart_rate", key="bpm", value=72, *, timestamp=100.0,
            visibility=Visibility.PUBLIC, message="Heart rate is available",
            confidence=.8, quality=.7, ttl=30.0):
    return Result(module, key, value, confidence, Severity.INFO, message, ttl=ttl,
                  visibility=visibility, timestamp=timestamp, quality=quality,
                  source="test_detector", location="living_room")


def test_broker_exposes_public_and_private_results_with_provenance_and_uncertainty():
    broker = AgentContextBroker()
    public = _result()
    private = _result("skin_vision", "hypothesis",
                      {"finding": "uncertain", "image_data": "forbidden"},
                      visibility=Visibility.AGENT_ONLY,
                      message="An uncertain visual hypothesis")
    broker.ingest([public, private], now=105.0)

    context = broker.build("What have you noticed?", [
        ConversationTurn("user", "What have you noticed?", 105.0)],
        workflows=[{"protocol": "balance", "stage": "positioning"}],
        capabilities=[{"name": "camera", "status": "ready"}],
        recent_events=[{"kind": "arrival", "timestamp": 104.0}],
        history_trends=[{"module": "heart_rate", "key": "bpm", "mean_24h": 71.0}])

    records = {item.id: item.prompt_record() for item in context.items}
    private_record = next(record for record in records.values()
                          if record["module"] == "skin_vision")
    assert private_record["visibility"] == "agent_only"
    assert "not a fact" in private_record["uncertainty"]
    assert "image_data" not in json.dumps(private_record)
    assert {item.kind for item in context.items} >= {
        "live_result", "workflow", "capability", "event_summary", "history_trend"}
    assert all(item.id in context.selected_item_ids for item in context.items)


def test_broker_drops_expired_results_and_diagnostics_never_expose_values():
    broker = AgentContextBroker()
    broker.ingest([_result(value="private-value", timestamp=10.0, ttl=2.0)], now=20.0)
    assert broker.items() == []
    assert "private-value" not in repr(broker.diagnostics())


def test_topic_queue_asks_about_private_hypothesis_without_repeating_it():
    broker = AgentContextBroker()
    result = _result("vlm", "private", "condition-name", timestamp=100.0,
                     visibility=Visibility.AGENT_ONLY,
                     message="condition-name", confidence=.7)
    result.severity = Severity.NOTICE
    broker.ingest([result], now=101.0)
    queue = TopicQueue(repeat_cooldown=600)
    queue.observe(broker.items(), 101.0)
    topic = queue.next(101.0)
    assert topic is not None
    assert "condition-name" not in topic.fallback
    queue.mark_raised(topic, 101.0)
    assert queue.next(102.0) is None


def test_info_rows_are_queryable_but_never_drive_unsolicited_topics():
    broker = AgentContextBroker()
    broker.ingest([_result("replay", "fixture", "loaded", timestamp=100.0,
                           message="Development fixture loaded")], now=101.0)
    queue = TopicQueue()
    queue.observe(broker.items(), 101.0)
    assert broker.items()
    assert queue.next(101.0) is None


def test_private_hypothesis_assertion_is_replaced_by_safe_fallback():
    broker = AgentContextBroker()
    broker.ingest([_result("skin_vision", "hypothesis", "eczema", timestamp=100.0,
                           visibility=Visibility.AGENT_ONLY,
                           message="Possible eczema hypothesis")], now=101.0)
    text, reason = guard_agent_only_speech(
        "It looks like you have eczema.", broker.items(),
        "I have an uncertain observation; would you like to check it together?")
    assert "eczema" not in text.lower()
    assert reason == "agent_only_assertion_blocked"


def test_change_driven_vision_keeps_only_one_due_frame_and_has_heartbeat():
    cadence = VisionCadence(heartbeat_seconds=20, min_spacing_seconds=8,
                            change_threshold=5)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cadence.observe(frame, active=True, now=0.0)
    first = cadence.take_due()
    assert first is not None and cadence.diagnostics()["in_flight"] is True
    cadence.observe(np.full_like(frame, 255), active=True, now=1.0)
    assert cadence.take_due() is None
    cadence.complete(1.0)
    cadence.observe(np.full_like(frame, 255), active=True, now=10.0)
    changed = cadence.take_due()
    assert changed is not None
    cadence.complete(10.0)
    cadence.observe(frame, active=True, now=31.0)
    assert cadence.take_due() is not None


class _Listener:
    available = True

    def __init__(self):
        self.queue = []

    def pop_utterances(self):
        out, self.queue = self.queue, []
        return out

    def close(self):
        pass


def test_requested_assessment_waits_for_explicit_confirmation(monkeypatch):
    import agent.moondream_client as moondream_module
    from agent.voice_agent import VoiceAgent

    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: None)
    listener = _Listener()
    agent = VoiceAgent(speak=False, listener=listener, moondream_enabled=False,
                       min_gap=0)
    agent.elicitation.clear()
    try:
        listener.queue.append(("Please check my tremor", 1000.0))
        proposal = agent.tick([], now=1000.0)
        assert proposal and "would you like" in proposal.lower()
        assert not agent.elicitation.active("hold_still")
        listener.queue.append(("yes, please", 1010.0))
        instruction = agent.tick([], now=1010.0)
        assert instruction and "eight seconds" in instruction.lower()
        assert agent.elicitation.active("hold_still")
    finally:
        agent.elicitation.clear()
        agent.close()


def test_denied_action_is_cancelled_without_starting(monkeypatch):
    import agent.moondream_client as moondream_module
    from agent.voice_agent import VoiceAgent

    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: None)
    listener = _Listener()
    agent = VoiceAgent(speak=False, listener=listener, moondream_enabled=False,
                       min_gap=0)
    agent.elicitation.clear()
    try:
        listener.queue.append(("Check my hand tremor", 1000.0))
        assert agent.tick([], now=1000.0)
        listener.queue.append(("No, cancel that", 1010.0))
        agent.tick([], now=1010.0)
        assert agent._pending_action is None
        assert not agent.elicitation.active("hold_still")
    finally:
        agent.elicitation.clear()
        agent.close()


class _StructuredProvider:
    def __init__(self):
        self.request = None

    def submit_response(self, messages, context_items, image=None):
        self.request = (messages, context_items, image)
        return "request"

    def poll_response(self, _request_id):
        return True, AgentResponse("I can answer that. By the way, shall we discuss the new observation?",
                                   provider_status="ready")

    def submit_generation(self, *_args):
        raise AssertionError("structured path should be used")

    def select_topic(self, *_args):
        return None

    def status(self):
        return {"active": True}

    def close(self):
        pass


def test_free_speech_uses_multiturn_structured_context_and_answer_first_steering(monkeypatch):
    import agent.moondream_client as moondream_module
    from agent.voice_agent import VoiceAgent

    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: None)
    listener = _Listener()
    agent = VoiceAgent(speak=False, listener=listener, moondream_enabled=False,
                       min_gap=0)
    agent.moondream.close()
    provider = _StructuredProvider()
    agent.moondream = provider
    try:
        listener.queue.append(("What is my heart rate?", 1001.0))
        changed = _result("routine", "change", True, timestamp=1000.0,
                          message="Activity changed", confidence=.8)
        changed.severity = Severity.NOTICE
        snapshot = [_result(timestamp=1000.0), changed]
        assert agent.tick(snapshot, now=1001.0) is None
        messages, items, image = provider.request
        assert image is None
        assert any(message["role"] == "user" and
                   "What is my heart rate?" in message["content"]
                   for message in messages)
        assert any("First answer" in message["content"] for message in messages)
        assert {item.module for item in items} >= {"heart_rate", "routine"}
        spoken = agent.tick(snapshot, now=1002.0)
        assert spoken.startswith("I can answer")
        assert agent.memory.dialogue[-1][0] == "you"
    finally:
        agent.close()


def test_conversation_diagnostics_contain_no_transcript_or_private_value(monkeypatch):
    import agent.moondream_client as moondream_module
    from agent.voice_agent import VoiceAgent

    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: None)
    agent = VoiceAgent(speak=False, moondream_enabled=False)
    try:
        agent.memory.person_said("my private transcript", time.time())
        agent.context_broker.ingest([
            _result(value="private-value", timestamp=time.time())])
        diagnostic = repr(agent.conversation_diagnostics())
        assert "private transcript" not in diagnostic
        assert "private-value" not in diagnostic
    finally:
        agent.close()


def test_moondream_structured_response_sends_multiturn_context_and_bounded_image(monkeypatch):
    import agent.moondream_client as moondream_module
    from agent.moondream_client import MoondreamClient

    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: "secret")
    captured = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": "The room looks calm."}}]}).encode()

    def urlopen(request, timeout):
        captured.append((json.loads(request.data), timeout))
        return _Response()

    monkeypatch.setattr(moondream_module.urllib.request, "urlopen", urlopen)
    broker = AgentContextBroker()
    broker.ingest([_result(timestamp=time.time())])
    client = MoondreamClient(enabled=True)
    try:
        response = client.respond(
            [{"role": "user", "content": "What do you see?"}], broker.items(),
            image=np.zeros((1080, 1920, 3), dtype=np.uint8))
        assert response and response.text == "The room looks calm."
        payload = captured[0][0]
        assert [message["role"] for message in payload["messages"][:2]] == [
            "system", "system"]
        image_parts = [part for message in payload["messages"]
                       if isinstance(message["content"], list)
                       for part in message["content"] if part["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert len(image_parts[0]["image_url"]["url"]) < 700_000
        assert "secret" not in json.dumps(payload)
    finally:
        client.close()
