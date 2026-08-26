"""Async answer classification and non-blocking topic steering (VoiceAgent).

Keywords decide; the model only refines an "unclear". Every test here is
offline: the provider is a controllable fake, except the final regression proof
which uses the real MoondreamClient with `urlopen` booby-trapped so that any
synchronous network call made *on the frame-loop thread* is recorded and fails
the test.
"""
from __future__ import annotations

import threading
import urllib.error

import pytest

from core.events import Result, Severity


_UNRESOLVED = object()
_UNKNOWN = object()


class _Listener:
    available = True

    def __init__(self):
        self.queue = []

    def pop_utterances(self):
        out, self.queue = self.queue, []
        return out

    def close(self):
        pass


class _ClassifyProvider:
    """Fake classify lane with one in-flight slot, like MoondreamClient's."""

    def __init__(self, accept: bool = True, auto: str | None = None):
        self.accept = accept          # False models unavailable / circuit open
        self.auto = auto              # verdict a submission resolves to at once
        self.submitted: list[tuple[str, str]] = []
        self.polled: list[str] = []
        self.pending: dict[str, object] = {}
        self._n = 0

    # -- classify lane -----------------------------------------------------
    def submit_classification(self, question, answer):
        self.submitted.append((question, answer))
        if not self.accept:
            return None
        if any(value is _UNRESOLVED for value in self.pending.values()):
            return None               # one shared in-flight slot
        self._n += 1
        request_id = f"c{self._n}"
        self.pending[request_id] = self.auto if self.auto else _UNRESOLVED
        return request_id

    def poll_classification(self, request_id):
        self.polled.append(request_id)
        value = self.pending.get(request_id, _UNKNOWN)
        if value is _UNKNOWN:
            return True, None
        if value is _UNRESOLVED:
            return False, None
        self.pending.pop(request_id)
        return True, value

    def resolve(self, request_id, verdict):
        self.pending[request_id] = verdict

    # -- everything else stays offline so tick() speaks the fallback -------
    def submit_generation(self, *_args):
        return None

    def status(self):
        return {"active": False}

    def close(self):
        pass


class _SteerProvider:
    """Fake async topic-selection lane (no synchronous select_topic at all)."""

    def __init__(self):
        self.submitted: list[tuple[list, str]] = []
        self.request_id = "s1"
        self.choice = _UNRESOLVED

    def submit_topic_selection(self, candidates, context):
        self.submitted.append((list(candidates), context))
        return self.request_id

    def poll_topic_selection(self, request_id):
        if self.choice is _UNRESOLVED:
            return False, None
        return True, self.choice

    def submit_generation(self, *_args):
        return None

    def status(self):
        return {"active": False}

    def close(self):
        pass


def _hit(module, key, confidence=0.3):
    return Result(module=module, key=key, value=1, message="cue",
                  severity=Severity.NOTICE, confidence=confidence)


@pytest.fixture
def build_agent(monkeypatch):
    """VoiceAgent with a swapped provider, no speech, no routine chatter."""
    import agent.moondream_client as moondream_module
    from agent.voice_agent import VoiceAgent

    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: None)
    created = []

    def _build(provider, listener=None):
        agent = VoiceAgent(speak=False, listener=listener,
                           moondream_enabled=False, min_gap=0,
                           small_talk_interval=1e12)
        agent.moondream.close()
        agent.moondream = provider
        agent.elicitation.clear()
        created.append(agent)
        return agent

    yield _build
    for agent in created:
        agent.elicitation.clear()
        agent.close()


def _ask_topic(agent, topic="skin_changes", module="rash", key="rash",
               now=1000.0):
    """Put one corroboration topic into the 'asked' state at `now`."""
    agent.corroboration.observe([_hit(module, key)], now)
    agent.corroboration.mark_asked(topic, now)
    return topic


# ------------------------------------------------------- pending + apply

def test_unclear_answer_leaves_a_pending_classification_and_takes_no_action(build_agent):
    listener = _Listener()
    provider = _ClassifyProvider()
    agent = build_agent(provider, listener)
    _ask_topic(agent)

    listener.queue.append(("hmm the birds are singing", 1000.0))
    agent.tick([], now=1000.0)

    pending = agent._pending_classification
    assert pending is not None
    assert (pending.lane, pending.target) == ("corroboration", "skin_changes")
    assert provider.submitted == [
        (agent.corroboration.rules["skin_changes"].question,
         "hmm the birds are singing")]
    # The state machine has NOT moved: no re-ask, no verdict, no event.
    assert agent.corroboration.status("skin_changes") == "asked"
    assert agent.corroboration.topics["skin_changes"].asks == 1
    assert agent.pop_conversation_results() == []
    assert agent.conversation_diagnostics()["classification"] == {
        "pending": True, "lane": "corroboration", "dropped_stale": 0,
        "dropped_deadline": 0, "superseded": 0}


def test_verdict_lands_on_a_later_tick_and_applies_to_the_right_topic(build_agent):
    listener = _Listener()
    provider = _ClassifyProvider()
    agent = build_agent(provider, listener)
    _ask_topic(agent)

    listener.queue.append(("hmm the birds are singing", 1000.0))
    agent.tick([], now=1000.0)
    provider.resolve(agent._pending_classification.request_id, "confirmed")

    agent.tick([], now=1001.0)

    assert agent._pending_classification is None
    assert agent.corroboration.status("skin_changes") == "confirmed"
    results = agent.pop_conversation_results()
    assert [r.key for r in results] == ["skin_changes_confirmed"]


def test_verdict_for_a_re_asked_question_is_dropped_as_stale(build_agent):
    listener = _Listener()
    provider = _ClassifyProvider()
    agent = build_agent(provider, listener)
    _ask_topic(agent)

    listener.queue.append(("hmm the birds are singing", 1000.0))
    agent.tick([], now=1000.0)
    request_id = agent._pending_classification.request_id
    assert agent._pending_classification.question_id == "ask:skin_changes:1"

    # The topic goes round again — a real re-ask, driven through the agent, so
    # `asks` (and with it the question id) moves on before the verdict lands.
    agent.corroboration.apply_verdict("skin_changes", "unclear", 1001.0)
    assert agent.corroboration.status("skin_changes") == "flagged"
    spoken = agent.tick([], now=1001.0)
    assert spoken == agent.corroboration.rules["skin_changes"].question
    assert agent.corroboration.topics["skin_changes"].asks == 2

    provider.resolve(request_id, "confirmed")
    agent.tick([], now=1002.0)

    assert agent.corroboration.status("skin_changes") == "asked"   # untouched
    assert agent.pop_conversation_results() == []
    diagnostics = agent.conversation_diagnostics()["classification"]
    assert diagnostics["dropped_stale"] == 1
    assert diagnostics["pending"] is False


def test_late_verdict_past_the_local_deadline_is_dropped(build_agent):
    listener = _Listener()
    provider = _ClassifyProvider()
    agent = build_agent(provider, listener)
    _ask_topic(agent)

    listener.queue.append(("hmm the birds are singing", 1000.0))
    agent.tick([], now=1000.0)
    provider.resolve(agent._pending_classification.request_id, "confirmed")

    agent.tick([], now=1009.0)          # well past submitted_at + 3.0

    assert agent.corroboration.status("skin_changes") == "asked"
    assert agent.conversation_diagnostics()["classification"][
        "dropped_deadline"] == 1


# ---------------------------------------------------------- supersession

def test_second_unclear_replaces_the_slot_and_reaps_the_first_future(build_agent):
    listener = _Listener()
    provider = _ClassifyProvider(auto="confirmed")
    agent = build_agent(provider, listener)
    _ask_topic(agent)

    # Both utterances arrive in one batch, so the second supersedes the first
    # before any poll at the top of the next tick could apply it.
    listener.queue.append(("hmm the birds are singing", 1000.0))
    listener.queue.append(("well the kettle is on", 1000.5))
    agent.tick([], now=1000.0)

    pending = agent._pending_classification
    assert pending is not None and pending.request_id == "c2"
    assert "c1" in provider.polled            # reaped, not leaked
    assert "c1" not in provider.pending
    assert agent.corroboration.status("skin_changes") == "asked"
    assert agent.conversation_diagnostics()["classification"]["superseded"] == 1


def test_keyword_confident_answer_cancels_a_pending_classification(build_agent):
    listener = _Listener()
    provider = _ClassifyProvider()
    agent = build_agent(provider, listener)
    _ask_topic(agent)

    listener.queue.append(("hmm the birds are singing", 1000.0))
    agent.tick([], now=1000.0)
    request_id = agent._pending_classification.request_id

    listener.queue.append(("no, nothing like that", 1001.0))
    agent.tick([], now=1001.0)

    assert agent._pending_classification is None
    assert request_id in provider.polled
    assert agent.corroboration.status("skin_changes") == "denied"
    assert agent.conversation_diagnostics()["classification"]["superseded"] == 1


# ------------------------------------------- bounded re-ask, action lane

def test_unavailable_provider_falls_back_to_exactly_one_bounded_re_ask(build_agent):
    listener = _Listener()
    provider = _ClassifyProvider(accept=False)     # circuit open / unavailable
    agent = build_agent(provider, listener)

    listener.queue.append(("Please check my tremor", 1000.0))
    assert "would you like" in agent.tick([], now=1000.0).lower()

    listener.queue.append(("hmm what was that", 1001.0))
    first = agent.tick([], now=1001.0)
    assert first == "Sorry — should I start the hold still? Just yes or no."
    assert agent._pending_action_unclear_rephrased is True
    assert agent._pending_classification is None
    assert not agent.elicitation.active("hold_still")

    listener.queue.append(("hmm what was that", 1002.0))
    second = agent.tick([], now=1002.0)
    assert second == "Okay, I'll leave it for now."
    assert agent._pending_action is None
    assert agent._pending_action_unclear_rephrased is False
    assert not agent.elicitation.active("hold_still")
    # Exactly once: the give-up line replaced the re-ask, it did not repeat it.
    # Both unclears were offered to the model and both were refused, so no
    # speculative verdict was ever pending.
    assert len(provider.submitted) == 2
    assert agent._pending_classification is None


# ------------------------------------------------- async steering (Goal A)

def test_steer_falls_back_this_tick_and_applies_the_async_choice_later(build_agent):
    provider = _SteerProvider()
    agent = build_agent(provider)
    agent.corroboration.observe([_hit("rash", "rash")], 100.0)
    agent.corroboration.observe([_hit("drowsiness", "perclos")], 200.0)

    signatures = [i.signature for i in agent._extra_intents(1000.0, [], False)]
    assert "ask:skin_changes:0" in signatures     # deterministic oldest, now
    assert agent._pending_topic_selection is not None
    assert [topic for topic, _q in provider.submitted[0][0]] == [
        "skin_changes", "tiredness"]

    provider.choice = "tiredness"
    agent._poll_topic_selection(1001.0)
    assert agent._pending_topic_selection is None

    signatures = [i.signature for i in agent._extra_intents(1001.0, [], False)]
    assert "ask:tiredness:0" in signatures        # the steer applies next tick
    assert "ask:skin_changes:0" not in signatures


def test_async_steer_choice_outside_the_offered_set_is_ignored(build_agent):
    provider = _SteerProvider()
    agent = build_agent(provider)
    agent.corroboration.observe([_hit("rash", "rash")], 100.0)
    agent.corroboration.observe([_hit("drowsiness", "perclos")], 200.0)
    agent._extra_intents(1000.0, [], False)

    provider.choice = "not_a_topic"
    agent._poll_topic_selection(1001.0)

    signatures = [i.signature for i in agent._extra_intents(1001.0, [], False)]
    assert "ask:skin_changes:0" in signatures     # membership check holds


# --------------------------------------------------- non-blocking proof

def test_tick_never_calls_the_network_on_the_frame_loop(monkeypatch):
    """A full tick with a flagged topic AND a pending classification must not
    make a single synchronous HTTP call. Before `_steer` was made async this
    failed: `select_topic` reached `urlopen(..., timeout=8.0)` inside tick()."""
    import agent.moondream_client as moondream_module
    from agent.voice_agent import VoiceAgent

    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: "secret")
    frame_loop = threading.current_thread()
    blocking_calls: list[str] = []

    def urlopen(*_args, **_kwargs):
        if threading.current_thread() is frame_loop:
            blocking_calls.append("frame_loop")
            raise AssertionError("tick() reached the network on the frame loop")
        raise urllib.error.URLError("offline")   # worker lanes stay contained

    monkeypatch.setattr(moondream_module.urllib.request, "urlopen", urlopen)
    listener = _Listener()
    agent = VoiceAgent(speak=False, listener=listener, moondream_enabled=True,
                       min_gap=0, small_talk_interval=1e12)
    agent.elicitation.clear()
    try:
        # One topic awaiting an answer (drives the classification lane) and a
        # second still flagged (drives the steer selector).
        agent.corroboration.observe([_hit("rash", "rash")], 1000.0)
        agent.corroboration.observe([_hit("drowsiness", "perclos")], 1000.0)
        agent.corroboration.mark_asked("skin_changes", 1000.0)
        listener.queue.append(("hmm the birds are singing", 1000.0))

        agent.tick([], now=1000.0)

        assert blocking_calls == []
        assert agent._pending_classification is not None
        assert agent.corroboration.status("tiredness") == "flagged"
    finally:
        agent.elicitation.clear()
        agent.close()
