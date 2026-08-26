"""The phrasing and classify lanes run concurrently but share one breaker."""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.moondream_client as moondream_module
from agent.moondream_client import MoondreamClient

_DEADLINE = 5.0


class _Response:
    def __init__(self, text: str):
        self.body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class _Gate:
    """A urlopen stand-in that parks every request until the test releases it."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = {"generation": threading.Event(),
                        "classification": threading.Event()}
        self.lanes: list[str] = []
        self._lock = threading.Lock()

    def urlopen(self, request, timeout):
        """Record which lane called, then block until the test releases it."""
        prompt = json.loads(request.data)["messages"][0]["content"]
        if "exactly one word" in prompt:
            lane, reply = "classification", "confirmed"
        elif "topic id" in prompt:
            lane, reply = "classification", "tiredness"
        else:
            lane, reply = "generation", "A short friendly line."
        with self._lock:
            self.lanes.append(lane)
        self.entered[lane].set()
        if not self.release.wait(_DEADLINE):
            raise TimeoutError("test gate never released")
        return _Response(reply)

    def wait_entered(self, lane: str) -> bool:
        """Block until a request of `lane` has actually reached the socket."""
        return self.entered[lane].wait(_DEADLINE)


def _gated_client(monkeypatch, enabled=True):
    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: "secret-key")
    client = MoondreamClient(enabled=enabled)
    gate = _Gate()
    monkeypatch.setattr(moondream_module.urllib.request, "urlopen", gate.urlopen)
    return client, gate


def _drain(poll, request_id):
    deadline = time.time() + _DEADLINE
    done, value = False, None
    while not done and time.time() < deadline:
        done, value = poll(request_id)
    return done, value


def test_classification_starts_while_a_phrasing_request_is_still_open(monkeypatch):
    client, gate = _gated_client(monkeypatch)
    try:
        generation_id = client.submit_generation("greet", "context")
        assert generation_id is not None
        assert gate.wait_entered("generation")

        classify_id = client.submit_classification("Are you okay?", "yes")
        assert classify_id is not None
        assert classify_id != generation_id
        assert gate.wait_entered("classification")

        status = client.status()
        assert status["in_flight"] is True
        assert status["classify_in_flight"] is True
        assert sorted(gate.lanes) == ["classification", "generation"]

        # Each lane still enforces exactly one flight of its own.
        assert client.submit_generation("second", "context") is None
        assert client.submit_classification("Are you okay?", "no") is None
        assert client.submit_topic_selection([("tiredness", "Tired?")], "ctx") is None

        gate.release.set()
        assert _drain(client.poll_generation, generation_id) == (
            True, "A short friendly line.")
        assert _drain(client.poll_classification, classify_id) == (True, "confirmed")
        assert client.status()["classify_in_flight"] is False
    finally:
        gate.release.set()
        client.close()


def test_phrasing_starts_while_a_classification_is_still_open(monkeypatch):
    client, gate = _gated_client(monkeypatch)
    try:
        classify_id = client.submit_topic_selection(
            [("tiredness", "Have you been tired?")], "recent context")
        assert classify_id is not None
        assert gate.wait_entered("classification")

        generation_id = client.submit_generation("greet", "context")
        assert generation_id is not None
        assert gate.wait_entered("generation")

        gate.release.set()
        assert _drain(client.poll_topic_selection, classify_id) == (True, "tiredness")
        assert _drain(client.poll_generation, generation_id) == (
            True, "A short friendly line.")
        assert client.status()["classification_requests"] == 1
        assert client.status()["generation_requests"] == 1
    finally:
        gate.release.set()
        client.close()


def test_authorization_failure_on_the_classify_lane_stops_both_lanes(monkeypatch):
    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: "secret-key")
    client = MoondreamClient(enabled=True)
    calls = []

    def unauthorized(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(moondream_module.urllib.request, "urlopen", unauthorized)
    try:
        classify_id = client.submit_classification("Are you okay?", "yes")
        assert classify_id is not None
        assert _drain(client.poll_classification, classify_id) == (True, None)

        status = client.status()
        assert status["authorization_failed"] is True
        assert status["circuit_state"] == "authorization_failed"
        # One credential, one latch: neither lane may spend another request.
        assert client.submit_classification("Are you okay?", "yes") is None
        assert client.submit_topic_selection([("tiredness", "Tired?")], "ctx") is None
        assert client.submit_generation("greet", "context") is None
        assert client.submit_response([{"role": "user", "content": "hi"}], []) is None
        assert len(calls) == 1
    finally:
        client.close()


def test_open_circuit_or_disabled_client_refuses_classification(monkeypatch):
    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: "secret-key")
    disabled = MoondreamClient(enabled=False)
    try:
        assert disabled.submit_classification("Are you okay?", "yes") is None
        assert disabled.submit_topic_selection([("tiredness", "Tired?")], "ctx") is None
        assert disabled.status()["classify_in_flight"] is False
    finally:
        disabled.close()

    client = MoondreamClient(enabled=True)
    try:
        with client._lock:
            client._circuit_open_until = time.time() + 60.0
        assert client.status()["circuit_state"] == "open"
        assert client.submit_classification("Are you okay?", "yes") is None
        assert client.submit_topic_selection([("tiredness", "Tired?")], "ctx") is None
        assert client.submit_generation("greet", "context") is None
    finally:
        client.close()


def test_close_clears_both_registries_and_shuts_both_executors(monkeypatch):
    client, gate = _gated_client(monkeypatch)
    try:
        generation_id = client.submit_generation("greet", "context")
        classify_id = client.submit_classification("Are you okay?", "yes")
        assert generation_id and classify_id
        assert gate.wait_entered("generation")
        assert gate.wait_entered("classification")

        gate.release.set()
        client.close()

        assert client._async == {}
        assert client._async_classify == {}
        assert client.status()["in_flight"] is False
        assert client.status()["classify_in_flight"] is False
        assert client.poll_generation(generation_id) == (True, None)
        assert client.poll_classification(classify_id) == (True, None)
        assert client.submit_generation("greet", "context") is None
        assert client.submit_classification("Are you okay?", "yes") is None
        with pytest.raises(RuntimeError):
            client._executor.submit(lambda: None)
        with pytest.raises(RuntimeError):
            client._classify_executor.submit(lambda: None)
    finally:
        gate.release.set()
        client.close()
