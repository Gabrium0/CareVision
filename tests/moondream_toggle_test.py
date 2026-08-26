"""Runtime Moondream controls and authentication must remain credential-safe."""
from __future__ import annotations

import json
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.moondream_client as moondream_module
from agent.moondream_client import MoondreamClient
from agent.voice_agent import VoiceAgent


class _Response:
    def __init__(self, text: str):
        self.body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def _client(monkeypatch, enabled=False):
    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: "secret-key")
    client = MoondreamClient(enabled=enabled)
    calls = []

    def urlopen(request, timeout):
        calls.append((request, timeout))
        prompt = json.loads(request.data)["messages"][0]["content"]
        return _Response("confirmed" if "exactly one word" in prompt else
                         "A short friendly line.")

    monkeypatch.setattr(moondream_module.urllib.request, "urlopen", urlopen)
    return client, calls


def test_disabled_client_makes_zero_generation_or_classification_calls(monkeypatch):
    client, calls = _client(monkeypatch, enabled=False)

    assert client.generate("greet", "context") is None
    assert client.classify_answer("Are you okay?", "yes") is None
    assert calls == []
    assert client.status()["generation_requests"] == 0
    assert client.status()["classification_requests"] == 0


def test_toggle_enables_calls_and_uses_documented_moondream_header(monkeypatch):
    client, calls = _client(monkeypatch, enabled=False)
    assert client.status()["status"] == "configured"
    assert client.toggle_enabled() is True
    assert client.generate("greet", "context") == "A short friendly line."
    assert client.status()["status"] == "ready"
    assert client.classify_answer("Are you okay?", "yes") == "confirmed"
    assert len(calls) == 2
    assert calls[0][0].get_header("X-moondream-auth") == "secret-key"
    assert calls[0][0].get_header("Authorization") is None
    assert calls[0][0].get_header("User-agent") == "Humanoid-Care-Agent/1.0"
    assert calls[0][0].full_url == "https://api.moondream.ai/v1/chat/completions"
    assert client.status()["provider"] == "moondream"

    assert client.toggle_enabled() is False
    assert client.generate("greet", "context") is None
    assert len(calls) == 2
    client.close()


def test_async_generation_is_one_flight_and_nonblocking(monkeypatch):
    client, calls = _client(monkeypatch, enabled=True)
    request_id = client.submit_generation("greet", "context")
    assert request_id is not None
    assert client.submit_generation("second", "context") is None
    deadline = __import__("time").time() + 2.0
    done, text = False, None
    while not done and __import__("time").time() < deadline:
        done, text = client.poll_generation(request_id)
    assert done and text == "A short friendly line."
    assert len(calls) == 1
    client.close()


def test_status_and_toggle_are_safe_from_multiple_threads(monkeypatch):
    client, _calls = _client(monkeypatch, enabled=False)

    def exercise(index):
        return client.toggle_enabled() if index % 2 else client.status()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(exercise, range(200)))
    status = client.status()
    assert status["available"] is True
    assert isinstance(status["enabled"], bool)
    assert status["active"] == status["enabled"]
    assert "secret-key" not in repr(status)


def test_voice_agent_keeps_templated_speech_when_moondream_disabled(monkeypatch):
    monkeypatch.setattr(moondream_module, "moondream_api_key", lambda: None)
    agent = VoiceAgent(speak=False, moondream_enabled=False, min_gap=0)
    try:
        text = agent.tick([], now=1000.0)
        assert text
        assert agent.moondream_status()["generation_requests"] == 0
        assert agent.moondream_status()["classification_requests"] == 0
    finally:
        agent.close()


def test_authorization_failure_latches_until_explicit_toggle(monkeypatch):
    client, calls = _client(monkeypatch, enabled=True)
    def forbidden(request, timeout):
        calls.append((request, timeout))
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)
    monkeypatch.setattr(moondream_module.urllib.request, "urlopen", forbidden)

    assert client.generate("greet", "context") is None
    assert client.generate("greet", "context") is None
    status = client.status()
    assert len(calls) == 1
    assert status["authorization_failed"] is True
    assert status["retryable"] is False
    assert status["circuit_state"] == "authorization_failed"
    assert status["status"] == "authorization_failed"

    assert client.toggle_enabled() is False
    assert client.toggle_enabled() is True
    assert client.status()["authorization_failed"] is False
    assert client.status()["status"] == "configured"
    client.close()
