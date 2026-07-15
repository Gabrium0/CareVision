"""Runtime Gemini controls must prevent accidental API token use."""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.gemini_client as gemini_module
from agent.gemini_client import GeminiClient
from agent.voice_agent import VoiceAgent


class _Models:
    def __init__(self):
        self.calls: list[str] = []

    def generate_content(self, *, model, contents):
        self.calls.append(contents)
        text = "confirmed" if "CONFIRM" in contents else "A short friendly line."
        return SimpleNamespace(text=text)


def _client(monkeypatch, enabled=False):
    monkeypatch.setattr(gemini_module, "gemini_api_key", lambda: None)
    client = GeminiClient(enabled=enabled)
    models = _Models()
    client._client = SimpleNamespace(models=models)
    client.available = True
    return client, models


def test_disabled_client_makes_zero_generation_or_classification_calls(monkeypatch):
    client, models = _client(monkeypatch, enabled=False)

    assert client.generate("greet", "context") is None
    assert client.classify_answer("Are you okay?", "yes") is None
    assert models.calls == []
    assert client.status()["generation_requests"] == 0
    assert client.status()["classification_requests"] == 0


def test_toggle_enables_both_call_types_then_blocks_future_calls(monkeypatch):
    client, models = _client(monkeypatch, enabled=False)
    assert client.available is True
    assert client.status()["active"] is False

    assert client.toggle_enabled() is True
    assert client.generate("greet", "context") == "A short friendly line."
    assert client.classify_answer("Are you okay?", "yes") == "confirmed"
    assert len(models.calls) == 2
    assert client.status()["generation_requests"] == 1
    assert client.status()["classification_requests"] == 1

    assert client.toggle_enabled() is False
    assert client.generate("greet", "context") is None
    assert client.classify_answer("Are you okay?", "yes") is None
    assert len(models.calls) == 2


def test_status_and_toggle_are_safe_from_multiple_threads(monkeypatch):
    client, _models = _client(monkeypatch, enabled=False)

    def exercise(index):
        if index % 2:
            client.toggle_enabled()
        else:
            client.status()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(exercise, range(200)))
    status = client.status()
    assert status["available"] is True
    assert isinstance(status["enabled"], bool)
    assert status["active"] == status["enabled"]


def test_voice_agent_keeps_templated_speech_when_gemini_disabled(monkeypatch):
    monkeypatch.setattr(gemini_module, "gemini_api_key", lambda: None)
    agent = VoiceAgent(speak=False, gemini_enabled=False, min_gap=0)
    try:
        text = agent.tick([], now=1000.0)
        assert text
        assert agent.gemini_status()["generation_requests"] == 0
        assert agent.gemini_status()["classification_requests"] == 0
    finally:
        agent.close()
