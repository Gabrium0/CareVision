"""Typed replies (--type-input) reaching the agent's existing answer path.

Whisper invents utterances from room noise, so a showcase drives the agent by
typing instead. The safety property under test is that typing changes only the
*transport*: a TypedListener satisfies the same polling contract as
audio.stt.Listener, and /say-control is as narrow as the other control routes,
so corroboration confirm/deny logic is reached unmodified. Runs against a real
CompanionServer on an ephemeral port -- no camera, pipeline, or microphone.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from audio.typed import TypedListener
from webui.server import CompanionServer


def _post(port: int, body: dict, path: str = "/say-control"):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    def _body(raw: bytes):
        # Unknown routes answer with plain "not found", not JSON.
        try:
            return json.loads(raw.decode())
        except json.JSONDecodeError:
            return raw.decode()

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, _body(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _body(exc.read())


# ------------------------------------------------------------- TypedListener

def test_push_then_pop_matches_listener_contract():
    listener = TypedListener()
    assert listener.available is True
    assert listener.push("no, I feel fine", timestamp=100.0) is True
    assert listener.pop_utterances() == [("no, I feel fine", 100.0)]
    assert listener.pop_utterances() == []      # draining is destructive


def test_whitespace_is_normalized_and_blank_rejected():
    listener = TypedListener()
    assert listener.push("  yes   I  am  ", timestamp=1.0) is True
    assert listener.pop_utterances() == [("yes I am", 1.0)]
    for blank in ("", "   ", "\n\t", None):
        assert listener.push(blank) is False
    assert listener.pop_utterances() == []


def test_overlong_text_is_truncated():
    listener = TypedListener()
    listener.push("a" * 900, timestamp=2.0)
    assert len(listener.pop_utterances()[0][0]) == 400


def test_ordering_is_preserved():
    listener = TypedListener()
    for index, text in enumerate(("first", "second", "third")):
        listener.push(text, timestamp=float(index))
    assert [t for t, _ in listener.pop_utterances()] == ["first", "second", "third"]


def test_interface_parity_hooks_are_harmless():
    # main.py:503 indexes the result, and voice_agent calls these unconditionally.
    listener = TypedListener()
    assert listener.pop_metrics() == []
    assert listener.mark_agent_spoke(1.0) is None
    assert listener.close() is None


# -------------------------------------------------------------- /say-control

@pytest.fixture
def served():
    """Server whose say_handler feeds a real TypedListener, as main.py does."""
    listener = TypedListener()

    def handler(text):
        if not listener.push(text):
            raise ValueError("empty reply")
        return {"text": " ".join(str(text).split())[:400]}

    server = CompanionServer(port=0, say_handler=handler)
    server.start()
    try:
        yield server, server._httpd.server_address[1], listener
    finally:
        server.stop()


def test_posted_text_reaches_the_listener(served):
    _server, port, listener = served
    status, body = _post(port, {"text": "no, nothing hurts"})
    assert status == 200
    assert body["ok"] is True and body["said"]["text"] == "no, nothing hurts"
    assert [t for t, _ in listener.pop_utterances()] == ["no, nothing hurts"]


def test_empty_text_is_rejected_without_queueing(served):
    _server, port, listener = served
    status, body = _post(port, {"text": "   "})
    assert status == 400
    assert body["ok"] is False and "empty reply" in body["error"]
    assert listener.pop_utterances() == []


def test_probe_reports_disabled_when_no_handler():
    # The companion page hides its reply bar on this exact response.
    server = CompanionServer(port=0)     # e.g. a run without --type-input
    server.start()
    try:
        status, body = _post(server._httpd.server_address[1], {"text": ""})
        assert status == 400
        assert body["ok"] is False
        assert "not enabled" in body["error"]
    finally:
        server.stop()


def test_unknown_post_route_is_still_rejected(served):
    _server, port, _listener = served
    status, _body = _post(port, {"text": "hi"}, path="/say")
    assert status == 404
