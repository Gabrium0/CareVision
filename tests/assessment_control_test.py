"""End-to-end wiring test for the /assessment-control endpoint that backs the
big-screen /demo "try this" picker (webui/demo.html).

The picker POSTs {action, protocol}; the server must forward that to the
injected assessment_handler (in production, main.py's closure over
voice_agent.request_test / start_demo_circuit) and reject anything malformed
or unavailable with HTTP 400 -- the same narrow contract as /replay-control.
Runs against a real CompanionServer on an ephemeral port so no camera or
pipeline is needed.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from webui.server import CompanionServer


def _post(port: int, body: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/assessment-control",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


@pytest.fixture
def served():
    """Start a server on an OS-assigned port; yield (server, port, calls)."""
    calls = []

    def handler(action, protocol=None):
        calls.append((action, protocol))
        if action == "start" and protocol not in ("balance", "arm_drift"):
            raise ValueError("unknown protocol")
        return {"action": action, "protocol": protocol, "started": True}

    server = CompanionServer(port=0, assessment_handler=handler)
    server.start()
    port = server._httpd.server_address[1]
    try:
        yield server, port, calls
    finally:
        server.stop()


def test_start_forwards_protocol_to_handler(served):
    _server, port, calls = served
    status, body = _post(port, {"action": "start", "protocol": "balance"})
    assert status == 200
    assert body["ok"] is True
    assert body["assessment"]["protocol"] == "balance"
    assert calls == [("start", "balance")]


def test_circuit_forwards_with_no_protocol(served):
    _server, port, calls = served
    status, body = _post(port, {"action": "circuit"})
    assert status == 200
    assert body["ok"] is True
    assert calls == [("circuit", None)]


def test_unknown_action_is_rejected_before_the_handler(served):
    _server, port, calls = served
    status, body = _post(port, {"action": "explode"})
    assert status == 400
    assert body["ok"] is False
    assert calls == []  # validated at the endpoint, handler never invoked


def test_handler_valueerror_becomes_400(served):
    _server, port, calls = served
    status, body = _post(port, {"action": "start", "protocol": "not_a_protocol"})
    assert status == 400
    assert body["ok"] is False
    assert calls == [("start", "not_a_protocol")]


def test_endpoint_unavailable_when_no_handler_wired():
    server = CompanionServer(port=0)  # no assessment_handler (e.g. --webui off path)
    server.start()
    port = server._httpd.server_address[1]
    try:
        status, body = _post(port, {"action": "circuit"})
        assert status == 400
        assert body["ok"] is False
    finally:
        server.stop()
