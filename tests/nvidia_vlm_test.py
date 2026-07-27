"""Deterministic tests for the shared NVIDIA VLM transport."""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.error

import cv2
import numpy as np
import pytest

from integrations.nvidia_vlm import (
    NvidiaVLMClient,
    NvidiaVLMError,
    NvidiaVLMResponse,
    _COORDINATOR,
)


@pytest.fixture(autouse=True)
def _reset_nvidia_coordinator():
    _COORDINATOR.reset_for_tests()
    yield
    _COORDINATOR.reset_for_tests()


class _Response:
    def __init__(self, body, status=200, headers=None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        if isinstance(self._body, bytes):
            return self._body
        return json.dumps(self._body).encode()


def _success(content="{}", *, finish_reason="stop", request_id="request-1"):
    return _Response(
        {"choices": [{"message": {"content": content},
                      "finish_reason": finish_reason}]},
        headers={"NVCF-REQID": request_id},
    )


def _client(**kwargs):
    return NvidiaVLMClient(
        "private-key",
        "https://integrate.api.nvidia.com/v1/chat/completions",
        "meta/llama-3.2-11b-vision-instruct",
        **kwargs,
    )


def test_request_returns_structured_sanitized_metadata(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return _success('{"ok":true}')

    monkeypatch.setattr("integrations.nvidia_vlm._open_no_redirect", fake_urlopen)
    result = _client().request("private prompt", [b"jpeg"], purpose="manual_arm_check")

    assert isinstance(result, NvidiaVLMResponse)
    assert result.content == '{"ok":true}'
    assert result.status == 200
    assert result.finish_reason == "stop"
    assert result.request_id == "request-1"
    assert result.poll_count == 0
    assert result.encoded_image_bytes == 4
    assert result.latency_ms >= 0
    assert len(captured["payload"]["messages"][0]["content"]) == 2
    diagnostic = json.dumps(_client().coordinator_diagnostics())
    assert "private prompt" not in diagnostic
    assert "private-key" not in diagnostic
    assert "jpeg" not in diagnostic


def test_encode_uses_fixed_adaptive_ladder_and_byte_budget():
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, (900, 1200, 3), dtype=np.uint8)
    client = _client(max_inline_image_bytes=24_000)

    encoded = client.encode(frame)
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)

    assert 0 < len(encoded) <= 24_000
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 768


def test_request_reencodes_one_oversized_jpeg_and_rejects_multiple(monkeypatch):
    rng = np.random.default_rng(9)
    frame = rng.integers(0, 256, (1000, 1000, 3), dtype=np.uint8)
    ok, original = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok and len(original) > 16_000
    captured = {}

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode())
        url = payload["messages"][0]["content"][1]["image_url"]["url"]
        captured["image"] = base64.b64decode(url.split(",", 1)[1])
        return _success()

    monkeypatch.setattr("integrations.nvidia_vlm._open_no_redirect", fake_urlopen)
    client = _client(max_inline_image_bytes=16_000)
    result = client.request("prompt", [original.tobytes()])
    assert len(captured["image"]) <= 16_000
    assert result.encoded_image_bytes == len(captured["image"])

    with pytest.raises(NvidiaVLMError) as error:
        client.request("prompt", [b"one", b"two"])
    assert error.value.kind == "payload_size"
    assert error.value.retryable is False


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (400, "client", False),
        (401, "authentication", False),
        (403, "authentication", False),
        (413, "payload_size", True),
        (429, "rate_limit", True),
        (503, "server", True),
    ],
)
def test_http_failures_are_classified(monkeypatch, status, kind, retryable):
    envelope = json.dumps({
        "error": {
            "message": "safe message data:image/jpeg;base64,private-media private-key"
        }
    }).encode()
    exc = urllib.error.HTTPError(
        "https://integrate.api.nvidia.com", status, "bad",
        {"Retry-After": "99", "NVCF-REQID": "request-safe"},
        io.BytesIO(envelope),
    )
    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(exc),
    )

    with pytest.raises(NvidiaVLMError) as error:
        _client().request("private prompt", [b"private-media"])
    failure = error.value
    assert failure.status == status
    assert failure.kind == kind
    assert failure.retryable is retryable
    assert failure.retry_after == 10.0
    assert failure.request_id == "request-safe"
    assert "private-key" not in str(failure)
    assert "private-media" not in str(failure)


def test_timeout_and_network_failures_are_distinct(monkeypatch):
    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )
    with pytest.raises(NvidiaVLMError) as timeout:
        _client().request("prompt", [b"jpeg"])
    assert timeout.value.kind == "timeout"
    assert timeout.value.retryable

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("dns")),
    )
    with pytest.raises(NvidiaVLMError) as network:
        _client().request("prompt", [b"jpeg"])
    assert network.value.kind == "network"
    assert network.value.retryable


def test_202_polls_only_provider_supplied_nvidia_https_url(monkeypatch):
    calls = []
    responses = iter([
        _Response(
            {"statusUrl": "https://api.nvcf.nvidia.com/v2/status/abc"},
            status=202,
            headers={"Retry-After": "0"},
        ),
        _success("done", request_id="polled-request"),
    ])

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.get_method()))
        return next(responses)

    monkeypatch.setattr("integrations.nvidia_vlm._open_no_redirect", fake_urlopen)
    result = _client().request("prompt", [b"jpeg"])

    assert calls == [
        ("https://integrate.api.nvidia.com/v1/chat/completions", "POST"),
        ("https://api.nvcf.nvidia.com/v2/status/abc", "GET"),
    ]
    assert result.content == "done"
    assert result.poll_count == 1
    assert result.request_id == "polled-request"


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        "http://api.nvcf.nvidia.com/status/abc",
        "https://nvidia.com.evil.invalid/status/abc",
        "/v2/status/abc",
    ],
)
def test_202_without_trusted_provider_url_is_retryable_pending(monkeypatch, candidate):
    body = {} if candidate is None else {"statusUrl": candidate}
    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect",
        lambda *_args, **_kwargs: _Response(body, status=202),
    )
    with pytest.raises(NvidiaVLMError) as error:
        _client().request("prompt", [b"jpeg"])
    assert error.value.kind == "pending"
    assert error.value.status == 202
    assert error.value.retryable


@pytest.mark.parametrize(
    ("body", "kind"),
    [
        ({}, "schema_validation"),
        ({"choices": []}, "schema_validation"),
        ({"choices": [{"message": {}}]}, "schema_validation"),
        ({"choices": [{"message": {"content": ""}}]}, "empty_content"),
        (b"not-json", "json_parse"),
    ],
)
def test_response_shape_and_empty_content_are_classified(monkeypatch, body, kind):
    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect",
        lambda *_args, **_kwargs: _Response(body),
    )
    with pytest.raises(NvidiaVLMError) as error:
        _client().request("prompt", [b"jpeg"])
    assert error.value.kind == kind
    assert error.value.retryable


def test_process_wide_coordinator_prioritizes_manual_then_guided(monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    order = []
    lock = threading.Lock()

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode())
        prompt = payload["messages"][0]["content"][0]["text"]
        with lock:
            order.append(prompt)
        if prompt == "first-passive":
            first_started.set()
            assert release_first.wait(2)
        return _success(prompt)

    monkeypatch.setattr("integrations.nvidia_vlm._open_no_redirect", fake_urlopen)
    client_a = _client()
    client_b = _client()
    results = {}

    def invoke(name, client, purpose):
        results[name] = client.request(name, [b"jpeg"], purpose=purpose).content

    first = threading.Thread(
        target=invoke, args=("first-passive", client_a, "passive"))
    queued_passive = threading.Thread(
        target=invoke, args=("queued-passive", client_b, "passive"))
    guided = threading.Thread(
        target=invoke, args=("guided", client_a, "guided_closeup"))
    manual = threading.Thread(
        target=invoke, args=("manual", client_b, "manual_arm_check"))
    first.start()
    assert first_started.wait(1)
    queued_passive.start()
    guided.start()
    manual.start()
    for expected in (3,):
        deadline = time.monotonic() + 1
        while client_a.coordinator_diagnostics()["queued"] < expected:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    release_first.set()
    for thread in (first, queued_passive, guided, manual):
        thread.join(2)
        assert not thread.is_alive()

    assert order == ["first-passive", "manual", "guided", "queued-passive"]
    assert results == {
        "first-passive": "first-passive",
        "queued-passive": "queued-passive",
        "guided": "guided",
        "manual": "manual",
    }
    assert client_a.coordinator_diagnostics()["queued"] == 0
    assert client_a.coordinator_diagnostics()["active"] is None
