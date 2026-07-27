"""Security and process-wide reliability tests for the NVIDIA VLM transport."""
from __future__ import annotations

import io
import json
import threading
import time
import urllib.error

import pytest

from integrations.nvidia_vlm import (
    NvidiaVLMClient,
    NvidiaVLMError,
    _COORDINATOR,
)


@pytest.fixture(autouse=True)
def _isolated_coordinator():
    _COORDINATOR.reset_for_tests()
    yield
    _COORDINATOR.reset_for_tests()


class _Response:
    def __init__(self, body, *, status=200, headers=None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=None):
        if isinstance(self._body, bytes):
            return self._body
        return json.dumps(self._body).encode("utf-8")


def _success(content="{}", *, finish_reason="stop"):
    return _Response({
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }],
    })


def _client(
    *,
    api_key="shared-private-key",
    endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
):
    return NvidiaVLMClient(
        api_key,
        endpoint,
        "meta/llama-3.2-11b-vision-instruct",
    )


def _authorization(request):
    return dict(request.header_items()).get("Authorization")


def test_empty_content_error_retains_length_finish_reason(monkeypatch):
    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect",
        lambda *_args, **_kwargs: _success("", finish_reason="length"),
    )

    with pytest.raises(NvidiaVLMError) as failure:
        _client().request("prompt", [b"jpeg"], purpose="manual_arm_check")

    assert failure.value.kind == "empty_content"
    assert failure.value.finish_reason == "length"


def test_same_origin_https_custom_status_url_is_trusted(monkeypatch):
    endpoint = "https://vlm.example.test/v1/chat/completions"
    status_url = "https://vlm.example.test/jobs/request-1"
    calls = []
    responses = iter([
        _Response(
            {"statusUrl": status_url},
            status=202,
            headers={"Retry-After": "0"},
        ),
        _success('{"ok":true}'),
    ])

    def fake_open(request, timeout):
        calls.append((
            request.full_url,
            request.get_method(),
            _authorization(request),
        ))
        return next(responses)

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)

    result = _client(endpoint=endpoint).request("prompt", [b"jpeg"])

    assert result.content == '{"ok":true}'
    assert result.poll_count == 1
    assert calls == [
        (endpoint, "POST", "Bearer shared-private-key"),
        (status_url, "GET", "Bearer shared-private-key"),
    ]


def test_custom_endpoint_cannot_forward_credential_to_nvidia(monkeypatch):
    endpoint = "https://vlm.example.test/v1/chat/completions"
    cross_origin = "https://api.nvcf.nvidia.com/v2/status/request-1"
    calls = []

    def fake_open(request, timeout):
        calls.append((request.full_url, _authorization(request)))
        return _Response({"statusUrl": cross_origin}, status=202)

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)

    with pytest.raises(NvidiaVLMError) as failure:
        _client(endpoint=endpoint).request("prompt", [b"jpeg"])

    assert failure.value.kind == "pending"
    assert calls == [(endpoint, "Bearer shared-private-key")]


def test_untrusted_poll_redirect_is_never_opened_or_authorized(monkeypatch):
    endpoint = "https://vlm.example.test/v1/chat/completions"
    trusted_status_url = "https://vlm.example.test/jobs/request-1"
    untrusted_url = "https://attacker.example/collect"
    calls = []

    def fake_open(request, timeout):
        calls.append((
            request.full_url,
            request.get_method(),
            _authorization(request),
        ))
        if request.full_url == endpoint:
            return _Response(
                {"statusUrl": trusted_status_url},
                status=202,
                headers={"Retry-After": "0"},
            )
        raise urllib.error.HTTPError(
            trusted_status_url,
            302,
            "redirect",
            {"Location": untrusted_url},
            io.BytesIO(b"{}"),
        )

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)

    with pytest.raises(NvidiaVLMError) as failure:
        _client(endpoint=endpoint).request("prompt", [b"jpeg"])

    assert failure.value.kind == "client"
    assert [url for url, _, _ in calls] == [endpoint, trusted_status_url]
    assert all(url != untrusted_url for url, _, _ in calls)
    assert all(
        authorization == "Bearer shared-private-key"
        for _, _, authorization in calls
    )


@pytest.mark.parametrize("failure_kind", ["http", "network"])
def test_raw_transport_exceptions_are_not_retained(monkeypatch, failure_kind):
    if failure_kind == "http":
        raw_failure = urllib.error.HTTPError(
            "https://integrate.api.nvidia.com",
            503,
            "provider-body",
            {},
            io.BytesIO(b"{}"),
        )
    else:
        raw_failure = urllib.error.URLError("private-network-detail")

    def fail_open(*_args, **_kwargs):
        raise raw_failure

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fail_open)

    with pytest.raises(NvidiaVLMError) as failure:
        _client().request("prompt", [b"jpeg"])

    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None


def test_manual_reservation_suppresses_passive_before_manual_ticket(
        monkeypatch):
    calls = []
    passive_finished = threading.Event()
    passive_result = {}
    deadline = time.monotonic() + 2.0
    token = NvidiaVLMClient.reserve_request("manual_arm_check", deadline)

    def fake_open(request, timeout):
        prompt = json.loads(
            request.data.decode("utf-8")
        )["messages"][0]["content"][0]["text"]
        calls.append(prompt)
        return _success(prompt)

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)

    def run_passive():
        try:
            passive_result["content"] = _client().request(
                "passive",
                [b"jpeg"],
                purpose="passive",
                deadline=deadline,
            ).content
        except BaseException as exc:  # surfaced by the main test thread
            passive_result["error"] = exc
        finally:
            passive_finished.set()

    passive = threading.Thread(target=run_passive)
    passive.start()
    try:
        wait_until = time.monotonic() + 1.0
        while _client().coordinator_diagnostics()["queued"] != 1:
            assert time.monotonic() < wait_until
            time.sleep(0.002)

        assert calls == []
        assert not passive_finished.is_set()

        manual = _client().request(
            "manual",
            [b"jpeg"],
            purpose="manual_arm_check",
            deadline=deadline,
            reservation_token=token,
        )
        assert manual.content == "manual"
        assert calls == ["manual"]
    finally:
        NvidiaVLMClient.cancel_reservation(token)
        passive.join(1.0)

    assert not passive.is_alive()
    assert passive_result == {"content": "passive"}
    assert calls == ["manual", "passive"]


def test_process_wide_retry_after_blocks_manual_request(monkeypatch):
    calls = []

    def fake_open(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "0.2"},
                io.BytesIO(b"{}"),
            )
        return _success()

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)

    with pytest.raises(NvidiaVLMError) as initial:
        _client().request("passive", [b"jpeg"], purpose="passive")
    assert initial.value.kind == "rate_limit"
    assert initial.value.retry_after == pytest.approx(0.2)

    with pytest.raises(NvidiaVLMError) as blocked:
        _client().request(
            "manual",
            [b"jpeg"],
            purpose="manual_arm_check",
            deadline=time.monotonic() + 0.03,
        )

    assert blocked.value.kind == "timeout"
    assert calls == [
        "https://integrate.api.nvidia.com/v1/chat/completions",
    ]


def test_auth_failure_blocks_same_credential_across_clients(monkeypatch):
    calls = []

    def fake_open(request, timeout):
        calls.append(_authorization(request))
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "unauthorized",
                {},
                io.BytesIO(b"{}"),
            )
        return _success()

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)
    first = _client()
    second = _client()

    with pytest.raises(NvidiaVLMError) as initial:
        first.request("first", [b"jpeg"], purpose="passive")
    with pytest.raises(NvidiaVLMError) as blocked:
        second.request("second", [b"jpeg"], purpose="manual_arm_check")

    assert initial.value.kind == "authentication"
    assert blocked.value.kind == "authentication"
    assert calls == ["Bearer shared-private-key"]

    different_credential = _client(api_key="different-private-key")
    assert different_credential.request(
        "third", [b"jpeg"], purpose="manual_arm_check"
    ).content == "{}"
    assert calls == [
        "Bearer shared-private-key",
        "Bearer different-private-key",
    ]


def test_cancelled_queued_request_never_sends_media(monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    cancel_second = threading.Event()
    calls = []
    outcomes = {}

    def fake_open(request, timeout):
        prompt = json.loads(
            request.data.decode("utf-8")
        )["messages"][0]["content"][0]["text"]
        calls.append(prompt)
        if prompt == "first":
            first_started.set()
            assert release_first.wait(2)
        return _success(prompt)

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)

    def invoke(name, *, cancel_event=None):
        try:
            outcomes[name] = _client().request(
                name,
                [b"jpeg"],
                purpose="passive",
                deadline=time.monotonic() + 2.0,
                cancel_event=cancel_event,
            ).content
        except BaseException as exc:
            outcomes[name] = getattr(exc, "kind", type(exc).__name__)

    first = threading.Thread(target=invoke, args=("first",))
    second = threading.Thread(
        target=invoke, args=("second",),
        kwargs={"cancel_event": cancel_second})
    first.start()
    assert first_started.wait(1)
    second.start()
    deadline = time.monotonic() + 1
    while _client().coordinator_diagnostics()["queued"] != 1:
        assert time.monotonic() < deadline
        time.sleep(0.002)

    cancel_second.set()
    NvidiaVLMClient.notify_cancellation()
    second.join(1)
    release_first.set()
    first.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert calls == ["first"]
    assert outcomes == {"second": "client", "first": "first"}
    diagnostics = _client().coordinator_diagnostics()
    assert diagnostics["queued"] == 0
    assert diagnostics["active"] is None


def test_cancelled_pending_poll_keeps_process_wide_retry_after(monkeypatch):
    cancel_passive = threading.Event()
    pending_seen = threading.Event()
    calls = []
    outcomes = {}

    def fake_open(request, timeout):
        calls.append((request.full_url, time.monotonic()))
        if len(calls) == 1:
            pending_seen.set()
            return _Response(
                {"statusUrl": "https://api.nvcf.nvidia.com/v2/status/abc"},
                status=202,
                headers={"Retry-After": "0.25"},
            )
        return _success("manual")

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)

    def run_passive():
        try:
            _client().request(
                "passive",
                [b"jpeg"],
                purpose="passive",
                cancel_event=cancel_passive,
            )
        except BaseException as exc:
            outcomes["passive"] = getattr(
                exc, "kind", type(exc).__name__)

    passive = threading.Thread(target=run_passive)
    passive.start()
    assert pending_seen.wait(1)
    deadline = time.monotonic() + 1
    while _client().coordinator_diagnostics()[
            "retry_after_seconds"] <= 0:
        assert time.monotonic() < deadline
        time.sleep(0.002)

    cancel_passive.set()
    NvidiaVLMClient.notify_cancellation()
    manual_started = time.monotonic()
    manual = _client().request(
        "manual",
        [b"jpeg"],
        purpose="manual_arm_check",
        deadline=manual_started + 1.0,
    )
    elapsed = time.monotonic() - manual_started
    passive.join(1)

    assert not passive.is_alive()
    assert outcomes == {"passive": "client"}
    assert manual.content == "manual"
    assert len(calls) == 2
    assert elapsed >= 0.18
