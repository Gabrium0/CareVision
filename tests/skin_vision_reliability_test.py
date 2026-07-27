"""Deterministic reliability coverage for manual NVIDIA skin checks."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest

from core.context import FrameContext
from integrations.nvidia_vlm import (
    NvidiaVLMError,
    NvidiaVLMResponse,
)
from modules.skin_vision import (
    SkinAnalysis,
    SkinVision,
    SkinVisionAPIError,
)


def _raw_analysis() -> dict:
    return {
        "image_quality": "good",
        "visual_source": "live_skin",
        "sufficient_skin_visible": True,
        "finding_present": False,
        "visible_features": [],
        "body_region": "left forearm",
        "confidence": 0.72,
        "possible_conditions": [],
        "follow_up_topics": [],
    }


def _response(content, *, finish_reason="stop", status=200,
              encoded_bytes=4321, request_id="request-safe"):
    return NvidiaVLMResponse(
        content=content,
        status=status,
        finish_reason=finish_reason,
        request_id=request_id,
        latency_ms=12.5,
        poll_count=0,
        encoded_image_bytes=encoded_bytes,
        queue_ms=3.0,
    )


def _module(monkeypatch, **params) -> SkinVision:
    monkeypatch.setattr("modules.skin_vision.nvidia_api_key", lambda: "secret-key")
    return SkinVision(
        consent=True,
        manual_retry_deadline=60.0,
        manual_attempt_timeout=20.0,
        **params,
    )


def _no_retry_wait(monkeypatch) -> None:
    monkeypatch.setattr("modules.skin_vision.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "modules.skin_vision.random.uniform", lambda _start, _end: 0.0)


def test_timeout_then_freeform_retry_recovers_and_records_manual_metrics(
        monkeypatch):
    module = _module(monkeypatch)
    _no_retry_wait(monkeypatch)
    calls = []

    def request(prompt, images, **kwargs):
        calls.append((prompt, images, kwargs))
        if len(calls) == 1:
            raise NvidiaVLMError(
                "provider request timed out", retryable=True, kind="timeout")
        return _response(json.dumps(_raw_analysis()))

    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"adaptive-jpeg")
    monkeypatch.setattr(module._client, "request", request)
    try:
        analysis = module._call_api(
            np.zeros((32, 32, 3), np.uint8), "closeup", None,
            purpose="manual_arm_check")
        assert analysis.body_region == "left forearm"
        assert len(calls) == 2
        assert calls[0][2]["response_format"]["type"] == "json_schema"
        assert "strict" not in calls[0][2]["response_format"]["json_schema"]
        assert calls[1][2]["response_format"] is None
        assert "JSON contract:" not in calls[0][0]
        assert "JSON contract:" in calls[1][0]
        assert calls[0][2]["max_tokens"] == 450
        assert calls[1][2]["max_tokens"] == 450

        diagnostics = module.diagnostics()
        attempts = diagnostics["last_attempt"]["attempts"]
        assert [attempt["attempt"] for attempt in attempts] == [1, 2]
        assert [attempt["failure_category"] for attempt in attempts] == [
            "timeout", None]
        assert [attempt["structured"] for attempt in attempts] == [True, False]
        assert diagnostics["manual_metrics"]["completed"] == 1
        assert diagnostics["manual_metrics"]["retried"] == 1
        assert diagnostics["manual_metrics"]["retry_recovery"] == 1
        assert diagnostics["manual_metrics"]["first_attempt_success"] == 0
    finally:
        module.close()


def test_timeout_then_empty_retry_then_repaired_success(monkeypatch):
    module = _module(monkeypatch)
    _no_retry_wait(monkeypatch)
    calls = []

    def request(_prompt, _images, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise NvidiaVLMError(
                "provider request timed out", retryable=True, kind="timeout")
        if len(calls) == 2:
            return _response(None)
        return _response(json.dumps(_raw_analysis()))

    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(module._client, "request", request)
    try:
        module._call_api(
            np.zeros((32, 32, 3), np.uint8), "closeup", None,
            purpose="manual_arm_check")
        assert len(calls) == 3
        assert calls[0]["response_format"] is not None
        assert calls[1]["response_format"] is None
        assert calls[2]["response_format"] is None
        attempts = module.diagnostics()["last_attempt"]["attempts"]
        assert [attempt["failure_category"] for attempt in attempts] == [
            "timeout", "empty_content", None]
    finally:
        module.close()


def test_finish_reason_length_increases_only_retry_output_budget(monkeypatch):
    module = _module(monkeypatch)
    _no_retry_wait(monkeypatch)
    calls = []

    def request(_prompt, _images, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _response("{", finish_reason="length")
        return _response(json.dumps(_raw_analysis()))

    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(module._client, "request", request)
    try:
        module._call_api(
            np.zeros((32, 32, 3), np.uint8), "closeup", None,
            purpose="manual_arm_check")
        assert [call["max_tokens"] for call in calls] == [450, 700]
        attempts = module.diagnostics()["last_attempt"]["attempts"]
        assert attempts[0]["finish_reason"] == "length"
        assert attempts[0]["failure_category"] == "json_parse"
    finally:
        module.close()


@pytest.mark.parametrize(
    ("status", "kind", "retry_after"),
    [
        (413, "payload_size", None),
        (429, "rate_limit", 0.0),
        (503, "server", None),
    ],
)
def test_retryable_http_failures_recover(
        monkeypatch, status, kind, retry_after):
    module = _module(monkeypatch)
    _no_retry_wait(monkeypatch)
    calls = []
    encodes = []

    def encode(_frame, **kwargs):
        encodes.append(kwargs)
        return b"jpeg"

    def request(_prompt, _images, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise NvidiaVLMError(
                f"HTTP {status}", status, retryable=True, kind=kind,
                retry_after=retry_after)
        return _response(json.dumps(_raw_analysis()))

    monkeypatch.setattr(module._client, "encode", encode)
    monkeypatch.setattr(module._client, "request", request)
    try:
        module._call_api(
            np.zeros((32, 32, 3), np.uint8), "closeup", None,
            purpose="manual_arm_check")
        assert len(calls) == 2
        assert encodes[1]["max_image_dim"] < encodes[0]["max_image_dim"]
        attempts = module.diagnostics()["last_attempt"]["attempts"]
        assert attempts[0]["failure_category"] == kind
        assert attempts[1]["outcome"] == "success"
    finally:
        module.close()


@pytest.mark.parametrize(
    ("status", "kind"),
    [(400, "client"), (401, "authentication"), (403, "authentication")],
)
def test_nonretryable_client_and_authentication_fail_once(
        monkeypatch, status, kind):
    module = _module(monkeypatch)
    calls = []

    def request(*_args, **_kwargs):
        calls.append(1)
        raise NvidiaVLMError(
            f"HTTP {status}", status, retryable=False, kind=kind)

    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(module._client, "request", request)
    try:
        with pytest.raises(SkinVisionAPIError) as error:
            module._call_api(
                np.zeros((32, 32, 3), np.uint8), "closeup", None,
                purpose="manual_arm_check")
        assert len(calls) == 1
        assert error.value.status == status
        assert error.value.kind == kind
        if status in (401, 403):
            assert module._manual_authorization_blocked is True
    finally:
        module.close()


def test_manual_capture_queues_behind_passive_and_owns_frame(monkeypatch):
    module = _module(monkeypatch, scan_interval=0.0)
    passive = Future()
    module._pending = passive
    module._pending_stage = "preliminary"
    module._pending_purpose = "passive_scan"
    module._pending_correlation_id = "passive"
    module._arm_check_capture_mode = "cloud_closeup"
    module._arm_check_correlation_id = "manual"
    original = np.full((20, 30, 3), 17, np.uint8)
    submitted = []

    def submit(frame, stage, now, **kwargs):
        submitted.append((frame.copy(), stage, now, kwargs))

    monkeypatch.setattr(module, "_submit", submit)
    try:
        module._queue_manual_capture(
            original, local_frame=None, arm_crop_label=None,
            paired_context=False)
        original[:] = 99
        assert np.all(module._queued_manual.frame == 17)
        assert module.diagnostics()["arm_check"]["queued"] is True

        ctx = FrameContext(
            np.zeros((40, 60, 3), np.uint8), 100.0, 0, 30.0,
            person_present=True)
        assert module.process(ctx) is None
        assert submitted == []

        passive.set_result(SkinAnalysis(
            "good", True, False, (), "visible skin", 0.2, (), ()))
        assert module.process(ctx) is None
        assert len(submitted) == 1
        frame, stage, _, kwargs = submitted[0]
        assert np.all(frame == 17)
        assert stage == "closeup"
        assert kwargs["purpose"] == "manual_arm_check"
        assert kwargs["deadline_monotonic"] > kwargs["captured_monotonic"]
        assert module._last_scan == -1e9
    finally:
        module._pending = None
        module.close()


def test_manual_bypasses_passive_circuit_but_honors_retry_after(monkeypatch):
    module = _module(monkeypatch)
    module._next_allowed = 1e18
    response = json.dumps(_raw_analysis())
    calls = []
    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(
        module._client, "request",
        lambda *_args, **_kwargs: calls.append(1) or response)
    try:
        module._submit(
            np.zeros((12, 12, 3), np.uint8), "closeup", 100.0,
            purpose="manual_arm_check")
        module._pending.result(timeout=2)
        assert calls == [1]

        module._pending = None
        module._manual_next_allowed = time.monotonic() + 30.0
        module._arm_check_capture_mode = "cloud_closeup"
        module._queue_manual_capture(
            np.zeros((12, 12, 3), np.uint8), local_frame=None,
            arm_crop_label=None, paired_context=False)
        assert module._drain_queued_manual(101.0) == []
        assert module._queued_manual is not None
        assert module.diagnostics()["arm_check"]["queued"] is True
    finally:
        module._pending = None
        module.close()


def test_queue_deadline_is_terminal_unavailable_with_zero_provider_attempts(
        monkeypatch):
    module = _module(monkeypatch)
    module._arm_check_capture_mode = "cloud_closeup"
    module._queue_manual_capture(
        np.zeros((12, 12, 3), np.uint8), local_frame=None,
        arm_crop_label=None, paired_context=False)
    module._queued_manual.deadline_monotonic = time.monotonic() - 0.01
    try:
        results = module._drain_queued_manual(100.0)
        assert len(results) == 1
        assert results[0].value["status"] == "unavailable"
        assert results[0].value["failure_category"] == "timeout"
        failure = module.diagnostics()["last_manual_failure"]
        assert failure["failure_category"] == "timeout"
        assert failure["attempt_count"] == 0
        assert failure["attempts"] == []
    finally:
        module.close()


def test_last_manual_failure_survives_passive_success_and_diagnostics_redact(
        monkeypatch):
    module = _module(monkeypatch, manual_max_attempts=1)
    secret_response = "raw-provider-content-must-not-remain"

    def fail(*_args, **_kwargs):
        raise NvidiaVLMError(
            "Bearer secret-key data:image/jpeg;base64,private-media "
            + secret_response,
            503, retryable=False, kind="server",
            request_id="request-safe")

    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"private-jpeg")
    monkeypatch.setattr(module._client, "request", fail)
    try:
        with pytest.raises(SkinVisionAPIError):
            module._call_api(
                np.zeros((12, 12, 3), np.uint8), "closeup", None,
                purpose="manual_arm_check")
        manual_failure = module.diagnostics()["last_manual_failure"]

        monkeypatch.setattr(
            module._client, "request",
            lambda *_args, **_kwargs: _response(
                json.dumps(_raw_analysis()), request_id="passive-safe"))
        module._call_api(
            np.zeros((12, 12, 3), np.uint8), "closeup", None,
            purpose="passive_scan")
        diagnostics = module.diagnostics()
        assert diagnostics["last_attempt"]["purpose"] == "passive_scan"
        assert diagnostics["last_manual_failure"] == manual_failure
        assert len(diagnostics["recent_requests_by_purpose"][
            "manual_arm_check"]) == 1
        assert len(diagnostics["recent_requests_by_purpose"][
            "passive_scan"]) == 1
        serialized = json.dumps(diagnostics)
        assert "secret-key" not in serialized
        assert "private-media" not in serialized
        assert "private-jpeg" not in serialized
        assert secret_response not in serialized
        assert json.dumps(_raw_analysis()) not in serialized
    finally:
        module.close()


def test_manual_deadline_does_not_start_retry_under_five_seconds(monkeypatch):
    module = _module(monkeypatch)
    calls = []

    def request(*_args, **_kwargs):
        calls.append(1)
        raise NvidiaVLMError(
            "provider request timed out", retryable=True, kind="timeout")

    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(module._client, "request", request)
    monkeypatch.setattr(
        "modules.skin_vision.time.sleep",
        lambda _seconds: pytest.fail("deadline should prevent retry backoff"))
    try:
        with pytest.raises(SkinVisionAPIError):
            module._call_api(
                np.zeros((12, 12, 3), np.uint8), "closeup", None,
                purpose="manual_arm_check",
                deadline_monotonic=time.monotonic() + 4.9)
        assert calls == [1]
        assert module.diagnostics()["last_attempt"]["attempt_count"] == 1
    finally:
        module.close()


@pytest.mark.parametrize(
    "malformed",
    [
        {
            "image_quality": "good",
            "visual_source": "live_skin",
            "sufficient_skin_visible": True,
            "body_region": "left forearm",
        },
        {**_raw_analysis(), "confidence": float("nan")},
    ],
)
def test_partial_or_nonfinite_json_can_never_become_clear(
        monkeypatch, malformed):
    module = _module(monkeypatch)
    _no_retry_wait(monkeypatch)
    calls = []
    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(
        module._client, "request",
        lambda *_args, **_kwargs: (
            calls.append(1) or _response(json.dumps(malformed))))
    module._arm_check_capture_mode = "cloud_closeup"
    try:
        module._submit(
            np.zeros((12, 12, 3), np.uint8), "closeup", 100.0,
            purpose="manual_arm_check")
        assert isinstance(module._pending.exception(timeout=2),
                          SkinVisionAPIError)
        result = module._consume_pending(101.0)
        assert len(calls) == 3
        assert len(result) == 1
        assert result[0].value["status"] == "unavailable"
        assert result[0].value["failure_category"] == "schema_validation"
        assert "finding_present" not in result[0].value
        diagnostics = module.diagnostics()
        assert diagnostics["last_manual_failure"][
            "failure_category"] == "schema_validation"
    finally:
        module.close()


@pytest.mark.parametrize(
    "contradictory_positive",
    [
        {**_raw_analysis(), "finding_present": True,
         "visible_features": ["redness"], "confidence": 0.1},
        {**_raw_analysis(), "finding_present": True,
         "visible_features": [], "confidence": 0.8},
    ],
)
def test_unsupported_positive_can_never_be_normalized_to_clear(
        monkeypatch, contradictory_positive):
    module = _module(monkeypatch)
    _no_retry_wait(monkeypatch)
    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(
        module._client, "request",
        lambda *_args, **_kwargs: _response(
            json.dumps(contradictory_positive)))
    module._arm_check_capture_mode = "cloud_closeup"
    try:
        module._submit(
            np.zeros((12, 12, 3), np.uint8), "closeup", 100.0,
            purpose="manual_arm_check")
        assert isinstance(module._pending.exception(timeout=2),
                          SkinVisionAPIError)
        result = module._consume_pending(101.0)
        assert len(result) == 1
        assert result[0].value["status"] == "unavailable"
        assert result[0].value["failure_category"] == "schema_validation"
        assert "finding_present" not in result[0].value
    finally:
        module.close()


def test_valid_negative_wrapped_in_prose_can_never_become_clear(monkeypatch):
    module = _module(monkeypatch)
    _no_retry_wait(monkeypatch)
    content = "I am unsure " + json.dumps(_raw_analysis()) + " trailing prose"
    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(
        module._client, "request",
        lambda *_args, **_kwargs: _response(content))
    module._arm_check_capture_mode = "cloud_closeup"
    try:
        module._submit(
            np.zeros((12, 12, 3), np.uint8), "closeup", 100.0,
            purpose="manual_arm_check")
        assert isinstance(module._pending.exception(timeout=2),
                          SkinVisionAPIError)
        result = module._consume_pending(101.0)
        assert len(result) == 1
        assert result[0].value["status"] == "unavailable"
        assert result[0].value["failure_category"] == "json_parse"
        assert "finding_present" not in result[0].value
    finally:
        module.close()


def test_raising_local_classifier_cannot_block_manual_cloud_analysis(
        monkeypatch):
    local_called = threading.Event()

    class RaisingLocal:
        backend = "cuda"
        model = "local-test"
        revision = "test-revision"
        target = "vitiligo"

        @staticmethod
        def predict(_frame):
            local_called.set()
            raise RuntimeError("simulated local backend failure")

        @staticmethod
        def diagnostics():
            return {"ready": True, "last_status": "ready"}

        @staticmethod
        def close():
            pass

    module = _module(monkeypatch)
    module._local_classifier = RaisingLocal()
    module._local_ready = True
    module._arm_check_capture_mode = "pose_crop_with_context"
    calls = []
    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(
        module._client, "request",
        lambda *_args, **_kwargs: (
            calls.append(1)
            or _response(json.dumps(_raw_analysis()))))
    try:
        module._submit(
            np.zeros((12, 12, 3), np.uint8), "closeup", 100.0,
            purpose="manual_arm_check",
            local_frame=np.zeros((8, 8, 3), np.uint8))
        module._pending.result(timeout=2)
        result = module._consume_pending(101.0)
        assert local_called.is_set()
        assert calls == [1]
        assert any(item.key == "arm_check"
                   and item.value["status"] == "succeeded"
                   for item in result)
        diagnostics = module.diagnostics()
        assert diagnostics["last_attempt"]["status"] == "success"
        assert diagnostics["manual_metrics"]["completed"] == 1
    finally:
        module._local_classifier = None
        module.close()


def test_blocking_local_classifier_cannot_delay_manual_cloud_deadline(
        monkeypatch):
    local_started = threading.Event()
    release_local = threading.Event()

    class BlockingLocal:
        backend = "cuda"
        model = "local-test"
        revision = "test-revision"
        target = "vitiligo"

        @staticmethod
        def predict(_frame):
            local_started.set()
            release_local.wait(2)
            return None

        @staticmethod
        def diagnostics():
            return {"ready": True, "last_status": "ready"}

        @staticmethod
        def close():
            pass

    module = _module(monkeypatch)
    module._local_classifier = BlockingLocal()
    module._local_ready = True
    module._arm_check_capture_mode = "pose_crop_with_context"
    cloud_called = threading.Event()
    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")

    def cloud_response(*_args, **_kwargs):
        cloud_called.set()
        return _response(json.dumps(_raw_analysis()))

    monkeypatch.setattr(module._client, "request", cloud_response)
    try:
        started = time.monotonic()
        module._submit(
            np.zeros((12, 12, 3), np.uint8), "closeup", 100.0,
            purpose="manual_arm_check",
            local_frame=np.zeros((8, 8, 3), np.uint8),
            deadline_monotonic=time.monotonic() + 5.0)
        module._pending.result(timeout=1)
        elapsed = time.monotonic() - started
        assert local_started.is_set()
        assert cloud_called.is_set()
        assert elapsed < 0.75
        result = module._consume_pending(101.0)
        assert any(item.key == "arm_check"
                   and item.value["status"] == "succeeded"
                   for item in result)
    finally:
        release_local.set()
        module._local_classifier = None
        module.close()


def test_recent_logical_request_history_is_capped_at_twenty(monkeypatch):
    module = _module(monkeypatch)
    monkeypatch.setattr(module._client, "encode",
                        lambda _frame, **_kwargs: b"jpeg")
    monkeypatch.setattr(
        module._client, "request",
        lambda *_args, **_kwargs: _response(json.dumps(_raw_analysis())))
    try:
        for _ in range(21):
            module._call_api(
                np.zeros((4, 4, 3), np.uint8), "closeup", None,
                purpose="passive_scan")
        assert len(module.diagnostics()["recent_requests"]) == 20
    finally:
        module.close()
