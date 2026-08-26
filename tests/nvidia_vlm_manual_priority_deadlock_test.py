"""Regression coverage for the three-party NVIDIA priority handoff."""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np

from integrations.nvidia_vlm import (
    NvidiaVLMClient,
    _COORDINATOR,
)
from modules.skin_vision import SkinVision


class _Response:
    """Minimal urllib response carrying one OpenAI-compatible choice."""

    status = 200
    headers: dict[str, str] = {}

    def __init__(self, content: str):
        self._raw = json.dumps({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": content},
            }],
        }).encode("utf-8")

    def read(self, _size: int = -1) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _closeup_result() -> dict[str, Any]:
    return {
        "image_quality": "good",
        "visual_source": "live_skin",
        "sufficient_skin_visible": True,
        "finding_present": True,
        "visible_features": ["discoloration"],
        "body_region": "left forearm",
        "confidence": 0.8,
        "possible_conditions": ["possible bruise"],
        "follow_up_topics": ["duration"],
    }


def _preliminary_result() -> dict[str, Any]:
    """Valid fallback body used only to let an unexpected HTTP call terminate."""
    return {
        **_closeup_result(),
        "visual_source": "live_skin",
        "eye_redness": "unclear",
        "under_eye_darkness": "unclear",
        "eyelid_swelling": "unclear",
        "lip_pallor": "unclear",
        "nasal_discharge_visible": "unclear",
        "facial_cue_confidence": 0.0,
    }


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for deterministic handoff")
        time.sleep(0.002)


def test_manual_capture_withdraws_skin_passive_behind_active_scene(
        monkeypatch):
    """A manual reservation cannot deadlock with queued Skin passive work."""
    _COORDINATOR.reset_for_tests()
    scene_started = threading.Event()
    release_scene = threading.Event()
    manual_http_started = threading.Event()
    calls: list[str] = []
    calls_lock = threading.Lock()
    scene_outcome: dict[str, Any] = {}

    def fake_open(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        prompt = payload["messages"][0]["content"][0]["text"]
        if prompt == "scene-active":
            kind = "scene-active"
        elif prompt.startswith("Screen the visible person"):
            kind = "skin-passive"
        else:
            kind = "manual"
        with calls_lock:
            calls.append(kind)
        if kind == "scene-active":
            scene_started.set()
            if not release_scene.wait(3.0):
                raise TimeoutError("test did not release active scene request")
            return _Response("{}")
        if kind == "manual":
            manual_http_started.set()
            return _Response(json.dumps(_closeup_result()))
        return _Response(json.dumps(_preliminary_result()))

    monkeypatch.setattr(
        "integrations.nvidia_vlm._open_no_redirect", fake_open)
    monkeypatch.setattr(
        "modules.skin_vision.nvidia_api_key", lambda: "test-private-key")

    endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
    model = "meta/llama-3.2-11b-vision-instruct"
    scene_client = NvidiaVLMClient(
        "test-private-key", endpoint, model, timeout=3.0)
    module = SkinVision(
        consent=True,
        request_timeout=3.0,
        manual_attempt_timeout=3.0,
        manual_retry_deadline=8.0,
        manual_max_attempts=1,
    )
    monkeypatch.setattr(
        module._client, "encode", lambda _frame, **_kwargs: b"jpeg")

    def run_scene() -> None:
        try:
            scene_outcome["content"] = scene_client.request(
                "scene-active",
                [b"jpeg"],
                purpose="passive_scene_scan",
                timeout=3.0,
                deadline=time.monotonic() + 8.0,
            ).content
        except BaseException as exc:  # retained only within this test
            scene_outcome["error"] = exc

    scene_thread = threading.Thread(
        target=run_scene, name="test-active-scene")
    scene_thread.start()
    try:
        assert scene_started.wait(1.0)

        # This Skin request queues behind the active Scene-like request and
        # must never get far enough to invoke the HTTP opener.
        module._submit(
            np.zeros((32, 32, 3), np.uint8),
            "preliminary",
            100.0,
            purpose="passive_scan",
        )
        _wait_until(
            lambda: NvidiaVLMClient.coordinator_diagnostics()
            ["queued_by_purpose"]["passive"] == 1)
        passive_future = module._pending
        assert passive_future is not None

        module._arm_check_capture_mode = "cloud_closeup"
        module._arm_check_correlation_id = "manual-priority-regression"
        module._queue_manual_capture(
            np.full((32, 32, 3), 17, np.uint8),
            local_frame=None,
            arm_crop_label="left forearm",
            paired_context=False,
        )

        _wait_until(passive_future.done)
        diagnostics = NvidiaVLMClient.coordinator_diagnostics()
        assert diagnostics["active"] == "passive"
        assert diagnostics["queued_by_purpose"]["passive"] == 0
        assert diagnostics["reserved_by_purpose"]["manual"] == 1
        assert calls == ["scene-active"]

        # Clear the withdrawn Skin Future, then enqueue its owned manual
        # capture while the Scene-like request is still the active blocker.
        module._consume_pending(101.0)
        assert module._pending is None
        assert module._drain_queued_manual(101.0) == []
        manual_future = module._pending
        assert manual_future is not None
        _wait_until(
            lambda: NvidiaVLMClient.coordinator_diagnostics()
            ["queued_by_purpose"]["manual"] == 1)
        assert not manual_http_started.is_set()

        release_scene.set()
        scene_thread.join(2.0)
        assert not scene_thread.is_alive()
        assert scene_outcome == {"content": "{}"}

        result = manual_future.result(timeout=2.0)
        assert result.finding_present is True
        assert result.visible_features == ("discoloration",)
        assert manual_http_started.is_set()
        assert calls == ["scene-active", "manual"]

        module._consume_pending(102.0)
        _wait_until(
            lambda: NvidiaVLMClient.coordinator_diagnostics()["active"] is None)
        diagnostics = NvidiaVLMClient.coordinator_diagnostics()
        assert diagnostics["queued"] == 0
        assert diagnostics["reserved_by_purpose"]["manual"] == 0
    finally:
        release_scene.set()
        NvidiaVLMClient.notify_cancellation()
        module.close()
        scene_thread.join(2.0)
        _wait_until(
            lambda: (
                NvidiaVLMClient.coordinator_diagnostics()["active"] is None
                and NvidiaVLMClient.coordinator_diagnostics()["queued"] == 0
            ))
        _COORDINATOR.reset_for_tests()
