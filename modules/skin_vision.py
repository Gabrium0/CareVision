"""Opt-in, two-stage skin screening through an NVIDIA vision model.

The module sends infrequent in-memory JPEG stills only after explicit runtime
consent. A preliminary whole-frame observation asks the voice agent to request
a close-up; only the close-up can emit a public, non-diagnostic observation.
Possible condition names remain agent-only and are never persisted here.
"""
from __future__ import annotations

import base64
import copy
import json
import queue
import re
import threading
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from agent.env import nvidia_api_key
from core.context import FrameContext
from core.elicitation import ElicitationState
from core.events import PersistencePolicy, Severity, Visibility
from core.registry import register
from modules.base import DetectionModule
from core.one_flight import DaemonOneFlight
from modules.local_skin_classifier import (
    LocalSkinPrediction,
    build_local_skin_classifier,
)
from integrations.nvidia_vlm import NvidiaVLMClient, NvidiaVLMError
from core.capabilities import CapabilityRegistry, CapabilityStatus


_FEATURES = {
    "redness", "discoloration", "swelling", "scaling", "blistering",
    "rash-like texture", "dryness", "lesion", "bruising", "irritation",
}
_TOPICS = {
    "itching", "pain", "duration", "spreading", "fever_unwell",
    "new_medication", "new_product_exposure", "blisters",
}
_QUALITY = {"poor", "fair", "good"}
_APPEARANCE_LEVELS = {"none", "mild", "marked", "unclear"}
_NASAL_LEVELS = {"no", "yes", "unclear"}
_FACIAL_KEYS = (
    "under_eye_darkness", "under_eye_puffiness", "nose_redness",
    "cheek_redness", "lip_dryness", "nasal_discharge_visible",
)
_FACIAL_LABELS = {
    "under_eye_darkness": "under-eye darkness",
    "under_eye_puffiness": "under-eye puffiness",
    "nose_redness": "nose redness",
    "cheek_redness": "cheek redness",
    "lip_dryness": "lip dryness",
    "nasal_discharge_visible": "visible nasal discharge",
}
_SCHEMA_TEXT = """Return exactly one JSON object matching the contract printed
below. Include every required field, use enum values exactly, and emit no prose
or markdown. Use possible_conditions only for uncertain internal hypotheses.
Do not infer a condition when the image is unclear."""
_COMPOSITE_WIDTH = 1024
_WHOLE_PANEL_HEIGHT = 576
_FACE_PANEL_HEIGHT = 448
_PANEL_BACKGROUND = 32


def _skin_properties() -> dict[str, Any]:
    return {
        "image_quality": {"type": "string", "enum": sorted(_QUALITY)},
        "sufficient_skin_visible": {"type": "boolean"},
        "finding_present": {"type": "boolean"},
        "visible_features": {
            "type": "array", "items": {"type": "string", "enum": sorted(_FEATURES)},
            "maxItems": 5,
        },
        "body_region": {"type": "string", "maxLength": 60},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "possible_conditions": {
            "type": "array", "items": {"type": "string", "maxLength": 80},
            "maxItems": 3,
        },
        "follow_up_topics": {
            "type": "array", "items": {"type": "string", "enum": sorted(_TOPICS)},
            "maxItems": 5,
        },
    }


_CLOSEUP_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": _skin_properties(),
    "required": list(_skin_properties()),
}
_PRELIMINARY_PROPERTIES = {
    **_skin_properties(),
    **{key: {"type": "string", "enum": sorted(_APPEARANCE_LEVELS)}
       for key in _FACIAL_KEYS[:-1]},
    "nasal_discharge_visible": {"type": "string", "enum": sorted(_NASAL_LEVELS)},
    "facial_cue_confidence": {
        "type": "number", "minimum": 0.0, "maximum": 1.0,
    },
}
_PRELIMINARY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": _PRELIMINARY_PROPERTIES,
    "required": list(_PRELIMINARY_PROPERTIES),
}


def _response_format(stage: str) -> dict[str, Any]:
    schema = _PRELIMINARY_SCHEMA if stage == "preliminary" else _CLOSEUP_SCHEMA
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"skin_{stage}_analysis",
            "strict": True,
            "schema": schema,
        },
    }


def _schema_contract(stage: str) -> str:
    """Put the provider-enforced contract in the prompt as a safe fallback."""
    schema = _PRELIMINARY_SCHEMA if stage == "preliminary" else _CLOSEUP_SCHEMA
    return json.dumps(schema, separators=(",", ":"), sort_keys=True)


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-fit a BGR image into a neutral in-memory panel."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
        raise ValueError("invalid composite image")
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    target_w = max(1, min(width, int(round(source_w * scale))))
    target_h = max(1, min(height, int(round(source_h * scale))))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    panel = np.full((height, width, 3), _PANEL_BACKGROUND, dtype=np.uint8)
    x = (width - target_w) // 2
    y = (height - target_h) // 2
    panel[y:y + target_h, x:x + target_w] = resized
    return panel


def _label_panel(panel: np.ndarray, label: str) -> None:
    """Add a compact high-contrast view label without retaining extra data."""
    cv2.rectangle(panel, (10, 10), (230, 43), (0, 0, 0), thickness=-1)
    cv2.putText(panel, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)


def _compose_preliminary_frame(whole: np.ndarray, face: np.ndarray | None = None,
                               arm: np.ndarray | None = None,
                               arm_label: str = "ARM CROP") -> np.ndarray:
    """Combine whole-frame context with enlarged face/arm detail panels.

    With both crops the bottom row splits face | arm side by side; with one
    crop it fills the row, preserving the original two-panel layout.
    """
    top = _fit_panel(whole, _COMPOSITE_WIDTH, _WHOLE_PANEL_HEIGHT)
    _label_panel(top, "WHOLE FRAME")
    if face is not None and arm is not None:
        half = _COMPOSITE_WIDTH // 2
        left = _fit_panel(face, half, _FACE_PANEL_HEIGHT)
        right = _fit_panel(arm, _COMPOSITE_WIDTH - half, _FACE_PANEL_HEIGHT)
        _label_panel(left, "FACE CROP")
        _label_panel(right, arm_label)
        bottom = np.hstack((left, right))
        cv2.line(bottom, (half - 1, 0), (half - 1, _FACE_PANEL_HEIGHT - 1),
                 (220, 220, 220), 2)
    elif arm is not None:
        bottom = _fit_panel(arm, _COMPOSITE_WIDTH, _FACE_PANEL_HEIGHT)
        _label_panel(bottom, arm_label)
    else:
        bottom = _fit_panel(face, _COMPOSITE_WIDTH, _FACE_PANEL_HEIGHT)
        _label_panel(bottom, "FACE CROP")
    composite = np.vstack((top, bottom))
    cv2.line(composite, (0, _WHOLE_PANEL_HEIGHT - 2),
             (_COMPOSITE_WIDTH - 1, _WHOLE_PANEL_HEIGHT - 2),
             (220, 220, 220), 4)
    return composite


@dataclass(frozen=True)
class FacialCues:
    """Validated visible facial appearance cues from a clear face crop."""

    under_eye_darkness: str = "unclear"
    under_eye_puffiness: str = "unclear"
    nose_redness: str = "unclear"
    cheek_redness: str = "unclear"
    lip_dryness: str = "unclear"
    nasal_discharge_visible: str = "unclear"
    confidence: float = 0.0

    def positive(self) -> dict[str, str]:
        """Return only affirmative observable cues, omitting none/unclear values."""
        out = {}
        for key in _FACIAL_KEYS[:-1]:
            value = getattr(self, key)
            if value in ("mild", "marked"):
                out[key] = value
        if self.nasal_discharge_visible == "yes":
            out["nasal_discharge_visible"] = "yes"
        return out


@dataclass(frozen=True)
class SkinAnalysis:
    """Validated, bounded output from the remote vision model."""

    image_quality: str
    sufficient_skin_visible: bool
    finding_present: bool
    visible_features: tuple[str, ...]
    body_region: str
    confidence: float
    possible_conditions: tuple[str, ...]
    follow_up_topics: tuple[str, ...]
    facial_cues: FacialCues | None = None

    def private_value(self, local: LocalSkinPrediction | None = None) -> dict[str, Any]:
        """Return JSON-safe private context for the agent."""
        value = {
            "image_quality": self.image_quality,
            "visible_features": list(self.visible_features),
            "body_region": self.body_region,
            "confidence": self.confidence,
            "possible_conditions": list(self.possible_conditions),
            "follow_up_topics": list(self.follow_up_topics),
        }
        if self.facial_cues is not None:
            value["facial_cues"] = self.facial_cues.positive()
            value["facial_cue_confidence"] = self.facial_cues.confidence
        if local is not None:
            value["local_classifier"] = local.private_value()
        return value


class SkinVisionAPIError(RuntimeError):
    """Remote inference failed without retaining response or image data."""

    def __init__(self, message: str, status: int | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = bool(retryable)


@dataclass(frozen=True)
class _CloseupOutcome:
    """Independent cloud/local outcomes from one close-up frame."""

    analysis: SkinAnalysis | None
    local: LocalSkinPrediction | None
    cloud_error: BaseException | None = None


class _DaemonOneFlight:
    """One queued daemon task; shutdown never waits on a blocked HTTP call."""

    def __init__(self, name: str):
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn, *args) -> Future:
        future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("worker closed")
            self._queue.put_nowait((future, fn, args))
        return future

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, fn, args = item
            if not future.set_running_or_notify_cancel():
                with self._lock:
                    if self._closed:
                        return
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
            with self._lock:
                if self._closed:
                    return

    def shutdown(self, timeout: float = 0.5) -> bool:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
        self._thread.join(timeout=max(0.0, float(timeout)))
        return not self._thread.is_alive()


def _bounded_text(value: Any, limit: int) -> str:
    text = re.sub(r"[\r\n\t]+", " ", str(value or "")).strip()
    return re.sub(r"\s+", " ", text)[:limit]


def _string_list(value: Any, allowed: set[str] | None = None,
                 limit: int = 3) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out = []
    for item in value:
        text = _bounded_text(item, 80).lower()
        if not text or (allowed is not None and text not in allowed) or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return tuple(out)


def validate_analysis(raw: Any, min_confidence: float = 0.35, *,
                      allow_facial_cues: bool = False,
                      min_facial_confidence: float = 0.45,
                      face_crop_available: bool = True) -> SkinAnalysis:
    """Validate and normalize the model JSON; reject unsafe loose structures."""
    if not isinstance(raw, dict):
        raise ValueError("skin response must be a JSON object")
    quality = _bounded_text(raw.get("image_quality"), 10).lower()
    if quality not in _QUALITY:
        raise ValueError("invalid image_quality")
    sufficient = raw.get("sufficient_skin_visible") is True
    finding = raw.get("finding_present") is True
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid confidence") from exc
    features = _string_list(raw.get("visible_features"), _FEATURES, limit=5)
    topics = _string_list(raw.get("follow_up_topics"), _TOPICS, limit=5)
    conditions = _string_list(raw.get("possible_conditions"), limit=3)
    region = _bounded_text(raw.get("body_region") or "visible skin", 60)
    finding = bool(finding and sufficient and quality != "poor"
                   and confidence >= min_confidence and features)
    if not finding:
        conditions = ()
        topics = ()
    facial_cues = None
    if allow_facial_cues:
        values = {}
        for key in _FACIAL_KEYS[:-1]:
            value = raw.get(key)
            if value not in _APPEARANCE_LEVELS:
                raise ValueError(f"invalid {key}")
            values[key] = value
        nasal = raw.get("nasal_discharge_visible")
        if nasal not in _NASAL_LEVELS:
            raise ValueError("invalid nasal_discharge_visible")
        try:
            facial_confidence = max(0.0, min(1.0,
                float(raw.get("facial_cue_confidence", 0.0))))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid facial_cue_confidence") from exc
        if (not face_crop_available or quality == "poor"
                or facial_confidence < min_facial_confidence):
            values = {key: "unclear" for key in _FACIAL_KEYS[:-1]}
            nasal = "unclear"
            facial_confidence = 0.0
        facial_cues = FacialCues(**values, nasal_discharge_visible=nasal,
                                  confidence=round(facial_confidence, 3))
    return SkinAnalysis(quality, sufficient, finding, features, region,
                        round(confidence, 3), conditions, topics, facial_cues)


def _extract_json(content: Any) -> dict:
    if isinstance(content, list):
        content = "".join(str(p.get("text", "")) for p in content
                          if isinstance(p, dict))
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response did not contain JSON")
    return json.loads(text[start:end + 1])


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@register("skin_vision")
class SkinVision(DetectionModule):
    """Background NVIDIA whole-frame screening followed by a guided close-up."""

    interval = 0.0
    requires = ()
    consent = False
    endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
    model = "meta/llama-3.2-11b-vision-instruct"
    scan_interval = 60.0
    request_timeout = 25.0
    max_image_dim = 1024
    jpeg_quality = 85
    closeup_seconds = 10.0
    closeup_positioning_delay = 2.0
    min_confidence = 0.35
    min_facial_confidence = 0.45
    min_face_crop_size = 64
    min_arm_crop_size = 64
    backoff_base = 15.0
    backoff_max = 900.0
    manual_retry_deadline = 45.0
    manual_retry_max_image_dim = 768
    manual_retry_jpeg_quality = 80
    manual_retry_max_tokens = 450
    local_classifier = None

    def __init__(self, **params):
        super().__init__(**params)
        local_config = self.local_classifier if isinstance(self.local_classifier, dict) else {}
        self._local_mode = str(local_config.get("mode", "debug")).strip().lower()
        if self._local_mode not in {"debug", "screening"}:
            self._local_mode = "debug"
        self._local_config_error: str | None = None
        self._local_load_timeout = float(local_config.get("load_timeout_seconds", 120.0))
        self._local_load_started: float | None = None
        self._local_load_timed_out = False
        try:
            self._local_classifier = build_local_skin_classifier(local_config)
        except (TypeError, ValueError) as exc:
            self._local_classifier = None
            self._local_config_error = f"{type(exc).__name__}: {str(exc).strip()}"[:160]
        self._local_worker = DaemonOneFlight("skin-local-classifier")
        self._local_load_future: Future | None = None
        self._local_load_attempted = False
        self._local_ready = False
        self._key = nvidia_api_key()
        self._client = NvidiaVLMClient(self._key or "", self.endpoint, self.model,
                                       float(self.request_timeout), int(self.max_image_dim),
                                       int(self.jpeg_quality))
        self._executor = _DaemonOneFlight("skin-vision")
        self._closed = False
        self._pending: Future | None = None
        self._pending_stage: str | None = None
        self._pending_purpose: str | None = None
        self._pending_correlation_id: str | None = None
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_status = "idle"
        self._diagnostic_current_stage: str | None = None
        self._diagnostic_started_at: float | None = None
        self._diagnostic_started_monotonic: float | None = None
        self._diagnostic_request_count = 0
        self._diagnostic_success_count = 0
        self._diagnostic_failure_count = 0
        self._diagnostic_last_attempt: dict[str, Any] | None = None
        self._diagnostic_purpose: str | None = None
        self._diagnostic_correlation_id: str | None = None
        self._diagnostic_attempt = 0
        self._diagnostic_payload_mode = "normal"
        self._diagnostic_deadline_at: float | None = None
        self._last_scan = -1e9
        self._next_allowed = -1e9
        self._failures = 0
        self._preliminary: SkinAnalysis | None = None
        self._preliminary_at = 0.0
        self._awaiting_closeup = False
        self._sampling_closeup = False
        self._arm_check_sampling = False
        self._arm_check_state = "idle"
        self._arm_check_last_error: str | None = None
        self._best_frame: np.ndarray | None = None
        self._best_sharpness = -1.0
        self._correlation_id: str | None = None
        self._elicitation = ElicitationState.instance()
        if self.consent and self._key:
            print(f"[skin-vision] cloud screening enabled (model {self.model})")
            CapabilityRegistry.instance().set("nvidia_skin", "cloud",
                                               CapabilityStatus.LOADING,
                                               "configured; first validated response pending")
        elif self.consent:
            print("[skin-vision] consent given but NVIDIA_API_KEY is missing; disabled")
            CapabilityRegistry.instance().set("nvidia_skin", "cloud",
                                               CapabilityStatus.UNCONFIGURED, "credential unavailable")
        else:
            print("[skin-vision] cloud screening disabled (use --enable-cloud-skin)")
            CapabilityRegistry.instance().set("nvidia_skin", "cloud",
                                               CapabilityStatus.UNCONFIGURED, "consent off")
        if self._local_config_error:
            CapabilityRegistry.instance().set(
                "local_skin_classifier", "model", CapabilityStatus.FAILED,
                self._local_config_error)
        elif self._local_classifier is None:
            CapabilityRegistry.instance().set(
                "local_skin_classifier", "model", CapabilityStatus.UNCONFIGURED,
                "local close-up classifier disabled")

    def start(self) -> None:
        """Begin idempotent asynchronous local-model preload."""
        if (self._local_classifier is None or self._local_load_attempted
                or self._closed):
            return
        self._local_load_attempted = True
        self._local_load_started = time.monotonic()
        CapabilityRegistry.instance().set(
            "local_skin_classifier", "model", CapabilityStatus.LOADING,
            "loading pinned close-up classifier")
        try:
            self._local_load_future = self._local_worker.submit(
                self._local_classifier.preload)
        except BaseException as exc:
            self._local_config_error = f"{type(exc).__name__}: {str(exc).strip()}"[:160]
            CapabilityRegistry.instance().set(
                "local_skin_classifier", "model", CapabilityStatus.FAILED,
                self._local_config_error)

    def _refresh_local_load(self) -> None:
        future = self._local_load_future
        if future is None:
            return
        if not future.done():
            if (not self._local_load_timed_out and self._local_load_started is not None
                    and time.monotonic() - self._local_load_started
                    > self._local_load_timeout):
                self._local_load_timed_out = True
                CapabilityRegistry.instance().set(
                    "local_skin_classifier", "model", CapabilityStatus.DEGRADED,
                    "local model load exceeded deadline; late recovery remains possible")
            return
        self._local_load_future = None
        try:
            future.result()
            self._local_ready = bool(self._local_classifier and self._local_classifier.ready)
        except BaseException as exc:
            self._local_ready = False
            self._local_config_error = f"{type(exc).__name__}: {str(exc).strip()}"[:160]
        if self._local_ready:
            details = self._local_classifier.diagnostics()
            CapabilityRegistry.instance().set(
                "local_skin_classifier", "model", CapabilityStatus.READY,
                f"{details.get('backend', 'local')} close-up classifier ready")
        else:
            CapabilityRegistry.instance().set(
                "local_skin_classifier", "model", CapabilityStatus.FAILED,
                self._local_config_error or "local classifier failed to load")

    @property
    def available(self) -> bool:
        """Whether this run has both explicit consent and credentials."""
        return bool(self.consent and self._key and self.endpoint and self.model)

    def _encode(self, frame: np.ndarray) -> bytes:
        return self._client.encode(frame)

    def diagnostics(self) -> dict[str, Any]:
        """Return the latest private, in-memory NVIDIA request diagnostics."""
        local = (self._local_classifier.diagnostics()
                 if self._local_classifier is not None else {
                     "ready": False,
                     "last_status": "unconfigured",
                     "last_error": self._local_config_error,
                 })
        local["mode"] = self._local_mode
        local["worker"] = self._local_worker.diagnostics()
        with self._diagnostic_lock:
            return {
                "available": self.available,
                "consent": bool(self.consent),
                "model": self.model,
                "status": self._diagnostic_status,
                "current_stage": self._diagnostic_current_stage,
                "current_purpose": self._diagnostic_purpose,
                "correlation_id": self._diagnostic_correlation_id,
                "request_started_at": self._diagnostic_started_at,
                "request_count": self._diagnostic_request_count,
                "success_count": self._diagnostic_success_count,
                "failure_count": self._diagnostic_failure_count,
                "success_rate": round(self._diagnostic_success_count /
                                      max(1, self._diagnostic_request_count), 3),
                "consecutive_failures": self._failures,
                "circuit_state": ("open" if time.time() < self._next_allowed else "closed"),
                "retry_after_seconds": round(max(0.0, self._next_allowed - time.time()), 1),
                "attempt": self._diagnostic_attempt,
                "payload_mode": self._diagnostic_payload_mode,
                "deadline_remaining_seconds": (round(max(
                    0.0, self._diagnostic_deadline_at - time.time()), 1)
                    if self._diagnostic_deadline_at else None),
                "arm_check": {
                    "state": self._arm_check_state,
                    "last_error": self._arm_check_last_error,
                    "correlation_id": (self._diagnostic_correlation_id
                                       if self._diagnostic_purpose == "manual_arm_check"
                                       else None),
                },
                "local_classifier": local,
                "last_attempt": copy.deepcopy(self._diagnostic_last_attempt),
            }

    def _begin_request_diagnostics(self, stage: str, purpose: str,
                                   correlation_id: str,
                                   deadline_seconds: float | None = None) -> None:
        with self._diagnostic_lock:
            self._diagnostic_status = "in_flight"
            self._diagnostic_current_stage = stage
            self._diagnostic_started_at = time.time()
            self._diagnostic_started_monotonic = time.monotonic()
            self._diagnostic_request_count += 1
            self._diagnostic_purpose = purpose
            self._diagnostic_correlation_id = correlation_id
            self._diagnostic_attempt = 0
            self._diagnostic_payload_mode = "normal"
            self._diagnostic_deadline_at = (time.time() + deadline_seconds
                                            if deadline_seconds else None)

    def _finish_request_diagnostics(self, status: str, stage: str, *,
                                    error: str | None = None,
                                    http_status: int | None = None,
                                    validation: dict[str, Any] | None = None,
                                    repair_attempted: bool = False,
                                    retryable: bool = False,
                                    terminal_reason: str | None = None) -> None:
        completed_at = time.time()
        completed_monotonic = time.monotonic()
        with self._diagnostic_lock:
            started_at = self._diagnostic_started_at
            started_monotonic = self._diagnostic_started_monotonic
            latency_ms = (round((completed_monotonic - started_monotonic) * 1000.0, 1)
                          if started_monotonic is not None else None)
            self._diagnostic_status = status
            self._diagnostic_current_stage = None
            self._diagnostic_started_at = None
            self._diagnostic_started_monotonic = None
            if status == "success":
                self._diagnostic_success_count += 1
                CapabilityRegistry.instance().set(
                    "nvidia_skin", "cloud", CapabilityStatus.READY,
                    "validated structured response received")
            else:
                self._diagnostic_failure_count += 1
                CapabilityRegistry.instance().set(
                    "nvidia_skin", "cloud", CapabilityStatus.DEGRADED,
                    terminal_reason or error or status)
            self._diagnostic_last_attempt = {
                "status": status,
                "stage": stage,
                "purpose": self._diagnostic_purpose,
                "correlation_id": self._diagnostic_correlation_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "latency_ms": latency_ms,
                "http_status": http_status,
                "error": error,
                "validation": copy.deepcopy(validation),
                "repair_attempted": bool(repair_attempted),
                "retryable": bool(retryable),
                "attempt_count": self._diagnostic_attempt,
                "payload_mode": self._diagnostic_payload_mode,
                "terminal_reason": terminal_reason,
            }

    def _prompt(self, stage: str, previous: SkinAnalysis | None,
                face_crop_available: bool = False,
                arm_crop_label: str | None = None) -> str:
        if stage == "preliminary":
            task = ("Screen the visible person for an obvious possible skin change. "
                    "This is a low-confidence screening step, not a diagnosis. "
                    "Name the body region precisely enough to request a close-up. ")
            if face_crop_available and arm_crop_label:
                task += ("The single image is a labeled composite: the top panel is the "
                         "whole camera frame, the bottom-left panel is an enlarged face "
                         "crop, and the bottom-right panel is an enlarged crop of the "
                         f"person's {arm_crop_label}. Use the whole-frame and arm-crop "
                         "panels for skin context and report only directly visible "
                         "facial appearance cues from the face-crop panel. Do not "
                         "infer tiredness, illness, allergies, dehydration, or any diagnosis. ")
            elif face_crop_available:
                task += ("The single image is a labeled composite: the top panel is the "
                         "whole camera frame and the bottom panel is an enlarged face crop. "
                         "Use the whole-frame panel for skin context and report only directly "
                         "visible facial appearance cues from the face-crop panel. Do not "
                         "infer tiredness, illness, allergies, dehydration, or any diagnosis. ")
            elif arm_crop_label:
                task += ("The single image is a labeled composite: the top panel is the "
                         "whole camera frame and the bottom panel is an enlarged crop of "
                         f"the person's {arm_crop_label}. Use both panels for skin "
                         "context. No usable face crop is available; set every facial "
                         "appearance field and nasal_discharge_visible to unclear, with "
                         "facial cue confidence 0. ")
            else:
                task += ("The single image is the whole camera frame. No usable face crop "
                         "is available; set every facial appearance "
                         "field and nasal_discharge_visible to unclear, with facial cue "
                         "confidence 0. ")
        else:
            task = ("Inspect this user-provided close-up for visible skin changes. "
                    "Be conservative and non-diagnostic.")
            if previous is not None:
                task += (f" The preliminary frame indicated {', '.join(previous.visible_features)} "
                         f"around {previous.body_region}.")
        return task + "\n\n" + _SCHEMA_TEXT + "\nJSON contract:\n" + _schema_contract(stage)

    @staticmethod
    def _validation_detail(raw: Any, stage: str, exc: BaseException) -> dict[str, Any]:
        schema = _PRELIMINARY_SCHEMA if stage == "preliminary" else _CLOSEUP_SCHEMA
        missing = ([key for key in schema["required"] if key not in raw]
                   if isinstance(raw, dict) else list(schema["required"]))
        return {"reason": _bounded_text(str(exc), 160) or type(exc).__name__,
                "missing_fields": missing[:20],
                "response_type": type(raw).__name__}

    def _call_api(self, frames: list[np.ndarray] | bytes, stage: str,
                  previous: SkinAnalysis | None, face_crop_available: bool = False,
                  arm_crop_label: str | None = None,
                  purpose: str = "passive_scan") -> SkinAnalysis:
        preencoded = isinstance(frames, (bytes, bytearray))
        if preencoded:
            frames = [bytes(frames)]
        prompt = self._prompt(stage, previous, face_crop_available, arm_crop_label)
        validation = None
        manual = purpose == "manual_arm_check"
        deadline = (time.monotonic() + float(self.manual_retry_deadline)
                    if manual else None)
        compact = False
        for attempt in range(2):
            repair = attempt == 1 and validation is not None
            with self._diagnostic_lock:
                self._diagnostic_attempt = attempt + 1
                self._diagnostic_payload_mode = "compact_retry" if compact else "normal"
            request_prompt = prompt
            if repair:
                request_prompt += ("\nYour prior response failed validation. Repair the "
                                   "format now; include every required field and JSON only.")
            try:
                images = ([self._client.encode(
                    item, max_image_dim=int(self.manual_retry_max_image_dim),
                    jpeg_quality=int(self.manual_retry_jpeg_quality)) for item in frames]
                    if compact and not preencoded else
                    [bytes(item) for item in frames] if preencoded else
                    [self._encode(item) for item in frames])
                remaining = ((deadline - time.monotonic()) if deadline is not None
                             else float(self.request_timeout))
                if remaining <= 0:
                    raise NvidiaVLMError("manual arm deadline exceeded", retryable=False)
                content = self._client.request(
                    request_prompt, images,
                    max_tokens=(int(self.manual_retry_max_tokens) if compact else 700),
                    response_format=_response_format(stage),
                    timeout=min(float(self.request_timeout), remaining))
            except NvidiaVLMError as exc:
                may_retry = bool(manual and attempt == 0 and exc.retryable
                                 and deadline is not None and time.monotonic() < deadline)
                if may_retry:
                    compact = True
                    validation = None
                    continue
                self._finish_request_diagnostics(
                    "error", stage, error=str(exc), http_status=exc.status,
                    validation=validation, repair_attempted=repair,
                    retryable=exc.retryable, terminal_reason="provider_failure")
                raise SkinVisionAPIError(str(exc), status=exc.status,
                                         retryable=exc.retryable) from exc
            raw: Any = None
            try:
                raw = _extract_json(content)
                analysis = validate_analysis(
                    raw, float(self.min_confidence),
                    allow_facial_cues=stage == "preliminary",
                    min_facial_confidence=float(self.min_facial_confidence),
                    face_crop_available=face_crop_available)
                self._finish_request_diagnostics(
                    "success", stage, validation=None, repair_attempted=repair)
                return analysis
            except (KeyError, IndexError, TypeError, ValueError,
                    json.JSONDecodeError) as exc:
                validation = self._validation_detail(raw, stage, exc)
        self._finish_request_diagnostics(
            "invalid_response", stage, error="invalid structured response",
            validation=validation, repair_attempted=True,
            terminal_reason="schema_validation_failed")
        raise SkinVisionAPIError("invalid structured response")

    def _call_closeup(self, frames: list[np.ndarray], previous: SkinAnalysis | None,
                      purpose: str, use_cloud: bool) -> _CloseupOutcome:
        """Analyze one close-up locally and in the cloud without coupling failures."""
        local = None
        if self._local_classifier is not None and self._local_ready:
            local = self._local_classifier.predict(frames[0])
            if local.status == "unavailable":
                CapabilityRegistry.instance().set(
                    "local_skin_classifier", "model", CapabilityStatus.DEGRADED,
                    local.abstain_reason or "local inference unavailable")
            else:
                CapabilityRegistry.instance().set(
                    "local_skin_classifier", "model", CapabilityStatus.READY,
                    f"{local.backend} close-up inference {local.inference_ms:.1f} ms")
        analysis = None
        cloud_error = None
        if use_cloud:
            try:
                analysis = self._call_api(
                    frames, "closeup", previous, purpose=purpose)
            except BaseException as exc:  # cloud degradation must not discard local evidence
                cloud_error = exc
        return _CloseupOutcome(analysis, local, cloud_error)

    @staticmethod
    def _local_corroborates(analysis: SkinAnalysis,
                            local: LocalSkinPrediction | None) -> bool:
        return bool(local is not None and local.status == "accepted"
                    and local.target == "vitiligo"
                    and "discoloration" in analysis.visible_features
                    and analysis.image_quality in {"fair", "good"})

    def _private_analysis(self, analysis: SkinAnalysis,
                          local: LocalSkinPrediction | None) -> dict[str, Any]:
        value = analysis.private_value(local)
        if local is None:
            return value
        corroborated = self._local_corroborates(analysis, local)
        value["local_fusion"] = ("corroborated" if corroborated
                                 else "disagreed" if local.status == "accepted"
                                 else "abstained")
        if corroborated:
            conditions = list(value.get("possible_conditions", []))
            if local.target not in {str(item).lower() for item in conditions}:
                conditions.append(local.target)
            value["possible_conditions"] = conditions[:3]
        return value

    def _local_only_results(self, purpose: str, correlation_id: str | None,
                            local: LocalSkinPrediction | None):
        """Publish private debug evidence, or a neutral validated screening result."""
        if local is None:
            return []
        private_value = {
            "image_quality": "unknown",
            "visible_features": [],
            "body_region": "visible close-up area",
            "possible_conditions": ([local.target] if local.status == "accepted" else []),
            "follow_up_topics": [],
            "local_classifier": local.private_value(),
            "local_fusion": "local_only",
        }
        if self._local_mode != "screening" or local.status != "accepted":
            return [self.result(
                "local_analysis", private_value, local.probability,
                Severity.INFO, "Local close-up classifier result is private and experimental",
                ttl=120.0, visibility=Visibility.AGENT_ONLY,
                correlation_id=correlation_id, source="local_skin_classifier")]
        region = "the visible arm" if purpose == "manual_arm_check" else "the visible area"
        public_key = "arm_check" if purpose == "manual_arm_check" else "visible_skin_change"
        public_value = ({"status": "succeeded", "source": "local_skin_classifier",
                         "finding_present": True, "body_region": region,
                         "visible_features": ["discoloration"], "image_quality": "unknown"}
                        if purpose == "manual_arm_check" else
                        {"body_region": region, "visible_features": ["discoloration"],
                         "image_quality": "unknown"})
        public = self.result(
            public_key, public_value, min(local.probability, 0.65), Severity.NOTICE,
            f"Possible pigment change on {region}", ttl=120.0,
            correlation_id=correlation_id, source="local_skin_classifier",
            location=region)
        private_key = "arm_analysis" if purpose == "manual_arm_check" else "analysis"
        private_value["body_region"] = region
        private_value["visible_features"] = ["discoloration"]
        private_value["follow_up_topics"] = ["duration", "spreading"]
        private = self.result(
            private_key, private_value, min(local.probability, 0.65), Severity.NOTICE,
            "Private experimental pigment-change hypothesis", ttl=120.0,
            visibility=Visibility.AGENT_ONLY, correlation_id=correlation_id,
            source="local_skin_classifier")
        return [public, private]

    def _submit(self, frame: np.ndarray | list[np.ndarray], stage: str, now: float,
                face_crop_available: bool = False,
                arm_crop_label: str | None = None,
                purpose: str | None = None) -> None:
        if self._pending is not None:
            return
        frames = frame if isinstance(frame, list) else [frame]
        frames = [np.array(item, copy=True) for item in frames]
        previous = self._preliminary
        purpose = purpose or ("passive_scan" if stage == "preliminary"
                              else "guided_closeup")
        use_cloud = bool(self.available and now >= self._next_allowed)
        if stage == "preliminary" and not use_cloud:
            return
        if stage == "closeup" and not use_cloud and not self._local_ready:
            return
        correlation_id = (self._correlation_id or uuid.uuid4().hex)
        self._pending_stage = stage
        self._pending_purpose = purpose
        self._pending_correlation_id = correlation_id
        if use_cloud:
            self._begin_request_diagnostics(
                stage, purpose, correlation_id,
                float(self.manual_retry_deadline) if purpose == "manual_arm_check" else None)
        if purpose == "manual_arm_check":
            self._arm_check_state = "pending"
            self._arm_check_last_error = None
        try:
            if stage == "closeup" and self._local_classifier is not None \
                    and self._local_ready:
                self._pending = self._executor.submit(
                    self._call_closeup, frames, previous, purpose, use_cloud)
            else:
                self._pending = self._executor.submit(
                    self._call_api, frames, stage, previous, face_crop_available,
                    arm_crop_label, purpose)
        except BaseException as exc:
            if use_cloud:
                self._finish_request_diagnostics(
                    "error", stage, error=type(exc).__name__)
            self._pending_stage = None
            raise
        if stage == "preliminary":
            self._last_scan = now

    def _reset_closeup(self) -> None:
        self._preliminary = None
        self._preliminary_at = 0.0
        self._awaiting_closeup = False
        self._sampling_closeup = False
        self._best_frame = None
        self._best_sharpness = -1.0
        self._correlation_id = None

    def _failure(self, now: float, exc: BaseException) -> None:
        self._failures += 1
        delay = min(float(self.backoff_max),
                    float(self.backoff_base) * (2 ** (self._failures - 1)))
        self._next_allowed = now + delay
        status = getattr(exc, "status", None)
        label = f"HTTP {status}" if status else type(exc).__name__
        print(f"[skin-vision] inference unavailable ({label}); retrying later")

    def _consume_pending(self, now: float):
        if self._pending is None or not self._pending.done():
            return []
        pending, stage = self._pending, self._pending_stage
        purpose = self._pending_purpose or "passive_scan"
        correlation_id = self._pending_correlation_id
        self._pending = None
        self._pending_stage = None
        self._pending_purpose = None
        self._pending_correlation_id = None
        local: LocalSkinPrediction | None = None
        cloud_error: BaseException | None = None
        try:
            completed = pending.result()
            if isinstance(completed, _CloseupOutcome):
                analysis = completed.analysis
                local = completed.local
                cloud_error = completed.cloud_error
            else:
                analysis = completed
        except BaseException as exc:  # noqa: BLE001
            self._failure(now, exc)
            if stage == "closeup":
                self._reset_closeup()
            if purpose == "manual_arm_check":
                self._arm_check_state = "unavailable"
                self._arm_check_last_error = type(exc).__name__
                if self._elicitation.test == "arm_check":
                    self._elicitation.clear()
                return [self.result(
                    "arm_check", {"status": "unavailable", "source": "nvidia_vlm"},
                    0.0, Severity.INFO,
                    "NVIDIA arm analysis was unavailable; the local camera check is separate",
                    ttl=30.0, source="nvidia_vlm", correlation_id=correlation_id)]
            return []
        if cloud_error is not None:
            self._failure(now, cloud_error)
        elif analysis is not None:
            self._failures = 0
            self._next_allowed = now
        if stage == "closeup" and analysis is None:
            self._reset_closeup()
            local_results = self._local_only_results(purpose, correlation_id, local)
            screening_succeeded = bool(
                self._local_mode == "screening" and local is not None
                and local.status == "accepted")
            if purpose == "manual_arm_check":
                self._arm_check_state = "succeeded" if screening_succeeded else "unavailable"
                self._arm_check_last_error = None if screening_succeeded else (
                    type(cloud_error).__name__ if cloud_error is not None else
                    (local.abstain_reason if local is not None else "no_backend"))
                if self._elicitation.test == "arm_check":
                    self._elicitation.clear()
                if not screening_succeeded:
                    local_results.insert(0, self.result(
                        "arm_check", {"status": "unavailable", "source": "skin_screening"},
                        0.0, Severity.INFO,
                        "Skin close-up analysis was unavailable or inconclusive",
                        ttl=30.0, source="skin_screening",
                        correlation_id=correlation_id))
            return local_results
        if stage == "preliminary":
            results = []
            cues = analysis.facial_cues
            positive = cues.positive() if cues is not None else {}
            if positive:
                labels = []
                for key, value in positive.items():
                    label = _FACIAL_LABELS[key]
                    labels.append(label if value == "yes" else f"{value} {label}")
                results.append(self.result(
                    "facial_appearance",
                    {"cues": positive, "image_quality": analysis.image_quality,
                     "confidence": cues.confidence},
                    confidence=cues.confidence, severity=Severity.INFO,
                    message="Visible facial appearance cues: " + ", ".join(labels),
                    ttl=120.0, source="nvidia_vlm",
                    quality={"poor": .2, "fair": .6, "good": .9}[analysis.image_quality]))
            if not analysis.finding_present:
                return results
            self._preliminary = analysis
            self._correlation_id = uuid.uuid4().hex
            self._preliminary_at = now
            self._awaiting_closeup = True
            value = analysis.private_value()
            value["closeup_seconds"] = float(self.closeup_seconds)
            results.append(self.result(
                "closeup_request", value,
                confidence=min(analysis.confidence, 0.55),
                severity=Severity.NOTICE, ttl=45.0,
                visibility=Visibility.AGENT_ONLY,
                correlation_id=self._correlation_id))
            return results
        if purpose == "manual_arm_check":
            self._arm_check_state = "succeeded"
            self._arm_check_last_error = None
            if self._elicitation.test == "arm_check":
                self._elicitation.clear()
            region = analysis.body_region or "the visible arm"
            corroborated = self._local_corroborates(analysis, local)
            value = {
                "status": "succeeded",
                "source": "nvidia_vlm",
                "finding_present": bool(analysis.finding_present),
                "body_region": region,
                "visible_features": list(analysis.visible_features),
                "image_quality": analysis.image_quality,
            }
            message = (f"Possible pigment change on {region}"
                       if corroborated else
                       f"NVIDIA arm VLM noticed a possible visible change on {region}: "
                       + ", ".join(analysis.visible_features[:3])
                       if analysis.finding_present else
                       "NVIDIA arm VLM did not identify a clear visible skin change in this image")
            public = self.result(
                "arm_check", value, min(analysis.confidence, 0.65),
                Severity.NOTICE if analysis.finding_present else Severity.INFO,
                message, ttl=120.0, source="nvidia_vlm",
                correlation_id=correlation_id,
                quality={"poor": .2, "fair": .6, "good": .9}[analysis.image_quality],
                location=region)
            private = self.result(
                "arm_analysis", self._private_analysis(analysis, local),
                min(analysis.confidence, 0.65),
                Severity.NOTICE if analysis.finding_present else Severity.INFO,
                message, ttl=120.0, visibility=Visibility.AGENT_ONLY,
                correlation_id=correlation_id, source="nvidia_vlm")
            self._reset_closeup()
            return [public, private]
        correlation_id = self._correlation_id
        self._reset_closeup()
        if not analysis.finding_present:
            return []
        region = analysis.body_region or "the visible area"
        features = ", ".join(analysis.visible_features[:3])
        corroborated = self._local_corroborates(analysis, local)
        public = self.result(
            "visible_skin_change",
            {"body_region": region,
             "visible_features": list(analysis.visible_features),
             "image_quality": analysis.image_quality},
            confidence=min(analysis.confidence, 0.65),
            severity=Severity.NOTICE,
            message=(f"Possible pigment change on {region}" if corroborated else
                     f"Possible visible skin change on {region}: {features}"),
            ttl=120.0, correlation_id=correlation_id,
            persistence=PersistencePolicy.EVENT,
            quality={"poor": .2, "fair": .6, "good": .9}[analysis.image_quality],
            location=region)
        private = self.result(
            "analysis", self._private_analysis(analysis, local),
            confidence=min(analysis.confidence, 0.65),
            severity=Severity.NOTICE, ttl=120.0,
            visibility=Visibility.AGENT_ONLY, correlation_id=correlation_id)
        return [public, private]

    def _collect_closeup(self, ctx: FrameContext) -> None:
        if ctx.timestamp < self._elicitation.started + float(self.closeup_positioning_delay):
            return
        score = _sharpness(ctx.frame)
        if score > self._best_sharpness:
            self._best_sharpness = score
            self._best_frame = ctx.frame.copy()

    def _best_arm_crop(self, ctx: FrameContext) -> tuple[np.ndarray, str] | None:
        """Largest bare-arm crop big enough for an enlarged detail panel."""
        from modules._util import arm_rois, arm_skin_region
        best = None
        for label, poly, anchors in arm_rois(ctx):
            region = arm_skin_region(ctx, poly, anchors)
            if region is None:
                continue
            patch, _, (x1, y1, x2, y2) = region
            short = min(x2 - x1, y2 - y1)
            if short < int(self.min_arm_crop_size):
                continue
            if best is None or short > best[0]:
                best = (short, patch.copy(), label)
        if best is None:
            return None
        return best[1], best[2]

    def process(self, ctx: FrameContext):
        """Advance asynchronous screening and return newly completed results."""
        self.start()  # direct/replay callers also receive idempotent preload
        self._refresh_local_load()
        now = ctx.timestamp
        if self._elicitation.active("arm_check", now=now):
            self._arm_check_state = "sampling"
        results = self._consume_pending(now)
        if not self.available and not self._local_ready:
            if self._elicitation.test == "arm_check" and now >= self._elicitation.until:
                self._arm_check_state = "unavailable"
                self._arm_check_last_error = "skin_backends_unconfigured"
                self._elicitation.clear()
            return results or None

        active = self._elicitation.active("skin_closeup", now=now)
        if active and self._awaiting_closeup:
            self._sampling_closeup = True
            self._collect_closeup(ctx)
        elif (self._sampling_closeup and self._elicitation.test == "skin_closeup"
              and now >= self._elicitation.until):
            frame = self._best_frame
            self._sampling_closeup = False
            self._awaiting_closeup = False
            self._elicitation.clear()
            if (frame is not None and self._pending is None
                    and (self._local_ready or now >= self._next_allowed)):
                self._submit(frame, "closeup", now)
            else:
                self._reset_closeup()

        # A user-initiated arm check reuses the sharpest-frame close-up
        # machinery without needing a preliminary finding first. The window
        # is owned by the voice agent / hotkey, so it expires on its own.
        if (self._elicitation.active("arm_check", now=now)
                and not self._sampling_closeup):
            self._arm_check_sampling = True
            self._collect_closeup(ctx)
        elif (self._arm_check_sampling and self._elicitation.test == "arm_check"
              and now >= self._elicitation.until):
            frame = self._best_frame
            self._arm_check_sampling = False
            self._best_frame = None
            self._best_sharpness = -1.0
            if (frame is not None and self._pending is None
                    and (self._local_ready or now >= self._next_allowed)):
                self._submit(frame, "closeup", now,
                             purpose="manual_arm_check")
            else:
                self._arm_check_state = "unavailable"
                self._arm_check_last_error = ("no_usable_frame" if frame is None
                                              else "cloud_backoff_active")
                self._elicitation.clear()

        if (self._awaiting_closeup and not self._sampling_closeup
                and now - self._preliminary_at > 60.0):
            self._reset_closeup()

        ready_for_scan = (self.available and ctx.person_present and self._pending is None
                          and not self._awaiting_closeup
                          and not self._arm_check_sampling
                          and self._preliminary is None
                          and now >= self._next_allowed
                          and now - self._last_scan >= float(self.scan_interval))
        if ready_for_scan:
            try:
                frame = ctx.frame.copy()
                face_crop = None
                if ctx.face is not None and isinstance(ctx.face.crop, np.ndarray) \
                        and ctx.face.crop.ndim == 3 and ctx.face.crop.size \
                        and min(ctx.face.crop.shape[:2]) >= int(self.min_face_crop_size):
                    face_crop = ctx.face.crop
                arm = self._best_arm_crop(ctx)
                arm_crop_label = arm[1] if arm is not None else None
                if face_crop is not None or arm is not None:
                    frame = _compose_preliminary_frame(
                        frame, face_crop, arm[0] if arm is not None else None,
                        arm_label=(arm_crop_label or "arm crop").upper())
                self._submit(frame, "preliminary", now, face_crop is not None,
                             arm_crop_label)
            except (ValueError, cv2.error) as exc:
                self._failure(now, exc)
        return results or None

    def close(self) -> None:
        """Cancel pending work without waiting on a slow network request."""
        if self._closed:
            return
        self._closed = True
        if self._pending is not None:
            self._pending.cancel()
        if self._local_load_future is not None:
            self._local_load_future.cancel()
        stopped = self._executor.shutdown(timeout=0.5)
        self._local_worker.shutdown(wait=False, cancel_futures=True, timeout=0.5)
        if self._local_classifier is not None:
            self._local_classifier.close()
        if not stopped:
            print("[skin-vision] cloud worker still finishing a bounded request; shutdown continues")
