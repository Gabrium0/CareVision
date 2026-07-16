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
import re
import threading
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
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
_SCHEMA_TEXT = """Return only the JSON object required by the supplied response
schema. Use enum values exactly. Use possible_conditions only for uncertain
internal hypotheses. Do not infer a condition when the image is unclear, and
do not include prose outside the JSON."""
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


def _compose_preliminary_frame(whole: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Combine whole-frame context and face detail into one NVIDIA-compatible image."""
    top = _fit_panel(whole, _COMPOSITE_WIDTH, _WHOLE_PANEL_HEIGHT)
    bottom = _fit_panel(face, _COMPOSITE_WIDTH, _FACE_PANEL_HEIGHT)
    _label_panel(top, "WHOLE FRAME")
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

    def private_value(self) -> dict[str, Any]:
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
        return value


class SkinVisionAPIError(RuntimeError):
    """Remote inference failed without retaining response or image data."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


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
    backoff_base = 15.0
    backoff_max = 900.0

    def __init__(self, **params):
        super().__init__(**params)
        self._key = nvidia_api_key()
        self._client = NvidiaVLMClient(self._key or "", self.endpoint, self.model,
                                       float(self.request_timeout), int(self.max_image_dim),
                                       int(self.jpeg_quality))
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="skin-vision")
        self._pending: Future | None = None
        self._pending_stage: str | None = None
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_status = "idle"
        self._diagnostic_current_stage: str | None = None
        self._diagnostic_started_at: float | None = None
        self._diagnostic_started_monotonic: float | None = None
        self._diagnostic_request_count = 0
        self._diagnostic_success_count = 0
        self._diagnostic_failure_count = 0
        self._diagnostic_last_attempt: dict[str, Any] | None = None
        self._last_scan = -1e9
        self._next_allowed = -1e9
        self._failures = 0
        self._preliminary: SkinAnalysis | None = None
        self._preliminary_at = 0.0
        self._awaiting_closeup = False
        self._sampling_closeup = False
        self._best_frame: np.ndarray | None = None
        self._best_sharpness = -1.0
        self._correlation_id: str | None = None
        self._elicitation = ElicitationState.instance()
        if self.consent and self._key:
            print(f"[skin-vision] cloud screening enabled (model {self.model})")
            CapabilityRegistry.instance().set("nvidia_skin", "cloud",
                                               CapabilityStatus.READY, "consented background VLM")
        elif self.consent:
            print("[skin-vision] consent given but NVIDIA_API_KEY is missing; disabled")
            CapabilityRegistry.instance().set("nvidia_skin", "cloud",
                                               CapabilityStatus.UNAVAILABLE, "credential unavailable")
        else:
            print("[skin-vision] cloud screening disabled (use --enable-cloud-skin)")
            CapabilityRegistry.instance().set("nvidia_skin", "cloud",
                                               CapabilityStatus.UNAVAILABLE, "consent off")

    @property
    def available(self) -> bool:
        """Whether this run has both explicit consent and credentials."""
        return bool(self.consent and self._key and self.endpoint and self.model)

    def _encode(self, frame: np.ndarray) -> bytes:
        return self._client.encode(frame)

    def diagnostics(self) -> dict[str, Any]:
        """Return the latest private, in-memory NVIDIA request diagnostics."""
        with self._diagnostic_lock:
            return {
                "available": self.available,
                "consent": bool(self.consent),
                "model": self.model,
                "status": self._diagnostic_status,
                "current_stage": self._diagnostic_current_stage,
                "request_started_at": self._diagnostic_started_at,
                "request_count": self._diagnostic_request_count,
                "success_count": self._diagnostic_success_count,
                "failure_count": self._diagnostic_failure_count,
                "last_attempt": copy.deepcopy(self._diagnostic_last_attempt),
            }

    def _begin_request_diagnostics(self, stage: str) -> None:
        with self._diagnostic_lock:
            self._diagnostic_status = "in_flight"
            self._diagnostic_current_stage = stage
            self._diagnostic_started_at = time.time()
            self._diagnostic_started_monotonic = time.monotonic()
            self._diagnostic_request_count += 1

    def _finish_request_diagnostics(self, status: str, stage: str, *,
                                    raw_model_content: Any = None,
                                    error: str | None = None,
                                    http_status: int | None = None) -> None:
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
            else:
                self._diagnostic_failure_count += 1
            self._diagnostic_last_attempt = {
                "status": status,
                "stage": stage,
                "started_at": started_at,
                "completed_at": completed_at,
                "latency_ms": latency_ms,
                "http_status": http_status,
                "error": error,
                "raw_model_content": copy.deepcopy(raw_model_content),
            }

    def _prompt(self, stage: str, previous: SkinAnalysis | None,
                face_crop_available: bool = False) -> str:
        if stage == "preliminary":
            task = ("Screen the visible person for an obvious possible skin change. "
                    "This is a low-confidence screening step, not a diagnosis. "
                    "Name the body region precisely enough to request a close-up. ")
            if face_crop_available:
                task += ("The single image is a labeled composite: the top panel is the "
                         "whole camera frame and the bottom panel is an enlarged face crop. "
                         "Use the whole-frame panel for skin context and report only directly "
                         "visible facial appearance cues from the face-crop panel. Do not "
                         "infer tiredness, illness, allergies, dehydration, or any diagnosis. ")
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
        return task + "\n\n" + _SCHEMA_TEXT

    def _call_api(self, jpeg: bytes | list[bytes], stage: str,
                  previous: SkinAnalysis | None,
                  face_crop_available: bool = False) -> SkinAnalysis:
        images = jpeg if isinstance(jpeg, list) else [jpeg]
        try:
            content = self._client.request(
                self._prompt(stage, previous, face_crop_available), images,
                response_format=_response_format(stage))
        except NvidiaVLMError as exc:
            self._finish_request_diagnostics(
                "error", stage, error=str(exc), http_status=exc.status)
            raise SkinVisionAPIError(str(exc), status=exc.status) from exc
        try:
            raw = _extract_json(content)
            analysis = validate_analysis(
                raw, float(self.min_confidence),
                allow_facial_cues=stage == "preliminary",
                min_facial_confidence=float(self.min_facial_confidence),
                face_crop_available=face_crop_available)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._finish_request_diagnostics(
                "invalid_response", stage, raw_model_content=content,
                error="invalid structured response")
            raise SkinVisionAPIError("invalid structured response") from exc
        self._finish_request_diagnostics(
            "success", stage, raw_model_content=content)
        return analysis

    def _submit(self, frame: np.ndarray | list[np.ndarray], stage: str, now: float,
                face_crop_available: bool = False) -> None:
        if self._pending is not None:
            return
        frames = frame if isinstance(frame, list) else [frame]
        jpeg = [self._encode(item) for item in frames]
        previous = self._preliminary
        self._pending_stage = stage
        self._begin_request_diagnostics(stage)
        try:
            self._pending = self._executor.submit(
                self._call_api, jpeg, stage, previous, face_crop_available)
        except BaseException as exc:
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
        self._pending = None
        self._pending_stage = None
        try:
            analysis = pending.result()
        except BaseException as exc:  # noqa: BLE001
            self._failure(now, exc)
            if stage == "closeup":
                self._reset_closeup()
            return []
        self._failures = 0
        self._next_allowed = now
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
        correlation_id = self._correlation_id
        self._reset_closeup()
        if not analysis.finding_present:
            return []
        region = analysis.body_region or "the visible area"
        features = ", ".join(analysis.visible_features[:3])
        public = self.result(
            "visible_skin_change",
            {"body_region": region,
             "visible_features": list(analysis.visible_features),
             "image_quality": analysis.image_quality},
            confidence=min(analysis.confidence, 0.65),
            severity=Severity.NOTICE,
            message=f"Possible visible skin change on {region}: {features}",
            ttl=120.0, correlation_id=correlation_id,
            persistence=PersistencePolicy.EVENT,
            quality={"poor": .2, "fair": .6, "good": .9}[analysis.image_quality],
            location=region)
        private = self.result(
            "analysis", analysis.private_value(),
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

    def process(self, ctx: FrameContext):
        """Advance asynchronous screening and return newly completed results."""
        if not self.available:
            return None
        now = ctx.timestamp
        results = self._consume_pending(now)

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
            if frame is not None and self._pending is None and now >= self._next_allowed:
                self._submit(frame, "closeup", now)
            else:
                self._reset_closeup()

        if (self._awaiting_closeup and not self._sampling_closeup
                and now - self._preliminary_at > 60.0):
            self._reset_closeup()

        ready_for_scan = (ctx.person_present and self._pending is None
                          and not self._awaiting_closeup
                          and self._preliminary is None
                          and now >= self._next_allowed
                          and now - self._last_scan >= float(self.scan_interval))
        if ready_for_scan:
            try:
                frame = ctx.frame.copy()
                face_crop_available = False
                if ctx.face is not None and isinstance(ctx.face.crop, np.ndarray) \
                        and ctx.face.crop.ndim == 3 and ctx.face.crop.size \
                        and min(ctx.face.crop.shape[:2]) >= int(self.min_face_crop_size):
                    frame = _compose_preliminary_frame(frame, ctx.face.crop)
                    face_crop_available = True
                self._submit(frame, "preliminary", now, face_crop_available)
            except (ValueError, cv2.error) as exc:
                self._failure(now, exc)
        return results or None

    def close(self) -> None:
        """Cancel pending work without waiting on a slow network request."""
        if self._pending is not None:
            self._pending.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
