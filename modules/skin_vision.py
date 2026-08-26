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
import math
import queue
import random
import re
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import Counter, deque
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from agent.env import nvidia_api_key
from core.context import FrameContext
from core.elicitation import ElicitationState
from core.events import PersistencePolicy, Result, Severity, Visibility
from core.registry import register
from modules.base import DetectionModule
from core.one_flight import DaemonOneFlight, detach_exception_context
from modules.local_skin_classifier import (
    LocalSkinPrediction,
    build_local_skin_classifier,
)
from integrations.nvidia_vlm import NvidiaVLMClient, NvidiaVLMError
from core.capabilities import CapabilityRegistry, CapabilityStatus


_FEATURES = {
    "redness", "discoloration", "swelling", "scaling", "blistering",
    "rash-like texture", "dryness", "lesion", "bruising", "irritation",
    "pigment_loss",
}
_TOPICS = {
    "itching", "pain", "duration", "spreading", "fever_unwell",
    "new_medication", "new_product_exposure", "blisters",
}
_QUALITY = {"poor", "fair", "good"}
_VISUAL_SOURCES = {"live_skin", "displayed_photo", "unclear"}
_APPEARANCE_LEVELS = {"none", "mild", "marked", "unclear"}
_NASAL_LEVELS = {"no", "yes", "unclear"}
# ORDER MATTERS: everything downstream slices `_FACIAL_KEYS[:-1]` to mean "the
# none/mild/marked cues", so `nasal_discharge_visible` (yes/no) must stay last
# and new appearance cues must be inserted before it. Adding a key here also
# extends the provider-enforced JSON schema and the validator loop for free.
_FACIAL_KEYS = (
    "under_eye_darkness", "under_eye_puffiness", "nose_redness",
    "cheek_redness", "lip_dryness", "forehead_shine", "eye_redness",
    "visible_skin_marking", "nasal_discharge_visible",
)
_FACIAL_LABELS = {
    "under_eye_darkness": "under-eye darkness",
    "under_eye_puffiness": "under-eye puffiness",
    "nose_redness": "nose redness",
    "cheek_redness": "cheek redness",
    "lip_dryness": "lip dryness",
    "forehead_shine": "forehead shine",
    "eye_redness": "eye redness",
    "visible_skin_marking": "visible skin marking",
    "nasal_discharge_visible": "visible nasal discharge",
}

# Where each cue is published so it lands on the detector card a presenter
# would actually look at, beside that detector's own heuristic reading.
#
# The `vlm_` key prefix is load-bearing, not cosmetic: agent/corroboration.py
# matches its follow-up rules with `key.startswith(rule.key)`, so a key named
# `lip_dryness_vlm` would satisfy the ("dry_lips", "lip_dryness") rule and let
# a model's guess trigger a *spoken* check-in that is meant to be driven only
# by the heuristic. Prefixing keeps model opinion out of that path entirely.
# `nasal_discharge_visible` is intentionally unrouted -- no detector owns it.
_CUE_ROUTES = {
    "lip_dryness":          ("dry_lips", "vlm_lip_dryness"),
    "under_eye_puffiness":  ("facial_swelling", "vlm_swelling"),
    "cheek_redness":        ("skin_color", "vlm_flushing"),
    "nose_redness":         ("skin_color", "vlm_nose_redness"),
    "under_eye_darkness":   ("drowsiness", "vlm_under_eye_darkness"),
    "forehead_shine":       ("sweating", "vlm_forehead_shine"),
    "eye_redness":          ("eye_redness", "vlm_eye_redness"),
    "visible_skin_marking": ("rash", "vlm_skin_marking"),
}
_QUALITY_SCORE = {"poor": .2, "fair": .6, "good": .9}
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
        "visual_source": {"type": "string", "enum": sorted(_VISUAL_SOURCES)},
        "sufficient_skin_visible": {"type": "boolean"},
        "finding_present": {"type": "boolean"},
        "visible_features": {
            "type": "array", "items": {"type": "string", "enum": sorted(_FEATURES)},
            "maxItems": 3,
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
    """Request parseable JSON while strict stage validation remains local."""
    del stage
    return {"type": "json_object"}


def _schema_contract(stage: str) -> str:
    """Return a compact prompt contract; strict validation remains local."""
    fields = [
        "image_quality: string, one of poor|fair|good",
        "visual_source: string, one of displayed_photo|live_skin|unclear",
        "sufficient_skin_visible: boolean",
        "finding_present: boolean",
        "visible_features: array of up to 3 from "
        + "|".join(sorted(_FEATURES)),
        "body_region: string",
        "confidence: number from 0 to 1",
        "possible_conditions: array of up to 3 short strings",
        "follow_up_topics: array of up to 5 from "
        + "|".join(sorted(_TOPICS)),
    ]
    if stage == "preliminary":
        fields.extend(
            f"{key}: string, one of none|mild|marked|unclear"
            for key in _FACIAL_KEYS[:-1])
        fields.extend((
            "nasal_discharge_visible: string, one of no|yes|unclear",
            "facial_cue_confidence: number from 0 to 1",
        ))
    return (
        "All keys below are required exactly once; use JSON booleans and "
        "numbers, and add no other keys.\n- " + "\n- ".join(fields)
    )


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
    forehead_shine: str = "unclear"
    eye_redness: str = "unclear"
    visible_skin_marking: str = "unclear"
    nasal_discharge_visible: str = "unclear"
    confidence: float = 0.0
    # True when the model did answer but below the spoken-path confidence bar
    # (or on a poor-quality frame). The readings stay legible to the console
    # via observed(); positive() stays empty so nothing reaches the agent.
    gated: bool = False

    def positive(self) -> dict[str, str]:
        """Return only affirmative observable cues, omitting none/unclear values.

        Empty while `gated`: this feeds the spoken/agent path, which must never
        open a conversation about a low-confidence model guess.
        """
        if self.gated:
            return {}
        out = {}
        for key in _FACIAL_KEYS[:-1]:
            value = getattr(self, key)
            if value in ("mild", "marked"):
                out[key] = value
        if self.nasal_discharge_visible == "yes":
            out["nasal_discharge_visible"] = "yes"
        return out

    def observed(self) -> dict[str, str]:
        """Every cue the model committed to, negatives included.

        `positive()` keeps only affirmatives because the agent must never
        open a conversation about something it did not see. A console has the
        opposite need: a confident "no dryness" is a real answer, and showing
        it beside the heuristic's own reading is the whole point. Only
        "unclear" — the model declining to answer — is withheld.
        """
        return {key: getattr(self, key) for key in _FACIAL_KEYS
                if getattr(self, key) != "unclear"}


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
    visual_source: str = "live_skin"

    def private_value(self, local: LocalSkinPrediction | None = None) -> dict[str, Any]:
        """Return JSON-safe private context for the agent."""
        value = {
            "image_quality": self.image_quality,
            "visual_source": self.visual_source,
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
                 retryable: bool = False, *, kind: str = "provider",
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = bool(retryable)
        self.kind = _bounded_text(kind, 40) or "provider"
        self.retry_after = retry_after


@dataclass(frozen=True)
class _CloseupOutcome:
    """Independent cloud/local outcomes from one close-up frame."""

    analysis: SkinAnalysis | None
    local: LocalSkinPrediction | None
    cloud_error: BaseException | None = None


@dataclass
class _QueuedManual:
    """Captured manual request waiting for the single NVIDIA lane."""

    frame: np.ndarray
    local_frame: np.ndarray | None
    arm_crop_label: str | None
    paired_context: bool
    capture_mode: str
    correlation_id: str
    arm_attempt: int
    queued_at: float
    deadline_monotonic: float
    view_labels: tuple[str, ...]
    reservation_token: int | None


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
                    closed = self._closed
                del item, future, fn, args
                if closed:
                    return
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(detach_exception_context(exc))
            with self._lock:
                closed = self._closed
            # A daemon blocked in queue.get() otherwise retains the last
            # task's frame arrays through these loop locals indefinitely.
            del item, future, fn, args
            if closed:
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


def _validate_schema_shape(raw: dict[str, Any], *,
                           allow_facial_cues: bool) -> None:
    """Enforce the provider contract before safety-sensitive normalization."""
    schema = _PRELIMINARY_SCHEMA if allow_facial_cues else _CLOSEUP_SCHEMA
    properties = schema["properties"]
    required = set(schema["required"])
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("missing required fields: " + ", ".join(missing[:5]))
    extras = sorted(set(raw) - set(properties))
    if extras:
        raise ValueError("unexpected fields: " + ", ".join(extras[:5]))
    for name, spec in properties.items():
        value = raw[name]
        value_type = spec.get("type")
        if value_type == "boolean":
            valid_type = type(value) is bool
        elif value_type == "number":
            valid_type = (isinstance(value, (int, float))
                          and not isinstance(value, bool)
                          and math.isfinite(float(value)))
        elif value_type == "string":
            valid_type = isinstance(value, str)
        elif value_type == "array":
            valid_type = isinstance(value, list)
        else:
            valid_type = False
        if not valid_type:
            raise ValueError(f"invalid type for {name}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"invalid enum for {name}")
        if value_type == "number":
            if value < spec.get("minimum", value) \
                    or value > spec.get("maximum", value):
                raise ValueError(f"out-of-range value for {name}")
        elif value_type == "string" and len(value) > spec.get(
                "maxLength", len(value)):
            raise ValueError(f"value too long for {name}")
        elif value_type == "array":
            if len(value) > spec.get("maxItems", len(value)):
                raise ValueError(f"too many items for {name}")
            item_spec = spec.get("items", {})
            for item in value:
                if item_spec.get("type") == "string" \
                        and not isinstance(item, str):
                    raise ValueError(f"invalid item type for {name}")
                if "enum" in item_spec and item not in item_spec["enum"]:
                    raise ValueError(f"invalid item enum for {name}")
                if isinstance(item, str) and len(item) > item_spec.get(
                        "maxLength", len(item)):
                    raise ValueError(f"item too long for {name}")


def validate_analysis(raw: Any, min_confidence: float = 0.35, *,
                      allow_facial_cues: bool = False,
                      min_facial_confidence: float = 0.45,
                      face_crop_available: bool = True,
                      strict_schema: bool = False) -> SkinAnalysis:
    """Validate and normalize the model JSON; reject unsafe loose structures."""
    if not isinstance(raw, dict):
        raise ValueError("skin response must be a JSON object")
    if strict_schema:
        _validate_schema_shape(
            raw, allow_facial_cues=allow_facial_cues)
    quality = _bounded_text(raw.get("image_quality"), 10).lower()
    if quality not in _QUALITY:
        raise ValueError("invalid image_quality")
    visual_source = _bounded_text(raw.get("visual_source"), 20).lower()
    if visual_source not in _VISUAL_SOURCES:
        raise ValueError("invalid visual_source")
    sufficient = raw.get("sufficient_skin_visible") is True
    finding = raw.get("finding_present") is True
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid confidence") from exc
    features = _string_list(raw.get("visible_features"), _FEATURES, limit=3)
    if strict_schema and {"pigment_loss", "bruising"}.issubset(features):
        raise ValueError("pigment_loss and bruising are mutually exclusive")
    topics = _string_list(raw.get("follow_up_topics"), _TOPICS, limit=5)
    conditions = _string_list(raw.get("possible_conditions"), limit=3)
    region = _bounded_text(raw.get("body_region") or "visible skin", 60)
    if strict_schema and finding and (
            confidence < min_confidence or not features):
        # Never normalize a provider-asserted positive into a reassuring
        # negative solely because its asserted evidence is unsupported.
        # Poor/insufficient imagery is handled as explicitly inconclusive by
        # the manual quality gates before any clear result can be emitted.
        raise ValueError("positive finding lacks usable supporting evidence")
    # A deliberately displayed full photo (the user holding a phone to the
    # camera) is often rated "poor" for screen glare/reflections even when the
    # change is plainly visible; trust a confident, evidenced finding there
    # rather than discarding it. Live skin stays conservative on poor quality.
    quality_blocks = quality == "poor" and visual_source != "displayed_photo"
    finding = bool(finding and sufficient and not quality_blocks
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
        # No usable face crop means the model was describing a face it could
        # not actually see, so those values are discarded outright. A low
        # confidence or poor frame is different: the model did look and did
        # commit to an answer, so the reading is kept for the console and only
        # `gated` out of the spoken path. Blanking it here used to leave a
        # manual scan showing nothing at all.
        gated = False
        if not face_crop_available:
            values = {key: "unclear" for key in _FACIAL_KEYS[:-1]}
            nasal = "unclear"
            facial_confidence = 0.0
        elif quality == "poor" or facial_confidence < min_facial_confidence:
            gated = True
        facial_cues = FacialCues(**values, nasal_discharge_visible=nasal,
                                  confidence=round(facial_confidence, 3),
                                  gated=gated)
    return SkinAnalysis(
        quality, sufficient, finding, features, region,
        round(confidence, 3), conditions, topics, facial_cues,
        visual_source=visual_source)


def _normalize_rescued(raw: dict[str, Any], stage: str) -> dict[str, Any]:
    """Make a prose-derived rescue object survivable without softening safety.

    The rescue path re-encodes an observation the vision model already made,
    against a contract of twenty fields for `preliminary`. `_validate_schema_shape`
    rejects on any extra key and any missing key, so a single chatty addition or
    one forgotten cue used to discard a legitimate screen outright.

    Two bounded liberties are taken here and nowhere else:

    * unknown keys are dropped -- an extra key is the model being talkative,
      not the screen being wrong; and
    * absent *facial cue* keys are filled with the schema's own neutral values
      ("unclear" / 0.0).

    Everything that decides whether something was seen -- `finding_present`,
    `sufficient_skin_visible`, `visible_features`, `confidence`,
    `image_quality`, `visual_source` -- is deliberately NOT filled. A rescue
    that omits those still fails validation. This keeps the transform
    one-directional: it can only ever yield a result that is less alarming than
    the model's own words, never more. A filled cue is `unclear`, which both
    `FacialCues.positive()` and `.observed()` already discard, so it reaches
    neither the spoken path nor the console.
    """
    schema = _PRELIMINARY_SCHEMA if stage == "preliminary" else _CLOSEUP_SCHEMA
    cleaned = {key: value for key, value in raw.items()
               if key in schema["properties"]}
    if stage != "preliminary":
        return cleaned
    for key in _FACIAL_KEYS[:-1]:
        if key not in cleaned:
            cleaned[key] = "unclear"
    cleaned.setdefault("nasal_discharge_visible", "unclear")
    cleaned.setdefault("facial_cue_confidence", 0.0)
    return cleaned


def _extract_json(content: Any) -> dict:
    """Salvage the JSON object from the model reply.

    The model reliably returns the schema object but sometimes wraps it in a
    markdown fence, a leading disclaimer, or appends a stray trailing character
    (e.g. ``{...}.``). Extract the first ``{`` through the last ``}`` — matching
    the tolerant approach in ``modules/scene_vision.py`` — instead of requiring
    the whole reply to be bare JSON. A reply with no object at all still raises
    ``ValueError`` so it is classified as ``json_parse`` upstream.
    """
    if isinstance(content, list):
        content = "".join(str(p.get("text", "")) for p in content
                          if isinstance(p, dict))
    text = str(content or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("response was not a JSON object")
    parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("skin response must be a JSON object")
    return parsed


def _content_text(content: Any) -> str:
    """Extract the raw text from provider content (str or list-of-dict)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict))
    return str(content) if content is not None else ""


def _response_content_diagnostic(content: Any) -> dict[str, Any]:
    """Describe provider content structurally without retaining its text."""
    def text_shape(value: str) -> dict[str, Any]:
        stripped = value.strip()
        starts_object = stripped.startswith("{")
        ends_object = stripped.endswith("}")
        has_start = "{" in stripped
        has_end = "}" in stripped
        if not stripped:
            leading_kind = "empty"
        elif starts_object:
            leading_kind = "object"
        elif stripped.startswith("["):
            leading_kind = "array"
        elif stripped.startswith("```"):
            leading_kind = "markdown_fence"
        else:
            leading_kind = "prose"
        return {
            "character_count": len(value),
            "line_count": value.count("\n") + (1 if value else 0),
            "leading_kind": leading_kind,
            "starts_with_object": starts_object,
            "ends_with_object": ends_object,
            "has_object_start": has_start,
            "has_object_end": has_end,
            "has_object_bounds": bool(
                has_start and has_end and value.rfind("}") > value.find("{")),
            "has_markdown_fence": "```" in value,
        }

    if isinstance(content, str):
        return {"content_type": "str", **text_shape(content)}
    if isinstance(content, list):
        text_parts = [
            part.get("text")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        joined = "".join(text_parts)
        return {
            "content_type": "list",
            "part_count": len(content),
            "text_part_count": len(text_parts),
            "non_text_part_count": len(content) - len(text_parts),
            **text_shape(joined),
        }
    if isinstance(content, dict):
        return {
            "content_type": "dict",
            "field_count": len(content),
            "leading_kind": "object",
        }
    if content is None:
        return {"content_type": "none", "leading_kind": "empty"}
    return {
        "content_type": type(content).__name__,
        "leading_kind": "unsupported",
    }


def _response_diagnostic_hint(raw: Any,
                              shape: dict[str, Any]) -> str:
    """Return an actionable categorical cause, never provider content."""
    if shape.get("leading_kind") == "empty":
        return "provider_empty_content"
    if raw is not None:
        return "provider_json_failed_schema_validation"
    if shape.get("has_object_start") != shape.get("has_object_end"):
        return "provider_partial_json"
    if shape.get("has_object_bounds"):
        return "provider_malformed_json"
    if shape.get("leading_kind") in {"prose", "markdown_fence", "array"}:
        return "provider_non_json_text"
    return "provider_content_not_json_object"


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
    # Shortest gap between console-requested scans while the provider is
    # returning non-retryable errors, so repeated clicks re-probe rather than
    # burst. Ignored entirely once the provider looks healthy again.
    manual_probe_interval = 30.0
    manual_retry_deadline = 60.0
    manual_max_attempts = 3
    manual_attempt_timeout = 20.0
    # The preliminary stage asks for more than twice the fields of a close-up
    # (`_PRELIMINARY_SCHEMA` adds eight cue enums, nasal, and a confidence), so
    # it needs at least the manual path's attempt budget rather than the bare
    # two-attempt/no-deadline fallback it used to inherit. A smaller token
    # budget than the old 700 keeps generation inside one attempt timeout;
    # `finish_reason == "length"` still escalates to 700.
    preliminary_deadline = 75.0
    preliminary_max_attempts = 3
    preliminary_attempt_timeout = 25.0
    preliminary_max_tokens = 520
    max_inline_image_bytes = 174080
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
                                       int(self.jpeg_quality),
                                       int(self.max_inline_image_bytes))
        self._executor = _DaemonOneFlight("skin-vision")
        self._closed = False
        self._pending: Future | None = None
        self._pending_stage: str | None = None
        self._pending_purpose: str | None = None
        self._pending_correlation_id: str | None = None
        self._pending_capture_mode: str | None = None
        self._pending_view_labels: tuple[str, ...] = ()
        self._pending_image_count = 0
        self._pending_arm_attempt = 0
        self._pending_deadline_monotonic: float | None = None
        self._pending_reservation_token: int | None = None
        self._pending_cancel_event: threading.Event | None = None
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
        self._diagnostic_capture_mode: str | None = None
        self._diagnostic_view_labels: tuple[str, ...] = ()
        self._diagnostic_image_count = 0
        self._diagnostic_encoded_bytes = 0
        self._diagnostic_attempts: list[dict[str, Any]] = []
        self._diagnostic_history: deque[dict[str, Any]] = deque(maxlen=20)
        self._diagnostic_last_manual_failure: dict[str, Any] | None = None
        self._diagnostic_by_purpose: dict[str, dict[str, Any]] = {}
        # Set by request_scan() and cleared only once a scan is actually
        # submitted, so a console request survives the frames where a gate
        # (no person yet, a request still in flight) is momentarily closed.
        self._manual_scan_requested = False
        # Provider health as last observed on the passive path, so a console
        # request can be spaced sensibly and the reason shown on the card.
        self._last_failure_retryable = True
        self._last_failure_category: str | None = None
        self._last_failure_status: int | None = None
        self._last_failure_at = 0.0
        self._last_scan = -1e9
        self._next_allowed = -1e9
        self._next_allowed_monotonic = -1e9
        self._manual_next_allowed = -1e9
        self._manual_authorization_blocked = False
        self._failures = 0
        self._preliminary: SkinAnalysis | None = None
        self._preliminary_at = 0.0
        self._awaiting_closeup = False
        self._sampling_closeup = False
        self._arm_check_sampling = False
        self._arm_check_window_id: float | None = None
        self._arm_check_state = "idle"
        self._arm_check_last_error: str | None = None
        self._arm_check_correlation_id: str | None = None
        self._arm_check_attempt = 0
        self._arm_check_capture_mode: str | None = None
        self._arm_check_view_labels: tuple[str, ...] = ()
        self._arm_check_image_count = 0
        self._manual_queue: deque[_QueuedManual] = deque()
        self._queued_manual: _QueuedManual | None = None
        self._best_arm_frame: np.ndarray | None = None
        self._best_arm_context: np.ndarray | None = None
        self._best_arm_label: str | None = None
        self._best_arm_sharpness = -1.0
        self._best_arm_fallback: np.ndarray | None = None
        self._best_arm_fallback_sharpness = -1.0
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

    def _routed_cue_results(self, cues: FacialCues, quality: str) -> list[Result]:
        """Publish each observed cue under the detector that owns its subject.

        Deliberately bypasses self.result(), which hardcodes module=self.name;
        modules/replay_events.py overrides `module` the same way. That also
        means re-doing the confidence sanitising self.result() would have
        given us. `source="nvidia_vlm"` keeps provenance unambiguous so the
        console can never present this as the heuristic having fired.
        """
        confidence = float(cues.confidence)
        if not math.isfinite(confidence):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        out = []
        for cue, value in cues.observed().items():
            route = _CUE_ROUTES.get(cue)
            if route is None:
                continue
            module, key = route
            out.append(Result(
                module=module, key=key,
                value={"cue": cue, "value": value},
                confidence=confidence, severity=Severity.INFO,
                message=f"{_FACIAL_LABELS[cue].capitalize()}: {value} (VLM)",
                ttl=120.0, source="nvidia_vlm",
                quality=_QUALITY_SCORE.get(quality)))
        return out

    def provider_health(self) -> dict[str, Any]:
        """Bounded, LAN-safe view of cloud availability for the console.

        diagnostics() carries correlation ids, provider request ids and model
        detail and is only ever exposed on the private debug port. This is the
        subset a caregiver console may show beside the scan button, so that a
        provider outage reads as an outage instead of a dead button.
        """
        waiting = max(0.0, float(self._next_allowed) - time.time())
        return {
            "available": bool(self.available),
            "ok": bool(self.available and self._failures == 0),
            "consecutive_failures": int(self._failures),
            "retry_in_seconds": round(waiting, 1),
            "reason": self._last_failure_category,
            "http_status": self._last_failure_status,
            "retryable": bool(self._last_failure_retryable),
        }

    def request_scan(self) -> bool:
        """Ask for a face scan on the next frame that can carry one.

        A 60s wait is unremarkable in a home and reads as the system doing
        nothing in front of an audience. Beyond clearing the passive cadence
        this also lifts the provider backoff for this one request and latches
        the intent: after a run of provider failures `_next_allowed` can sit
        minutes in the future, which used to swallow a console request without
        a trace. `_failures` is deliberately left alone so passive backoff
        resumes its normal curve afterwards.

        Consent, a person being present and no request already being in flight
        still apply, so this asks for a scan rather than forcing one.
        """
        if not self.available:
            return False
        self._manual_scan_requested = True
        self._last_scan = -1e9
        if self._last_failure_retryable:
            self._next_allowed = -1e9
        else:
            # A non-retryable provider error (a degraded model, a rejected
            # request, bad credentials) fails again the instant it is asked,
            # so clicking repeatedly must not turn into a burst of doomed
            # calls. Space the re-probe instead of either hammering or making
            # the operator sit out a backoff that can reach 15 minutes.
            self._next_allowed = min(
                self._next_allowed,
                self._last_failure_at + float(self.manual_probe_interval))
        return True

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
        return self._client.encode(
            frame, max_inline_image_bytes=int(self.max_inline_image_bytes))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        """Return a deterministic nearest-rank percentile for a small sample."""
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        index = max(0, min(len(ordered) - 1,
                           int(math.ceil(percentile * len(ordered))) - 1))
        return round(ordered[index], 1)

    def _safe_diagnostic_text(self, value: Any, limit: int = 160) -> str | None:
        """Bound diagnostic text and remove the configured credential."""
        text = _bounded_text(value, limit)
        if self._key:
            text = text.replace(self._key, "[redacted]")
        return text or None

    def _metrics_snapshot_locked(self, purpose: str) -> dict[str, Any]:
        stats = self._diagnostic_by_purpose.get(purpose, {})
        completed = int(stats.get("completed", 0))
        successes = int(stats.get("successes", 0))
        retried = int(stats.get("retried", 0))
        latencies = list(stats.get("latencies_ms", ()))
        return {
            "completed": completed,
            "successes": successes,
            "failures": int(stats.get("failures", 0)),
            "completion_rate": round(successes / max(1, completed), 3),
            "first_attempt_success": int(
                stats.get("first_attempt_successes", 0)),
            "first_attempt_success_rate": round(
                int(stats.get("first_attempt_successes", 0))
                / max(1, completed), 3),
            "retried": retried,
            "retry_recovery": int(stats.get("retry_recoveries", 0)),
            "retry_recovery_rate": round(
                int(stats.get("retry_recoveries", 0)) / max(1, retried), 3),
            "p50_latency_ms": self._percentile(latencies, 0.50),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "encoded_bytes": int(stats.get("last_encoded_bytes", 0)),
            "failure_categories": dict(
                sorted(stats.get("failure_categories", {}).items())),
        }

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
            completed = (self._diagnostic_success_count
                         + self._diagnostic_failure_count)
            now_monotonic = time.monotonic()
            queued = self._queued_manual
            purposes = sorted(set(self._diagnostic_by_purpose) | {
                str(record.get("purpose", "unknown"))
                for record in self._diagnostic_history})
            by_purpose = {
                purpose: [
                    copy.deepcopy(record)
                    for record in self._diagnostic_history
                    if record.get("purpose") == purpose
                ]
                for purpose in purposes
            }
            metrics = {
                purpose: self._metrics_snapshot_locked(purpose)
                for purpose in purposes
            }
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
                                      max(1, completed), 3),
                "consecutive_failures": self._failures,
                "circuit_state": (
                    "open"
                    if now_monotonic < self._next_allowed_monotonic
                    else "closed"),
                "retry_after_seconds": round(max(
                    0.0, self._next_allowed_monotonic - now_monotonic), 1),
                "attempt": self._diagnostic_attempt,
                "payload_mode": self._diagnostic_payload_mode,
                "deadline_remaining_seconds": (round(max(
                    0.0, self._diagnostic_deadline_at - time.time()), 1)
                    if self._diagnostic_deadline_at else None),
                "arm_check": {
                    "state": self._arm_check_state,
                    "last_error": self._arm_check_last_error,
                    "queued": queued is not None,
                    "queue_depth": max(
                        len(self._manual_queue),
                        1 if queued is not None else 0),
                    "queued_for_seconds": (
                        round(max(0.0, now_monotonic - queued.queued_at), 1)
                        if queued is not None else None),
                    "deadline_remaining_seconds": (
                        round(max(0.0, queued.deadline_monotonic
                                  - now_monotonic), 1)
                        if queued is not None else None),
                    "attempt": self._arm_check_attempt,
                    "capture_mode": self._arm_check_capture_mode,
                    "view_count": len(self._arm_check_view_labels),
                    "view_labels": list(self._arm_check_view_labels),
                    "image_count": self._arm_check_image_count,
                    "correlation_id": (self._diagnostic_correlation_id
                                       if self._diagnostic_purpose == "manual_arm_check"
                                       else self._arm_check_correlation_id),
                },
                "local_classifier": local,
                "last_attempt": copy.deepcopy(self._diagnostic_last_attempt),
                "last_manual_failure": copy.deepcopy(
                    self._diagnostic_last_manual_failure),
                "recent_requests": copy.deepcopy(
                    list(self._diagnostic_history)),
                "recent_requests_by_purpose": by_purpose,
                "metrics_by_purpose": metrics,
                "manual_metrics": copy.deepcopy(
                    metrics.get("manual_arm_check", {
                        "completed": 0, "successes": 0, "failures": 0,
                        "completion_rate": 0.0, "first_attempt_success": 0,
                        "first_attempt_success_rate": 0.0, "retried": 0,
                        "retry_recovery": 0, "retry_recovery_rate": 0.0,
                        "p50_latency_ms": None, "p95_latency_ms": None,
                        "encoded_bytes": 0,
                        "failure_categories": {},
                    })),
                "coordinator": self._client.coordinator_diagnostics(),
            }

    def _begin_request_diagnostics(self, stage: str, purpose: str,
                                   correlation_id: str,
                                   deadline_monotonic: float | None = None, *,
                                   capture_mode: str | None = None,
                                   view_labels: tuple[str, ...] = (),
                                   image_count: int = 0,
                                   started_monotonic: float | None = None) -> None:
        now_monotonic = time.monotonic()
        started_monotonic = (now_monotonic if started_monotonic is None
                             else float(started_monotonic))
        with self._diagnostic_lock:
            self._diagnostic_status = "in_flight"
            self._diagnostic_current_stage = stage
            self._diagnostic_started_at = (
                time.time() - max(0.0, now_monotonic - started_monotonic))
            self._diagnostic_started_monotonic = started_monotonic
            self._diagnostic_request_count += 1
            self._diagnostic_purpose = purpose
            self._diagnostic_correlation_id = correlation_id
            self._diagnostic_attempt = 0
            self._diagnostic_payload_mode = "normal"
            self._diagnostic_deadline_at = (
                time.time() + max(0.0, deadline_monotonic - now_monotonic)
                if deadline_monotonic is not None else None)
            self._diagnostic_capture_mode = capture_mode
            self._diagnostic_view_labels = tuple(view_labels)
            self._diagnostic_image_count = max(0, int(image_count))
            self._diagnostic_encoded_bytes = 0
            self._diagnostic_attempts = []

    def _record_attempt_diagnostics(
            self, *, attempt: int, outcome: str, payload_mode: str,
            structured: bool, max_tokens: int, latency_ms: float,
            encoded_bytes: int = 0, http_status: int | None = None,
            finish_reason: str | None = None, request_id: str | None = None,
            retry_after: float | None = None, retryable: bool = False,
            category: str | None = None, error: str | None = None,
            validation: dict[str, Any] | None = None,
            poll_count: int = 0, queue_ms: float = 0.0) -> None:
        """Retain categorical attempt metadata, never request/response media."""
        record = {
            "attempt": max(1, int(attempt)),
            "outcome": _bounded_text(outcome, 40),
            "payload_mode": _bounded_text(payload_mode, 40),
            "structured": bool(structured),
            "max_tokens": max(1, int(max_tokens)),
            "latency_ms": round(max(0.0, float(latency_ms)), 1),
            "encoded_bytes": max(0, int(encoded_bytes)),
            "http_status": http_status,
            "finish_reason": self._safe_diagnostic_text(finish_reason, 40),
            "request_id": self._safe_diagnostic_text(request_id, 160),
            "retry_after_seconds": (
                round(max(0.0, float(retry_after)), 2)
                if retry_after is not None else None),
            "retryable": bool(retryable),
            "failure_category": self._safe_diagnostic_text(category, 40),
            "error": self._safe_diagnostic_text(error, 160),
            "validation": copy.deepcopy(validation),
            "poll_count": max(0, int(poll_count)),
            "queue_ms": round(max(0.0, float(queue_ms)), 1),
        }
        with self._diagnostic_lock:
            self._diagnostic_attempt = record["attempt"]
            self._diagnostic_payload_mode = record["payload_mode"]
            self._diagnostic_encoded_bytes = max(
                self._diagnostic_encoded_bytes, record["encoded_bytes"])
            self._diagnostic_attempts.append(record)

    def _note_rescue_outcome(self, outcome: str) -> None:
        """Tag the most recent attempt with how its prose rescue ended."""
        if not outcome:
            return
        with self._diagnostic_lock:
            if not self._diagnostic_attempts:
                return
            record = self._diagnostic_attempts[-1]
            validation = record.get("validation")
            if not isinstance(validation, dict):
                validation = {}
                record["validation"] = validation
            validation["rescue_outcome"] = _bounded_text(outcome, 40)

    def _update_metrics_locked(self, logical: dict[str, Any]) -> None:
        """Update bounded aggregate counters from one completed logical request."""
        purpose = str(logical.get("purpose") or "unknown")
        attempts = logical.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        stats = self._diagnostic_by_purpose.setdefault(purpose, {
            "completed": 0,
            "successes": 0,
            "failures": 0,
            "first_attempt_successes": 0,
            "retried": 0,
            "retry_recoveries": 0,
            "latencies_ms": deque(maxlen=20),
            "last_encoded_bytes": 0,
            "failure_categories": Counter(),
        })
        stats["completed"] += 1
        stats["last_encoded_bytes"] = max(
            0, int(logical.get("encoded_bytes") or 0))
        stats["latencies_ms"].append(float(logical.get("latency_ms") or 0.0))
        was_retried = len(attempts) > 1
        if was_retried:
            stats["retried"] += 1
        if logical.get("status") == "success":
            stats["successes"] += 1
            if len(attempts) <= 1:
                stats["first_attempt_successes"] += 1
            if was_retried:
                stats["retry_recoveries"] += 1
        else:
            stats["failures"] += 1
            category = str(logical.get("failure_category")
                           or logical.get("status") or "provider")
            stats["failure_categories"][category] += 1
            if purpose == "manual_arm_check":
                self._diagnostic_last_manual_failure = copy.deepcopy(logical)

    def _finish_request_diagnostics(self, status: str, stage: str, *,
                                    error: str | None = None,
                                    http_status: int | None = None,
                                    validation: dict[str, Any] | None = None,
                                    repair_attempted: bool = False,
                                    retryable: bool = False,
                                    terminal_reason: str | None = None,
                                    failure_category: str | None = None) -> None:
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
            attempts = copy.deepcopy(self._diagnostic_attempts)
            category = self._safe_diagnostic_text(
                failure_category or terminal_reason, 40)
            logical = {
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
                "encoded_bytes": self._diagnostic_encoded_bytes,
                "capture_mode": self._diagnostic_capture_mode,
                "view_count": len(self._diagnostic_view_labels),
                "view_labels": list(self._diagnostic_view_labels),
                "image_count": self._diagnostic_image_count,
                "terminal_reason": terminal_reason,
                "failure_category": category,
                "attempts": attempts,
            }
            self._diagnostic_last_attempt = logical
            self._diagnostic_history.append(copy.deepcopy(logical))
            self._update_metrics_locked(logical)

    def _prompt(self, stage: str, previous: SkinAnalysis | None,
                face_crop_available: bool = False,
                arm_crop_label: str | None = None,
                purpose: str = "passive_scan",
                paired_context: bool = False, *,
                include_contract: bool = True) -> str:
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
            if purpose == "manual_arm_check":
                label = arm_crop_label or "candidate close-up"
                task = (
                    "Validate and inspect this manual skin-check image. It may show "
                    "either live bare forearm/upper-arm skin or a phone screen displaying "
                    "a close-up skin photo. Set visual_source to live_skin only for skin "
                    "physically in front of the camera, displayed_photo only when the "
                    "skin is inside a phone display, and unclear otherwise. For a "
                    "displayed photo, inspect the photo content rather than rejecting it "
                    "for not being live; ignore the phone bezel, gallery controls, glare, "
                    "reflections, and other UI. Set sufficient_skin_visible false when "
                    "glare, blur, scale, obstruction, or non-skin content prevents a "
                    "reliable visual screen. For live_skin, require a prominent bare "
                    "forearm or upper arm; reject a face, neck, torso, hand only, clothing, "
                    "or an uncertain body region. For displayed_photo, use the photographed "
                    "body region when clear, otherwise use 'skin area in displayed photo'. "
                    "Be conservative and non-diagnostic. "
                    f"The capture source describes it as: {label}. ")
                if paired_context:
                    task += (
                        "The single image is a labeled composite from one camera moment. "
                        "The top panel is the complete camera frame and may contain a "
                        "phone displaying the actual skin photo. The bottom panel is an "
                        f"enlarged pose crop described as {label}. Inspect both panels "
                        "before deciding. A visible change that appears only in the top "
                        "whole-frame panel is still relevant; do not assume the bottom "
                        "arm crop is the intended evidence. ")
            else:
                task = ("Inspect this user-provided close-up for visible skin changes. "
                        "Be conservative and non-diagnostic.")
            if previous is not None:
                task += (f" The preliminary frame indicated {', '.join(previous.visible_features)} "
                         f"around {previous.body_region}.")
        if include_contract:
            prompt = (
                "MACHINE-READABLE VISUAL ATTRIBUTE EXTRACTION. Return exactly "
                "one raw JSON object and nothing else. The first character "
                "must be { and the last character must be }. Never output an "
                "explanation, disclaimer, diagnosis, recommendation, prose, "
                "or markdown. Record only directly visible attributes; this "
                "is not a request for medical advice. "
                + _SCHEMA_TEXT
                + "\nJSON contract:\n" + _schema_contract(stage)
            )
            if purpose == "manual_arm_check":
                prompt += (
                    "\nFeature-label guidance: first compare the affected area's "
                    "color with the immediately surrounding skin. A directly "
                    "visible lighter or white, well-demarcated hypopigmented patch "
                    "requires pigment_loss. Do not label a lighter or white patch "
                    "as bruising because of edge shadows, screen tint, or contrast. "
                    "Use bruising only when the affected skin itself is darker with "
                    "an injury-like red, purple, blue, or brown color. Never return "
                    "both pigment_loss and bruising. Use discoloration only when a "
                    "color change is visible but cannot be classified more "
                    "specifically. Select no more than three directly supported "
                    "features and never fill the list speculatively. "
                    "Do not infer vitiligo or any diagnosis. If no visible change "
                    "is present, set finding_present false and visible_features "
                    "empty. If the view is unusable, set sufficient_skin_visible "
                    "false and visual_source unclear. Use image_quality fair when "
                    "a finding remains recognizable despite moderate phone glare "
                    "or blur; use poor only when visual labeling is not possible. "
                    "Include both possible_conditions and follow_up_topics as "
                    "empty arrays for every manual arm check."
                )
            elif stage == "preliminary":
                # The manual path earns its reliability partly from the block
                # above: concrete, enumerated, worked instructions. Preliminary
                # asks for twenty fields and used to ship only the bare
                # contract, which is where the prose drift came from.
                prompt += (
                    "\nOutput shape: begin the reply with { and end it with }. "
                    "Every one of the "
                    f"{len(_PRELIMINARY_PROPERTIES)} keys below appears "
                    "exactly once, with no extra keys and no commentary "
                    "between them:\n"
                    + ", ".join(_PRELIMINARY_PROPERTIES) + "\n"
                    "Each of "
                    + ", ".join(_FACIAL_KEYS[:-1])
                    + " is exactly one of none, mild, marked, or unclear -- "
                    "never a sentence, never a number. "
                    "nasal_discharge_visible is exactly no, yes, or unclear. "
                    "Use unclear whenever the panel does not let you judge "
                    "that cue, and set facial_cue_confidence to how well the "
                    "face was actually visible. Report only what is directly "
                    "visible; do not infer tiredness, illness, or any "
                    "diagnosis from a cue."
                )
            return prompt + "\nVisual extraction task:\n" + task
        return (
            "Return exactly one raw JSON object matching the provider-supplied "
            "response schema, with no prose, markdown, explanation, disclaimer, "
            "diagnosis, or recommendation. Record only directly visible "
            "attributes; this is not a request for medical advice.\n\n"
            + task
        )

    @staticmethod
    def _validation_detail(raw: Any, stage: str, exc: BaseException,
                           content: Any) -> dict[str, Any]:
        schema = _PRELIMINARY_SCHEMA if stage == "preliminary" else _CLOSEUP_SCHEMA
        missing = ([key for key in schema["required"] if key not in raw]
                   if isinstance(raw, dict) else list(schema["required"]))
        shape = _response_content_diagnostic(content)
        return {"reason": _bounded_text(str(exc), 160) or type(exc).__name__,
                "missing_fields": missing[:20],
                "response_type": type(raw).__name__,
                "provider_content": shape,
                "diagnostic_hint": _response_diagnostic_hint(raw, shape)}

    def _reformat_prompt(self, stage: str, prose: str,
                         feedback: str = "") -> str:
        """Turn a vision reply that drifted into prose into a JSON encode task.

        The NVIDIA Llama-3.2 vision endpoint ignores ``response_format`` and
        guided decoding on image calls and frequently answers with a scene
        description instead of the schema object. A text-only follow-up that
        re-encodes that description does honor ``response_format`` and reliably
        yields schema-valid JSON, so the finding the model already observed is
        preserved instead of being discarded as ``json_parse``.
        """
        return (
            "You are a JSON extraction function. The observation below was "
            "written by a vision model describing a single image. Encode it into "
            "exactly one raw JSON object and nothing else: the first character "
            "must be { and the last must be }. Emit no preamble, prose, markdown, "
            "explanation, or disclaimer. Record only attributes the observation "
            "supports; when it does not mention an attribute, use the schema's "
            "neutral value (false, an empty array, \"unclear\", or 0). Do not "
            "invent findings the observation does not describe. "
            "This is not a request for medical advice.\n"
            + _schema_contract(stage)
            + ("\n\nThe previous encode attempt was rejected: " + feedback
               + " Return the corrected object with every required key."
               if feedback else "")
            + "\n\nObservation to encode:\n" + prose
        )

    def _reformat_json(self, prose: str, stage: str, *, timeout: float,
                       purpose: str, deadline: float | None,
                       reservation_token: int | None,
                       cancel_event: threading.Event | None,
                       feedback: str = "") -> dict | None:
        """Best-effort text-only rescue of a prose vision reply into JSON.

        Returns the parsed object, or ``None`` on any transport or parse
        failure so the caller falls through to its normal retry path. Never
        logs the prose, which may describe the person in view.
        """
        prose = (prose or "").strip()
        if not prose:
            return None
        try:
            response = self._client.request(
                self._reformat_prompt(stage, prose, feedback),
                None,
                max_tokens=700,
                response_format=_response_format(stage),
                timeout=timeout,
                purpose=purpose,
                deadline=deadline,
                reservation_token=reservation_token,
                cancel_event=cancel_event)
        except NvidiaVLMError:
            return None
        content = getattr(response, "content", response)
        try:
            raw = _extract_json(content)
        except (ValueError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def _call_api(self, frames: list[np.ndarray] | np.ndarray | bytes, stage: str,
                  previous: SkinAnalysis | None, face_crop_available: bool = False,
                  arm_crop_label: str | None = None,
                  purpose: str = "passive_scan",
                  paired_context: bool = False,
                  deadline_monotonic: float | None = None,
                  reservation_token: int | None = None,
                  cancel_event: threading.Event | None = None) -> SkinAnalysis:
        """Hold logical priority from capture through every provider retry."""
        priority_work = purpose in {"manual_arm_check", "guided_closeup"}
        if deadline_monotonic is None and purpose == "manual_arm_check":
            deadline_monotonic = (
                time.monotonic() + float(self.manual_retry_deadline))
        elif deadline_monotonic is None and purpose == "guided_closeup":
            deadline_monotonic = (
                time.monotonic() + max(
                    60.0, float(self.request_timeout) * 3.0))
        if priority_work and reservation_token is None:
            reservation_token = self._client.reserve_request(
                purpose, float(deadline_monotonic))
        try:
            return self._call_api_attempts(
                frames, stage, previous, face_crop_available,
                arm_crop_label, purpose, paired_context,
                deadline_monotonic, reservation_token, cancel_event)
        finally:
            if priority_work:
                self._client.cancel_reservation(reservation_token)

    def _call_api_attempts(
                  self, frames: list[np.ndarray] | np.ndarray | bytes, stage: str,
                  previous: SkinAnalysis | None, face_crop_available: bool = False,
                  arm_crop_label: str | None = None,
                  purpose: str = "passive_scan",
                  paired_context: bool = False,
                  deadline_monotonic: float | None = None,
                  reservation_token: int | None = None,
                  cancel_event: threading.Event | None = None) -> SkinAnalysis:
        """Run bounded structured generation with locally validated retries."""
        preencoded = isinstance(frames, (bytes, bytearray))
        if preencoded:
            frames = [bytes(frames)]
        elif isinstance(frames, np.ndarray):
            frames = [frames]
        if len(frames) != 1:
            self._finish_request_diagnostics(
                "error", stage, error="exactly one composite image is required",
                retryable=False, terminal_reason="payload_size",
                failure_category="payload_size")
            raise SkinVisionAPIError(
                "exactly one composite image is required",
                retryable=False, kind="payload_size")

        manual = purpose == "manual_arm_check"
        # Budgets used to be manual-only: every other stage fell through to a
        # bare two-attempt/700-token branch, and only supplied its own deadline
        # if the caller passed one (guided close-ups do; the preliminary scan
        # did not). That left the preliminary scan asking for the largest
        # schema with the smallest allowance and no wall clock at all, which is
        # what made it the flaky one. Each stage now names its own budget
        # instead of inheriting the fallback.
        if manual:
            attempt_budget: float | None = float(self.manual_retry_deadline)
            max_attempts = max(1, int(self.manual_max_attempts))
            request_timeout = float(self.manual_attempt_timeout)
            max_tokens = int(self.manual_retry_max_tokens)
        elif stage == "preliminary":
            attempt_budget = float(self.preliminary_deadline)
            max_attempts = max(1, int(self.preliminary_max_attempts))
            request_timeout = float(self.preliminary_attempt_timeout)
            max_tokens = int(self.preliminary_max_tokens)
        else:
            attempt_budget = None
            max_attempts = 2
            request_timeout = float(self.request_timeout)
            max_tokens = 700
        deadline = (float(deadline_monotonic)
                    if deadline_monotonic is not None else
                    time.monotonic() + attempt_budget
                    if attempt_budget is not None else None)
        # True once this request is answerable against a wall clock, so the
        # "don't start a retry that cannot finish" guards apply to any bounded
        # stage rather than only to manual checks.
        bounded = deadline is not None
        # A `json_parse` line could not previously distinguish "the rescue never
        # ran" from "the rescue ran and its JSON was rejected"; both printed the
        # same thing. Carried across attempts so the terminal diagnostic names
        # which one actually happened.
        rescue_outcome = ""
        rescue_feedback = ""
        validation: dict[str, Any] | None = None
        terminal_category = "schema_validation"
        terminal_status: int | None = None
        terminal_error = "invalid structured response"
        terminal_diagnostic_error = "invalid structured response"
        terminal_retryable = False

        with self._diagnostic_lock:
            diagnostics_active = bool(
                self._diagnostic_status == "in_flight"
                and self._diagnostic_current_stage == stage
                and self._diagnostic_purpose == purpose)
        if not diagnostics_active:
            self._begin_request_diagnostics(
                stage, purpose, uuid.uuid4().hex, deadline,
                capture_mode=(self._arm_check_capture_mode if manual else None),
                view_labels=(self._arm_check_view_labels if manual else ()),
                image_count=1)

        for attempt_index in range(max_attempts):
            attempt = attempt_index + 1
            is_retry = attempt > 1
            attempt_tokens = max_tokens
            payload_mode = "compact_retry" if is_retry else "normal"
            request_prompt = self._prompt(
                stage, previous, face_crop_available, arm_crop_label, purpose,
                paired_context=paired_context, include_contract=True)
            if is_retry:
                missing = (
                    validation.get("missing_fields", [])
                    if isinstance(validation, dict) else [])
                missing_text = (
                    " Missing required fields: "
                    + ", ".join(str(name) for name in missing[:8]) + "."
                    if missing else "")
                validation_reason = (
                    self._safe_diagnostic_text(
                        validation.get("reason"), 120)
                    if isinstance(validation, dict) else "")
                validation_text = (
                    " Validation issue: " + validation_reason + "."
                    if validation_reason else "")
                request_prompt += (
                    "\nThe prior attempt did not produce a validated object. "
                    "Return repaired JSON with every required field."
                    + missing_text + validation_text)
            request_format = None if is_retry else _response_format(stage)
            if is_retry and terminal_category == "schema_validation":
                validation = validation or {
                    "reason": "prior structured result was invalid"}

            remaining = ((deadline - time.monotonic())
                         if deadline is not None else request_timeout)
            if remaining <= 0 or (is_retry and bounded and remaining < 5.0):
                terminal_category = "timeout"
                terminal_error = ("manual analysis deadline exceeded" if manual
                                  else f"{stage} analysis deadline exceeded")
                terminal_diagnostic_error = terminal_error
                terminal_retryable = False
                break
            attempt_timeout = min(request_timeout, remaining)
            encode_started = time.monotonic()
            encoded: bytes | None = None
            try:
                if preencoded:
                    encoded = bytes(frames[0])
                    if is_retry:
                        decoded = cv2.imdecode(
                            np.frombuffer(encoded, dtype=np.uint8),
                            cv2.IMREAD_COLOR)
                        if decoded is not None:
                            encoded = self._client.encode(
                                decoded,
                                max_image_dim=max(
                                    256, int(self.manual_retry_max_image_dim)
                                    - 128 * min(2, attempt_index)),
                                jpeg_quality=max(
                                    50, int(self.manual_retry_jpeg_quality)
                                    - 10 * min(2, attempt_index)),
                                max_inline_image_bytes=min(
                                    int(self.max_inline_image_bytes), 130560))
                else:
                    max_dimension = (
                        int(self.manual_retry_max_image_dim)
                        if not is_retry else
                        max(256, int(self.manual_retry_max_image_dim)
                            - 128 * min(2, attempt_index)))
                    quality = (
                        int(self.manual_retry_jpeg_quality)
                        if not is_retry else
                        max(50, int(self.manual_retry_jpeg_quality)
                            - 10 * min(2, attempt_index)))
                    encoded = self._client.encode(
                        frames[0], max_image_dim=max_dimension,
                        jpeg_quality=quality,
                        max_inline_image_bytes=(
                            int(self.max_inline_image_bytes)
                            if not is_retry else
                            min(int(self.max_inline_image_bytes), 130560)))
                response = self._client.request(
                    request_prompt, [encoded],
                    max_tokens=attempt_tokens,
                    response_format=request_format,
                    timeout=attempt_timeout,
                    purpose=purpose,
                    deadline=deadline,
                    reservation_token=reservation_token,
                    cancel_event=cancel_event)
            except NvidiaVLMError as exc:
                category = {
                    "response_shape": "schema_validation",
                }.get(str(getattr(exc, "kind", "")),
                      str(getattr(exc, "kind", "") or "network")
                      )
                terminal_category = category
                terminal_status = exc.status
                terminal_error = str(exc)
                terminal_diagnostic_error = (
                    f"HTTP {exc.status}" if exc.status is not None
                    else category)
                terminal_retryable = bool(exc.retryable)
                retry_after = getattr(exc, "retry_after", None)
                finish_reason = getattr(exc, "finish_reason", None)
                if finish_reason == "length":
                    max_tokens = 700
                if manual:
                    if category == "authentication":
                        self._manual_authorization_blocked = True
                    if retry_after is not None:
                        self._manual_next_allowed = max(
                            self._manual_next_allowed,
                            time.monotonic() + max(
                                0.0, min(10.0, float(retry_after))))
                self._record_attempt_diagnostics(
                    attempt=attempt, outcome="error",
                    payload_mode=payload_mode,
                    structured=request_format is not None,
                    max_tokens=attempt_tokens,
                    latency_ms=(time.monotonic() - encode_started) * 1000.0,
                    encoded_bytes=len(encoded or b""),
                    http_status=exc.status,
                    finish_reason=finish_reason,
                    request_id=getattr(exc, "request_id", None),
                    retry_after=retry_after,
                    retryable=exc.retryable, category=category,
                    error=terminal_diagnostic_error)
                may_retry = bool(
                    attempt < max_attempts and exc.retryable
                    and category != "authentication"
                    and (deadline is None
                         or deadline - time.monotonic() >= 5.0))
                if may_retry:
                    remaining_after = (
                        deadline - time.monotonic()
                        if deadline is not None else request_timeout)
                    if retry_after is not None:
                        delay = max(0.0, min(10.0, float(retry_after)))
                    else:
                        delay = (0.35 * (2 ** attempt_index)
                                 + random.uniform(0.0, 0.20))
                    if bounded and remaining_after - delay < 5.0:
                        break
                    if delay > 0:
                        time.sleep(min(delay, max(0.0, remaining_after)))
                    continue
                self._finish_request_diagnostics(
                    "error", stage, error=terminal_diagnostic_error,
                    http_status=exc.status,
                    validation=validation, repair_attempted=is_retry,
                    retryable=exc.retryable, terminal_reason=category,
                    failure_category=category)
                wrapped = SkinVisionAPIError(
                    terminal_error,
                    status=terminal_status,
                    retryable=terminal_retryable,
                    kind=category,
                    retry_after=retry_after,
                )
                detach_exception_context(exc)
                raise wrapped from None

            content = getattr(response, "content", response)
            response_status = getattr(response, "status", 200)
            finish_reason = getattr(response, "finish_reason", None)
            encoded_bytes = int(
                getattr(response, "encoded_image_bytes", len(encoded or b"")))
            response_latency_ms = float(getattr(
                response, "latency_ms",
                (time.monotonic() - encode_started) * 1000.0))
            raw: Any = None
            try:
                raw = _extract_json(content)
                analysis = validate_analysis(
                    raw, float(self.min_confidence),
                    allow_facial_cues=stage == "preliminary",
                    min_facial_confidence=float(self.min_facial_confidence),
                    face_crop_available=face_crop_available,
                    strict_schema=True)
                if manual and analysis.possible_conditions:
                    analysis = replace(analysis, possible_conditions=())
                self._record_attempt_diagnostics(
                    attempt=attempt, outcome="success",
                    payload_mode=payload_mode,
                    structured=request_format is not None,
                    max_tokens=attempt_tokens, latency_ms=response_latency_ms,
                    encoded_bytes=encoded_bytes,
                    http_status=response_status,
                    finish_reason=finish_reason,
                    request_id=getattr(response, "request_id", None),
                    retry_after=getattr(response, "retry_after", None),
                    poll_count=getattr(response, "poll_count", 0),
                    queue_ms=getattr(response, "queue_ms", 0.0))
                self._finish_request_diagnostics(
                    "success", stage, validation=None,
                    repair_attempted=is_retry)
                if manual:
                    self._manual_next_allowed = -1e9
                    self._manual_authorization_blocked = False
                return analysis
            except (KeyError, IndexError, TypeError, ValueError,
                    json.JSONDecodeError) as exc:
                validation = self._validation_detail(
                    raw, stage, exc, content)
                empty = (
                    content is None
                    or isinstance(content, str) and not content.strip()
                    or isinstance(content, list) and not content)
                category = (
                    "empty_content" if empty else
                    "json_parse" if raw is None else
                    "schema_validation")
                terminal_category = category
                terminal_status = response_status
                terminal_error = self._safe_diagnostic_text(
                    str(exc), 160) or category
                terminal_diagnostic_error = terminal_error
                terminal_retryable = True
                self._record_attempt_diagnostics(
                    attempt=attempt, outcome="invalid_response",
                    payload_mode=payload_mode,
                    structured=request_format is not None,
                    max_tokens=attempt_tokens,
                    latency_ms=response_latency_ms,
                    encoded_bytes=encoded_bytes,
                    http_status=response_status,
                    finish_reason=finish_reason,
                    request_id=getattr(response, "request_id", None),
                    retry_after=getattr(response, "retry_after", None),
                    retryable=True, category=category,
                    error=terminal_error, validation=validation,
                    poll_count=getattr(response, "poll_count", 0),
                    queue_ms=getattr(response, "queue_ms", 0.0))
                if category == "json_parse" and finish_reason != "length":
                    cancelled = (cancel_event is not None
                                 and cancel_event.is_set())
                    rescued_analysis = None
                    # One feedback round: a first encode that misses a key or
                    # mislabels an enum is worth correcting, because the vision
                    # model has already done the seeing. Bounded at two so a
                    # rescue can never outlive the request's own deadline.
                    for rescue_round in range(2):
                        reformat_remaining = (
                            (deadline - time.monotonic())
                            if deadline is not None else request_timeout)
                        if reformat_remaining < 5.0 or cancelled:
                            rescue_outcome = rescue_outcome or "skipped"
                            break
                        rescued = self._reformat_json(
                            _content_text(content), stage,
                            timeout=min(request_timeout, reformat_remaining),
                            purpose=purpose, deadline=deadline,
                            reservation_token=reservation_token,
                            cancel_event=cancel_event,
                            feedback=rescue_feedback)
                        if rescued is None:
                            rescue_outcome = "transport_failed"
                            break
                        try:
                            rescued_analysis = validate_analysis(
                                _normalize_rescued(rescued, stage),
                                float(self.min_confidence),
                                allow_facial_cues=stage == "preliminary",
                                min_facial_confidence=float(
                                    self.min_facial_confidence),
                                face_crop_available=face_crop_available,
                                strict_schema=True)
                            rescue_outcome = "ok"
                            break
                        except (KeyError, IndexError, TypeError, ValueError,
                                json.JSONDecodeError) as rescue_exc:
                            rescued_analysis = None
                            rescue_outcome = "validation_failed"
                            rescue_feedback = self._safe_diagnostic_text(
                                str(rescue_exc), 120) or "invalid object"
                            if rescue_round:
                                break
                    if rescued_analysis is not None:
                        if manual and rescued_analysis.possible_conditions:
                            rescued_analysis = replace(
                                rescued_analysis, possible_conditions=())
                        self._record_attempt_diagnostics(
                            attempt=attempt, outcome="success",
                            payload_mode="reformat_rescue",
                            structured=True, max_tokens=attempt_tokens,
                            latency_ms=response_latency_ms,
                            encoded_bytes=encoded_bytes,
                            http_status=response_status,
                            finish_reason=finish_reason)
                        self._finish_request_diagnostics(
                            "success", stage, validation=None,
                            repair_attempted=True)
                        if manual:
                            self._manual_next_allowed = -1e9
                            self._manual_authorization_blocked = False
                        return rescued_analysis
                    self._note_rescue_outcome(rescue_outcome)
                if finish_reason == "length":
                    max_tokens = 700
                if (attempt < max_attempts
                        and (deadline is None
                             or deadline - time.monotonic() >= 5.0)):
                    delay = (0.35 * (2 ** attempt_index)
                             + random.uniform(0.0, 0.20))
                    if deadline is not None \
                            and deadline - time.monotonic() - delay < 5.0:
                        break
                    if delay > 0:
                        time.sleep(delay)
                    continue
                break

        status = ("invalid_response"
                  if terminal_category in {
                      "empty_content", "json_parse", "schema_validation"}
                  else "error")
        self._finish_request_diagnostics(
            status, stage, error=terminal_diagnostic_error,
            http_status=terminal_status, validation=validation,
            repair_attempted=len(self._diagnostic_attempts) > 1,
            retryable=terminal_retryable,
            terminal_reason=terminal_category,
            failure_category=terminal_category)
        raise SkinVisionAPIError(
            terminal_error, status=terminal_status,
            retryable=terminal_retryable, kind=terminal_category)

    @staticmethod
    def _manual_analysis_problem(analysis: SkinAnalysis) -> str | None:
        """Return why a manual live/photo screen is inconclusive, if anything."""
        displayed = analysis.visual_source == "displayed_photo"
        # A deliberately displayed full photo must not be sent back for
        # repositioning over phone-screen glare; gate quality for live skin
        # only. Genuinely unusable displayed photos are still caught by the
        # sufficient_skin_visible check below.
        if not displayed and analysis.image_quality not in {"fair", "good"}:
            return "image_quality_poor"
        if not analysis.sufficient_skin_visible:
            return ("displayed_photo_not_clear" if displayed
                    else "insufficient_skin_visible")
        if analysis.visual_source == "unclear":
            return "visual_source_unclear"
        if analysis.visual_source == "displayed_photo":
            return None
        region = " ".join(re.findall(r"[a-z]+", analysis.body_region.lower()))
        arm_region = bool(re.search(r"\b(?:forearm|arm)\b", region))
        wrong_region = bool(re.search(r"\b(?:face|neck|torso|hand)\b", region))
        return None if arm_region and not wrong_region else "arm_region_not_confirmed"

    @staticmethod
    def _manual_arm_analysis_usable(analysis: SkinAnalysis) -> bool:
        """Compatibility predicate for a usable live or displayed-photo screen."""
        return SkinVision._manual_analysis_problem(analysis) is None

    def _call_closeup(self, frames: list[np.ndarray], previous: SkinAnalysis | None,
                      purpose: str, use_cloud: bool,
                      arm_crop_label: str | None = None,
                      local_frame: np.ndarray | None = None,
                      paired_context: bool = False,
                      deadline_monotonic: float | None = None,
                      reservation_token: int | None = None,
                      cancel_event: threading.Event | None = None) -> _CloseupOutcome:
        """Analyze one close-up locally and in the cloud without coupling failures."""
        local = None
        local_future: Future | None = None
        if self._local_classifier is not None and self._local_ready:
            local_input = local_frame if local_frame is not None else frames[0]
            try:
                if use_cloud:
                    # Optional local corroboration runs beside NVIDIA and may
                    # never delay the manual/guided terminal result.
                    local_future = self._local_worker.submit(
                        self._local_classifier.predict, local_input)
                else:
                    local = self._local_classifier.predict(local_input)
            except BaseException as exc:  # local GPU/model failure must not block NVIDIA
                local = LocalSkinPrediction.unavailable(
                    str(getattr(self._local_classifier, "backend", "local")),
                    str(getattr(self._local_classifier, "model", "unknown")),
                    str(getattr(
                        self._local_classifier, "revision", "unknown")),
                    str(getattr(self._local_classifier, "target", "vitiligo")),
                    type(exc).__name__,
                )
                detach_exception_context(exc)
        analysis = None
        cloud_error = None
        if use_cloud:
            try:
                analysis = self._call_api(
                    frames, "closeup", previous, arm_crop_label=arm_crop_label,
                    purpose=purpose, paired_context=paired_context,
                    deadline_monotonic=deadline_monotonic,
                    reservation_token=reservation_token,
                    cancel_event=cancel_event)
            except BaseException as exc:  # cloud degradation must not discard local evidence
                cloud_error = detach_exception_context(exc)
        if local_future is not None:
            remaining = (
                deadline_monotonic - time.monotonic()
                if deadline_monotonic is not None else 0.25)
            try:
                local = local_future.result(
                    timeout=max(0.0, min(0.25, remaining)))
            except FutureTimeoutError:
                local_future.cancel()
                local = None
            except BaseException as exc:
                local = LocalSkinPrediction.unavailable(
                    str(getattr(self._local_classifier, "backend", "local")),
                    str(getattr(self._local_classifier, "model", "unknown")),
                    str(getattr(
                        self._local_classifier, "revision", "unknown")),
                    str(getattr(self._local_classifier, "target", "vitiligo")),
                    type(exc).__name__,
                )
                detach_exception_context(exc)
        if local is not None:
            if local.status == "unavailable":
                CapabilityRegistry.instance().set(
                    "local_skin_classifier", "model", CapabilityStatus.DEGRADED,
                    local.abstain_reason or "local inference unavailable")
            else:
                CapabilityRegistry.instance().set(
                    "local_skin_classifier", "model", CapabilityStatus.READY,
                    f"{local.backend} close-up inference {local.inference_ms:.1f} ms")
        return _CloseupOutcome(analysis, local, cloud_error)

    @staticmethod
    def _local_corroborates(analysis: SkinAnalysis,
                            local: LocalSkinPrediction | None) -> bool:
        return bool(local is not None and local.status == "accepted"
                    and local.target == "vitiligo"
                    and any(feature in {"discoloration", "pigment_loss"}
                            for feature in analysis.visible_features)
                    and analysis.image_quality in {"fair", "good"})

    @staticmethod
    def _is_pigment_change(analysis: SkinAnalysis,
                           corroborated: bool = False) -> bool:
        """Return whether public wording should use neutral pigment language."""
        return bool(corroborated or "pigment_loss" in analysis.visible_features)

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
                         "visible_features": ["discoloration"],
                         "visual_source": "live_skin",
                         "attempt": self._arm_check_attempt,
                         "capture_mode": self._arm_check_capture_mode}
                        if purpose == "manual_arm_check" else
                        {"body_region": region, "visible_features": ["discoloration"],
                         "visual_source": "live_skin"})
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
                purpose: str | None = None, *,
                local_frame: np.ndarray | None = None,
                paired_context: bool = False,
                deadline_monotonic: float | None = None,
                captured_monotonic: float | None = None,
                reservation_token: int | None = None) -> None:
        if self._pending is not None:
            return
        frames = frame if isinstance(frame, list) else [frame]
        frames = [np.array(item, copy=True) for item in frames]
        local_frame = (np.array(local_frame, copy=True)
                       if local_frame is not None else None)
        previous = self._preliminary
        purpose = purpose or ("passive_scan" if stage == "preliminary"
                              else "guided_closeup")
        manual = purpose == "manual_arm_check"
        if manual and deadline_monotonic is None:
            deadline_monotonic = (
                time.monotonic() + float(self.manual_retry_deadline))
        use_cloud = bool(
            self.available
            and (not manual and now >= self._next_allowed
                 or manual and not self._manual_authorization_blocked
                 and time.monotonic() >= self._manual_next_allowed))
        if stage == "preliminary" and not use_cloud:
            return
        if stage == "closeup" and not use_cloud and not self._local_ready:
            return
        priority_work = purpose in {"manual_arm_check", "guided_closeup"}
        if use_cloud and priority_work and deadline_monotonic is None:
            deadline_monotonic = time.monotonic() + max(
                60.0, float(self.request_timeout) * 3.0)
        if use_cloud and priority_work and reservation_token is None:
            reservation_token = self._client.reserve_request(
                purpose, float(deadline_monotonic))
        cancel_event = threading.Event() if use_cloud else None
        correlation_id = (
            self._arm_check_correlation_id
            if purpose == "manual_arm_check" and self._arm_check_correlation_id
            else self._correlation_id or uuid.uuid4().hex)
        self._pending_stage = stage
        self._pending_purpose = purpose
        self._pending_correlation_id = correlation_id
        self._pending_capture_mode = (
            self._arm_check_capture_mode if purpose == "manual_arm_check" else None)
        if purpose == "manual_arm_check":
            if self._pending_capture_mode == "pose_crop_with_context":
                view_labels = ("whole_frame", "arm_crop")
            elif self._pending_capture_mode == "pose_crop":
                view_labels = ("arm_crop",)
            else:
                view_labels = ("whole_frame",)
        else:
            view_labels = ()
        self._pending_view_labels = view_labels
        self._pending_image_count = len(frames)
        self._pending_arm_attempt = (
            self._arm_check_attempt if manual else 0)
        self._pending_deadline_monotonic = deadline_monotonic
        self._pending_reservation_token = reservation_token
        self._pending_cancel_event = cancel_event
        if use_cloud:
            self._begin_request_diagnostics(
                stage, purpose, correlation_id, deadline_monotonic,
                capture_mode=self._pending_capture_mode,
                view_labels=view_labels,
                image_count=len(frames),
                started_monotonic=captured_monotonic)
        if manual:
            self._arm_check_view_labels = view_labels
            self._arm_check_image_count = len(frames)
            self._arm_check_state = "pending"
            self._arm_check_last_error = None
            if use_cloud:
                print(
                    "[skin-vision] manual check submitted "
                    f"(mode={self._pending_capture_mode or 'unknown'}, "
                    f"views={'+'.join(view_labels) or 'none'}, "
                    f"images={len(frames)})")
        try:
            local_allowed = not (
                purpose == "manual_arm_check"
                and self._arm_check_capture_mode == "cloud_closeup")
            if stage == "closeup" and local_allowed \
                    and self._local_classifier is not None and self._local_ready:
                self._pending = self._executor.submit(
                    self._call_closeup, frames, previous, purpose, use_cloud,
                    arm_crop_label, local_frame, paired_context,
                    deadline_monotonic, reservation_token, cancel_event)
            else:
                self._pending = self._executor.submit(
                    self._call_api, frames, stage, previous, face_crop_available,
                    arm_crop_label, purpose, paired_context,
                    deadline_monotonic, reservation_token, cancel_event)
        except BaseException as exc:
            self._client.cancel_reservation(reservation_token)
            if use_cloud:
                self._finish_request_diagnostics(
                    "error", stage, error=type(exc).__name__)
            self._pending_stage = None
            self._pending_purpose = None
            self._pending_correlation_id = None
            self._pending_capture_mode = None
            self._pending_view_labels = ()
            self._pending_image_count = 0
            self._pending_arm_attempt = 0
            self._pending_deadline_monotonic = None
            self._pending_reservation_token = None
            self._pending_cancel_event = None
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

    def _failure(self, now: float, exc: BaseException,
                 purpose: str = "passive_scan") -> None:
        category = self._safe_diagnostic_text(
            getattr(exc, "kind", None), 40) or "provider"
        status = getattr(exc, "status", None)
        if purpose == "manual_arm_check":
            retry_after = getattr(exc, "retry_after", None)
            if category == "authentication":
                self._manual_authorization_blocked = True
            if retry_after is not None:
                self._manual_next_allowed = max(
                    self._manual_next_allowed,
                    time.monotonic() + max(
                        0.0, min(10.0, float(retry_after))))
        else:
            self._failures += 1
            delay = min(float(self.backoff_max),
                        float(self.backoff_base) * (2 ** (self._failures - 1)))
            self._next_allowed = now + delay
            self._next_allowed_monotonic = time.monotonic() + delay
            # Remembered so a console request can tell "try again, it was a
            # blip" apart from "this will fail again the moment we ask".
            self._last_failure_retryable = bool(
                getattr(exc, "retryable", True))
            self._last_failure_category = category
            self._last_failure_status = status
            self._last_failure_at = now
        status_text = f", HTTP {status}" if status is not None else ""
        diagnostic_text = ""
        with self._diagnostic_lock:
            logical = copy.deepcopy(self._diagnostic_last_attempt)
        if (isinstance(logical, dict)
                and logical.get("purpose") == purpose):
            attempts = logical.get("attempts")
            last = attempts[-1] if isinstance(attempts, list) and attempts else {}
            validation = (last.get("validation")
                          if isinstance(last, dict) else None)
            shape = (validation.get("provider_content")
                     if isinstance(validation, dict) else None)
            details = []
            if logical.get("stage"):
                details.append(f"stage={logical['stage']}")
            if logical.get("attempt_count"):
                details.append(f"attempts={logical['attempt_count']}")
            if isinstance(validation, dict) and validation.get(
                    "diagnostic_hint"):
                details.append(f"hint={validation['diagnostic_hint']}")
            if isinstance(validation, dict) and validation.get(
                    "rescue_outcome"):
                details.append(f"rescue={validation['rescue_outcome']}")
            if isinstance(shape, dict):
                details.append(
                    "content="
                    f"{shape.get('content_type', 'unknown')}/"
                    f"{shape.get('leading_kind', 'unknown')}")
                if "character_count" in shape:
                    details.append(f"chars={shape['character_count']}")
                if "has_object_bounds" in shape:
                    details.append(
                        "object_bounds="
                        f"{'yes' if shape['has_object_bounds'] else 'no'}")
            finish_reason = (last.get("finish_reason")
                             if isinstance(last, dict) else None)
            if finish_reason:
                details.append(f"finish={finish_reason}")
            if details:
                diagnostic_text = ", " + ", ".join(details)
        print(
            "[skin-vision] inference unavailable "
            f"(category={category}{status_text}{diagnostic_text}); "
            "retrying later")

    def _consume_pending(self, now: float):
        if self._pending is None or not self._pending.done():
            return []
        pending, stage = self._pending, self._pending_stage
        purpose = self._pending_purpose or "passive_scan"
        correlation_id = self._pending_correlation_id
        capture_mode = self._pending_capture_mode
        view_labels = self._pending_view_labels
        view_text = "+".join(view_labels) or "none"
        image_count = self._pending_image_count
        arm_attempt = self._pending_arm_attempt
        deadline_monotonic = self._pending_deadline_monotonic
        reservation_token = self._pending_reservation_token
        self._pending = None
        self._pending_stage = None
        self._pending_purpose = None
        self._pending_correlation_id = None
        self._pending_capture_mode = None
        self._pending_view_labels = ()
        self._pending_image_count = 0
        self._pending_arm_attempt = 0
        self._pending_deadline_monotonic = None
        self._pending_reservation_token = None
        self._pending_cancel_event = None
        self._client.cancel_reservation(reservation_token)
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
            self._failure(now, exc, purpose)
            if stage == "closeup":
                self._reset_closeup()
            if purpose == "manual_arm_check":
                self._arm_check_state = "unavailable"
                category = self._safe_diagnostic_text(
                    getattr(exc, "kind", None), 40) or "provider"
                status = getattr(exc, "status", None)
                self._arm_check_last_error = (
                    f"{category}:http_{status}" if status is not None
                    else category)
                self._clear_matching_arm_elicitation(correlation_id)
                result = self.result(
                    "arm_check",
                    {"status": "unavailable", "source": "nvidia_vlm",
                      "attempt": arm_attempt, "capture_mode": capture_mode,
                     "visual_source": "unclear",
                     "reason": "provider_failure",
                     "failure_category": category,
                     "http_status": status},
                    0.0, Severity.INFO,
                    "The arm check could not be completed because visual analysis was unavailable",
                    ttl=30.0, source="nvidia_vlm", correlation_id=correlation_id)
                self._reset_arm_capture()
                _exc_msg = self._safe_diagnostic_text(str(exc), 120)
                _retryable = getattr(exc, "retryable", None)
                _req_id = getattr(exc, "request_id", None)
                print(
                    "[skin-vision] manual check completed "
                    f"(status=unavailable, mode={capture_mode or 'unknown'}, "
                    f"views={view_text}, images={image_count}, "
                    "visual_source=unclear, finding_present=unknown, "
                    f"category={category}"
                    f"{f', http_status={status}' if status is not None else ''}"
                    f"{f', error={_exc_msg!r}' if _exc_msg else ''}"
                    f"{f', retryable={_retryable}' if _retryable is not None else ''}"
                    f"{f', request_id={_req_id}' if _req_id else ''})")
                return [result]
            return []
        if cloud_error is not None:
            self._failure(now, cloud_error, purpose)
        elif analysis is not None:
            if purpose == "manual_arm_check":
                self._manual_next_allowed = -1e9
            else:
                self._failures = 0
                self._next_allowed = now
                self._next_allowed_monotonic = time.monotonic()
                self._last_failure_retryable = True
                self._last_failure_category = None
                self._last_failure_status = None
        if stage == "closeup" and analysis is None:
            self._reset_closeup()
            local_results = self._local_only_results(purpose, correlation_id, local)
            screening_succeeded = bool(
                self._local_mode == "screening" and local is not None
                and local.status == "accepted")
            if purpose == "manual_arm_check":
                self._arm_check_state = "succeeded" if screening_succeeded else "unavailable"
                self._arm_check_last_error = None if screening_succeeded else (
                    (self._safe_diagnostic_text(
                        getattr(cloud_error, "kind", None), 40)
                     or "provider") if cloud_error is not None else
                    (local.abstain_reason if local is not None else "no_backend"))
                self._clear_matching_arm_elicitation(correlation_id)
                if not screening_succeeded:
                    local_results.insert(0, self.result(
                        "arm_check",
                        {"status": "unavailable", "source": "skin_screening",
                          "attempt": arm_attempt, "capture_mode": capture_mode,
                         "visual_source": ("live_skin"
                                           if capture_mode in (
                                               "pose_crop",
                                               "pose_crop_with_context")
                                           else "unclear"),
                         "reason": self._arm_check_last_error},
                        0.0, Severity.INFO,
                        "Skin close-up analysis was unavailable or inconclusive",
                        ttl=30.0, source="skin_screening",
                        correlation_id=correlation_id))
                _ce_kind = getattr(cloud_error, "kind", None) if cloud_error is not None else None
                _ce_msg = self._safe_diagnostic_text(str(cloud_error), 120) if cloud_error is not None else None
                _loc_status = local.status if local is not None else "no_local"
                _loc_abstain = local.abstain_reason if local is not None and local.abstain_reason else None
                print(
                    "[skin-vision] manual check completed "
                    f"(status={'succeeded' if screening_succeeded else 'unavailable'}, "
                    f"mode={capture_mode or 'unknown'}, "
                    f"views={view_text}, images={image_count}, "
                    f"visual_source={'live_skin' if capture_mode in ('pose_crop', 'pose_crop_with_context') else 'unclear'}, "
                    f"finding_present={'true' if screening_succeeded else 'unknown'}"
                    f"{f', cloud_error={_ce_kind}' if _ce_kind is not None else ''}"
                    f"{f', cloud_error_msg={_ce_msg!r}' if _ce_msg is not None else ''}"
                    f"{f', local_status={_loc_status}' if _loc_status else ''}"
                    f"{f', local_abstain={_loc_abstain}' if _loc_abstain is not None else ''})")
            return local_results
        if stage == "preliminary":
            results = []
            cues = analysis.facial_cues
            positive = cues.positive() if cues is not None else {}
            # Every validated field the model committed to, so a scan that
            # shows nothing on a card can be told apart from a scan that was
            # never answered. Enum values only -- never provider prose, which
            # describes whoever is in view.
            _observed = cues.observed() if cues is not None else {}
            print(
                "[skin-vision] face scan result "
                f"(quality={analysis.image_quality}, "
                f"facial_conf={cues.confidence if cues is not None else 0.0}, "
                f"gated={'yes' if cues is not None and cues.gated else 'no'}, "
                f"finding_present={str(analysis.finding_present).lower()}, "
                f"features={list(analysis.visible_features)}, "
                f"region={analysis.body_region!r}, "
                f"cues={_observed or '{}'})")
            if positive:
                labels = []
                for key, value in positive.items():
                    label = _FACIAL_LABELS[key]
                    labels.append(label if value == "yes" else f"{value} {label}")
                results.append(self.result(
                    "facial_appearance",
                    {"cues": positive, "confidence": cues.confidence},
                    confidence=cues.confidence, severity=Severity.INFO,
                    message="Visible facial appearance cues: " + ", ".join(labels),
                    ttl=120.0, source="nvidia_vlm",
                    quality=_QUALITY_SCORE[analysis.image_quality]))
            if cues is not None:
                # Same readings again, routed to the detector cards that own
                # each subject so a "no dryness seen" is visible next to the
                # lip heuristic rather than buried under skin_vision.
                results.extend(
                    self._routed_cue_results(cues, analysis.image_quality))
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
            problem = self._manual_analysis_problem(analysis)
            if problem is not None:
                result = self._manual_capture_result(
                    problem, visual_source=analysis.visual_source)
                self._clear_matching_arm_elicitation(correlation_id)
                self._reset_arm_capture()
                print(
                    "[skin-vision] manual check completed "
                    f"(status={result.value['status']}, mode={capture_mode or 'unknown'}, "
                    f"views={view_text}, images={image_count}, "
                    f"visual_source={analysis.visual_source}, finding_present=unknown, "
                    f"problem={problem})")
                return [result]
            self._arm_check_state = "succeeded"
            self._arm_check_last_error = None
            self._clear_matching_arm_elicitation(correlation_id)
            region = analysis.body_region or "the visible arm"
            corroborated = self._local_corroborates(analysis, local)
            pigment_change = self._is_pigment_change(analysis, corroborated)
            value = {
                "status": "succeeded",
                "source": "nvidia_vlm",
                "finding_present": bool(analysis.finding_present),
                "body_region": region,
                "visible_features": list(analysis.visible_features),
                "visual_source": analysis.visual_source,
                "attempt": arm_attempt,
                "capture_mode": capture_mode,
            }
            photo_prefix = "In the photo shown on the phone, "
            message = (photo_prefix + f"a possible pigment change appears on {region}"
                       if (analysis.visual_source == "displayed_photo"
                           and analysis.finding_present and pigment_change) else
                       photo_prefix + f"a possible visible change appears on {region}: "
                       + ", ".join(analysis.visible_features[:3])
                       if (analysis.visual_source == "displayed_photo"
                           and analysis.finding_present) else
                       photo_prefix
                       + "the NVIDIA VLM did not identify a clear visible skin change"
                       if analysis.visual_source == "displayed_photo" else
                       f"Possible pigment change on {region}"
                       if pigment_change else
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
            print(
                "[skin-vision] manual check completed "
                f"(status=succeeded, mode={capture_mode or 'unknown'}, "
                f"views={view_text}, images={image_count}, "
                f"visual_source={analysis.visual_source}, "
                f"finding_present={'true' if analysis.finding_present else 'false'})")
            # Also surface the cloud verdict on the local arm detector's card so
            # the modules console shows local + cloud side by side. Routed under
            # module="arm_skin" with source="nvidia_vlm" (mirrors
            # _routed_cue_results) so the console labels it a VLM second opinion
            # and never as the local heuristic having fired.
            cloud_verdict = (
                f"possible pigment change on {region}"
                if analysis.finding_present and pigment_change else
                "possible " + ", ".join(analysis.visible_features[:3])
                + f" on {region}"
                if analysis.finding_present and analysis.visible_features else
                "no clear visible skin change")
            routed = Result(
                module="arm_skin", key="vlm_arm_check",
                value={"status": "succeeded",
                       "finding_present": bool(analysis.finding_present),
                       "visible_features": list(analysis.visible_features),
                       "visual_source": analysis.visual_source,
                       "body_region": region},
                confidence=min(analysis.confidence, 0.65),
                severity=(Severity.NOTICE if analysis.finding_present
                          else Severity.INFO),
                message="Cloud photo check: " + cloud_verdict,
                ttl=120.0, source="nvidia_vlm",
                correlation_id=correlation_id,
                quality={"poor": .2, "fair": .6, "good": .9}[
                    analysis.image_quality],
                location=region)
            self._reset_closeup()
            self._reset_arm_capture()
            return [public, private, routed]
        correlation_id = self._correlation_id
        self._reset_closeup()
        if not analysis.finding_present:
            return []
        region = analysis.body_region or "the visible area"
        features = ", ".join(analysis.visible_features[:3])
        corroborated = self._local_corroborates(analysis, local)
        pigment_change = self._is_pigment_change(analysis, corroborated)
        public = self.result(
            "visible_skin_change",
            {"body_region": region,
             "visible_features": list(analysis.visible_features)},
            confidence=min(analysis.confidence, 0.65),
            severity=Severity.NOTICE,
            message=(f"Possible pigment change on {region}" if pigment_change else
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

    def _reset_arm_capture(self) -> None:
        self._arm_check_sampling = False
        self._arm_check_window_id = None
        self._best_arm_frame = None
        self._best_arm_context = None
        self._best_arm_label = None
        self._best_arm_sharpness = -1.0
        self._best_arm_fallback = None
        self._best_arm_fallback_sharpness = -1.0

    def _clear_matching_arm_elicitation(
            self, correlation_id: str | None) -> None:
        """Never let completion of an older request cancel a newer arm window."""
        if (self._elicitation.test == "arm_check"
                and self._elicitation.correlation_id == correlation_id):
            self._elicitation.clear()

    def _queue_manual_capture(
            self, frame: np.ndarray, *, local_frame: np.ndarray | None,
            arm_crop_label: str | None, paired_context: bool) -> None:
        """Own the best completed capture until the NVIDIA lane is available."""
        captured_at = time.monotonic()
        deadline = captured_at + float(self.manual_retry_deadline)
        reservation_token = self._client.reserve_request(
            "manual_arm_check", deadline)
        if (self._pending is not None
                and self._pending_purpose != "manual_arm_check"
                and self._pending_cancel_event is not None):
            # Withdraw this module's lower-priority coordinator ticket. A
            # manual reservation must not block the passive/guided Future
            # whose completion is required before manual submission.
            self._pending_cancel_event.set()
            self._client.notify_cancellation()
        capture_mode = self._arm_check_capture_mode or "cloud_closeup"
        view_labels = (
            ("whole_frame", "arm_crop")
            if capture_mode == "pose_crop_with_context" else
            ("arm_crop",) if capture_mode == "pose_crop" else
            ("whole_frame",))
        queued = _QueuedManual(
            frame=np.array(frame, copy=True),
            local_frame=(np.array(local_frame, copy=True)
                         if local_frame is not None else None),
            arm_crop_label=arm_crop_label,
            paired_context=bool(paired_context),
            capture_mode=capture_mode,
            correlation_id=(
                self._arm_check_correlation_id or uuid.uuid4().hex),
            arm_attempt=self._arm_check_attempt,
            queued_at=captured_at,
            deadline_monotonic=deadline,
            view_labels=view_labels,
            reservation_token=reservation_token,
        )
        self._manual_queue.append(queued)
        if self._queued_manual is None:
            self._queued_manual = queued
        head = self._queued_manual
        self._arm_check_state = "queued"
        self._arm_check_last_error = None
        self._arm_check_capture_mode = head.capture_mode
        self._arm_check_correlation_id = head.correlation_id
        self._arm_check_attempt = head.arm_attempt
        self._arm_check_view_labels = head.view_labels
        self._arm_check_image_count = 1
        print(
            "[skin-vision] manual check queued "
            f"(mode={capture_mode}, views={'+'.join(view_labels)}, images=1, "
            f"depth={len(self._manual_queue)})")

    def _remove_queued_manual(self, queued: _QueuedManual) -> None:
        """Remove exactly one owned capture and expose the next queue head."""
        for index, item in enumerate(self._manual_queue):
            if item is queued:
                del self._manual_queue[index]
                break
        self._queued_manual = (
            self._manual_queue[0] if self._manual_queue else None)

    def _record_queued_manual_failure(
            self, queued: _QueuedManual, category: str,
            error: str) -> None:
        """Complete a queued logical request without clobbering active telemetry."""
        completed_at = time.time()
        completed_monotonic = time.monotonic()
        latency_ms = round(
            max(0.0, completed_monotonic - queued.queued_at) * 1000.0, 1)
        logical = {
            "status": "error",
            "stage": "closeup",
            "purpose": "manual_arm_check",
            "correlation_id": queued.correlation_id,
            "started_at": completed_at - latency_ms / 1000.0,
            "completed_at": completed_at,
            "latency_ms": latency_ms,
            "http_status": None,
            "error": self._safe_diagnostic_text(error, 160),
            "validation": None,
            "repair_attempted": False,
            "retryable": False,
            "attempt_count": 0,
            "payload_mode": "queued",
            "encoded_bytes": 0,
            "capture_mode": queued.capture_mode,
            "view_count": len(queued.view_labels),
            "view_labels": list(queued.view_labels),
            "image_count": 1,
            "terminal_reason": category,
            "failure_category": category,
            "attempts": [],
        }
        with self._diagnostic_lock:
            self._diagnostic_request_count += 1
            self._diagnostic_failure_count += 1
            self._diagnostic_history.append(copy.deepcopy(logical))
            self._update_metrics_locked(logical)
            if self._diagnostic_status != "in_flight":
                self._diagnostic_status = "error"
                self._diagnostic_last_attempt = copy.deepcopy(logical)
        CapabilityRegistry.instance().set(
            "nvidia_skin", "cloud", CapabilityStatus.DEGRADED, category)

    def _fail_queued_manual(
            self, queued: _QueuedManual, category: str, message: str) -> Result:
        self._record_queued_manual_failure(queued, category, message)
        self._client.cancel_reservation(queued.reservation_token)
        self._remove_queued_manual(queued)
        next_queued = self._queued_manual
        self._arm_check_state = (
            "queued" if next_queued is not None else "unavailable")
        self._arm_check_last_error = (
            None if next_queued is not None else category)
        self._arm_check_capture_mode = (
            next_queued.capture_mode if next_queued is not None
            else queued.capture_mode)
        self._arm_check_correlation_id = (
            next_queued.correlation_id if next_queued is not None
            else queued.correlation_id)
        self._arm_check_attempt = (
            next_queued.arm_attempt if next_queued is not None
            else queued.arm_attempt)
        self._arm_check_view_labels = (
            next_queued.view_labels if next_queued is not None
            else queued.view_labels)
        self._arm_check_image_count = 1
        self._clear_matching_arm_elicitation(queued.correlation_id)
        self._reset_arm_capture()
        print(
            "[skin-vision] manual check completed "
            f"(status=unavailable, mode={queued.capture_mode}, "
            f"views={'+'.join(queued.view_labels)}, images=1, "
            f"category={category}, message={message!r})")
        return self.result(
            "arm_check",
            {"status": "unavailable", "source": "nvidia_vlm",
             "attempt": queued.arm_attempt,
             "capture_mode": queued.capture_mode,
             "visual_source": "unclear",
             "reason": "provider_failure",
             "failure_category": category},
            0.0, Severity.INFO,
            "The arm check could not be completed because visual analysis was unavailable",
            ttl=30.0, source="nvidia_vlm",
            correlation_id=queued.correlation_id)

    def _drain_queued_manual(self, now: float) -> list[Result]:
        """Submit a captured manual request ahead of all new passive work."""
        queued = self._queued_manual
        if queued is None:
            return []
        current = time.monotonic()
        if current >= queued.deadline_monotonic:
            return [self._fail_queued_manual(
                queued, "timeout", "manual analysis deadline exceeded in queue")]
        if self._manual_authorization_blocked:
            return [self._fail_queued_manual(
                queued, "authentication",
                "provider authentication is unavailable")]
        if self._pending is not None or current < self._manual_next_allowed:
            self._arm_check_state = "queued"
            return []
        if not self.available:
            return [self._fail_queued_manual(
                queued, "authentication", "provider credential is unavailable")]

        self._arm_check_correlation_id = queued.correlation_id
        self._arm_check_attempt = queued.arm_attempt
        self._arm_check_capture_mode = queued.capture_mode
        self._arm_check_view_labels = queued.view_labels
        self._arm_check_image_count = 1
        self._remove_queued_manual(queued)
        try:
            self._submit(
                queued.frame, "closeup", now,
                arm_crop_label=queued.arm_crop_label,
                purpose="manual_arm_check",
                local_frame=queued.local_frame,
                paired_context=queued.paired_context,
                deadline_monotonic=queued.deadline_monotonic,
                captured_monotonic=queued.queued_at,
                reservation_token=queued.reservation_token)
        except BaseException as exc:  # preserve capture across handoff failure
            if time.monotonic() < queued.deadline_monotonic:
                queued.reservation_token = self._client.reserve_request(
                    "manual_arm_check", queued.deadline_monotonic)
                self._manual_queue.appendleft(queued)
                self._queued_manual = queued
                self._arm_check_state = "queued"
                self._arm_check_last_error = (
                    self._safe_diagnostic_text(type(exc).__name__, 40)
                    or "provider")
                return []
            return [self._fail_queued_manual(
                queued, "timeout",
                "manual analysis deadline exceeded during handoff")]
        if self._pending is None:
            # A test seam or a transient local worker race may decline the
            # handoff without raising. Preserve the owned capture and retry
            # until its original deadline instead of discarding it.
            self._manual_queue.appendleft(queued)
            self._queued_manual = queued
            self._arm_check_state = "queued"
        return []

    def _collect_arm_check(self, ctx: FrameContext) -> None:
        """Keep a pose crop with matching context plus a whole-frame fallback."""
        if ctx.timestamp < self._elicitation.started + float(
                self.closeup_positioning_delay):
            return
        fallback_score = _sharpness(ctx.frame)
        if fallback_score > self._best_arm_fallback_sharpness:
            self._best_arm_fallback_sharpness = fallback_score
            self._best_arm_fallback = ctx.frame.copy()
        arm = self._best_arm_crop(ctx)
        if arm is None:
            return
        crop, label = arm
        score = _sharpness(crop)
        if score > self._best_arm_sharpness:
            self._best_arm_sharpness = score
            self._best_arm_frame = crop.copy()
            self._best_arm_context = ctx.frame.copy()
            self._best_arm_label = label

    def _manual_capture_result(
            self, reason: str, visual_source: str = "unclear") -> Result:
        """Return one reposition request, then an honest terminal failure."""
        retry = self._arm_check_attempt == 0
        status = "reposition_required" if retry else "unavailable"
        self._arm_check_state = status
        self._arm_check_last_error = reason
        phone = visual_source == "displayed_photo" or reason.startswith(
            "displayed_photo")
        message = (
            "The photo on the phone was not clear enough; please bring it closer, "
            "reduce glare, and hold it steady once"
            if retry and phone else
            "The camera did not capture a clear bare-arm or phone-photo view; "
            "please reposition once"
            if retry else
            "The phone-photo skin check could not be completed after repositioning"
            if phone else
            "The arm check could not be completed after the repositioning attempt")
        return self.result(
            "arm_check",
            {"status": status, "source": "skin_screening",
             "reason": reason, "attempt": self._arm_check_attempt,
             "visual_source": visual_source,
             "capture_mode": self._arm_check_capture_mode or "none"},
            0.0, Severity.INFO, message, ttl=30.0,
            source="skin_screening",
            correlation_id=self._arm_check_correlation_id)

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
        arm_active = self._elicitation.active("arm_check", now=now)
        manual_busy = bool(
            self._queued_manual is not None
            or self._pending is not None
            and self._pending_purpose == "manual_arm_check")
        if arm_active and not manual_busy:
            if self._arm_check_window_id != self._elicitation.started:
                self._reset_arm_capture()
                self._arm_check_window_id = self._elicitation.started
                self._arm_check_sampling = True
                self._arm_check_correlation_id = self._elicitation.correlation_id
                self._arm_check_attempt = self._elicitation.attempt
                self._arm_check_capture_mode = None
            self._arm_check_state = "sampling"
            self._collect_arm_check(ctx)
        results = self._consume_pending(now)
        results.extend(self._drain_queued_manual(now))
        if not self.available and not self._local_ready:
            if (self._arm_check_sampling and self._elicitation.test == "arm_check"
                    and now >= self._elicitation.until):
                pose_captured = self._best_arm_frame is not None
                self._arm_check_capture_mode = "pose_crop" if pose_captured else "none"
                failure = None if pose_captured else self._manual_capture_result(
                    "cloud_required_for_forearm_only")
                if pose_captured:
                    self._arm_check_state = "local_only"
                    self._arm_check_last_error = None
                self._elicitation.clear()
                self._reset_arm_capture()
                if failure is not None:
                    results.append(failure)
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
                    and self._queued_manual is None
                    and (self._local_ready or now >= self._next_allowed)):
                self._submit(frame, "closeup", now)
            else:
                self._reset_closeup()

        # Cloud checks pair a pose crop with its exact whole-frame context so a
        # phone screen cannot be cropped out. The whole-frame fallback remains
        # cloud-only when no pose crop exists.
        if (self._arm_check_sampling and self._elicitation.test == "arm_check"
                and now >= self._elicitation.until):
            can_cloud = bool(
                self.available and not self._manual_authorization_blocked)
            frame = None
            label = None
            if self._best_arm_frame is not None:
                label = self._best_arm_label
                if can_cloud and self._best_arm_context is not None:
                    frame = _compose_preliminary_frame(
                        self._best_arm_context, None, self._best_arm_frame,
                        arm_label=(label or "ARM CROP").upper())
                    self._arm_check_capture_mode = "pose_crop_with_context"
                else:
                    frame = self._best_arm_frame
                    self._arm_check_capture_mode = "pose_crop"
            elif self._best_arm_fallback is not None and can_cloud:
                frame = self._best_arm_fallback
                self._arm_check_capture_mode = "cloud_closeup"
            if frame is not None and can_cloud:
                self._arm_check_sampling = False
                self._queue_manual_capture(
                    frame, arm_crop_label=label,
                    local_frame=(
                        self._best_arm_frame
                        if self._arm_check_capture_mode
                        == "pose_crop_with_context" else None),
                    paired_context=(
                        self._arm_check_capture_mode
                        == "pose_crop_with_context"))
                self._elicitation.clear()
                self._reset_arm_capture()
                results.extend(self._drain_queued_manual(now))
            elif (frame is not None and self._pending is None
                  and self._local_ready
                  and self._arm_check_capture_mode == "pose_crop"):
                self._arm_check_sampling = False
                self._submit(
                    frame, "closeup", now, arm_crop_label=label,
                    purpose="manual_arm_check")
                self._elicitation.clear()
                self._reset_arm_capture()
            else:
                reason = ("cloud_required_for_forearm_only"
                          if self._best_arm_frame is None and not can_cloud else
                          "no_usable_frame")
                results.append(self._manual_capture_result(reason))
                self._elicitation.clear()
                self._reset_arm_capture()

        if (self._awaiting_closeup and not self._sampling_closeup
                and now - self._preliminary_at > 60.0):
            self._reset_closeup()

        # A latched console request waits out the busy gates above rather than
        # being dropped on the frame it arrived, and skips only the two timing
        # gates (backoff, cadence) the operator just overrode by asking.
        manual_scan = self._manual_scan_requested
        ready_for_scan = (self.available and ctx.person_present and self._pending is None
                          and not self._awaiting_closeup
                          and not self._arm_check_sampling
                          and self._queued_manual is None
                          and self._arm_check_state not in (
                              "sampling", "queued", "pending",
                              "reposition_required")
                          and self._preliminary is None
                          and (manual_scan or now >= self._next_allowed)
                          and (manual_scan
                               or now - self._last_scan >= float(self.scan_interval)))
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
                self._manual_scan_requested = False
                if manual_scan:
                    print("[skin-vision] face scan requested "
                          f"(face_crop={'yes' if face_crop is not None else 'no'}, "
                          f"arm_crop={arm_crop_label or 'none'})")
            except (ValueError, cv2.error) as exc:
                self._manual_scan_requested = False
                self._failure(now, exc)
        return results or None

    def close(self) -> None:
        """Cancel pending work without waiting on a slow network request."""
        if self._closed:
            return
        self._closed = True
        if self._pending_cancel_event is not None:
            self._pending_cancel_event.set()
        self._client.notify_cancellation()
        self._client.cancel_reservation(self._pending_reservation_token)
        self._pending_reservation_token = None
        self._pending_cancel_event = None
        for queued in self._manual_queue:
            self._client.cancel_reservation(queued.reservation_token)
        if self._pending is not None:
            self._pending.cancel()
        self._pending = None
        if self._local_load_future is not None:
            self._local_load_future.cancel()
        self._local_load_future = None
        self._manual_queue.clear()
        self._queued_manual = None
        self._reset_arm_capture()
        self._reset_closeup()
        stopped = self._executor.shutdown(timeout=0.5)
        self._local_worker.shutdown(wait=False, cancel_futures=True, timeout=0.5)
        if self._local_classifier is not None:
            self._local_classifier.close()
        if not stopped:
            print("[skin-vision] cloud worker still finishing a bounded request; shutdown continues")
