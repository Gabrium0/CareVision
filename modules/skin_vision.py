"""Opt-in, two-stage skin screening through an NVIDIA vision model.

The module sends infrequent in-memory JPEG stills only after explicit runtime
consent. A preliminary whole-frame observation asks the voice agent to request
a close-up; only the close-up can emit a public, non-diagnostic observation.
Possible condition names remain agent-only and are never persisted here.
"""
from __future__ import annotations

import base64
import json
import re
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
from core.events import Severity, Visibility
from core.registry import register
from modules.base import DetectionModule


_FEATURES = {
    "redness", "discoloration", "swelling", "scaling", "blistering",
    "rash-like texture", "dryness", "lesion", "bruising", "irritation",
}
_TOPICS = {
    "itching", "pain", "duration", "spreading", "fever_unwell",
    "new_medication", "new_product_exposure", "blisters",
}
_QUALITY = {"poor", "fair", "good"}
_SCHEMA_TEXT = """Return exactly one JSON object with this schema:
{
  "image_quality": "poor|fair|good",
  "sufficient_skin_visible": true,
  "finding_present": false,
  "visible_features": ["redness"],
  "body_region": "left forearm",
  "confidence": 0.0,
  "possible_conditions": ["private hypothesis"],
  "follow_up_topics": ["itching", "duration"]
}
Allowed visible_features: redness, discoloration, swelling, scaling,
blistering, rash-like texture, dryness, lesion, bruising, irritation.
Allowed follow_up_topics: itching, pain, duration, spreading, fever_unwell,
new_medication, new_product_exposure, blisters.
Use possible_conditions only for uncertain internal hypotheses. Do not infer a
condition when the image is unclear. Do not include prose outside the JSON."""


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

    def private_value(self) -> dict[str, Any]:
        """Return JSON-safe private context for the agent."""
        return {
            "image_quality": self.image_quality,
            "visible_features": list(self.visible_features),
            "body_region": self.body_region,
            "confidence": self.confidence,
            "possible_conditions": list(self.possible_conditions),
            "follow_up_topics": list(self.follow_up_topics),
        }


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


def validate_analysis(raw: Any, min_confidence: float = 0.35) -> SkinAnalysis:
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
    return SkinAnalysis(quality, sufficient, finding, features, region,
                        round(confidence, 3), conditions, topics)


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
    backoff_base = 15.0
    backoff_max = 900.0

    def __init__(self, **params):
        super().__init__(**params)
        self._key = nvidia_api_key()
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="skin-vision")
        self._pending: Future | None = None
        self._pending_stage: str | None = None
        self._last_scan = -1e9
        self._next_allowed = -1e9
        self._failures = 0
        self._preliminary: SkinAnalysis | None = None
        self._preliminary_at = 0.0
        self._awaiting_closeup = False
        self._sampling_closeup = False
        self._best_frame: np.ndarray | None = None
        self._best_sharpness = -1.0
        self._elicitation = ElicitationState.instance()
        if self.consent and self._key:
            print(f"[skin-vision] cloud screening enabled (model {self.model})")
        elif self.consent:
            print("[skin-vision] consent given but NVIDIA_API_KEY is missing; disabled")
        else:
            print("[skin-vision] cloud screening disabled (use --enable-cloud-skin)")

    @property
    def available(self) -> bool:
        """Whether this run has both explicit consent and credentials."""
        return bool(self.consent and self._key and self.endpoint and self.model)

    def _encode(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > int(self.max_image_dim):
            scale = float(self.max_image_dim) / longest
            frame = cv2.resize(frame, (max(1, int(w * scale)),
                                       max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return encoded.tobytes()

    def _prompt(self, stage: str, previous: SkinAnalysis | None) -> str:
        if stage == "preliminary":
            task = ("Screen the visible person for an obvious possible skin change. "
                    "This is a low-confidence screening step, not a diagnosis. "
                    "Name the body region precisely enough to request a close-up.")
        else:
            task = ("Inspect this user-provided close-up for visible skin changes. "
                    "Be conservative and non-diagnostic.")
            if previous is not None:
                task += (f" The preliminary frame indicated {', '.join(previous.visible_features)} "
                         f"around {previous.body_region}.")
        return task + "\n\n" + _SCHEMA_TEXT

    def _call_api(self, jpeg: bytes, stage: str,
                  previous: SkinAnalysis | None) -> SkinAnalysis:
        image = base64.b64encode(jpeg).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt(stage, previous)},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + image}},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 700,
        }
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=float(self.request_timeout)) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SkinVisionAPIError(f"HTTP {exc.code}", status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SkinVisionAPIError(type(exc).__name__) from exc
        try:
            content = body["choices"][0]["message"]["content"]
            raw = _extract_json(content)
            return validate_analysis(raw, float(self.min_confidence))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SkinVisionAPIError("invalid structured response") from exc

    def _submit(self, frame: np.ndarray, stage: str, now: float) -> None:
        if self._pending is not None:
            return
        jpeg = self._encode(frame)
        previous = self._preliminary
        self._pending_stage = stage
        self._pending = self._executor.submit(self._call_api, jpeg, stage, previous)
        if stage == "preliminary":
            self._last_scan = now

    def _reset_closeup(self) -> None:
        self._preliminary = None
        self._preliminary_at = 0.0
        self._awaiting_closeup = False
        self._sampling_closeup = False
        self._best_frame = None
        self._best_sharpness = -1.0

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
            if not analysis.finding_present:
                return []
            self._preliminary = analysis
            self._preliminary_at = now
            self._awaiting_closeup = True
            value = analysis.private_value()
            value["closeup_seconds"] = float(self.closeup_seconds)
            return [self.result(
                "closeup_request", value,
                confidence=min(analysis.confidence, 0.55),
                severity=Severity.NOTICE, ttl=45.0,
                visibility=Visibility.AGENT_ONLY)]
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
            ttl=120.0)
        private = self.result(
            "analysis", analysis.private_value(),
            confidence=min(analysis.confidence, 0.65),
            severity=Severity.NOTICE, ttl=120.0,
            visibility=Visibility.AGENT_ONLY)
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
                self._submit(ctx.frame.copy(), "preliminary", now)
            except (ValueError, cv2.error) as exc:
                self._failure(now, exc)
        return results or None

    def close(self) -> None:
        """Cancel pending work without waiting on a slow network request."""
        if self._pending is not None:
            self._pending.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
