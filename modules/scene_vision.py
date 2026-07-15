"""Opt-in background NVIDIA room, object, activity, and hazard summaries."""
from __future__ import annotations

import json
import re
import uuid
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from agent.env import nvidia_api_key
from core.capabilities import CapabilityRegistry, CapabilityStatus
from core.events import PersistencePolicy, Result, Severity
from core.registry import register
from integrations.nvidia_vlm import NvidiaVLMClient
from modules.base import DetectionModule


@dataclass(frozen=True)
class SceneAnalysis:
    """Validated public scene description with bounded topic suggestions."""
    objects: tuple[str, ...]
    activities: tuple[str, ...]
    locations: tuple[str, ...]
    hazards: tuple[str, ...]
    quality: float
    confidence: float
    topics: tuple[str, ...]


def _items(value: Any, limit: int = 8) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item)).strip().lower()[:80]
        if text and text not in out:
            out.append(text)
    return tuple(out[:limit])


def validate_scene(raw: Any) -> SceneAnalysis:
    """Reject loose VLM output and normalize confidence and quality."""
    if not isinstance(raw, dict):
        raise ValueError("scene response must be an object")
    required = {"objects", "activities", "locations", "visible_hazards",
                "quality", "confidence", "conversation_topics"}
    if not required.issubset(raw):
        raise ValueError("scene response is missing required fields")
    if any(not isinstance(raw[key], list) for key in
           ("objects", "activities", "locations", "visible_hazards", "conversation_topics")):
        raise ValueError("scene collection fields must be lists")
    quality = max(0.0, min(1.0, float(raw.get("quality", 0))))
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
    return SceneAnalysis(_items(raw.get("objects")), _items(raw.get("activities")),
                         _items(raw.get("locations"), 3), _items(raw.get("visible_hazards"), 5),
                         quality, confidence, _items(raw.get("conversation_topics"), 3))


_PROMPT = """Describe only visibly supported room context. Do not identify people, infer health,
or diagnose. Return exactly JSON: {"objects":[],"activities":[],"locations":[],
"visible_hazards":[],"quality":0.0,"confidence":0.0,"conversation_topics":[]}.
Look for eating, drinking, reading, exercise, rest, cooking, leaving; cups, meals,
medication containers, mobility aids, glasses, phones, blankets; clutter, spills,
poor lighting, blocked paths, open exterior doors, and unattended cooking context."""


@register("scene_vision")
class SceneVision(DetectionModule):
    """Infrequent one-flight cloud scene analysis with exponential backoff."""
    name = "scene_vision"
    interval = 0.0
    consent = False
    endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
    model = "meta/llama-3.2-11b-vision-instruct"
    scan_interval = 90.0
    request_timeout = 25.0
    max_image_dim = 1024
    jpeg_quality = 80
    backoff_base = 15.0
    backoff_max = 900.0
    location = "unspecified"
    window_frames = 3
    window_spacing = 0.5

    def __init__(self, **params):
        super().__init__(**params)
        key = nvidia_api_key()
        self.available = bool(self.consent and key)
        self._client = NvidiaVLMClient(key or "", self.endpoint, self.model,
                                       self.request_timeout, self.max_image_dim, self.jpeg_quality)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-vision")
        self._pending: Future | None = None
        self._last_scan = self._next_allowed = -1e9
        self._failures = 0
        self._capture_frames: list[bytes] = []
        self._capture_started: float | None = None
        self._last_capture_frame = -1e9
        self._hazard_history: deque[tuple[float, tuple[str, ...]]] = deque(maxlen=8)
        status = CapabilityStatus.READY if self.available else CapabilityStatus.UNAVAILABLE
        detail = "consented background VLM" if self.available else "requires --enable-cloud-scene and key"
        CapabilityRegistry.instance().set("nvidia_scene", "cloud", status, detail)

    def _analyze(self, images: list[bytes]) -> SceneAnalysis:
        content = self._client.request(_PROMPT, images)
        if isinstance(content, list):
            content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        text = str(content).strip()
        return validate_scene(json.loads(text[text.find("{"):text.rfind("}") + 1]))

    def process(self, ctx):
        """Poll completed inference and schedule the next in-memory still."""
        now, out = ctx.timestamp, []
        if self._pending is not None and self._pending.done():
            try:
                scene = self._pending.result()
                correlation_id = uuid.uuid4().hex
                self._failures, self._next_allowed = 0, now
                CapabilityRegistry.instance().set("nvidia_scene", "cloud",
                    CapabilityStatus.READY, "consented background VLM")
                base = {"source": "nvidia_scene", "quality": scene.quality,
                        "location": self.location, "conversation_tags": scene.topics,
                        "persistence": PersistencePolicy.EVENT,
                        "correlation_id": correlation_id}
                if scene.objects:
                    out.append(Result(self.name, "objects", list(scene.objects), scene.confidence,
                                      Severity.INFO, "Visible objects: " + ", ".join(scene.objects), **base))
                if scene.activities:
                    out.append(Result(self.name, "activities", list(scene.activities), scene.confidence,
                                      Severity.INFO, "Visible activities: " + ", ".join(scene.activities), **base))
                if scene.hazards:
                    self._hazard_history.append((now, scene.hazards))
                    recent = [hazard for when, hazards in self._hazard_history
                              if now - when <= 10 * 60 for hazard in hazards]
                    confirmed = [hazard for hazard, count in Counter(recent).items() if count >= 2]
                    severity = Severity.NOTICE if confirmed else Severity.INFO
                    label = confirmed or list(scene.hazards)
                    message = ("Repeated visible context to check: " if confirmed
                               else "Unconfirmed visible context: ") + ", ".join(label)
                    out.append(Result(self.name, "hazards", label, scene.confidence,
                                      severity, message, **base))
            except Exception as exc:  # noqa: BLE001
                self._failures += 1
                self._next_allowed = now + min(self.backoff_max,
                                                self.backoff_base * (2 ** (self._failures - 1)))
                CapabilityRegistry.instance().set("nvidia_scene", "cloud",
                    CapabilityStatus.DEGRADED, f"backoff after {type(exc).__name__}")
            self._pending = None
        due = (self.available and self._pending is None and now >= self._next_allowed
               and now - self._last_scan >= self.scan_interval)
        if due and self._capture_started is None:
            self._capture_started = now
            self._capture_frames = []
            self._last_capture_frame = -1e9
        if self._capture_started is not None and self._pending is None \
                and now - self._last_capture_frame >= self.window_spacing:
            self._capture_frames.append(self._client.encode(ctx.frame))
            self._last_capture_frame = now
            if len(self._capture_frames) >= max(1, int(self.window_frames)):
                images = self._capture_frames
                self._capture_frames = []
                self._capture_started = None
                self._pending = self._executor.submit(self._analyze, images)
                self._last_scan = now
        return out or None

    def close(self) -> None:
        """Cancel pending work without retaining image payloads."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._capture_frames.clear()
