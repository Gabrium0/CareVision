"""Structured, privacy-aware context and control contracts for conversation.

The conversational provider may see every live result intended for the agent,
but it never receives an opaque Python object or an unlabelled detector string.
This module converts observations into bounded data records, ranks fresh topics,
and keeps proposed runtime actions behind an explicit confirmation gate.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from core.events import Result, Severity, Visibility


_ORDER = {Severity.INFO: 0, Severity.NOTICE: 1,
          Severity.WARNING: 2, Severity.ALERT: 3}
_SENSITIVE_KEYS = ("image", "frame", "audio", "media", "base64", "data_url",
                   "credential", "password", "token", "secret", "api_key")
_WORD = re.compile(r"[a-z0-9_]+")


def _bounded(value: Any, depth: int = 0) -> Any:
    """Return small JSON-safe data while refusing media- and secret-like fields."""
    if depth > 3:
        return "<bounded>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, str):
        low = value.lower()
        if low.startswith("data:") or len(value) > 2000:
            return "<redacted>"
        return " ".join(value.split())[:500]
    if isinstance(value, dict):
        out = {}
        for key, item in list(value.items())[:24]:
            name = str(key)[:80]
            if any(term in name.lower() for term in _SENSITIVE_KEYS):
                continue
            out[name] = _bounded(item, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_bounded(item, depth + 1) for item in list(value)[:24]]
    return str(type(value).__name__)


def guard_agent_only_speech(text: str, items: list["ContextItem"],
                            fallback: str) -> tuple[str, str | None]:
    """Refuse speech that repeats a private hypothesis as a person-facing fact."""
    low = text.lower()
    private_terms: set[str] = set()
    for item in items:
        if item.visibility != Visibility.AGENT_ONLY.value:
            continue
        raw = f"{item.value} {item.message}".lower()
        private_terms.update(word for word in _WORD.findall(raw) if len(word) >= 5)
    # Common connective words carry no private meaning and would over-block.
    private_terms -= {"there", "their", "about", "could", "would", "uncertain",
                      "visual", "model", "finding", "person", "observation"}
    asserted = any(term in low for term in private_terms) and any(
        phrase in low for phrase in ("you have", "you are", "it is", "looks like",
                                     "i can see", "i noticed", "appears to be"))
    if asserted:
        return fallback, "agent_only_assertion_blocked"
    return text, None


@dataclass(frozen=True)
class ConversationTurn:
    """One session-only utterance passed to a conversational provider."""
    role: str
    text: str
    timestamp: float


@dataclass(frozen=True)
class ContextItem:
    """One bounded observation with provenance and explicit privacy semantics."""
    id: str
    kind: str
    subject_id: str
    module: str
    key: str
    value: Any
    message: str
    confidence: float | None
    quality: float | None
    severity: str
    source: str
    visibility: str
    location: str | None
    timestamp: float
    expires_at: float | None
    freshness_seconds: float
    uncertainty: str | None = None

    def prompt_record(self) -> dict:
        """Return provider-safe structured data, never executable instructions."""
        data = asdict(self)
        data["value"] = _bounded(self.value)
        data["message"] = _bounded(self.message)
        return data


@dataclass
class AgentContext:
    """Bounded context selected for one model turn."""
    items: list[ContextItem] = field(default_factory=list)
    turns: list[ConversationTurn] = field(default_factory=list)
    workflows: list[dict] = field(default_factory=list)
    capabilities: list[dict] = field(default_factory=list)
    recent_events: list[dict] = field(default_factory=list)
    history_trends: list[dict] = field(default_factory=list)
    selected_item_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProposedAction:
    """Allowlisted action that remains inert until the person confirms it."""
    action: str
    target: str | None
    reason: str
    created_at: float
    expires_at: float


@dataclass
class AgentResponse:
    """Provider-neutral result for one conversational turn."""
    text: str
    cited_context_ids: list[str] = field(default_factory=list)
    proposed_action: ProposedAction | None = None
    provider_status: str = "unknown"
    fallback_reason: str | None = None


class ConversationalProvider(Protocol):
    """Minimal provider contract; providers never receive executable callbacks."""
    def respond(self, messages: list[dict], context_items: list[ContextItem],
                image=None) -> AgentResponse | None: ...


@dataclass
class TopicCandidate:
    """A fresh observation that may be raised at a natural conversation gap."""
    id: str
    context_id: str
    signature: str
    prompt: str
    fallback: str
    score: float
    created_at: float
    expires_at: float
    visibility: str
    status: str = "queued"
    suppression_reason: str | None = None


class AgentContextBroker:
    """Make all approved agent data queryable and select relevant bounded context."""
    def __init__(self, max_items: int = 64, max_chars: int = 14_000):
        self.max_items = max(8, int(max_items))
        self.max_chars = max(2000, int(max_chars))
        self._items: dict[str, ContextItem] = {}
        self._latest: dict[tuple[str, str, str], str] = {}
        self._last_selected: list[str] = []
        self._vision_item: ContextItem | None = None

    @staticmethod
    def from_result(result: Result, now: float) -> ContextItem:
        private = result.visibility == Visibility.AGENT_ONLY
        item_id = (f"result:{result.subject_id}:{result.module}:{result.key}:"
                   f"{int(result.timestamp * 1000)}")
        return ContextItem(
            id=item_id, kind="live_result", subject_id=result.subject_id,
            module=result.module, key=result.key, value=_bounded(result.value),
            message=str(_bounded(result.message) or ""),
            confidence=round(float(result.confidence), 4),
            quality=(round(float(result.quality), 4)
                     if result.quality is not None and math.isfinite(float(result.quality))
                     else None),
            severity=result.severity.value, source=str(result.source)[:120],
            visibility=result.visibility.value, location=result.location,
            timestamp=float(result.timestamp), expires_at=result.timestamp + result.ttl,
            freshness_seconds=max(0.0, now - result.timestamp),
            uncertainty=("Private model hypothesis: uncertain, ephemeral, and not a fact. "
                         "Use only to ask a gentle clarifying question."
                         if private else None))

    def ingest(self, snapshot: list[Result], now: float | None = None) -> None:
        """Refresh the live catalogue, including agent-only observations."""
        now = time.time() if now is None else now
        active = {}
        latest = {}
        for result in snapshot:
            if now - result.timestamp > result.ttl:
                continue
            item = self.from_result(result, now)
            active[item.id] = item
            latest[(item.subject_id, item.module, item.key)] = item.id
        if self._vision_item is not None and (self._vision_item.expires_at or 0) > now:
            active[self._vision_item.id] = self._vision_item
            latest[(self._vision_item.subject_id, self._vision_item.module,
                    self._vision_item.key)] = self._vision_item.id
        elif self._vision_item is not None:
            self._vision_item = None
        self._items, self._latest = active, latest

    def add_vision_observation(self, text: str, now: float | None = None,
                               ttl: float = 45.0) -> ContextItem | None:
        """Store only a short-lived caption; never retain its source frame."""
        now = time.time() if now is None else now
        clean = str(_bounded(text) or "").strip()
        if not clean or clean == "<redacted>":
            return None
        self._vision_item = ContextItem(
            id=f"vision:primary:{int(now * 1000)}", kind="vision_summary",
            subject_id="primary", module="conversation_vision", key="scene_summary",
            value=clean, message=clean, confidence=None, quality=None,
            severity=Severity.INFO.value, source="moondream_periodic_vision",
            visibility=Visibility.AGENT_ONLY.value, location=None, timestamp=now,
            expires_at=now + ttl, freshness_seconds=0.0,
            uncertainty="Ephemeral cloud vision summary; verify before relying on it.")
        return self._vision_item

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {word for word in _WORD.findall(text.lower()) if len(word) > 2}

    def build(self, query: str, turns: list[ConversationTurn], *,
              workflows: list[dict] | None = None,
              capabilities: list[dict] | None = None,
              recent_events: list[dict] | None = None,
              history_trends: list[dict] | None = None) -> AgentContext:
        """Select relevant context first, then fill remaining budget by salience."""
        terms = self._terms(query)
        ranked = []
        supplemental: list[ContextItem] = []
        now = time.time()
        groups = (("workflow", workflows or []),
                  ("capability", capabilities or []),
                  ("event_summary", recent_events or []),
                  ("history_trend", history_trends or []))
        for kind, rows in groups:
            for index, row in enumerate(rows[:64]):
                safe = _bounded(row)
                timestamp = (float(row.get("timestamp", now))
                             if isinstance(row, dict) else now)
                module = str(row.get("module") or kind) if isinstance(row, dict) else kind
                key = str(row.get("key") or row.get("name") or index) \
                    if isinstance(row, dict) else str(index)
                supplemental.append(ContextItem(
                    id=f"{kind}:{module}:{key}:{int(timestamp * 1000)}",
                    kind=kind, subject_id=str(row.get("subject_id", "primary"))
                    if isinstance(row, dict) else "primary",
                    module=module, key=key, value=safe, message="",
                    confidence=None, quality=None, severity=Severity.INFO.value,
                    source=kind, visibility=Visibility.PUBLIC.value, location=None,
                    timestamp=timestamp, expires_at=None,
                    freshness_seconds=max(0.0, now - timestamp)))
        for item in [*self._items.values(), *supplemental]:
            haystack = f"{item.module} {item.key} {item.message} {item.value}".lower()
            overlap = len(terms & self._terms(haystack))
            severity = {"info": 0, "notice": 1, "warning": 2, "alert": 3}.get(
                item.severity, 0)
            private_penalty = 0.2 if item.visibility == "agent_only" else 0.0
            score = overlap * 10 + severity * 2 + (item.confidence or 0) - private_penalty
            ranked.append((score, -item.freshness_seconds, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected, chars = [], 0
        for _score, _freshness, item in ranked:
            size = len(json.dumps(item.prompt_record(), ensure_ascii=True))
            if selected and (len(selected) >= self.max_items or chars + size > self.max_chars):
                continue
            selected.append(item)
            chars += size
        self._last_selected = [item.id for item in selected]
        return AgentContext(
            items=selected, turns=turns[-12:],
            workflows=[_bounded(row) for row in (workflows or [])[:8]],
            capabilities=[_bounded(row) for row in (capabilities or [])[:64]],
            recent_events=[_bounded(row) for row in (recent_events or [])[-12:]],
            history_trends=[_bounded(row) for row in (history_trends or [])[:24]],
            selected_item_ids=list(self._last_selected))

    def items(self) -> list[ContextItem]:
        """Return the current immutable item records for local ranking."""
        return list(self._items.values())

    def diagnostics(self) -> dict:
        """Return counts and identifiers only; never private values or dialogue."""
        counts: dict[str, int] = {}
        for item in self._items.values():
            key = f"{item.kind}:{item.visibility}"
            counts[key] = counts.get(key, 0) + 1
        return {"catalogue_items": len(self._items), "counts": counts,
                "selected_item_ids": list(self._last_selected),
                "vision_summary_present": self._vision_item is not None}


class TopicQueue:
    """Rank changed observations without granting the model alert authority."""
    def __init__(self, repeat_cooldown: float = 600.0, max_topics: int = 32):
        self.repeat_cooldown = repeat_cooldown
        self.max_topics = max_topics
        self._seen: dict[tuple[str, str, str], str] = {}
        self._topics: list[TopicCandidate] = []
        self._last_raised: dict[str, float] = {}

    def observe(self, items: list[ContextItem], now: float,
                excluded: frozenset = frozenset()) -> None:
        """Queue changed observations that no other layer already speaks for.

        `excluded` holds `(module, key)` pairs owned by the declarative topic
        table or a corroboration follow-up rule (see
        `agent/topics.py::covered_keys`). Those signals already have
        hand-authored wording, so promoting them here as well would say the
        same thing twice in two different voices. Everything else — post-answer
        steering, the agent-only path, and unspec'd signals — is unaffected.
        """
        for item in items:
            if item.kind not in ("live_result", "vision_summary"):
                continue
            if (item.module, item.key) in excluded:
                continue
            if item.severity == Severity.ALERT.value or not item.message:
                continue
            if item.kind == "live_result" and item.severity == Severity.INFO.value:
                # INFO rows remain queryable context, but routine status/metrics
                # are too noisy to initiate unsolicited conversation.
                continue
            if item.confidence is not None and item.confidence < 0.35:
                continue
            if item.quality is not None and item.quality < 0.25:
                continue
            stable_key = (item.subject_id, item.module, item.key)
            signature = json.dumps(_bounded(item.value), sort_keys=True, ensure_ascii=True)
            if self._seen.get(stable_key) == signature:
                continue
            self._seen[stable_key] = signature
            topic_id = f"{item.subject_id}:{item.module}:{item.key}"
            if now - self._last_raised.get(topic_id, -1e9) < self.repeat_cooldown:
                continue
            severity = {"info": 0, "notice": 1, "warning": 2}.get(item.severity, 0)
            private = item.visibility == Visibility.AGENT_ONLY.value
            # Turn an observation into ONE warm, specific question TO the person,
            # never a robotic "By the way, <label> would you like to talk about
            # that?". A behavioural cue (e.g. frequent face touching) should sound
            # like caring concern ("Is your face feeling itchy or sore?"), and a
            # low-confidence one must never be named as fact.
            prompt = ("Gently turn what you have noticed into ONE warm, caring "
                      "question about how the person feels right now, spoken to them "
                      "as 'you'. Do not state the observation as fact or name it, and "
                      "never say 'would you like to talk about that'."
                      if private else
                      "Turn what you have noticed into ONE warm, specific, caring "
                      "question spoken to the person as 'you' — e.g. if they keep "
                      "touching their face, kindly ask whether it feels itchy or sore. "
                      "Do not state the observation as a label and never say 'would "
                      "you like to talk about that'.")
            # Fallback (used only if the model output is rejected): a gentle,
            # non-naming check-in rather than reading the raw observation aloud.
            fallback = "By the way, how are you feeling right now — is anything bothering you?"
            self._topics.append(TopicCandidate(
                id=f"topic:{topic_id}:{int(now * 1000)}", context_id=item.id,
                signature=topic_id, prompt=prompt, fallback=fallback,
                score=severity * 10 + (item.confidence or 0.4) +
                      (item.quality if item.quality is not None else 0.5),
                created_at=now, expires_at=item.expires_at or now + 45,
                visibility=item.visibility))
        self._topics = sorted(
            [topic for topic in self._topics if topic.expires_at > now],
            key=lambda topic: (topic.status == "queued", topic.score),
            reverse=True)[:self.max_topics]

    def next(self, now: float) -> TopicCandidate | None:
        self._topics = [topic for topic in self._topics if topic.expires_at > now]
        return next((topic for topic in self._topics if topic.status == "queued"), None)

    def mark_raised(self, topic: TopicCandidate, now: float) -> None:
        topic.status = "raised"
        self._last_raised[topic.signature] = now

    def suppress(self, topic: TopicCandidate, reason: str) -> None:
        topic.status, topic.suppression_reason = "suppressed", reason[:120]

    def diagnostics(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        return {"queued": sum(topic.status == "queued" and topic.expires_at > now
                              for topic in self._topics),
                "topics": [{"id": topic.id, "context_id": topic.context_id,
                            "status": topic.status,
                            "suppression_reason": topic.suppression_reason}
                           for topic in self._topics[:8]]}


class VisionCadence:
    """Hold at most one ephemeral frame for change-driven cloud vision."""
    def __init__(self, heartbeat_seconds: float = 20.0,
                 min_spacing_seconds: float = 8.0, change_threshold: float = 12.0):
        self.heartbeat_seconds = heartbeat_seconds
        self.min_spacing_seconds = min_spacing_seconds
        self.change_threshold = change_threshold
        self._signature = None
        self._last_submitted = -1e9
        self._pending_frame = None
        self._in_flight = False
        self._requests = 0
        self._last_reason: str | None = None

    def observe(self, frame, *, active: bool, now: float) -> None:
        """Sample a tiny signature and copy a due frame; retain no frame otherwise."""
        if not active or frame is None or self._in_flight or self._pending_frame is not None:
            return
        try:
            import numpy as np
            sample = np.asarray(frame)[::32, ::32]
            if sample.ndim == 3:
                sample = sample.mean(axis=2)
            changed = (self._signature is None or sample.shape != self._signature.shape
                       or float(np.mean(np.abs(sample.astype(float) -
                                            self._signature.astype(float)))) >= self.change_threshold)
            heartbeat = now - self._last_submitted >= self.heartbeat_seconds
            spaced = now - self._last_submitted >= self.min_spacing_seconds
            self._signature = sample.copy()
            if spaced and (changed or heartbeat):
                self._pending_frame = np.asarray(frame).copy()
                self._last_reason = "change" if changed else "heartbeat"
        except Exception:
            self._last_reason = "frame_unavailable"

    def take_due(self):
        frame, self._pending_frame = self._pending_frame, None
        if frame is not None:
            self._in_flight = True
            self._requests += 1
        return frame

    def force_refresh(self) -> None:
        """Make the next active observation eligible for a heartbeat refresh."""
        self._last_submitted = -1e9

    def complete(self, now: float) -> None:
        self._in_flight = False
        self._last_submitted = now

    def diagnostics(self) -> dict:
        return {"heartbeat_seconds": self.heartbeat_seconds,
                "min_spacing_seconds": self.min_spacing_seconds,
                "in_flight": self._in_flight,
                "pending": self._pending_frame is not None,
                "requests": self._requests, "last_reason": self._last_reason,
                "ephemeral_frame_pending": self._pending_frame is not None,
                "raw_frame_persisted": False}
