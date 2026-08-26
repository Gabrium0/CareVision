"""Shared NVIDIA VLM transport with bounded payloads and safe diagnostics."""
from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


_ENCODING_LADDER = (
    (768, 80),
    (768, 70),
    (640, 75),
    (640, 65),
    (512, 70),
    (512, 60),
    (448, 55),
    (384, 60),
    (320, 50),
    (256, 45),
    (192, 40),
    (128, 30),
    (96, 25),
    (64, 15),
    (64, 1),
)
_NVIDIA_HOST = re.compile(r"(?:^|\.)nvidia\.com$", re.IGNORECASE)
_DATA_URL = re.compile(r"data:[^,\s\"']+,[^\s\"']+", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+[^\s,;]+", re.IGNORECASE)
_REQUEST_ID = re.compile(r"[^A-Za-z0-9._:/-]+")
_CREDENTIAL_SALT = secrets.token_bytes(16)
_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_ERROR_BYTES = 64 * 1024


def _sanitize_text(value: Any, limit: int = 300, secrets: tuple[str, ...] = ()) -> str:
    """Bound remote text and remove obvious credentials or inline media."""
    text = str(value) if value is not None else ""
    text = _DATA_URL.sub("[redacted-media]", text)
    text = _BEARER.sub("Bearer [redacted]", text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _safe_request_id(value: Any) -> str | None:
    """Return a bounded opaque identifier containing no response content."""
    if not isinstance(value, (str, int)):
        return None
    cleaned = _REQUEST_ID.sub("", str(value).strip())[:160]
    return cleaned or None


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except (AttributeError, TypeError):
        value = None
    return str(value).strip() if value is not None else None


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        try:
            status = response.getcode()
        except (AttributeError, TypeError):
            status = 200
    try:
        return int(status)
    except (TypeError, ValueError):
        return 200


def _request_id(headers: Any, body: Any = None) -> str | None:
    for name in ("NVCF-REQID", "X-Request-ID", "Request-ID"):
        value = _safe_request_id(_header(headers, name))
        if value:
            return value
    if isinstance(body, dict):
        for name in ("request_id", "requestId", "id"):
            value = _safe_request_id(body.get(name))
            if value:
                return value
    return None


def _retry_after(headers: Any, *, now: float | None = None,
                 maximum: float = 10.0) -> float | None:
    """Parse delta-seconds or an HTTP date and clamp it to a safe wait."""
    value = _header(headers, "Retry-After")
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            delay = parsed.timestamp() - (time.time() if now is None else float(now))
        except (TypeError, ValueError, OverflowError):
            return None
    if not np.isfinite(delay):
        return None
    return max(0.0, min(float(maximum), delay))


def _https_origin(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (parsed.scheme.lower() != "https" or parsed.username is not None
            or parsed.password is not None or not parsed.hostname):
        return None
    return ("https", parsed.hostname.lower(), 443 if port is None else port)


def _trusted_poll_url(value: Any, endpoint: str | None = None) -> str | None:
    """Accept only the provider boundary established by the endpoint."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    origin = _https_origin(candidate)
    if origin is None:
        return None
    _, hostname, port = origin
    endpoint_origin = _https_origin(endpoint) if endpoint else None
    if endpoint_origin is None:
        trusted = bool(_NVIDIA_HOST.search(hostname) and port == 443)
    else:
        _, endpoint_host, endpoint_port = endpoint_origin
        endpoint_is_nvidia = bool(
            _NVIDIA_HOST.search(endpoint_host) and endpoint_port == 443)
        trusted = (
            bool(_NVIDIA_HOST.search(hostname) and port == 443)
            if endpoint_is_nvidia else endpoint_origin == origin)
    if not trusted:
        return None
    return candidate


def _pending_url(headers: Any, body: Any,
                 endpoint: str | None = None) -> str | None:
    candidates: list[Any] = [_header(headers, "Location")]
    if isinstance(body, dict):
        candidates.extend((body.get("statusUrl"), body.get("status_url")))
    for candidate in candidates:
        trusted = _trusted_poll_url(candidate, endpoint)
        if trusted:
            return trusted
    return None


def _kind_for_status(status: int | None) -> tuple[str, bool]:
    if status in (401, 403):
        return "authentication", False
    if status == 413:
        return "payload_size", True
    if status == 429:
        return "rate_limit", True
    if status in (408, 425):
        return "timeout", True
    if status == 202:
        return "pending", True
    if status is not None and status >= 500:
        return "server", True
    return "client", False


def _credential_fingerprint(value: str) -> str:
    """Return a process-local, non-reversible coordinator key."""
    return hashlib.blake2s(
        str(value).encode("utf-8", errors="ignore"),
        key=_CREDENTIAL_SALT,
        digest_size=16,
    ).hexdigest()


class NvidiaVLMError(RuntimeError):
    """Sanitized failure metadata containing no credentials or payloads."""

    def __init__(self, message: str, status: int | None = None,
                 retryable: bool | None = None, *, kind: str | None = None,
                 retry_after: float | None = None,
                 request_id: str | None = None,
                 finish_reason: str | None = None):
        inferred_kind, inferred_retryable = _kind_for_status(status)
        self.kind = str(kind or inferred_kind)
        self.status = status
        self.retryable = (inferred_retryable if retryable is None
                          else bool(retryable))
        self.retry_after = (None if retry_after is None
                            else max(0.0, float(retry_after)))
        self.request_id = _safe_request_id(request_id)
        self.finish_reason = (
            _sanitize_text(finish_reason, 40)
            if finish_reason is not None else None)
        super().__init__(_sanitize_text(message) or self.kind)


@dataclass(frozen=True)
class NvidiaVLMResponse:
    """Sanitized metadata for one completed NVIDIA inference request."""

    content: Any
    status: int
    finish_reason: str | None
    request_id: str | None
    latency_ms: float
    poll_count: int
    retry_after: float | None = None
    encoded_image_bytes: int = 0
    queue_ms: float = 0.0


@dataclass(eq=False)
class _QueueTicket:
    purpose: str
    priority: int
    sequence: int
    enqueued_at: float
    credential_id: str
    reservation_token: int | None = None


def _purpose_class(purpose: str | None) -> tuple[str, int]:
    value = str(purpose or "passive").strip().lower()
    if value in {"manual", "manual_arm", "manual_arm_check"}:
        return "manual", 0
    if value in {"guided", "guided_closeup", "guided_close-up", "closeup"}:
        return "guided", 1
    return "passive", 2


class _NvidiaRequestCoordinator:
    """Process-wide priority, reservation, Retry-After, and auth gate."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active: _QueueTicket | None = None
        self._waiting: list[_QueueTicket] = []
        self._sequence = 0
        self._reservation_sequence = 0
        self._reservations: dict[int, tuple[str, int, float]] = {}
        self._retry_until = 0.0
        self._authentication_blocked: dict[str, int] = {}

    def _prune_reservations_locked(self, now: float) -> None:
        expired = [
            token for token, (_, _, deadline) in self._reservations.items()
            if deadline <= now
        ]
        for token in expired:
            self._reservations.pop(token, None)

    def reserve(self, purpose: str | None, deadline: float) -> int:
        purpose_class, priority = _purpose_class(purpose)
        with self._condition:
            self._prune_reservations_locked(time.monotonic())
            self._reservation_sequence += 1
            token = self._reservation_sequence
            self._reservations[token] = (
                purpose_class, priority, float(deadline))
            self._condition.notify_all()
            return token

    def cancel_reservation(self, token: int | None) -> None:
        if token is None:
            return
        with self._condition:
            self._reservations.pop(int(token), None)
            self._condition.notify_all()

    def note_error(self, error: NvidiaVLMError,
                   credential_id: str) -> None:
        with self._condition:
            if error.retry_after is not None:
                self._retry_until = max(
                    self._retry_until,
                    time.monotonic() + max(
                        0.0, min(10.0, float(error.retry_after))))
            if error.kind == "authentication":
                self._authentication_blocked[credential_id] = int(
                    error.status or 401)
            self._condition.notify_all()

    def acquire(self, purpose: str | None, deadline: float,
                credential_id: str,
                reservation_token: int | None = None,
                cancel_event: threading.Event | None = None) -> float:
        purpose_class, priority = _purpose_class(purpose)
        enqueued_at = time.monotonic()
        with self._condition:
            self._sequence += 1
            ticket = _QueueTicket(
                purpose_class, priority, self._sequence, enqueued_at,
                credential_id, reservation_token)
            self._waiting.append(ticket)
            while True:
                now = time.monotonic()
                self._prune_reservations_locked(now)
                if cancel_event is not None and cancel_event.is_set():
                    if ticket in self._waiting:
                        self._waiting.remove(ticket)
                    self._condition.notify_all()
                    raise NvidiaVLMError(
                        "request was cancelled",
                        retryable=False,
                        kind="client",
                    )
                auth_status = self._authentication_blocked.get(credential_id)
                if auth_status is not None:
                    if ticket in self._waiting:
                        self._waiting.remove(ticket)
                    self._condition.notify_all()
                    raise NvidiaVLMError(
                        "provider authentication is unavailable",
                        status=auth_status,
                        retryable=False,
                        kind="authentication",
                    )
                winner = min(
                    self._waiting,
                    key=lambda item: (item.priority, item.sequence),
                    default=None,
                )
                higher_reservations = [
                    reservation_deadline
                    for _, reservation_priority, reservation_deadline
                    in self._reservations.values()
                    if reservation_priority < priority
                ]
                retry_blocked = now < self._retry_until
                if (self._active is None and winner is ticket
                        and not higher_reservations
                        and not retry_blocked):
                    self._waiting.remove(ticket)
                    self._active = ticket
                    return max(0.0, (time.monotonic() - enqueued_at) * 1000.0)
                remaining = deadline - now
                if remaining <= 0:
                    if ticket in self._waiting:
                        self._waiting.remove(ticket)
                    self._condition.notify_all()
                    raise NvidiaVLMError(
                        "request queue deadline exceeded",
                        retryable=True,
                        kind="timeout",
                    )
                wake_after = remaining
                if retry_blocked:
                    wake_after = min(wake_after, self._retry_until - now)
                if higher_reservations:
                    wake_after = min(
                        wake_after, max(0.001, min(higher_reservations) - now))
                self._condition.wait(timeout=max(0.001, wake_after))

    def release(self) -> None:
        with self._condition:
            self._active = None
            self._condition.notify_all()

    def notify_waiters(self) -> None:
        """Wake queued requests so per-module cancellation is immediate."""
        with self._condition:
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        """Return categorical queue state only; no request content is retained."""
        with self._condition:
            now = time.monotonic()
            self._prune_reservations_locked(now)
            counts = {"manual": 0, "guided": 0, "passive": 0}
            for ticket in self._waiting:
                counts[ticket.purpose] += 1
            reservations = {"manual": 0, "guided": 0, "passive": 0}
            for purpose, _, _ in self._reservations.values():
                reservations[purpose] += 1
            return {
                "active": self._active.purpose if self._active else None,
                "queued": sum(counts.values()),
                "queued_by_purpose": counts,
                "reserved_by_purpose": reservations,
                "retry_after_seconds": round(max(
                    0.0, self._retry_until - now), 2),
                "authentication_blocked": bool(
                    self._authentication_blocked),
            }

    def reset_for_tests(self) -> None:
        """Clear categorical gate state after deterministic isolated tests."""
        with self._condition:
            if self._active is not None or self._waiting:
                raise RuntimeError("cannot reset an active coordinator")
            self._reservations.clear()
            self._retry_until = 0.0
            self._authentication_blocked.clear()
            self._condition.notify_all()


_COORDINATOR = _NvidiaRequestCoordinator()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward an Authorization header through an automatic redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _open_no_redirect(request: urllib.request.Request, timeout: float):
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def _sanitized_http_error(exc: urllib.error.HTTPError,
                          secrets: tuple[str, ...] = ()) -> str:
    """Return only NVIDIA's bounded error message, never its response envelope."""
    label = f"HTTP {exc.code}"
    try:
        raw = exc.read(_MAX_ERROR_BYTES + 1)
        if len(raw) > _MAX_ERROR_BYTES:
            return label
        body = json.loads(raw.decode("utf-8", errors="replace"))
        detail = body.get("error", {}).get("message")
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return label
    if not isinstance(detail, str):
        return label
    detail = _sanitize_text(detail, secrets=secrets)
    return f"{label}: {detail}" if detail else label


class NvidiaVLMClient:
    """Encode one in-memory JPEG and call an NVIDIA chat endpoint."""

    def __init__(self, api_key: str, endpoint: str, model: str,
                 timeout: float = 25, max_image_dim: int = 1024,
                 jpeg_quality: int = 85,
                 max_inline_image_bytes: int = 174080):
        self._key, self.endpoint, self.model = api_key, endpoint, model
        self._credential_id = _credential_fingerprint(api_key)
        self.timeout = max(0.1, float(timeout))
        self.max_image_dim = max(64, int(max_image_dim))
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.max_inline_image_bytes = max(1024, int(max_inline_image_bytes))

    @staticmethod
    def coordinator_diagnostics() -> dict[str, Any]:
        return _COORDINATOR.snapshot()

    @staticmethod
    def reserve_request(purpose: str, deadline: float) -> int:
        """Reserve logical priority across queueing, backoff, and retries."""
        return _COORDINATOR.reserve(purpose, deadline)

    @staticmethod
    def cancel_reservation(token: int | None) -> None:
        _COORDINATOR.cancel_reservation(token)

    @staticmethod
    def notify_cancellation() -> None:
        _COORDINATOR.notify_waiters()

    def _encoding_ladder(self, max_image_dim: int | None,
                         jpeg_quality: int | None) -> tuple[tuple[int, int], ...]:
        max_dim = min(768, self.max_image_dim if max_image_dim is None
                      else max(64, int(max_image_dim)))
        quality = min(80, self.jpeg_quality if jpeg_quality is None
                      else max(1, min(100, int(jpeg_quality))))
        steps: list[tuple[int, int]] = []
        for dimension, step_quality in _ENCODING_LADDER:
            candidate = (min(max_dim, dimension), min(quality, step_quality))
            if candidate not in steps:
                steps.append(candidate)
        return tuple(steps)

    def encode(self, frame: np.ndarray, *, max_image_dim: int | None = None,
               jpeg_quality: int | None = None,
               max_inline_image_bytes: int | None = None) -> bytes:
        """Adaptively resize and JPEG-encode one frame without touching disk."""
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
            raise NvidiaVLMError(
                "invalid image frame", retryable=False, kind="payload_size")
        target = (self.max_inline_image_bytes if max_inline_image_bytes is None
                  else max(1, int(max_inline_image_bytes)))
        height, width = frame.shape[:2]
        for max_dim, quality in self._encoding_ladder(max_image_dim, jpeg_quality):
            candidate = frame
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                candidate = cv2.resize(
                    frame,
                    (max(1, int(round(width * scale))),
                     max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(
                ".jpg", candidate, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                raise NvidiaVLMError(
                    "JPEG encoding failed", retryable=False, kind="payload_size")
            data = encoded.tobytes()
            if len(data) <= target:
                return data
        raise NvidiaVLMError(
            "image exceeds inline payload budget",
            status=413,
            retryable=True,
            kind="payload_size",
        )

    def _single_image(self, images: list[bytes]) -> bytes:
        if len(images) != 1:
            raise NvidiaVLMError(
                "exactly one composite image is required",
                retryable=False,
                kind="payload_size",
            )
        image = images[0]
        if not isinstance(image, (bytes, bytearray, memoryview)) or not image:
            raise NvidiaVLMError(
                "invalid image payload", retryable=False, kind="payload_size")
        data = bytes(image)
        if len(data) <= self.max_inline_image_bytes:
            return data
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise NvidiaVLMError(
                "image exceeds inline payload budget",
                status=413,
                retryable=True,
                kind="payload_size",
            )
        return self.encode(decoded)

    def _http_error(self, exc: urllib.error.HTTPError) -> NvidiaVLMError:
        status = int(exc.code)
        kind, retryable = _kind_for_status(status)
        headers = getattr(exc, "headers", None)
        return NvidiaVLMError(
            _sanitized_http_error(exc, (self._key,)),
            status,
            retryable,
            kind=kind,
            retry_after=_retry_after(headers),
            request_id=_request_id(headers),
        )

    @staticmethod
    def _network_error(exc: BaseException) -> NvidiaVLMError:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return NvidiaVLMError(
                "provider request timed out", retryable=True, kind="timeout")
        return NvidiaVLMError(
            "provider network request failed", retryable=True, kind="network")

    @staticmethod
    def _read_response(response: Any) -> tuple[int, Any, Any]:
        status = _response_status(response)
        headers = getattr(response, "headers", None)
        try:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except TypeError:  # lightweight deterministic response fakes
            raw = response.read()
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise NvidiaVLMError(
                "provider response exceeded the size limit",
                status=status,
                retryable=True,
                kind="schema_validation",
                request_id=_request_id(headers),
            )
        if not raw:
            return status, headers, None
        parsed: Any = None
        parse_failed = False
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, ValueError,
                json.JSONDecodeError):
            parse_failed = True
        if parse_failed:
            raw = b""
            raise NvidiaVLMError(
                "provider returned invalid JSON",
                status=status,
                retryable=True,
                kind="json_parse",
                request_id=_request_id(headers),
            )
        return status, headers, parsed

    @staticmethod
    def _content_is_empty(content: Any) -> bool:
        if content is None:
            return True
        if isinstance(content, str):
            return not content.strip()
        if isinstance(content, list):
            if not content:
                return True
            return not any(
                (isinstance(part, str) and part.strip())
                or (isinstance(part, dict)
                    and isinstance(part.get("text"), str)
                    and part["text"].strip())
                for part in content
            )
        return False

    def _validated_response(self, body: Any, status: int, headers: Any, *,
                            started: float, poll_count: int,
                            encoded_bytes: int, queue_ms: float) -> NvidiaVLMResponse:
        request_id = _request_id(headers, body)
        if not isinstance(body, dict):
            raise NvidiaVLMError(
                "provider response is not an object",
                status=status,
                retryable=True,
                kind="schema_validation",
                request_id=request_id,
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise NvidiaVLMError(
                "provider response has no choice",
                status=status,
                retryable=True,
                kind="schema_validation",
                request_id=request_id,
            )
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        finish_reason = (_sanitize_text(finish_reason, 40)
                         if finish_reason is not None else None)
        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise NvidiaVLMError(
                "provider response has no message content",
                status=status,
                retryable=True,
                kind="schema_validation",
                request_id=request_id,
                finish_reason=finish_reason,
            )
        content = message.get("content")
        if self._content_is_empty(content):
            raise NvidiaVLMError(
                "provider returned empty content",
                status=status,
                retryable=True,
                kind="empty_content",
                request_id=request_id,
                finish_reason=finish_reason,
            )
        return NvidiaVLMResponse(
            content=content,
            status=status,
            finish_reason=finish_reason,
            request_id=request_id,
            latency_ms=round((time.monotonic() - started) * 1000.0, 1),
            poll_count=poll_count,
            retry_after=_retry_after(headers),
            encoded_image_bytes=encoded_bytes,
            queue_ms=round(queue_ms, 1),
        )

    def request(self, prompt: str, images: list[bytes] | None = None,
                max_tokens: int = 700,
                response_format: dict[str, Any] | None = None,
                timeout: float | None = None, *, purpose: str = "passive",
                deadline: float | None = None,
                reservation_token: int | None = None,
                cancel_event: threading.Event | None = None) -> NvidiaVLMResponse:
        """Submit one composite image and return sanitized response metadata.

        ``deadline`` is an absolute ``time.monotonic()`` value. Queue time is
        charged to it, while ``timeout`` caps provider I/O after admission.
        """
        started = time.monotonic()
        attempt_timeout = (self.timeout if timeout is None
                           else max(0.1, float(timeout)))
        queue_deadline = (float(deadline) if deadline is not None
                          else started + max(60.0, attempt_timeout * 3.0))
        queue_ms = _COORDINATOR.acquire(
            purpose, queue_deadline, self._credential_id,
            reservation_token, cancel_event)
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise NvidiaVLMError(
                    "request was cancelled", retryable=False, kind="client")
            io_deadline = min(
                queue_deadline,
                time.monotonic() + attempt_timeout,
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            image = b""
            if images:
                image = self._single_image(images)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                               + base64.b64encode(image).decode("ascii")
                    },
                })
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "max_tokens": int(max_tokens),
            }
            if response_format is not None:
                payload["response_format"] = response_format
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            poll_url: str | None = None
            poll_count = 0
            redirect_count = 0
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise NvidiaVLMError(
                        "request was cancelled",
                        retryable=False,
                        kind="client",
                    )
                remaining = io_deadline - time.monotonic()
                if remaining <= 0:
                    raise NvidiaVLMError(
                        "provider request timed out",
                        retryable=True,
                        kind="timeout",
                    )
                current_request = request
                if poll_url is not None:
                    current_request = urllib.request.Request(
                        poll_url,
                        method="GET",
                        headers={
                            "Authorization": f"Bearer {self._key}",
                            "Accept": "application/json",
                        },
                    )
                transport_error: NvidiaVLMError | None = None
                try:
                    with _open_no_redirect(
                            current_request, timeout=max(0.1, remaining)) as response:
                        response_status = _response_status(response)
                        response_headers = getattr(response, "headers", None)
                        try:
                            status, headers, body = self._read_response(response)
                        except NvidiaVLMError:
                            header_poll = (
                                _trusted_poll_url(
                                    _header(response_headers, "Location"),
                                    self.endpoint)
                                if response_status == 202 else None)
                            if header_poll is None:
                                raise
                            status, headers, body = (
                                response_status, response_headers, None)
                except urllib.error.HTTPError as exc:
                    if poll_url is not None and int(exc.code) in {
                            301, 302, 303, 307, 308}:
                        candidate = _trusted_poll_url(
                            _header(getattr(exc, "headers", None), "Location"),
                            self.endpoint)
                        if candidate is not None and redirect_count < 8:
                            poll_url = candidate
                            redirect_count += 1
                            continue
                    transport_error = self._http_error(exc)
                    exc.__traceback__ = None
                    exc.__cause__ = None
                    exc.__context__ = None
                    try:
                        exc.close()
                    except (AttributeError, OSError):
                        pass
                except (urllib.error.URLError, TimeoutError, socket.timeout,
                        ConnectionError, OSError) as exc:
                    transport_error = self._network_error(exc)
                    exc.__traceback__ = None
                    exc.__cause__ = None
                    exc.__context__ = None
                if transport_error is not None:
                    _COORDINATOR.note_error(
                        transport_error, self._credential_id)
                    raise transport_error
                if status == 202:
                    candidate = _pending_url(headers, body, self.endpoint)
                    if candidate is not None:
                        poll_url = candidate
                    if poll_url is None:
                        error = NvidiaVLMError(
                            "provider response is still pending without a trusted status URL",
                            status=202,
                            retryable=True,
                            kind="pending",
                            retry_after=_retry_after(headers),
                            request_id=_request_id(headers, body),
                        )
                        _COORDINATOR.note_error(error, self._credential_id)
                        raise error
                    delay = _retry_after(headers)
                    if delay is not None:
                        _COORDINATOR.note_error(
                            NvidiaVLMError(
                                "provider response is pending",
                                status=202,
                                retryable=True,
                                kind="pending",
                                retry_after=delay,
                                request_id=_request_id(headers, body),
                            ),
                            self._credential_id,
                        )
                    delay = 0.25 if delay is None else delay
                    remaining = io_deadline - time.monotonic()
                    if delay > 0 and remaining > 0:
                        delay = min(delay, remaining)
                        if cancel_event is not None:
                            cancel_event.wait(delay)
                        else:
                            time.sleep(delay)
                    poll_count += 1
                    continue
                if status < 200 or status >= 300:
                    kind, retryable = _kind_for_status(status)
                    error = NvidiaVLMError(
                        f"HTTP {status}",
                        status,
                        retryable,
                        kind=kind,
                        retry_after=_retry_after(headers),
                        request_id=_request_id(headers, body),
                    )
                    _COORDINATOR.note_error(error, self._credential_id)
                    raise error
                return self._validated_response(
                    body,
                    status,
                    headers,
                    started=started,
                    poll_count=poll_count,
                    encoded_bytes=len(image),
                    queue_ms=queue_ms,
                )
        finally:
            _COORDINATOR.release()
