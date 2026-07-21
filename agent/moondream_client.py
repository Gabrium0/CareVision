"""Moondream Cloud natural-language generation with an offline fallback."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
import uuid

from agent.env import moondream_api_key

_PERSONA = (
    "You are a warm, calm companion robot for an elderly person who may live "
    "alone. You speak out loud in ONE short, natural sentence (max ~25 words), "
    "conversational and kind, never clinical or alarming. You do not give "
    "medical diagnoses. If you mention something you noticed, be gentle and "
    "offer, don't instruct."
)


class MoondreamClient:
    """Small credential-safe wrapper around Moondream's chat API."""

    endpoint = "https://api.moondream.ai/v1/chat/completions"

    def __init__(self, model: str = "moondream3.1-9B-A2B", enabled: bool = True,
                 timeout: float = 8.0):
        self.model = model
        self.timeout = float(timeout)
        self._key = moondream_api_key()
        self.available = bool(self._key)
        self._enabled = bool(enabled)
        self._lock = threading.RLock()
        self._generation_requests = 0
        self._classification_requests = 0
        self._failures = 0
        self._successes = 0
        self._last_request_at: float | None = None
        self._last_error: str | None = None
        self._last_http_status: int | None = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._authorization_failed = False
        self._retryable = True
        self._lifecycle = "configured" if self.available else "unconfigured"
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="moondream")
        self._async: dict[str, Future] = {}
        self._closed = False
        if self.available:
            print(f"[agent/moondream] configured (model {self.model}); authorization pending")
        else:
            print("[agent/moondream] no X-Moondream-Auth in .env; using templated speech")

    @property
    def enabled(self) -> bool:
        """Whether the user currently permits Moondream API requests."""
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> bool:
        """Enable or disable future API requests."""
        with self._lock:
            was_enabled = self._enabled
            self._enabled = bool(enabled)
            if self._enabled and not was_enabled:
                self._reset_authorization_latch()
            return self._enabled

    def toggle_enabled(self) -> bool:
        """Toggle future API requests; an in-flight request may finish."""
        with self._lock:
            self._enabled = not self._enabled
            if self._enabled:
                self._reset_authorization_latch()
            return self._enabled

    def _reset_authorization_latch(self) -> None:
        """Caller holds `_lock`; an explicit off/on toggle authorizes retry."""
        self._authorization_failed = False
        self._retryable = True
        self._circuit_open_until = 0.0
        self._consecutive_failures = 0
        self._lifecycle = "configured"

    def status(self) -> dict:
        """Return credential-free operational diagnostics."""
        with self._lock:
            return {
                "provider": "moondream",
                "available": self.available,
                "enabled": self._enabled,
                "active": self.available and self._enabled,
                "model": self.model,
                "status": self._lifecycle,
                "generation_requests": self._generation_requests,
                "classification_requests": self._classification_requests,
                "failures": self._failures,
                "successes": self._successes,
                "success_rate": round(self._successes /
                                      max(1, self._successes + self._failures), 3),
                "last_request_at": self._last_request_at,
                "last_error": self._last_error,
                "last_http_status": self._last_http_status,
                "consecutive_failures": self._consecutive_failures,
                "authorization_failed": self._authorization_failed,
                "retryable": self._retryable,
                "circuit_state": ("authorization_failed" if self._authorization_failed
                                  else "open" if time.time() < self._circuit_open_until
                                  else "closed"),
                "retry_after_seconds": round(max(0.0, self._circuit_open_until - time.time()), 1),
                "in_flight": any(not future.done() for future in self._async.values()),
            }

    def _begin_request(self, kind: str) -> str | None:
        with self._lock:
            if (self._closed or not self._enabled or not self.available or not self._key
                    or self._authorization_failed
                    or time.time() < self._circuit_open_until):
                return None
            if kind == "generation":
                self._generation_requests += 1
            else:
                self._classification_requests += 1
            self._last_request_at = time.time()
            return self._key

    def _record_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failures += 1
            status = int(exc.code) if isinstance(exc, urllib.error.HTTPError) else None
            self._last_http_status = status
            retryable = status is None or status in (408, 425, 429) or status >= 500
            self._retryable = retryable
            label = f"HTTP {status}" if status is not None else type(exc).__name__
            if status in (401, 403):
                self._authorization_failed = True
                self._retryable = False
                self._last_error = f"{label}: authorization_failed"
                self._lifecycle = "authorization_failed"
            else:
                self._last_error = f"{label}: {'retryable' if retryable else 'request rejected'}"[:240]
                self._lifecycle = "degraded"
            self._consecutive_failures += 1
            if retryable and self._consecutive_failures >= 3:
                delay = min(300.0, 15.0 * (2 ** (self._consecutive_failures - 3)))
                self._circuit_open_until = time.time() + delay

    def _record_success(self) -> None:
        with self._lock:
            self._successes += 1
            self._last_error = None
            self._last_http_status = None
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._lifecycle = "ready"
            self._retryable = True

    def _complete(self, prompt: str, kind: str) -> str | None:
        key = self._begin_request(kind)
        if key is None:
            return None
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_completion_tokens": 120,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=payload, method="POST",
            headers={
                "X-Moondream-Auth": key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Moondream's edge currently rejects Python-urllib's default
                # user agent (HTTP 403 / code 1010) before API authentication.
                "User-Agent": "Humanoid-Care-Agent/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            self._record_success()
            return text.strip() if isinstance(text, str) and text.strip() else None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                OSError, KeyError, IndexError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            self._record_failure(exc)
            print(f"[agent/moondream] {kind} request failed ({type(exc).__name__})")
            return None

    def submit_generation(self, intent: str, context: str,
                          detail: str = "") -> str | None:
        """Submit one generation without blocking the caller; one flight only."""
        with self._lock:
            self._async = {key: future for key, future in self._async.items()
                           if not future.done()}
            if (self._closed or not self._enabled or not self.available
                    or self._authorization_failed
                    or time.time() < self._circuit_open_until
                    or any(not future.done() for future in self._async.values())):
                return None
            request_id = uuid.uuid4().hex
            self._async[request_id] = self._executor.submit(
                self.generate, intent, context, detail)
            return request_id

    def poll_generation(self, request_id: str) -> tuple[bool, str | None]:
        with self._lock:
            future = self._async.get(request_id)
        if future is None:
            return True, None
        if not future.done():
            return False, None
        try:
            value = future.result()
        except BaseException as exc:  # noqa: BLE001
            self._record_failure(exc if isinstance(exc, Exception) else RuntimeError(type(exc).__name__))
            value = None
        with self._lock:
            self._async.pop(request_id, None)
        return True, value

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = list(self._async.values())
            self._async.clear()
        for future in futures:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def classify_answer(self, question: str, answer: str) -> str | None:
        """Classify a reply as confirmed, denied, or unclear."""
        prompt = (
            "A companion robot asked this gentle yes/no check-in: "
            f"{question!r}. The person replied: {answer!r}. Answer with exactly "
            "one word: confirmed, denied, or unclear."
        )
        text = self._complete(prompt, "classification")
        if text is None:
            return None
        word = text.strip().lower().split(maxsplit=1)[0].strip(".,:;!?")
        return word if word in {"confirmed", "denied", "unclear"} else None

    def generate(self, intent: str, context: str, detail: str = "") -> str | None:
        """Generate one short spoken line, or None for the local template path."""
        prompt = (f"{_PERSONA}\n\nWhat you know right now: {context}\n\n"
                  f"Intent: {intent}. {detail}\n\nSay the single line you would speak now:")
        text = self._complete(prompt, "generation")
        return text.split("\n", 1)[0].strip().strip('"')[:240] if text else None
