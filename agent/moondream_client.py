"""Moondream Cloud natural-language generation with an offline fallback."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

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
                 timeout: float = 20.0):
        self.model = model
        self.timeout = float(timeout)
        self._key = moondream_api_key()
        self.available = bool(self._key)
        self._enabled = bool(enabled)
        self._lock = threading.RLock()
        self._generation_requests = 0
        self._classification_requests = 0
        self._failures = 0
        self._last_request_at: float | None = None
        self._last_error: str | None = None
        if self.available:
            print(f"[agent/moondream] ready (model {self.model})")
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
            self._enabled = bool(enabled)
            return self._enabled

    def toggle_enabled(self) -> bool:
        """Toggle future API requests; an in-flight request may finish."""
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled

    def status(self) -> dict:
        """Return credential-free operational diagnostics."""
        with self._lock:
            return {
                "provider": "moondream",
                "available": self.available,
                "enabled": self._enabled,
                "active": self.available and self._enabled,
                "model": self.model,
                "generation_requests": self._generation_requests,
                "classification_requests": self._classification_requests,
                "failures": self._failures,
                "last_request_at": self._last_request_at,
                "last_error": self._last_error,
            }

    def _begin_request(self, kind: str) -> str | None:
        with self._lock:
            if not self._enabled or not self.available or not self._key:
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
            self._last_error = f"{type(exc).__name__}: request failed"[:240]

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
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            return text.strip() if isinstance(text, str) and text.strip() else None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                OSError, KeyError, IndexError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            self._record_failure(exc)
            print(f"[agent/moondream] {kind} request failed ({type(exc).__name__})")
            return None

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
