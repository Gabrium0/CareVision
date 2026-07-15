"""Gemini natural-language generation for the voice agent.

Turns an utterance intent + the accumulated person context into one short, warm
spoken line. Key is read from .env (GEMINI_API_KEY / GOOGLE_API_KEY). If the
SDK or key is missing, `available` is False and the caller uses the templated
fallback, so the agent still speaks offline.
"""
from __future__ import annotations

import threading
import time

from agent.env import gemini_api_key

_PERSONA = (
    "You are a warm, calm companion robot for an elderly person who may live "
    "alone. You speak out loud in ONE short, natural sentence (max ~25 words), "
    "conversational and kind, never clinical or alarming. You do not give "
    "medical diagnoses. If you mention something you noticed (like their "
    "clothing or mood), be gentle and offer, don't instruct."
)


class GeminiClient:
    """Gemini NLG wrapper for the voice agent, with an offline templated fallback."""
    def __init__(self, model: str = "gemini-2.5-flash", enabled: bool = True):
        self.model = model
        self.available = False
        self._enabled = bool(enabled)
        self._client = None
        self._lock = threading.RLock()
        self._generation_requests = 0
        self._classification_requests = 0
        self._failures = 0
        self._last_request_at: float | None = None
        self._last_error: str | None = None
        key = gemini_api_key()
        if not key:
            print("[agent/gemini] no GEMINI_API_KEY in .env; using templated speech")
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
            self.available = True
            print(f"[agent/gemini] ready (model {self.model})")
        except Exception as e:  # noqa: BLE001
            print(f"[agent/gemini] unavailable ({type(e).__name__}: {e}); "
                  "using templated speech")

    @property
    def enabled(self) -> bool:
        """Whether the user currently permits Gemini API requests."""
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> bool:
        """Enable or disable future API requests; return the new state."""
        with self._lock:
            self._enabled = bool(enabled)
            return self._enabled

    def toggle_enabled(self) -> bool:
        """Toggle future API requests; an already-started request may finish."""
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled

    def status(self) -> dict:
        """Return a thread-safe, credential-free diagnostic snapshot."""
        with self._lock:
            available = bool(self.available and self._client is not None)
            return {
                "available": available,
                "enabled": self._enabled,
                "active": available and self._enabled,
                "model": self.model,
                "generation_requests": self._generation_requests,
                "classification_requests": self._classification_requests,
                "failures": self._failures,
                "last_request_at": self._last_request_at,
                "last_error": self._last_error,
            }

    def _begin_request(self, kind: str):
        """Reserve one permitted request and return the configured SDK client."""
        with self._lock:
            if not self._enabled or not self.available or self._client is None:
                return None
            if kind == "generation":
                self._generation_requests += 1
            else:
                self._classification_requests += 1
            self._last_request_at = time.time()
            return self._client

    def _record_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failures += 1
            self._last_error = f"{type(exc).__name__}: {exc}"[:240]

    def classify_answer(self, question: str, answer: str) -> str | None:
        """Classify a spoken reply to a yes/no health check-in.

        Returns 'confirmed', 'denied', or 'unclear' — or None when the API
        is unavailable/errors, so the caller falls back to keyword matching.
        """
        client = self._begin_request("classification")
        if client is None:
            return None
        prompt = (
            "A companion robot asked an elderly person a gentle yes/no "
            f"check-in question: \"{question}\"\n"
            f"The person replied: \"{answer}\"\n\n"
            "Does the reply CONFIRM the concern, DENY it, or is it UNCLEAR? "
            "Answer with exactly one word: confirmed, denied, or unclear.")
        try:
            resp = client.models.generate_content(
                model=self.model, contents=prompt)
            word = (getattr(resp, "text", "") or "").strip().lower()
            for verdict in ("confirmed", "denied", "unclear"):
                if verdict in word:
                    return verdict
            return None
        except Exception as e:  # noqa: BLE001
            self._record_failure(e)
            print(f"[agent/gemini] classification failed: {e}")
            return None

    def generate(self, intent: str, context: str, detail: str = "") -> str | None:
        """Generate one short spoken line, or None if unavailable."""
        client = self._begin_request("generation")
        if client is None:
            return None
        prompt = (f"{_PERSONA}\n\nWhat you know right now: {context}\n\n"
                  f"Intent: {intent}. {detail}\n\n"
                  "Say the single line you would speak now:")
        try:
            resp = client.models.generate_content(
                model=self.model, contents=prompt)
            text = (getattr(resp, "text", "") or "").strip().strip('"')
            return text.split("\n")[0][:240] if text else None
        except Exception as e:  # noqa: BLE001
            self._record_failure(e)
            print(f"[agent/gemini] generation failed: {e}")
            return None
