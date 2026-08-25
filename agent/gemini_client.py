"""Free Gemini backend that duck-types MoondreamClient for local speech testing.

Production speaks through Moondream (paid). For iterating on speech quality
without cost, this points the identical OpenAI-compatible request machinery at
Google's free Gemini endpoint (``gemini-2.5-flash``, 500 requests/day free).
Because it subclasses ``MoondreamClient``, it exposes the exact same async
surface the ``VoiceAgent`` calls (submit_response/poll_response,
submit_generation/poll_generation, classify/select, status, close), so the agent
cannot tell the difference — the deterministic guards and control flow under
test are identical to production.

Not wired into production; the harness swaps it onto ``agent.moondream``.
"""
from __future__ import annotations

from agent.env import gemini_api_key
from agent.moondream_client import MoondreamClient

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient(MoondreamClient):
    """MoondreamClient retargeted at Gemini's OpenAI-compatible endpoint."""

    endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    # Gemini's OpenAI-compat layer expects ``max_tokens``, not the newer name.
    token_param = "max_tokens"

    def __init__(self, model: str | None = None, enabled: bool = True,
                 timeout: float = 15.0):
        super().__init__(model=model or DEFAULT_MODEL, enabled=enabled,
                         timeout=timeout)
        # Re-key onto Gemini after the base wired everything for Moondream.
        self._key = gemini_api_key()
        self.available = bool(self._key)
        self._lifecycle = "configured" if self.available else "unconfigured"

    def _auth_headers(self, key: str) -> dict:
        return {"Authorization": f"Bearer {key}"}

    def _payload_extra(self) -> dict:
        # Gemini 2.5 does server-side "thinking" that consumes the token budget
        # and truncates short spoken lines mid-sentence; disable it so a one-line
        # check-in comes back whole.
        return {"reasoning_effort": "none"}
