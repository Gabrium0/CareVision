"""NVIDIA hosted-inference backend that duck-types MoondreamClient.

NVIDIA's build.nvidia.com serves many open models through one
OpenAI-compatible endpoint (``integrate.api.nvidia.com/v1/chat/completions``)
with far higher free limits than Gemini's 500 requests/day. Because it
subclasses ``MoondreamClient`` it exposes the identical async surface the
``VoiceAgent`` calls, so the agent's deterministic guards and control flow are
unchanged — only the mouth is stronger.

Pick an *instruct* model for spoken lines (e.g. ``meta/llama-3.3-70b-instruct``).
Avoid *reasoning* models like ``deepseek-ai/deepseek-r1``: they emit long
``<think>`` output that is slow and truncates a one-line reply.
"""
from __future__ import annotations

from agent.env import nvidia_api_key
from agent.moondream_client import MoondreamClient

# NVIDIA retired the whole Llama 3.x hosted line on 2026-08-26 (HTTP 410 Gone).
# nemotron-3-nano is a live MoE (3B active) that returns a warm one-liner in
# ~0.7s over the free tier — ideal latency for the spoken companion.
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


class NvidiaClient(MoondreamClient):
    """MoondreamClient retargeted at NVIDIA's OpenAI-compatible endpoint."""

    endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
    # NVIDIA's OpenAI-compat layer expects ``max_tokens``, not the newer name.
    token_param = "max_tokens"

    def __init__(self, model: str | None = None, enabled: bool = True,
                 timeout: float = 20.0):
        super().__init__(model=model or DEFAULT_MODEL, enabled=enabled,
                         timeout=timeout)
        # Re-key onto NVIDIA after the base wired everything for Moondream.
        self._key = nvidia_api_key()
        self.available = bool(self._key)
        self._lifecycle = "configured" if self.available else "unconfigured"

    def _auth_headers(self, key: str) -> dict:
        return {"Authorization": f"Bearer {key}"}

    def _payload_extra(self) -> dict:
        # nemotron reasons by default and emits its chain-of-thought as the
        # reply, which swamps a one-line spoken turn. Disable server-side
        # thinking so a short check-in comes back as just the line to speak.
        return {"chat_template_kwargs": {"thinking": False}}
