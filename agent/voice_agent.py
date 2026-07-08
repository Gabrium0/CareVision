"""VoiceAgent: orchestrates memory -> policy -> Gemini -> speech.

Call `tick(snapshot)` once per frame (cheap; rate-limited internally). The agent
updates its memory, lets the policy pick at most one thing to say, phrases it
with Gemini (falling back to a templated line offline), speaks it via TTS, and
records it so it won't repeat. The last utterance is exposed for the dashboard.
"""
from __future__ import annotations

import time

from agent.state import ObservationMemory
from agent.policy import Policy
from agent.gemini_client import GeminiClient
from audio.tts import Speaker


class VoiceAgent:
    """Orchestrates memory -> policy -> Gemini -> speech."""
    def __init__(self, name: str = "there", speak: bool = True,
                 model: str = "gemini-2.5-flash", **policy_kwargs):
        self.memory = ObservationMemory(name=name)
        self.policy = Policy(**policy_kwargs)
        self.gemini = GeminiClient(model=model)
        self.speaker = Speaker(enabled=speak)
        self.last_utterance = ""
        mode = "Gemini" if self.gemini.available else "templated"
        print(f"[agent] voice agent ready (name={name}, speech={mode})")

    def tick(self, snapshot, now: float | None = None) -> str | None:
        """Advance one step: update state and act if warranted."""
        now = time.time() if now is None else now
        self.memory.ingest(snapshot, now)
        intent = self.policy.next_intent(self.memory, now)
        if intent is None:
            return None
        text = self.gemini.generate(intent.llm_intent, self.memory.context_text(),
                                    intent.detail) or intent.fallback
        self.policy.mark_spoken(intent, now)
        self.last_utterance = text
        self.speaker.say(text)
        return text

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self.speaker.close()
