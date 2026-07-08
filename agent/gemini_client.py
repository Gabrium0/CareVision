"""Gemini natural-language generation for the voice agent.

Turns an utterance intent + the accumulated person context into one short, warm
spoken line. Key is read from .env (GEMINI_API_KEY / GOOGLE_API_KEY). If the
SDK or key is missing, `available` is False and the caller uses the templated
fallback, so the agent still speaks offline.
"""
from __future__ import annotations

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
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.available = False
        self._client = None
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

    def generate(self, intent: str, context: str, detail: str = "") -> str | None:
        """Generate one short spoken line, or None if unavailable."""
        if not self.available:
            return None
        prompt = (f"{_PERSONA}\n\nWhat you know right now: {context}\n\n"
                  f"Intent: {intent}. {detail}\n\n"
                  "Say the single line you would speak now:")
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=prompt)
            text = (getattr(resp, "text", "") or "").strip().strip('"')
            return text.split("\n")[0][:240] if text else None
        except Exception as e:  # noqa: BLE001
            print(f"[agent/gemini] generation failed: {e}")
            return None
