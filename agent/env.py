"""Environment / secrets loading for the agent (keys come from .env)."""
from __future__ import annotations

import os

_loaded = False


def load_env() -> None:
    """Load .env into os.environ once (no-op if python-dotenv is absent)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        # minimal fallback parser so a .env still works without python-dotenv
        path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def moondream_api_key() -> str | None:
    """Return the Moondream Cloud key without exposing it to diagnostics."""
    load_env()
    return (os.environ.get("X-Moondream-Auth")
            or os.environ.get("MOONDREAM_API_KEY"))


def moondream_model() -> str:
    """Return the Moondream model id (env override, else the current API id)."""
    load_env()
    return os.environ.get("MOONDREAM_MODEL") or "moondream/moondream3-preview"


def nvidia_api_key() -> str | None:
    """Return the NVIDIA hosted-inference API key, or None."""
    load_env()
    return os.environ.get("NVIDIA_API_KEY")


def gemini_api_key() -> str | None:
    """Return the Google Gemini API key (free tier), or None."""
    load_env()
    return os.environ.get("GEMINI_API_KEY")


def groq_api_key() -> str | None:
    """Return the Groq API key for cloud speech-to-text, or None."""
    load_env()
    return os.environ.get("GROQ_API_KEY")
