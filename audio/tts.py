"""Offline text-to-speech — the agent's mouth.

Uses pyttsx3 (offline, cross-platform). Speech is queued to a worker thread so
`say()` never blocks the video loop. If pyttsx3 is unavailable, utterances are
printed instead, so the agent still "speaks" in text.
"""
from __future__ import annotations

import queue
import threading


class Speaker:
    """Offline text-to-speech worker on its own thread (the agent's voice)."""
    def __init__(self, rate: int = 165, volume: float = 1.0, enabled: bool = True):
        self.enabled = enabled
        self.speaking = False   # True while TTS is audible (read by audio/stt.Listener)
        self._engine = None
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        if enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
                self._engine.setProperty("volume", volume)
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
                print("[tts] pyttsx3 speaker ready")
            except Exception as e:  # noqa: BLE001
                print(f"[tts] pyttsx3 unavailable ({e}); printing utterances")
                self._engine = None

    def _run(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                break
            # Flag flips inside the worker around the blocking call, so it is
            # exact by construction — the Listener mutes while this is True.
            self.speaking = True
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:  # noqa: BLE001
                print(f"[tts] speak failed: {e}")
            finally:
                self.speaking = False

    def say(self, text: str) -> None:
        """Queue a line to be spoken (non-blocking)."""
        if not text:
            return
        print(f"[agent speaks] {text}")
        if self._engine is not None:
            self._q.put(text)

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        if self._thread is not None:
            self._q.put(None)
