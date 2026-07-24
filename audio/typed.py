"""Keyboard-driven listener implementing the live listener interface.

`audio.stt.Listener` is the right instrument for a real deployment and the wrong
one for a showcase room: whisper turns ambient noise into confident sentences
("Thank you very much.") and the agent answers them, which reads to an audience
as the system inventing a conversation.

`TypedListener` satisfies the same polling contract, so replies typed into the
companion page travel the *same* corroboration path as speech. Nothing
downstream distinguishes the two, which is the point -- the demo exercises the
real confirm/deny logic instead of a presentation-only branch.
"""
from __future__ import annotations

import queue
import time


class TypedListener:
    """Typed-answer source with the same polling contract as Listener."""
    available = True

    def __init__(self):
        self._utterances: queue.Queue[tuple[str, float]] = queue.Queue()

    def push(self, text: str, timestamp: float | None = None) -> bool:
        """Queue one typed reply; False when the text is unusable."""
        cleaned = " ".join(str(text or "").split())[:400]
        if not cleaned:
            return False
        self._utterances.put((cleaned, time.time() if timestamp is None
                              else float(timestamp)))
        return True

    def pop_utterances(self) -> list[tuple[str, float]]:
        """Drain typed utterances exactly like the live ASR listener."""
        out = []
        while True:
            try:
                out.append(self._utterances.get_nowait())
            except queue.Empty:
                return out

    def pop_metrics(self) -> list[dict]:
        """No speech timing exists without a microphone."""
        return []

    def close(self) -> None:
        """Typed listeners hold no external resources."""

    def mark_agent_spoke(self, timestamp: float | None = None) -> None:
        """Accept the live listener turn-timing hook for interface parity."""
