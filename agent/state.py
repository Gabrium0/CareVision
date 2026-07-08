"""ObservationMemory: the agent's evolving knowledge of the person.

Ingests the aggregator snapshot each tick, remembers the latest value per
(module, key) and when each value was first seen (for novelty), and builds a
compact natural-language context the LLM uses to personalize speech. This is
what lets the agent accumulate information over time rather than react to a
single frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.events import Result, Severity


@dataclass
class ObservationMemory:
    """The agent's evolving knowledge of the person, accumulated over time."""
    name: str = "there"
    latest: dict = field(default_factory=dict)        # (module,key) -> Result
    first_seen: dict = field(default_factory=dict)    # (module,key,value) -> ts
    arrived_at: float | None = None

    def ingest(self, snapshot: list[Result], now: float | None = None) -> None:
        """Merge new results into the current person-state."""
        now = time.time() if now is None else now
        for r in snapshot:
            self.latest[(r.module, r.key)] = r
            sig = (r.module, r.key, str(r.value))
            self.first_seen.setdefault(sig, now)
        if self.get("presence", "arrival") is not None and self.arrived_at is None:
            self.arrived_at = now
        if self.get_val("presence", "present") is None and \
                self.get("presence", "arrival") is None:
            # person gone: allow a fresh greeting next arrival
            self.arrived_at = None

    def get(self, module: str, key: str) -> Result | None:
        """Return the latest result for (module, key), or a default."""
        return self.latest.get((module, key))

    def get_val(self, module: str, key: str, default=None):
        """Return the latest value for (module, key), or a default."""
        r = self.latest.get((module, key))
        return r.value if r is not None else default

    def is_new(self, module: str, key: str, value, within: float = 6.0,
               now: float | None = None) -> bool:
        """True if this (module,key,value) first appeared within `within` sec."""
        now = time.time() if now is None else now
        ts = self.first_seen.get((module, key, str(value)))
        return ts is not None and now - ts <= within

    @staticmethod
    def time_of_day() -> str:
        """Return 'morning' / 'afternoon' / 'evening' for now."""
        h = time.localtime().tm_hour
        return "morning" if h < 12 else ("afternoon" if h < 18 else "evening")

    def _first_present_backend(self, base: str):
        """Return the first available value among <base>_<backend> keys."""
        for (mod, key), r in self.latest.items():
            if key.startswith(base + "_") and r.value not in ("...", "unknown", None):
                return r
        return None

    def mood(self) -> str | None:
        """Return a coarse mood label from the latest emotion/valence."""
        val = self._first_present_backend("valence")
        if val is not None:
            v = float(val.value)
            return "positive" if v > 0.15 else ("low" if v < -0.15 else "neutral")
        emo = self._first_present_backend("emotion")
        return str(emo.value) if emo is not None else None

    def context_text(self) -> str:
        """Compact description of what the agent currently knows, for the LLM."""
        bits = [f"Person: {self.name}.", f"Time of day: {self.time_of_day()}."]
        mood = self.mood()
        if mood:
            bits.append(f"Apparent mood: {mood}.")
        clothing = self.get_val("clothing", "upper_body")
        if clothing and clothing not in ("...", "unknown"):
            bits.append(f"Wearing: {clothing}.")
        feels = self.get_val("weather", "feels_like_c",
                             self.get_val("weather", "temperature_c"))
        if feels is not None:
            bits.append(f"Weather feels like {float(feels):.0f}C.")
        hr = self.get("heart_rate", "bpm_open_rppg") or self.get("heart_rate", "bpm_classical")
        if hr is not None and hr.confidence >= 0.4:
            bits.append(f"Heart rate about {float(hr.value):.0f} bpm.")
        notable = [r.message for r in self.latest.values()
                   if r.severity in (Severity.NOTICE, Severity.WARNING) and r.message]
        if notable:
            bits.append("Recent observations: " + "; ".join(notable[:4]) + ".")
        return " ".join(bits)
