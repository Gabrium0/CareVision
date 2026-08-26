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

from core.events import Result, Severity, Visibility


@dataclass
class ObservationMemory:
    """The agent's evolving knowledge of the person, accumulated over time."""
    name: str = "there"
    latest: dict = field(default_factory=dict)        # (module,key) -> Result
    first_seen: dict = field(default_factory=dict)    # (module,key,value) -> ts
    arrived_at: float | None = None
    dialogue: list = field(default_factory=list)      # (speaker, text, ts)
    dialogue_keep: int = 20                           # turns retained (user+agent combined)

    def person_said(self, text: str, ts: float | None = None) -> None:
        """Record something the person said (from the ASR listener)."""
        self.dialogue.append(("them", text, time.time() if ts is None else ts))
        del self.dialogue[:-self.dialogue_keep]

    def agent_said(self, text: str, ts: float | None = None) -> None:
        """Record something the agent spoke, so replies stay in context."""
        self.dialogue.append(("you", text, time.time() if ts is None else ts))
        del self.dialogue[:-self.dialogue_keep]

    def ingest(self, snapshot: list[Result], now: float | None = None) -> None:
        """Merge new results into the current person-state."""
        now = time.time() if now is None else now
        for r in snapshot:
            # Dedicated workflows may consume agent-only data directly from
            # the current snapshot. Never copy it into general conversation
            # memory, where it would outlive the result TTL or enter prompts.
            if r.visibility == Visibility.AGENT_ONLY:
                continue
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

    def facial_cues(self) -> dict[str, str]:
        """Return current public NVIDIA appearance cues for corroboration only."""
        result = self.get("skin_vision", "facial_appearance")
        if result is None or result.expired or not isinstance(result.value, dict):
            return {}
        cues = result.value.get("cues")
        if not isinstance(cues, dict):
            return {}
        return {str(key): str(value) for key, value in cues.items()}

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
        hr = (self.get("heart_rate", "bpm")
              or self.get("heart_rate", "bpm_classical")
              or self.get("heart_rate", "bpm_open_rppg"))
        if hr is not None and hr.confidence >= 0.4:
            bits.append(f"Heart rate about {float(hr.value):.0f} bpm.")
        resp = self.get("respiration", "breaths_per_min")
        if resp is not None and resp.confidence >= 0.4:
            bits.append(f"Breathing about {float(resp.value):.0f} per minute.")
        age = self.get("age_estimation", "age_range")
        if age is not None and age.confidence >= 0.3:
            bits.append(f"Estimated age range {age.value}.")
        # Alertness: PERCLOS is % of time eyes closed; give the LLM a word, not a number.
        perclos = self.get("drowsiness", "perclos")
        if perclos is not None and perclos.confidence >= 0.5:
            p = float(perclos.value)
            alertness = "alert" if p < 0.15 else ("tired" if p < 0.30 else "very drowsy")
            bits.append(f"Alertness: {alertness}.")
        attention = self.get_val("attention", "eye_contact")
        if attention is not None:
            bits.append("They are looking at you." if attention
                        else "They are not looking at you right now.")
        gesture = self.get("gesture", "gesture")
        if gesture is not None and self.is_new("gesture", "gesture", gesture.value):
            bits.append(f"Gesture just now: {gesture.message or gesture.value}.")
        if self.get_val("gesture", "waving") and self.is_new("gesture", "waving", True):
            bits.append("They are waving at you.")
        height = self.get("height_distance", "height_m")
        if height is not None and height.confidence >= 0.4:
            bits.append(f"Height about {float(height.value):.2f} m.")
        dist = self.get("height_distance", "distance_m")
        if dist is not None and dist.confidence >= 0.4:
            bits.append(f"Standing about {float(dist.value):.1f} m away.")
        notable = [r.message for r in self.latest.values()
                   if r.severity in (Severity.NOTICE, Severity.WARNING) and r.message]
        if notable:
            bits.append("Recent observations: " + "; ".join(notable[:4]) + ".")
        if self.dialogue:
            turns = "; ".join(f"{who}: \"{text}\""
                              for who, text, _ts in self.dialogue[-4:])
            bits.append("Recent conversation (you = you): " + turns + ".")
        return " ".join(bits)
