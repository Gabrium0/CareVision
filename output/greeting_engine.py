"""Rule-based greeting + recommendation engine.

Consumes the aggregator snapshot and produces a human-facing message when a
person arrives (or on demand). It is deliberately rule-based and
transparent so a caregiver can audit exactly why something was said. Health
indicators are phrased as gentle suggestions, never diagnoses, and only
above a confidence floor. ALERT-severity findings always surface.

Swap this class for an LLM-backed generator later — it only needs the
snapshot list, so the rest of the system is unaffected.
"""
from __future__ import annotations

import time

from core.events import Result, Severity

_CONF_FLOOR = 0.35


class GreetingEngine:
    def __init__(self, name: str = "there", cooldown_s: float = 30.0):
        self.name = name
        self.cooldown_s = cooldown_s
        self._last_greeting = 0.0

    def _time_of_day(self) -> str:
        h = time.localtime().tm_hour
        if h < 12:
            return "Good morning"
        if h < 18:
            return "Good afternoon"
        return "Good evening"

    def alerts(self, snapshot: list[Result]) -> list[str]:
        return [r.message for r in snapshot
                if r.severity == Severity.ALERT and r.message]

    def maybe_greet(self, aggregator, force: bool = False) -> str | None:
        """Return a greeting string if a person just arrived (or forced)."""
        arrival = aggregator.get("presence", "arrival")
        now = time.time()
        if not force:
            if arrival is None:
                return None
            if now - self._last_greeting < self.cooldown_s:
                return None
        self._last_greeting = now
        return self.compose(aggregator.snapshot())

    def compose(self, snapshot: list[Result]) -> str:
        by = {(r.module, r.key): r for r in snapshot}
        lines: list[str] = [f"{self._time_of_day()}, {self.name}!"]

        emo = by.get(("emotion", "emotion"))
        if emo and emo.confidence >= _CONF_FLOOR:
            mood = str(emo.value)
            greet = {
                "happy": "You look cheerful today — lovely to see.",
                "sad": "You seem a little down. I'm here if you'd like some company.",
                "angry": "You look a bit tense. Shall we take a slow breath together?",
                "surprise": "Something caught your attention?",
                "neutral": "Hope you're having a calm day.",
            }.get(mood, "Hope you're having a good day.")
            lines.append(greet)

        # Health suggestions (gentle, confidence-gated)
        suggestions: list[str] = []
        recs = {
            ("skin_color", "pallor"): "You look a little pale — maybe rest and some water would help.",
            ("skin_color", "flushing"): "Your face looks flushed. Are you feeling warm?",
            ("skin_color", "cyanosis"): "Your lips look a little blue — let's make sure you're breathing easy; consider calling someone.",
            ("skin_color", "jaundice_tint"): "I noticed a slight yellow tint — worth mentioning to your doctor.",
            ("rash", "rash_fraction"): "There may be a rash on your skin — keep an eye on it.",
            ("dry_lips", "lip_dryness"): "Your lips look dry — a glass of water might be nice.",
            ("eye_redness", "sclera_redness"): "Your eyes look a bit red — resting them could help.",
            ("drowsiness", "perclos"): "You seem sleepy — a short rest might do you good.",
            ("yawn", "yawn"): "Lots of yawning — perhaps some rest is due.",
            ("tremor", "tremor_right"): "I noticed some hand tremor — no rush, take your time.",
            ("tremor", "tremor_left"): "I noticed some hand tremor — no rush, take your time.",
            ("balance", "postural_sway"): "You seem a little unsteady — please hold something sturdy.",
            ("gait", "gait_asymmetry"): "Your walking looks uneven today — take care on your feet.",
            ("pain", "pain"): "You look uncomfortable — are you in any pain?",
            ("respiration", "breaths_per_min"): None,   # handled only if atypical
            ("clothing_advice", "recommendation"): None,  # use module message directly
        }
        for (mod, key), text in recs.items():
            r = by.get((mod, key))
            if r and text and r.confidence >= _CONF_FLOOR and r.severity != Severity.INFO:
                suggestions.append(text)

        hr = by.get(("heart_rate", "bpm"))
        if hr and hr.confidence >= 0.4 and hr.severity == Severity.WARNING:
            suggestions.append(f"Your heart rate looks around {hr.value:.0f} — "
                               "if you feel unwell, let's check with someone.")

        clothing = by.get(("clothing_advice", "recommendation"))
        if clothing and clothing.confidence >= _CONF_FLOOR and clothing.severity != Severity.INFO:
            suggestions.append(str(clothing.value))

        # de-dupe while preserving order, cap length
        seen, deduped = set(), []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        lines.extend(deduped[:3])

        # Alerts always appended, prominently
        for a in self.alerts(snapshot):
            lines.append(f"⚠ {a}")

        if len(lines) == 1:
            lines.append("Everything looks fine. Let me know if you need anything.")
        return "\n".join(lines)
