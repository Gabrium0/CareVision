"""Deterministic cross-signal fusion and meaningful daily summaries."""
from __future__ import annotations

from core.events import PersistencePolicy, Result, Severity


class MultimodalReasoner:
    """Combine independent public signals; no LLM or VLM result can alert alone."""
    def evaluate(self, snapshot: list[Result]) -> list[Result]:
        """Emit conservative public-safe correlations from independent evidence."""
        out = []
        for subject in {r.subject_id for r in snapshot}:
            rows = [r for r in snapshot if r.subject_id == subject]
            keys = {(r.module, r.key): r for r in rows}
            coughs = [r for r in rows if r.module == "sound_event" and "cough" in r.key]
            low_activity = (keys.get(("activity_level", "activity_drop"))
                            or keys.get(("unresponsive", "stillness")))
            symptom = keys.get(("conversation", "symptoms_confirmed"))
            if coughs and low_activity and symptom:
                out.append(self._result("cough_activity",
                    "Repeated cough sounds, reduced activity, and a user-confirmed symptom coincided.", subject))
            near_fall = keys.get(("near_fall", "recovered"))
            balance = keys.get(("guided_assessments", "balance"))
            support = bool(balance and isinstance(balance.value, dict)
                           and balance.value.get("support_used"))
            dizzy = keys.get(("conversation", "dizziness_confirmed"))
            if near_fall and support and dizzy:
                out.append(self._result("stability_check",
                    "A recovered stumble, support use, and reported dizziness coincided.", subject))
            drink = any(r.module == "scene_vision" and "drinking" in str(r.value).lower()
                        for r in rows)
            interval = keys.get(("routine", "hours_since_drink"))
            if drink and interval and isinstance(interval.value, (int, float)) and interval.value >= 3:
                out.append(self._result("hydration_opportunity",
                    "A drink is visible after a long interval without a recorded drink opportunity.", subject))
            difficulty = keys.get(("routine", "repeated_difficulty_standing"))
            slower = keys.get(("routine", "slower_sit_to_stand"))
            if difficulty and slower:
                out.append(self._result("standing_change",
                    "Repeated stand attempts and a slower personal sit-to-stand baseline coincided.", subject))
        return out

    @staticmethod
    def _result(key: str, message: str, subject_id: str) -> Result:
        return Result("multimodal_reasoning", key, message, .75, Severity.NOTICE,
                      message, ttl=60, subject_id=subject_id, source="deterministic_fusion",
                      persistence=PersistencePolicy.EVENT)
