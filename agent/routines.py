"""Subject-aware routine baselines and deterministic temporal events."""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from core.events import PersistencePolicy, Result, Severity
from storage.history_store import HistoryStore


class RoutineReasoner:
    """Learn routine opportunities and report meaningful deviations without diagnosis."""
    def __init__(self, interval: float = 30.0, summary_hour: int = 20):
        self.interval = interval
        self.summary_hour = summary_hour
        self.store = HistoryStore.instance()
        self._last_run: dict[str, float] = defaultdict(lambda: -1e9)
        self._seen: set[tuple[str, str, str, float]] = set()
        self._last_summary_day: dict[str, int] = {}

    def _new(self, result: Result) -> bool:
        signature = (result.subject_id, result.module, result.key, result.timestamp)
        if signature in self._seen:
            return False
        self._seen.add(signature)
        if len(self._seen) > 4096:
            cutoff = time.time() - 2 * 86400
            self._seen = {s for s in self._seen if s[3] >= cutoff}
        return True

    @staticmethod
    def _contains(result: Result, terms: tuple[str, ...]) -> bool:
        value = " ".join(map(str, result.value)) if isinstance(result.value, list) else str(result.value)
        low = value.lower()
        return any(term in low for term in terms)

    def _record(self, result: Result, now: float) -> None:
        subject = result.subject_id
        if result.module == "presence" and result.key in ("present", "arrival"):
            self.store.add("routine", "occupied", 1, now, subject)
            location_key = "occupied_" + "".join(
                ch if ch.isalnum() else "_" for ch in (result.location or "unspecified").lower())
            self.store.add("routine", location_key, 1, now, subject)
            self.store.add("routine", "presence_hour", time.localtime(now).tm_hour, now, subject)
            if result.key == "arrival":
                self.store.add("routine", "return", 1, now, subject)
        if result.module == "scene_vision":
            if self._contains(result, ("eating", "meal", "breakfast", "lunch", "dinner")):
                self.store.add("routine", "meal_opportunity", 1, now, subject)
            if self._contains(result, ("drinking", "cup", "glass", "bottle")):
                self.store.add("routine", "drink_opportunity", 1, now, subject)
            if self._contains(result, ("medication", "pill", "medicine")):
                self.store.add("routine", "medication_opportunity", 1, now, subject)
            if self._contains(result, ("walker", "cane", "wheelchair", "mobility aid")):
                self.store.add("routine", "mobility_aid_visible", 1, now, subject)
            if self._contains(result, ("leaving", "preparing to leave", "open exterior door")):
                self.store.add("routine", "leaving", 1, now, subject)
        if result.module == "activity_level":
            value = float(result.value) if isinstance(result.value, (int, float)) else 0.0
            self.store.add("routine", "activity", value, now, subject)
        if result.module == "speech_timing":
            self.store.add("routine", "conversation", 1, now, subject)
        if result.module == "guided_assessments" and result.key == "sit_to_stand" \
                and isinstance(result.value, dict):
            self.store.add("routine", "sit_to_stand_seconds",
                           float(result.value.get("total_time_seconds", 0)), now, subject)
            self.store.add("routine", "stand_failed_attempts",
                           float(result.value.get("failed_attempts", 0)), now, subject)

    def _baseline_result(self, subject: str, now: float, location: str | None) -> Result | None:
        hours = [value for _ts, value in self.store.recent(
            "routine", "presence_hour", 30 * 86400, subject, now)]
        if len(hours) < 5:
            return None
        wake = int(np.percentile(hours, 10))
        sleep = int(np.percentile(hours, 90))
        counts = {key: self.store.count_since("routine", key, 7 * 86400, subject, now)
                  for key in ("meal_opportunity", "drink_opportunity", "conversation",
                              "mobility_aid_visible", "leaving", "return")}
        activity = [value for _ts, value in self.store.recent(
            "routine", "activity", 7 * 86400, subject, now)]
        location_key = "occupied_" + "".join(
            ch if ch.isalnum() else "_" for ch in (location or "unspecified").lower())
        value = {"wake_window_start": wake, "sleep_window_start": sleep,
                 "weekly_opportunities": counts,
                 "room_occupancy_observations_7d": self.store.count_since(
                     "routine", location_key, 7 * 86400, subject, now),
                 "mean_activity_7d": (round(float(np.mean(activity)), 3)
                                      if activity else None),
                 "inactive_observations_7d": sum(value < .1 for value in activity),
                 "location": location or "unspecified"}
        return Result("routine", "baseline", value, .65, Severity.INFO,
                      "Personal routine baseline updated", ttl=120,
                      subject_id=subject, source="personal_baseline", location=location,
                      persistence=PersistencePolicy.BASELINE)

    def _temporal_events(self, snapshot: list[Result], subject: str,
                         now: float) -> list[Result]:
        rows = [r for r in snapshot if r.subject_id == subject]
        keys = {(r.module, r.key): r for r in rows}
        out: list[Result] = []
        location = next((r.location for r in rows if r.location), None)
        baseline = self._baseline_result(subject, now, location)
        if baseline:
            out.append(baseline)
            wake, sleep = baseline.value["wake_window_start"], baseline.value["sleep_window_start"]
            hour = time.localtime(now).tm_hour
            if any(r.module == "presence" for r in rows) and (hour < max(0, wake-2) or hour > min(23, sleep+2)):
                out.append(self._result(subject, "unusual_night_activity",
                    "Activity occurred outside the person's usual observed window.", .65, location))
        for key, label, hours in (("meal_opportunity", "meal", 10),
                                  ("drink_opportunity", "drink", 6)):
            last = self.store.last("routine", key, subject)
            weekly = self.store.count_since("routine", key, 7*86400, subject, now)
            if weekly >= 3 and last and now - last[0] > hours * 3600:
                out.append(self._result(subject, f"missed_{label}_opportunity",
                    f"No {label} opportunity has been recorded during the usual interval.", .6, location))
        med_now = any(r.module == "scene_vision" and self._contains(r, ("medication", "pill", "medicine")) for r in rows)
        if med_now:
            out.append(Result("routine", "medication_opportunity", True, .65, Severity.INFO,
                "A possible medication opportunity is visible; adherence is unknown.", ttl=120,
                subject_id=subject, source="deterministic_routine", location=location,
                persistence=PersistencePolicy.EVENT))
        last_drink = self.store.last("routine", "drink_opportunity", subject)
        if last_drink:
            hours = round(max(0.0, (now-last_drink[0])/3600), 1)
            out.append(Result("routine", "hours_since_drink", hours, .7, Severity.INFO,
                "Hours since the last recorded drink opportunity", ttl=120,
                subject_id=subject, source="deterministic_routine", location=location))
        assessment = keys.get(("guided_assessments", "sit_to_stand"))
        if assessment and isinstance(assessment.value, dict):
            failed = int(assessment.value.get("failed_attempts", 0))
            current = float(assessment.value.get("total_time_seconds", 0))
            baseline_time = self.store.mean_since("routine", "sit_to_stand_seconds",
                                                  30*86400, subject, now)
            if failed >= 2:
                out.append(self._result(subject, "repeated_difficulty_standing",
                    "Several stand attempts were needed during the guided assessment.", .75, location))
            if baseline_time and current > baseline_time * 1.25:
                out.append(self._result(subject, "slower_sit_to_stand",
                    "The guided sit-to-stand took longer than this person's recent baseline.", .7, location))
        if keys.get(("fall", "fall")) and keys.get(("unresponsive", "immobility")):
            out.append(Result("routine", "post_fall_immobility", True, .9, Severity.ALERT,
                "A deterministic fall signal is followed by prolonged immobility.", ttl=30,
                subject_id=subject, source="deterministic_fusion", location=location,
                persistence=PersistencePolicy.EVENT))
        pacing = keys.get(("wandering", "pacing"))
        if pacing:
            out.append(self._result(subject, "repetitive_pacing",
                "Repeated back-and-forth movement is occurring in this camera view.",
                pacing.confidence, location))
        return out

    @staticmethod
    def _result(subject: str, key: str, message: str, confidence: float,
                location: str | None) -> Result:
        return Result("routine", key, True, confidence, Severity.NOTICE, message,
                      ttl=120, subject_id=subject, source="deterministic_routine",
                      location=location, persistence=PersistencePolicy.EVENT)

    def evaluate(self, snapshot: list[Result], now: float) -> list[Result]:
        """Update baselines and emit subject-specific temporal changes."""
        subjects = {r.subject_id for r in snapshot} or {"primary"}
        for result in snapshot:
            if self._new(result):
                self._record(result, result.timestamp)
        out = []
        for subject in subjects:
            if now - self._last_run[subject] < self.interval:
                continue
            self._last_run[subject] = now
            changes = self._temporal_events(snapshot, subject, now)
            out.extend(changes)
            day = int(now // 86400)
            meaningful = [r.message for r in changes if r.severity in (Severity.NOTICE, Severity.WARNING, Severity.ALERT)]
            if time.localtime(now).tm_hour >= self.summary_hour and meaningful \
                    and self._last_summary_day.get(subject) != day:
                self._last_summary_day[subject] = day
                out.append(Result("routine", "daily_summary", meaningful[:5], .7,
                    Severity.INFO, "Daily meaningful changes: " + "; ".join(meaningful[:3]),
                    ttl=3600, subject_id=subject, source="deterministic_summary",
                    persistence=PersistencePolicy.EVENT))
        return out
