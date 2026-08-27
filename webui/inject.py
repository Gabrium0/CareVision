"""Synthetic signal injection for live showcases (opt-in via --demo-inject).

The companion agent never reads a shared "signals" dict -- it consumes the
per-frame ``list[Result]`` snapshot the real detectors feed through the
Aggregator. So a demo "inject" is nothing more than adding a ``Result`` to that
same stream: it then travels the identical path (smoothing, thresholds,
corroboration, policy cadence, dashboards, alerting) and the agent reacts for
real, rather than the UI faking a reaction.

A single one-frame injection would be median-diluted by ``Aggregator._smooth``
and would expire on its TTL, so a trigger instead *holds* a scenario active for
a few seconds: the frame loop re-emits fresh Results every frame while the hold
lasts. The median of several identical held frames is the held value, so the
number lands cleanly and stays non-expired -- much like a real reading that
persists for a moment.

This module does no file I/O; it only builds in-memory Results and tracks an
in-memory per-scenario expiry. It is thread-safe because the web server calls
``trigger`` on its daemon thread while the frame loop calls ``drain`` on the
main thread.
"""
from __future__ import annotations

import threading
from typing import Callable

from core.events import Result, Severity


def _high_heart_rate(now: float) -> list[Result]:
    return [Result(
        "heart_rate", "bpm", 128, confidence=0.7, severity=Severity.WARNING,
        message="Elevated heart rate", ttl=6.0, timestamp=now,
        source="demo_inject")]


def _frequent_yawns(now: float) -> list[Result]:
    # The visible count for the screen, plus the drowsiness cue that actually
    # drives the spoken "you seem tired" line (perclos at NOTICE severity).
    return [
        Result("yawn", "yawn_count_3min", 7, confidence=0.8,
               severity=Severity.NOTICE, message="Frequent yawning", ttl=6.0,
               timestamp=now, source="demo_inject"),
        Result("drowsiness", "perclos", 0.34, confidence=0.7,
               severity=Severity.NOTICE, message="Signs of fatigue", ttl=6.0,
               timestamp=now, source="demo_inject"),
    ]


def _low_mood(now: float) -> list[Result]:
    # Negative valence drives ObservationMemory.mood()=="low" (the policy mood
    # line); the low-confidence expressivity cue drives the corroboration ask.
    return [
        Result("emotion", "valence_hsemotion", -0.4, confidence=0.7,
               severity=Severity.INFO, message="Low mood", ttl=6.0,
               timestamp=now, source="demo_inject"),
        Result("expressivity", "expressivity_low", 0.5, confidence=0.5,
               severity=Severity.NOTICE, message="Quieter than usual", ttl=6.0,
               timestamp=now, source="demo_inject"),
    ]


def _possible_rash(now: float) -> list[Result]:
    # Low confidence on purpose: the corroboration engine only *asks* about a
    # cue when it is uncertain (confidence <= the rule's max_confidence of 0.6).
    return [Result(
        "rash", "rash", True, confidence=0.4, severity=Severity.NOTICE,
        message="Possible skin change", ttl=6.0, timestamp=now,
        source="demo_inject")]


# scenario_id -> (button label, hold seconds, factory(now) -> fresh Results)
SCENARIOS: dict[str, tuple[str, float, Callable[[float], list[Result]]]] = {
    "high_heart_rate": ("High heart rate", 5.0, _high_heart_rate),
    "frequent_yawns": ("Frequent yawns", 5.0, _frequent_yawns),
    "low_mood": ("Low mood", 5.0, _low_mood),
    "possible_rash": ("Possible rash", 5.0, _possible_rash),
}


class DemoInjector:
    """Holds active demo scenarios and re-emits their signals each frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, float] = {}   # scenario_id -> expiry timestamp

    def trigger(self, scenario_id: str, now: float) -> dict:
        """Mark a scenario active for its hold window.

        Raises ValueError on an unknown id (rendered as HTTP 400 by the server).
        """
        entry = SCENARIOS.get(scenario_id)
        if entry is None:
            raise ValueError("unknown scenario")
        label, hold, _ = entry
        with self._lock:
            self._active[scenario_id] = now + hold
        return {"scenario": scenario_id, "label": label, "hold": hold}

    def drain(self, now: float) -> list[Result]:
        """Return fresh Results for every still-active scenario; drop expired."""
        with self._lock:
            live = [sid for sid, expiry in self._active.items() if expiry > now]
            self._active = {sid: self._active[sid] for sid in live}
        results: list[Result] = []
        for sid in live:
            results.extend(SCENARIOS[sid][2](now))
        return results

    @staticmethod
    def catalog() -> list[dict]:
        """The scenario list the /demo page renders its buttons from."""
        return [{"id": sid, "label": label}
                for sid, (label, _, _) in SCENARIOS.items()]
