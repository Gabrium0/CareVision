"""Post-aggregation advice rules.

Advisors consume the latest aggregated detector state and emit normal Result
objects, so advice can flow through the same dashboard, voice, and alert paths
as raw detections.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.events import Result, Severity
from agent.multimodal_reasoning import MultimodalReasoner
from agent.routines import RoutineReasoner


_ORDER = {Severity.INFO: 0, Severity.NOTICE: 1, Severity.WARNING: 2, Severity.ALERT: 3}
_DISTRESS_EMOTIONS = {"angry", "fear", "fearful", "sad", "surprise"}


def _severity_max(a: Severity, b: Severity) -> Severity:
    return a if _ORDER[a] >= _ORDER[b] else b


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, "..."):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_numeric(snapshot: list[Result], module: str, key_prefix: str,
                  min_confidence: float) -> Result | None:
    candidates = [
        r for r in snapshot
        if r.module == module
        and r.key.startswith(key_prefix)
        and r.confidence >= min_confidence
        and _numeric(r.value) is not None
    ]
    return max(candidates, key=lambda r: r.confidence, default=None)


def _preferred_bpm(snapshot: list[Result], min_confidence: float) -> Result | None:
    canonical = next(
        (r for r in snapshot
         if r.module == "heart_rate"
         and r.key == "bpm"
         and r.confidence >= min_confidence
         and _numeric(r.value) is not None),
        None,
    )
    return canonical or _best_numeric(snapshot, "heart_rate", "bpm_", min_confidence)


def _preferred_breaths(snapshot: list[Result], min_confidence: float) -> Result | None:
    """Best available breathing rate, from either module that measures one.

    The rPPG backends publish `heart_rate.breaths_per_min_<backend>` and
    modules/respiration.py publishes a standalone `respiration.breaths_per_min`
    from torso motion. Only the first was ever read here, so a rig running
    respiration without an rPPG backend produced no breathing advice at all.
    The rPPG-derived value keeps priority because it is measured on the same
    waveform as the rest of this advisor's vitals.
    """
    return (_best_numeric(snapshot, "heart_rate", "breaths_per_min_", min_confidence)
            or _best_numeric(snapshot, "respiration", "breaths_per_min", min_confidence))


@dataclass
class VitalsAdvisor:
    """Conservative cross-signal vitals advice.

    The thresholds intentionally produce gentle check-in prompts rather than
    diagnoses; deterministic caregiver escalation remains in alerts/.
    """

    enabled: bool = True
    interval: float = 8.0
    bpm_high: float = 100.0
    bpm_warning: float = 120.0
    bpm_low: float = 55.0
    bpm_low_warning: float = 45.0
    hrv_low: float = 20.0
    resp_high: float = 24.0
    # Camera SpO2 is a trend, not an oximeter. These are the "worth a gentle
    # word" and "worth suggesting a check" bands, and they are only ever
    # consulted for a reading modules/spo2.py itself raised above INFO — which
    # that module does only once it is calibrated. Deterministic caregiver
    # escalation stays in alerts/.
    spo2_low: float = 92.0
    spo2_warning: float = 88.0
    min_confidence: float = 0.35
    ttl: float = 30.0
    _last_run: float = field(default=-1e9, init=False)

    def evaluate(self, snapshot: list[Result], now: float) -> list[Result]:
        """Evaluate the latest snapshot and act on it."""
        if not self.enabled or now - self._last_run < self.interval:
            return []
        self._last_run = now

        observations: list[str] = []
        severity = Severity.INFO
        confidence = 0.0

        bpm_r = _preferred_bpm(snapshot, self.min_confidence)
        bpm = _numeric(bpm_r.value) if bpm_r else None
        if bpm is not None:
            confidence = max(confidence, bpm_r.confidence)
            if bpm >= self.bpm_warning:
                observations.append(f"heart rate looks high at about {bpm:.0f} bpm")
                severity = _severity_max(severity, Severity.WARNING)
            elif bpm >= self.bpm_high:
                observations.append(f"heart rate is a bit elevated at about {bpm:.0f} bpm")
                severity = _severity_max(severity, Severity.NOTICE)
            elif bpm <= self.bpm_low_warning:
                observations.append(f"heart rate looks low at about {bpm:.0f} bpm")
                severity = _severity_max(severity, Severity.WARNING)
            elif bpm <= self.bpm_low:
                observations.append(f"heart rate is on the low side at about {bpm:.0f} bpm")
                severity = _severity_max(severity, Severity.NOTICE)

        hrv_r = _best_numeric(snapshot, "heart_rate", "hrv_rmssd_ms_", self.min_confidence * 0.8)
        hrv = _numeric(hrv_r.value) if hrv_r else None
        if hrv is not None and hrv < self.hrv_low:
            observations.append("heart-rate variability looks low")
            confidence = max(confidence, hrv_r.confidence)
            severity = _severity_max(severity, Severity.NOTICE)

        resp_r = _preferred_breaths(snapshot, self.min_confidence * 0.8)
        resp = _numeric(resp_r.value) if resp_r else None
        if resp is not None and resp >= self.resp_high:
            observations.append(f"breathing rate looks elevated at about {resp:.0f} per minute")
            confidence = max(confidence, resp_r.confidence)
            severity = _severity_max(severity, Severity.NOTICE)

        # Only a reading modules/spo2.py already raised above INFO: that module
        # keeps an uncalibrated estimate at INFO deliberately, and a trend
        # without a baseline is not something to speak to a person about.
        spo2_r = next(
            (r for r in snapshot
             if r.module == "spo2" and r.key == "spo2"
             and r.confidence >= self.min_confidence
             and _ORDER[r.severity] >= _ORDER[Severity.NOTICE]
             and _numeric(r.value) is not None),
            None,
        )
        spo2 = _numeric(spo2_r.value) if spo2_r else None
        if spo2 is not None and spo2 <= self.spo2_low:
            wording = ("oxygen reading looks low" if spo2 <= self.spo2_warning
                       else "oxygen reading is a little low")
            observations.append(f"{wording} at about {spo2:.0f} percent")
            confidence = max(confidence, spo2_r.confidence)
            severity = _severity_max(
                severity, Severity.WARNING if spo2 <= self.spo2_warning
                else Severity.NOTICE)

        pain = next((r for r in snapshot if r.module == "pain" and r.key == "pain"), None)
        if pain is not None and _ORDER[pain.severity] >= _ORDER[Severity.WARNING]:
            observations.append("there may be discomfort or pain")
            confidence = max(confidence, pain.confidence)
            severity = _severity_max(severity, Severity.WARNING)

        drowsy = next((r for r in snapshot if r.module == "drowsiness" and r.key == "perclos"), None)
        if drowsy is not None and _ORDER[drowsy.severity] >= _ORDER[Severity.NOTICE]:
            observations.append("they may be getting drowsy")
            confidence = max(confidence, drowsy.confidence)
            severity = _severity_max(severity, drowsy.severity)

        emotion = next(
            (r for r in snapshot
             if r.module == "emotion"
             and r.key.startswith("emotion_")
             and str(r.value).lower() in _DISTRESS_EMOTIONS
             and r.confidence >= self.min_confidence),
            None,
        )
        if emotion is not None:
            observations.append(f"facial expression may look {str(emotion.value).lower()}")
            confidence = max(confidence, emotion.confidence)
            if severity == Severity.NOTICE:
                severity = Severity.WARNING

        if not observations or severity == Severity.INFO:
            return []

        # Mention at most two signals to keep spoken advice short and non-alarming.
        if severity == Severity.WARNING:
            advice = "I noticed " + ", and ".join(observations[:2]) + ". Consider checking in or resting for a moment."
        else:
            advice = "I noticed " + observations[0] + ". A brief rest or check-in may help."
        return [Result(
            module="vitals_advice",
            key="recommendation",
            value=advice,
            confidence=round(max(0.5, min(confidence, 0.85)), 2),
            severity=severity,
            message=advice,
            ttl=self.ttl,
        )]


@dataclass
class ColdSymptomAdvisor:
    """Cold-symptom composite: behavior + color signals, none diagnostic alone.

    Sneezes (modules/sneeze.py), nose/face-touch frequency
    (modules/face_touch.py), facial flushing (modules/skin_color.py), and
    drowsiness combine into a gentle "maybe a cold coming on" check-in.
    Requiring at least two independent signals keeps a single false sneeze
    or a warm room from triggering it.
    """

    enabled: bool = True
    interval: float = 60.0            # a cold develops over minutes, not frames
    sneeze_count: int = 2             # sneezes in the 10-min window that count alone
    face_touch_count: int = 4         # touches in the window that count as a signal
    min_confidence: float = 0.3
    ttl: float = 120.0
    _last_run: float = field(default=-1e9, init=False)

    def evaluate(self, snapshot: list[Result], now: float) -> list[Result]:
        """Evaluate the latest snapshot and act on it."""
        if not self.enabled or now - self._last_run < self.interval:
            return []
        self._last_run = now

        signals: list[str] = []
        confidence = 0.0

        sneezes = next((r for r in snapshot if r.module == "sneeze"
                        and r.key == "sneeze_count_10min"), None)
        n_sneeze = int(_numeric(sneezes.value) or 0) if sneezes else 0
        if n_sneeze >= 1:
            signals.append(f"{n_sneeze} sneeze{'s' if n_sneeze != 1 else ''} recently"
                           if n_sneeze < self.sneeze_count
                           else f"{n_sneeze} sneezes in the last few minutes")
            confidence = max(confidence, sneezes.confidence)

        touches = next((r for r in snapshot if r.module == "face_touch"
                        and r.key == "face_touch_count_10min"), None)
        n_touch = int(_numeric(touches.value) or 0) if touches else 0
        if n_touch >= self.face_touch_count:
            signals.append("frequent nose/face touching")
            confidence = max(confidence, touches.confidence)

        flush = next((r for r in snapshot if r.module == "skin_color"
                      and r.key == "flushing"
                      and r.confidence >= self.min_confidence), None)
        if flush is not None:
            signals.append("some facial flushing")
            confidence = max(confidence, flush.confidence)

        drowsy = next((r for r in snapshot if r.module == "drowsiness"
                       and r.key == "perclos"
                       and _ORDER[r.severity] >= _ORDER[Severity.NOTICE]), None)
        if drowsy is not None:
            signals.append("they seem tired")
            confidence = max(confidence, drowsy.confidence)

        # Two independent signals, or a run of sneezes on its own.
        if len(signals) < 2 and n_sneeze < self.sneeze_count:
            return []

        advice = ("I noticed " + ", and ".join(signals[:3]) +
                  " — could be a cold coming on. Maybe take it easy and "
                  "drink something warm.")
        return [Result(
            module="wellness_advice",
            key="cold_symptoms",
            value=advice,
            confidence=round(max(0.4, min(confidence, 0.75)), 2),
            severity=Severity.NOTICE,
            message=advice,
            ttl=self.ttl,
        )]


class AdvisorEngine:
    """Runs post-aggregation advisors over the snapshot and emits advice results."""
    def __init__(self, advisors: list[Any] | None = None, enabled: bool = True):
        self.enabled = enabled
        self.advisors = advisors or []

    @classmethod
    def from_config(cls, cfg: dict | None):
        """Build an instance from its config dict."""
        if cfg is None:
            return cls(enabled=False)
        if not cfg.get("enabled", True):
            return cls(enabled=False)

        advisors = []
        vitals_cfg = cfg.get("vitals", {}) or {}
        if vitals_cfg.get("enabled", True):
            params = {k: v for k, v in vitals_cfg.items() if k != "enabled"}
            advisors.append(VitalsAdvisor(**params))
        cold_cfg = cfg.get("cold", {}) or {}
        if cold_cfg.get("enabled", True):
            params = {k: v for k, v in cold_cfg.items() if k != "enabled"}
            advisors.append(ColdSymptomAdvisor(**params))
        advisors.append(MultimodalReasoner())
        routine_cfg = cfg.get("routine", {}) or {}
        if routine_cfg.get("enabled", True):
            advisors.append(RoutineReasoner(**{k: v for k, v in routine_cfg.items()
                                               if k != "enabled"}))
        return cls(advisors=advisors)

    def evaluate(self, snapshot: list[Result], now: float | None = None) -> list[Result]:
        """Evaluate the latest snapshot and act on it."""
        if not self.enabled:
            return []
        now = time.time() if now is None else now
        # Advice should not recursively react to advice it emitted in earlier frames.
        detector_snapshot = [r for r in snapshot if not r.module.endswith("_advice")]
        out: list[Result] = []
        for advisor in self.advisors:
            try:
                out.extend(advisor.evaluate(detector_snapshot, now))
            except TypeError:
                out.extend(advisor.evaluate(detector_snapshot))
        return out
