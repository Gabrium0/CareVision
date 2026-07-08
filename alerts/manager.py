"""AlertManager: turn ALERT-severity detections into caregiver notifications
with confirmation, dedupe, rate-limiting, quiet hours, and escalation.

Deterministic safety layer. It consumes the aggregator snapshot each tick and
decides whether/what to notify — it never calls the LLM.

Per alert key (module, key) it tracks:
- first_seen:     when the alert first appeared (confirmation window)
- first_notified: when we first sent it
- last_notified:  for cooldown
- escalated:      whether the escalation tier was already notified
An alert must persist for `confirm_seconds` before the first notification
(filters transient false positives). If it is still active `escalate_after`
seconds after the first notification, it escalates once.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.events import Result, Severity
from alerts.notifier import build_channels


@dataclass
class _AlertState:
    first_seen: float
    first_notified: float | None = None
    last_notified: float | None = None
    escalated: bool = False


@dataclass
class AlertManager:
    """Turns ALERT results into caregiver notifications with confirm/dedupe/escalate."""
    channels: list = field(default_factory=lambda: build_channels(["console"]))
    confirm_seconds: float = 3.0          # must persist this long before firing
    cooldown_seconds: float = 120.0       # min gap between repeats of one alert
    escalate_after: float = 300.0         # escalate if still active this long
    quiet_hours: tuple | None = None      # (start_hour, end_hour) or None
    quiet_suppress: tuple = ()            # alert keys held during quiet hours
    _state: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict):
        """Build an instance from its config dict."""
        cfg = cfg or {}
        qh = cfg.get("quiet_hours")
        return cls(
            channels=build_channels(cfg.get("channels", ["console"])),
            confirm_seconds=float(cfg.get("confirm_seconds", 3.0)),
            cooldown_seconds=float(cfg.get("cooldown_seconds", 120.0)),
            escalate_after=float(cfg.get("escalate_after", 300.0)),
            quiet_hours=tuple(qh) if qh else None,
            quiet_suppress=tuple(cfg.get("quiet_suppress", [])),
        )

    def _in_quiet_hours(self, now: float) -> bool:
        if not self.quiet_hours:
            return False
        h = time.localtime(now).tm_hour
        a, b = self.quiet_hours
        return a <= h < b if a <= b else (h >= a or h < b)

    def _dispatch(self, subject: str, body: str) -> None:
        for ch in self.channels:
            ch.send(subject, body)

    def evaluate(self, snapshot: list[Result], now: float | None = None) -> None:
        """Evaluate the latest snapshot and act on it."""
        now = time.time() if now is None else now
        active = {(r.module, r.key): r for r in snapshot
                  if r.severity == Severity.ALERT}

        for key, r in active.items():
            st = self._state.get(key)
            if st is None:
                st = self._state[key] = _AlertState(first_seen=now)
            if now - st.first_seen < self.confirm_seconds:
                continue                                  # still confirming
            if key in self.quiet_suppress and self._in_quiet_hours(now):
                continue                                  # held during quiet hrs

            subject = str(r.message).split(" — ")[0][:80] or f"{r.module} alert"
            when = time.strftime("%H:%M:%S", time.localtime(now))
            if st.first_notified is None:
                self._dispatch(subject, f"{r.message}\nDetected {when} "
                                        f"(confidence {r.confidence:.2f}).")
                st.first_notified = st.last_notified = now
            elif (not st.escalated and now - st.first_notified >= self.escalate_after):
                self._dispatch(f"ESCALATION: {subject}",
                               f"{r.message}\nStill ongoing at {when} — no "
                               "resolution since first alert. Please respond.")
                st.escalated = True
                st.last_notified = now
            elif now - (st.last_notified or 0) >= self.cooldown_seconds:
                self._dispatch(f"REMINDER: {subject}",
                               f"{r.message}\nStill ongoing at {when}.")
                st.last_notified = now

        # clear state for alerts that resolved (no longer active)
        for key in list(self._state):
            if key not in active:
                self._state.pop(key, None)
