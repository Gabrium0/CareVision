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

from core.events import Result, Severity, Visibility
from alerts.notifier import build_channels


@dataclass
class _AlertState:
    first_seen: float
    first_notified: float | None = None
    last_notified: float | None = None
    escalated: bool = False
    case_id: str | None = None


@dataclass
class AlertManager:
    """Turns ALERT results into caregiver notifications with confirm/dedupe/escalate."""
    channels: list = field(default_factory=lambda: build_channels(["console"]))
    confirm_seconds: float = 3.0          # must persist this long before firing
    cooldown_seconds: float = 120.0       # min gap between repeats of one alert
    escalate_after: float = 300.0         # escalate if still active this long
    quiet_hours: tuple | None = None      # (start_hour, end_hour) or None
    quiet_suppress: tuple = ()            # alert keys held during quiet hours
    case_store: object | None = None       # deterministic durable-case boundary
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

    def _dispatch(self, subject: str, body: str) -> list[tuple[str, bool]]:
        outcomes = []
        for ch in self.channels:
            outcomes.append((str(getattr(ch, "name", "channel")), bool(ch.send(subject, body))))
        return outcomes

    def _notify(self, state: _AlertState, subject: str, body: str, now: float,
                tier: str) -> None:
        for channel, success in self._dispatch(subject, body):
            if self.case_store is not None and state.case_id is not None:
                self.case_store.record_case_delivery(
                    state.case_id, channel, success, timestamp=now,
                    notification=tier)

    def evaluate(self, snapshot: list[Result], now: float | None = None) -> None:
        """Evaluate the latest snapshot and act on it."""
        now = time.time() if now is None else now
        active = {(r.subject_id, r.module, r.key): r for r in snapshot
                  if r.severity == Severity.ALERT and r.visibility == Visibility.PUBLIC}

        for key, r in active.items():
            st = self._state.get(key)
            if st is None:
                st = self._state[key] = _AlertState(first_seen=now)
            if now - st.first_seen < self.confirm_seconds:
                continue                                  # still confirming
            if st.case_id is None and self.case_store is not None:
                opened = self.case_store.open_alert_case(r, timestamp=now)
                st.case_id = opened["id"]
                if opened.get("reused"):
                    prior = self.case_store.case(st.case_id)
                    sent = [action for action in prior.get("actions", [])
                            if action["action"].endswith("_notification_sent")]
                    if sent:
                        st.first_notified = min(item["timestamp"] for item in sent)
                        st.last_notified = max(item["timestamp"] for item in sent)
                        st.escalated = any(item["action"].startswith("escalation_")
                                           for item in sent)
            case_status = None
            if self.case_store is not None and st.case_id is not None:
                case = self.case_store.case(st.case_id)
                case_status = case["status"] if case is not None else None
            if case_status == "resolved":
                continue
            quiet_key = (r.module, r.key)
            if (key in self.quiet_suppress or quiet_key in self.quiet_suppress
                    or r.key in self.quiet_suppress) and self._in_quiet_hours(now):
                continue                                  # held during quiet hrs

            # Every subject's alert still opens/tracks a case above (so a
            # secondary visitor's fall stays reviewable in the caregiver
            # portal), but only the primary subject's alert ever reaches a
            # notification channel -- this process makes no identity claim
            # about a visitor and must not page anyone over them.
            if r.subject_id != "primary":
                continue
            subject = str(r.message).split(" — ")[0][:80] or f"{r.module} alert"
            when = time.strftime("%H:%M:%S", time.localtime(now))
            if st.first_notified is None:
                self._notify(st, subject, f"{r.message}\nDetected {when} "
                                          f"(confidence {r.confidence:.2f}).", now,
                             "initial")
                st.first_notified = st.last_notified = now
            elif (not st.escalated and now - st.first_notified >= self.escalate_after):
                self._notify(st, f"ESCALATION: {subject}",
                             f"{r.message}\nStill ongoing at {when} — no "
                             "resolution since first alert. Please respond.", now,
                             "escalation")
                st.escalated = True
                st.last_notified = now
            elif case_status != "acknowledged" and \
                    now - (st.last_notified or 0) >= self.cooldown_seconds:
                self._notify(st, f"REMINDER: {subject}",
                             f"{r.message}\nStill ongoing at {when}.", now,
                             "reminder")
                st.last_notified = now

        # clear state for alerts that resolved (no longer active)
        for key in list(self._state):
            if key not in active:
                state = self._state.pop(key, None)
                if self.case_store is not None and state and state.case_id is not None:
                    self.case_store.mark_case_signal(state.case_id, False, timestamp=now)
