"""Privacy-enforcing event store for public, JSON-safe showcase events."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from core.events import PersistencePolicy, Result, Visibility


_SENSITIVE_KEYS = ("frame", "image", "jpeg", "audio", "waveform", "embedding",
                   "biometric", "base64", "possible_conditions", "private_hypotheses")
_CASE_COLUMNS = ("id", "subject_id", "module", "key", "correlation_id",
                 "opened_at", "last_seen_at", "signal_active", "cleared_at",
                 "status", "severity", "confidence", "quality", "location",
                 "summary", "version", "expires_at")


class CaseNotFoundError(LookupError):
    """Requested caregiver case does not exist."""


class CaseConflictError(RuntimeError):
    """Caregiver case changed after the caller read it."""


class CaseValidationError(ValueError):
    """Caregiver case mutation failed validation."""


def _json_safe(value: Any, path: str = "payload") -> Any:
    """Return JSON-compatible data while refusing raw media-like values."""
    if isinstance(value, str):
        lowered = value.lstrip().lower()
        if lowered.startswith(("data:image/", "data:audio/", "data:video/")):
            raise TypeError(f"encoded media refused: {path}")
        if len(value) > 10_000:
            raise TypeError(f"oversized event string refused: {path}")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = str(key)
            if any(term in name.lower() for term in _SENSITIVE_KEYS):
                raise TypeError(f"sensitive event field refused: {path}.{name}")
            out[name] = _json_safe(item, f"{path}.{name}")
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise TypeError(f"oversized event sequence refused: {path}")
        return [_json_safe(v, f"{path}[]") for v in value]
    raise TypeError(f"event value is not JSON-safe: {type(value).__name__}")


class EventStore:
    """SQLite timeline for summaries; raw media and private hypotheses are refused."""
    _instance = None

    def __init__(self, path: str | Path = "data/events.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._pending_writes = 0
        self._last_commit = time.monotonic()
        self._db.execute("""CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL,
            subject_id TEXT NOT NULL, source TEXT NOT NULL, module TEXT,
            key TEXT, severity TEXT, confidence REAL, quality REAL,
            location TEXT, correlation_id TEXT, payload TEXT NOT NULL)""")
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(events)")}
        if "persistence" not in columns:
            self._db.execute("ALTER TABLE events ADD COLUMN persistence TEXT NOT NULL DEFAULT 'event'")
        if "expires_at" not in columns:
            self._db.execute("ALTER TABLE events ADD COLUMN expires_at REAL")
        self._db.execute("UPDATE events SET expires_at=ts+? WHERE expires_at IS NULL",
                         (30 * 86400,))
        self._db.execute("CREATE INDEX IF NOT EXISTS events_ts ON events(ts)")
        self._db.execute("""CREATE TABLE IF NOT EXISTS care_cases (
            id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, module TEXT NOT NULL,
            key TEXT NOT NULL, correlation_id TEXT, opened_at REAL NOT NULL,
            last_seen_at REAL NOT NULL, signal_active INTEGER NOT NULL DEFAULT 1,
            cleared_at REAL, status TEXT NOT NULL, severity TEXT NOT NULL,
            confidence REAL, quality REAL, location TEXT, summary TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1, expires_at REAL)""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS care_case_actions (
            id TEXT PRIMARY KEY, case_id TEXT NOT NULL, ts REAL NOT NULL,
            action TEXT NOT NULL, actor TEXT NOT NULL, note TEXT,
            channel TEXT, success INTEGER)""")
        self._db.execute("CREATE INDEX IF NOT EXISTS care_cases_subject_status "
                         "ON care_cases(subject_id,status,opened_at)")
        self._db.execute("CREATE INDEX IF NOT EXISTS care_actions_case "
                         "ON care_case_actions(case_id,ts)")
        self._db.commit()

    @classmethod
    def instance(cls) -> "EventStore":
        """Return the process-wide event store."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(self, kind: str, payload: dict, *, subject_id: str = "primary",
               source: str = "system", module: str | None = None,
               key: str | None = None, severity: str = "info",
               confidence: float | None = None, quality: float | None = None,
               location: str | None = None, correlation_id: str | None = None,
               timestamp: float | None = None,
               persistence: PersistencePolicy = PersistencePolicy.EVENT,
               retention_seconds: float | None = None) -> str:
        """Store one safe summary event and return its opaque identifier."""
        safe = _json_safe(payload)
        event_id = uuid.uuid4().hex
        ts = time.time() if timestamp is None else timestamp
        default_retention = {PersistencePolicy.EVENT: 30 * 86400,
                             PersistencePolicy.BASELINE: 365 * 86400,
                             PersistencePolicy.NONE: 0}[persistence]
        expires = ts + (default_retention if retention_seconds is None else retention_seconds)
        with self._lock:
            self._db.execute("""INSERT INTO events
                (id,ts,kind,subject_id,source,module,key,severity,confidence,quality,
                 location,correlation_id,payload,persistence,expires_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                event_id, ts, kind,
                subject_id, source, module, key, severity, confidence, quality,
                location, correlation_id, json.dumps(safe, separators=(",", ":")),
                persistence.value, expires))
            self._pending_writes += 1
            if self._pending_writes >= 64 or time.monotonic() - self._last_commit >= 1.0:
                self._flush_locked()
        return event_id

    def _flush_locked(self) -> None:
        if self._pending_writes:
            self._db.commit()
            self._pending_writes = 0
            self._last_commit = time.monotonic()

    def flush(self) -> None:
        """Commit any batched event writes immediately."""
        with self._lock:
            self._flush_locked()

    def record_result(self, result: Result) -> str | None:
        """Persist an opted-in public result; private or ephemeral data is ignored."""
        if result.visibility != Visibility.PUBLIC or result.persistence == PersistencePolicy.NONE:
            return None
        return self.record("observation", {"value": result.value, "message": result.message,
                                           "tags": result.conversation_tags,
                                           "evidence_window": result.evidence_window},
                           subject_id=result.subject_id, source=result.source,
                           module=result.module, key=result.key,
                           severity=result.severity.value, confidence=result.confidence,
                           quality=result.quality, location=result.location,
                           correlation_id=result.correlation_id,
                           timestamp=result.timestamp, persistence=result.persistence)

    def record_assessment(self, action: str, protocol: str, **metadata) -> str:
        """Record a public-safe assessment lifecycle event."""
        return self.record("assessment", {"action": action, "protocol": protocol}, **metadata)

    def record_question(self, topic: str, ordinal: int, **metadata) -> str:
        """Record an approved question topic without generative private context."""
        return self.record("question", {"topic": topic, "ordinal": ordinal}, **metadata)

    def record_answer(self, classification: str, **metadata) -> str:
        """Record only an answer classification, never raw microphone audio."""
        return self.record("answer", {"classification": classification}, **metadata)

    def record_recommendation(self, topic: str, text: str, **metadata) -> str:
        """Record one public-safe recommendation in its causal chain."""
        return self.record("recommendation", {"topic": topic, "text": text[:500]}, **metadata)

    def recent(self, limit: int = 100, *, subject_id: str | None = None) -> list[dict]:
        """Return newest safe summaries in chronological order."""
        sql = ("SELECT id,ts,kind,subject_id,source,module,key,severity,confidence,"
               "quality,location,correlation_id,payload,persistence,expires_at FROM events")
        args: list[Any] = []
        if subject_id is not None:
            sql += " WHERE subject_id=?"
            args.append(subject_id)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(max(1, min(limit, 1000)))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        names = ["id", "timestamp", "kind", "subject_id", "source", "module",
                 "key", "severity", "confidence", "quality", "location",
                 "correlation_id", "payload", "persistence", "expires_at"]
        out = []
        for row in reversed(rows):
            item = dict(zip(names, row))
            item["payload"] = json.loads(item["payload"])
            out.append(item)
        return out

    def query(self, *, start: float, end: float, subject_id: str | None = None,
              limit: int = 5000) -> list[dict]:
        """Return bounded safe summaries for caregiver trends and exports."""
        sql = ("SELECT id,ts,kind,subject_id,source,module,key,severity,confidence,"
               "quality,location,correlation_id,payload,persistence,expires_at "
               "FROM events WHERE ts>=? AND ts<=?")
        args: list[Any] = [float(start), float(end)]
        if subject_id and subject_id != "all":
            sql += " AND subject_id=?"
            args.append(subject_id)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(max(1, min(int(limit), 10_000)))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        names = ["id", "timestamp", "kind", "subject_id", "source", "module",
                 "key", "severity", "confidence", "quality", "location",
                 "correlation_id", "payload", "persistence", "expires_at"]
        out = []
        for row in reversed(rows):
            item = dict(zip(names, row))
            item["payload"] = json.loads(item["payload"])
            out.append(item)
        return out

    @staticmethod
    def _case(row) -> dict | None:
        if row is None:
            return None
        item = dict(zip(_CASE_COLUMNS, row))
        item["signal_active"] = bool(item["signal_active"])
        return item

    def _case_row_locked(self, case_id: str):
        return self._db.execute(
            f"SELECT {','.join(_CASE_COLUMNS)} FROM care_cases WHERE id=?",
            (case_id,)).fetchone()

    def _case_action_locked(self, case_id: str, action: str, timestamp: float,
                            *, note: str | None = None,
                            channel: str | None = None,
                            success: bool | None = None,
                            actor: str = "system") -> None:
        self._db.execute(
            "INSERT INTO care_case_actions (id,case_id,ts,action,actor,note,channel,success) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, case_id, timestamp, action, actor,
             note, channel, None if success is None else int(success)))

    def open_alert_case(self, result: Result, timestamp: float | None = None) -> dict:
        """Open or reactivate the unresolved case for one confirmed alert."""
        if result.visibility != Visibility.PUBLIC:
            raise CaseValidationError("private results cannot create caregiver cases")
        now = time.time() if timestamp is None else float(timestamp)
        subject = str(result.subject_id or "primary")
        summary = _json_safe({"summary": str(result.message)[:500]})["summary"]
        with self._lock:
            row = self._db.execute(
                f"SELECT {','.join(_CASE_COLUMNS)} FROM care_cases "
                "WHERE subject_id=? AND module=? AND key=? AND status!='resolved' "
                "AND (signal_active=1 OR cleared_at IS NULL) "
                "ORDER BY opened_at DESC LIMIT 1",
                (subject, result.module, result.key)).fetchone()
            case = self._case(row)
            if case is not None:
                reused = True
                self._db.execute(
                    "UPDATE care_cases SET last_seen_at=?,signal_active=1,cleared_at=NULL,"
                    "severity=?,confidence=?,quality=?,location=?,summary=?,version=version+1 "
                    "WHERE id=?",
                    (now, result.severity.value, result.confidence, result.quality,
                     result.location, summary, case["id"]))
                case_id = case["id"]
                self._case_action_locked(case_id, "signal_reconfirmed", now)
            else:
                reused = False
                case_id = uuid.uuid4().hex
                self._db.execute(
                    """INSERT INTO care_cases
                    (id,subject_id,module,key,correlation_id,opened_at,last_seen_at,
                     signal_active,status,severity,confidence,quality,location,summary,version)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                    (case_id, subject, result.module, result.key,
                     result.correlation_id, now, now, 1, "open",
                     result.severity.value, result.confidence, result.quality,
                     result.location, summary))
                self._case_action_locked(case_id, "opened", now)
            self._db.commit()
            opened = self._case(self._case_row_locked(case_id))
            opened["reused"] = reused
            return opened

    def record_case_delivery(self, case_id: str, channel: str, success: bool,
                             timestamp: float | None = None,
                             notification: str = "initial") -> None:
        """Append one notification-channel delivery outcome."""
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            if self._case_row_locked(case_id) is None:
                raise CaseNotFoundError(case_id)
            if notification not in ("initial", "reminder", "escalation"):
                raise CaseValidationError("invalid notification tier")
            self._case_action_locked(case_id, f"{notification}_notification_"
                                     f"{'sent' if success else 'failed'}", now,
                                     channel=str(channel)[:80], success=success)
            self._db.commit()

    def mark_case_signal(self, case_id: str, active: bool,
                         timestamp: float | None = None) -> dict:
        """Update whether the originating signal remains active."""
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            case = self._case(self._case_row_locked(case_id))
            if case is None:
                raise CaseNotFoundError(case_id)
            if case["signal_active"] != bool(active):
                self._db.execute(
                    "UPDATE care_cases SET signal_active=?,cleared_at=?,last_seen_at=?,"
                    "version=version+1 WHERE id=?",
                    (int(active), None if active else now, now, case_id))
                self._case_action_locked(case_id, "signal_reconfirmed" if active
                                         else "signal_cleared", now)
                self._db.commit()
            return self._case(self._case_row_locked(case_id))

    def case(self, case_id: str) -> dict | None:
        """Return one case and its append-only audit actions."""
        with self._lock:
            item = self._case(self._case_row_locked(case_id))
            if item is None:
                return None
            rows = self._db.execute(
                "SELECT id,case_id,ts,action,actor,note,channel,success "
                "FROM care_case_actions WHERE case_id=? ORDER BY ts DESC,id DESC LIMIT 5000",
                (case_id,)).fetchall()
        names = ("id", "case_id", "timestamp", "action", "actor", "note",
                 "channel", "success")
        item["actions"] = [dict(zip(names, row)) for row in reversed(rows)]
        for action in item["actions"]:
            if action["success"] is not None:
                action["success"] = bool(action["success"])
        return item

    def list_cases(self, *, subject_id: str = "primary", status: str = "all",
                   start: float | None = None, limit: int = 500) -> list[dict]:
        """List bounded caregiver cases with anonymous-subject filtering."""
        if status not in ("all", "open", "acknowledged", "resolved"):
            raise CaseValidationError("invalid case status")
        sql = f"SELECT {','.join(_CASE_COLUMNS)} FROM care_cases WHERE 1=1"
        args: list[Any] = []
        if subject_id != "all":
            sql += " AND subject_id=?"
            args.append(subject_id)
        if status != "all":
            sql += " AND status=?"
            args.append(status)
        if start is not None:
            sql += " AND (status!='resolved' OR opened_at>=?)"
            args.append(float(start))
        sql += " ORDER BY opened_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
        with self._lock:
            return [self._case(row) for row in self._db.execute(sql, args).fetchall()]

    def mutate_case(self, case_id: str, action: str, expected_version: int,
                    note: str | None = None, timestamp: float | None = None) -> dict:
        """Apply an acknowledged, resolved, or note action with version checking."""
        if action not in ("acknowledge", "resolve", "note"):
            raise CaseValidationError("unsupported case action")
        clean_note = None if note is None else str(note).strip()
        if clean_note is not None and len(clean_note) > 500:
            raise CaseValidationError("note must be at most 500 characters")
        if action == "note" and not clean_note:
            raise CaseValidationError("note is required")
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            case = self._case(self._case_row_locked(case_id))
            if case is None:
                raise CaseNotFoundError(case_id)
            if case["version"] != int(expected_version):
                raise CaseConflictError("case changed; refresh and retry")
            status = case["status"]
            if action == "acknowledge":
                if status != "open":
                    raise CaseValidationError("only open cases can be acknowledged")
                status = "acknowledged"
            elif action == "resolve":
                if status not in ("open", "acknowledged"):
                    raise CaseValidationError("case is already resolved")
                status = "resolved"
            expires = now + 90 * 86400 if action == "resolve" else case["expires_at"]
            self._db.execute(
                "UPDATE care_cases SET status=?,version=version+1,expires_at=? "
                "WHERE id=? AND version=?",
                (status, expires, case_id, int(expected_version)))
            self._case_action_locked(case_id, action + "d" if action != "note"
                                     else "note", now, note=clean_note,
                                     actor="local_caregiver")
            self._db.commit()
            return self.case(case_id)

    def subjects(self) -> list[str]:
        """Return retained anonymous subject identifiers, primary first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT subject_id FROM events UNION SELECT subject_id FROM care_cases").fetchall()
        values = {str(row[0]) for row in rows if row[0]}
        values.add("primary")
        return sorted(values, key=lambda value: (value != "primary", value))

    def interrupt_active_cases(self, timestamp: float | None = None) -> int:
        """Record a monitoring restart boundary for previously active cases."""
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            ids = [row[0] for row in self._db.execute(
                "SELECT id FROM care_cases WHERE signal_active=1").fetchall()]
            for case_id in ids:
                self._db.execute(
                    "UPDATE care_cases SET signal_active=0,cleared_at=NULL,version=version+1 WHERE id=?",
                    (case_id,))
                self._case_action_locked(case_id, "monitoring_restarted", now)
            self._db.commit()
        return len(ids)

    def retention_status(self, now: float | None = None) -> dict:
        """Summarize retention counts without exposing stored payloads."""
        now = time.time() if now is None else float(now)
        with self._lock:
            events, expired_events = self._db.execute(
                "SELECT COUNT(*),SUM(CASE WHEN expires_at<=? THEN 1 ELSE 0 END) FROM events",
                (now,)).fetchone()
            cases, expired_cases, unresolved = self._db.execute(
                """SELECT COUNT(*),SUM(CASE WHEN status='resolved' AND expires_at<=? THEN 1 ELSE 0 END),
                SUM(CASE WHEN status!='resolved' THEN 1 ELSE 0 END) FROM care_cases""",
                (now,)).fetchone()
        return {"event_rows": int(events or 0),
                "expired_event_rows": int(expired_events or 0),
                "case_rows": int(cases or 0),
                "expired_case_rows": int(expired_cases or 0),
                "unresolved_case_rows": int(unresolved or 0),
                "event_retention_days": 30, "resolved_case_retention_days": 90}

    def chain(self, correlation_id: str) -> list[dict]:
        """Return one complete observation-to-recommendation causal chain."""
        with self._lock:
            rows = self._db.execute(
                """SELECT id,ts,kind,subject_id,source,module,key,severity,confidence,
                quality,location,correlation_id,payload,persistence,expires_at
                FROM events WHERE correlation_id=? ORDER BY ts""", (correlation_id,)).fetchall()
        names = ["id", "timestamp", "kind", "subject_id", "source", "module",
                 "key", "severity", "confidence", "quality", "location",
                 "correlation_id", "payload", "persistence", "expires_at"]
        out = []
        for row in rows:
            item = dict(zip(names, row))
            item["payload"] = json.loads(item["payload"])
            out.append(item)
        return out

    def purge_expired(self, now: float | None = None) -> int:
        """Delete expired summaries according to their explicit retention policy."""
        now = time.time() if now is None else now
        with self._lock:
            cursor = self._db.execute("DELETE FROM events WHERE expires_at IS NOT NULL AND expires_at<=?", (now,))
            event_count = int(cursor.rowcount)
            expired_cases = [row[0] for row in self._db.execute(
                "SELECT id FROM care_cases WHERE status='resolved' AND expires_at<=?", (now,))]
            for case_id in expired_cases:
                self._db.execute("DELETE FROM care_case_actions WHERE case_id=?", (case_id,))
                self._db.execute("DELETE FROM care_cases WHERE id=?", (case_id,))
            self._db.commit()
            self._pending_writes = 0
            self._last_commit = time.monotonic()
            return event_count + len(expired_cases)

    def close(self) -> None:
        """Close the SQLite handle."""
        with self._lock:
            self._flush_locked()
            self._db.close()
