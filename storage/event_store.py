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
            self._db.commit()
        return event_id

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
            self._db.commit()
            return int(cursor.rowcount)

    def close(self) -> None:
        """Close the SQLite handle."""
        with self._lock:
            self._db.close()
