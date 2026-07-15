"""Persistent longitudinal store (SQLite).

Longitudinal modules (activity trends, presence, grooming, weight) have a
different lifetime than in-memory frame buffers: they must survive restarts
and span days. They write time-stamped samples here and query recent
history. One tiny table keeps it schema-simple.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"


class HistoryStore:
    """SQLite store of timestamped longitudinal samples."""
    _instance = None

    def __init__(self, path: Path = _DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS samples ("
            "  ts REAL, module TEXT, key TEXT, value REAL)")
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(samples)")}
        if "subject_id" not in columns:
            self.conn.execute("ALTER TABLE samples ADD COLUMN subject_id TEXT NOT NULL DEFAULT 'primary'")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mk ON samples(subject_id, module, key, ts)")
        self.conn.commit()

    @classmethod
    def instance(cls) -> "HistoryStore":
        """Return the process-wide singleton, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, module: str, key: str, value: float, ts: float | None = None,
            subject_id: str = "primary"):
        """Append a timestamped sample to the store."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO samples (ts,module,key,value,subject_id) VALUES (?,?,?,?,?)",
                (ts or time.time(), module, key, float(value), subject_id))
            self.conn.commit()

    def recent(self, module: str, key: str, seconds: float,
               subject_id: str = "primary", now: float | None = None):
        """Return recent (timestamp, value) samples within a time window."""
        cutoff = (time.time() if now is None else now) - seconds
        with self._lock:
            cur = self.conn.execute(
                "SELECT ts, value FROM samples WHERE subject_id=? AND module=? AND key=? AND ts>=? "
                "ORDER BY ts", (subject_id, module, key, cutoff))
            return cur.fetchall()

    def mean_since(self, module: str, key: str, seconds: float,
                   subject_id: str = "primary", now: float | None = None):
        """Return the mean value over the trailing time window."""
        cutoff = (time.time() if now is None else now) - seconds
        with self._lock:
            cur = self.conn.execute(
                "SELECT AVG(value) FROM samples WHERE subject_id=? AND module=? AND key=? AND ts>=?",
                (subject_id, module, key, cutoff))
            row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def last(self, module: str, key: str, subject_id: str = "primary"):
        """Return the newest (timestamp, value) for a subject and signal."""
        with self._lock:
            return self.conn.execute(
                "SELECT ts,value FROM samples WHERE subject_id=? AND module=? AND key=? ORDER BY ts DESC LIMIT 1",
                (subject_id, module, key)).fetchone()

    def count_since(self, module: str, key: str, seconds: float,
                    subject_id: str = "primary", now: float | None = None) -> int:
        """Count samples inside a subject-specific trailing window."""
        cutoff = (time.time() if now is None else now) - seconds
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM samples WHERE subject_id=? AND module=? AND key=? AND ts>=?",
                (subject_id, module, key, cutoff)).fetchone()
        return int(row[0] if row else 0)
