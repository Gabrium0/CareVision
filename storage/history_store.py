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
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._pending_writes = 0
        self._last_commit = time.monotonic()
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
                (time.time() if ts is None else ts, module, key, float(value), subject_id))
            self._pending_writes += 1
            if self._pending_writes >= 64 or time.monotonic() - self._last_commit >= 1.0:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self._pending_writes:
            self.conn.commit()
            self._pending_writes = 0
            self._last_commit = time.monotonic()

    def flush(self) -> None:
        """Commit any batched numeric-history writes immediately."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Commit pending samples and close the SQLite connection."""
        with self._lock:
            self._flush_locked()
            self.conn.close()

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

    def portal_series(self, subject_id: str, start: float, end: float,
                      max_points: int = 300) -> list[dict]:
        """Return bounded numeric trend buckets for the caregiver portal."""
        sql = ("SELECT ts,module,key,value,subject_id FROM samples "
               "WHERE ts>=? AND ts<=?")
        args: list = [float(start), float(end)]
        if subject_id != "all":
            sql += " AND subject_id=?"
            args.append(subject_id)
        sql += " ORDER BY ts DESC LIMIT 100000"
        with self._lock:
            rows = list(reversed(self.conn.execute(sql, args).fetchall()))
        grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
        for ts, module, key, value, subject in rows:
            grouped.setdefault((subject, module, key), []).append((float(ts), float(value)))
        span = max(1.0, float(end) - float(start))
        buckets = max(1, min(int(max_points), 300))
        width = span / buckets
        out = []
        for (subject, module, key), points in sorted(grouped.items()):
            compact: dict[int, list[float]] = {}
            for ts, value in points:
                index = min(buckets - 1, max(0, int((ts - start) / width)))
                compact.setdefault(index, []).append(value)
            values = []
            for index, samples in sorted(compact.items()):
                values.append({"timestamp": start + (index + .5) * width,
                               "mean": sum(samples) / len(samples),
                               "min": min(samples), "max": max(samples),
                               "count": len(samples)})
            out.append({"subject_id": subject, "module": module, "key": key,
                        "points": values})
        return out

    def subjects(self) -> list[str]:
        """Return anonymous subject identifiers represented in numeric history."""
        with self._lock:
            return [str(row[0]) for row in self.conn.execute(
                "SELECT DISTINCT subject_id FROM samples ORDER BY subject_id").fetchall()]

    def purge_before(self, cutoff: float) -> int:
        """Delete numeric history older than the 365-day retention boundary."""
        with self._lock:
            cursor = self.conn.execute("DELETE FROM samples WHERE ts<?", (float(cutoff),))
            self.conn.commit()
            self._pending_writes = 0
            self._last_commit = time.monotonic()
            return int(cursor.rowcount)

    def retention_status(self, cutoff: float) -> dict:
        """Return numeric-history totals and expired-row counts."""
        with self._lock:
            total, expired = self.conn.execute(
                "SELECT COUNT(*),SUM(CASE WHEN ts<? THEN 1 ELSE 0 END) FROM samples",
                (float(cutoff),)).fetchone()
        return {"history_rows": int(total or 0),
                "expired_history_rows": int(expired or 0),
                "history_retention_days": 365}
