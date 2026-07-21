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
from collections import deque
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"


class HistoryStore:
    """SQLite store of timestamped longitudinal samples."""
    _instance = None

    def __init__(self, path: Path = _DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = Path(path)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._pending_writes = 0
        self._last_commit = time.monotonic()
        self._write_cv = threading.Condition()
        self._write_queue: deque[tuple[float, str, str, float, str]] = deque(maxlen=4096)
        self._writer_stop = False
        self._writer_failures = 0
        self._writer_drops = 0
        self._closed = False
        self._aggregate_cv = threading.Condition()
        self._aggregate_queue: deque[tuple[str, str, str, float]] = deque()
        self._aggregates: dict[tuple[str, str, str, float], dict] = {}
        self._aggregate_failures = 0
        self._aggregate_stop = False
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS samples ("
            "  ts REAL, module TEXT, key TEXT, value REAL)")
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(samples)")}
        if "subject_id" not in columns:
            self.conn.execute("ALTER TABLE samples ADD COLUMN subject_id TEXT NOT NULL DEFAULT 'primary'")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mk ON samples(subject_id, module, key, ts)")
        self.conn.commit()
        self._writer = threading.Thread(
            target=self._writer_loop, name="history-writer", daemon=True)
        self._writer.start()
        self._aggregate_worker = threading.Thread(
            target=self._aggregate_loop, name="history-aggregates", daemon=True)
        self._aggregate_worker.start()

    @classmethod
    def instance(cls) -> "HistoryStore":
        """Return the process-wide singleton, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, module: str, key: str, value: float, ts: float | None = None,
            subject_id: str = "primary"):
        """Append a timestamped sample to the store."""
        item = (time.time() if ts is None else float(ts), str(module), str(key),
                float(value), str(subject_id))
        with self._write_cv:
            if self._closed:
                return
            if len(self._write_queue) == self._write_queue.maxlen:
                self._writer_drops += 1
                # Preserve data rather than block a detector indefinitely.
                self._write_queue.popleft()
            self._write_queue.append(item)
            self._write_cv.notify()
        self._update_aggregates(item)

    def _update_aggregates(self, item) -> None:
        ts, module, key, value, subject = item
        with self._aggregate_cv:
            for cache_key, state in self._aggregates.items():
                cm, ck, cs, _seconds = cache_key
                if (cm, ck, cs) != (module, key, subject):
                    continue
                if state["ready"]:
                    state["values"].append((ts, value))
                    state["sum"] += value
                else:
                    state["pending"].append((ts, value))

    def _aggregate_loop(self) -> None:
        """Bootstrap requested rolling windows away from detector threads."""
        connection = sqlite3.connect(str(self._path), check_same_thread=False)
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            while True:
                with self._aggregate_cv:
                    while not self._aggregate_queue and not self._aggregate_stop:
                        self._aggregate_cv.wait(timeout=1.0)
                    if self._aggregate_stop:
                        return
                    cache_key = self._aggregate_queue.popleft()
                    state = self._aggregates.get(cache_key)
                    registered_at = state["registered_at"] if state else time.time()
                module, key, subject, seconds = cache_key
                cutoff = registered_at - seconds
                try:
                    rows = connection.execute(
                        "SELECT ts,value FROM samples WHERE subject_id=? AND module=? "
                        "AND key=? AND ts>=? AND ts<? ORDER BY ts",
                        (subject, module, key, cutoff, registered_at)).fetchall()
                    with self._aggregate_cv:
                        state = self._aggregates.get(cache_key)
                        if state is None:
                            continue
                        values = deque((float(ts), float(value)) for ts, value in rows)
                        values.extend(state["pending"])
                        state["pending"].clear()
                        state["values"] = values
                        state["sum"] = sum(value for _ts, value in values)
                        state["ready"] = True
                        state["loaded_at"] = time.time()
                        self._aggregate_cv.notify_all()
                except Exception:  # noqa: BLE001
                    self._aggregate_failures += 1
        finally:
            connection.close()

    def rolling_mean(self, module: str, key: str, seconds: float,
                     subject_id: str = "primary", now: float | None = None):
        """Return an O(1), asynchronously bootstrapped rolling mean.

        The first call registers the window and returns ``None`` until the
        aggregate worker has loaded its baseline.  It never touches SQLite on
        the caller's thread.
        """
        cache_key = (str(module), str(key), str(subject_id), float(seconds))
        current = float(time.time() if now is None else now)
        with self._aggregate_cv:
            state = self._aggregates.get(cache_key)
            if state is None:
                state = {"ready": False, "registered_at": current,
                         "loaded_at": None, "values": deque(), "pending": deque(),
                         "sum": 0.0}
                self._aggregates[cache_key] = state
                self._aggregate_queue.append(cache_key)
                self._aggregate_cv.notify()
                return None
            if not state["ready"]:
                return None
            cutoff = current - float(seconds)
            values = state["values"]
            while values and values[0][0] < cutoff:
                state["sum"] -= values.popleft()[1]
            return state["sum"] / len(values) if values else None

    def wait_aggregates(self, timeout: float = 1.0) -> bool:
        """Wait for registered baselines during startup/tests, never frame processing."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._aggregate_cv:
            while any(not state["ready"] for state in self._aggregates.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._aggregate_cv.wait(timeout=remaining)
            return True

    def _drain_queue_locked(self) -> int:
        """Move queued writes into SQLite; caller holds the database lock."""
        with self._write_cv:
            items = list(self._write_queue)
            self._write_queue.clear()
        if items:
            self.conn.executemany(
                "INSERT INTO samples (ts,module,key,value,subject_id) VALUES (?,?,?,?,?)",
                items)
            self._pending_writes += len(items)
        return len(items)

    def _writer_loop(self) -> None:
        while True:
            with self._write_cv:
                if not self._write_queue and not self._writer_stop:
                    self._write_cv.wait(timeout=1.0)
                if self._writer_stop and not self._write_queue:
                    return
            try:
                with self._lock:
                    self._drain_queue_locked()
                    if (self._pending_writes >= 64 or
                            time.monotonic() - self._last_commit >= 1.0):
                        self._flush_locked()
            except Exception:  # noqa: BLE001
                self._writer_failures += 1

    def _flush_locked(self) -> None:
        if self._pending_writes:
            self.conn.commit()
            self._pending_writes = 0
            self._last_commit = time.monotonic()

    def flush(self) -> None:
        """Commit any batched numeric-history writes immediately."""
        with self._lock:
            self._drain_queue_locked()
            self._flush_locked()

    def close(self) -> None:
        """Commit pending samples and close the SQLite connection."""
        if self._closed:
            return
        self._closed = True
        with self._aggregate_cv:
            self._aggregate_stop = True
            self._aggregate_cv.notify_all()
        with self._write_cv:
            self._writer_stop = True
            self._write_cv.notify_all()
        self._writer.join(timeout=1.0)
        self._aggregate_worker.join(timeout=1.0)
        with self._lock:
            self._drain_queue_locked()
            self._flush_locked()
            self.conn.close()

    def diagnostics(self) -> dict:
        with self._write_cv:
            queued = len(self._write_queue)
        with self._aggregate_cv:
            aggregate_loading = sum(1 for state in self._aggregates.values()
                                    if not state["ready"])
            aggregate_windows = len(self._aggregates)
        return {"alive": self._writer.is_alive() and not self._closed,
                "queued": queued, "dropped": self._writer_drops,
                "failures": self._writer_failures,
                "aggregate_failures": self._aggregate_failures,
                "aggregates": {"alive": self._aggregate_worker.is_alive()
                                and not self._closed,
                               "windows": aggregate_windows,
                               "loading": aggregate_loading,
                               "failures": self._aggregate_failures}}

    def recent(self, module: str, key: str, seconds: float,
               subject_id: str = "primary", now: float | None = None):
        """Return recent (timestamp, value) samples within a time window."""
        cutoff = (time.time() if now is None else now) - seconds
        with self._lock:
            self._drain_queue_locked()
            cur = self.conn.execute(
                "SELECT ts, value FROM samples WHERE subject_id=? AND module=? AND key=? AND ts>=? "
                "ORDER BY ts", (subject_id, module, key, cutoff))
            return cur.fetchall()

    def mean_since(self, module: str, key: str, seconds: float,
                   subject_id: str = "primary", now: float | None = None):
        """Return the mean value over the trailing time window."""
        cutoff = (time.time() if now is None else now) - seconds
        with self._lock:
            self._drain_queue_locked()
            cur = self.conn.execute(
                "SELECT AVG(value) FROM samples WHERE subject_id=? AND module=? AND key=? AND ts>=?",
                (subject_id, module, key, cutoff))
            row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def last(self, module: str, key: str, subject_id: str = "primary"):
        """Return the newest (timestamp, value) for a subject and signal."""
        with self._lock:
            self._drain_queue_locked()
            return self.conn.execute(
                "SELECT ts,value FROM samples WHERE subject_id=? AND module=? AND key=? ORDER BY ts DESC LIMIT 1",
                (subject_id, module, key)).fetchone()

    def count_since(self, module: str, key: str, seconds: float,
                    subject_id: str = "primary", now: float | None = None) -> int:
        """Count samples inside a subject-specific trailing window."""
        cutoff = (time.time() if now is None else now) - seconds
        with self._lock:
            self._drain_queue_locked()
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
            self._drain_queue_locked()
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
            self._drain_queue_locked()
            return [str(row[0]) for row in self.conn.execute(
                "SELECT DISTINCT subject_id FROM samples ORDER BY subject_id").fetchall()]

    def purge_before(self, cutoff: float) -> int:
        """Delete numeric history older than the 365-day retention boundary."""
        with self._lock:
            self._drain_queue_locked()
            cursor = self.conn.execute("DELETE FROM samples WHERE ts<?", (float(cutoff),))
            self.conn.commit()
            self._pending_writes = 0
            self._last_commit = time.monotonic()
            return int(cursor.rowcount)

    def retention_status(self, cutoff: float) -> dict:
        """Return numeric-history totals and expired-row counts."""
        with self._lock:
            self._drain_queue_locked()
            total, expired = self.conn.execute(
                "SELECT COUNT(*),SUM(CASE WHEN ts<? THEN 1 ELSE 0 END) FROM samples",
                (float(cutoff),)).fetchone()
        return {"history_rows": int(total or 0),
                "expired_history_rows": int(expired or 0),
                "history_retention_days": 365}
