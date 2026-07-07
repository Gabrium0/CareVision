"""Persistent longitudinal store (SQLite).

Longitudinal modules (activity trends, presence, grooming, weight) have a
different lifetime than in-memory frame buffers: they must survive restarts
and span days. They write time-stamped samples here and query recent
history. One tiny table keeps it schema-simple.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"


class HistoryStore:
    _instance = None

    def __init__(self, path: Path = _DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS samples ("
            "  ts REAL, module TEXT, key TEXT, value REAL)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mk ON samples(module, key, ts)")
        self.conn.commit()

    @classmethod
    def instance(cls) -> "HistoryStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, module: str, key: str, value: float, ts: float | None = None):
        self.conn.execute("INSERT INTO samples VALUES (?,?,?,?)",
                          (ts or time.time(), module, key, float(value)))
        self.conn.commit()

    def recent(self, module: str, key: str, seconds: float):
        cutoff = time.time() - seconds
        cur = self.conn.execute(
            "SELECT ts, value FROM samples WHERE module=? AND key=? AND ts>=? "
            "ORDER BY ts", (module, key, cutoff))
        return cur.fetchall()

    def mean_since(self, module: str, key: str, seconds: float):
        cutoff = time.time() - seconds
        cur = self.conn.execute(
            "SELECT AVG(value) FROM samples WHERE module=? AND key=? AND ts>=?",
            (module, key, cutoff))
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None
