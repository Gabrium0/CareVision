"""SQLite hot-path writes are batched but remain readable and flushable."""
from pathlib import Path

from storage.event_store import EventStore
from storage.history_store import HistoryStore


def test_history_store_batches_and_reads_own_writes(tmp_path: Path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.add("presence", "present", 1.0, ts=100.0)
    assert store._pending_writes == 1
    assert store.last("presence", "present") == (100.0, 1.0)
    store.flush()
    assert store._pending_writes == 0
    store.close()


def test_event_store_batches_and_flushes(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.record("test", {"value": 1}, timestamp=100.0)
    assert store._pending_writes == 1
    assert len(store.recent()) == 1
    store.flush()
    assert store._pending_writes == 0
    store.close()
