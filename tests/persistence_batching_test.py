"""SQLite hot-path writes are batched but remain readable and flushable."""
from pathlib import Path

from storage.event_store import EventStore
from storage.history_store import HistoryStore


def test_history_store_batches_and_reads_own_writes(tmp_path: Path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.add("presence", "present", 1.0, ts=100.0)
    assert store.last("presence", "present") == (100.0, 1.0)
    assert store.diagnostics()["failures"] == 0
    store.flush()
    assert store._pending_writes == 0
    store.close()
    store.close()


def test_event_store_batches_and_flushes(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.record("test", {"value": 1}, timestamp=100.0)
    assert store._pending_writes == 1
    assert len(store.recent()) == 1
    store.flush()
    assert store._pending_writes == 0
    store.close()


def test_rolling_aggregate_bootstraps_and_updates_without_detector_sql(tmp_path: Path):
    store = HistoryStore(tmp_path / "rolling.sqlite3")
    try:
        assert store.rolling_mean("grooming", "hair", 3600.0, now=100.0) is None
        assert store.wait_aggregates()
        store.add("grooming", "hair", 2.0, ts=101.0)
        store.add("grooming", "hair", 4.0, ts=102.0)
        assert store.rolling_mean("grooming", "hair", 3600.0, now=103.0) == 3.0
        state = store.diagnostics()["aggregates"]
        assert state["alive"] and state["windows"] == 1 and state["failures"] == 0
    finally:
        store.close()
