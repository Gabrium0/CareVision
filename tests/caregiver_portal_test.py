"""Durable caregiver cases, alert semantics, and loopback portal contracts."""
from __future__ import annotations

import csv
import io
import json
import threading
import urllib.error
import urllib.request

import pytest
import numpy as np

from alerts.manager import AlertManager
from alerts.notifier import Channel
from core.events import Result, Severity, Visibility
from core.context import FrameContext
from modules.replay_events import ReplayEvents
from storage.event_store import (CaseConflictError, EventStore)
from storage.history_store import HistoryStore
from webui.caregiver_server import CaregiverServer


def alert(subject="primary", key="fall"):
    return Result("fall", key, True, .91, Severity.ALERT,
                  "Fall-like event confirmed", subject_id=subject,
                  source="camera", quality=.88, location="living_room")


class Recorder(Channel):
    name = "recorder"

    def __init__(self):
        self.messages = []

    def send(self, subject, body):
        self.messages.append((subject, body))
        return True


def test_case_lifecycle_subject_isolation_restart_and_retention(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    first = store.open_alert_case(alert("primary"), timestamp=10)
    visitor = store.open_alert_case(alert("track-2"), timestamp=11)
    assert first["id"] != visitor["id"]
    assert store.subjects() == ["primary", "track-2"]

    acknowledged = store.mutate_case(first["id"], "acknowledge", first["version"],
                                     "Checked in", timestamp=12)
    assert acknowledged["status"] == "acknowledged"
    with pytest.raises(CaseConflictError):
        store.mutate_case(first["id"], "note", first["version"], "stale")
    noted = store.mutate_case(first["id"], "note", acknowledged["version"],
                              "Walking normally", timestamp=13)
    resolved = store.mutate_case(first["id"], "resolve", noted["version"],
                                 timestamp=14)
    assert resolved["status"] == "resolved"
    store.mark_case_signal(first["id"], False, timestamp=15)
    recurrence = store.open_alert_case(alert("primary"), timestamp=16)
    assert recurrence["id"] != first["id"]

    store.interrupt_active_cases(timestamp=17)
    interrupted = store.case(recurrence["id"])
    assert interrupted["signal_active"] is False
    assert interrupted["actions"][-1]["action"] == "monitoring_restarted"
    reused = store.open_alert_case(alert("primary"), timestamp=18)
    assert reused["id"] == recurrence["id"]

    unresolved = store.open_alert_case(alert("track-3"), timestamp=20)
    store.mark_case_signal(unresolved["id"], False, timestamp=21)
    new_activation = store.open_alert_case(alert("track-3"), timestamp=22)
    assert new_activation["id"] != unresolved["id"]

    assert store.purge_expired(now=14 + 90 * 86400 + 1) == 1
    assert store.case(first["id"]) is None
    assert store.case(visitor["id"]) is not None
    store.close()


def test_restart_reuses_acknowledged_case_without_resending_initial_notification(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    channel = Recorder()
    first_manager = AlertManager(channels=[channel], confirm_seconds=0,
                                 cooldown_seconds=1, escalate_after=5,
                                 case_store=store)
    first_manager.evaluate([alert()], now=0)
    case = store.list_cases()[0]
    store.mutate_case(case["id"], "acknowledge", case["version"], timestamp=1)
    store.interrupt_active_cases(timestamp=2)

    restarted = AlertManager(channels=[channel], confirm_seconds=0,
                             cooldown_seconds=1, escalate_after=5,
                             case_store=store)
    restarted.evaluate([alert()], now=2)
    assert len(channel.messages) == 1
    restarted.evaluate([alert()], now=6)
    assert len(channel.messages) == 2
    assert channel.messages[-1][0].startswith("ESCALATION:")
    store.close()


def test_acknowledgement_pauses_reminders_but_keeps_escalation_and_resolution(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    channel = Recorder()
    manager = AlertManager(channels=[channel], confirm_seconds=0,
                           cooldown_seconds=1, escalate_after=5,
                           case_store=store)
    result = alert()
    manager.evaluate([result], now=0)
    case = store.list_cases()[0]
    assert len(channel.messages) == 1
    case = store.mutate_case(case["id"], "acknowledge", case["version"],
                             timestamp=1)
    manager.evaluate([result], now=2)
    assert len(channel.messages) == 1
    manager.evaluate([result], now=6)
    assert len(channel.messages) == 2
    assert channel.messages[-1][0].startswith("ESCALATION:")

    case = store.case(case["id"])
    store.mutate_case(case["id"], "resolve", case["version"], timestamp=7)
    manager.evaluate([result], now=8)
    assert len(channel.messages) == 2
    manager.evaluate([], now=9)
    manager.evaluate([result], now=10)
    assert len(channel.messages) == 3
    assert len(store.list_cases()) == 2
    store.close()


def test_concurrent_case_version_allows_only_one_writer(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    case = store.open_alert_case(alert(), timestamp=1)
    outcomes = []

    def update(note):
        try:
            store.mutate_case(case["id"], "note", case["version"], note)
            outcomes.append("ok")
        except CaseConflictError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=update, args=(value,)) for value in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["conflict", "ok"]
    store.close()


def test_private_or_encoded_media_alert_cannot_create_case(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    private = alert()
    private.visibility = Visibility.AGENT_ONLY
    with pytest.raises(ValueError):
        store.open_alert_case(private)
    encoded = alert()
    encoded.message = "data:image/jpeg;base64,AAAA"
    with pytest.raises(TypeError):
        store.open_alert_case(encoded)
    assert store.list_cases() == []
    store.close()


def _request(url, *, method="GET", body=None, token=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Caregiver-CSRF"] = token
    return urllib.request.urlopen(urllib.request.Request(
        url, data=data, headers=headers, method=method), timeout=5)


def test_loopback_portal_routes_security_exports_and_purge(tmp_path):
    events = EventStore(tmp_path / "events.sqlite3")
    history = HistoryStore(tmp_path / "history.sqlite3")
    history.add("routine", "activity", .4, ts=100, subject_id="track-2")
    events.record("observation", {"value": 72}, subject_id="track-2",
                  module="heart_rate", key="bpm", timestamp=100)
    case = events.open_alert_case(alert("track-2"), timestamp=100)
    server = CaregiverServer(events, history, port=0)
    server.service.now = lambda: 110.0
    server.start()
    try:
        host, port = server._httpd.server_address
        assert host == "127.0.0.1"
        base = f"http://127.0.0.1:{port}/caregiver"
        with _request(base) as response:
            html = response.read().decode("utf-8")
            assert "Caregiver portal" in html and "portal.js" in html
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        with _request(base + "/portal.js") as response:
            script = response.read().decode("utf-8")
            assert "createElementNS" in script and "innerHTML" not in script
        with _request(base + "/api/bootstrap") as response:
            bootstrap = json.load(response)
        assert bootstrap["subjects"] == ["primary", "track-2"]
        token = bootstrap["csrf_token"]
        with _request(base + "/api/summary?subject_id=track-2&range=24h") as response:
            summary = json.load(response)
        assert summary["counts"] and summary["series"]
        assert all(item["subject_id"] == "track-2" for item in summary["cases"])

        with pytest.raises(urllib.error.HTTPError) as missing_csrf:
            _request(base + f"/api/cases/{case['id']}/acknowledge",
                     method="POST", body={"version": case["version"]})
        assert missing_csrf.value.code == 403
        with _request(base + f"/api/cases/{case['id']}/acknowledge",
                      method="POST", token=token,
                      body={"version": case["version"],
                            "note": "=HYPERLINK(\"bad\")"}) as response:
            acknowledged = json.load(response)["case"]
        assert acknowledged["status"] == "acknowledged"
        with pytest.raises(urllib.error.HTTPError) as stale:
            _request(base + f"/api/cases/{case['id']}/note", method="POST",
                     token=token, body={"version": case["version"], "note": "stale"})
        assert stale.value.code == 409

        with _request(base + "/api/export?subject_id=track-2&range=24h&format=csv") as response:
            rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))
            assert "attachment" in response.headers["Content-Disposition"]
        assert {row["record_type"] for row in rows} >= {"event", "case", "case_action"}
        assert next(row for row in rows if row["action"] == "acknowledged")["note"].startswith("'")
        with _request(base + "/api/export?subject_id=track-2&range=24h&format=json") as response:
            exported = json.load(response)
        assert exported["subject_id"] == "track-2"
        assert "frame" not in json.dumps(exported).lower()

        with _request(base + "/api/retention/purge-expired", method="POST",
                      token=token, body={"confirm": "purge-expired"}) as response:
            assert json.load(response)["ok"] is True
    finally:
        server.stop()
        events.close()
        history.close()


def test_history_series_downsamples_and_purges(tmp_path):
    history = HistoryStore(tmp_path / "history.sqlite3")
    for index in range(900):
        history.add("sensor", "heart_rate_bpm", index % 100,
                    ts=index, subject_id="primary")
    series = history.portal_series("primary", 0, 900)
    assert len(series) == 1 and len(series[0]["points"]) <= 300
    assert set(series[0]["points"][0]) == {"timestamp", "mean", "min", "max", "count"}
    assert history.purge_before(450) == 450
    history.close()


def test_replay_alert_acknowledge_clear_resolve_and_recur(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    channel = Recorder()
    manager = AlertManager(channels=[channel], confirm_seconds=0,
                           cooldown_seconds=30, escalate_after=300,
                           case_store=store)
    adapter = ReplayEvents()

    def replay_result(timestamp):
        ctx = FrameContext(np.zeros((2, 2, 3), dtype=np.uint8), 1, timestamp, 10)
        ctx.extras["replay_events"] = [{"module": "fall", "key": "fall",
                                        "value": True, "severity": "alert",
                                        "message": "Replay fall confirmed",
                                        "subject_id": "track-2"}]
        return adapter.process(ctx)[0]

    manager.evaluate([replay_result(1)], now=1)
    first = store.list_cases(subject_id="track-2")[0]
    acknowledged = store.mutate_case(first["id"], "acknowledge", first["version"],
                                     timestamp=2)
    manager.evaluate([], now=3)
    cleared = store.case(first["id"])
    assert cleared["signal_active"] is False
    store.mutate_case(first["id"], "resolve", cleared["version"], timestamp=4)
    manager.evaluate([replay_result(10)], now=10)
    cases = store.list_cases(subject_id="track-2")
    assert len(cases) == 2 and cases[0]["id"] != acknowledged["id"]
    store.close()
