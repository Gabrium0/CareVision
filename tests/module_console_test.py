"""Anti-drift gates for the /modules showcase console.

The console exists to make capability AND limitation legible during a client
demo, so the risky failure is silent drift: a roster claiming a module needs
something it does not, offering a button that fires nothing, or quietly losing
a module's honest reliability rating. Every fact the page shows is therefore
asserted against the registered classes themselves, and /module-control is
held to the same narrow contract as the other control routes.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import core.registry as registry
from output.dashboard import (_MODULE_RELIABILITY, _MODULE_TRIGGERS,
                              _reliability, to_payload)
from webui.server import CompanionServer

_TIERS = {"HIGH", "MEDIUM", "LOW", "N/A"}
_ELICITATIONS = {"hold_still", "arm_check"}


def _modules(system=None) -> dict[str, dict]:
    """Roster from a real to_payload call, keyed by module slug."""
    registry.discover()
    payload = to_payload([], fps=30.0, system=system or {})
    return {entry["module"]: entry for entry in payload["modules"]}


# ------------------------------------------------------------------- roster

def test_every_registered_module_is_in_the_roster():
    registry.discover()
    roster = _modules()
    assert set(roster) == set(registry.all_registered())


def test_roster_mirrors_the_registered_classes_exactly():
    # The page tells the audience what a detector needs; if this drifts, the
    # console explains an idle module with the wrong reason.
    registry.discover()
    roster = _modules()
    for slug, cls in registry.all_registered().items():
        entry = roster[slug]
        assert entry["requires"] == list(getattr(cls, "requires", ()) or ())
        assert entry["interval"] == pytest.approx(
            round(float(getattr(cls, "interval", 0.0) or 0.0), 2))
        assert entry["consent"] is bool(getattr(cls, "consent", False))


def test_requires_only_uses_tokens_the_scheduler_understands():
    for entry in _modules().values():
        assert set(entry["requires"]) <= {"face", "pose", "person", "depth"}


def test_reliability_tiers_are_known_and_shaped():
    for entry in _modules().values():
        rating = entry["reliability"]
        if rating is None:
            continue
        assert rating["tier"] in _TIERS
        assert rating["note"], f"{entry['module']} has a tier but no reason"


def test_reliability_table_has_no_entries_for_unregistered_modules():
    registry.discover()
    unknown = set(_MODULE_RELIABILITY) - set(registry.all_registered())
    assert not unknown, f"reliability rates modules that do not exist: {unknown}"


def test_every_registered_module_is_rated():
    # An unrated detector renders as "UNRATED" to a client audience, which
    # reads as an oversight rather than the honest limit it is meant to convey.
    registry.discover()
    unrated = sorted(set(registry.all_registered()) - set(_MODULE_RELIABILITY))
    assert not unrated, f"add a reliability tier for: {unrated}"


def test_unrated_module_returns_none_rather_than_a_guess():
    assert _reliability("some_module_that_does_not_exist") is None


# ------------------------------------------------------------------ triggers

def test_triggers_reference_real_modules_and_valid_actions():
    registry.discover()
    from assessments import PROTOCOLS
    valid_targets = set(PROTOCOLS) | _ELICITATIONS
    unknown = set(_MODULE_TRIGGERS) - set(registry.all_registered())
    assert not unknown, f"triggers on modules that do not exist: {unknown}"
    for slug, trigger in _MODULE_TRIGGERS.items():
        assert trigger["action"] in ("test", "circuit", "vlm_scan")
        assert trigger["label"]
        if trigger["action"] == "test":
            # Must match main.py's module_handler allowlist or the button 400s.
            assert trigger["target"] in valid_targets, slug
        else:
            assert trigger["target"] is None


def test_passive_modules_expose_no_trigger():
    # A button that fires nothing is worse than no button in front of clients.
    roster = _modules()
    for slug, entry in roster.items():
        if slug not in _MODULE_TRIGGERS:
            assert entry["trigger"] is None


# ------------------------------------------------------- feature availability

def test_frame_features_reach_the_payload():
    features = {"face": True, "pose": False, "person": True, "depth": False}
    registry.discover()
    payload = to_payload([], fps=30.0, system={"features": features})
    assert payload["system"]["features"] == features


def test_missing_features_block_does_not_break_the_roster():
    assert _modules(system={})  # console must still render


def test_console_renders_complete_values_and_retains_last_known_readings():
    html = (Path(__file__).resolve().parents[1] / "webui" / "modules.html").read_text(
        encoding="utf-8")
    assert "module_readings" in html
    assert "reading.value" in html
    assert "readingCache" in html
    assert "Last known" in html


# -------------------------------------------------------- /module-control

def _post(port: int, body: dict, path: str = "/module-control"):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})

    def _body(raw: bytes):
        try:
            return json.loads(raw.decode())
        except json.JSONDecodeError:
            return raw.decode()

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, _body(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _body(exc.read())


@pytest.fixture
def served():
    """Server with a handler mirroring main.py's module_handler contract."""
    calls = []

    def handler(action, target=None):
        calls.append((action, target))
        if action == "circuit":
            return {"action": "circuit", "started": True}
        if action != "test":
            raise ValueError("unsupported action")
        if target not in _ELICITATIONS:
            raise ValueError("unknown test")
        return {"action": "test", "target": target, "started": True}

    server = CompanionServer(port=0, module_handler=handler)
    server.start()
    try:
        yield server, server._httpd.server_address[1], calls
    finally:
        server.stop()


def test_test_action_forwards_target(served):
    _server, port, calls = served
    status, body = _post(port, {"action": "test", "target": "arm_check"})
    assert status == 200
    assert body["ok"] is True and body["module"]["target"] == "arm_check"
    assert calls == [("test", "arm_check")]


def test_circuit_action_needs_no_target(served):
    _server, port, calls = served
    status, body = _post(port, {"action": "circuit"})
    assert status == 200 and body["module"]["started"] is True
    assert calls == [("circuit", None)]


def test_unknown_action_and_target_are_rejected(served):
    _server, port, _calls = served
    assert _post(port, {"action": "explode"})[0] == 400
    assert _post(port, {"action": "test", "target": "nope"})[0] == 400


def test_route_is_unavailable_without_a_handler():
    server = CompanionServer(port=0)
    server.start()
    try:
        status, body = _post(server._httpd.server_address[1],
                             {"action": "circuit"})
        assert status == 400
        assert body["ok"] is False and "not available" in body["error"]
    finally:
        server.stop()
