"""ModuleGate scope semantics, and the /module-control HTTP toggle contract.

Two layers, matching how the feature is actually wired:

1. `ModuleGate` (core/module_gate.py) directly -- enable/disable/snapshot and
   the primary/secondary scope isolation it promises in its own docstring.
2. The HTTP route (webui/server.py's `/module-control` POST branch) against a
   real `CompanionServer` on an ephemeral port, with a stub `module_handler`
   that mirrors main.py's actual enable/disable contract (ValueError on an
   unknown scope or a target not loaded for that scope -> HTTP 400), plus the
   loopback-only guard new to this feature: enable/disable is rejected 403
   for a non-loopback caller unless `allow_remote_module_toggle=True`.

For (2)'s loopback boundary, a real `urllib.request` client run in the same
test process necessarily connects via 127.0.0.1, so it cannot organically
produce a non-loopback `client_address`. Rather than skip that path, the
socket-accept layer (`QuietThreadingHTTPServer.get_request`) is monkeypatched
to report a spoofed public IP for the already-accepted connection -- this
still exercises the real HTTP route end-to-end (request parsing, JSON
response, status code), only the peer-address fact is faked. The pure
`_is_loopback` predicate the guard is built on is also unit-tested directly
per the task's fallback guidance, so the address-classification logic has
coverage independent of the monkeypatch.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from core.module_gate import ModuleGate, seed_start_paused
from webui._http import QuietThreadingHTTPServer
from webui.server import CompanionServer, _is_loopback

# --------------------------------------------------------------- ModuleGate


def test_gate_defaults_to_all_disabled():
    gate = ModuleGate()
    assert gate.enabled("presence") is False
    assert gate.enabled("presence", scope="secondary") is False
    assert gate.snapshot() == {"primary": [], "secondary": []}


def test_gate_seeds_from_constructor_sets():
    gate = ModuleGate(primary_enabled={"presence", "gait"},
                      secondary_enabled={"presence"})
    assert gate.enabled("presence") is True
    assert gate.enabled("gait") is True
    assert gate.enabled("presence", scope="secondary") is True
    assert gate.enabled("gait", scope="secondary") is False
    assert gate.snapshot() == {"primary": ["gait", "presence"],
                               "secondary": ["presence"]}


def test_set_enable_disable_round_trips():
    gate = ModuleGate()
    assert gate.enabled("presence") is False
    gate.set("presence", True)
    assert gate.enabled("presence") is True
    gate.set("presence", False)
    assert gate.enabled("presence") is False


def test_scopes_are_isolated():
    # Toggling primary must never leak into secondary and vice versa -- the
    # class docstring promises exactly this.
    gate = ModuleGate()
    gate.set("presence", True, scope="primary")
    assert gate.enabled("presence", scope="primary") is True
    assert gate.enabled("presence", scope="secondary") is False

    gate.set("presence", True, scope="secondary")
    gate.set("presence", False, scope="primary")
    assert gate.enabled("presence", scope="primary") is False
    assert gate.enabled("presence", scope="secondary") is True


def test_snapshot_is_sorted_public_state():
    gate = ModuleGate()
    gate.set("zeta", True)
    gate.set("alpha", True)
    gate.set("alpha", True, scope="secondary")
    snap = gate.snapshot()
    assert snap == {"primary": ["alpha", "zeta"], "secondary": ["alpha"]}


# ------------------------------------------------------- start_paused seed


def test_start_paused_boots_modules_off_in_both_scopes():
    gate = ModuleGate(primary_enabled={"clothing", "presence"},
                      secondary_enabled={"clothing"})
    applied, ignored = seed_start_paused(gate, ["clothing"],
                                         {"clothing", "presence"})
    assert applied == ["clothing"]
    assert ignored == []
    assert gate.enabled("clothing", "primary") is False
    assert gate.enabled("clothing", "secondary") is False
    # Untouched modules keep their boot state.
    assert gate.enabled("presence", "primary") is True


def test_start_paused_modules_stay_warm_and_re_enableable():
    """The whole point of the gate: pausing never unloads, so recovery is
    one toggle, not a restart."""
    gate = ModuleGate(primary_enabled={"grooming"})
    applied, _ignored = seed_start_paused(gate, ["grooming"], {"grooming"})
    assert applied == ["grooming"]
    gate.set("grooming", True)
    assert gate.enabled("grooming", "primary") is True


@pytest.mark.parametrize("module_name", ["pain", "facial_asymmetry"])
def test_showcase_noise_modules_stay_warm_and_re_enableable(module_name):
    gate = ModuleGate(primary_enabled={module_name},
                      secondary_enabled={module_name})
    applied, ignored = seed_start_paused(gate, [module_name], {module_name})
    assert applied == [module_name]
    assert ignored == []
    assert gate.enabled(module_name, "primary") is False
    assert gate.enabled(module_name, "secondary") is False

    gate.set(module_name, True, scope="primary")
    assert gate.enabled(module_name, "primary") is True
    assert gate.enabled(module_name, "secondary") is False


def test_start_paused_ignores_unknown_names_without_side_effects():
    gate = ModuleGate(primary_enabled={"presence"})
    applied, ignored = seed_start_paused(gate, ["no_such_module", "presence"],
                                         {"presence"})
    assert applied == ["presence"]
    assert ignored == ["no_such_module"]
    assert gate.snapshot() == {"primary": [], "secondary": []}


def test_start_paused_empty_config_is_a_no_op():
    gate = ModuleGate(primary_enabled={"presence"})
    applied, ignored = seed_start_paused(gate, None, {"presence"})
    assert (applied, ignored) == ([], [])
    assert gate.enabled("presence", "primary") is True


def test_shipped_start_paused_config_is_wellformed():
    """Guard the hand-edited YAML: the key must be a list of plain names,
    and every entry must be a registered module -- a typo here would only
    surface as an 'ignoring unknown' log line nobody reads."""
    import yaml
    from core.registry import all_registered, discover
    discover()
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "modules.yaml")
        .read_text(encoding="utf-8"))
    paused = ((config or {}).get("runtime") or {}).get("start_paused")
    assert isinstance(paused, list) and paused
    assert all(isinstance(name, str) and name for name in paused)
    assert {"pain", "facial_asymmetry"} <= set(paused)
    unknown = [name for name in paused if name not in all_registered()]
    assert unknown == []


# ------------------------------------------------------- loopback predicate


def test_is_loopback_accepts_known_loopback_forms():
    assert _is_loopback("127.0.0.1") is True
    assert _is_loopback("::1") is True
    assert _is_loopback("::ffff:127.0.0.1") is True


def test_is_loopback_rejects_lan_and_public_addresses():
    assert _is_loopback("192.168.1.50") is False
    assert _is_loopback("10.0.0.5") is False
    assert _is_loopback("203.0.113.5") is False
    assert _is_loopback("::2") is False


# ------------------------------------------------------------ HTTP contract


def _post(port: int, body: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/module-control",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _stub_handler(gate, primary_loaded, secondary_loaded):
    """Mirrors main.py's module_handler enable/disable contract exactly:
    ValueError for an unknown scope or a target not loaded for that scope,
    a gate mutation plus its resulting state on success."""
    calls = []

    def handler(action, target=None, scope=None):
        calls.append((action, target, scope))
        if action not in ("enable", "disable"):
            raise ValueError("unsupported action")
        effective_scope = scope or "primary"
        if effective_scope not in ("primary", "secondary"):
            raise ValueError("unknown scope")
        loaded = primary_loaded if effective_scope == "primary" else secondary_loaded
        if target not in loaded:
            raise ValueError(
                "module is not loaded for this scope; enable it in "
                "config/modules.yaml and restart")
        gate.set(target, action == "enable", scope=effective_scope)
        return {"action": action, "target": target, "scope": effective_scope,
                "enabled": gate.enabled(target, effective_scope)}

    return handler, calls


@pytest.fixture
def toggle_served():
    """Real CompanionServer wired to a stub gate-backed module_handler,
    loopback-only (default) since the test client itself connects via
    127.0.0.1."""
    gate = ModuleGate()
    handler, calls = _stub_handler(gate, primary_loaded={"presence"},
                                   secondary_loaded={"presence"})
    server = CompanionServer(port=0, module_handler=handler)
    server.start()
    try:
        yield server, server._httpd.server_address[1], gate, calls
    finally:
        server.stop()


def test_enable_disable_round_trip_through_the_gate(toggle_served):
    _server, port, gate, _calls = toggle_served
    status, body = _post(port, {"action": "enable", "target": "presence",
                                "scope": "primary"})
    assert status == 200
    assert body["ok"] is True
    assert body["module"] == {"action": "enable", "target": "presence",
                              "scope": "primary", "enabled": True}
    assert gate.enabled("presence", "primary") is True

    status, body = _post(port, {"action": "disable", "target": "presence",
                                "scope": "primary"})
    assert status == 200
    assert body["module"]["enabled"] is False
    assert gate.enabled("presence", "primary") is False


def test_scope_defaults_to_primary_when_omitted(toggle_served):
    _server, port, gate, _calls = toggle_served
    status, body = _post(port, {"action": "enable", "target": "presence"})
    assert status == 200
    assert body["module"]["scope"] == "primary"
    assert gate.enabled("presence", "primary") is True


def test_secondary_scope_toggles_independently_of_primary(toggle_served):
    _server, port, gate, _calls = toggle_served
    status, body = _post(port, {"action": "enable", "target": "presence",
                                "scope": "secondary"})
    assert status == 200
    assert body["module"]["scope"] == "secondary"
    assert gate.enabled("presence", "secondary") is True
    assert gate.enabled("presence", "primary") is False  # untouched


def test_unknown_scope_yields_400_with_error_body(toggle_served):
    _server, port, _gate, _calls = toggle_served
    status, body = _post(port, {"action": "enable", "target": "presence",
                                "scope": "tertiary"})
    assert status == 400
    assert body["ok"] is False
    assert "scope" in body["error"]


def test_unloaded_target_yields_400_with_error_body(toggle_served):
    _server, port, _gate, _calls = toggle_served
    status, body = _post(port, {"action": "enable", "target": "no_such_module",
                                "scope": "primary"})
    assert status == 400
    assert body["ok"] is False
    assert "not loaded" in body["error"]


# ---------------------------------------------------- loopback-only guard


def _spoof_client_ip(monkeypatch, fake_ip: str):
    """Make every connection accepted by QuietThreadingHTTPServer report
    `fake_ip` as its peer address, without touching the real socket -- a
    same-process urllib client cannot otherwise present as non-loopback."""
    real_get_request = QuietThreadingHTTPServer.get_request

    def spoofed(self):
        conn, addr = real_get_request(self)
        return conn, (fake_ip, addr[1])

    monkeypatch.setattr(QuietThreadingHTTPServer, "get_request", spoofed)


def test_remote_toggle_rejected_403_when_not_allowed(monkeypatch):
    gate = ModuleGate()
    handler, _calls = _stub_handler(gate, primary_loaded={"presence"},
                                    secondary_loaded=set())
    server = CompanionServer(port=0, module_handler=handler)  # default: loopback only
    server.start()
    port = server._httpd.server_address[1]
    _spoof_client_ip(monkeypatch, "203.0.113.5")
    try:
        status, body = _post(port, {"action": "enable", "target": "presence",
                                    "scope": "primary"})
        assert status == 403
        assert body["ok"] is False
        assert "loopback" in body["error"]
        assert gate.enabled("presence", "primary") is False  # handler never ran
    finally:
        server.stop()


def test_remote_toggle_allowed_when_opted_in(monkeypatch):
    gate = ModuleGate()
    handler, _calls = _stub_handler(gate, primary_loaded={"presence"},
                                    secondary_loaded=set())
    server = CompanionServer(port=0, module_handler=handler,
                             allow_remote_module_toggle=True)
    server.start()
    port = server._httpd.server_address[1]
    _spoof_client_ip(monkeypatch, "203.0.113.5")
    try:
        status, body = _post(port, {"action": "enable", "target": "presence",
                                    "scope": "primary"})
        assert status == 200
        assert body["ok"] is True
        assert gate.enabled("presence", "primary") is True
    finally:
        server.stop()


def test_non_toggle_actions_reach_the_handler_from_a_remote_client(monkeypatch):
    # Only enable/disable are gated; a "circuit"-style action (mirrored here
    # via a handler that returns an error to prove it was actually called
    # rather than rejected by the 403 gate) must still reach module_handler.
    calls = []

    def handler(action, target=None, scope=None):
        calls.append((action, target, scope))
        raise ValueError("unsupported action")

    server = CompanionServer(port=0, module_handler=handler)
    server.start()
    port = server._httpd.server_address[1]
    _spoof_client_ip(monkeypatch, "203.0.113.5")
    try:
        status, body = _post(port, {"action": "circuit"})
        assert status == 400  # from the handler's ValueError, not the 403 gate
        assert calls == [("circuit", None, None)]
        assert "unsupported action" in body["error"]
    finally:
        server.stop()
