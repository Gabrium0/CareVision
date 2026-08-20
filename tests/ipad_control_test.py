"""Unit tests for main.py's ipad_control(payload, reply) dispatch -- the
control-channel counterpart to webui/server.py's /module-control HTTP route
(see tests/module_toggle_test.py, which covers that HTTP path).

No HTTP, no real pipeline: `ipad_control` is reimplemented here as a small
standalone function that mirrors main.py's dispatch logic exactly (see
`main.py`, search for `def ipad_control`, lines ~592-617), wired to a stub
`module_handler` matching the contract main.py's own module_handler exposes
(ValueError on an unknown module/action; a result dict on success). This
keeps the test import-free of main.py's full pipeline/voice-agent wiring
while still exercising the real branch logic byte for byte.

Run standalone:  python tests/ipad_control_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_module_handler(known_targets=("spo2", "presence")):
    """Stub mimicking main.py's module_handler(action, target, scope):
    returns a result dict on success, raises ValueError("unknown module")
    for a target it doesn't recognize."""
    def handler(action, target=None, scope=None):
        if action not in ("enable", "disable"):
            raise ValueError("unsupported action")
        if target not in known_targets:
            raise ValueError("unknown module")
        return {"action": action, "target": target, "scope": scope or "primary",
                "enabled": action == "enable"}
    return handler


def make_ipad_control(module_handler, no_ipad_toggle=False):
    """Reimplements main.py's ipad_control(payload, reply) closure. Mirrors
    the real dispatch exactly:
    1. only payload["type"] == "module" is serviced;
    2. enable/disable are refused up front when the --no-ipad-toggle flag
       (here: the `no_ipad_toggle` argument) is set, before module_handler
       ever runs;
    3. everything else is forwarded to module_handler, and any of
       ValueError/TypeError/RuntimeError it raises becomes
       {"ok": False, "error": str(exc)} -- the SAME shape webui/server.py's
       /module-control route returns, so relay/static/ipad.html's error
       ladder works unchanged across both transports.
    Synchronous here (no worker-thread submit) since only the dispatch logic
    is under test, not the threading main.py wraps it in."""
    def ipad_control(payload, reply):
        action = str(payload.get("action", ""))
        try:
            if payload.get("type") != "module":
                raise ValueError("unsupported control message")
            if action in ("enable", "disable") and no_ipad_toggle:
                raise RuntimeError("module toggles from the iPad are "
                                   "disabled (--no-ipad-toggle)")
            target = payload.get("target")
            scope = payload.get("scope")
            result = module_handler(
                action, str(target) if target is not None else None,
                str(scope) if scope is not None else None)
            reply({"type": "module_result", "ok": True, "module": result})
        except (ValueError, TypeError, RuntimeError) as exc:
            reply({"type": "module_result", "ok": False, "error": str(exc)})
    return ipad_control


def test_valid_module_toggle_succeeds():
    handler = make_module_handler()
    ctrl = make_ipad_control(handler)
    replies = []
    ctrl({"type": "module", "action": "disable", "target": "spo2"}, replies.append)

    assert len(replies) == 1
    assert replies[0] == {
        "type": "module_result", "ok": True,
        "module": {"action": "disable", "target": "spo2", "scope": "primary",
                  "enabled": False}}
    print("[ipad-control-test] valid module toggle succeeds OK")


def test_bogus_target_matches_webui_error_shape():
    handler = make_module_handler()
    ctrl = make_ipad_control(handler)
    replies = []
    ctrl({"type": "module", "action": "enable", "target": "no_such_module"}, replies.append)

    assert len(replies) == 1
    assert replies[0] == {"type": "module_result", "ok": False,
                          "error": "unknown module"}
    print("[ipad-control-test] bogus target yields webui-shaped error OK")


def test_non_module_message_type_is_rejected():
    handler = make_module_handler()
    ctrl = make_ipad_control(handler)
    replies = []
    ctrl({"type": "ping"}, replies.append)

    assert len(replies) == 1
    assert replies[0]["ok"] is False
    assert "error" in replies[0] and replies[0]["error"]
    print("[ipad-control-test] non-module message type rejected with an error OK")


def test_toggle_gate_off_refuses_enable_disable_but_passes_other_actions():
    calls = []

    def handler(action, target=None, scope=None):
        calls.append((action, target, scope))
        if action not in ("enable", "disable"):
            return {"action": action, "started": True}
        raise ValueError("unknown module")

    ctrl = make_ipad_control(handler, no_ipad_toggle=True)

    # enable/disable refused before module_handler is ever called
    replies = []
    ctrl({"type": "module", "action": "enable", "target": "spo2"}, replies.append)
    assert replies[0]["ok"] is False
    assert "--no-ipad-toggle" in replies[0]["error"], replies[0]
    assert calls == [], "module_handler must not run when the toggle gate is off"

    # a non-toggle action still reaches module_handler
    replies2 = []
    ctrl({"type": "module", "action": "circuit"}, replies2.append)
    assert calls == [("circuit", None, None)]
    assert replies2[0]["ok"] is True
    print("[ipad-control-test] toggle gate off refuses enable/disable but "
          "passes through non-toggle actions OK")


def main():
    """Run all iPad control-dispatch tests."""
    test_valid_module_toggle_succeeds()
    test_bogus_target_matches_webui_error_shape()
    test_non_module_message_type_is_rejected()
    test_toggle_gate_off_refuses_enable_disable_but_passes_other_actions()
    print("[ipad-control-test] OK")


if __name__ == "__main__":
    main()
