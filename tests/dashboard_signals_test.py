"""Signals in to_payload() must carry module/key so the /demo webui page
(webui/demo.html) can group live observations by originating module without
guessing from the message text."""
from __future__ import annotations

from core.events import Result, Severity
from output.dashboard import to_payload


def test_signals_expose_module_and_key():
    snapshot = [
        Result(module="heart_rate", key="bpm", value=71, message="Heart rate 71 bpm",
               severity=Severity.INFO, confidence=0.8),
        Result(module="facial_asymmetry", key="mouth", value=0.03,
               message="Mouth symmetry normal", severity=Severity.INFO, confidence=0.7),
    ]
    payload = to_payload(snapshot, fps=30.0)
    by_module = {s["module"]: s for s in payload["signals"]}
    assert by_module["heart_rate"]["key"] == "bpm"
    assert by_module["facial_asymmetry"]["key"] == "mouth"
