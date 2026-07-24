"""Signals in to_payload() must carry module/key so the /demo webui page
(webui/demo.html) can group live observations by originating module without
guessing from the message text.

Also covers the guest-facing vocabulary added on top: every signal must
carry a plain-English label/blurb, and the payload must expose the full
module roster (with running/registered counts) so /demo can show what the
system watches for even when a given module hasn't fired a signal yet.
"""
from __future__ import annotations

import core.registry as registry
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


def test_signals_carry_guest_facing_label_and_blurb():
    snapshot = [
        Result(module="heart_rate", key="bpm", value=71, message="Heart rate 71 bpm",
               severity=Severity.INFO, confidence=0.8),
        Result(module="facial_asymmetry", key="mouth", value=0.03,
               message="Mouth symmetry normal", severity=Severity.INFO, confidence=0.7),
    ]
    payload = to_payload(snapshot, fps=30.0)
    by_module = {s["module"]: s for s in payload["signals"]}
    assert by_module["heart_rate"]["label"] == "Heart rate"
    assert by_module["heart_rate"]["blurb"]
    assert by_module["facial_asymmetry"]["label"] == "Facial symmetry"
    assert by_module["facial_asymmetry"]["blurb"]


def test_payload_exposes_launchable_assessment_roster():
    from assessments import PROTOCOLS
    payload = to_payload([], fps=30.0)
    roster = payload["assessments"]
    # Every runnable protocol is offered to the /demo picker, and only those.
    assert {a["protocol"] for a in roster} == set(PROTOCOLS)
    # Guest-safe: a human label, never the raw underscored slug.
    for a in roster:
        assert a["label"] and a["label"] != a["protocol"]
        assert "_" not in a["label"]
    # Sorted by label so the picker order is stable.
    assert [a["label"] for a in roster] == sorted(a["label"] for a in roster)


def test_payload_exposes_module_roster_with_running_flags():
    registry.discover()
    registered = registry.all_registered()
    payload = to_payload([], fps=30.0, system={"modules_enabled": ["heart_rate", "fall"]})

    assert payload["module_counts"]["registered"] == len(registered)
    assert payload["module_counts"]["running"] == 2

    by_module = {m["module"]: m for m in payload["modules"]}
    assert set(by_module) == set(registered)
    assert by_module["heart_rate"]["running"] is True
    assert by_module["fall"]["running"] is True
    other = [m for m in registered if m not in ("heart_rate", "fall")]
    if other:
        assert by_module[other[0]]["running"] is False

    labels = [m["label"] for m in payload["modules"]]
    assert labels == sorted(labels)


def test_module_roster_running_is_false_when_modules_enabled_is_absent():
    payload_no_system = to_payload([], fps=30.0)
    assert payload_no_system["module_counts"]["running"] == 0
    assert all(m["running"] is False for m in payload_no_system["modules"])

    payload_empty_system = to_payload([], fps=30.0, system={})
    assert payload_empty_system["module_counts"]["running"] == 0
    assert all(m["running"] is False for m in payload_empty_system["modules"])


def test_low_confidence_notice_is_not_promotable_but_high_confidence_info_is():
    # Reproduces the real observed case: masked_face (notice, 0.277 confidence)
    # must not be able to win the guest headline over a confident info signal.
    snapshot = [
        Result(module="masked_face", key="flat_affect", value=0.28,
               message="Reduced facial expressiveness (flat affect / masking — screening)",
               severity=Severity.NOTICE, confidence=0.16),
        Result(module="heart_rate", key="bpm", value=71, message="Heart rate 71 bpm",
               severity=Severity.INFO, confidence=0.9),
    ]
    payload = to_payload(snapshot, fps=30.0)
    by_module = {s["module"]: s for s in payload["signals"]}
    assert by_module["masked_face"]["promote"] is False
    assert by_module["heart_rate"]["promote"] is True


def test_warning_and_alert_are_always_promotable_even_at_low_confidence():
    snapshot = [
        Result(module="fall", key="detected", value=True, message="Possible fall detected",
               severity=Severity.WARNING, confidence=0.05),
        Result(module="unresponsive", key="detected", value=True, message="Unresponsive",
               severity=Severity.ALERT, confidence=0.05),
    ]
    payload = to_payload(snapshot, fps=30.0)
    by_module = {s["module"]: s for s in payload["signals"]}
    assert by_module["fall"]["promote"] is True
    assert by_module["unresponsive"]["promote"] is True


def test_confidence_exactly_at_guest_floor_is_promotable():
    snapshot = [
        Result(module="grooming", key="texture", value=0.5,
               message="Hair looks more textured/unkempt than the usual weekly average",
               severity=Severity.NOTICE, confidence=0.45),
    ]
    payload = to_payload(snapshot, fps=30.0)
    by_module = {s["module"]: s for s in payload["signals"]}
    assert by_module["grooming"]["promote"] is True


def test_low_confidence_notices_are_still_present_not_suppressed():
    # The promotion filter must never drop a signal from the payload -- it
    # only controls whether it may become the headline / a moment card.
    snapshot = [
        Result(module="masked_face", key="flat_affect", value=0.28,
               message="Reduced facial expressiveness (flat affect / masking — screening)",
               severity=Severity.NOTICE, confidence=0.277),
        Result(module="grooming", key="texture", value=0.4,
               message="Hair looks more textured/unkempt than the usual weekly average",
               severity=Severity.NOTICE, confidence=0.157),
    ]
    payload = to_payload(snapshot, fps=30.0)
    modules_present = {s["module"] for s in payload["signals"]}
    assert "masked_face" in modules_present
    assert "grooming" in modules_present
