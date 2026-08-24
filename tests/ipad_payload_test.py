"""Anti-drift gates for the telemetry pushed to the paired iPad.

The iPad's dashboard is fed over the WebRTC control channel, not HTTP, so two
things must never silently break:

* **Size.** aiortc advertises a=max-message-size:65536 and Safari throws on
  send() past it. If ipad_payload() ever grows past the ceiling, telemetry
  stops arriving with no error the operator can see -- so the size is asserted
  here against a full module roster.
* **Shape.** The page renders vitals tiles, a signal feed, and a module grid
  straight off this dict; a dropped or renamed field blanks part of the
  dashboard mid-demo. The fields the page depends on are pinned below.
"""
from __future__ import annotations

import json

import core.registry as registry
from core.events import Result, Severity
from output.dashboard import (IPAD_PAYLOAD_MAX_BYTES, _IPAD_MAX_SIGNALS,
                              _IPAD_VITALS, ipad_payload)


def _busy_snapshot() -> list[Result]:
    """A deliberately noisy snapshot: every vital present (canonical +
    backend-suffixed) plus more signals than the feed will ever show."""
    now = 1_000_000.0
    snap = [
        Result("heart_rate", "bpm", 72.0, 0.8, Severity.INFO,
               "HR ~72 bpm (classical)", ttl=8.0, timestamp=now),
        Result("heart_rate", "bpm_open_rppg", 69.0, 0.6, Severity.INFO,
               "HR ~69 bpm (open_rppg)", ttl=8.0, timestamp=now),
        Result("respiration", "breaths_per_min", 15.0, 0.7, Severity.INFO,
               "Respiration ~15 breaths/min", ttl=10.0, timestamp=now),
        Result("spo2", "spo2", 97.0, 0.5, Severity.INFO,
               "SpO₂ ~97% (uncalibrated, trend only)", ttl=8.0, timestamp=now),
        Result("emotion", "emotion_hsemotion", "Content", 0.6, Severity.INFO,
               "Mood: content", ttl=6.0, timestamp=now),
    ]
    # A pile of warnings so the feed has to trim to _IPAD_MAX_SIGNALS.
    for i in range(40):
        snap.append(Result("agitation", f"note_{i}", True, 0.5, Severity.WARNING,
                           f"restless movement sample {i}", ttl=10.0, timestamp=now))
    return snap


def _payload(system=None):
    registry.discover()
    return ipad_payload(_busy_snapshot(), fps=18.0,
                        greeting="Good afternoon.", system=system or {})


def test_serialized_payload_stays_under_the_channel_ceiling():
    # The worst case that matters: every registered module in the roster.
    encoded = json.dumps(_payload()).encode("utf-8")
    assert len(encoded) < IPAD_PAYLOAD_MAX_BYTES, (
        f"iPad telemetry is {len(encoded)}B, over the "
        f"{IPAD_PAYLOAD_MAX_BYTES}B control-channel ceiling; trim it or the "
        f"paired iPad silently stops receiving data.")


def test_is_json_serializable_and_tagged():
    payload = _payload()
    assert payload["type"] == "telemetry"
    assert payload["v"] == 1
    # No non-JSON leakage (enums, numpy, bytes) from the snapshot.
    json.dumps(payload)


def test_every_vital_tile_is_present_in_order():
    vitals = _payload()["vitals"]
    assert [v["id"] for v in vitals] == [spec[0] for spec in _IPAD_VITALS]
    by_id = {v["id"]: v for v in vitals}
    # Canonical keys resolve to a present tile with the emitted value.
    assert by_id["hr"]["present"] and by_id["hr"]["value"] == "72"
    assert by_id["resp"]["present"] and by_id["resp"]["value"] == "15"
    assert by_id["spo2"]["present"] and by_id["spo2"]["value"] == "97"
    # Emotion has no canonical key; the best backend label wins.
    assert by_id["mood"]["present"] and by_id["mood"]["value"] == "Content"


def test_absent_vital_renders_as_measuring_not_dropped():
    # No spo2/emotion in the snapshot -> tiles stay, marked not present, so the
    # layout never reflows mid-demo.
    now = 1_000_000.0
    snap = [Result("heart_rate", "bpm", 80.0, 0.9, Severity.INFO, "HR ~80 bpm",
                   ttl=8.0, timestamp=now)]
    registry.discover()
    vitals = {v["id"]: v for v in ipad_payload(snap)["vitals"]}
    assert vitals["hr"]["present"] is True
    assert vitals["spo2"]["present"] is False
    assert vitals["spo2"]["value"] is None
    # Even absent, the tile keeps its honest reliability rating for the badge.
    assert vitals["spo2"]["reliability"]["tier"] == "LOW"


def test_signal_feed_is_capped():
    signals = _payload()["signals"]
    assert len(signals) <= _IPAD_MAX_SIGNALS


def test_module_roster_matches_registry():
    payload = _payload()
    assert {m["module"] for m in payload["modules"]} == set(registry.all_registered())
    # Fields the settings drawer's toggle grid needs.
    for m in payload["modules"]:
        assert set(m["enabled"]) == {"primary", "secondary"}
        assert set(m["toggleable"]) == {"primary", "secondary"}
