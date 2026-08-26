"""Checks for trusted canonical heart-rate output vs debug backend values.

Run standalone:  python tests/heart_rate_canonical_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext
from modules.heart_rate import HeartRate


class FakeBackend:
    """Minimal backend stub for HeartRate.process()."""

    available = True

    def __init__(self, label: str, reading: dict | None):
        self.label = label
        self.reading = reading

    def update(self, ctx):
        pass

    def compute(self):
        return self.reading

    def close(self):
        pass


def _ctx() -> FrameContext:
    return FrameContext(frame=np.zeros((8, 8, 3), dtype=np.uint8), timestamp=0.0,
                        frame_index=0, fps=30.0)


def _hr(debug: bool, backends: list[FakeBackend]) -> HeartRate:
    hr = HeartRate(backends=["classical"], debug_backend_values=debug)
    hr._backends = backends
    return hr


def test_normal_mode_emits_only_canonical_bpm():
    hr = _hr(False, [
        FakeBackend("classical", {"bpm": 54.0, "confidence": 0.08,
                                  "quality": 0.42}),
        FakeBackend("open-rppg", {
            "raw_bpm": 136.0,
            "raw_confidence": 0.16,
            "rejected_reason": "low SQI 0.16<0.35",
            "status": "low SQI 0.16<0.35",
        }),
    ])
    out = hr.process(_ctx()) or []
    by_key = {r.key: r for r in out}
    assert by_key["bpm"].value == 54.0
    assert by_key["bpm"].confidence == 0.08
    assert by_key["bpm"].quality == 0.42
    assert "bpm_classical" not in by_key
    assert "bpm_open_rppg" not in by_key
    print("[heart-rate-canonical-test] normal mode canonical-only OK")


def test_debug_mode_keeps_rejected_backend_value_visible():
    hr = _hr(True, [
        FakeBackend("classical", {"bpm": 54.0, "confidence": 0.08}),
        FakeBackend("open-rppg", {
            "raw_bpm": 136.0,
            "raw_confidence": 0.16,
            "rejected_reason": "low SQI 0.16<0.35",
            "status": "low SQI 0.16<0.35",
        }),
    ])
    out = hr.process(_ctx()) or []
    by_key = {r.key: r for r in out}
    assert by_key["bpm"].value == 54.0
    assert by_key["bpm_classical"].value == 54.0
    assert by_key["bpm_open_rppg"].value == 136.0
    assert by_key["bpm_open_rppg"].confidence <= 0.2
    assert "low SQI" in by_key["bpm_open_rppg"].message
    print("[heart-rate-canonical-test] debug rejected backend visible OK")


def test_low_confidence_jump_is_not_canonical():
    hr = _hr(False, [FakeBackend("classical", {"bpm": 54.0, "confidence": 0.5})])
    first = hr.process(_ctx()) or []
    assert {r.key: r for r in first}["bpm"].value == 54.0
    hr._backends = [FakeBackend("classical", {"bpm": 84.0, "confidence": 0.2})]
    second = hr.process(_ctx()) or []
    assert not any(r.key == "bpm" for r in second), second
    print("[heart-rate-canonical-test] low-confidence jump rejected OK")


def test_low_quality_candidate_is_debug_only():
    hr = _hr(True, [FakeBackend("classical", {
        "bpm": 61.0, "confidence": 0.9, "quality": 0.2})])
    out = hr.process(_ctx()) or []
    by_key = {result.key: result for result in out}
    assert "bpm" not in by_key
    assert by_key["bpm_classical"].value == 61.0
    assert by_key["bpm_classical"].quality == 0.2


def main():
    test_normal_mode_emits_only_canonical_bpm()
    test_debug_mode_keeps_rejected_backend_value_visible()
    test_low_confidence_jump_is_not_canonical()
    test_low_quality_candidate_is_debug_only()
    print("[heart-rate-canonical-test] OK")


if __name__ == "__main__":
    main()
