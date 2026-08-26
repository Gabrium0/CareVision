"""Tests for privacy-safe localhost rPPG calibration summaries."""
import pytest

from live_rppg_calibration import (_parse_checkpoint, _reference_at, _sample,
                                   _summaries, _localhost_url)


def test_checkpoint_interpolation():
    points = [_parse_checkpoint("0:60"), _parse_checkpoint("30:66")]
    assert _reference_at(points, 15.0) == 63.0


def test_samples_only_summary_fields_and_reports_metrics():
    payload = {"system": {"vitals": {
        "fast_path": {"mode": "tracked"},
        "backends": [{"name": "classical", "effective_sample_hz": 25.0,
                      "measurement": {"bpm": 62.0, "accepted": True,
                                      "confidence": .8, "quality": .9}}]}}}
    rows = _sample(payload, 5.0, [(0.0, 61.0), (10.0, 61.0)])
    assert rows[0]["absolute_error_bpm"] == 1.0
    assert "frame" not in rows[0] and "landmarks" not in rows[0]
    summary = _summaries(rows)[0]
    assert summary["mae_bpm"] == 1.0
    assert summary["coverage_pct"] == 100.0


def test_calibration_rejects_non_loopback_debug_url():
    assert _localhost_url("http://127.0.0.1:8771/debug/state")
    with pytest.raises(Exception):
        _localhost_url("https://example.com/debug/state")
