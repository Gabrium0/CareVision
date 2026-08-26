"""Unit coverage for wearable-reference alignment in the rPPG benchmark."""
from __future__ import annotations

from types import SimpleNamespace

from benchmark_rppg_models import (_load_reference, _reference_at,
                                   _ubfc_reference, _pure_reference,
                                   _dataset_items)


def test_timestamped_reference_is_interpolated(tmp_path):
    path = tmp_path / "wearable.csv"
    path.write_text("seconds,bpm\n0,60\n10,80\n", encoding="utf-8")
    args = SimpleNamespace(reference_bpm=None, reference_csv=str(path))
    points = _load_reference(args)
    assert _reference_at(points, 5.0) == 70.0


def test_bpm_only_reference_uses_mean(tmp_path):
    path = tmp_path / "wearable.csv"
    path.write_text("bpm\n60\n80\n", encoding="utf-8")
    args = SimpleNamespace(reference_bpm=None, reference_csv=str(path))
    assert _reference_at(_load_reference(args), 500.0) == 70.0


def test_ubfc_adapter_reads_official_three_row_layout(tmp_path):
    recording = tmp_path / "subject1"
    recording.mkdir()
    (recording / "vid.avi").write_bytes(b"placeholder")
    truth = recording / "ground_truth.txt"
    truth.write_text("0.1 0.2 0.3\n60 61 62\n5 6 7\n", encoding="utf-8")
    assert _ubfc_reference(truth) == [(0.0, 60.0), (1.0, 61.0), (2.0, 62.0)]
    items = _dataset_items("ubfc", str(tmp_path))
    assert items[0]["recording"] == "subject1"


def test_pure_adapter_reads_timestamped_pulse_rate(tmp_path):
    metadata = tmp_path / "01-01.json"
    metadata.write_text(
        '{"/FullPackage": ['
        '{"Timestamp": 1000000000, "Value": {"pulseRate": 60}},'
        '{"Timestamp": 2000000000, "Value": {"pulseRate": 62}}]}',
        encoding="utf-8")
    assert _pure_reference(metadata) == [(0.0, 60.0), (1.0, 62.0)]
