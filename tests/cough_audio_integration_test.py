"""Real-audio integration benchmark for the production YAMNet detector."""
from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from audio.bus import AudioBus
from audio.intelligence import SoundEventDetector


FIXTURES = Path(__file__).parent / "fixtures" / "cough"


def _yamnet():
    try:
        import tensorflow_hub as hub
    except ImportError:
        pytest.skip("YAMNet unavailable; install requirements-audio-events.txt")
    try:
        model = hub.load("https://tfhub.dev/google/yamnet/1")
    except Exception as exc:  # noqa: BLE001 - optional download/cache boundary
        pytest.skip(f"YAMNet model unavailable: {type(exc).__name__}")
    class_map = model.class_map_path().numpy().decode()
    with open(class_map, encoding="utf-8") as source:
        labels = [row[2] for row in csv.reader(source)][1:]
    return model, labels


def _samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        assert source.getframerate() == 16000
        assert source.getnchannels() == 1 and source.getsampwidth() == 2
        raw = source.readframes(source.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


@pytest.mark.integration
def test_coswara_cough_detection_benchmark():
    """Detect licensed cough controls while rejecting respiratory/speech controls."""
    model, labels = _yamnet()
    manifest = json.loads((FIXTURES / "ATTRIBUTION.json").read_text("utf-8"))
    rows = []
    for fixture in manifest["fixtures"]:
        detector = SoundEventDetector(
            AudioBus(), model=model, labels=labels, allowed_events={"cough"},
            start_worker=False)
        samples = _samples(FIXTURES / fixture["file"])
        timestamp = 0.0
        for start in range(0, len(samples), 1600):
            detector.feed(samples[start:start + 1600], timestamp)
            timestamp += 0.1
        for _ in range(40):
            detector.feed(np.zeros(1600, dtype=np.float32), timestamp)
            timestamp += 0.1
        detector.flush(timestamp)
        results = [row for row in detector.pop_results()
                   if row.key == "cough_episode"]
        rows.append({"category": fixture["category"],
                     "expected": fixture["expected"],
                     "detected": bool(results),
                     "episodes": len(results),
                     "max_confidence": detector.diagnostics()["max_scores"].get(
                         "cough", 0.0)})
        detector.close()
    positives = [row for row in rows if row["expected"] == "cough"]
    negatives = [row for row in rows if row["expected"] == "non_cough"]
    hits = sum(row["detected"] for row in positives)
    false_positives = sum(row["detected"] for row in negatives)
    for row in rows:
        print("\n{category:18} detected={detected!s:5} episodes={episodes} "
              "max_cough={max_confidence:.3f}".format(**row))
    print(f"\nrecall={hits}/{len(positives)}; "
          f"false_positive_rate={false_positives}/{len(negatives)}")
    assert hits >= 9
    assert false_positives <= 1
