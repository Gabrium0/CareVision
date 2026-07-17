"""Unit contracts for shared microphone capture and cough episodes."""
from __future__ import annotations

import threading
import time
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from audio.bus import AudioBus
from audio.intelligence import SoundEventDetector
from audio.microphone import MicrophoneProducer
from audio.stt import Listener
from core.events import PersistencePolicy
from core.context import FrameContext
from modules.replay_events import ReplayEvents
from storage.event_store import EventStore


class _SequenceModel:
    """Return deterministic Cough/Speech scores for consecutive windows."""

    def __init__(self, cough_scores):
        self._scores = iter(cough_scores)

    def __call__(self, _audio):
        cough = next(self._scores, 0.0)
        scores = np.array([[cough, 1.0 - cough]], dtype=np.float32)
        return scores, np.empty((0,)), np.empty((0,))


def _detector(scores, **kwargs):
    return SoundEventDetector(
        AudioBus(), model=_SequenceModel(scores), labels=["Cough", "Speech"],
        allowed_events={"cough"}, start_worker=False, **kwargs)


def test_overlapping_windows_emit_one_private_safe_episode():
    """Two positive windows confirm a cough and three quiet seconds close it."""
    detector = _detector([.8, .9, 0, 0, 0, 0, 0, 0])
    for index in range(9):
        detector.feed(np.zeros(8000, dtype=np.float32), index * .5)
    rows = detector.pop_results()
    assert len(rows) == 1
    result = rows[0]
    assert result.module == "sound_event" and result.key == "cough_episode"
    assert result.value["count"] == 1
    assert result.evidence_window == (result.value["started_at"],
                                      result.value["ended_at"])
    assert result.persistence == PersistencePolicy.EVENT
    assert not hasattr(result.value, "dtype")
    detector.close()


def test_isolated_positive_is_ignored_and_nearby_bursts_are_counted():
    """A singleton is rejected while two confirmed bursts merge into one episode."""
    detector = _detector([], merge_seconds=2, quiet_seconds=3)
    detector._observe_cough(.8, 1.0)
    detector._observe_cough(0, 1.5)
    assert detector.pop_results() == []
    for confidence, timestamp in ((.8, 2), (.9, 2.5), (0, 3),
                                  (.85, 3.5), (.9, 4), (0, 5),
                                  (0, 6), (0, 7)):
        detector._observe_cough(confidence, timestamp)
    rows = detector.pop_results()
    assert len(rows) == 1 and rows[0].value["count"] == 2
    detector.close()


def test_strong_cough_window_still_requires_adjacent_support():
    """High confidence uses hysteresis but never confirms without a second window."""
    detector = _detector([])
    detector._observe_cough(.95, 1)
    detector._observe_cough(0, 1.1)
    assert detector.pop_results() == []
    detector._observe_cough(.2, 2)
    detector._observe_cough(.9, 2.1)
    detector.flush()
    assert detector.pop_results()[0].value["count"] == 1
    detector.close()


def test_cough_only_mode_does_not_emit_other_sound_classes():
    """Restricting events to cough prevents speech or other taxonomy output."""
    detector = _detector([0, 0, 0])
    for index in range(4):
        detector.feed(np.zeros(8000, dtype=np.float32), index * .5)
    assert detector.pop_results() == []
    detector.close()


def test_diagnostics_report_live_cough_state_without_raw_audio():
    """Operational telemetry updates while samples and model tensors stay private."""
    detector = _detector([.8, .9])
    detector.feed(np.zeros(16000, dtype=np.float32), 10.0)
    detector.feed(np.zeros(1600, dtype=np.float32), 11.0)
    diagnostics = detector.diagnostics()
    assert diagnostics["status"] == "ready"
    assert diagnostics["mode"] == "cough_only"
    assert diagnostics["windows_processed"] == 2
    assert diagnostics["latest_cough_confidence"] == pytest.approx(.9)
    assert diagnostics["peak_cough_confidence"] == pytest.approx(.9)
    assert diagnostics["max_scores"]["cough"] == pytest.approx(.9)
    assert diagnostics["last_inference_at"] is not None
    assert diagnostics["pending_cough_episode"] is True
    assert diagnostics["pending_burst_count"] == 1
    assert not any(isinstance(value, np.ndarray) for value in diagnostics.values())
    assert not {"audio", "samples", "embeddings", "spectrogram"} & diagnostics.keys()
    detector.close()


class _FakeStream:
    def __init__(self, **kwargs):
        self.blocksize = kwargs["blocksize"]
        self.started = False
        self.closed = threading.Event()

    def start(self):
        self.started = True

    def read(self, _size):
        if self.closed.wait(.01):
            raise RuntimeError("closed")
        return np.ones((self.blocksize, 1), dtype=np.float32), False

    def stop(self):
        self.closed.set()

    def close(self):
        self.closed.set()


def test_one_microphone_stream_fans_out_to_audio_consumers():
    """A single physical stream publishes copied blocks to STT and sound events."""
    bus = AudioBus()
    stt = bus.subscribe("speech-to-text")
    yamnet = bus.subscribe("sound-events")
    streams = []

    def factory(**kwargs):
        stream = _FakeStream(**kwargs)
        streams.append(stream)
        return stream

    producer = MicrophoneProducer(bus, stream_factory=factory)
    one = stt.get(timeout=1)[0]
    two = yamnet.get(timeout=1)[0]
    producer.close()
    assert len(streams) == 1 and streams[0].started
    assert len(one) == len(two) == 1600
    assert one is not two


def test_microphone_failure_is_nonfatal():
    """A missing or failed input device disables capture without raising."""
    def failed(**_kwargs):
        raise OSError("no device")

    producer = MicrophoneProducer(AudioBus(), stream_factory=failed)
    assert producer.available is False
    producer.close()


def test_whisper_listener_segments_shared_bus_without_opening_mic(monkeypatch):
    """STT consumes the producer bus and has no sounddevice dependency of its own."""
    monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
    monkeypatch.setattr(Listener, "_transcribe_loop", lambda self: None)
    bus = AudioBus()
    listener = Listener(audio_bus=bus, min_voiced_seconds=.3, silence_seconds=.8)
    for index in range(4):
        bus.publish(np.full(1600, .1, dtype=np.float32), index * .1)
    for index in range(4, 14):
        bus.publish(np.zeros(1600, dtype=np.float32), index * .1)
    deadline = time.time() + 1
    while listener._segments.empty() and time.time() < deadline:
        time.sleep(.01)
    segment = listener._segments.get_nowait()[0]
    listener.close()
    assert len(segment) >= 4 * 1600


def test_replay_cough_episode_keeps_counted_payload():
    """The scripted cough showcase enters the ordinary pipeline as an episode."""
    scenarios = json.loads((Path(__file__).parents[1] / "config" /
                            "replay_scenarios.json").read_text("utf-8"))
    scripted = next(event for event in scenarios["cough_followup"]["events"]
                    if event.get("key") == "cough_episode")
    ctx = FrameContext(np.zeros((2, 2, 3), dtype=np.uint8), 1, 0, 10)
    ctx.extras["replay_events"] = [scripted]
    result = ReplayEvents().process(ctx)[0]
    assert result.key == "cough_episode"
    assert result.value == {"count": 2, "started_at": .1, "ended_at": 1.2}


def test_cough_episode_persists_summary_without_audio(tmp_path):
    """Persistence accepts the counted summary and stores no sample arrays."""
    detector = _detector([])
    detector._observe_cough(.8, 1)
    detector._observe_cough(.9, 1.1)
    detector.flush()
    result = detector.pop_results()[0]
    result.correlation_id = "cough-test"
    store = EventStore(tmp_path / "events.sqlite3")
    store.record_result(result)
    store.flush()
    payload = store.chain("cough-test")[0]["payload"]
    assert payload["value"]["count"] == 1
    assert "audio" not in json.dumps(payload).lower()
    store.close()
    detector.close()
