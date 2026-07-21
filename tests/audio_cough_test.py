"""Unit contracts for shared microphone capture and cough episodes."""
from __future__ import annotations

import threading
import time
import json
import builtins
import queue
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from audio.bus import AudioBus
from audio.intelligence import SoundEventDetector
from audio.microphone import MicrophoneProducer
import audio.stt as stt_module
from audio.stt import Listener, _whisper_worker
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


class _FakeProcess:
    """Small multiprocessing.Process stand-in for listener lifecycle tests."""

    def __init__(self, alive=True, exitcode=None):
        self.alive = alive
        self.exitcode = exitcode
        self.join_calls = []
        self.terminated = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def terminate(self):
        self.terminated = True
        self.alive = False


def _fake_stt_worker(listener):
    listener._segments = queue.Queue()
    listener._worker_out = queue.Queue()
    listener._worker = _FakeProcess()
    return True


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
    monkeypatch.setattr(stt_module, "_dependency_available", lambda: True)
    monkeypatch.setattr(Listener, "_start_worker", _fake_stt_worker)
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


def test_listener_dependency_probe_does_not_import_native_asr(monkeypatch):
    """The parent checks package presence without loading CTranslate2 DLLs."""
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "faster_whisper" or name.startswith("ctranslate2"):
            imported.append(name)
            raise AssertionError(f"native ASR imported in parent: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(stt_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(Listener, "_start_worker", _fake_stt_worker)

    listener = Listener(audio_bus=AudioBus())
    listener.close()

    assert imported == []


def test_whisper_worker_emits_bounded_ready_and_result_messages(monkeypatch):
    """The child returns transcript summaries, never the raw audio array."""
    words = [types.SimpleNamespace(start=0.0, end=0.2),
             types.SimpleNamespace(start=0.9, end=1.1)]
    segments = [types.SimpleNamespace(text=" hello ", words=words)]

    class FakeWhisperModel:
        def __init__(self, model_size, device, compute_type):
            assert (model_size, device, compute_type) == ("base", "cpu", "int8")

        def transcribe(self, audio, **kwargs):
            assert kwargs["word_timestamps"] is True
            assert len(audio) == 16000
            return iter(segments), object()

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    in_q, out_q = queue.Queue(), queue.Queue()
    in_q.put((np.zeros(16000, dtype=np.float32), 10.0, 11.0, 2))
    in_q.put(None)

    _whisper_worker(in_q, out_q, "base", "en")

    ready, result = out_q.get_nowait(), out_q.get_nowait()
    assert {key: ready[key] for key in ("event", "model", "device")} == {
        "event": "ready", "model": "base", "device": "cpu"}
    assert ready["pid"] > 0
    assert ready["native_runtime"]["torch_import_blocked"] is True
    assert ready["native_runtime"]["torch_loaded"] is False
    assert result == {"event": "result", "text": "hello", "timestamp": 10.0,
                      "ended_at": 11.0, "interruptions": 2, "duration": 1.0,
                      "word_count": 2, "pauses": 1}
    assert "audio" not in result


def test_whisper_worker_reports_load_and_transcription_errors(monkeypatch):
    """Worker failures are sanitized into protocol messages instead of escaping."""
    original_meta_path = tuple(sys.meta_path)
    class LoadFailure:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("broken native runtime")

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = LoadFailure
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    out_q = queue.Queue()
    _whisper_worker(queue.Queue(), out_q, "base", "en")
    assert tuple(sys.meta_path) == original_meta_path
    message = out_q.get_nowait()
    assert message["event"] == "error" and message["phase"] == "load"
    assert "broken native runtime" in message["error"]

    class TranscriptionFailure:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, *_args, **_kwargs):
            raise ValueError("bad segment")

    fake_module.WhisperModel = TranscriptionFailure
    in_q, out_q = queue.Queue(), queue.Queue()
    in_q.put((np.zeros(10, dtype=np.float32), 1.0, 2.0, 0))
    in_q.put(None)
    _whisper_worker(in_q, out_q, "base", "en")
    assert tuple(sys.meta_path) == original_meta_path
    assert out_q.get_nowait()["event"] == "ready"
    message = out_q.get_nowait()
    assert message["event"] == "error" and message["phase"] == "transcribe"
    assert "bad segment" in message["error"]


def test_listener_worker_failure_is_nonfatal_and_close_is_bounded(capsys):
    """A dead native worker disables listening; repeated close remains safe."""
    listener = Listener(enabled=False, audio_bus=AudioBus())
    listener.available = True
    listener._worker_out = queue.Queue()
    listener._worker = _FakeProcess(alive=False, exitcode=127)

    assert listener.pop_utterances() == []
    assert listener.available is False
    assert "exited unexpectedly" in capsys.readouterr().out

    listener._segments = queue.Queue()
    listener._worker_out = queue.Queue()
    worker = _FakeProcess(alive=True)
    listener._worker = worker
    listener.close()
    listener.close()

    assert worker.terminated is True
    assert worker.join_calls == [2.0, 0.5]


def test_listener_worker_start_failure_disables_only_listening(capsys):
    """Process creation errors stay inside the optional listener boundary."""
    class BrokenContext:
        def Queue(self):
            return queue.Queue()

        def Process(self, **_kwargs):
            class BrokenProcess:
                def start(self):
                    raise OSError("spawn denied")
            return BrokenProcess()

    listener = Listener(enabled=False, audio_bus=AudioBus())
    listener._ctx = BrokenContext()

    assert listener._start_worker() is False
    assert listener._worker is None
    assert listener._segments is None and listener._worker_out is None
    assert "spawn denied" in capsys.readouterr().out
    listener.close()


def test_listener_publishes_worker_summary_with_parent_history_metrics():
    """Transcript timing and history calculations remain in the parent process."""
    class FakeHistory:
        def __init__(self):
            self.added = []

        def mean_since(self, *_args):
            return 100.0

        def add(self, *args):
            self.added.append(args)

    listener = Listener(enabled=False, audio_bus=AudioBus())
    listener._history = FakeHistory()
    listener._last_agent_speech = 9.0
    listener._last_user_end = 8.0
    listener._publish_result({"text": "hello there", "timestamp": 10.0,
                              "ended_at": 12.0, "duration": 2.0,
                              "word_count": 4, "pauses": 1,
                              "interruptions": 2})

    assert listener.pop_utterances() == [("hello there", 10.0)]
    metrics = listener.pop_metrics()
    assert metrics == [{"timestamp": 10.0, "duration": 2.0, "word_count": 4,
                        "words_per_minute": 120.0, "pauses": 1,
                        "pause_frequency": 0.5, "response_latency": 1.0,
                        "turn_gap": 2.0, "interruptions": 2,
                        "baseline_change": 0.2, "quality": 1.0}]
    assert listener._history.added == [
        ("speech_timing", "words_per_minute", 120.0, 10.0),
        ("speech_timing", "pause_frequency", 0.5, 10.0),
    ]
    listener.close()


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
