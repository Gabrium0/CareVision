"""Focused contracts for bounded Groq speech-to-text over the shared audio bus."""
from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audio.stt_groq as groq_stt
from audio.bus import AudioBus
from audio.stt_groq import GroqListener, _wav_buffer


class FakeHistory:
    def __init__(self):
        self.added = []

    def rolling_mean(self, *_args):
        return 100.0

    def add(self, *args):
        self.added.append(args)


def _disabled_listener() -> GroqListener:
    listener = GroqListener(enabled=False, audio_bus=AudioBus())
    listener._history = FakeHistory()
    return listener


def test_wav_upload_is_16khz_mono_pcm_without_disk_io():
    upload = _wav_buffer(np.linspace(-1.0, 1.0, 1600, dtype=np.float32))
    assert upload.name == "speech.wav"
    with wave.open(upload, "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 1600


def test_dict_response_filters_hallucinations_and_publishes_safe_summary():
    class Transcriptions:
        def create(self, **kwargs):
            assert kwargs["model"] == "whisper-large-v3-turbo"
            assert kwargs["response_format"] == "verbose_json"
            assert kwargs["file"].read(4) == b"RIFF"
            return {
                "text": "thanks for watching hello there",
                "segments": [
                    {"text": "thanks for watching", "avg_logprob": -0.1,
                     "no_speech_prob": 0.0},
                    {"text": "hello there", "avg_logprob": -0.1,
                     "no_speech_prob": 0.0,
                     "words": [{"start": 0.0, "end": 0.2},
                               {"start": 0.8, "end": 1.0}]},
                ],
            }

    listener = _disabled_listener()
    listener._client = types.SimpleNamespace(
        audio=types.SimpleNamespace(transcriptions=Transcriptions()))
    listener._last_agent_speech = 9.0
    listener._last_user_end = 8.0

    listener._transcribe_segment(
        np.zeros(16000, dtype=np.float32), 10.0, 11.0, 2)

    assert listener.pop_utterances() == [("hello there", 10.0)]
    metrics = listener.pop_metrics()
    assert metrics == [{
        "timestamp": 10.0, "duration": 1.0, "word_count": 2,
        "words_per_minute": 120.0, "pauses": 1,
        "pause_frequency": 1.0, "response_latency": 1.0,
        "turn_gap": 2.0, "interruptions": 2,
        "baseline_change": 0.2, "quality": 0.5,
    }]
    assert listener._history.added == [
        ("speech_timing", "words_per_minute", 120.0, 10.0),
        ("speech_timing", "pause_frequency", 1.0, 10.0),
    ]


def test_segment_queue_is_bounded_and_keeps_newest_utterances():
    listener = _disabled_listener()
    for marker in (1, 2, 3):
        listener._enqueue_segment((np.array([marker]), marker, marker, 0))
    assert listener._segments.qsize() == 2
    assert listener._segments_dropped == 1
    assert listener._segments.get_nowait()[1] == 2
    assert listener._segments.get_nowait()[1] == 3


def test_missing_key_disables_only_groq_listener(monkeypatch, capsys):
    monkeypatch.setattr(groq_stt, "_dependency_available", lambda: True)
    monkeypatch.setattr(groq_stt, "groq_api_key", lambda: None)
    listener = GroqListener(audio_bus=AudioBus())
    assert listener.available is False
    assert "GROQ_API_KEY not set" in capsys.readouterr().out
    listener.close()


def test_client_lifecycle_is_bounded_and_diagnostics_hide_credentials(monkeypatch):
    created = {}

    class FakeGroq:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.audio = types.SimpleNamespace(
                transcriptions=types.SimpleNamespace(create=lambda **_kwargs: None))

    fake_module = types.ModuleType("groq")
    fake_module.Groq = FakeGroq
    monkeypatch.setitem(sys.modules, "groq", fake_module)
    monkeypatch.setattr(groq_stt, "_dependency_available", lambda: True)
    monkeypatch.setattr(groq_stt, "groq_api_key", lambda: "secret-test-key")

    listener = GroqListener(audio_bus=AudioBus())
    try:
        assert listener.available is True
        assert created == {"api_key": "secret-test-key", "timeout": 15.0,
                           "max_retries": 1}
        diagnostics = listener.diagnostics()
        assert diagnostics["worker_alive"] is True
        assert "secret-test-key" not in str(diagnostics)
    finally:
        listener.close()
    assert listener.diagnostics()["worker_alive"] is False


def test_object_response_attributes_extracted_correctly():
    class SegmentObj:
        def __init__(self, text, avg_logprob, no_speech_prob, words=None):
            self.text = text
            self.avg_logprob = avg_logprob
            self.no_speech_prob = no_speech_prob
            self.words = words or []

    class WordObj:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class ResponseObj:
        def __init__(self):
            self.text = "how are you"
            self.segments = [
                SegmentObj("how are you", -0.2, 0.05, [
                    WordObj(0.0, 0.2), WordObj(0.3, 0.5), WordObj(0.6, 0.8)
                ])
            ]
            self.words = []

    class Transcriptions:
        def create(self, **_kwargs):
            return ResponseObj()

    listener = _disabled_listener()
    listener._client = types.SimpleNamespace(
        audio=types.SimpleNamespace(transcriptions=Transcriptions()))

    listener._transcribe_segment(
        np.zeros(16000, dtype=np.float32), 20.0, 21.0, 0)

    assert listener.pop_utterances() == [("how are you", 20.0)]
    metrics = listener.pop_metrics()
    assert len(metrics) == 1
    assert metrics[0]["word_count"] == 3
    assert metrics[0]["timestamp"] == 20.0


def test_transcription_error_sets_last_error_and_preserves_listener():
    class FailingTranscriptions:
        def create(self, **_kwargs):
            raise RuntimeError("API rate limit exceeded (429)")

    listener = _disabled_listener()
    listener._client = types.SimpleNamespace(
        audio=types.SimpleNamespace(transcriptions=FailingTranscriptions()))

    listener._transcribe_segment(
        np.zeros(16000, dtype=np.float32), 30.0, 31.0, 0)

    assert listener.pop_utterances() == []
    diag = listener.diagnostics()
    assert "API rate limit exceeded" in diag["last_error"]


def test_mark_agent_spoke_mutes_capture_window():
    listener = _disabled_listener()
    assert listener._agent_is_speaking() is False

    listener.mark_agent_spoke(timestamp=100.0, estimated_seconds=4.0)
    assert listener._last_agent_speech == 100.0
    assert listener._agent_speech_until == 104.0
