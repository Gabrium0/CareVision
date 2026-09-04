"""Groq cloud speech-to-text — the agent's ears via Groq Whisper API.

`GroqListener` mirrors `audio/stt.Listener`'s interface so the main loop
can swap implementations without changes. It uses local energy VAD
to avoid sending silence, then uploads completed segments to Groq's
Whisper API (whisper-large-v3-turbo on free tier).

Turn-taking: the robot must not transcribe its own voice. Any segment that
overlapped agent speech (`Speaker.speaking`, plus a short tail for room
echo) is dropped at capture time — same logic as the local listener.

Optional dependency: requires `groq` package and `GROQ_API_KEY` in .env.
When missing, the listener self-disables with a hint (matching
`audio/stt.Listener` pattern).
"""
from __future__ import annotations

import io
import queue
import re
import threading
import time
import wave
from typing import Optional

import numpy as np
from agent.env import groq_api_key
from storage.history_store import HistoryStore

_SAMPLE_RATE = 16000
_BLOCK_SECONDS = 0.1

_MAX_AGENT_MUTE_SECONDS = 10.0

_NO_SPEECH_MAX = 0.6
_AVG_LOGPROB_MIN = -1.0
_HALLUCINATION_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
    "amara org",
)


def _normalize_transcript(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace for phrase matching."""
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def _segment_is_trustworthy(text: str, avg_logprob: float | None,
                            no_speech_prob: float | None) -> bool:
    """Whether a transcribed segment is real speech, not a whisper hallucination."""
    if no_speech_prob is not None and float(no_speech_prob) >= _NO_SPEECH_MAX:
        return False
    if avg_logprob is not None and float(avg_logprob) <= _AVG_LOGPROB_MIN:
        return False
    normalized = _normalize_transcript(text)
    if any(phrase in normalized for phrase in _HALLUCINATION_PHRASES):
        return False
    return True


def _dependency_available() -> bool:
    """Check for groq package without importing heavy deps."""
    try:
        import groq  # noqa: F401
        return True
    except (ImportError, ValueError):
        return False


def _error_detail(exc: BaseException) -> str:
    """Return a bounded worker error suitable for terminal diagnostics."""
    message = str(exc).strip()
    detail = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return detail[:240]


def _field(value, name: str, default=None):
    """Read SDK response objects and plain JSON dictionaries uniformly."""
    return value.get(name, default) if isinstance(value, dict) \
        else getattr(value, name, default)


def _wav_buffer(audio: np.ndarray) -> io.BytesIO:
    """Encode float mono samples as a named in-memory WAV upload."""
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
           * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    buffer.seek(0)
    buffer.name = "speech.wav"
    return buffer


class GroqListener:
    """Microphone -> local energy VAD -> Groq Whisper API -> text."""

    def __init__(self, enabled: bool = True, model: str = "whisper-large-v3-turbo",
                 energy_threshold: float = 0.02, silence_seconds: float = 0.5,
                 min_voiced_seconds: float = 0.3, max_segment_seconds: float = 12.0,
                 speech_tail_seconds: float = 0.5, language: str = "en",
                 speaker=None, audio_bus=None):
        self.available = False
        self.energy_threshold = energy_threshold
        self.silence_seconds = silence_seconds
        self.min_voiced_seconds = min_voiced_seconds
        self.max_segment_seconds = max_segment_seconds
        self.speech_tail_seconds = speech_tail_seconds
        self.language = language
        self.model = model
        self.speaker = speaker
        if audio_bus is None:
            from audio.bus import AudioBus
            audio_bus = AudioBus()
        self.audio_bus = audio_bus
        self._out: queue.Queue[tuple] = queue.Queue()
        self._metrics: queue.Queue[dict] = queue.Queue()
        self._segments: queue.Queue[tuple | None] = queue.Queue(maxsize=2)
        self._segments_dropped = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_agent_speech = -1e9
        self._agent_speech_until = -1e9
        self._last_user_end = -1e9
        self._interruption_count = 0
        self._history = HistoryStore.instance()
        if hasattr(self._history, "rolling_mean"):
            self._history.rolling_mean(
                "speech_timing", "words_per_minute", 30 * 86400)
        self._closed = False
        self._client = None
        self._last_error: Optional[str] = None

        if not enabled:
            return
        if not _dependency_available():
            print("[stt-groq] listener unavailable (groq package not installed); "
                  "agent is speak-only. Install with: pip install -r requirements-asr.txt")
            return
        api_key = groq_api_key()
        if not api_key:
            print("[stt-groq] listener unavailable (GROQ_API_KEY not set in .env); "
                  "agent is speak-only.")
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key, timeout=15.0, max_retries=1)
        except Exception as exc:
            print(f"[stt-groq] failed to initialize client ({_error_detail(exc)}); "
                  "agent is speak-only")
            return

        self.available = True
        self._audio_in = self.audio_bus.subscribe("speech-to-text")
        for target, name in ((self._transcription_loop, "stt-groq-api"),
                             (self._segment_loop, "stt-groq-segment")):
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()
            self._threads.append(thread)
        print(f"[stt-groq] listener ready (model={model}, language={language})")

    def _agent_is_speaking(self) -> bool:
        """True while the agent talks (or just finished — room-echo tail)."""
        now = time.time()
        if self.speaker is not None and getattr(self.speaker, "speaking", False):
            self._last_agent_speech = now
            self._agent_speech_until = max(
                self._agent_speech_until, now + self.speech_tail_seconds)
            return True
        if now < self._agent_speech_until:
            return True
        return now - self._last_agent_speech < self.speech_tail_seconds

    def _segment_loop(self) -> None:
        """Consume shared microphone blocks and close speech segments by energy."""
        buf: list[np.ndarray] = []
        voiced_time = 0.0
        last_voiced = None
        segment_started = None
        tainted = False
        _log_started = False

        try:
            while not self._stop.is_set():
                try:
                    audio, now = self._audio_in.get(timeout=0.5)
                except queue.Empty:
                    continue
                if not _log_started:
                    print(f"[stt-groq] shared audio attached (energy VAD, "
                          f"threshold={self.energy_threshold}, model={self.model})")
                    _log_started = True
                block_seconds = len(audio) / _SAMPLE_RATE
                if block_seconds <= 0:
                    continue
                audio = np.asarray(audio, dtype=np.float32).ravel()
                if self._agent_is_speaking():
                    tainted = bool(buf) or tainted
                    if not buf:
                        continue
                rms = float(np.sqrt(np.mean(audio ** 2)))
                voiced = rms >= self.energy_threshold
                if voiced:
                    if not buf:
                        segment_started = now
                    buf.append(audio.copy())
                    voiced_time += block_seconds
                    last_voiced = now
                elif buf:
                    buf.append(audio.copy())
                if not buf:
                    continue
                seg_len = sum(len(part) for part in buf) / _SAMPLE_RATE
                silence = (now - last_voiced) if last_voiced else 0.0
                if silence >= self.silence_seconds or seg_len >= self.max_segment_seconds:
                    if voiced_time >= self.min_voiced_seconds and not tainted:
                        self._enqueue_segment((
                            np.concatenate(buf).ravel(), segment_started or now,
                            last_voiced or now, self._interruption_count))
                        self._interruption_count = 0
                    elif tainted:
                        self._interruption_count += 1
                        print(f"[stt-groq] dropped tainted segment (agent speaking), "
                              f"interruptions={self._interruption_count}")
                    buf, voiced_time, last_voiced, segment_started, tainted = [], 0.0, None, None, False
        except Exception as e:
            print(f"[stt-groq] microphone failed ({e}); listener stopped")
            self.available = False

    def _enqueue_segment(self, segment: tuple) -> None:
        """Keep API work bounded and prefer the newest completed utterance."""
        try:
            self._segments.put_nowait(segment)
        except queue.Full:
            try:
                self._segments.get_nowait()
                self._segments.put_nowait(segment)
                self._segments_dropped += 1
            except (queue.Empty, queue.Full):
                self._segments_dropped += 1

    def _transcription_loop(self) -> None:
        """Run cloud calls away from audio segmentation and the video loop."""
        while not self._stop.is_set():
            try:
                segment = self._segments.get(timeout=0.5)
            except queue.Empty:
                continue
            if segment is None:
                return
            self._transcribe_segment(*segment)

    def _transcribe_segment(self, audio: np.ndarray, started_at: float,
                            ended_at: float, interruptions: int) -> None:
        """Send audio segment to Groq Whisper API."""
        if self._client is None:
            return
        try:
            response = self._client.audio.transcriptions.create(
                file=_wav_buffer(audio),
                model=self.model,
                language=self.language,
                response_format="verbose_json",
                temperature=0.0,
            )

            text = str(_field(response, "text", "") or "").strip()
            if not text:
                return

            raw_segments = list(_field(response, "segments", None) or [])
            segments = []
            for segment in raw_segments:
                segment_text = str(_field(segment, "text", "") or "").strip()
                if segment_text and _segment_is_trustworthy(
                        segment_text, _field(segment, "avg_logprob"),
                        _field(segment, "no_speech_prob")):
                    segments.append(segment)
            if raw_segments:
                text = " ".join(str(_field(segment, "text", "")).strip()
                                for segment in segments).strip()
            if not text:
                return

            words = []
            for segment in segments:
                words.extend(list(_field(segment, "words", None) or []))
            if not words:
                words.extend(list(_field(response, "words", None) or []))

            pauses = sum(
                1 for previous, current in zip(words, words[1:])
                if float(_field(current, "start", 0.0))
                - float(_field(previous, "end", 0.0)) >= 0.5)

            duration = len(audio) / _SAMPLE_RATE
            word_count = len(words) or len(text.split())

            print(f"[person says] {text}")
            self._out.put((text, started_at))

            wpm = word_count / max(duration, .1) * 60
            pause_frequency = pauses / max(duration, .1)
            if hasattr(self._history, "rolling_mean"):
                baseline = self._history.rolling_mean(
                    "speech_timing", "words_per_minute", 30 * 86400)
            else:
                baseline = self._history.mean_since(
                    "speech_timing", "words_per_minute", 30 * 86400)
            baseline_change = ((wpm - baseline) / max(abs(baseline), 1.0)
                               if baseline is not None else 0.0)
            response_latency = (max(0.0, started_at - self._last_agent_speech)
                                if self._last_agent_speech > 0 else None)
            turn_gap = (max(0.0, started_at - self._last_user_end)
                        if self._last_user_end > 0 else None)
            quality = min(1.0, word_count / 4) * min(1.0, duration / .8)

            self._metrics.put({
                "timestamp": started_at,
                "duration": round(duration, 2),
                "word_count": word_count,
                "words_per_minute": round(wpm, 1),
                "pauses": pauses,
                "pause_frequency": round(pause_frequency, 2),
                "response_latency": (round(response_latency, 2)
                                     if response_latency is not None else None),
                "turn_gap": round(turn_gap, 2) if turn_gap is not None else None,
                "interruptions": interruptions,
                "baseline_change": round(baseline_change, 3),
                "quality": round(quality, 2),
            })
            self._history.add("speech_timing", "words_per_minute", wpm, started_at)
            self._history.add("speech_timing", "pause_frequency",
                              pause_frequency, started_at)
            self._last_user_end = ended_at
            self._last_error = None

        except Exception as exc:
            self._last_error = _error_detail(exc)
            print(f"[stt-groq] transcription failed ({self._last_error})")

    def pop_utterances(self) -> list[tuple]:
        """Drain and return [(text, timestamp), ...] heard since last call."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def diagnostics(self) -> dict:
        """Credential- and transcript-free worker lifecycle state."""
        return {
            "available": bool(self.available),
            "status": "closed" if self._closed else "ready" if self.available else "unavailable",
            "model": self.model,
            "device": "cloud",
            "compute_type": "groq-api",
            "worker_alive": any(thread.is_alive() for thread in self._threads),
            "worker_pid": None,
            "native_runtime": {},
            "worker_ready": bool(self.available),
            "worker_exit_code": None,
            "load_latency_ms": None,
            "last_error": self._last_error,
            "segments_pending": self._segments.qsize(),
            "segments_dropped": self._segments_dropped,
        }

    def close(self) -> None:
        """Release any resources held here."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if hasattr(self, "_audio_in"):
            self.audio_bus.unsubscribe("speech-to-text")
        try:
            self._segments.put_nowait(None)
        except queue.Full:
            try:
                self._segments.get_nowait()
                self._segments.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self._client = None
        self.available = False

    def mark_agent_spoke(self, timestamp: float | None = None,
                         estimated_seconds: float | None = None) -> None:
        """Mark the start of an agent turn for response-latency measurement."""
        start = time.time() if timestamp is None else timestamp
        self._last_agent_speech = start
        hold = self.speech_tail_seconds
        if estimated_seconds is not None:
            hold = max(hold, min(float(estimated_seconds), _MAX_AGENT_MUTE_SECONDS))
        self._agent_speech_until = start + hold

    def pop_metrics(self) -> list[dict]:
        """Drain speech timing summaries; transcript text is intentionally absent."""
        out = []
        while True:
            try:
                out.append(self._metrics.get_nowait())
            except queue.Empty:
                return out
