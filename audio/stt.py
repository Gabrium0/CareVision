"""Offline speech-to-text — the agent's ears.

`Listener` mirrors `audio/tts.py::Speaker`'s worker-thread design so the
video loop never blocks on audio: it subscribes to the shared 16 kHz mono
audio bus and segments speech with a simple energy VAD
(rolling RMS above a threshold opens a segment; ~0.8 s of silence closes
it); a second thread transcribes closed segments with faster-whisper
(local CTranslate2 Whisper — offline-safe, so the showcase doesn't depend
on venue Wi-Fi). The main loop polls `pop_utterances()` each tick.

Turn-taking: the robot must not transcribe its own voice. Any segment that
overlapped agent speech (`Speaker.speaking`, plus a short tail for room
echo) is dropped at capture time.

Both dependencies are optional extras (requirements-asr.txt); when either
is missing the Listener self-disables with a hint, matching the pattern of
modules/gesture.py, and the agent simply runs speak-only as before.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
from storage.history_store import HistoryStore

_SAMPLE_RATE = 16000
_BLOCK_SECONDS = 0.1


class Listener:
    """Microphone -> VAD segments -> Whisper text, on worker threads."""

    def __init__(self, enabled: bool = True, model_size: str = "base",
                 energy_threshold: float = 0.01, silence_seconds: float = 0.8,
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
        self.speaker = speaker                 # audio/tts.Speaker, for turn-taking
        if audio_bus is None:
            from audio.bus import AudioBus
            audio_bus = AudioBus()
        self.audio_bus = audio_bus
        self._out: queue.Queue[tuple] = queue.Queue()
        self._metrics: queue.Queue[dict] = queue.Queue()
        self._segments: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_agent_speech = -1e9
        self._last_user_end = -1e9
        self._interruption_count = 0
        self._history = HistoryStore.instance()
        if not enabled:
            return
        try:
            import faster_whisper  # noqa: F401
        except Exception as e:  # noqa: BLE001
            print(f"[stt] listener unavailable ({e}); agent is speak-only. "
                  "Install with: pip install -r requirements-asr.txt")
            return
        self.model_size = model_size
        self.available = True
        self._audio_in = self.audio_bus.subscribe("speech-to-text")
        for target, name in ((self._segment_loop, "stt-segment"),
                             (self._transcribe_loop, "stt-transcribe")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    # ----------------------------------------------------------- segmentation

    def _agent_is_speaking(self) -> bool:
        """True while the agent talks (or just finished — room-echo tail)."""
        if self.speaker is not None and getattr(self.speaker, "speaking", False):
            self._last_agent_speech = time.time()
            return True
        return time.time() - self._last_agent_speech < self.speech_tail_seconds

    def _segment_loop(self) -> None:
        """Consume shared microphone blocks and close speech segments by energy."""
        buf: list[np.ndarray] = []
        voiced_time = 0.0
        last_voiced = None
        segment_started = None
        tainted = False                        # overlapped agent speech
        print("[stt] shared audio attached (energy VAD, "
              f"whisper-{self.model_size})")
        try:
            while not self._stop.is_set():
                try:
                    audio, now = self._audio_in.get(timeout=0.5)
                except queue.Empty:
                    continue
                block_seconds = len(audio) / _SAMPLE_RATE
                if block_seconds <= 0:
                    continue
                audio = np.asarray(audio, dtype=np.float32).ravel()
                if self._agent_is_speaking():
                    tainted = bool(buf) or tainted
                    if not buf:
                        continue           # don't even open a segment
                rms = float(np.sqrt(np.mean(audio ** 2)))
                voiced = rms >= self.energy_threshold
                if voiced:
                    if not buf:
                        segment_started = now
                    buf.append(audio.copy())
                    voiced_time += block_seconds
                    last_voiced = now
                elif buf:
                    buf.append(audio.copy())   # keep trailing context
                if not buf:
                    continue
                seg_len = sum(len(part) for part in buf) / _SAMPLE_RATE
                silence = (now - last_voiced) if last_voiced else 0.0
                if silence >= self.silence_seconds or \
                        seg_len >= self.max_segment_seconds:
                    if voiced_time >= self.min_voiced_seconds and not tainted:
                        self._segments.put(
                            (np.concatenate(buf).ravel(), segment_started or now,
                             last_voiced or now, self._interruption_count))
                        self._interruption_count = 0
                    elif tainted:
                        self._interruption_count += 1
                    buf, voiced_time, last_voiced, segment_started, tainted = [], 0.0, None, None, False
        except Exception as e:  # noqa: BLE001
            print(f"[stt] microphone failed ({e}); listener stopped")
            self.available = False

    # --------------------------------------------------------- transcribe

    def _transcribe_loop(self) -> None:
        """Load Whisper once, then transcribe queued segments to text."""
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(self.model_size, device="cpu",
                                 compute_type="int8")
            print(f"[stt] whisper-{self.model_size} ready")
        except Exception as e:  # noqa: BLE001
            print(f"[stt] whisper load failed ({e}); listener stopped")
            self.available = False
            return
        while not self._stop.is_set():
            try:
                audio, ts, ended_at, interruptions = self._segments.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                segments, _info = model.transcribe(
                    audio, language=self.language, beam_size=1,
                    vad_filter=True, word_timestamps=True)
                segments = list(segments)
                text = " ".join(s.text.strip() for s in segments).strip()
                if text:
                    print(f"[person says] {text}")
                    self._out.put((text, ts))
                    words = [w for s in segments for w in (s.words or [])]
                    duration = len(audio) / _SAMPLE_RATE
                    pauses = sum(1 for a, b in zip(words, words[1:])
                                 if float(b.start) - float(a.end) >= 0.5)
                    wpm = len(words) / max(duration, .1) * 60
                    baseline = self._history.mean_since("speech_timing", "words_per_minute",
                                                        30 * 86400)
                    baseline_change = ((wpm - baseline) / max(abs(baseline), 1.0)
                                       if baseline is not None else 0.0)
                    response_latency = max(0.0, ts - self._last_agent_speech) \
                        if self._last_agent_speech > 0 else None
                    turn_gap = max(0.0, ts - self._last_user_end) \
                        if self._last_user_end > 0 else None
                    quality = min(1.0, len(words) / 4) * min(1.0, duration / .8)
                    metrics = {"timestamp": ts, "duration": round(duration, 2),
                               "word_count": len(words), "words_per_minute": round(wpm, 1),
                               "pauses": pauses, "pause_frequency": round(pauses/max(duration, .1), 2),
                               "response_latency": (round(response_latency, 2)
                                                    if response_latency is not None else None),
                               "turn_gap": round(turn_gap, 2) if turn_gap is not None else None,
                               "interruptions": interruptions,
                               "baseline_change": round(baseline_change, 3),
                               "quality": round(quality, 2)}
                    self._metrics.put(metrics)
                    self._history.add("speech_timing", "words_per_minute", wpm, ts)
                    self._history.add("speech_timing", "pause_frequency",
                                      pauses/max(duration, .1), ts)
                    self._last_user_end = ended_at
            except Exception as e:  # noqa: BLE001
                print(f"[stt] transcription failed: {e}")

    # -------------------------------------------------------------- public

    def pop_utterances(self) -> list[tuple]:
        """Drain and return [(text, timestamp), ...] heard since last call."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self._stop.set()
        if hasattr(self, "_audio_in"):
            self.audio_bus.unsubscribe("speech-to-text")
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def mark_agent_spoke(self, timestamp: float | None = None) -> None:
        """Mark the start of an agent turn for response-latency measurement."""
        self._last_agent_speech = time.time() if timestamp is None else timestamp

    def pop_metrics(self) -> list[dict]:
        """Drain speech timing summaries; transcript text is intentionally absent."""
        out = []
        while True:
            try:
                out.append(self._metrics.get_nowait())
            except queue.Empty:
                return out
