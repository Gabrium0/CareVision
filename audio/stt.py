"""Offline speech-to-text — the agent's ears.

`Listener` mirrors `audio/tts.py::Speaker`'s worker-thread design so the
video loop never blocks on audio: a capture thread owns the microphone
(sounddevice, 16 kHz mono) and segments speech with a simple energy VAD
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

_SAMPLE_RATE = 16000
_BLOCK_SECONDS = 0.1


class Listener:
    """Microphone -> VAD segments -> Whisper text, on worker threads."""

    def __init__(self, enabled: bool = True, model_size: str = "base",
                 energy_threshold: float = 0.01, silence_seconds: float = 0.8,
                 min_voiced_seconds: float = 0.3, max_segment_seconds: float = 12.0,
                 speech_tail_seconds: float = 0.5, language: str = "en",
                 speaker=None):
        self.available = False
        self.energy_threshold = energy_threshold
        self.silence_seconds = silence_seconds
        self.min_voiced_seconds = min_voiced_seconds
        self.max_segment_seconds = max_segment_seconds
        self.speech_tail_seconds = speech_tail_seconds
        self.language = language
        self.speaker = speaker                 # audio/tts.Speaker, for turn-taking
        self._out: queue.Queue[tuple] = queue.Queue()
        self._segments: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_agent_speech = -1e9
        if not enabled:
            return
        try:
            import sounddevice  # noqa: F401  (validate before starting threads)
            import faster_whisper  # noqa: F401
        except Exception as e:  # noqa: BLE001
            print(f"[stt] listener unavailable ({e}); agent is speak-only. "
                  "Install with: pip install -r requirements-asr.txt")
            return
        self.model_size = model_size
        self.available = True
        for target, name in ((self._capture_loop, "stt-capture"),
                             (self._transcribe_loop, "stt-transcribe")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    # ------------------------------------------------------------- capture

    def _agent_is_speaking(self) -> bool:
        """True while the agent talks (or just finished — room-echo tail)."""
        if self.speaker is not None and getattr(self.speaker, "speaking", False):
            self._last_agent_speech = time.time()
            return True
        return time.time() - self._last_agent_speech < self.speech_tail_seconds

    def _capture_loop(self) -> None:
        """Own the mic; segment speech by energy; queue closed segments."""
        import sounddevice as sd
        block = int(_SAMPLE_RATE * _BLOCK_SECONDS)
        buf: list[np.ndarray] = []
        voiced_time = 0.0
        last_voiced = None
        tainted = False                        # overlapped agent speech
        try:
            with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=block) as stream:
                print("[stt] microphone open (energy VAD, "
                      f"whisper-{self.model_size})")
                while not self._stop.is_set():
                    audio, _overflow = stream.read(block)
                    now = time.time()
                    if self._agent_is_speaking():
                        tainted = bool(buf) or tainted
                        if not buf:
                            continue           # don't even open a segment
                    rms = float(np.sqrt(np.mean(audio ** 2)))
                    voiced = rms >= self.energy_threshold
                    if voiced:
                        buf.append(audio.copy())
                        voiced_time += _BLOCK_SECONDS
                        last_voiced = now
                    elif buf:
                        buf.append(audio.copy())   # keep trailing context
                    if not buf:
                        continue
                    seg_len = len(buf) * _BLOCK_SECONDS
                    silence = (now - last_voiced) if last_voiced else 0.0
                    if silence >= self.silence_seconds or \
                            seg_len >= self.max_segment_seconds:
                        if voiced_time >= self.min_voiced_seconds and not tainted:
                            self._segments.put(
                                (np.concatenate(buf).ravel(), last_voiced))
                        buf, voiced_time, last_voiced, tainted = [], 0.0, None, False
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
                audio, ts = self._segments.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                segments, _info = model.transcribe(
                    audio, language=self.language, beam_size=1)
                text = " ".join(s.text.strip() for s in segments).strip()
                if text:
                    print(f"[person says] {text}")
                    self._out.put((text, ts))
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
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
