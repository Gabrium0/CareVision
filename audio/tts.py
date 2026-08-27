"""Offline text-to-speech — the agent's mouth.

Engine chain, first available wins:
  1. Piper   — local neural voice (ONNX on CPU; natural sound, no network).
               Voice models live in assets/tts; fetch one ahead of a demo
               with ``python -m audio.tts_piper --download``.
  2. pyttsx3 — offline system voice (the original engine).
  3. print   — utterances are only printed, so the agent still "speaks".

Speech is queued to a worker thread so ``say()`` never blocks the video loop;
the model load and a warm-up synthesis happen on that same thread after start,
so startup latency is hidden and the demo's first spoken line is instant.
Utterances are synthesized sentence by sentence, so audio begins while the rest
is still being synthesized.

If everything is unavailable, utterances are printed instead, matching the
graceful-degrade pattern of the rest of the runtime.
"""
from __future__ import annotations

import queue
import re
import threading
import time

import numpy as np

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")
_MAX_CHUNK = 280        # characters per synthesis chunk before clause split
_SENTENCE_PAUSE = 0.28  # calm inter-sentence breathing room (seconds)
# On the device (remote-sink) route there is no local player to block on, so the
# speaker holds `speaking` for each chunk's real duration plus this short tail,
# letting the far-end buffer drain before audio/stt unmutes the microphone —
# otherwise the agent's own device-played voice is transcribed back as a reply.
_REMOTE_TAIL_SECONDS = 0.35


def _split_chunks(text: str) -> list[str]:
    """Split an utterance into short, natural synthesis chunks."""
    chunks: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > _MAX_CHUNK:
            cut = -1
            for match in _CLAUSE_SPLIT.finditer(sentence, 1, _MAX_CHUNK):
                cut = match.start()
            if cut <= 0:
                cut = sentence.rfind(" ", 1, _MAX_CHUNK)
            if cut <= 0:
                cut = _MAX_CHUNK
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


class Speaker:
    """Offline text-to-speech worker on its own thread (the agent's voice)."""
    def __init__(self, rate: int = 165, volume: float = 1.0, enabled: bool = True,
                 engine: str = "auto"):
        self.enabled = enabled
        self.speaking = False   # True while TTS is audible (read by audio/stt.Listener)
        self.rate = int(rate)   # words per minute-ish; maps to Piper length_scale
        self.volume = float(volume)
        self._engine_pref = "piper" if engine == "auto" else engine
        self.engine_name = "print"   # settled synchronously below
        self.ready = False           # True once the chosen engine finished loading
        self.error: str | None = None
        self.model = ""
        self._engine = None          # piper.PiperVoice or pyttsx3 engine
        self._backend = None         # audio.tts_piper.PiperBackend when piper
        self.remote_sink = None      # callable(samples float32, rate) — device route
        self.local_playback = True   # False mutes laptop speakers while routed
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        if not enabled:
            return
        backend = self._select_backend()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="tts-speaker")
        self._thread.start()
        if backend is None:
            print(f"[tts] {self.error or 'no speech engine'}; printing utterances")

    # ------------------------------------------------------------- selection

    def _select_backend(self):
        """Pick the best available mouth without loading anything heavy."""
        if self._engine_pref in ("piper", "auto"):
            from audio import tts_piper
            if tts_piper.dependency_available():
                model_path = tts_piper.resolve_voice(None)
                if model_path is not None:
                    length_scale = min(2.0, max(0.5, 165.0 / max(1, self.rate)))
                    self._backend = tts_piper.PiperBackend(
                        model_path, volume=self.volume,
                        length_scale=length_scale)
                    self.engine_name = "piper"
                    self.model = model_path.stem
                    return self._backend
                self.error = ("no Piper voice in assets/tts "
                              "(python -m audio.tts_piper --download)")
            else:
                self.error = "piper-tts not installed"
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            self.engine_name = "pyttsx3"
            self.model = "system"
            print("[tts] pyttsx3 speaker ready")
            return self._engine
        except Exception as exc:  # noqa: BLE001
            self.error = f"pyttsx3 unavailable ({exc})"
            return None

    def status(self) -> dict:
        """Credential-safe diagnostics for the capability registry/UI."""
        return {"engine": self.engine_name, "ready": self.ready,
                "model": self.model, "error": self.error,
                "remote_sink": self.remote_sink is not None,
                "local_playback": bool(self.local_playback)}

    def set_remote_sink(self, sink, local_playback: bool = False) -> None:
        """Route synthesized speech to a remote output (e.g. the paired iPad).

        ``sink(samples_float32, rate)`` is called from the TTS worker thread for
        every synthesized chunk; it must never block (the WebRTC link's paced
        buffer satisfies this). Only the Piper engine produces PCM a remote
        sink can carry — system-voice fallbacks keep playing locally and say
        so once. Turn-taking (`speaking`) still covers routed speech.
        """
        self.remote_sink = sink
        self.local_playback = bool(local_playback)
        if sink is not None and self.engine_name != "piper":
            print(f"[tts] engine {self.engine_name!r} cannot route to a remote "
                  "device; keeping laptop speakers on")

    # ----------------------------------------------------------------- worker

    def _run(self) -> None:
        player = None
        if self._backend is not None:
            try:
                import sounddevice as sd

                player = sd
            except Exception as exc:  # noqa: BLE001
                self.error = f"audio output unavailable ({exc})"
                self._downgrade_to_pyttsx3()
        if self._backend is not None:
            try:
                self._engine = self._backend.load()
            except Exception as exc:  # noqa: BLE001
                self.error = f"piper load failed ({exc})"
                self._downgrade_to_pyttsx3()
        if self._backend is not None:
            # Warm-up: synthesize-and-discard so the first real line is instant.
            try:
                for _chunk in self._backend.synthesize(self._engine, "Hello."):
                    pass
            except Exception:  # noqa: BLE001
                pass
            self.ready = True
            print(f"[tts] piper speaker ready ({self.model})")
        elif self._engine is not None:
            self.ready = True
        while True:
            text = self._q.get()
            if text is None:
                break
            # Flag flips inside the worker around the blocking call, so it is
            # exact by construction — the Listener mutes while this is True.
            self.speaking = True
            try:
                self._emit(text, player)
            except Exception as exc:  # noqa: BLE001
                print(f"[tts] speak failed: {exc}")
            finally:
                self.speaking = False

    def _downgrade_to_pyttsx3(self) -> None:
        self._backend = None
        self.model = ""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            self.engine_name = "pyttsx3"
            self.model = "system"
            self.ready = True
        except Exception as exc:  # noqa: BLE001
            self.error = f"{self.error}; pyttsx3 unavailable ({exc})"
            self.engine_name = "print"
            self._engine = None

    def _emit(self, text: str, player) -> None:
        """Speak one utterance through whichever engine survived selection."""
        if self._backend is not None:
            first = True
            routed_remote = False
            for chunk in _split_chunks(text):
                samples = None
                sample_rate = 22050
                for audio, rate in self._backend.synthesize(self._engine, chunk):
                    samples = audio
                    sample_rate = rate
                if samples is None or len(samples) == 0:
                    continue
                if self.remote_sink is not None:
                    try:
                        self.remote_sink(
                            np.asarray(samples, dtype=np.float32), sample_rate)
                        routed_remote = True
                    except Exception as exc:  # noqa: BLE001 - a dead link mutes
                        print(f"[tts] remote sink failed ({exc}); "
                              "falling back to laptop audio")
                        self.local_playback = True
                if not self.local_playback:
                    # No local player blocks here, so hold `speaking` for the
                    # chunk's real audible duration; the mic stays muted for the
                    # whole time the device is playing this line.
                    time.sleep(len(samples) / float(sample_rate))
                    first = False
                    continue
                if not first and _SENTENCE_PAUSE > 0:
                    time.sleep(_SENTENCE_PAUSE)
                first = False
                player.play(samples, sample_rate)
                player.wait()
            if routed_remote and not self.local_playback:
                time.sleep(_REMOTE_TAIL_SECONDS)   # let the far-end buffer drain
            return
        if self._engine is not None:
            self._engine.say(text)
            self._engine.runAndWait()

    # ------------------------------------------------------------------ api

    def say(self, text: str) -> None:
        """Queue a line to be spoken (non-blocking)."""
        if not text:
            return
        print(f"[agent speaks] {text}")
        if self.enabled:
            self._q.put(text)

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self.remote_sink = None
        if self._thread is not None:
            self._q.put(None)
