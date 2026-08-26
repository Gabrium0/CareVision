"""Offline speech-to-text — the agent's ears.

`Listener` mirrors `audio/tts.py::Speaker`'s worker-thread design so the
video loop never blocks on audio: it subscribes to the shared 16 kHz mono
audio bus and segments speech with a simple energy VAD
(rolling RMS above a threshold opens a segment; ~0.8 s of silence closes
it); a spawned process transcribes closed segments with faster-whisper
(local CTranslate2 Whisper — offline-safe, so the showcase doesn't depend
on venue Wi-Fi). Native CTranslate2 libraries stay in that process so they
cannot collide with PyTorch's CUDA libraries. The main loop polls
`pop_utterances()` each tick.

Turn-taking: the robot must not transcribe its own voice. Any segment that
overlapped agent speech (`Speaker.speaking`, plus a short tail for room
echo) is dropped at capture time.

Both dependencies are optional extras (requirements-asr.txt); when either
is missing the Listener self-disables with a hint, matching the pattern of
modules/gesture.py, and the agent simply runs speak-only as before.
"""
from __future__ import annotations

import importlib.util
import importlib.abc
import os
import queue
import sys
import threading
import time
from multiprocessing import get_context

import numpy as np
from storage.history_store import HistoryStore

_SAMPLE_RATE = 16000
_BLOCK_SECONDS = 0.1


def _dependency_available() -> bool:
    """Check for faster-whisper without importing its native CTranslate2 DLLs."""
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def _error_detail(exc: BaseException) -> str:
    """Return a bounded worker error suitable for terminal diagnostics."""
    message = str(exc).strip()
    detail = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return detail[:240]


class _TorchImportBlocker(importlib.abc.MetaPathFinder):
    """Make optional Torch model-spec support unavailable inside ASR only."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(
                "Torch is disabled in the isolated CPU Whisper worker",
                name=fullname)
        return None


def _whisper_worker(in_q, out_q, model_size: str, language: str) -> None:
    """Own faster-whisper/CTranslate2 in an isolated spawned process."""
    blocker = _TorchImportBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        _run_whisper_worker(in_q, out_q, model_size, language)
    finally:
        # Spawned workers normally exit immediately after this function, but
        # keeping the import hook scoped is important for embedded callers and
        # unit tests that exercise the protocol in-process.
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass


def _run_whisper_worker(in_q, out_q, model_size: str, language: str) -> None:
    """Run the worker protocol while the caller owns native import isolation."""
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        out_q.put({
            "event": "ready", "model": model_size, "device": "cpu",
            "pid": os.getpid(),
            "native_runtime": {
                "ctranslate2_loaded": "ctranslate2" in sys.modules,
                "torch_loaded": "torch" in sys.modules,
                "torch_import_blocked": True,
                "cuda_requested": False,
            },
        })
    except BaseException as exc:  # native-runtime failures must stay isolated
        try:
            out_q.put({"event": "error", "phase": "load",
                       "error": _error_detail(exc)})
        except Exception:  # noqa: BLE001 - parent may already be gone
            pass
        return

    while True:
        try:
            item = in_q.get()
        except (EOFError, OSError, KeyboardInterrupt):
            return
        if item is None:
            return
        audio, timestamp, ended_at, interruptions = item
        try:
            segments, _info = model.transcribe(
                audio, language=language, beam_size=1,
                vad_filter=True, word_timestamps=True)
            segments = list(segments)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            words = [word for segment in segments for word in (segment.words or [])]
            pauses = sum(
                1 for previous, current in zip(words, words[1:])
                if float(current.start) - float(previous.end) >= 0.5)
            out_q.put({
                "event": "result",
                "text": text,
                "timestamp": float(timestamp),
                "ended_at": float(ended_at),
                "interruptions": int(interruptions),
                "duration": len(audio) / _SAMPLE_RATE,
                "word_count": len(words),
                "pauses": pauses,
            })
        except Exception as exc:  # noqa: BLE001 - one bad segment is nonfatal
            try:
                out_q.put({"event": "error", "phase": "transcribe",
                           "error": _error_detail(exc)})
            except Exception:  # noqa: BLE001 - parent may already be gone
                return


class Listener:
    """Microphone -> parent VAD -> isolated Whisper worker -> text."""

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
        self.model_size = model_size
        self.speaker = speaker                 # audio/tts.Speaker, for turn-taking
        if audio_bus is None:
            from audio.bus import AudioBus
            audio_bus = AudioBus()
        self.audio_bus = audio_bus
        self._out: queue.Queue[tuple] = queue.Queue()
        self._metrics: queue.Queue[dict] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_agent_speech = -1e9
        self._last_user_end = -1e9
        self._interruption_count = 0
        self._history = HistoryStore.instance()
        if hasattr(self._history, "rolling_mean"):
            # Bootstrap away from the camera/agent tick; transcript handling
            # must never perform a synchronous SQLite query.
            self._history.rolling_mean(
                "speech_timing", "words_per_minute", 30 * 86400)
        self._ctx = get_context("spawn")
        self._segments = None
        self._worker_out = None
        self._worker = None
        self._worker_ready = False
        self._worker_started_at: float | None = None
        self._worker_ready_at: float | None = None
        self._worker_pid: int | None = None
        self._worker_runtime: dict = {}
        self._worker_last_error: str | None = None
        self._worker_failure_reported = False
        self._closed = False
        if not enabled:
            return
        if not _dependency_available():
            print("[stt] listener unavailable (faster-whisper not installed); "
                  "agent is speak-only. "
                  "Install with: pip install -r requirements-asr.txt")
            return
        if not self._start_worker():
            return
        self.available = True
        self._audio_in = self.audio_bus.subscribe("speech-to-text")
        t = threading.Thread(target=self._segment_loop, daemon=True,
                             name="stt-segment")
        t.start()
        self._threads.append(t)

    def _start_worker(self) -> bool:
        """Spawn the process that exclusively owns faster-whisper."""
        try:
            self._segments = self._ctx.Queue()
            self._worker_out = self._ctx.Queue()
            self._worker = self._ctx.Process(
                target=_whisper_worker,
                args=(self._segments, self._worker_out,
                      self.model_size, self.language),
                daemon=True)
            self._worker.start()
            self._worker_started_at = time.time()
            return True
        except Exception as exc:  # noqa: BLE001 - optional feature boundary
            print(f"[stt] listener worker failed to start ({_error_detail(exc)}); "
                  "agent is speak-only")
            self._close_worker_queues()
            self._worker = None
            return False

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
                        try:
                            self._segments.put(
                                (np.concatenate(buf).ravel(), segment_started or now,
                                 last_voiced or now, self._interruption_count))
                            self._interruption_count = 0
                        except (EOFError, OSError, ValueError):
                            self.available = False
                            return
                    elif tainted:
                        self._interruption_count += 1
                    buf, voiced_time, last_voiced, segment_started, tainted = [], 0.0, None, None, False
        except Exception as e:  # noqa: BLE001
            print(f"[stt] microphone failed ({e}); listener stopped")
            self.available = False

    # --------------------------------------------------------- transcribe

    def _publish_result(self, message: dict) -> None:
        """Turn a bounded worker result into the existing public queues."""
        text = str(message.get("text") or "").strip()
        if not text:
            return
        timestamp = float(message["timestamp"])
        ended_at = float(message["ended_at"])
        duration = float(message.get("duration") or 0.0)
        word_count = int(message.get("word_count") or 0)
        pauses = int(message.get("pauses") or 0)
        interruptions = int(message.get("interruptions") or 0)
        print(f"[person says] {text}")
        self._out.put((text, timestamp))
        wpm = word_count / max(duration, .1) * 60
        pause_frequency = pauses / max(duration, .1)
        if hasattr(self._history, "rolling_mean"):
            baseline = self._history.rolling_mean(
                "speech_timing", "words_per_minute", 30 * 86400)
        else:  # lightweight compatibility stores used by integrations/tests
            baseline = self._history.mean_since(
                "speech_timing", "words_per_minute", 30 * 86400)
        baseline_change = ((wpm - baseline) / max(abs(baseline), 1.0)
                           if baseline is not None else 0.0)
        response_latency = (max(0.0, timestamp - self._last_agent_speech)
                            if self._last_agent_speech > 0 else None)
        turn_gap = (max(0.0, timestamp - self._last_user_end)
                    if self._last_user_end > 0 else None)
        quality = min(1.0, word_count / 4) * min(1.0, duration / .8)
        self._metrics.put({
            "timestamp": timestamp,
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
        self._history.add("speech_timing", "words_per_minute", wpm, timestamp)
        self._history.add("speech_timing", "pause_frequency",
                          pause_frequency, timestamp)
        self._last_user_end = ended_at

    def _drain_worker(self) -> None:
        """Drain worker events and detect a native worker exit without blocking."""
        if self._worker_out is not None:
            while True:
                try:
                    message = self._worker_out.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, OSError, ValueError):
                    break
                event = message.get("event")
                if event == "ready":
                    self._worker_ready = True
                    self._worker_ready_at = time.time()
                    self._worker_pid = int(message.get("pid") or 0) or None
                    runtime = message.get("native_runtime")
                    self._worker_runtime = dict(runtime) if isinstance(runtime, dict) else {}
                    print(f"[stt] whisper-{message.get('model', self.model_size)} "
                          f"ready ({message.get('device', 'cpu')} worker)")
                elif event == "result":
                    self._publish_result(message)
                elif event == "error":
                    phase = message.get("phase", "worker")
                    self._worker_last_error = str(
                        message.get("error", "unknown error"))[:240]
                    print(f"[stt] whisper {phase} failed "
                          f"({message.get('error', 'unknown error')})")
                    if phase == "load":
                        self.available = False
                        self._worker_failure_reported = True
        worker = self._worker
        if (worker is not None and not self._stop.is_set()
                and not worker.is_alive() and not self._worker_failure_reported):
            self.available = False
            self._worker_failure_reported = True
            print(f"[stt] whisper worker exited unexpectedly "
                  f"(exit code {getattr(worker, 'exitcode', 'unknown')}); "
                  "agent is speak-only")

    # -------------------------------------------------------------- public

    def pop_utterances(self) -> list[tuple]:
        """Drain and return [(text, timestamp), ...] heard since last call."""
        self._drain_worker()
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def diagnostics(self) -> dict:
        """Credential- and transcript-free worker lifecycle state."""
        self._drain_worker()
        worker = self._worker
        return {
            "available": bool(self.available),
            "status": ("closed" if self._closed else "ready" if self._worker_ready
                       else "starting" if worker is not None and worker.is_alive()
                       else "unavailable"),
            "model": self.model_size,
            "device": "cpu",
            "compute_type": "int8",
            "worker_alive": bool(worker is not None and worker.is_alive()),
            "worker_pid": self._worker_pid,
            "native_runtime": dict(self._worker_runtime),
            "worker_ready": bool(self._worker_ready),
            "worker_exit_code": getattr(worker, "exitcode", None) if worker else None,
            "load_latency_ms": (round((self._worker_ready_at - self._worker_started_at)
                                      * 1000.0, 1)
                                if self._worker_ready_at and self._worker_started_at else None),
            "last_error": self._worker_last_error,
        }

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if hasattr(self, "_audio_in"):
            self.audio_bus.unsubscribe("speech-to-text")
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        if self._segments is not None:
            try:
                self._segments.put_nowait(None)
            except (EOFError, OSError, ValueError, queue.Full):
                pass
        worker = self._worker
        if worker is not None:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.5)
        self._close_worker_queues()
        self._worker = None
        self.available = False

    def _close_worker_queues(self) -> None:
        """Close multiprocessing queues without waiting on feeder threads."""
        for worker_queue in (self._segments, self._worker_out):
            if worker_queue is None:
                continue
            try:
                worker_queue.close()
                worker_queue.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass
        self._segments = None
        self._worker_out = None

    def mark_agent_spoke(self, timestamp: float | None = None) -> None:
        """Mark the start of an agent turn for response-latency measurement."""
        self._last_agent_speech = time.time() if timestamp is None else timestamp

    def pop_metrics(self) -> list[dict]:
        """Drain speech timing summaries; transcript text is intentionally absent."""
        self._drain_worker()
        out = []
        while True:
            try:
                out.append(self._metrics.get_nowait())
            except queue.Empty:
                return out
