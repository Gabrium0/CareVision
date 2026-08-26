"""PCM bridges between the WebRTC iPad link and the shared 16 kHz audio bus.

Deliberately free of aiortc/PyAV imports so `core.ipad_camera` stays
importable without the optional WebRTC stack (the same rule
`core/ipad_link.py` follows): everything here is plain numpy + threading.

Two directions live here:

* ``to_mono_16k`` — convert a decoded browser audio frame (Opus arrives as
  48 kHz stereo s16 through aiortc) into the float32 mono 16 kHz shape the
  shared `audio.bus.AudioBus` and `audio.stt.Listener` already speak.
* ``PacedPcmSource`` — a small thread-safe FIFO for agent speech flowing the
  other way. The TTS worker thread writes whole synthesized chunks from any
  thread; the link's asyncio loop drains it at wall-clock pace, so the
  outbound WebRTC track looks like a live microphone instead of bursting.
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

SAMPLE_RATE = 16000


def _as_float(arr: np.ndarray) -> np.ndarray:
    """Scale integer PCM into [-1, 1] floats without clipping headroom."""
    if arr.dtype.kind == "i":
        peak = np.float32(32768.0 if arr.dtype.itemsize <= 2 else 2147483648.0)
        return arr.astype(np.float32) / peak
    return np.clip(arr.astype(np.float32), -1.0, 1.0)


def pcm_to_mono(samples, channels: int = 1) -> np.ndarray:
    """Downmix one decoded frame to mono floats.

    Handles both PyAV layouts: planar frames arrive as ``(channels, n)``,
    packed interleaved frames as ``(1, n * channels)`` (or flat). A single
    malformed shape must never take the audio path down, so unknown layouts
    degrade to a flatten rather than raising.
    """
    arr = _as_float(np.asarray(samples))
    channels = max(1, int(channels))
    if channels == 1 or arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim != 2:
        return arr.reshape(-1)
    if arr.shape[0] == channels:            # planar (channels, n)
        return arr.mean(axis=0)
    if arr.shape[1] == channels:            # (n, channels)
        return arr.mean(axis=1)
    if arr.size % channels == 0:            # packed interleaved
        return arr.reshape(-1, channels).mean(axis=1)
    return arr.reshape(-1)


def to_mono_16k(samples, rate: int = 48000, channels: int = 1) -> np.ndarray:
    """Resample one frame to the bus's 16 kHz mono float32 contract."""
    mono = pcm_to_mono(samples, channels)
    rate = int(rate)
    if rate == SAMPLE_RATE or mono.size == 0:
        return np.ascontiguousarray(mono, dtype=np.float32)
    if rate > SAMPLE_RATE and rate % SAMPLE_RATE == 0:
        factor = rate // SAMPLE_RATE         # 48k -> 16k decimation
        usable = (mono.size // factor) * factor
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        return mono[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)
    if mono.size < 2:                        # too short to interpolate
        return np.empty(0, dtype=np.float32)
    out_len = int(round(mono.size * SAMPLE_RATE / rate))
    positions = np.linspace(0.0, mono.size - 1, num=out_len)
    return np.interp(positions, np.arange(mono.size), mono).astype(np.float32)


class PacedPcmSource:
    """Thread-safe FIFO of 16 kHz mono speech with real-time drain pacing.

    The writer side is called from the TTS worker thread; the reader side runs
    on the link's asyncio loop. Pacing math lives with the reader's clock (see
    `_AgentAudioTrack` in core/ipad_link.py): this class only guarantees FIFO
    order, bounded partial reads, and lock-free-adjacent safety.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 # 20ms blocks: 320 samples @16kHz resample to exactly 960 @48kHz,
                 # a 1:1 match with aiortc's Opus 20ms frame. Larger blocks make
                 # recv() emit a burst then sleep, which jitters into choppy audio.
                 block_samples: int = 20 * SAMPLE_RATE // 1000):
        self.sample_rate = int(sample_rate)
        self.block_samples = max(1, int(block_samples))
        self._parts: deque[np.ndarray] = deque()
        self._pending = 0
        self._lock = threading.Lock()

    def write(self, samples) -> int:
        """Queue one mono block; returns the number of samples accepted."""
        arr = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).ravel())
        if arr.size == 0 or not np.isfinite(arr).all():
            return 0
        with self._lock:
            self._parts.append(arr)
            self._pending += arr.size
        return arr.size

    def pending(self) -> int:
        """Samples currently buffered (diagnostics only)."""
        with self._lock:
            return self._pending

    def read(self, max_samples: int | None = None) -> np.ndarray | None:
        """Pop up to one block of samples, oldest first; None when empty."""
        want = self.block_samples if max_samples is None else max(1, int(max_samples))
        with self._lock:
            if self._pending == 0:
                return None
            parts: list[np.ndarray] = []
            need = min(want, self._pending)
            while need > 0 and self._parts:
                head = self._parts[0]
                take = head[:need]
                parts.append(take)
                need -= take.size
                self._pending -= take.size
                if take.size == head.size:
                    self._parts.popleft()
                else:
                    self._parts[0] = head[take.size:]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def clear(self) -> None:
        """Drop everything buffered (peer teardown must not speak stale text)."""
        with self._lock:
            self._parts.clear()
            self._pending = 0
