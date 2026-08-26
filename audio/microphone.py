"""Single-owner microphone capture published onto the shared audio bus."""
from __future__ import annotations

import threading

from core.capabilities import CapabilityRegistry, CapabilityStatus


class MicrophoneProducer:
    """Open one 16 kHz mono input stream and publish non-blocking audio blocks."""

    def __init__(self, bus, enabled: bool = True, sample_rate: int = 16000,
                 block_seconds: float = 0.1, device=None, stream_factory=None):
        self.bus = bus
        self.sample_rate = sample_rate
        self.block_size = max(1, int(sample_rate * block_seconds))
        self.available = False
        self._stop = threading.Event()
        self._thread = None
        self._stream = None
        registry = CapabilityRegistry.instance()
        if not enabled:
            registry.set("microphone", "hardware", CapabilityStatus.UNAVAILABLE,
                         "disabled")
            return
        try:
            if stream_factory is None:
                import sounddevice as sd
                stream_factory = sd.InputStream
            self._stream = stream_factory(samplerate=sample_rate, channels=1,
                                          dtype="float32", blocksize=self.block_size,
                                          device=device)
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - optional hardware boundary
            registry.set("microphone", "hardware", CapabilityStatus.UNAVAILABLE,
                         f"input unavailable: {type(exc).__name__}")
            self._close_stream()
            return
        self.available = True
        registry.set("microphone", "hardware", CapabilityStatus.READY,
                     "16 kHz mono shared audio bus")
        self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                        name="microphone-capture")
        self._thread.start()

    def _capture_loop(self) -> None:
        """Read fixed blocks without allowing downstream consumers to block capture."""
        import time

        try:
            while not self._stop.is_set():
                audio, _overflow = self._stream.read(self.block_size)
                self.bus.publish(audio, time.time())
        except Exception as exc:  # noqa: BLE001 - device may disappear at runtime
            if not self._stop.is_set():
                self.available = False
                CapabilityRegistry.instance().set(
                    "microphone", "hardware", CapabilityStatus.UNAVAILABLE,
                    f"capture failed: {type(exc).__name__}")

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:  # noqa: BLE001 - best-effort hardware cleanup
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001 - best-effort hardware cleanup
            pass

    def close(self) -> None:
        """Stop capture and release the physical input device."""
        self._stop.set()
        self._close_stream()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.available = False
