"""Deterministic replay producers implementing live audio/listener interfaces."""
from __future__ import annotations

import queue
import wave
from pathlib import Path

import numpy as np


class ReplayListener:
    """Scripted-answer source with the same polling contract as Listener."""
    available = True

    def __init__(self):
        self._utterances: queue.Queue[tuple[str, float]] = queue.Queue()
        self._metrics: queue.Queue[dict] = queue.Queue()

    def feed_context(self, ctx) -> None:
        """Queue synchronized scripted answers from a replay frame."""
        for event in ctx.extras.get("replay_channels", {}).get("answer", []):
            text = str(event.get("text", event.get("value", ""))).strip()
            if text:
                self._utterances.put((text, ctx.timestamp))
            if isinstance(event.get("metrics"), dict):
                self._metrics.put(dict(event["metrics"]))

    def pop_utterances(self) -> list[tuple[str, float]]:
        """Drain scripted utterances exactly like the live ASR listener."""
        out = []
        while True:
            try:
                out.append(self._utterances.get_nowait())
            except queue.Empty:
                return out

    def pop_metrics(self) -> list[dict]:
        """Drain scripted speech timing summaries."""
        out = []
        while True:
            try:
                out.append(self._metrics.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        """Replay listeners hold no external resources."""

    def mark_agent_spoke(self, timestamp: float | None = None) -> None:
        """Accept the live listener turn-timing hook for interface parity."""


class ReplayAudioProducer:
    """Publish synchronized WAV fixtures onto the production AudioBus."""
    def __init__(self, bus, base_path: str | Path = "config"):
        self.bus = bus
        self.base_path = Path(base_path)

    def feed_context(self, ctx) -> None:
        """Decode due WAV fixtures in memory and publish 16 kHz float samples."""
        for event in ctx.extras.get("replay_channels", {}).get("audio", []):
            wav_path = event.get("wav")
            if wav_path:
                path = (self.base_path / str(wav_path)).resolve()
                with wave.open(str(path), "rb") as stream:
                    channels, width, rate = stream.getnchannels(), stream.getsampwidth(), stream.getframerate()
                    raw = stream.readframes(stream.getnframes())
                if width != 2:
                    raise ValueError("replay WAV fixtures must be 16-bit PCM")
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                if rate != 16000:
                    old = np.arange(len(samples), dtype=float)
                    new = np.linspace(0, max(0, len(samples) - 1),
                                      max(1, int(len(samples) * 16000 / rate)))
                    samples = np.interp(new, old, samples).astype(np.float32)
            elif event.get("synthetic"):
                rng = np.random.default_rng(int(event.get("seed", 1)))
                samples = rng.normal(0, .02, 8000).astype(np.float32)
                samples *= np.exp(-np.linspace(0, 6, len(samples))).astype(np.float32)
            else:
                continue
            for start in range(0, len(samples), 1600):
                self.bus.publish(samples[start:start + 1600], ctx.timestamp + start / 16000)
