"""Regression: FaceExtractor must feed MediaPipe strictly-increasing millisecond
timestamps even when ctx.timestamp repeats or goes backwards.

The iPad capture clock keeps float timestamps monotonic only by a 0.1ms nudge,
which collapses to equal/lower integer milliseconds during reconnect resync
bursts. MediaPipe's VIDEO mode then raised
``ValueError: Input timestamp must be monotonically increasing`` on every frame,
the face worker swallowed it, and ctx.face went permanently None — silencing the
yawn/blink/head-nod detectors. FaceExtractor now clamps the ms it passes down.
"""
from __future__ import annotations

import types

import numpy as np

import extractors.face as face_mod
from core.context import FrameContext
from extractors.face import FaceExtractor


class _StubLandmarker:
    """Mimics MediaPipe: rejects a timestamp that is not strictly increasing."""

    def __init__(self):
        self.calls: list[int] = []
        self._last: int | None = None

    def detect_for_video(self, _image, ts_ms):
        if self._last is not None and ts_ms <= self._last:
            raise ValueError("Input timestamp must be monotonically increasing.")
        self._last = ts_ms
        self.calls.append(ts_ms)
        return types.SimpleNamespace(face_landmarks=[])  # no face -> extract returns early

    def close(self):
        pass


def _extractor(monkeypatch) -> tuple[FaceExtractor, _StubLandmarker]:
    stub = _StubLandmarker()
    fake_model = types.SimpleNamespace(exists=lambda: True)
    monkeypatch.setattr(face_mod, "_MODEL", fake_model)
    monkeypatch.setattr(face_mod.vision.FaceLandmarker, "create_from_options",
                        staticmethod(lambda _opts: stub))
    return FaceExtractor(smooth=False), stub


def _feed(extractor: FaceExtractor, ts: float) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    extractor.extract(FrameContext(frame=frame, timestamp=ts, frame_index=0, fps=20.0))


def test_repeated_and_backwards_timestamps_never_raise(monkeypatch):
    extractor, stub = _extractor(monkeypatch)
    # Equal ms (1.000, 1.000), a sub-ms nudge (1.0004 -> still 1000ms), a big
    # backwards jump (reconnect resync), then forward again.
    for ts in (1.000, 1.000, 1.0004, 0.500, 2.000):
        _feed(extractor, ts)   # must not raise
    assert all(b > a for a, b in zip(stub.calls, stub.calls[1:])), stub.calls
    assert len(stub.calls) == 5


def test_reset_keeps_the_monotonic_guard(monkeypatch):
    # reset() (a source switch) must not let the ms counter go backwards: the
    # same landmarker keeps its internal clock, so our guard must persist too.
    extractor, stub = _extractor(monkeypatch)
    _feed(extractor, 5.000)
    extractor.reset()
    _feed(extractor, 1.000)      # lower wall time after the switch
    assert stub.calls[-1] > stub.calls[0], stub.calls
