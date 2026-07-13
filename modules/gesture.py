"""Hand gesture recognition (MediaPipe GestureRecognizer).

Method: the MediaPipe Tasks GestureRecognizer (hand landmarks + a gesture
classifier head) run in VIDEO mode on an interval, like the other optional-
model modules. Canonical labels: Closed_Fist, Open_Palm, Pointing_Up,
Thumb_Up, Thumb_Down, Victory, ILoveYou. On top of the per-frame label we
derive **waving** — an Open_Palm whose hand center oscillates horizontally
— because a wave is the single most natural way a showcase visitor opens
interaction with a robot, and no static label captures it.

Self-disables cleanly when models/gesture_recognizer.task is absent
(mirroring modules/age_estimation.py), so the pipeline runs unchanged
without the download.

Reliability: HIGH for the static labels with the hand in view (this is one
of MediaPipe's most mature models); waving is a heuristic on top, MEDIUM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer

_MODEL = Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"
_MESSAGES = {
    "Thumb_Up": "Thumbs up",
    "Thumb_Down": "Thumbs down",
    "Open_Palm": "Open palm raised",
    "Victory": "Victory sign",
    "Pointing_Up": "Pointing up",
    "ILoveYou": "'I love you' sign",
    "Closed_Fist": "Closed fist raised",
}


@register("gesture")
class Gesture(DetectionModule):
    """Hand gesture recognition via MediaPipe GestureRecognizer."""
    interval = 0.25              # classifier head is light but not free
    requires = ("person",)
    min_confidence = 0.5
    wave_window_seconds = 2.0
    wave_crossings = 3           # direction reversals that make a wave
    wave_span_ratio = 0.04       # min horizontal travel, fraction of frame width

    def __init__(self, **params):
        super().__init__(**params)
        self.recognizer = None
        self._wave_buf = TimedBuffer(self.wave_window_seconds)
        self._last_ts_ms = -1
        if _MODEL.exists():
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
                opts = vision.GestureRecognizerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                )
                self.recognizer = vision.GestureRecognizer.create_from_options(opts)
            except Exception as e:  # noqa: BLE001
                print(f"[gesture] GestureRecognizer load failed ({e}); disabled")
        else:
            print("[gesture] no model at models/gesture_recognizer.task; disabled. "
                  "Download: https://storage.googleapis.com/mediapipe-models/"
                  "gesture_recognizer/gesture_recognizer/float16/latest/"
                  "gesture_recognizer.task")

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        if self.recognizer is None:
            return None
        import mediapipe as mp
        rgb = np.ascontiguousarray(ctx.frame[:, :, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(ctx.timestamp * 1000)
        if ts_ms <= self._last_ts_ms:      # VIDEO mode needs monotonic stamps
            return None
        self._last_ts_ms = ts_ms
        out = self.recognizer.recognize_for_video(mp_img, ts_ms)
        if not out.gestures:
            self._wave_buf = TimedBuffer(self.wave_window_seconds)
            return None

        results = []
        top = out.gestures[0][0]           # best gesture of the first hand
        label, score = top.category_name, float(top.score)
        if label in _MESSAGES and score >= self.min_confidence:
            results.append(self.result(
                "gesture", label, score, Severity.NOTICE,
                _MESSAGES[label], ttl=3.0))

        # Waving: Open_Palm whose hand center swings back and forth.
        if label == "Open_Palm" and out.hand_landmarks:
            xs = [p.x for p in out.hand_landmarks[0]]
            self._wave_buf.push(ctx.timestamp, float(np.mean(xs)))
            if self._wave_buf.span() > 0.8:
                _, v = self._wave_buf.arrays()
                span = float(np.max(v) - np.min(v))
                centered = v - np.mean(v)
                crossings = int(np.sum(np.diff(np.sign(centered)) != 0))
                if span > self.wave_span_ratio and crossings >= self.wave_crossings:
                    results.append(self.result(
                        "waving", True, min(0.85, score), Severity.NOTICE,
                        "Waving hello", ttl=3.0))
        else:
            self._wave_buf = TimedBuffer(self.wave_window_seconds)
        return results or None
