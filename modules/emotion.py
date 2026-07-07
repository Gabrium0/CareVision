"""Emotion / mood recognition — multi-backend.

Runs several emotion backends together and reports each one's label so the
hand-rolled heuristic and the tested models show side by side (like the rPPG
backends). Configure in config/modules.yaml:

    emotion:
      backends: [heuristic, hsemotion, deepface]

Backends that lack their dependency/weights self-disable, so this always at
least runs the heuristic. HSEmotion additionally provides a continuous
valence ("mood") used by the greeting engine.
"""
from __future__ import annotations

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules.emotion_backends.heuristic import HeuristicEmotionBackend
from modules.emotion_backends.ferplus import FerPlusBackend
from modules.emotion_backends.hsemotion import HSEmotionBackend
from modules.emotion_backends.deepface_backend import DeepFaceBackend

_BACKENDS = {
    "heuristic": HeuristicEmotionBackend,
    "ferplus": FerPlusBackend,
    "hsemotion": HSEmotionBackend,
    "deepface": DeepFaceBackend,
}


@register("emotion")
class Emotion(DetectionModule):
    interval = 0.4
    requires = ("face",)
    backends = ["heuristic"]          # overridden by config

    def __init__(self, **params):
        super().__init__(**params)
        self._backends = []
        for name in self.backends:
            cls = _BACKENDS.get(name)
            if cls is None:
                print(f"[emotion] unknown backend '{name}', skipping")
                continue
            inst = cls()
            if getattr(inst, "available", True):
                self._backends.append(inst)
        if not self._backends:
            self._backends.append(HeuristicEmotionBackend())

    def process(self, ctx: FrameContext):
        results = []
        for be in self._backends:
            be.update(ctx)
            reading = be.compute()
            if not reading:
                continue
            label = be.label
            emo = reading.get("emotion")
            conf = float(reading.get("confidence", 0.4))
            if emo is not None:
                results.append(self.result(
                    f"emotion_{label}", emo, conf, Severity.INFO,
                    f"Emotion ({label}): {emo}", ttl=3.0))
            if "valence" in reading:
                v = reading["valence"]
                mood = "positive" if v > 0.15 else ("negative" if v < -0.15 else "neutral")
                results.append(self.result(
                    f"valence_{label}", round(v, 2), conf, Severity.INFO,
                    f"Mood ({label}): {mood} ({v:+.2f})", ttl=3.0))
        return results or None

    def close(self):
        for be in self._backends:
            be.close()
