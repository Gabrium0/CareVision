"""Facial expressivity / flat-affect screening (GDS-15-adjacent cues).

Method: action-unit-style analysis validated against depression scores in
seniors uses reduced smiling, flattened expression variance, and blink-rate
changes. We approximate those with landmark-only features (no new model):

- **smile level** — mouth-corner span / face width; rises when smiling.
  Frames with a wide-open mouth (talking/yawning, MAR above a gate) are
  skipped so speech doesn't read as expression.
- **expression variance** — rolling std of the smile level: a lively face
  moves, a flat one doesn't. This is the core "flattened affect" cue.
- **blink rate** — EAR dips below a fraction of its personal baseline
  (same measure modules/drowsiness.py uses).

Each window's aggregates are persisted to storage/history_store.py, and
`expressivity_low` is raised only when the current window sits well below
the person's own 30-day history for several consecutive windows — a
cross-visit change screen, never a mood label. The agent phrases it as a
check-in ("you seem quieter than usual") via agent/corroboration.py, and
only says more if the person confirms.

Reliability: LOW-MEDIUM as an instantaneous signal; its value is the
longitudinal comparison. Single-known-user assumption (no face re-ID).
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, mouth_aspect_ratio
from modules.drowsiness import _ear
from storage.history_store import HistoryStore
from extractors import face_landmarks as FL


@register("expressivity")
class Expressivity(DetectionModule):
    """Expressivity / flat-affect screening from smile, variance, blinks."""
    interval = 0.0
    requires = ("face",)
    window_seconds = 60.0
    talk_mar_gate = 0.35         # skip frames with a wide-open mouth
    blink_drop_ratio = 0.75      # EAR below this fraction of baseline = blink
    store_every = 180.0          # seconds between HistoryStore samples
    history_days = 30.0
    low_ratio = 0.85             # window must sit below this x history
    low_hits_needed = 3          # consecutive low windows before a NOTICE

    def __init__(self, **params):
        super().__init__(**params)
        self.smile_buf = TimedBuffer(self.window_seconds)
        self.blink_buf = TimedBuffer(self.window_seconds)   # blink events
        self.ear_baseline = None
        self._eye_closed = False
        self._last_store = 0.0
        self._low_hits = 0
        # instance() singleton, overridable in tests (see tests/grooming_test.py)
        self.store = HistoryStore.instance()

    def _features(self, ctx) -> float | None:
        """Current smile level, or None on talking/yawning frames."""
        mar = mouth_aspect_ratio(ctx)
        if mar is None or mar > self.talk_mar_gate:
            return None
        px = ctx.face_px()
        face_w = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        return float(np.linalg.norm(px[FL.MOUTH_LEFT] - px[FL.MOUTH_RIGHT]) / face_w)

    def _track_blinks(self, ctx) -> None:
        px = ctx.face_px()
        ear = 0.5 * (_ear(px, FL.LEFT_EYE_EAR) + _ear(px, FL.RIGHT_EYE_EAR))
        self.ear_baseline = (ear if self.ear_baseline is None
                             else 0.99 * self.ear_baseline + 0.01 * ear)
        closed = ear < self.blink_drop_ratio * self.ear_baseline
        if closed and not self._eye_closed:
            self.blink_buf.push(ctx.timestamp, 1.0)
        self._eye_closed = closed

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        self._track_blinks(ctx)
        smile = self._features(ctx)
        if smile is not None:
            self.smile_buf.push(ctx.timestamp, smile)
        if self.smile_buf.span() < 20.0:
            return None

        _, smiles = self.smile_buf.arrays()
        smile_mean = float(np.mean(smiles))
        expr_std = float(np.std(smiles))
        blink_rate = len(self.blink_buf) * 60.0 / max(self.blink_buf.seconds, 1.0)
        results = [
            self.result("smile_level", round(smile_mean, 3), 0.5,
                        Severity.INFO, "", ttl=15.0),
            self.result("expression_variance", round(expr_std, 4), 0.5,
                        Severity.INFO, "", ttl=15.0),
            self.result("blink_rate", round(blink_rate, 0), 0.5,
                        Severity.INFO, "", ttl=15.0),
        ]

        if ctx.timestamp - self._last_store >= self.store_every:
            self._last_store = ctx.timestamp
            store = self.store
            day_seconds = self.history_days * 86400.0
            mean = getattr(store, "rolling_mean", None) or store.mean_since
            hist_smile = mean("expressivity", "smile_mean", day_seconds)
            hist_std = mean("expressivity", "expr_std", day_seconds)
            store.add("expressivity", "smile_mean", smile_mean, ts=ctx.timestamp)
            store.add("expressivity", "expr_std", expr_std, ts=ctx.timestamp)
            store.add("expressivity", "blink_rate", blink_rate, ts=ctx.timestamp)
            low = (hist_smile is not None and hist_std is not None
                   and smile_mean < self.low_ratio * hist_smile
                   and expr_std < self.low_ratio * hist_std)
            self._low_hits = self._low_hits + 1 if low else 0
            if self._low_hits >= self.low_hits_needed:
                results.append(self.result(
                    "expressivity_low", round(smile_mean, 3), 0.45,
                    Severity.NOTICE,
                    "Expressivity looks lower than their usual baseline "
                    "(check-in cue, not a mood assessment)", ttl=120.0))
        return results
