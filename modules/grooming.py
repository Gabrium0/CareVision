"""Grooming / hygiene appearance drift (longitudinal, self-neglect proxy).

Method: track two coarse face-crop texture proxies each interval -- edge
density in the head-top/hairline band (unkempt hair) and in the jaw/beard
band (facial-hair growth) -- and log them to the persistent history store
(storage/history_store.py), the same pattern modules/activity_level.py
uses for its motion trend. Compares today's mean against the trailing
week's mean per feature: a sustained rise (messier hair, heavier stubble
than usual for this person) flags a possible grooming/hygiene change worth
a caregiver glance.

This is deliberately coarse and slow-moving -- it says nothing meaningful
on day one, and routine events (a haircut, growing a beard on purpose)
will trigger it same as neglect would. Treat purely as a screening prompt,
not a diagnosis, and expect it to need days to weeks of runtime before the
weekly baseline is trustworthy.

Reliability: LOW; longitudinal drift indicator only, needs history.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import face_landmarks as FL
from storage.history_store import HistoryStore


def _edge_density(patch: np.ndarray | None) -> float | None:
    """Coarse texture/edge-energy proxy (mean abs Laplacian) of a BGR patch."""
    if patch is None or patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))))


@register("grooming")
class Grooming(DetectionModule):
    """Grooming / hygiene appearance drift (longitudinal, self-neglect proxy)."""
    interval = 30.0
    requires = ("face",)
    recent_seconds = 24 * 60 * 60          # trailing "today"
    baseline_seconds = 7 * 24 * 60 * 60    # trailing week
    rise_ratio = 1.5                       # today vs week mean that flags a change

    def __init__(self, **params):
        super().__init__(**params)
        self.store = HistoryStore.instance()
        rolling = getattr(self.store, "rolling_mean", None)
        if rolling is not None:
            for key in ("hair_texture", "jaw_texture"):
                rolling("grooming", key, self.recent_seconds)
                rolling("grooming", key, self.baseline_seconds)

    def _rois(self, ctx: FrameContext):
        """(hair_patch, jaw_patch) cropped from the face bbox/landmarks, or
        None where the region falls outside the frame."""
        x1, y1, x2, y2 = ctx.face.bbox
        face_h = max(1, y2 - y1)
        hair = ctx.frame[max(0, y1 - int(0.4 * face_h)):y1, x1:x2]

        px = ctx.face_px()
        chin_y = int(px[FL.CHIN][1])
        jaw = ctx.frame[chin_y:min(ctx.h, chin_y + int(0.18 * face_h)), x1:x2]
        return hair, jaw

    def _drift_result(self, key: str, label: str):
        """A Result if `key`'s recent mean has risen vs its weekly baseline, else None."""
        mean = getattr(self.store, "rolling_mean", None) or self.store.mean_since
        recent = mean("grooming", key, self.recent_seconds)
        baseline = mean("grooming", key, self.baseline_seconds)
        if recent is None or baseline is None or baseline < 1e-3:
            return None
        ratio = recent / baseline
        if ratio < self.rise_ratio:
            return None
        conf = float(min(0.5, 0.15 + (ratio - self.rise_ratio) * 0.3))
        return self.result(
            f"{key}_change", round(ratio, 2), conf, Severity.NOTICE,
            f"{label.capitalize()} looks more textured/unkempt than the usual "
            "weekly average (screening only)", ttl=6 * 60 * 60)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        hair_patch, jaw_patch = self._rois(ctx)
        hair_tex = _edge_density(hair_patch)
        jaw_tex = _edge_density(jaw_patch)
        if hair_tex is not None:
            self.store.add("grooming", "hair_texture", hair_tex, ctx.timestamp)
        if jaw_tex is not None:
            self.store.add("grooming", "jaw_texture", jaw_tex, ctx.timestamp)

        results = [r for r in (self._drift_result("hair_texture", "hair"),
                               self._drift_result("jaw_texture", "facial hair"))
                  if r is not None]
        return results or None
