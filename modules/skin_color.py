"""Skin color screening: pallor, flushing, cyanosis, jaundice.

Method: measure cheek skin in a white-balanced, illumination-normalized
way. We sample the face-skin mask, convert to normalized chromaticity
(r,g,b divided by intensity) so overall brightness/lighting cancels, and
learn a rolling personal baseline over the first ~30 s. Deviations from
baseline in specific directions map to indicators:
  - lower redness + higher paleness  -> pallor
  - higher redness                   -> flushing
  - lips bluish (low R, high B)       -> cyanosis
  - sclera/skin yellow (high R+G,B low, in normalized space) -> jaundice tint

Reliability: LOW-MEDIUM and highly dependent on white balance and lighting.
Emitted as WARNING/NOTICE indicators, never diagnoses. A fixed, color-
calibrated light source dramatically improves this; without one, treat as
"worth a human check" only.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import face_skin_mask, roi_patch
from extractors import face_landmarks as FL


def _norm_chroma(bgr_pixels: np.ndarray) -> np.ndarray:
    """Mean normalized (r,g,b) chromaticity of a set of BGR pixels."""
    px = bgr_pixels.reshape(-1, 3).astype(np.float32)
    b, g, r = px[:, 0], px[:, 1], px[:, 2]
    s = (r + g + b) + 1e-6
    return np.array([r.mean() / s.mean(), g.mean() / s.mean(), b.mean() / s.mean()])


class _Baseline:
    def __init__(self, learn_seconds=30.0, alpha=0.02):
        self.learn_seconds = learn_seconds
        self.alpha = alpha
        self.mean = None
        self.t0 = None

    def update(self, sample, timestamp):
        if self.t0 is None:
            self.t0, self.mean = timestamp, sample.copy()
            return False
        learning = (timestamp - self.t0) < self.learn_seconds
        a = 0.1 if learning else self.alpha
        self.mean = (1 - a) * self.mean + a * sample
        return not learning


@register("skin_color")
class SkinColor(DetectionModule):
    interval = 1.0
    requires = ("face",)

    def __init__(self, **params):
        super().__init__(**params)
        self.base = _Baseline()

    def process(self, ctx: FrameContext):
        mask = face_skin_mask(ctx)
        if mask is None or mask.sum() < 500:
            return None
        skin = ctx.frame[mask > 0]
        chroma = _norm_chroma(skin)          # [r, g, b] normalized
        ready = self.base.update(chroma, ctx.timestamp)
        if not ready:
            return None

        d = chroma - self.base.mean          # deviation from personal baseline
        results = []

        # Redness axis (r relative to g+b)
        redness = d[0] - 0.5 * (d[1] + d[2])
        if redness < -0.02:
            results.append(self.result(
                "pallor", round(float(-redness), 3), min(1.0, -redness * 20),
                Severity.WARNING, "Skin looks paler than baseline (possible pallor)",
                ttl=15.0))
        elif redness > 0.025:
            results.append(self.result(
                "flushing", round(float(redness), 3), min(1.0, redness * 20),
                Severity.NOTICE, "Facial flushing / redness above baseline",
                ttl=15.0))

        # Jaundice: yellow = high r & g, low b (in normalized space)
        yellow = 0.5 * (d[0] + d[1]) - d[2]
        if yellow > 0.03:
            results.append(self.result(
                "jaundice_tint", round(float(yellow), 3), min(0.8, yellow * 15),
                Severity.WARNING, "Yellowish skin tint vs baseline (verify white balance)",
                ttl=20.0))

        # Cyanosis: measured on the lips, which turn bluish with low oxygen
        lips = roi_patch(ctx, FL.MOUTH_BOTTOM_INNER, radius_frac=0.06)
        if lips is not None and lips.size:
            lc = _norm_chroma(lips)
            bluish = lc[2] - lc[0]           # blue minus red on the lip
            if bluish > 0.08:
                results.append(self.result(
                    "cyanosis", round(float(bluish), 3), min(0.8, bluish * 6),
                    Severity.ALERT, "Bluish lip tint (possible low oxygen — check on person)",
                    ttl=15.0))
        return results or None
