"""Facial / eyelid swelling (edema) screening via slow contour drift.

Method: track normalized eye-opening height and cheek fullness (cheek-to-
jaw width ratio) against a long rolling baseline. Puffiness reduces eye
aperture and increases lower-face fullness. Only slow, sustained changes
are reported (fast changes are expression, not edema).

Each frame's raw features are first temporally pooled over a short window
(`pooled_skin_sample`, modules/_util.py) before feeding the drift math --
at small face sizes (e.g. a low-resolution/distant webcam) landmark
placement jitters by a pixel or two frame-to-frame, which is otherwise a
larger fraction of these already-small eye-opening/cheek-width ratios.

With depth (RealSense D435i) a third, volumetric feature joins the drift
vector: the mean cheek/periorbital surface depth relative to the nose tip.
Real puffiness pushes those surfaces toward the camera (millimeters the
2D contour ratios can only infer indirectly), and depth separates it from
expression far better than geometry alone. The baseline resets when the
modality flips (camera switch) because the feature vectors differ.

Reliability: LOW on RGB; MEDIUM with depth. Longitudinal drift indicator.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, pooled_skin_sample
from extractors import face_landmarks as FL


@register("facial_swelling")
class FacialSwelling(DetectionModule):
    """Facial / eyelid swelling (edema) screening via slow contour drift."""
    interval = 2.0
    requires = ("face",)
    smoothing_seconds = 6.0   # short pooling window; damps small-face landmark jitter

    def __init__(self, **params):
        super().__init__(**params)
        self.base = None
        self.t0 = None
        self.feat_buf = TimedBuffer(self.smoothing_seconds)
        self._depth_mode = False

    def _protrusion(self, ctx, px):
        """Volumetric proxy: mean cheek + periorbital surface depth relative
        to the nose tip, in meters. Shrinks as puffiness pushes those
        surfaces toward the camera. None on depth holes."""
        nose_d = ctx.depth_m(px[FL.NOSE_TIP][0], px[FL.NOSE_TIP][1])
        if nose_d is None:
            return None
        ds = []
        for idx in (FL.LEFT_CHEEK, FL.RIGHT_CHEEK, 145, 374):  # cheeks + under-eyes
            d = ctx.depth_m(px[idx][0], px[idx][1])
            if d is None:
                return None
            ds.append(d - nose_d)
        return float(np.mean(ds))

    def _features(self, ctx):
        px = ctx.face_px()
        face_w = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
        eye_l = np.linalg.norm(px[159] - px[145]) / face_w
        eye_r = np.linalg.norm(px[386] - px[374]) / face_w
        cheek_w = np.linalg.norm(px[FL.LEFT_CHEEK] - px[FL.RIGHT_CHEEK]) / face_w
        feats = [(eye_l + eye_r) / 2.0, cheek_w]
        if ctx.depth is not None:
            protrusion = self._protrusion(ctx, px)
            if protrusion is None:
                return None                 # skip frame; don't mix modalities
            feats.append(protrusion)
        return np.array(feats)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        raw_f = self._features(ctx)
        if raw_f is None:
            return None
        depth_mode = len(raw_f) == 3
        if depth_mode != self._depth_mode:  # camera switched: feature vector changed
            self._depth_mode = depth_mode
            self.base, self.t0 = None, None
            self.feat_buf = TimedBuffer(self.smoothing_seconds)
        f = pooled_skin_sample(self.feat_buf, raw_f, ctx.timestamp)
        if self.t0 is None:
            self.t0, self.base = ctx.timestamp, f
            return None
        if ctx.timestamp - self.t0 < 30.0:      # long learning window
            self.base = 0.9 * self.base + 0.1 * f
            return None
        self.base = 0.999 * self.base + 0.001 * f
        eye_drop = (self.base[0] - f[0]) / (self.base[0] + 1e-6)
        cheek_gain = (f[1] - self.base[1]) / (self.base[1] + 1e-6)
        if len(f) == 3:
            # Volumetric term: cheeks/under-eyes moved toward the camera
            # relative to the nose. ~3 mm of protrusion ≈ full weight.
            protrusion_gain = max(0.0, (self.base[2] - f[2]) / 0.003)
            score = (0.3 * max(0, eye_drop) + 0.3 * max(0, cheek_gain)
                     + 0.4 * min(1.0, protrusion_gain) * 0.1)
        else:
            score = 0.5 * max(0, eye_drop) + 0.5 * max(0, cheek_gain)
        if score < 0.08:
            return None
        conf = float(min(0.5, score * 3))
        return self.result(
            "swelling", round(float(score), 3), conf, Severity.NOTICE,
            "Possible facial/eyelid puffiness vs baseline (screening only)",
            ttl=30.0)
