"""Metric height and distance of the subject (depth-only module).

Method: with aligned depth + intrinsics (RealSense D435i), deproject the
pose nose and ankle landmarks to 3D camera-frame points. Distance is the
median torso depth; standing height is the vertical span nose-to-ankles
plus fixed anatomical offsets for nose-to-crown (~12 cm) and ankle-to-sole
(~8 cm). Smoothed with an EMA because single-frame depth at landmarks is
noisy.

Requires `("pose", "depth")`, so the scheduler simply never runs it on an
RGB-only source — the graceful-degrade contract of the "depth" token.

Reliability: distance HIGH (that's what a depth camera measures); height
MEDIUM (~±5 cm) and only meaningful when the person stands upright with
ankles in view.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import pose as P

_NOSE_TO_CROWN_M = 0.12
_ANKLE_TO_SOLE_M = 0.08


@register("height_distance")
class HeightDistance(DetectionModule):
    """Metric subject height + distance from deprojected pose landmarks."""
    interval = 0.5
    requires = ("pose", "depth")
    min_visibility = 0.5
    ema = 0.2                    # per-update weight of the newest estimate

    def __init__(self, **params):
        super().__init__(**params)
        self._height = None
        self._distance = None

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        px = ctx.pose_px()
        results = []

        # Distance: median depth across the torso landmarks that are visible.
        torso = [i for i in (P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP)
                 if lm[i, 3] >= self.min_visibility]
        depths = [d for i in torso
                  if (d := ctx.depth_m(px[i][0], px[i][1])) is not None]
        if depths:
            dist = float(np.median(depths))
            self._distance = (dist if self._distance is None else
                              (1 - self.ema) * self._distance + self.ema * dist)
            results.append(self.result(
                "distance_m", round(self._distance, 2), 0.8, Severity.INFO,
                f"Standing ~{self._distance:.1f} m away", ttl=4.0))

        # Height: needs head and at least one ankle, person roughly upright.
        if lm[P.NOSE, 3] >= self.min_visibility:
            head = ctx.deproject(px[P.NOSE][0], px[P.NOSE][1])
            ankles = [a for i in (P.L_ANKLE, P.R_ANKLE)
                      if lm[i, 3] >= self.min_visibility
                      and (a := ctx.deproject(px[i][0], px[i][1])) is not None]
            if head is not None and ankles:
                foot_y = float(np.mean([a[1] for a in ankles]))
                # Camera y grows downward; vertical extent is foot_y - head_y.
                extent = foot_y - float(head[1])
                if extent > 0.8:            # rules out sitting/crouching poses
                    h = extent + _NOSE_TO_CROWN_M + _ANKLE_TO_SOLE_M
                    self._height = (h if self._height is None else
                                    (1 - self.ema) * self._height + self.ema * h)
                    results.append(self.result(
                        "height_m", round(self._height, 2), 0.6, Severity.INFO,
                        f"Height ~{self._height:.2f} m", ttl=6.0))
        return results or None
