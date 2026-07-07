"""Hazard-zone entry detection (stove, stairs, doorway).

Method: configured with named polygons in normalized (0..1) image
coordinates. When the person's foot position (ankle midpoint, or bbox
bottom center as fallback) enters a zone, an event is raised. Zones are
scene-specific and must be calibrated per camera install.

Reliability: MEDIUM given correct calibration; disabled by default.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from extractors import pose as P


@register("hazard_zones")
class HazardZones(DetectionModule):
    interval = 0.3
    requires = ("pose",)
    zones = []          # [{name, polygon:[[x,y],...] normalized}]

    def __init__(self, **params):
        super().__init__(**params)
        self._polys = [(z["name"], np.array(z["polygon"], dtype=np.float32))
                       for z in (self.zones or [])]
        self._inside = set()

    def process(self, ctx: FrameContext):
        if not self._polys:
            return None
        lm = ctx.pose.landmarks
        if lm[P.L_ANKLE, 3] > 0.4 and lm[P.R_ANKLE, 3] > 0.4:
            foot = np.array([(lm[P.L_ANKLE, 0] + lm[P.R_ANKLE, 0]) / 2.0,
                             (lm[P.L_ANKLE, 1] + lm[P.R_ANKLE, 1]) / 2.0])
        else:
            x1, y1, x2, y2 = ctx.pose.bbox
            foot = np.array([(x1 + x2) / 2.0 / ctx.w, y2 / ctx.h])
        results = []
        for name, poly in self._polys:
            inside = cv2.pointPolygonTest(poly, (float(foot[0]), float(foot[1])), False) >= 0
            if inside and name not in self._inside:
                self._inside.add(name)
                results.append(self.result(
                    "zone_entry", name, 0.7, Severity.WARNING,
                    f"Entered hazard zone: {name}", ttl=8.0))
            elif not inside:
                self._inside.discard(name)
        return results or None
