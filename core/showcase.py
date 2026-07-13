"""Stationary D435i guest-zone and capture-quality policy.

This is deliberately a presentation/measurement gate, not a diagnostic
classifier.  It writes a small, UI-safe state into ``FrameContext.extras``
and prevents modules from publishing measurements outside the framing in
which they are meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context import FrameContext
from .events import Result, Severity


@dataclass
class ShowcaseGate:
    enabled: bool = True
    conversation_min_m: float = 0.8
    conversation_max_m: float = 1.5
    movement_min_m: float = 2.5
    movement_max_m: float = 3.2
    min_face_px: int = 110
    min_brightness: float = 45.0
    max_brightness: float = 220.0
    max_motion: float = 12.0

    # Face/physiology estimates require the close, stable conversation zone.
    conversation_modules = frozenset({"heart_rate", "respiration", "facial_asymmetry",
                                      "facial_swelling", "skin_color", "rash", "bruise",
                                      "eye_redness", "sweating", "dry_lips"})
    # Whole-body movement estimates are demonstrated only at the movement marker.
    movement_modules = frozenset({"balance", "gait", "fall", "tremor", "bradykinesia"})

    @classmethod
    def from_config(cls, config: dict | None) -> "ShowcaseGate":
        cfg = (config or {}).get("showcase", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if hasattr(cls, k)})

    @staticmethod
    def _distance(ctx: FrameContext) -> float | None:
        if ctx.depth is None:
            return None
        if ctx.pose is not None:
            px, lm = ctx.pose_px(), ctx.pose.landmarks
            values = [ctx.depth_m(px[i][0], px[i][1]) for i in (11, 12, 23, 24)
                      if lm[i, 3] >= 0.5]
            values = [v for v in values if v is not None]
            if values:
                return float(np.median(values))
        if ctx.face is not None:
            x1, y1, x2, y2 = ctx.face.bbox
            return ctx.depth_m((x1 + x2) / 2, (y1 + y2) / 2)
        return None

    def assess(self, ctx: FrameContext) -> list[Result]:
        if not self.enabled:
            return []
        distance = self._distance(ctx)
        face_px = 0 if ctx.face is None else min(ctx.face.bbox[2] - ctx.face.bbox[0],
                                                  ctx.face.bbox[3] - ctx.face.bbox[1])
        # BGR luma approximation; avoids making this policy layer depend on OpenCV.
        brightness = float(np.dot(ctx.frame[..., :3], (0.114, 0.587, 0.299)).mean())
        face_count = int(ctx.extras.get("face_count", 0))
        pose_count = int(ctx.extras.get("pose_count", 0))
        multiple = face_count > 1 or pose_count > 1
        zone = "outside"
        if distance is not None and self.conversation_min_m <= distance <= self.conversation_max_m:
            zone = "conversation"
        elif distance is not None and self.movement_min_m <= distance <= self.movement_max_m:
            zone = "movement"
        stable = (zone == "conversation" and not multiple and ctx.face is not None
                  and face_px >= self.min_face_px
                  and self.min_brightness <= brightness <= self.max_brightness
                  and ctx.motion_energy <= self.max_motion)
        if multiple:
            guidance = "Please step forward one at a time."
        elif distance is None:
            guidance = "Depth is unavailable; please remain in front of me."
        elif distance < self.conversation_min_m:
            guidance = "Please move back slightly to the conversation marker."
        elif zone == "outside":
            guidance = "Please step onto the conversation marker about 1.2 m away."
        elif zone == "movement":
            guidance = "Movement zone ready for a walking or balance demonstration."
        elif face_px < self.min_face_px:
            guidance = "Please face me so I can see you clearly."
        elif not self.min_brightness <= brightness <= self.max_brightness:
            guidance = "Lighting is not suitable; please use the marked well-lit position."
        elif ctx.motion_energy > self.max_motion:
            guidance = "Please hold still for a moment."
        else:
            guidance = "Conversation zone ready."
        ctx.extras["showcase"] = {"zone": zone, "stable": stable, "distance_m": distance,
                                   "brightness": brightness, "multiple": multiple,
                                   "guidance": guidance}
        return [Result("showcase", "zone", zone, 1.0, Severity.INFO,
                       f"Showcase zone: {zone}", ttl=2.0),
                Result("showcase", "capture_ready", stable, 1.0, Severity.INFO,
                       guidance, ttl=2.0)]

    def allow(self, module_name: str, ctx: FrameContext) -> bool:
        if not self.enabled:
            return True
        state = ctx.extras.get("showcase", {})
        if module_name in self.conversation_modules:
            return bool(state.get("stable"))
        if module_name in self.movement_modules:
            return state.get("zone") == "movement" and not state.get("multiple")
        return True
