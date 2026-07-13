"""Facial asymmetry / droop screening (stroke-relevant, FAST 'F').

Method: using symmetric landmark pairs, mirror the face about its vertical
midline (defined by nose-tip to chin) and measure how far each left point
sits from its mirrored right counterpart, normalized by face width. Sudden
or sustained one-sided droop (especially mouth/eye) raises an alert.

Landmark pairs are grouped into regions (mouth, eye, brow, cheek/edge)
instead of averaged into one global number: an isolated mouth or eye droop
-- the FAST-relevant regions -- would otherwise be diluted by a symmetric
brow or an unrelated cheek measurement, and vice versa. Each region tracks
its own rolling personal baseline and sustained-hit counter so naturally
asymmetric faces don't false-positive; we flag a *change*, per region.

With depth (RealSense D435i), the same regional measurement runs in 3D:
landmark pairs are deprojected to camera-frame points and mirrored across
the 3D mid-sagittal plane (through the 3D nose and chin, normal along the
face's left-right axis). This removes the head-yaw confound that inflates
2D asymmetry — a turned head foreshortens one side of the projected face,
which the 2D mirror reads as droop; in 3D the geometry is pose-invariant.
Baselines reset when the modality flips (camera switch) since the two
residuals aren't numerically identical.

Reliability: MEDIUM as a screen (better with depth); NOT a diagnosis.
Pairs well with a prompt to smile/raise arms if triggered.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from storage.history_store import HistoryStore
from extractors import face_landmarks as FL

# region name -> (landmark pairs, may escalate to ALERT). Mouth/eye are the
# FAST-relevant regions for stroke droop; brow and cheek/face-edge are
# lower-specificity and cap at WARNING.
_REGIONS = {
    "mouth": ([(61, 291)], True),
    "eye": ([(159, 386), (145, 374)], True),
    "brow": ([(105, 334)], False),
    "cheek_edge": ([(50, 280), (234, 454)], False),
}


@register("facial_asymmetry")
class FacialAsymmetry(DetectionModule):
    """Facial asymmetry / droop screening (stroke-relevant, FAST 'F')."""
    interval = 0.5
    requires = ("face",)
    learning_seconds = 20.0
    threshold = 0.05
    alert_threshold = 0.08
    alert_after_count = 3
    history_days = 30.0
    trend_margin = 0.03       # settled baseline this far above history = drift
    persist_history = True    # HistoryStore read/write (off in some tests)

    def __init__(self, **params):
        super().__init__(**params)
        self.baseline: dict = {}
        self.t0 = None
        self._hits = {name: 0 for name in _REGIONS}
        self._depth_mode = False
        self._persisted = False
        # instance() singleton, overridable in tests (see tests/grooming_test.py)
        self.store = HistoryStore.instance() if self.persist_history else None

    def _reset_baseline(self) -> None:
        """Restart baseline learning (used when the 2D/3D modality flips)."""
        self.baseline = {}
        self.t0 = None
        self._hits = {name: 0 for name in _REGIONS}
        self._persisted = False

    def _persist_and_trend(self, now: float) -> list:
        """Once per session, after the baseline settles: compare each region
        against the person's 30-day history (the gradual Bell's-palsy-type
        drift a single session can't see), then store today's values.
        Separate keys per 2D/3D modality — the residuals aren't comparable.
        Assumes a single known user (no face re-ID yet)."""
        self._persisted = True
        if not self.persist_history or self.store is None:
            return []
        results = []
        store = self.store
        mode = "3d" if self._depth_mode else "2d"
        window = self.history_days * 86400.0
        for name, value in self.baseline.items():
            key = f"base_{name}_{mode}"
            hist = store.mean_since("facial_asymmetry", key, window)
            if hist is not None and value - hist > self.trend_margin:
                results.append(self.result(
                    f"asymmetry_trend_{name}", round(value - hist, 3),
                    min(0.6, (value - hist) * 8), Severity.NOTICE,
                    f"Baseline {name} asymmetry is higher than their "
                    "historical baseline (gradual change; screening only)",
                    ttl=60.0))
            store.add("facial_asymmetry", key, value, ts=now)
        return results

    def _region_asym(self, px, nose, normal, face_w, pairs) -> float:
        """Mean mirrored-landmark deviation (normalized by face width) for
        one region's symmetry pairs."""
        devs = []
        for li, ri in pairs:
            lp, rp = px[li], px[ri]
            # reflect right point across the midline axis through nose
            v = rp - nose
            rp_mirror = nose + v - 2 * np.dot(v, normal) * normal
            devs.append(np.linalg.norm(lp - rp_mirror) / face_w)
        return float(np.mean(devs))

    def _region_asym_3d(self, ctx, px) -> dict | None:
        """Regional mirrored-landmark deviations in 3D camera space, or None
        when any needed deprojection hits a depth hole (skip the frame
        rather than mixing 2D values into 3D baselines)."""
        nose3 = ctx.deproject(px[FL.NOSE_TIP][0], px[FL.NOSE_TIP][1])
        chin3 = ctx.deproject(px[FL.CHIN][0], px[FL.CHIN][1])
        le3 = ctx.deproject(px[FL.LEFT_FACE_EDGE][0], px[FL.LEFT_FACE_EDGE][1])
        re3 = ctx.deproject(px[FL.RIGHT_FACE_EDGE][0], px[FL.RIGHT_FACE_EDGE][1])
        if any(p is None for p in (nose3, chin3, le3, re3)):
            return None
        axis = chin3 - nose3                       # midline, in-plane
        n = np.linalg.norm(axis)
        face_w = np.linalg.norm(le3 - re3)
        if n < 1e-6 or face_w < 1e-6:
            return None
        axis = axis / n
        # Mid-sagittal plane normal: left-right direction made orthogonal to
        # the midline axis (Gram-Schmidt), so the plane contains nose + chin.
        lr = re3 - le3
        lr = lr - np.dot(lr, axis) * axis
        ln = np.linalg.norm(lr)
        if ln < 1e-6:
            return None
        normal = lr / ln
        out = {}
        for name, (pairs, _can_alert) in _REGIONS.items():
            devs = []
            for li, ri in pairs:
                lp = ctx.deproject(px[li][0], px[li][1])
                rp = ctx.deproject(px[ri][0], px[ri][1])
                if lp is None or rp is None:
                    return None
                v = rp - nose3
                rp_mirror = nose3 + v - 2 * np.dot(v, normal) * normal
                devs.append(np.linalg.norm(lp - rp_mirror) / face_w)
            out[name] = float(np.mean(devs))
        return out

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        px = ctx.face_px()
        depth_mode = ctx.depth is not None and ctx.intrinsics is not None
        if depth_mode != self._depth_mode:
            self._depth_mode = depth_mode
            self._reset_baseline()

        region_asym = None
        if depth_mode:
            region_asym = self._region_asym_3d(ctx, px)
            if region_asym is None:
                return None                # depth holes this frame: skip
        else:
            nose = px[FL.NOSE_TIP]
            chin = px[FL.CHIN]
            axis = chin - nose
            n = np.linalg.norm(axis)
            if n < 1e-3:
                return None
            axis = axis / n
            normal = np.array([-axis[1], axis[0]])   # perpendicular (mirror normal)
            face_w = np.linalg.norm(px[FL.LEFT_FACE_EDGE] - px[FL.RIGHT_FACE_EDGE]) + 1e-6
            region_asym = {name: self._region_asym(px, nose, normal, face_w, pairs)
                           for name, (pairs, _can_alert) in _REGIONS.items()}

        if self.t0 is None:
            self.t0 = ctx.timestamp
            self.baseline = dict(region_asym)
            return None
        learning = (ctx.timestamp - self.t0) < self.learning_seconds
        a = 0.1 if learning else 0.01
        for name, asym in region_asym.items():
            self.baseline[name] = (1 - a) * self.baseline[name] + a * asym
        if learning:
            return None

        results = []
        if not self._persisted:
            results.extend(self._persist_and_trend(ctx.timestamp))
        for name, (_pairs, can_alert) in _REGIONS.items():
            change = region_asym[name] - self.baseline[name]
            if change < self.threshold:
                self._hits[name] = 0
                continue
            self._hits[name] += 1
            if self._hits[name] < self.alert_after_count:
                continue
            conf = float(min(0.8, change * 10))
            if can_alert:
                sev = Severity.ALERT if change >= self.alert_threshold else Severity.WARNING
                msg = (f"Facial asymmetry increased vs baseline in the {name} "
                       "(possible droop — ask person to smile; consider stroke check)")
            else:
                sev = Severity.WARNING
                msg = (f"Facial asymmetry increased vs baseline in the {name} region "
                       "(lower-specificity; monitor)")
            results.append(self.result(
                f"asymmetry_{name}", round(change, 3), conf, sev, msg, ttl=12.0))
        return results or None
