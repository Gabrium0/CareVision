"""Arm skin-condition screening on pose-derived arm ROIs.

Method: within each visible bare arm segment (upper-arm / forearm oriented
boxes from pose landmarks; skin isolated by chroma against the person's own
face skin, background rejected by depth on RealSense), screen for
 - rash: red AND locally patchy clusters (same math as modules/rash.py),
 - bruise: purple/blue patches darker than surrounding arm skin
   (modules/bruise.py), sized in cm when depth gives a mm-per-px scale,
 - dryness/scaling: strongly textured patchy clusters WITHOUT redness,
 - dark spots: lesion-sized dark blobs (2-15 mm via depth mm-per-px) whose
   daily count is compared against a trailing-week HistoryStore baseline so
   tattoos, birthmarks, and long-standing moles don't re-flag.

Passive fractions are pooled over a short window to damp jitter. During an
`arm_check` elicitation window (the agent asks the person to hold a forearm
up to the camera), the module samples densely, suppresses per-frame chatter,
and publishes one consolidated `arm_check` result when the window closes —
mirroring modules/tremor.py's hold-still test.

Reliability: LOW. Screening prompts only, never a diagnosis.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.context import FrameContext
from core.elicitation import ElicitationState
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, arm_rois, arm_skin_region, pooled_skin_sample
from storage.history_store import HistoryStore


@register("arm_skin")
class ArmSkin(DetectionModule):
    """Arm skin-condition screening on pose-derived arm ROIs."""
    interval = 3.0
    requires = ("pose",)
    pool_seconds = 10.0            # damp per-frame chroma/texture jitter
    min_skin_px = 600              # enough bare-skin pixels for stable stats
    rash_notice = 0.02
    rash_warning = 0.06
    dryness_notice = 0.04
    bruise_blob_frac = 0.004       # min connected-blob area vs segment skin
    lesion_min_mm = 2.0
    lesion_max_mm = 15.0
    lesion_min_px = 3.0            # below ~3 px across, a "spot" is just noise
    spot_rise = 0.9                # ~one more spot than the weekly usual flags
    recent_seconds = 24 * 60 * 60
    baseline_seconds = 7 * 24 * 60 * 60
    window_interval = 0.4          # dense cadence inside an arm_check window
    min_window_samples = 3         # never call an unseen/unstable arm "clear"

    def __init__(self, **params):
        super().__init__(**params)
        self.store = HistoryStore.instance()
        rolling = getattr(self.store, "rolling_mean", None)
        if rolling is not None:
            for side in ("left", "right"):
                rolling("arm_skin", f"spots_{side}", self.recent_seconds)
                rolling("arm_skin", f"spots_{side}", self.baseline_seconds)
        self._pools: dict[tuple[str, str], TimedBuffer] = {}
        self._passive_interval = float(self.interval)
        self._window_id = None         # started-ts of the window being sampled
        self._window_best: dict[str, tuple[float, str]] = {}
        self._window_usable_samples = 0
        self._window_attempt = 0
        self._window_correlation_id: str | None = None

    # ------------------------------------------------------------- analysis

    def _pooled(self, side: str, metric: str, value: float, ts: float) -> float:
        buf = self._pools.setdefault((side, metric),
                                     TimedBuffer(float(self.pool_seconds)))
        return float(pooled_skin_sample(buf, value, ts))

    def _analyze_segment(self, ctx: FrameContext, label: str,
                         poly: np.ndarray, anchors: np.ndarray) -> dict | None:
        """Raw finding fractions for one bare arm segment, or None."""
        region = arm_skin_region(ctx, poly, anchors)
        if region is None:
            return None
        frame, mask, _ = region
        roi_scale = 1.0
        max_dim = max(frame.shape[:2])
        quality = str(ctx.extras.get("quality_profile", "maximum"))
        roi_cap = int(ctx.extras.get("detail_roi_cap", {
            "maximum": 640, "balanced": 480, "realtime": 320}.get(quality, 640)))
        if max_dim > roi_cap:
            roi_scale = float(roi_cap) / float(max_dim)
            size = (max(1, int(round(frame.shape[1] * roi_scale))),
                    max(1, int(round(frame.shape[0] * roi_scale))))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        skin = mask > 0
        n_skin = int(skin.sum())
        if n_skin < int(self.min_skin_px):
            return None
        a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        texture = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 3))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, v = hsv[:, :, 0].astype(np.float32), hsv[:, :, 2].astype(np.float32)

        a_skin = a[skin]
        red = (a > a_skin.mean() + 1.5 * a_skin.std()) & skin
        tex_skin = texture[skin]
        patchy = texture > tex_skin.mean() + 1.0 * tex_skin.std()
        rash_frac = float((patchy & red).sum()) / n_skin

        # Dry/scaling: strong texture clusters without redness. The higher
        # 2.5-sigma cut plus a morphological open keeps ordinary skin (whose
        # texture tail alone would pass a 1-sigma cut) from flagging.
        scaling = ((texture > tex_skin.mean() + 2.5 * tex_skin.std())
                   & skin & ~red).astype(np.uint8)
        scaling = cv2.morphologyEx(scaling, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8))
        dry_frac = float(scaling.sum()) / n_skin

        v_skin = v[skin]
        purple = (hue > 110) & (hue < 160)
        dark = v < (v_skin.mean() - 1.2 * v_skin.std())
        cand = (purple & dark & skin).astype(np.uint8)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(cand)
        big = [i for i in range(1, n)
               if stats[i, cv2.CC_STAT_AREA] > float(self.bruise_blob_frac) * n_skin]
        bruise_frac = (sum(stats[i, cv2.CC_STAT_AREA] for i in big) / n_skin
                       if big else 0.0)
        mid = anchors.mean(axis=0)
        mm_per_px = ctx.mm_per_px(mid[0], mid[1])
        bruise_cm = None
        if big and mm_per_px is not None:
            largest = max(big, key=lambda i: stats[i, cv2.CC_STAT_AREA])
            diam_px = (2.0 * np.sqrt(stats[largest, cv2.CC_STAT_AREA] / np.pi)
                       / roi_scale)
            bruise_cm = round(diam_px * mm_per_px / 10.0, 1)

        spots = 0
        max_spot_mm = 0.0
        if mm_per_px is not None:              # no metric scale -> no sizing
            darkest = (v < (v_skin.mean() - 1.5 * v_skin.std())) & skin
            blobs = cv2.morphologyEx(darkest.astype(np.uint8), cv2.MORPH_OPEN,
                                     np.ones((3, 3), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(blobs)
            for i in range(1, n):
                diam_px = (2.0 * np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi)
                           / roi_scale)
                mm = diam_px * mm_per_px
                if (diam_px >= float(self.lesion_min_px)
                        and float(self.lesion_min_mm) <= mm <= float(self.lesion_max_mm)):
                    spots += 1
                    max_spot_mm = max(max_spot_mm, mm)

        return {"label": label, "rash": rash_frac, "dry": dry_frac,
                "bruise": bruise_frac, "bruise_cm": bruise_cm,
                "spots": spots, "max_spot_mm": max_spot_mm,
                "sized": mm_per_px is not None}

    def _side_findings(self, ctx: FrameContext) -> dict[str, dict]:
        """Worst pooled finding per side across that side's bare segments."""
        sides: dict[str, dict] = {}
        for label, poly, anchors in arm_rois(ctx):
            seg = self._analyze_segment(ctx, label, poly, anchors)
            if seg is None:
                continue
            side = label.split()[0]
            cur = sides.setdefault(side, {"rash": 0.0, "dry": 0.0, "bruise": 0.0,
                                          "bruise_cm": None, "spots": 0,
                                          "max_spot_mm": 0.0, "sized": False})
            if seg["bruise"] >= cur["bruise"] and seg["bruise_cm"] is not None:
                cur["bruise_cm"] = seg["bruise_cm"]
            for key in ("rash", "dry", "bruise"):
                cur[key] = max(cur[key], seg[key])
            cur["spots"] += seg["spots"]
            cur["max_spot_mm"] = max(cur["max_spot_mm"], seg["max_spot_mm"])
            cur["sized"] = cur["sized"] or seg["sized"]
        for side, f in sides.items():
            for key in ("rash", "dry", "bruise"):
                f[key] = self._pooled(side, key, f[key], ctx.timestamp)
        return sides

    # -------------------------------------------------------------- results

    def _passive_results(self, ctx: FrameContext, sides: dict[str, dict]) -> list:
        results = []
        for side, f in sides.items():
            if f["rash"] >= float(self.rash_notice):
                sev = (Severity.WARNING if f["rash"] > float(self.rash_warning)
                       else Severity.NOTICE)
                results.append(self.result(
                    f"arm_rash_fraction_{side}", round(f["rash"], 3),
                    float(min(0.75, f["rash"] * 8)), sev,
                    f"Possible skin rash on ~{f['rash']*100:.0f}% of the "
                    f"{side} arm skin (screening only)", ttl=20.0))
            if f["bruise"] > 0.0:
                size = (f" (~{f['bruise_cm']} cm across)"
                        if f["bruise_cm"] else "")
                results.append(self.result(
                    f"arm_bruise_fraction_{side}", round(f["bruise"], 3),
                    float(min(0.7, f["bruise"] * 10)), Severity.NOTICE,
                    f"Possible bruising/discoloration on the {side} arm"
                    f"{size} (screening only)", ttl=25.0))
            if f["dry"] >= float(self.dryness_notice):
                results.append(self.result(
                    f"arm_dryness_fraction_{side}", round(f["dry"], 3),
                    float(min(0.6, f["dry"] * 6)), Severity.NOTICE,
                    f"Patchy dry/scaling texture on the {side} arm skin "
                    "(screening only)", ttl=25.0))
            results.extend(self._spot_drift(ctx, side, f))
        return results

    def _spot_drift(self, ctx: FrameContext, side: str, f: dict) -> list:
        """Log today's dark-spot count; flag a rise vs the weekly baseline."""
        if not f["sized"]:
            return []
        self.store.add("arm_skin", f"spots_{side}", float(f["spots"]),
                       ctx.timestamp)
        mean = getattr(self.store, "rolling_mean", None) or self.store.mean_since
        recent = mean("arm_skin", f"spots_{side}", self.recent_seconds)
        baseline = mean("arm_skin", f"spots_{side}", self.baseline_seconds)
        if recent is None or baseline is None:
            return []
        if recent < baseline + float(self.spot_rise):
            return []
        return [self.result(
            f"arm_dark_spots_{side}", round(recent, 1), 0.35, Severity.NOTICE,
            f"More small dark spots than usual on the {side} arm "
            f"(largest ~{f['max_spot_mm']:.0f} mm; screening only)",
            ttl=3600.0)]

    # ------------------------------------------------------ arm_check window

    def _track_window_best(self, sides: dict[str, dict]) -> None:
        for side, f in sides.items():
            checks = (("rash", f["rash"], self.rash_notice,
                       f"redness/rash-like texture on the {side} arm"),
                      ("bruise", f["bruise"], 1e-9,
                       f"bruise-like discoloration on the {side} arm"),
                      ("dry", f["dry"], self.dryness_notice,
                       f"dry/scaling texture on the {side} arm"))
            for metric, value, floor, note in checks:
                if value < float(floor):
                    continue
                best = self._window_best.get(f"{metric}_{side}")
                if best is None or value > best[0]:
                    self._window_best[f"{metric}_{side}"] = (value, note)

    def _window_result(self):
        """One consolidated result when the arm_check window closes."""
        if self._window_usable_samples < int(self.min_window_samples):
            return self.result(
                "arm_check",
                {"status": "unavailable", "source": "local_arm_skin",
                 "reason": "insufficient_arm_samples",
                 "attempt": self._window_attempt,
                 "capture_mode": "pose_crop"},
                0.0, Severity.INFO,
                "Local camera arm check could not capture a stable bare-arm view",
                ttl=15.0, source="local_arm_skin",
                correlation_id=self._window_correlation_id)
        if self._window_best:
            notes = [note for _, note in self._window_best.values()]
            worst = max(value for value, _ in self._window_best.values())
            return self.result(
                "arm_check",
                {"status": "succeeded", "source": "local_arm_skin",
                 "finding_present": True, "attempt": self._window_attempt,
                 "capture_mode": "pose_crop",
                 "findings": {key: round(value, 3) for key, (value, _)
                              in self._window_best.items()}},
                float(min(0.7, 0.3 + worst * 5)), Severity.NOTICE,
                "Local camera arm check: possible " + "; ".join(notes) +
                " (screening only)", ttl=15.0, source="local_arm_skin",
                correlation_id=self._window_correlation_id)
        return self.result(
            "arm_check",
            {"status": "succeeded", "source": "local_arm_skin",
             "finding_present": False, "attempt": self._window_attempt,
             "capture_mode": "pose_crop"},
            0.6, Severity.INFO,
            "Local camera arm check: the visible arm skin looked clear", ttl=15.0,
            source="local_arm_skin", correlation_id=self._window_correlation_id)

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        es = ElicitationState.instance()
        if es.active("arm_check", now=ctx.timestamp):
            if self._window_id != es.started:      # window just opened
                self._window_id = es.started
                self._window_best = {}
                self._window_usable_samples = 0
                self._window_attempt = es.attempt
                self._window_correlation_id = es.correlation_id
                self.interval = float(self.window_interval)
            sides = self._side_findings(ctx)
            if sides:
                self._window_usable_samples += 1
            self._track_window_best(sides)
            return None                            # one consolidated report
        if self._window_id is not None:            # window just closed
            self.interval = self._passive_interval
            result = self._window_result()
            self._window_id = None
            self._window_best = {}
            self._window_usable_samples = 0
            self._window_correlation_id = None
            return [result]
        sides = self._side_findings(ctx)
        if not sides:
            return None
        return self._passive_results(ctx, sides) or None
