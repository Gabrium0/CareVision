"""Respiratory rate from chest motion.

Method: sample a 1D chest-motion signal over a long window, bandpass to
0.1-0.5 Hz (6-30 breaths/min), take dominant frequency. The sampled value
depends on the camera:

- **RGB-only** (webcam): mean shoulder y-position — visible torso rise/fall,
  needs a fairly still subject.
- **Depth** (RealSense D435i): mean depth over a chest ROI below the
  shoulders. The chest wall moves sub-cm toward/away from the camera with
  each breath, which depth resolves directly — works through clothing and
  is robust to lighting. Same buffer/filter pipeline, only the sample
  changes; the buffer resets when the modality flips (camera switch) so
  pixel units never mix with meters.

Reliability: medium on RGB (still subject, visible torso); high with depth.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer, dominant_frequency, bandpass
from extractors import pose as P


@register("respiration")
class Respiration(DetectionModule):
    """Respiratory rate from shoulder vertical oscillation."""
    interval = 0.0
    requires = ("pose",)
    window_seconds = 25.0

    def __init__(self, **params):
        super().__init__(**params)
        self.buf = TimedBuffer(self.window_seconds)
        self._depth_mode = False

    def _chest_depth(self, ctx: FrameContext) -> float | None:
        """Mean valid depth (m) over a chest ROI just below the shoulder
        line, half a shoulder-width tall — the region whose distance to the
        camera breathes."""
        px = ctx.pose_px()
        l, r = px[P.L_SHOULDER], px[P.R_SHOULDER]
        width = float(np.linalg.norm(l - r))
        if width < 10:                      # subject too far for a usable ROI
            return None
        x1 = int(max(min(l[0], r[0]), 0))
        x2 = int(min(max(l[0], r[0]), ctx.w))
        y1 = int(max((l[1] + r[1]) / 2.0, 0))
        y2 = int(min(y1 + 0.5 * width, ctx.h))
        roi = ctx.depth[y1:y2, x1:x2]
        valid = roi[roi > 0]
        if valid.size < 20:                 # mostly holes: unusable
            return None
        return float(valid.mean()) * ctx.depth_scale

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        lm = ctx.pose.landmarks
        if lm[P.L_SHOULDER, 3] < 0.5 or lm[P.R_SHOULDER, 3] < 0.5:
            return None
        sample = self._chest_depth(ctx) if ctx.depth is not None else None
        depth_mode = sample is not None
        if not depth_mode:
            sample = float((lm[P.L_SHOULDER, 1] + lm[P.R_SHOULDER, 1]) / 2.0)
        if depth_mode != self._depth_mode:  # modality changed: units differ
            self.buf = TimedBuffer(self.window_seconds)
            self._depth_mode = depth_mode
        self.buf.push(ctx.timestamp, sample)

        rs = self.buf.resampled(fs=10.0)
        if rs is None or self.buf.span() < 15.0:
            return None
        signal, fs = rs
        filt = bandpass(signal, fs, 0.1, 0.5, order=2)
        if filt is None:
            return None
        dom = dominant_frequency(filt, fs, 0.1, 0.5)
        if dom is None:
            return None
        freq, prominence = dom
        brpm = freq * 60.0
        conf = round(min(1.0, prominence * 3.0) *
                     min(1.0, self.buf.span() / self.window_seconds), 2)
        sev = Severity.INFO
        msg = f"Respiration ~{brpm:.0f} breaths/min"
        if conf >= 0.35 and (brpm < 10 or brpm > 22):
            sev = Severity.NOTICE
            msg = f"Respiration ~{brpm:.0f} breaths/min (atypical)"
        return self.result("breaths_per_min", round(brpm, 1), conf, sev, msg, ttl=10.0)
