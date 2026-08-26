"""Sneeze event detection (cold-symptom screening input).

Method: a sneeze has a distinctive ~half-second temporal signature no single
frame shows — the head pitches down fast while the eyes reflexively snap
shut (the sneeze reflex closes them involuntarily). We track head pitch
(nose-below-eye-line, the same measure modules/head_nod.py uses) and EAR
(modules/drowsiness.py's eye-aspect ratio) in short rolling buffers and
fire when a rapid pitch drop coincides with an eye-closure dip. Ordinary
nodding is slower, and blinking lacks the pitch jerk, so requiring both in
the same sub-second window filters most confounds.

Counts feed the cold-symptom advisor (agent/advisor_engine.py) together
with modules/face_touch.py and skin_color flushing.

Reliability: MEDIUM — clear sneezes with the face in view; misses sneezes
turned away from the camera (people often turn to sneeze), so treat counts
as a lower bound.
"""
from __future__ import annotations

import numpy as np

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule
from modules._util import TimedBuffer
from modules.drowsiness import _ear
from extractors import face_landmarks as FL


@register("sneeze")
class Sneeze(DetectionModule):
    """Sneeze event detection from head-pitch jerk + reflex eye closure."""
    interval = 0.0
    requires = ("face",)
    window_seconds = 0.8         # a sneeze's jerk fits well inside this
    pitch_jerk_threshold = 0.08  # min pitch rise (fraction of face height) in window
    ear_drop_ratio = 0.55        # eyes must dip below this fraction of open baseline
    cooldown_seconds = 2.0       # one physical sneeze = one event
    count_window_seconds = 600.0

    def __init__(self, **params):
        super().__init__(**params)
        self.pitch_buf = TimedBuffer(self.window_seconds)
        self.ear_buf = TimedBuffer(self.window_seconds)
        self.events = TimedBuffer(self.count_window_seconds)
        self.ear_baseline = None
        self._last_event = -1e9

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        px = ctx.face_px()
        eye_mid = (px[159] + px[386]) / 2.0
        face_h = np.linalg.norm(px[FL.FOREHEAD_TOP] - px[FL.CHIN]) + 1e-6
        pitch = (px[FL.NOSE_TIP][1] - eye_mid[1]) / face_h   # larger = head down
        ear = 0.5 * (_ear(px, FL.LEFT_EYE_EAR) + _ear(px, FL.RIGHT_EYE_EAR))
        self.pitch_buf.push(ctx.timestamp, float(pitch))
        self.ear_buf.push(ctx.timestamp, float(ear))
        # Slow EMA of the open-eye EAR; the reflex closure is too brief to
        # drag it down, so it stays a valid "eyes open" reference.
        self.ear_baseline = (ear if self.ear_baseline is None
                             else 0.99 * self.ear_baseline + 0.01 * ear)

        results = [self.result("sneeze_count_10min", len(self.events), 0.6,
                               Severity.INFO, "", ttl=30.0)]
        if self.pitch_buf.span() < 0.3 or \
                ctx.timestamp - self._last_event < self.cooldown_seconds:
            return results

        t, p = self.pitch_buf.arrays()
        jerk = float(p[-1] - p.min())          # downward head snap within window
        _, e = self.ear_buf.arrays()
        eyes_shut = float(e.min()) < self.ear_drop_ratio * self.ear_baseline
        if jerk > self.pitch_jerk_threshold and eyes_shut:
            self._last_event = ctx.timestamp
            self.events.push(ctx.timestamp, 1.0)
            results.append(self.result(
                "sneeze", True, min(0.75, jerk * 6), Severity.NOTICE,
                "Sneeze detected", ttl=8.0))
            results.append(self.result("sneeze_count_10min", len(self.events),
                                       0.6, Severity.INFO, "", ttl=30.0))
        return results
