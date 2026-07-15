"""Draw detection results onto the frame for a live debug view."""
from __future__ import annotations

import cv2

from core.events import Severity

_COLORS = {
    Severity.INFO: (200, 200, 200),
    Severity.NOTICE: (0, 200, 255),
    Severity.WARNING: (0, 140, 255),
    Severity.ALERT: (0, 0, 255),
}


def draw_boxes(frame, ctx, fps: float, performance: dict | None = None):
    """Lightweight camera overlay: face/pose boxes + fps only. All textual
    data lives in the separate dashboard window (output/dashboard.py)."""
    if ctx.face is not None:
        x1, y1, x2, y2 = ctx.face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 1)
    if ctx.pose is not None:
        x1, y1, x2, y2 = ctx.pose.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 120, 0), 1)
    perf = performance or {}
    label = (f"C {perf.get('capture_fps', fps):.1f}  P {perf.get('preview_fps', 0):.1f}  "
             f"A {perf.get('analysis_fps', 0):.1f} fps" if perf else f"{fps:4.1f} fps")
    cv2.putText(frame, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def draw(frame, ctx, snapshot, fps: float):
    """Draw the detections overlay onto the frame."""
    if ctx.face is not None:
        x1, y1, x2, y2 = ctx.face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 1)
    if ctx.pose is not None:
        x1, y1, x2, y2 = ctx.pose.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 120, 0), 1)

    order = {Severity.ALERT: 3, Severity.WARNING: 2, Severity.NOTICE: 1, Severity.INFO: 0}
    rows = sorted([r for r in snapshot if r.message],
                  key=lambda r: (order[r.severity], r.confidence), reverse=True)[:14]
    y = 22
    cv2.putText(frame, f"{fps:4.1f} fps  |  {len(snapshot)} signals",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 22
    for r in rows:
        color = _COLORS[r.severity]
        txt = f"[{r.confidence:.2f}] {r.message}"
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
    return frame
