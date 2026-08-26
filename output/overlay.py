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

_PRIMARY_COLOR = (0, 220, 0)      # bright green -- same as the single-person face box
_SECONDARY_COLOR = (200, 160, 60)  # steel blue -- visually distinct from primary
_AMBIGUOUS_COLOR = (0, 140, 255)   # amber -- draws the tracker's own uncertainty


def subject_style(subject) -> tuple[tuple[int, int, int], int, str]:
    """Box color (BGR), line thickness, and label for one tracked subject
    (core/context.py SubjectView). Ambiguous assignment gets a distinct color
    regardless of primary/secondary, since AnonymousTracker's own uncertainty
    about who's who must not be drawn as if it were confident. The label uses
    only the tracker's own anonymous vocabulary (PRIMARY / track-N) -- never
    a name or anything that reads as identity."""
    if subject.ambiguous:
        color = _AMBIGUOUS_COLOR
    elif subject.primary:
        color = _PRIMARY_COLOR
    else:
        color = _SECONDARY_COLOR
    thickness = 2 if subject.primary else 1
    label = "PRIMARY" if subject.primary else subject.track_id
    if subject.ambiguous:
        label += " ?"
    return color, thickness, label


def draw_subjects(frame, ctx):
    """Draw one box + label per anonymously-tracked person, so a multi-person
    run maps on-screen people to the track-N ids used everywhere else (the
    web UI, per-subject readings). No-op if ctx.subjects is empty."""
    h, w = frame.shape[:2]
    for subject in ctx.subjects:
        bbox = (subject.pose.bbox if subject.pose is not None
                else subject.face.bbox if subject.face is not None
                else subject.bbox)
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        color, thickness, label = subject_style(subject)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        # Nested face-detail box only for the primary -- avoids doubling the
        # box count per secondary subject once several people are tracked.
        if subject.primary and subject.face is not None and subject.pose is not None:
            fx1, fy1, fx2, fy2 = subject.face.bbox
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 1)
        label_y = max(14, y1 - 6)                       # clamp off the top edge
        label_x = max(2, min(x1, max(2, w - 8 * len(label))))
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_boxes(frame, ctx, fps: float, performance: dict | None = None):
    """Lightweight camera overlay: face/pose boxes + fps only. All textual
    data lives in the separate dashboard window (output/dashboard.py)."""
    if ctx.subjects:
        draw_subjects(frame, ctx)
    else:
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


def _row_text(r) -> str:
    """One overlay text row for a detection result; subject-tags anything
    that isn't the primary so a secondary person's line (e.g. a non-
    escalating fall/unresponsive result) is never mistaken for the primary's
    in the one view an operator is actually watching live."""
    prefix = f"[{r.subject_id}] " if r.subject_id != "primary" else ""
    return f"[{r.confidence:.2f}] {prefix}{r.message}"


def draw(frame, ctx, snapshot, fps: float):
    """Draw the detections overlay onto the frame."""
    if ctx.subjects:
        draw_subjects(frame, ctx)
    else:
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
        cv2.putText(frame, _row_text(r), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
    return frame
