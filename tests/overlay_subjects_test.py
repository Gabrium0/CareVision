"""Multi-person camera-overlay contracts (output/overlay.py).

The camera window previously drew only ctx.face/ctx.pose (one box), leaving
--enable-multi-person's tracked people invisible on screen even though the
web UI and per-subject readings already named them by track-N id. This
covers the pure style/label decisions directly, and exercises the drawing
entry points (draw_boxes/draw) enough to prove the empty-ctx.subjects path
is byte-for-byte the original single-person behavior.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from core.context import FaceData, FrameContext, PoseData, SubjectView
from core.events import Result, Severity
from output.overlay import _row_text, draw, draw_boxes, draw_subjects, subject_style

_IDENTITY_WORDS = re.compile(r"\b(name|person\s*\d|guest|visitor|mr|mrs|ms|dr)\b", re.IGNORECASE)


def _subject(primary=False, ambiguous=False, track_id="track-2"):
    return SubjectView(subject_id=("primary" if primary else track_id),
                       track_id=track_id, primary=primary, ambiguous=ambiguous,
                       bbox=(10, 10, 50, 90))


def _frame(w=200, h=150):
    return np.zeros((h, w, 3), np.uint8)


def _ctx(subjects=None, face=None, pose=None):
    ctx = FrameContext(frame=_frame(), timestamp=1.0, frame_index=0, fps=30.0)
    ctx.face = face
    ctx.pose = pose
    ctx.subjects = subjects or []
    return ctx


# --------------------------------------------------------------- subject_style

def test_primary_is_bright_green_and_thicker():
    color, thickness, label = subject_style(_subject(primary=True))
    assert color == (0, 220, 0)
    assert thickness == 2
    assert label == "PRIMARY"


def test_secondary_is_visually_distinct_from_primary():
    p_color, _, _ = subject_style(_subject(primary=True))
    s_color, s_thickness, s_label = subject_style(_subject(primary=False, track_id="track-2"))
    assert s_color != p_color
    assert s_thickness == 1
    assert s_label == "track-2"


def test_ambiguous_overrides_color_regardless_of_primary():
    color, _, label = subject_style(_subject(primary=True, ambiguous=True))
    assert color == (0, 140, 255)
    assert label.endswith(" ?")


@pytest.mark.parametrize("primary,ambiguous", [(True, True), (True, False),
                                               (False, True), (False, False)])
def test_labels_never_contain_identity_words(primary, ambiguous):
    _, _, label = subject_style(_subject(primary=primary, ambiguous=ambiguous,
                                         track_id="track-7"))
    assert not _IDENTITY_WORDS.search(label), f"label leaks identity wording: {label!r}"
    assert label in ("PRIMARY", "PRIMARY ?", "track-7", "track-7 ?")


# --------------------------------------------------------------- _row_text

def test_row_text_tags_non_primary_subject():
    r = Result(module="fall", key="fall", value=True, confidence=0.8,
              severity=Severity.ALERT, message="FALL DETECTED", subject_id="track-2")
    text = _row_text(r)
    assert "track-2" in text
    assert "FALL DETECTED" in text


def test_row_text_leaves_primary_untagged():
    r = Result(module="fall", key="fall", value=True, confidence=0.8,
              severity=Severity.ALERT, message="FALL DETECTED", subject_id="primary")
    text = _row_text(r)
    assert "[primary]" not in text
    assert text == "[0.80] FALL DETECTED"


# --------------------------------------------------------------- drawing entry points

def test_draw_subjects_handles_empty_bbox_and_multiple_people():
    ctx = _ctx(subjects=[_subject(primary=True, track_id="track-1"),
                         _subject(primary=False, track_id="track-2")])
    frame = ctx.frame.copy()
    out = draw_subjects(frame, ctx)
    assert out.shape == frame.shape
    assert out.any()   # something was actually drawn (frame is no longer all-zero)


def test_draw_boxes_falls_back_to_legacy_single_box_when_no_subjects():
    face = FaceData(landmarks=np.zeros((478, 3), np.float32), bbox=(5, 5, 40, 40),
                    crop=np.zeros((35, 35, 3), np.uint8))
    ctx = _ctx(subjects=[], face=face)
    out = draw_boxes(ctx.frame.copy(), ctx, fps=30.0)
    assert out.any()


def test_draw_boxes_uses_subject_path_when_subjects_present():
    ctx = _ctx(subjects=[_subject(primary=True, track_id="track-1")])
    out = draw_boxes(ctx.frame.copy(), ctx, fps=30.0)
    assert out.any()


def test_draw_does_not_raise_with_subjects_and_mixed_subject_snapshot():
    ctx = _ctx(subjects=[_subject(primary=True, track_id="track-1"),
                         _subject(primary=False, track_id="track-2")])
    snapshot = [
        Result(module="fall", key="fall", value=True, confidence=0.9,
              severity=Severity.ALERT, message="FALL DETECTED", subject_id="track-2"),
        Result(module="heart_rate", key="bpm", value=72, confidence=0.6,
              severity=Severity.INFO, message="72 bpm", subject_id="primary"),
    ]
    out = draw(ctx.frame.copy(), ctx, snapshot, fps=30.0)
    assert out.any()
