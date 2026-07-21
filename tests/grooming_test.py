"""Unit test for the grooming/hygiene drift module (modules/grooming.py): a
sustained rise in today's face-texture proxies vs the trailing week's
average should flag a NOTICE, gated by HistoryStore.mean_since() the same
way modules/activity_level.py gates its activity-drop signal.

Uses a temp-file HistoryStore (never the production data/history.db).
HistoryStore.mean_since() windows against real wall-clock time.time(), so
samples must be backdated from the real "now", not a synthetic epoch.

Run standalone:  python tests/grooming_test.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FaceData, FrameContext
from extractors import face_landmarks as FL
from modules.grooming import Grooming
from storage.history_store import HistoryStore

FRAME_SIZE = 200
DAY = 24 * 60 * 60


def _make_ctx(texture_level: float, ts: float) -> FrameContext:
    """A face crop whose hair/jaw bands carry a controllable amount of
    high-frequency noise as a stand-in for texture (0 = smooth/groomed,
    higher = busier/less groomed)."""
    frame = np.full((FRAME_SIZE, FRAME_SIZE, 3), 120, dtype=np.uint8)
    if texture_level > 0:
        rng = np.random.default_rng(0)
        noise = rng.standard_normal((FRAME_SIZE, FRAME_SIZE, 3)) * texture_level
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    lm = np.zeros((478, 3), dtype=np.float32)
    lm[FL.CHIN] = [0.50, 0.75, 0.0]
    face = FaceData(landmarks=lm, bbox=(40, 30, 160, 170),
                    crop=np.zeros((10, 10, 3), dtype=np.uint8), has_iris=True)
    return FrameContext(frame=frame, timestamp=ts, frame_index=0, fps=1.0, face=face)


def _fresh_module(tmp_dir: Path) -> Grooming:
    mod = Grooming()
    mod.store = HistoryStore(path=tmp_dir / "grooming_test.db")
    for key in ("hair_texture", "jaw_texture"):
        mod.store.rolling_mean("grooming", key, mod.recent_seconds)
        mod.store.rolling_mean("grooming", key, mod.baseline_seconds)
    assert mod.store.wait_aggregates()
    return mod


def test_no_report_with_consistent_texture():
    with tempfile.TemporaryDirectory() as td:
        mod = _fresh_module(Path(td))
        try:
            now = time.time()
            for d in range(7, 0, -1):
                assert mod.process(_make_ctx(5.0, now - d * DAY)) is None
            assert mod.process(_make_ctx(5.0, now)) is None
        finally:
            mod.store.close()   # stop async writer/aggregate workers before releasing Windows file
    print("[grooming-test] steady texture over a week -> no report OK")


def test_reports_sustained_rise_vs_weekly_baseline():
    with tempfile.TemporaryDirectory() as td:
        mod = _fresh_module(Path(td))
        try:
            now = time.time()
            # A settled week of low-texture (well-groomed) history...
            for d in range(7, 1, -1):
                mod.process(_make_ctx(5.0, now - d * DAY))
            # ...then today: markedly more textured than usual.
            r = None
            for h in (16, 12, 8, 4, 0):
                r = mod.process(_make_ctx(40.0, now - h * 3600))
            assert r is not None, "sustained high texture vs weekly baseline should report"
            keys = {res.key for res in r}
            assert keys & {"hair_texture_change", "jaw_texture_change"}, keys
            for res in r:
                assert res.severity.value == "notice", res.severity
        finally:
            mod.store.close()
    print("[grooming-test] sustained rise vs weekly baseline reports OK")


def test_no_report_before_baseline_exists():
    """A single frame with no history yet must not report -- mean_since()
    over the 7-day window has nothing to compare against."""
    with tempfile.TemporaryDirectory() as td:
        mod = _fresh_module(Path(td))
        try:
            assert mod.process(_make_ctx(40.0, time.time())) is None
        finally:
            mod.store.close()
    print("[grooming-test] no baseline yet -> no report OK")


def main():
    """Run all grooming-drift tests."""
    test_no_report_with_consistent_texture()
    test_reports_sustained_rise_vs_weekly_baseline()
    test_no_report_before_baseline_exists()
    print("[grooming-test] OK")


if __name__ == "__main__":
    main()
