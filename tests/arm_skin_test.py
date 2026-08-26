"""Contract tests for the arm skin-condition screening path.

Synthetic frames only: pose landmarks are hand-crafted (as in
pose_modules_test.py) and findings are painted patches, so these tests
exercise arm ROI geometry, skin masking (chroma fallback, sleeve gate,
depth gate, hole filling), the rash/bruise/dryness/dark-spot heuristics,
and the arm_check elicitation window without a camera or network.

Run standalone:  python -m pytest -q tests/arm_skin_test.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext, Intrinsics, PoseData
from core.elicitation import ElicitationState
from core.events import Severity
from modules._util import arm_rois, arm_skin_mask
import extractors.pose as P

W, H = 640, 480
SKIN_BGR = (130, 160, 210)          # tan inside the classic YCrCb skin range


def _pose() -> PoseData:
    """Left forearm laid out horizontally: elbow (160,240) -> wrist (352,240)."""
    lm = np.zeros((33, 4), np.float32)
    lm[P.L_SHOULDER] = [0.10, 0.50, 0, 0.0]      # invisible: no upper-arm ROI
    lm[P.L_ELBOW] = [0.25, 0.50, 0, 1.0]
    lm[P.L_WRIST] = [0.55, 0.50, 0, 1.0]
    return PoseData(landmarks=lm, bbox=(0, 0, W, H))


def _ctx(frame=None, ts=100.0, depth=None, intr=None) -> FrameContext:
    frame = np.full((H, W, 3), SKIN_BGR, np.uint8) if frame is None else frame
    ctx = FrameContext(frame=frame, timestamp=ts, frame_index=0, fps=30.0)
    ctx.pose = _pose()
    ctx.person_present = True
    if depth is not None:
        ctx.depth = depth
    if intr is not None:
        ctx.intrinsics = intr
    return ctx


class _FakeStore:
    """In-memory stand-in for HistoryStore.add/mean_since."""

    def __init__(self, recent=None, baseline=None):
        self.added = []
        self.recent = recent
        self.baseline = baseline

    def add(self, namespace, key, value, ts):
        self.added.append((namespace, key, value, ts))

    def mean_since(self, namespace, key, seconds):
        return self.recent if seconds <= 24 * 60 * 60 else self.baseline


def _module(monkeypatch, store=None, **params):
    import modules.arm_skin as mod
    fake = store if store is not None else _FakeStore()
    monkeypatch.setattr(mod, "HistoryStore",
                        type("S", (), {"instance": staticmethod(lambda: fake)}))
    ElicitationState.instance().clear()
    return mod.ArmSkin(**params)


# ------------------------------------------------------------------ ROI/mask

def test_arm_rois_returns_only_visible_segments():
    rois = arm_rois(_ctx())
    assert [label for label, _, _ in rois] == ["left forearm"]
    _, poly, anchors = rois[0]
    assert 150 <= poly[:, 0].min() and poly[:, 0].max() <= 362
    assert np.allclose(anchors[0], (160, 240)) and np.allclose(anchors[1], (352, 240))


def test_arm_skin_mask_accepts_bare_and_rejects_sleeve():
    ctx = _ctx()
    _, poly, anchors = arm_rois(ctx)[0]
    mask = arm_skin_mask(ctx, poly, anchors)
    assert mask is not None and (mask > 0).sum() > 10000
    sleeved = _ctx(frame=np.full((H, W, 3), (200, 80, 30), np.uint8))  # blue sleeve
    assert arm_skin_mask(sleeved, poly, anchors) is None


def test_arm_skin_mask_depth_gate_drops_background():
    depth = np.full((H, W), 1000, np.uint16)     # arm plane at 1.0 m
    depth[260:, :] = 3000                        # far background below the arm
    ctx = _ctx(depth=depth)
    _, poly, anchors = arm_rois(ctx)[0]
    mask = arm_skin_mask(ctx, poly, anchors)
    assert mask is not None
    assert (mask[262:298, 170:340] > 0).sum() == 0
    assert (mask[182:238, 170:340] > 0).sum() > 5000


def test_arm_skin_mask_fills_discolored_holes():
    frame = np.full((H, W, 3), SKIN_BGR, np.uint8)
    frame[230:250, 240:280] = (140, 50, 90)      # deep purple: fails chroma test
    ctx = _ctx(frame=frame)
    _, poly, anchors = arm_rois(ctx)[0]
    mask = arm_skin_mask(ctx, poly, anchors)
    assert mask is not None
    assert (mask[235:245, 250:270] > 0).all()    # bruise interior kept


# ---------------------------------------------------------------- heuristics

def test_clean_skin_reports_nothing(monkeypatch):
    module = _module(monkeypatch)
    assert module.process(_ctx()) is None


def test_red_speckled_patch_flags_rash(monkeypatch):
    frame = np.full((H, W, 3), SKIN_BGR, np.uint8)
    for y in range(210, 270):                    # red AND locally patchy
        for x in range(220, 300):
            if (x // 2 + y // 2) % 2 == 0:
                frame[y, x] = (130, 140, 210)    # modestly redder skin tone
    module = _module(monkeypatch)
    results = module.process(_ctx(frame=frame)) or []
    rash = [r for r in results if r.key == "arm_rash_fraction_left"]
    assert rash and rash[0].value > 0.02
    assert rash[0].severity in (Severity.NOTICE, Severity.WARNING)


def test_purple_patch_flags_bruise(monkeypatch):
    frame = np.full((H, W, 3), SKIN_BGR, np.uint8)
    frame[230:250, 240:280] = (140, 50, 90)      # purple, darker than skin
    module = _module(monkeypatch)
    results = module.process(_ctx(frame=frame)) or []
    assert any(r.key == "arm_bruise_fraction_left" for r in results)


def test_dark_spot_sized_by_depth_and_drift_flagged(monkeypatch):
    frame = np.full((H, W, 3), SKIN_BGR, np.uint8)
    cv2.circle(frame, (280, 240), 4, (40, 50, 70), -1)   # ~13 mm at 1 m
    depth = np.full((H, W), 1000, np.uint16)
    intr = Intrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
    store = _FakeStore(recent=2.0, baseline=0.5)
    module = _module(monkeypatch, store=store)
    results = module.process(_ctx(frame=frame, depth=depth, intr=intr)) or []
    assert store.added and store.added[0][:2] == ("arm_skin", "spots_left")
    assert store.added[0][2] >= 1.0
    assert any(r.key == "arm_dark_spots_left" for r in results)


def test_no_depth_means_no_spot_logging(monkeypatch):
    store = _FakeStore()
    module = _module(monkeypatch, store=store)
    module.process(_ctx())
    assert store.added == []


# --------------------------------------------------------- arm_check window

def test_arm_check_window_consolidates_once(monkeypatch):
    module = _module(monkeypatch)
    es = ElicitationState.instance()
    es.begin("arm_check", 5.0, now=100.0)
    try:
        assert module.process(_ctx(ts=101.0)) is None    # sampling, no chatter
        assert module.process(_ctx(ts=102.0)) is None
        assert module.process(_ctx(ts=103.0)) is None
        assert module.interval == module.window_interval
        out = module.process(_ctx(ts=106.0))             # window closed
        assert module.interval == module._passive_interval
        assert len(out) == 1 and out[0].key == "arm_check"
        assert out[0].value["status"] == "succeeded"
        assert out[0].value["finding_present"] is False
        assert out[0].severity == Severity.INFO
        assert out[0].source == "local_arm_skin"
        assert out[0].message.startswith("Local camera arm check:")
    finally:
        es.clear()


def test_arm_check_without_usable_samples_is_never_clear(monkeypatch):
    module = _module(monkeypatch)
    es = ElicitationState.instance()
    es.begin("arm_check", 2.0, now=100.0, correlation_id="arm-session")
    hidden = _ctx(ts=101.0)
    hidden.pose = None
    try:
        assert module.process(hidden) is None
        out = module.process(_ctx(ts=103.0))
        assert len(out) == 1
        assert out[0].value["status"] == "unavailable"
        assert out[0].value["reason"] == "insufficient_arm_samples"
        assert out[0].correlation_id == "arm-session"
        assert "clear" not in out[0].message.lower()
    finally:
        es.clear()


def test_registry_knows_arm_skin():
    import modules.arm_skin  # noqa: F401  (import runs @register)
    from core.registry import all_registered
    assert "arm_skin" in all_registered()
