"""Unit test for runtime camera switching in core/camera.py (Camera.switch_to
/ _apply_switch): pressing 'c' in main.py toggles between the laptop cam and
the external OV2735 without restarting, and a failed open must fall back to the
previous device instead of killing the stream.

Uses a fake cv2.VideoCapture so it needs no real hardware and no reader-thread
timing. Cameras are opened with lock=False so _configure() returns immediately
(no exposure/WB settle loop) -- switching logic is what's under test here.

Run standalone:  python tests/camera_switch_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.camera as camera_mod
from core.camera import Camera


class FakeCapture:
    """Minimal stand-in for cv2.VideoCapture. Sources listed in `bad_sources`
    report isOpened()==False, simulating an unplugged/busy device."""
    opened_log: list = []
    bad_sources: set = set()

    def __init__(self, source, backend=None):
        FakeCapture.opened_log.append(source)
        self.source = source
        self._opened = source not in FakeCapture.bad_sources

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        return True, np.full((48, 64, 3), 100, np.uint8)

    def set(self, *_a):
        return True

    def get(self, *_a):
        return 0.0

    def release(self):
        self._opened = False


def _patched():
    """Swap cv2.VideoCapture for the fake; returns the original to restore."""
    orig = camera_mod.cv2.VideoCapture
    camera_mod.cv2.VideoCapture = FakeCapture
    return orig


def test_switch_to_flags_and_take_pending_clears():
    cam = Camera(source=0, lock=False)
    assert cam._take_pending() is None
    cam.switch_to(1)
    assert cam._pending == (1, {})
    assert cam._take_pending() == (1, {})
    assert cam._take_pending() is None          # cleared after taking
    print("[camera-switch-test] switch_to flags a pending switch, take clears it OK")


def test_apply_switch_swaps_source():
    orig = _patched()
    FakeCapture.bad_sources = set()
    try:
        cam = Camera(source=0, lock=False)
        cam.open()
        assert cam.source == 0
        cam._apply_switch(1, {})
        assert cam.source == 1, f"expected source 1 after switch, got {cam.source}"
        cam.release()
    finally:
        camera_mod.cv2.VideoCapture = orig
    print("[camera-switch-test] _apply_switch swaps to the new device OK")


def test_apply_switch_overrides_opts():
    orig = _patched()
    FakeCapture.bad_sources = set()
    try:
        cam = Camera(source=0, lock=False, target_brightness=90.0)
        cam.open()
        cam._apply_switch(1, {"target_brightness": 120.0})
        assert cam.source == 1
        assert cam.target_brightness == 120.0
        cam.release()
    finally:
        camera_mod.cv2.VideoCapture = orig
    print("[camera-switch-test] _apply_switch applies per-device opt overrides OK")


def test_failed_open_reverts_to_previous():
    orig = _patched()
    FakeCapture.bad_sources = {2}          # source 2 refuses to open
    try:
        cam = Camera(source=0, lock=False)
        cam.open()
        assert cam.source == 0
        cam._apply_switch(2, {})           # should fail and revert
        assert cam.source == 0, f"expected revert to 0, got {cam.source}"
        assert cam.cap is not None and cam.cap.isOpened(), "prior device not reopened"
        cam.release()
    finally:
        FakeCapture.bad_sources = set()
        camera_mod.cv2.VideoCapture = orig
    print("[camera-switch-test] failed open reverts to the previous camera OK")


def main():
    """Run all camera-switch tests."""
    test_switch_to_flags_and_take_pending_clears()
    test_apply_switch_swaps_source()
    test_apply_switch_overrides_opts()
    test_failed_open_reverts_to_previous()
    print("[camera-switch-test] OK")


if __name__ == "__main__":
    main()
