"""Unit test for Camera._tune_exposure_and_gain() (core/camera.py): fps must
be protected first, brightness second. No real camera needed — a fake
cv2.VideoCapture models a UVC-like sensor where a longer exposure caps
frame rate (integration_time = 2**exposure seconds) and both exposure and
gain contribute to brightness, mirroring the real log this fix targets:

    [camera] locked ... exposure=-3 ... brightness=182
    [camera] WARNING: delivering ~8.3 fps (< 30)

Run standalone:  python tests/camera_tuning_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import core.camera as camera_mod
from core.camera import Camera


class FakeCap:
    """Models a UVC-ish sensor: integration_time = 2**exposure seconds caps
    fps; brightness grows with both exposure (longer integration) and gain.
    Sleeps for the real simulated frame interval (not compressed) so the
    fps numbers the tuning algorithm measures are physically consistent —
    the whole test runs in a few real seconds since exposure quickly shrinks."""
    def __init__(self, request_fps: float, start_exposure: float = -3.0,
                 scene_light: float = 1456.0):
        self.props: dict = {}
        self._exposure = start_exposure
        self._gain = 0.0
        self.request_fps = request_fps
        self.scene_light = scene_light    # ambient light level (matches ~182 brightness at -3/0)

    def isOpened(self) -> bool:
        return True

    def set(self, prop, val) -> bool:
        if prop == cv2.CAP_PROP_EXPOSURE:
            self._exposure = val
        elif prop == cv2.CAP_PROP_GAIN:
            self._gain = val
        else:
            self.props[prop] = val
        return True

    def get(self, prop) -> float:
        if prop == cv2.CAP_PROP_EXPOSURE:
            return self._exposure
        if prop == cv2.CAP_PROP_GAIN:
            return self._gain
        if prop == cv2.CAP_PROP_WB_TEMPERATURE:
            return 4000.0
        return self.props.get(prop, 0.0)

    def read(self):
        integration = max(2.0 ** self._exposure, 1e-6)
        fps_cap = 1.0 / integration                    # the physical bottleneck under test
        frame_interval = 1.0 / min(self.request_fps, fps_cap)
        time.sleep(frame_interval)
        gain_mult = 2.0 ** (self._gain / 24.0)
        brightness = min(255.0, self.scene_light * integration * gain_mult)
        frame = np.full((10, 10, 3), int(round(brightness)), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        pass


def _make_camera(monkeypatch_target, request_fps=30.0, start_exposure=-3.0, **kwargs):
    fake = FakeCap(request_fps=request_fps, start_exposure=start_exposure)
    monkeypatch_target(lambda *a, **k: fake)
    cam = Camera(source=0, lock=True, exposure=None, request_fps=request_fps,
                request_size=(64, 48), target_width=64, settle_seconds=0.02,
                **kwargs)
    return cam, fake


def test_fps_protected_over_brightness():
    """Reproduces the reported scenario (exposure=-3 -> ~8fps ceiling) and
    asserts tuning shrinks exposure to protect fps, using gain (not a long
    exposure) to still reach target brightness."""
    orig = camera_mod.cv2.VideoCapture
    try:
        cam, fake = _make_camera(
            lambda factory: setattr(camera_mod.cv2, "VideoCapture", factory),
            request_fps=30.0, start_exposure=-3.0, target_brightness=90.0)
        cam.open()

        integration = 2.0 ** fake._exposure
        achieved_fps_cap = 1.0 / integration
        print(f"[camera-tuning-test] final exposure={fake._exposure:.1f} "
              f"gain={fake._gain:.1f} fps_cap={achieved_fps_cap:.1f}")

        assert achieved_fps_cap >= 0.9 * cam.request_fps, (
            f"fps ceiling {achieved_fps_cap:.1f} still below 90% of requested "
            f"{cam.request_fps:.1f} — exposure was not shrunk enough")

        bright = cam._measure_brightness()
        print(f"[camera-tuning-test] final brightness={bright:.0f}")
        assert bright >= cam.target_brightness * 0.9, (
            f"brightness {bright:.0f} did not recover toward target "
            f"{cam.target_brightness:.0f} via gain")

        # The whole point: exposure must have SHRUNK from the fps-collapsing
        # start value, not grown further (which is what the old brightness
        # -first logic would have done).
        assert fake._exposure < -3.0, (
            f"exposure {fake._exposure} did not shrink below the starting -3.0")
        cam.release()
        print("[camera-tuning-test] fps-protection OK")
    finally:
        camera_mod.cv2.VideoCapture = orig


def test_gain_disabled_still_protects_fps():
    """Even with gain boosting disabled, fps must still be protected (image
    may stay dim, but frequency-domain vitals need real samples first)."""
    orig = camera_mod.cv2.VideoCapture
    try:
        cam, fake = _make_camera(
            lambda factory: setattr(camera_mod.cv2, "VideoCapture", factory),
            request_fps=30.0, start_exposure=-3.0, target_brightness=90.0,
            allow_gain_boost=False)
        cam.open()
        achieved_fps_cap = 1.0 / (2.0 ** fake._exposure)
        assert achieved_fps_cap >= 0.9 * cam.request_fps, (
            f"fps ceiling {achieved_fps_cap:.1f} not protected with gain boost disabled")
        assert fake._gain == 0.0, "gain must stay untouched when allow_gain_boost=False"
        cam.release()
        print("[camera-tuning-test] gain-disabled fps-protection OK")
    finally:
        camera_mod.cv2.VideoCapture = orig


def main():
    """Run all camera tuning tests."""
    test_fps_protected_over_brightness()
    test_gain_disabled_still_protects_fps()
    print("[camera-tuning-test] OK")


if __name__ == "__main__":
    main()
