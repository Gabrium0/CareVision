"""Focused contracts for iPad CLI capture policy and live preview routing."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from main import _build_ipad_capture_config, _uses_decoupled_display


ROOT = Path(__file__).resolve().parent.parent


def test_capture_config_keeps_auto_resolution_adaptive():
    assert _build_ipad_capture_config(20.0, None) == {
        "fps": 20.0,
        "resolution": {"mode": "auto"},
    }


def test_capture_config_preserves_an_explicit_fixed_resolution():
    assert _build_ipad_capture_config(24.0, (1280, 720)) == {
        "fps": 24.0,
        "resolution": {"mode": "fixed", "width": 1280, "height": 720},
    }


@pytest.mark.parametrize(
    ("fps", "size"),
    [(0.0, None), (float("nan"), None), (31.0, None),
     (20.0, (0, 720)), (20.0, (1280, 5000))],
)
def test_capture_config_rejects_values_outside_browser_bounds(fps, size):
    with pytest.raises(ValueError):
        _build_ipad_capture_config(fps, size)


@pytest.mark.parametrize("source", ["0", "realsense", "rs", "d435i", "ipad",
                                     "browser", "webrtc:camera"])
def test_live_latest_frame_sources_use_decoupled_display(source):
    assert _uses_decoupled_display(True, source) is True


@pytest.mark.parametrize(
    ("display", "source"),
    [(False, "ipad"), (True, "replay:client_demo"), (True, "clip.mp4")],
)
def test_decoupled_display_keeps_existing_gates(display, source):
    assert _uses_decoupled_display(display, source) is False


def test_decoupled_display_survives_switches_to_and_from_ipad():
    assert _uses_decoupled_display(True, "clip.mp4", "ipad") is True
    assert _uses_decoupled_display(True, "ipad", "clip.mp4") is True
    assert _uses_decoupled_display(True, "replay:client_demo", "ipad") is True


def test_video_transport_is_rejected_before_runtime_startup():
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--ipad-transport", "video"],
        cwd=ROOT, capture_output=True, text=True, timeout=20, check=False)

    assert result.returncode == 2
    assert "--ipad-transport video is not implemented" in result.stderr
    assert "--ipad-transport datachannel" in result.stderr
