"""Focused contracts for decoupled metrics, RealSense profiles, and debug state."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import Result, Visibility
from core.realsense_camera import RealSenseCamera
from core.runtime_metrics import RuntimeMetrics
from webui.debug_server import DebugServer, build_debug_payload, safe_json


def test_runtime_metrics_separate_capture_preview_and_analysis():
    metrics = RuntimeMetrics()
    base = 100.0
    for i in range(31):
        metrics.note_capture(30.0, base + i / 30.0)
        metrics.note_preview(base + i / 30.0)
        if i % 3 == 0:
            metrics.note_analysis(i, 55.0, base + i / 30.0)
    snap = metrics.snapshot(now=base + 1.0)
    assert snap["capture_fps"] == 30.0
    assert 29.0 <= snap["preview_fps"] <= 31.0
    assert 9.0 <= snap["analysis_fps"] <= 11.0
    assert snap["skipped_analysis_frames"] == 20


def test_realsense_profile_ranking_prefers_depth_then_resolution():
    colors = {(1920, 1080, 30), (1280, 720, 30), (640, 480, 30)}
    depths = {(848, 480, 30), (640, 480, 30)}
    ranked = RealSenseCamera.rank_profile_candidates(
        colors, depths, (1920, 1080), 30, auto_resolution=True)
    assert ranked[0][:3] == ("color+depth", 1920, 1080)
    assert ranked[0][3:5] == (848, 480)
    assert [r[0] for r in ranked[:3]] == ["color+depth"] * 3
    assert ranked[3][:3] == ("color-only", 1920, 1080)


def test_realsense_explicit_resolution_does_not_expand_candidates():
    ranked = RealSenseCamera.rank_profile_candidates(
        {(1920, 1080, 30), (1280, 720, 30)}, {(640, 480, 30)},
        (1280, 720), 30, auto_resolution=False)
    assert ranked
    assert all(item[1:3] == (1280, 720) for item in ranked)


def test_private_debug_payload_includes_private_and_redacts_media():
    private = Result("agent", "hypothesis", np.float32(72.5),
                     visibility=Visibility.AGENT_ONLY,
                     conversation_tags=("private",))
    payload = build_debug_payload([private], {"capture_fps": 30.0},
                                  {"frame": np.zeros((2, 2)),
                                   "url": "data:image/jpeg;base64,AAAA"})
    encoded = json.dumps(payload)
    assert '"visibility": "agent_only"' in encoded
    assert '"value": 72.5' in encoded
    assert "AAAA" not in encoded
    assert encoded.count("<redacted-media>") == 2
    assert safe_json(float("nan")) is None


def test_debug_server_binds_loopback_and_serves_readable_private_dashboard():
    server = DebugServer(lambda: {"ok": True}, port=0)
    server.start()
    try:
        host, port = server._httpd.server_address
        assert host == "127.0.0.1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/debug") as response:
            html = response.read().decode("utf-8")
            assert response.headers["Cache-Control"] == "no-store"
            assert "Care Monitor — Private Debug" in html
            assert "fetch('/debug/state'" in html
            assert "textContent" in html
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/debug/state") as response:
            assert json.load(response) == {"ok": True}
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/data")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("private server must not serve companion routes")
    finally:
        server.stop()
