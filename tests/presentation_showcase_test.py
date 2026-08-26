"""Pytest wrapper so the presentation showcase stays green under CI.

The narrated runner in tests/presentation_showcase.py is the deliverable; this
file simply asserts that every scenario that ran actually passed, so a
regression in any demonstrated feature fails the suite.

Run standalone:  python -m pytest tests/presentation_showcase_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from presentation_showcase import run_all


def test_every_showcase_scenario_passes():
    """Each non-skipped showcase scenario holds all of its checks."""
    results = run_all()
    ran = [s for s in results if not s.skipped]
    assert ran, "no showcase scenarios ran"
    failures = [s.key for s in ran if not s.passed]
    assert not failures, f"showcase scenarios failed: {failures}"
