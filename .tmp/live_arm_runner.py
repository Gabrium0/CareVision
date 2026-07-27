"""Temporary isolated-cwd launcher for Codex runtime verification."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RUNTIME = Path(__file__).resolve().parent / "live-arm-runtime"
RUNTIME.mkdir(exist_ok=True)
os.chdir(RUNTIME)
sys.argv = [
    str(PROJECT / "main.py"),
    "--source", "replay:dev_hot_reload",
    "--headless", "--debug-endpoint", "--debug-port", "8871",
    "--dev-mode", "--no-voice", "--no-moondream",
    "--vitals-log-every", "0",
]
runpy.run_path(str(PROJECT / "main.py"), run_name="__main__")
