"""Hot-reload supervisor for autonomous local development.

The application owns native workers, HTTP servers, and camera resources, so a
fresh child process is a safer reload boundary than re-importing modules in the
running interpreter.  This supervisor watches source and UI files, gracefully
stops the child, and starts it again with the same command.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
WATCH_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".html", ".css", ".js"})
EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".venv", "__pycache__", "graphify-out", "models", "node_modules", "venv",
})
DEFAULT_COMMAND = (
    sys.executable, "main.py", "--source", "replay:dev_hot_reload", "--headless",
    "--debug-endpoint", "--dev-mode", "--no-voice", "--no-moondream",
    "--vitals-log-every", "0",
)


class FileWatcher:
    """Poll a bounded set of source trees and report changed paths."""

    def __init__(self, roots: Iterable[Path], suffixes: Iterable[str] = WATCH_SUFFIXES):
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.suffixes = frozenset(suffix.lower() for suffix in suffixes)
        self._state = self._snapshot()

    def _files(self) -> Iterable[Path]:
        for root in self.roots:
            if root.is_file():
                if root.suffix.lower() in self.suffixes:
                    yield root
                continue
            if not root.exists():
                continue
            for directory, names, filenames in os.walk(root):
                names[:] = [name for name in names if name not in EXCLUDED_DIRS]
                base = Path(directory)
                for filename in filenames:
                    path = base / filename
                    if path.suffix.lower() in self.suffixes:
                        yield path

    def _snapshot(self) -> dict[Path, tuple[int, int]]:
        state: dict[Path, tuple[int, int]] = {}
        for path in self._files():
            try:
                stat = path.stat()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            state[path] = (stat.st_mtime_ns, stat.st_size)
        return state

    def poll(self) -> set[Path]:
        """Return files added, removed, or modified since the previous poll."""
        current = self._snapshot()
        changed = {path for path in current.keys() | self._state.keys()
                   if current.get(path) != self._state.get(path)}
        self._state = current
        return changed


def _start_process(command: Sequence[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    kwargs: dict[str, object] = {"cwd": PROJECT_ROOT, "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    print(f"[reload] starting: {subprocess.list2cmdline(list(command))}", flush=True)
    return subprocess.Popen(list(command), **kwargs)


def _stop_process(process: subprocess.Popen, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
        pass
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


class HotReloader:
    """Run a command and restart it after a debounced source change."""

    def __init__(self, command: Sequence[str], watch_roots: Iterable[Path],
                 interval: float = 0.25, debounce: float = 0.2):
        self.command = tuple(command)
        self.watcher = FileWatcher(watch_roots)
        self.interval = max(0.05, interval)
        self.debounce = max(0.0, debounce)

    def run(self, stop_event: threading.Event | None = None) -> int:
        """Supervise until interrupted; return the last child exit code."""
        stop_event = stop_event or threading.Event()
        process = _start_process(self.command)
        last_exit = 0
        reported_exit: int | None = None
        try:
            while not stop_event.wait(self.interval):
                changed = self.watcher.poll()
                if not changed:
                    if process.poll() is not None:
                        last_exit = int(process.returncode or 0)
                        if reported_exit != process.returncode:
                            reported_exit = process.returncode
                            print(f"[reload] child exited with code {process.returncode}; "
                                  "waiting for a file change", flush=True)
                    continue
                deadline = time.monotonic() + self.debounce
                while not stop_event.is_set() and time.monotonic() < deadline:
                    time.sleep(min(self.interval, max(0.0, deadline - time.monotonic())))
                    more = self.watcher.poll()
                    if more:
                        changed.update(more)
                        deadline = time.monotonic() + self.debounce
                names = ", ".join(str(path.relative_to(PROJECT_ROOT))
                                  if path.is_relative_to(PROJECT_ROOT) else str(path)
                                  for path in sorted(changed))
                print(f"[reload] change detected: {names}", flush=True)
                _stop_process(process)
                process = _start_process(self.command)
                reported_exit = None
            if process.poll() is not None:
                last_exit = int(process.returncode or 0)
        except KeyboardInterrupt:
            print("\n[reload] stopping", flush=True)
        finally:
            _stop_process(process)
        return last_exit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart the app or a test command when source files change.")
    parser.add_argument("--watch", action="append", default=[], metavar="PATH",
                        help="file/tree to watch (repeatable; project root by default)")
    parser.add_argument("--interval", type=float, default=0.25,
                        help="poll interval in seconds (default: 0.25)")
    parser.add_argument("--debounce", type=float, default=0.2,
                        help="quiet period before restart in seconds (default: 0.2)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command after --; defaults to safe debug replay")
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the development supervisor CLI."""
    args = _parse_args(argv)
    command = tuple(args.command) or DEFAULT_COMMAND
    roots = [PROJECT_ROOT / path for path in args.watch] if args.watch else [PROJECT_ROOT]
    if not command:
        raise SystemExit("a child command is required")
    print("[reload] watching " + ", ".join(str(path.resolve()) for path in roots), flush=True)
    if command == DEFAULT_COMMAND:
        print("[reload] private diagnostics: http://127.0.0.1:8771/debug", flush=True)
        print("[reload] machine-readable state: http://127.0.0.1:8771/debug/state", flush=True)
    return HotReloader(command, roots, args.interval, args.debounce).run()


if __name__ == "__main__":
    raise SystemExit(main())
