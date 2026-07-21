"""Contracts for the dependency-free development hot reloader."""
from pathlib import Path
import sys
import threading
import time

from dev import DEFAULT_COMMAND, FileWatcher, HotReloader, _parse_args


def _wait_for_lines(path: Path, count: int, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text(encoding="utf-8").splitlines()) >= count:
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {count} launches in {path}")


def test_file_watcher_reports_source_changes_and_ignores_generated_trees(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    ignored = tmp_path / "graphify-out"
    ignored.mkdir()
    generated = ignored / "graph.json"
    generated.write_text("{}", encoding="utf-8")
    watcher = FileWatcher([tmp_path])

    source.write_text("VALUE = 200\n", encoding="utf-8")
    generated.write_text('{"changed": true}', encoding="utf-8")

    assert watcher.poll() == {source}


def test_custom_command_is_accepted_after_separator():
    args = _parse_args(["--watch", "core", "--", "python", "-m", "pytest", "-q"])
    assert args.watch == ["core"]
    assert args.command == ["python", "-m", "pytest", "-q"]


def test_default_command_is_private_hardware_free_and_lightweight():
    command = set(DEFAULT_COMMAND)
    assert "replay:dev_hot_reload" in command
    assert {"--headless", "--debug-endpoint", "--dev-mode",
            "--no-voice", "--no-moondream"} <= command


def test_hot_reloader_restarts_a_running_child_after_source_change(tmp_path):
    launches = tmp_path / "launches.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "from pathlib import Path\n"
        "import time\n"
        f"p = Path({str(launches)!r})\n"
        "with p.open('a', encoding='utf-8') as handle:\n"
        "    handle.write('started\\n')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    watched = tmp_path / "watched.py"
    watched.write_text("VALUE = 1\n", encoding="utf-8")
    stop = threading.Event()
    reloader = HotReloader([sys.executable, str(child)], [watched],
                           interval=0.05, debounce=0.05)
    thread = threading.Thread(target=reloader.run, args=(stop,), daemon=True)
    thread.start()
    try:
        _wait_for_lines(launches, 1)
        watched.write_text("VALUE = 2\n", encoding="utf-8")
        _wait_for_lines(launches, 2)
    finally:
        stop.set()
        thread.join(timeout=8)
    assert not thread.is_alive()
