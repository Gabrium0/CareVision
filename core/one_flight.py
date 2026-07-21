"""Tiny daemon-backed, bounded one-flight executor."""
from __future__ import annotations

import threading
from concurrent.futures import Future


class DaemonOneFlight:
    """Run at most one callable without creating non-daemon executor threads."""
    def __init__(self, name: str):
        self.name = str(name)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._future: Future | None = None
        self._closed = False

    def submit(self, fn, *args, **kwargs) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("one-flight worker is closed")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("one-flight worker is busy")
            future = Future()
            self._future = future

            def run():
                if not future.set_running_or_notify_cancel():
                    return
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:  # Future must preserve worker errors
                    future.set_exception(exc)

            self._thread = threading.Thread(target=run, name=self.name, daemon=True)
            self._thread.start()
            return future

    def shutdown(self, wait: bool = False, cancel_futures: bool = True,
                 timeout: float = 1.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future, thread = self._future, self._thread
        if cancel_futures and future is not None:
            future.cancel()
        if wait and thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))

    def diagnostics(self) -> dict:
        with self._lock:
            return {"alive": bool(self._thread and self._thread.is_alive()),
                    "closed": self._closed,
                    "in_flight": bool(self._future and not self._future.done())}
