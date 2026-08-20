"""Camera factory + cross-backend runtime switching.

`make_camera` is the single indirection `main.py` uses so `--source
realsense` (or `rs`/`d435i`) builds the depth-capable
`core.realsense_camera.RealSenseCamera` while every other value keeps the
existing UVC/file `core.camera.Camera` path untouched.

`SwitchableCamera` preserves the 'c'-hotkey runtime toggle *across backend
types*: same-type switches (laptop cam <-> OV2735) delegate to the inner
backend's own proven `switch_to` teardown/reopen, while UVC <-> RealSense
swaps rebuild the whole backend between yields — opening the new device
before releasing the old one, so a failed open (SDK missing, device
unplugged) falls back to the still-running current camera and the stream
never dies, matching `Camera._apply_switch`'s revert contract.
"""
from __future__ import annotations

import threading
from typing import Callable, Iterator, Optional

from .camera import Camera
from .context import FrameContext
from .ipad_camera import IPadCamera, is_ipad_source
from .realsense_camera import RealSenseCamera, is_realsense_source
from .replay import ReplayCamera


def make_backend(source, opts: dict | None = None):
    """Build the raw backend for a source: replay, RealSense, iPad, or UVC/file."""
    opts = opts or {}
    if isinstance(source, str) and source.startswith("replay:"):
        return ReplayCamera(source=source, **opts)
    if is_realsense_source(source):
        return RealSenseCamera(source=source, **opts)
    if is_ipad_source(source):
        return IPadCamera(source=source, **opts)
    return Camera(source=source, **opts)


def _kind(source) -> str:
    """Classify a source string by the backend that will serve it."""
    if isinstance(source, str) and source.startswith("replay:"):
        return "replay"
    if is_realsense_source(source):
        return "realsense"
    if is_ipad_source(source):
        return "ipad"
    return "uvc"


def _kind_of(backend) -> str:
    """Classify a live backend the same way `_kind` classifies a source."""
    if isinstance(backend, ReplayCamera):
        return "replay"
    if isinstance(backend, RealSenseCamera):
        return "realsense"
    if isinstance(backend, IPadCamera):
        return "ipad"
    return "uvc"


def make_camera(source, opts: dict | None = None) -> "SwitchableCamera":
    """Build the switchable camera the pipeline consumes."""
    return SwitchableCamera(source, opts)


class SwitchableCamera:
    """Facade holding the current backend; swaps it on cross-type switches."""

    def __init__(self, source, opts: dict | None = None):
        self.inner = make_backend(source, opts)
        self._hooks: list[Callable] = []
        self._ipad_control_handler: Optional[Callable] = None
        self._cross_pending: Optional[tuple] = None
        self._switch_lock = threading.Lock()

    @property
    def current_fps(self) -> float:
        """Delivered fps of the active backend."""
        return self.inner.current_fps

    def latest_frame(self):
        """Return the active backend's newest frame for live preview."""
        with self._switch_lock:
            inner = self.inner
        getter = getattr(inner, "latest_frame", None)
        return getter() if getter is not None else None

    def diagnostics(self) -> dict:
        with self._switch_lock:
            inner = self.inner
        getter = getattr(inner, "diagnostics", None)
        return getter() if getter is not None else {"backend": type(inner).__name__}

    def register_fast_hook(self, hook: Callable) -> None:
        """Register on the facade so hooks survive backend swaps."""
        self._hooks.append(hook)
        self.inner.register_fast_hook(hook)

    def switch_to(self, source, opts: dict | None = None) -> None:
        """Same-type switches ride the backend's own path; cross-type ones
        are queued for the frames loop (never swapped from a UI thread)."""
        target, current = _kind(source), _kind_of(self.inner)
        # Only UVC and RealSense backends can retarget in place. Replay has no
        # switch_to at all, and re-pairing an iPad means a fresh peer connection
        # and pairing code, so both kinds always take the rebuild path.
        if target == current and target in ("uvc", "realsense"):
            self.inner.switch_to(source, opts)
        else:
            with self._switch_lock:
                self._cross_pending = (source, opts or {})

    def _apply_cross_switch(self) -> None:
        """Release the old backend first, then open the new one.  Opening
        both simultaneously can exhaust USB bandwidth on some host
        controllers (e.g. laptop hubs sharing bandwidth between the
        internal webcam and an external RealSense).  If the new backend
        fails to open, we try to restore the old one — it was just
        released so it should be available again."""
        with self._switch_lock:
            source, opts = self._cross_pending
            self._cross_pending = None
            old = self.inner
        old.release()
        new = make_backend(source, opts)
        try:
            new.open()
        except Exception as e:  # noqa: BLE001
            print(f"[camera] failed to open {source!r} ({e}); "
                  f"restoring {old.source!r}")
            try:
                old.open()
                with self._switch_lock:
                    self.inner = old
                for hook in self._hooks:
                    old.register_fast_hook(hook)
            except Exception as e2:  # noqa: BLE001
                print(f"[camera] also failed to restore {old.source!r} "
                      f"({e2}); stream dead")
            return
        for hook in self._hooks:
            new.register_fast_hook(hook)
        if self._ipad_control_handler is not None and isinstance(new, IPadCamera):
            new.set_control_handler(self._ipad_control_handler)
        with self._switch_lock:
            self.inner = new

    def frames(self) -> Iterator[FrameContext]:
        """Yield from the active backend, swapping backends between yields
        when a cross-type switch is pending."""
        while True:
            for ctx in self.inner.frames():
                yield ctx
                with self._switch_lock:
                    pending = self._cross_pending is not None
                if pending:
                    break
            with self._switch_lock:
                pending = self._cross_pending is not None
            if not pending:
                return              # genuine end of stream (file done, device dead)
            self._apply_cross_switch()

    def release(self) -> None:
        """Release the active backend."""
        self.inner.release()

    def replay_control(self, action: str, value: float | None = None) -> dict:
        """Forward dashboard controls only when the active source is replay."""
        if not isinstance(self.inner, ReplayCamera):
            raise RuntimeError("active source is not a replay")
        return self.inner.control(action, value)

    def replay_status(self) -> dict | None:
        """Return active replay state or None for live sources."""
        return self.inner.status() if isinstance(self.inner, ReplayCamera) else None

    def ipad_control(self, payload: dict) -> None:
        """Push a state update to the paired iPad; no-op for other sources."""
        if isinstance(self.inner, IPadCamera):
            self.inner.send_control(payload)

    def ipad_status(self) -> dict | None:
        """Return active iPad link state or None for other sources."""
        return self.inner.status() if isinstance(self.inner, IPadCamera) else None

    def set_ipad_control_handler(self, handler) -> None:
        """Attach the iPad's control callback, surviving backend rebuilds.

        Held on the facade like the fast hooks are, so switching to an iPad via
        the 'c' hotkey (which rebuilds the backend) still gets a live handler.
        """
        self._ipad_control_handler = handler
        if isinstance(self.inner, IPadCamera):
            self.inner.set_control_handler(handler)
