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

from typing import Callable, Iterator, Optional

from .camera import Camera
from .context import FrameContext
from .realsense_camera import RealSenseCamera, is_realsense_source


def make_backend(source, opts: dict | None = None):
    """Build the raw backend for a source: RealSense or UVC/file Camera."""
    opts = opts or {}
    if is_realsense_source(source):
        return RealSenseCamera(source=source, **opts)
    return Camera(source=source, **opts)


def make_camera(source, opts: dict | None = None) -> "SwitchableCamera":
    """Build the switchable camera the pipeline consumes."""
    return SwitchableCamera(source, opts)


class SwitchableCamera:
    """Facade holding the current backend; swaps it on cross-type switches."""

    def __init__(self, source, opts: dict | None = None):
        self.inner = make_backend(source, opts)
        self._hooks: list[Callable] = []
        self._cross_pending: Optional[tuple] = None

    @property
    def current_fps(self) -> float:
        """Delivered fps of the active backend."""
        return self.inner.current_fps

    def register_fast_hook(self, hook: Callable) -> None:
        """Register on the facade so hooks survive backend swaps."""
        self._hooks.append(hook)
        self.inner.register_fast_hook(hook)

    def switch_to(self, source, opts: dict | None = None) -> None:
        """Same-type switches ride the backend's own path; cross-type ones
        are queued for the frames loop (never swapped from a UI thread)."""
        if is_realsense_source(source) == isinstance(self.inner, RealSenseCamera):
            self.inner.switch_to(source, opts)
        else:
            self._cross_pending = (source, opts or {})

    def _apply_cross_switch(self) -> None:
        """Release the old backend first, then open the new one.  Opening
        both simultaneously can exhaust USB bandwidth on some host
        controllers (e.g. laptop hubs sharing bandwidth between the
        internal webcam and an external RealSense).  If the new backend
        fails to open, we try to restore the old one — it was just
        released so it should be available again."""
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
                self.inner = old
                for hook in self._hooks:
                    old.register_fast_hook(hook)
            except Exception as e2:  # noqa: BLE001
                print(f"[camera] also failed to restore {old.source!r} "
                      f"({e2}); stream dead")
            return
        for hook in self._hooks:
            new.register_fast_hook(hook)
        self.inner = new

    def frames(self) -> Iterator[FrameContext]:
        """Yield from the active backend, swapping backends between yields
        when a cross-type switch is pending."""
        while True:
            for ctx in self.inner.frames():
                yield ctx
                if self._cross_pending is not None:
                    break
            if self._cross_pending is None:
                return              # genuine end of stream (file done, device dead)
            self._apply_cross_switch()

    def release(self) -> None:
        """Release the active backend."""
        self.inner.release()
