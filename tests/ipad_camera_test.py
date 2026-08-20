"""Unit tests for core/ipad_camera.py's IPadCamera.

No real iPad, WebRTC stack, or network needed: a FakeLink stands in for
core.ipad_link.IPadLink, exposing only the surface IPadCamera actually calls
(`take_frame()`, `status()`, `stop()`). Injected via the `link=` constructor
kwarg, which also means `open()` never tries to build a real link.

Run standalone:  python tests/ipad_camera_test.py
"""
import itertools
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.camera import Camera
from core.ipad_camera import IPadCamera, pack_frame_header


class FakeLink:
    """Minimal stand-in for core.ipad_link.IPadLink: a queue of already
    header-parsed (header_dict, jpeg_bytes) items, popped oldest-first by
    take_frame(). Returns None once drained (or always, if empty from the
    start) -- exactly what a real link does between pushes.

    Delivery is paced (one item at most every `delivery_interval` seconds)
    rather than dumped all at once: IPadCamera.frames() publishes only the
    *latest* decoded frame (like Camera's webcam path), so a consumer that
    is slower than the producer legitimately misses intermediate frames --
    that's the documented drop-oldest contract. Pacing the fake link to
    roughly real arrival cadence (~ every 20ms, faster than the reader
    thread's own polling) lets the test observe every frame the way a real
    ~20fps push actually would, without asserting anything about a
    scenario the design intentionally does not guarantee."""

    def __init__(self, items=None, delivery_interval: float = 0.02):
        self._queue = list(items or [])
        self._lock = threading.Lock()
        self.stopped = False
        self._delivery_interval = delivery_interval
        self._next_at = time.time()

    def take_frame(self):
        with self._lock:
            if not self._queue:
                return None
            now = time.time()
            if now < self._next_at:
                return None
            self._next_at = now + self._delivery_interval
            return self._queue.pop(0)

    def status(self):
        return {"relay": "connected", "ice": "connected", "dc": "open",
                "frames_rx": len(self._queue), "seq_gaps": 0,
                "last_rx_at": time.time(), "clock_source": "fake", "error": None}

    def stop(self):
        self.stopped = True


def _encode(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok, "test setup: JPEG encode failed"
    return buf.tobytes()


def _pull_n(camera: IPadCamera, n: int, timeout: float = 5.0):
    """Drive camera.frames() on a worker thread and collect n FrameContexts."""
    out = []

    def consume():
        for ctx in itertools.islice(camera.frames(), n):
            out.append(ctx)

    t = threading.Thread(target=consume, daemon=True, name="ipad-camera-test-consumer")
    t.start()
    t.join(timeout=timeout)
    return out, t


def test_frames_fire_fast_hooks_like_replay():
    """Rationale: Camera's file/URL path (_frames_sync) never fires fast
    hooks, which silently drops rPPG to the heavy loop's cadence. This test
    guards against the iPad backend regressing the same way -- every pushed
    frame must reach a registered fast hook, not just the ones the (slower)
    heavy detection loop gets around to consuming."""
    items = []
    for i in range(5):
        frame = np.full((60, 80, 3), fill_value=(20 * i, 60, 120), dtype=np.uint8)
        header = {"media_time": i * 0.05, "recv_wall": time.time() + i * 0.001}
        items.append((header, _encode(frame)))

    link = FakeLink(items)
    camera = IPadCamera(source="ipad", link=link)
    hook_calls = []
    camera.register_fast_hook(lambda frame, ts: hook_calls.append((frame, ts)))

    try:
        contexts, thread = _pull_n(camera, 5)
    finally:
        camera.release()

    assert len(contexts) == 5, f"expected 5 FrameContexts, got {len(contexts)}"
    assert len(hook_calls) == 5, f"expected 5 fast-hook calls, got {len(hook_calls)}"
    for frame, ts in hook_calls:
        assert frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3, (
            f"expected BGR (H,W,3) uint8, got shape {frame.shape} dtype {frame.dtype}")
    timestamps = [ts for _, ts in hook_calls]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:])), (
        f"fast-hook timestamps must be strictly increasing, got {timestamps}")
    print("[ipad-camera-test] fast hooks fire for every pushed frame, BGR uint8, "
          "strictly increasing timestamps OK")


def test_frames_generator_survives_starvation():
    """Rationale: Pipeline.run() treats generator exhaustion as end-of-stream
    and exits the whole app. Safari stops delivering frames every time the
    tab is backgrounded, so a starved link must look like a stalled camera,
    never a dead (StopIteration-raising) one."""
    link = FakeLink([])   # take_frame() always returns None
    camera = IPadCamera(source="ipad", link=link)

    escaped = {"stop_iteration": False}
    results = []

    def consume():
        try:
            for ctx in camera.frames():
                results.append(ctx)
        except StopIteration:
            escaped["stop_iteration"] = True

    thread = threading.Thread(target=consume, daemon=True, name="ipad-starvation-consumer")
    thread.start()
    time.sleep(0.3)

    try:
        assert thread.is_alive(), "frames() generator thread died during starvation"
        assert not escaped["stop_iteration"], "StopIteration escaped frames()"
        assert results == [], "no frames should have been yielded (link is empty)"
    finally:
        camera.release()
        thread.join(timeout=2.0)
    print("[ipad-camera-test] frames() survives indefinite starvation without "
          "ending the stream OK")


def test_backpressure_drops_oldest():
    """Exercises IPadLink's own depth-2 drop-oldest deque (core/ipad_link.py).
    IPadLink can be constructed without aiortc installed -- the import is
    lazy, inside the asyncio loop -- so this pushes frames straight through
    the private _ingest_frame() and inspects the deque directly."""
    try:
        from core.ipad_link import IPadLink
    except ImportError as exc:  # pragma: no cover - environment guard
        raise AssertionError(f"IPadLink must be importable without aiortc: {exc}")

    link = IPadLink(relay_url="https://x", room="r", secret="s", code="123456")
    for seq in range(10):
        buf = pack_frame_header(seq, seq * 0.05, 640, 480) + f"payload{seq}".encode()
        link._ingest_frame(buf)

    assert len(link._frames) <= 2, f"expected maxlen-2 deque, got {len(link._frames)} items"
    newest = link.take_frame()
    assert newest is not None
    header, payload = newest
    assert header["seq"] == 9, f"take_frame() must return the newest frame, got seq={header['seq']}"
    assert payload == b"payload9"
    print("[ipad-camera-test] backpressure drops oldest, take_frame returns newest OK")


def test_unknown_camera_opts_are_swallowed():
    """main.py passes ONE camera_opts dict to whichever backend gets built,
    so each backend's **_unused must swallow the other backend's options
    without raising TypeError."""
    cam = IPadCamera(source="ipad", lock=True, exposure=-6, auto_resolution=True,
                     target_brightness=90.0, allow_gain_boost=True)
    assert isinstance(cam, IPadCamera)

    cam2 = Camera(source=0, ipad_room="x", ipad_secret="y")
    assert isinstance(cam2, Camera)
    print("[ipad-camera-test] unknown camera opts swallowed symmetrically OK")


def test_bgr_channel_order():
    """Guards rPPG's px[:, ::-1] BGR->RGB flip: cv2.imencode takes BGR input
    and cv2.imdecode returns BGR output, so a frame that is pure red in BGR
    terms (channel index 2 set) must still have channel index 2 dominant
    after the iPad's push -> JPEG -> decode round trip."""
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[:, :, 2] = 255   # pure red in BGR (B=0, G=0, R=255)
    header = {"media_time": 0.0, "recv_wall": time.time()}
    link = FakeLink([(header, _encode(frame))])
    camera = IPadCamera(source="ipad", link=link)

    captured = {}
    camera.register_fast_hook(lambda f, ts: captured.setdefault("frame", f))

    try:
        _pull_n(camera, 1)
    finally:
        camera.release()

    assert "frame" in captured, "fast hook never fired"
    decoded = captured["frame"]
    b_mean = float(decoded[:, :, 0].mean())
    g_mean = float(decoded[:, :, 1].mean())
    r_mean = float(decoded[:, :, 2].mean())
    assert r_mean > b_mean and r_mean > g_mean, (
        f"expected channel index 2 (R in BGR) to dominate a red frame, "
        f"got B={b_mean:.1f} G={g_mean:.1f} R={r_mean:.1f}")
    print(f"[ipad-camera-test] BGR channel order preserved through JPEG round trip "
          f"(B={b_mean:.1f} G={g_mean:.1f} R={r_mean:.1f}) OK")


def main():
    """Run all iPad camera tests."""
    test_frames_fire_fast_hooks_like_replay()
    test_frames_generator_survives_starvation()
    test_backpressure_drops_oldest()
    test_unknown_camera_opts_are_swallowed()
    test_bgr_channel_order()
    print("[ipad-camera-test] OK")


if __name__ == "__main__":
    main()
