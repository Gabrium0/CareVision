"""iPad-as-camera backend: frames pushed in over WebRTC, not pulled from a device.

Why this exists: the pipeline normally *pulls* frames (`cv2.VideoCapture.read()`).
An iPad in Safari can only *push* them, and only over WebRTC — from an HTTPS
page, plain `http://`/`ws://` to a non-TLS laptop is mixed-content blocked, while
WebRTC is exempt because DTLS-SRTP encrypts it unconditionally.

This module deliberately imports neither asyncio nor aiortc, so the backend can
be constructed and unit-tested without the WebRTC stack installed — the same
split `core/realsense_camera.py` uses to keep `pyrealsense2` optional. All the
networking lives in `core.ipad_link`, which this module imports lazily inside
`open()`.

Two design points carry most of the weight:

* **The reader thread fires the fast hooks.** `Camera` only fires them on its
  threaded webcam path; the file/URL path never does, which is why those sources
  silently drop rPPG to the heavy loop's cadence. `ReplayCamera.frames()` fires
  them for the same reason. Vitals depend on this.
* **`frames()` never returns when frames stop arriving.** `Pipeline.run()` treats
  generator exhaustion as end-of-stream and shuts the whole app down, so a
  backgrounded Safari tab must look like a stalled camera, not a dead one.
"""
from __future__ import annotations

import struct
import threading
import time
from collections import deque
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .context import FrameContext

# Source strings that select this backend. "ipad" is the documented spelling;
# the other two are accepted because they are what people type.
IPAD_SOURCES = ("ipad", "browser", "webrtc")

# Wire format for one pushed frame: a fixed 24-byte header then the JPEG bytes.
# Binary rather than JSON so the receive path costs a single struct.unpack
# instead of a parse per frame at 20fps.
FRAME_MAGIC = b"IPF1"
FRAME_HEADER = struct.Struct("<4sIdHHHH")   # magic, seq, mediaTime, w, h, flags, reserved
FRAME_HEADER_SIZE = FRAME_HEADER.size       # 24


def is_ipad_source(source) -> bool:
    """True when `--source` selects the iPad backend."""
    if not isinstance(source, str):
        return False
    name = source.strip().lower()
    return name in IPAD_SOURCES or name.split(":", 1)[0] in IPAD_SOURCES


def pack_frame_header(seq: int, media_time: float, width: int, height: int,
                      chunk_index: int = 0, chunk_count: int = 1) -> bytes:
    """Build the 24-byte header the iPad page prepends to every JPEG chunk.

    The last two u16 slots carry chunking rather than flags: aiortc advertises
    `a=max-message-size:65536`, so a quality-0.92 frame does not fit in one SCTP
    message and Safari throws on send. Splitting keeps full image quality
    instead of trading it away to squeeze under the limit.
    """
    return FRAME_HEADER.pack(FRAME_MAGIC, int(seq) & 0xFFFFFFFF, float(media_time),
                             int(width) & 0xFFFF, int(height) & 0xFFFF,
                             int(chunk_index) & 0xFFFF, int(chunk_count) & 0xFFFF)


def unpack_frame_header(buf: bytes) -> Optional[dict]:
    """Parse a pushed frame's header, or None if it isn't one of ours.

    Returns None rather than raising: this runs on the asyncio receive path,
    where an exception would tear down the data channel over one bad message.
    """
    if len(buf) < FRAME_HEADER_SIZE:
        return None
    magic, seq, media_time, width, height, chunk_index, chunk_count = \
        FRAME_HEADER.unpack_from(buf, 0)
    if magic != FRAME_MAGIC:
        return None
    return {"seq": seq, "media_time": media_time, "width": width,
            "height": height, "flags": chunk_index,
            "chunk_index": chunk_index, "chunk_count": max(1, chunk_count)}


def jpeg_subsampling(data: bytes) -> Optional[str]:
    """Report a JPEG's chroma subsampling ("4:4:4", "4:2:0", ...) from its SOF.

    Worth measuring rather than assuming: WebKit may encode 4:4:4 at high
    quality, and if it does, the `compressed_channels` workaround in
    config/modules.yaml (green-instead-of-blue, chosen because green rides
    full-resolution luma under 4:2:0) stops being necessary.
    """
    if data[:2] != b"\xff\xd8":
        return None
    i, end = 2, len(data)
    while i + 11 < end:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        # SOF0/1/2/3 all carry the component table we need; the other SOFn
        # markers are arithmetic/hierarchical variants Safari never emits.
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            ncomp = data[i + 9]
            if ncomp < 1:
                return None
            if ncomp == 1:
                return "grayscale"
            h_samp = data[i + 11] >> 4
            v_samp = data[i + 11] & 0x0F
            return {(1, 1): "4:4:4", (2, 2): "4:2:0",
                    (2, 1): "4:2:2", (1, 2): "4:4:0"}.get((h_samp, v_samp),
                                                          f"{h_samp}x{v_samp}")
        if marker == 0xDA:      # start of scan; no SOF found before the image data
            return None
        i += 2 + seg_len
    return None


class CaptureClock:
    """Map the iPad's monotonic capture clock onto local wall time.

    We never consume the iPad's *absolute* time (its wall clock can be minutes
    off, and NTP can step it mid-session) — only its relative spacing, anchored
    to ours by a single offset.

    The offset is the **minimum** observed `recv_wall - media_time` over a
    rolling window, not the mean. The smallest one-way delay is the least-queued
    sample, so a minimum is immune to transient queueing; a mean tracks queueing
    delay, and that slowly-varying error lands inside the 0.7-3Hz band rPPG
    bandpasses, as well as inflating the timestamp irregularity that
    `rppg_input_quality` scores.
    """

    def __init__(self, window_seconds: float = 10.0, resync_gap: float = 2.0,
                 max_drift: float = 0.5):
        self.window_seconds = window_seconds
        self.resync_gap = resync_gap        # silence longer than this invalidates the anchor
        self.max_drift = max_drift          # mapped time this far from arrival = re-anchor
        self._samples: deque[tuple[float, float]] = deque()
        self._last_media: Optional[float] = None
        self._last_recv: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._resyncs = 0
        self._last_reason = ""

    def reset(self, reason: str) -> None:
        """Drop the anchor so the next frame re-establishes it.

        `_last_ts` deliberately survives: downstream buffers (TimedBuffer, the
        staleness check in Pipeline._fast_hook) require monotonic timestamps
        across a resync, not just within one.
        """
        self._samples.clear()
        self._last_media = None
        self._last_recv = None
        self._resyncs += 1
        self._last_reason = reason

    def map(self, media_time: float, recv_wall: float) -> float:
        """Convert a peer capture time to a local wall-clock timestamp."""
        if self._last_media is not None:
            if media_time < self._last_media - 1e-6:
                # mediaTime restarts at 0 on reload or track replacement
                self.reset("media_time_backwards")
            elif recv_wall - (self._last_recv or recv_wall) > self.resync_gap:
                self.reset("gap")

        delta = recv_wall - media_time
        self._samples.append((recv_wall, delta))
        while len(self._samples) > 1 and recv_wall - self._samples[0][0] > self.window_seconds:
            self._samples.popleft()

        ts = media_time + min(d for _, d in self._samples)
        if abs(ts - recv_wall) > self.max_drift:
            # The anchor has gone stale (clock drift, a long stall); re-anchor on
            # this frame rather than emitting timestamps that disagree with
            # arrival by more than the pipeline's staleness tolerance.
            self.reset("drift")
            self._samples.append((recv_wall, delta))
            ts = recv_wall

        self._last_media, self._last_recv = media_time, recv_wall
        if self._last_ts is not None and ts <= self._last_ts:
            ts = self._last_ts + 1e-4
        self._last_ts = ts
        return ts

    def diagnostics(self) -> dict:
        return {"resyncs": self._resyncs, "last_resync_reason": self._last_reason,
                "offset_samples": len(self._samples)}


class IPadCamera:
    """Camera backend fed by JPEG frames an iPad pushes over a WebRTC data channel."""

    def __init__(self, source: str = "ipad", link=None, target_width: int = 960,
                 request_fps: float = 20.0, request_size: tuple = (640, 480),
                 ipad_relay_url: Optional[str] = None, ipad_room: Optional[str] = None,
                 ipad_secret: Optional[str] = None, ipad_code: Optional[str] = None,
                 ipad_stun: tuple = (), ipad_transport: str = "datachannel",
                 ipad_pair_ttl: float = 600.0, connect_timeout: float = 5.0,
                 ipad_listen_host: Optional[str] = None,
                 ipad_listen_port: Optional[int] = None,
                 min_fps: float = 10.0, **_unused):
        # **_unused swallows UVC-only camera_opts (lock/exposure/gain...) so
        # main.py can pass one opts dict to whichever backend gets built.
        self.source = source
        self.target_width = target_width
        self.request_fps = request_fps
        self.request_size = request_size
        self.min_fps = min_fps
        self.connect_timeout = connect_timeout
        self._link_opts = {"relay_url": ipad_relay_url, "room": ipad_room,
                           "secret": ipad_secret, "code": ipad_code,
                           "stun": tuple(ipad_stun), "transport": ipad_transport,
                           "pair_ttl": ipad_pair_ttl,
                           "listen_host": ipad_listen_host,
                           "listen_port": ipad_listen_port}
        self._link = link                  # injected in tests; built lazily in open()
        self._owns_link = link is None

        self._fast_hooks: list[Callable[[np.ndarray, float], None]] = []
        self._capture_reset_hooks: list[Callable[[], None]] = []
        self._last_capture_profile_generation: Optional[int] = None
        self._clock = CaptureClock()
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: Optional[float] = None
        self._latest_index = -1
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()

        self._fps_smooth = float(request_fps)
        self._last_t: Optional[float] = None
        self._frames_decoded = 0
        self._decode_errors = 0
        self._subsampling: Optional[str] = None
        self._last_frame_at: Optional[float] = None
        self._waiting_since: Optional[float] = None
        self._waiting_notice = 0.0

    # ---- camera contract -------------------------------------------------

    @property
    def current_fps(self) -> float:
        """Smoothed delivered fps, readable from other threads/modules."""
        return self._fps_smooth

    def register_fast_hook(self, hook: Callable[[np.ndarray, float], None]) -> None:
        """Register `hook(frame, timestamp)` to run on the reader thread for
        every pushed frame, ahead of the slower per-frame detection pipeline.
        Vitals depend on this: without it heart-rate/SpO2 fall back to sampling
        once per heavy-loop tick."""
        self._fast_hooks.append(hook)

    def register_capture_reset_hook(self, hook: Callable[[], None]) -> None:
        """Run ``hook`` before pixels from a changed capture profile are used."""
        if hook not in self._capture_reset_hooks:
            self._capture_reset_hooks.append(hook)

    def open(self) -> None:
        """Bring up the link and start the reader thread.

        Fails fast when the relay is unreachable — SwitchableCamera releases the
        old backend *before* opening a new one and only reverts if open() raises,
        so a slow failure here would strand the pipeline with no camera. Pairing
        with the iPad stays asynchronous; only reaching the relay is awaited.
        """
        if self._link is None:
            link = self._build_link()
            link.start()
            deadline = time.time() + self.connect_timeout
            while time.time() < deadline:
                status = link.status()
                if status.get("relay") == "connected":
                    break
                if status.get("error"):
                    link.stop()
                    raise RuntimeError(f"iPad relay error: {status['error']}")
                time.sleep(0.05)
            else:
                link.stop()
                raise RuntimeError(
                    f"cannot reach iPad relay at {self._link_opts['relay_url']!r} "
                    f"within {self.connect_timeout:.0f}s")
            self._link = link
        self._waiting_since = time.time()
        self._start_reader()

    def frames(self) -> Iterator[FrameContext]:
        """Yield the newest pushed frame, dropping any the heavy loop missed.

        Deliberately does not end when frames stop arriving: Pipeline.run()
        treats a finished generator as end-of-stream and exits the whole app,
        and Safari stops delivering frames every time the tab is backgrounded.
        """
        if self._reader_thread is None:
            self.open()
        last_seen = -1
        out_idx = 0
        while not self._reader_stop.is_set():
            with self._latest_lock:
                idx, frame, ts = self._latest_index, self._latest_frame, self._latest_ts
            if idx == last_seen or frame is None:
                time.sleep(0.001)
                continue
            last_seen = idx
            ctx = FrameContext(frame=frame, timestamp=ts, frame_index=out_idx,
                               fps=self._fps_smooth)
            ctx.extras["capture_index"] = idx
            yield ctx
            out_idx += 1

    def latest_frame(self) -> Optional[tuple[int, np.ndarray, float]]:
        """Return the newest pushed frame for a non-consuming preview."""
        with self._latest_lock:
            if self._latest_frame is None or self._latest_ts is None:
                return None
            return self._latest_index, self._latest_frame, self._latest_ts

    def diagnostics(self) -> dict:
        with self._latest_lock:
            frame = self._latest_frame
            resolution = ([int(frame.shape[1]), int(frame.shape[0])]
                          if frame is not None else None)
        link = self._link.status() if self._link is not None else {"relay": "down"}
        received = int(link.get("frames_rx", 0))
        gaps = int(link.get("seq_gaps", 0))
        return {"backend": "ipad", "resolution": resolution,
                "depth_available": False, "requested_fps": self.request_fps,
                "delivered_fps": round(self._fps_smooth, 2),
                "frames_decoded": self._frames_decoded,
                "decode_errors": self._decode_errors,
                "frame_loss_ratio": round(gaps / max(received + gaps, 1), 4),
                "jpeg_subsampling": self._subsampling,
                "clock_source": link.get("clock_source"),
                "clock": self._clock.diagnostics(),
                "link": link}

    def release(self) -> None:
        """Stop the reader thread and tear the link down."""
        self._stop_reader()
        if self._link is not None and self._owns_link:
            self._link.stop()
            self._link = None

    def switch_to(self, source, opts: dict | None = None) -> None:
        """Not supported in place — the factory rebuilds this backend instead.

        Re-pairing means a new peer connection and a new pairing code, so there
        is nothing meaningful to swap without a rebuild. SwitchableCamera routes
        every iPad switch through _cross_pending for exactly this reason.
        """
        raise RuntimeError("iPad camera re-pairs by rebuild, not in-place switch")

    # ---- iPad-specific control (mirrors ReplayCamera.control/status) ------

    def status(self) -> dict:
        """Link and capture status for the dashboard and the /data payload."""
        return self.diagnostics()

    def send_control(self, payload: dict) -> None:
        """Push a state update down the control channel to the iPad page."""
        if self._link is not None:
            self._link.send_control(payload)

    def set_control_handler(self, handler) -> None:
        """Attach the callback that services control messages from the iPad.

        Late-bound on purpose: the handler closes over the pipeline, which does
        not exist yet when camera options are assembled, and the link itself is
        not built until open().
        """
        self._link_opts["on_control"] = handler
        if self._link is not None:
            self._link.set_control_handler(handler)

    def attach_audio_bus(self, bus) -> None:
        """Late-bind the shared audio bus for device-mic frames.

        Like set_control_handler, this survives the link not existing yet:
        options recorded here are applied to every link built by open(), and to
        a live link immediately.
        """
        self._link_opts["audio_bus"] = bus
        if self._link is not None:
            self._link.audio_bus = bus

    def send_agent_audio(self, samples, rate: int = 16000,
                         channels: int = 1) -> None:
        """Route one chunk of agent speech to the paired device (no-op unpiped)."""
        if self._link is not None:
            self._link.send_audio(samples, rate, channels)

    # ---- internals -------------------------------------------------------

    def _build_link(self):
        """Build the transport for this source, importing its stack lazily.

        ``lan`` is the native-app path: a direct WebSocket to the laptop over the
        hotspot, no relay/WebRTC. ``datachannel`` (default) keeps the browser
        WebRTC link. Both present the same duck-typed interface, so nothing else
        in this class changes between them.
        """
        transport = self._link_opts.get("transport", "datachannel")
        if transport == "lan":
            from .ipad_lan_link import IPadLanLink   # noqa: PLC0415 - lazy by design
            return IPadLanLink(**self._link_opts)
        from .ipad_link import IPadLink       # noqa: PLC0415 - lazy by design
        return IPadLink(**self._link_opts)

    def _start_reader(self) -> None:
        if self._reader_thread is not None:
            return
        self._reader_stop.clear()
        self._last_capture_profile_generation = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="ipad-reader")
        self._reader_thread.start()

    def _stop_reader(self) -> None:
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

    def _prep_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] > self.target_width:
            scale = self.target_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return frame

    def _note_fps(self, now: float) -> None:
        if self._last_t is not None:
            dt = max(now - self._last_t, 1e-3)
            self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * (1.0 / dt)
        self._last_t = now

    def _note_waiting(self) -> None:
        """Say something while unpaired — an iPad that never connects is
        otherwise indistinguishable from a camera that simply sees nothing."""
        if self._last_frame_at is not None or self._waiting_since is None:
            return
        waited = time.time() - self._waiting_since
        if waited - self._waiting_notice >= 15.0:
            self._waiting_notice = waited
            print(f"[ipad] waiting for pair ({waited:.0f}s)")

    def _apply_capture_profile_boundary(self, header: dict) -> None:
        """Reset consumers once, before decoding the first changed-profile frame."""
        generation = header.get("capture_profile_generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            return
        if self._last_capture_profile_generation is None:
            # Whatever generation the reader sees first is its initial profile;
            # no accumulated samples exist yet, so there is nothing to reset.
            self._last_capture_profile_generation = generation
            return
        if generation == self._last_capture_profile_generation:
            return
        # Commit the boundary before invoking user hooks so even a faulty hook
        # cannot make this same generation reset repeatedly on later frames.
        self._last_capture_profile_generation = generation
        for hook in tuple(self._capture_reset_hooks):
            try:
                hook()
            except Exception:  # noqa: BLE001 - capture must continue after reset failure
                pass

    def _reader_loop(self) -> None:
        """Decode pushed frames off the link, publish the newest, fire hooks.

        Decoding happens here rather than on the link's event loop so JPEG work
        never serializes against packet receive — that would reintroduce exactly
        the arrival jitter the capture-clock mapping exists to remove.
        """
        idx = 0
        while not self._reader_stop.is_set():
            item = self._link.take_frame() if self._link is not None else None
            if item is None:
                self._note_waiting()
                time.sleep(0.001)
                continue
            header, payload = item
            self._apply_capture_profile_boundary(header)
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self._decode_errors += 1
                continue
            if self._subsampling is None:
                self._subsampling = jpeg_subsampling(payload)
            frame = self._prep_frame(frame)
            recv_wall = header.get("recv_wall") or time.time()
            ts = self._clock.map(float(header.get("media_time", recv_wall)), recv_wall)

            with self._latest_lock:
                self._latest_frame = frame
                self._latest_ts = ts
                self._latest_index = idx
            for hook in self._fast_hooks:
                try:
                    hook(frame, ts)
                except Exception:  # noqa: BLE001
                    pass
            self._note_fps(ts)
            self._frames_decoded += 1
            self._last_frame_at = recv_wall
            idx += 1
