"""WebRTC transport between the laptop and an iPad browser.

Splits cleanly from `core.ipad_camera` so that backend stays importable without
aiortc installed — `core/realsense_camera.py` keeps `pyrealsense2` optional the
same way. Everything here is lazy-imported inside the loop thread.

Shape of the thing:

* A private asyncio loop runs on its own daemon thread. Nothing in the pipeline
  is async, so the loop never escapes this module.
* The relay (a tiny HTTPS service) carries only SDP and ICE. Media never touches
  it — once the peer connection is up, frames go straight iPad -> laptop.
* **The iPad is the offerer.** It creates the data channels, so it must offer;
  we passively answer. That keeps Safari's SDP authoritative and means aiortc
  never has to construct an offer.
* Frames land in a depth-2 drop-oldest slot. `IPadCamera`'s reader thread pops
  from it and does the JPEG decode, so decoding never serializes against packet
  receive on the event loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import threading
import time
from collections import deque
from typing import Callable, Optional

from .ipad_camera import FRAME_HEADER_SIZE, unpack_frame_header

# Free STUN, used only as a safety net: on a laptop hotspot both peers are on
# the same subnet, so ICE settles on host candidates and never needs it.
DEFAULT_STUN = ("stun:stun.l.google.com:19302",)

# The laptop and the iPad must arrive at the same `exp` without talking to each
# other first (the relay authenticates a room by checking both members present
# the identical signature for the identical expiry). Both sides therefore
# quantize to the next boundary of a fixed wall-clock window instead of using
# "now + TTL". Must stay in lockstep with PAIR_WINDOW_SECONDS in
# relay/static/ipad.html.
PAIR_WINDOW_SECONDS = 600


def pairing_expiry(now: Optional[float] = None) -> int:
    """Next pairing-window boundary, matching the iPad page's computation."""
    now = time.time() if now is None else now
    window = PAIR_WINDOW_SECONDS
    return int(math.ceil((now + 1) / window) * window)


def pairing_signature(secret: str, room: str, code: str, exp: int) -> str:
    """HMAC proving both peers know the shared secret and the pairing code.

    The relay never learns the code — it only checks that both members of a room
    presented the same signature for the same expiry, which is enough to keep
    strangers out without handing the code to the relay.
    """
    msg = f"{room}|{code}|{exp}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]


class IPadLink:
    """Owns the relay WebSocket, the peer connection, and the inbound frame slot."""

    def __init__(self, relay_url: Optional[str] = None, room: Optional[str] = None,
                 secret: Optional[str] = None, code: Optional[str] = None,
                 stun: tuple = (), transport: str = "datachannel",
                 pair_ttl: float = 600.0,
                 on_control: Optional[Callable[[dict, Callable], None]] = None):
        if not relay_url or not room or not secret or not code:
            raise RuntimeError(
                "iPad link needs relay url, room, secret and pairing code "
                "(set IPAD_RELAY_URL / IPAD_ROOM / RELAY_SECRET in .env)")
        self.relay_url = relay_url.rstrip("/")
        self.room = room
        self.transport = transport
        self.pair_ttl = pair_ttl
        self._secret = secret
        self._code = code
        self._stun = tuple(stun) or DEFAULT_STUN
        self._on_control = on_control

        self._frames: deque[tuple[dict, bytes]] = deque(maxlen=2)
        self._partial: dict[int, dict[int, bytes]] = {}   # seq -> {chunk_index: bytes}
        self._frames_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = {"relay": "idle", "ice": "new", "dc": "closed",
                       "frames_rx": 0, "seq_gaps": 0, "last_rx_at": None,
                       "clock_source": None, "error": None, "dropped": 0}

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._pc = None
        self._control_channel = None
        self._last_seq: Optional[int] = None

    # ---- thread-facing API (called from the pipeline's threads) ----------

    def start(self) -> None:
        """Spin up the asyncio loop thread. Returns immediately."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ipad-link")
        self._thread.start()

    def stop(self) -> None:
        """Ask the loop to shut down and wait briefly for the thread to exit."""
        loop, stop_event = self._loop, self._stop_event
        if loop is not None and stop_event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def take_frame(self) -> Optional[tuple[dict, bytes]]:
        """Pop the newest pushed frame, or None. Never blocks."""
        with self._frames_lock:
            if not self._frames:
                return None
            return self._frames.pop()

    def send_control(self, payload: dict) -> None:
        """Send a JSON control message to the iPad. Safe from any thread."""
        loop = self._loop
        if loop is None or self._control_channel is None or loop.is_closed():
            return
        try:
            data = json.dumps(payload)
        except (TypeError, ValueError):
            return
        try:
            loop.call_soon_threadsafe(self._send_control_now, data)
        except RuntimeError:            # loop shut down between the checks
            pass

    def set_control_handler(self, handler) -> None:
        """Swap the control callback on a live link (see IPadCamera's docstring)."""
        self._on_control = handler

    def status(self) -> dict:
        with self._state_lock:
            return dict(self._state)

    # ---- internals ------------------------------------------------------

    def _set(self, **kw) -> None:
        with self._state_lock:
            self._state.update(kw)

    def _bump(self, key: str, amount: int = 1) -> None:
        with self._state_lock:
            self._state[key] = self._state.get(key, 0) + amount

    def _send_control_now(self, data: str) -> None:
        channel = self._control_channel
        if channel is not None and getattr(channel, "readyState", "") == "open":
            try:
                channel.send(data)
            except Exception:  # noqa: BLE001 - a dead channel must not kill the loop
                pass

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception as exc:  # noqa: BLE001 - surfaced through status()
            self._set(error=f"{type(exc).__name__}: {exc}", relay="failed")
        finally:
            try:
                loop.run_until_complete(self._teardown())
            except Exception:  # noqa: BLE001
                pass
            loop.close()
            self._loop = None

    async def _main(self) -> None:
        try:
            import aiohttp        # noqa: PLC0415 - lazy by design
        except ImportError as exc:
            raise RuntimeError("iPad source needs: pip install -r "
                               "requirements-ipad.txt") from exc

        self._stop_event = asyncio.Event()
        ws_url = self.relay_url.replace("https://", "wss://", 1) \
                               .replace("http://", "ws://", 1) + "/ws"

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self._wake_relay(session)
            backoff = 1.0
            while not self._stop_event.is_set():
                try:
                    await self._session_once(aiohttp, session, ws_url)
                    backoff = 1.0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reconnect, don't die
                    self._set(relay="reconnecting",
                              error=f"{type(exc).__name__}: {exc}")
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _wake_relay(self, session) -> None:
        """Poke /healthz first: a free-tier service can take ~50s to spin up,
        and eating that here means the iPad never sees a hung tab."""
        self._set(relay="waking")
        deadline = time.time() + 90.0
        delay = 1.0
        while time.time() < deadline and not self._stop_event.is_set():
            try:
                async with session.get(f"{self.relay_url}/healthz") as resp:
                    if resp.status == 200:
                        return
            except Exception:  # noqa: BLE001 - still spinning up
                pass
            print("[ipad] waking relay (free tier, up to ~60s)…")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)

    async def _session_once(self, aiohttp, session, ws_url: str) -> None:
        """One relay connection: authenticate, then answer whatever the iPad offers."""
        async with session.ws_connect(ws_url, heartbeat=30.0) as ws:
            exp = pairing_expiry()
            await ws.send_json({
                "type": "hello", "room": self.room, "role": "host", "exp": exp,
                "sig": pairing_signature(self._secret, self.room, self._code, exp)})
            self._set(relay="connected", error=None)

            stop_task = asyncio.ensure_future(self._stop_event.wait())
            # Re-register just after the pairing window rolls over. Otherwise an
            # iPad that pairs in the *next* window computes a different exp than
            # the one we registered, and the relay rejects it with 4401 even
            # though both sides know the secret and the code.
            expiry_task = asyncio.ensure_future(
                asyncio.sleep(max(1.0, exp - time.time() + 1.0)))
            try:
                while not self._stop_event.is_set():
                    recv_task = asyncio.ensure_future(ws.receive())
                    done, _ = await asyncio.wait(
                        {recv_task, stop_task, expiry_task},
                        return_when=asyncio.FIRST_COMPLETED)
                    if stop_task in done or expiry_task in done:
                        recv_task.cancel()
                        await ws.close()
                        return
                    msg = recv_task.result()
                    if msg.type is not aiohttp.WSMsgType.TEXT:
                        return          # closed, errored, or binary (relay rejects those)
                    try:
                        payload = json.loads(msg.data)
                    except (ValueError, TypeError):
                        continue
                    try:
                        await self._handle_signal(ws, payload)
                    except Exception as exc:  # noqa: BLE001
                        # Keep the signalling socket alive: a failed offer
                        # should let the user retry from the page rather than
                        # drop the whole session, and the reason must be
                        # visible instead of surfacing as a silent stall.
                        detail = f"{type(exc).__name__}: {exc}"
                        self._set(error=f"signal {payload.get('type')}: {detail}")
                        print(f"[ipad] signalling failed on "
                              f"{payload.get('type')!r}: {detail}")
            finally:
                stop_task.cancel()
                expiry_task.cancel()
                self._set(relay="disconnected")

    async def _handle_signal(self, ws, payload: dict) -> None:
        kind = str(payload.get("type", ""))
        if kind == "offer":
            await self._answer_offer(ws, payload)
        elif kind == "candidate":
            await self._add_candidate(payload)
        elif kind == "bye":
            await self._teardown()

    async def _answer_offer(self, ws, payload: dict) -> None:
        from aiortc import (RTCConfiguration, RTCIceServer,  # noqa: PLC0415
                            RTCPeerConnection, RTCSessionDescription)

        await self._teardown()          # a new offer means the iPad reloaded
        config = RTCConfiguration(iceServers=[RTCIceServer(urls=list(self._stun))])
        pc = RTCPeerConnection(configuration=config)
        self._pc = pc
        self._last_seq = None

        @pc.on("connectionstatechange")
        async def _on_connection_state():
            self._set(ice=pc.connectionState)
            print(f"[ipad] peer connection: {pc.connectionState}")

        @pc.on("iceconnectionstatechange")
        async def _on_ice_state():
            self._set(ice=pc.iceConnectionState)
            print(f"[ipad] ice: {pc.iceConnectionState}")

        @pc.on("datachannel")
        def _on_datachannel(channel):
            print(f"[ipad] data channel opened: {channel.label!r}")
            if channel.label == "frames":
                self._set(dc="open")

                @channel.on("message")
                def _on_frame(message):
                    self._ingest_frame(message)

                @channel.on("close")
                def _on_frames_closed():
                    self._set(dc="closed")
            elif channel.label == "control":
                self._control_channel = channel

                @channel.on("message")
                def _on_control(message):
                    self._ingest_control(message)

        print("[ipad] offer received; answering")
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=payload.get("sdp", ""), type="offer"))
        answer = await pc.createAnswer()
        # setLocalDescription blocks here until aiortc finishes gathering ICE,
        # which is where a blocked STUN/UDP path shows up as a long stall.
        await pc.setLocalDescription(answer)
        print("[ipad] answer sent; waiting for ICE")
        # aiortc gathers ICE fully before setLocalDescription resolves, so the
        # answer already carries every candidate; we still accept the iPad's
        # trickled ones below.
        await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})

    async def _add_candidate(self, payload: dict) -> None:
        if self._pc is None:
            return
        candidate = payload.get("candidate")
        if not candidate:
            return
        try:
            from aiortc.sdp import candidate_from_sdp   # noqa: PLC0415
            text = (candidate.get("candidate", "") if isinstance(candidate, dict)
                    else str(candidate))
            if not text:
                return
            if text.startswith("candidate:"):
                text = text.split(":", 1)[1]
            ice = candidate_from_sdp(text)
            if isinstance(candidate, dict):
                ice.sdpMid = candidate.get("sdpMid")
                ice.sdpMLineIndex = candidate.get("sdpMLineIndex")
            await self._pc.addIceCandidate(ice)
        except Exception:  # noqa: BLE001 - a bad candidate must not kill the loop
            pass

    def _ingest_frame(self, message) -> None:
        """Header-parse a pushed frame and hand it to the reader thread.

        Runs on the event loop, so it must stay cheap: no JPEG decode here.
        """
        if not isinstance(message, (bytes, bytearray, memoryview)):
            return
        buf = bytes(message)
        header = unpack_frame_header(buf)
        if header is None:
            return
        seq = header["seq"]
        payload = buf[FRAME_HEADER_SIZE:]
        count = header["chunk_count"]
        if count > 1:
            # Reassemble: one JPEG spans several SCTP messages because aiortc
            # caps a message at 65536 bytes. The channel is unreliable, so a
            # frame missing any chunk is discarded rather than half-decoded.
            slot = self._partial.setdefault(seq, {})
            slot[header["chunk_index"]] = payload
            if len(slot) < count:
                if len(self._partial) > 8:      # bound memory on a lossy link
                    oldest = min(self._partial)
                    if oldest != seq:
                        self._partial.pop(oldest, None)
                        self._bump("incomplete")
                return
            payload = b"".join(slot[i] for i in sorted(slot))
            self._partial.pop(seq, None)

        if self._last_seq is not None:
            gap = (seq - self._last_seq - 1) & 0xFFFFFFFF
            if gap > 0xF0000000:
                # Arrived after a newer frame. Feeding it downstream would push
                # a backwards capture timestamp into CaptureClock and force a
                # resync, so drop it — newest-wins is the contract anyway.
                self._bump("out_of_order")
                return
            if 0 < gap < 1000:          # ignore wrap/reconnect discontinuities
                self._bump("seq_gaps", gap)
        self._last_seq = seq
        header["recv_wall"] = time.time()
        with self._frames_lock:
            if len(self._frames) == self._frames.maxlen:
                self._bump("dropped")
            self._frames.append((header, payload))
        self._bump("frames_rx")
        self._set(last_rx_at=header["recv_wall"])

    def _ingest_control(self, message) -> None:
        try:
            payload = json.loads(message)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        if payload.get("type") in ("clock_source", "hello"):
            # The page reports which timebase it is stamping frames with —
            # rvfc-mediaTime (capture clock, low jitter) or performance-now
            # (fallback). Surfaced in diagnostics so an rPPG quality regression
            # is attributable rather than mysterious.
            self._set(clock_source=payload.get("clock_source"))
            return
        if self._on_control is not None:
            # The callback is expected to hand work to an executor and return
            # immediately — blocking here would stall the whole event loop.
            try:
                self._on_control(payload, self.send_control)
            except Exception as exc:  # noqa: BLE001
                self.send_control({"type": "module_result", "ok": False,
                                   "error": str(exc)})

    async def _teardown(self) -> None:
        pc, self._pc = self._pc, None
        self._control_channel = None
        self._set(dc="closed")
        if pc is not None:
            try:
                await pc.close()
            except Exception:  # noqa: BLE001
                pass
