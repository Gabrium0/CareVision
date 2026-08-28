"""Direct-LAN WebSocket transport between the laptop and a NATIVE iPad app.

Why this exists alongside `core/ipad_link.py` (WebRTC): a native iPadOS app on the
Windows Mobile Hotspot can reach the laptop directly at ``192.168.137.x``, so the
two reasons the WebRTC relay exists — Safari's HTTPS-only ``getUserMedia`` and
Wi-Fi AP client isolation — both vanish. This link therefore drops WebRTC,
STUN/ICE, and the cloud relay entirely: the laptop binds a small aiohttp
WebSocket server on the hotspot interface, the app dials it, and the **same wire
contract** flows over one socket — IPF1 binary frame chunks (see
`core.ipad_camera`) plus control JSON, exactly as analysed for the browser page.

It presents the exact duck-typed interface `IPadCamera` drives
(``start``/``stop``/``take_frame``/``send_control``/``set_control_handler``/
``send_audio``/``status`` plus ``audio_bus``), so `IPadCamera._build_link` can
swap it in for ``--ipad-transport lan`` with nothing else downstream changing:
the reader thread, extractors, detectors, and dashboard push are unaffected.

Frame reassembly, capture-profile ordering, and the bounded telemetry validators
are the **same logic** as `IPadLink`; the security-critical validators
(`_validated_sender_stats`, `_validated_capture_profile`) and the pairing HMAC
(`pairing_signature`) are imported from there rather than reimplemented, so their
bounds never drift. The transport-coupled glue (WebSocket receive, single-flight
peer, text-vs-binary demux, frame ACK) is the only genuinely new code.

Auth: no relay mediates, so the laptop verifies the pairing HMAC itself. The app
sends the same ``hello`` the browser computes —
``sig = HMAC-SHA256(secret, "room|code|exp")[:32]`` with ``exp`` quantised to the
next `PAIR_WINDOW_SECONDS` boundary — and the laptop recomputes it against the
printed 6-digit code. Knowing this is strictly stronger than the relay design,
which never learned the code.

Audio: v1 pairs with native on-device TTS/ASR, so **no PCM media crosses this
socket** — agent speech is sent as guarded text (a ``speak`` control message the
backend emits via ``send_control``) and the person's words arrive as ``asr_text``
control messages routed to ``on_control``. ``send_audio`` is therefore a no-op
here; raw-PCM streaming for YAMNet is a later phase.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import threading
import time
from collections import deque
from typing import Callable, Optional

from .ipad_camera import FRAME_HEADER_SIZE, unpack_frame_header
from .ipad_link import (
    PAIR_WINDOW_SECONDS,
    _validated_capture_profile,
    _validated_sender_stats,
    pairing_signature,
)

# Default listen endpoint. The Windows Mobile Hotspot gateway is 192.168.137.1
# (see README "Why a Windows Mobile Hotspot is required"); binding there exposes
# the server on the hotspot interface only, not on every laptop interface. Pass
# --ipad-listen-host 0.0.0.0 to bind all interfaces when the gateway IP differs.
DEFAULT_LISTEN_HOST = "192.168.137.1"
DEFAULT_LISTEN_PORT = 8788

# A `hello` proves knowledge of the secret + code, but its `exp` is client-chosen.
# Accept only an `exp` that is still in the future and no further ahead than two
# pairing windows, so a captured hello cannot be replayed indefinitely.
_MAX_EXP_AHEAD = 2 * PAIR_WINDOW_SECONDS

# One reassembled JPEG can span several messages; bound pending sequences exactly
# as IPadLink does so a stale/corrupt peer cannot grow memory without limit.
_MAX_PENDING_SEQS = 8

# Matches the relay's 64 KiB ceiling in spirit, but frames flow here so allow a
# whole high-quality JPEG in one binary message (the app need not chunk on a LAN,
# though it may). Keep it bounded against a hostile peer.
_MAX_MSG_SIZE = 8 * 1024 * 1024
_CAPTURE_PROFILE_MAX_JSON = 512


class IPadLanLink:
    """Owns the LAN WebSocket server, one paired app socket, and the frame slot."""

    def __init__(self, room: Optional[str] = None, secret: Optional[str] = None,
                 code: Optional[str] = None,
                 listen_host: Optional[str] = None,
                 listen_port: Optional[int] = None,
                 transport: str = "lan", pair_ttl: float = 600.0,
                 audio_bus=None,
                 on_control: Optional[Callable[[dict, Callable], None]] = None,
                 **_unused):
        # **_unused swallows the WebRTC-only opts (relay_url, stun) so main.py can
        # hand one _link_opts dict to whichever transport gets built.
        if not room or not secret or not code:
            raise RuntimeError(
                "iPad LAN link needs room, secret and pairing code "
                "(set IPAD_ROOM / RELAY_SECRET in .env)")
        self.room = room
        self.transport = transport
        self.pair_ttl = pair_ttl
        self._secret = secret
        self._code = code
        self.listen_host = listen_host or DEFAULT_LISTEN_HOST
        self.listen_port = int(listen_port or DEFAULT_LISTEN_PORT)
        self._on_control = on_control
        # Late-bindable like IPadLink.audio_bus; unused in the native-TTS/ASR v1.
        self.audio_bus = audio_bus

        self._frames: deque[tuple[dict, bytes]] = deque(maxlen=2)
        self._partial: dict[int, dict[int, bytes]] = {}
        self._frames_lock = threading.Lock()
        self._state_lock = threading.Lock()
        # Same key shape as IPadLink.status() so the dashboard / diagnostics read
        # both transports identically. WebRTC-only fields keep sane LAN values.
        self._state = {"relay": "idle", "ice": "new", "dc": "closed",
                       "frames_rx": 0, "seq_gaps": 0, "last_rx_at": None,
                       "clock_source": None, "error": None, "dropped": 0,
                       "coalesced": 0, "sender_stats": None,
                       "unexpected_media_tracks": 0, "frame_acks_tx": 0,
                       "frame_ack_errors": 0, "capture_profile_generation": 0,
                       "capture_profile": None, "client_build": None,
                       "client_ua": None, "client_caps": None,
                       "audio_mic": False, "audio_out": False,
                       "audio_rx_blocks": 0, "audio_tx_chunks": 0,
                       "transport": "lan", "incomplete": 0, "out_of_order": 0}
        self._last_seq: Optional[int] = None

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._runner = None
        self._ws = None                 # the single active app socket, if any

    # ---- thread-facing API (identical contract to IPadLink) --------------

    def start(self) -> None:
        """Spin up the asyncio loop thread. Returns immediately."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ipad-lan-link")
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
        """Return only the newest pushed frame, or None. Never blocks.

        Newest-wins with the same coalescing accounting as IPadLink.take_frame:
        entries discarded here were already admitted, so count them as
        intentional ``coalesced`` frames rather than double-counting ``dropped``.
        """
        with self._frames_lock:
            if not self._frames:
                return None
            newest = self._frames.pop()
            coalesced = len(self._frames)
            self._frames.clear()
        if coalesced:
            self._bump("coalesced", coalesced)
        return newest

    def send_control(self, payload: dict) -> None:
        """Send a JSON control message to the app. Safe from any thread."""
        loop = self._loop
        if loop is None or self._ws is None or loop.is_closed():
            return
        try:
            data = json.dumps(payload)
        except (TypeError, ValueError):
            return
        try:
            loop.call_soon_threadsafe(self._send_text_now, data)
        except RuntimeError:            # loop shut down between the checks
            pass

    def set_control_handler(self, handler) -> None:
        """Swap the control callback on a live link (see IPadCamera's docstring)."""
        self._on_control = handler

    def send_audio(self, samples, rate: int = 16000, channels: int = 1) -> None:
        """No-op: v1 speaks on-device (native TTS), so no PCM crosses the LAN.

        Kept to satisfy the link interface; a later phase may stream raw PCM here
        to feed the laptop audio bus (YAMNet cough/smoke) alongside native voice.
        """
        return

    def status(self) -> dict:
        """Return a thread-safe diagnostics snapshot with nested telemetry copied."""
        with self._state_lock:
            state = dict(self._state)
            if isinstance(state.get("sender_stats"), dict):
                state["sender_stats"] = dict(state["sender_stats"])
            if isinstance(state.get("capture_profile"), dict):
                state["capture_profile"] = dict(state["capture_profile"])
            return state

    # ---- internals (transport-agnostic; mirrors IPadLink) ----------------

    def _set(self, **kw) -> None:
        with self._state_lock:
            self._state.update(kw)

    def _bump(self, key: str, amount: int = 1) -> None:
        with self._state_lock:
            self._state[key] = self._state.get(key, 0) + amount

    def _reset_peer_state(self) -> None:
        """Forget the previous app's partial frames and sequence counter.

        A fresh app connection restarts its sequence at zero; keeping stale
        partials could combine old and new chunks under one seq, and a queued
        complete frame would keep a stale preview alive across the reconnect.
        """
        self._partial.clear()
        self._last_seq = None
        with self._frames_lock:
            self._frames.clear()
        self._set(sender_stats=None, capture_profile=None)

    def _ingest_capture_profile(self, message) -> bool:
        """Apply one ordered profile, advancing the generation only on changes."""
        if not isinstance(message, str) or len(message) > _CAPTURE_PROFILE_MAX_JSON:
            return False
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return False
        profile = _validated_capture_profile(payload)
        if profile is None:
            return False
        with self._state_lock:
            if self._state.get("capture_profile") == profile:
                return False
            generation = int(self._state.get("capture_profile_generation", 0)) + 1
            self._state["capture_profile_generation"] = generation
            self._state["capture_profile"] = dict(profile)
        return True

    def _capture_profile_snapshot(self) -> tuple[int, Optional[dict]]:
        with self._state_lock:
            generation = int(self._state.get("capture_profile_generation", 0))
            profile = self._state.get("capture_profile")
            return generation, (dict(profile) if isinstance(profile, dict) else None)

    def _ingest_frame(self, message) -> Optional[int]:
        """Header-parse a pushed frame and hand it to the reader thread.

        Same reassembly + newest-wins contract as IPadLink._ingest_frame: partial,
        invalid, and out-of-order data return None (and so receive no ACK); the
        completed sequence is returned only after the full JPEG is admitted.
        """
        if not isinstance(message, (bytes, bytearray, memoryview)):
            return None
        buf = bytes(message)
        header = unpack_frame_header(buf)
        if header is None:
            return None
        seq = header["seq"]
        payload = buf[FRAME_HEADER_SIZE:]
        count = header["chunk_count"]
        if count > 1:
            slot = self._partial.setdefault(seq, {})
            slot[header["chunk_index"]] = payload
            if len(slot) < count:
                if len(self._partial) > _MAX_PENDING_SEQS:
                    oldest = min(self._partial)
                    if oldest != seq:
                        self._partial.pop(oldest, None)
                        self._bump("incomplete")
                return None
            payload = b"".join(slot[i] for i in sorted(slot))
            self._partial.pop(seq, None)

        if self._last_seq is not None:
            gap = (seq - self._last_seq - 1) & 0xFFFFFFFF
            if gap > 0xF0000000:
                # Arrived after a newer frame; feeding it downstream would push a
                # backwards capture timestamp into CaptureClock. Newest-wins.
                self._bump("out_of_order")
                return None
            if 0 < gap < 1000:          # ignore wrap/reconnect discontinuities
                self._bump("seq_gaps", gap)
        self._last_seq = seq
        profile_generation, profile = self._capture_profile_snapshot()
        header["capture_profile_generation"] = profile_generation
        header["capture_profile"] = profile
        header["recv_wall"] = time.time()
        with self._frames_lock:
            if len(self._frames) == self._frames.maxlen:
                self._bump("dropped")
            self._frames.append((header, payload))
        self._bump("frames_rx")
        self._set(last_rx_at=header["recv_wall"])
        return seq

    def _handle_control(self, payload: dict) -> None:
        """Dispatch a parsed control message, mirroring IPadLink._ingest_control."""
        if not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if kind in ("clock_source", "hello"):
            self._set(clock_source=payload.get("clock_source"))
            return
        if kind == "client_version":
            build = payload.get("build")
            ua = payload.get("ua")
            caps = payload.get("caps")
            safe_caps = None
            if isinstance(caps, dict):
                safe_caps = {}
                for key, value in list(caps.items())[:16]:
                    if not isinstance(key, str) or len(key) > 32:
                        continue
                    if isinstance(value, bool) or value is None:
                        safe_caps[key] = value
                    elif isinstance(value, int) and -1 <= value <= 1000:
                        safe_caps[key] = value
            self._set(
                client_build=str(build)[:64] if isinstance(build, str) else None,
                client_ua=str(ua)[:256] if isinstance(ua, str) else None,
                client_caps=safe_caps)
            return
        if kind == "camera_tuning":
            self._set(camera_tuning=payload.get("locked"),
                      camera_caps=payload.get("capabilities"),
                      camera_settings=payload.get("settings"))
            return
        if kind == "sender_stats":
            stats = _validated_sender_stats(payload)
            if stats:
                self._set(sender_stats=dict(stats))
            return
        # Everything else (module toggles, asr_text, demo injection) goes to the
        # pipeline callback, which must hand work off and return immediately.
        if self._on_control is not None:
            try:
                self._on_control(payload, self.send_control)
            except Exception as exc:  # noqa: BLE001
                self.send_control({"type": "module_result", "ok": False,
                                   "error": str(exc)})

    def _ingest_text(self, raw) -> None:
        """Route one inbound text frame: capture_profile ordering, else control."""
        if not isinstance(raw, str):
            return
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        if payload.get("type") == "capture_profile":
            # On the browser this rides the frames channel; over one LAN socket it
            # is just a text control message. Route it through the profile barrier
            # so rPPG buffers reset before the first frame of the new profile.
            self._ingest_capture_profile(raw)
            return
        self._handle_control(payload)

    # ---- pairing ---------------------------------------------------------

    def _verify_hello(self, raw) -> bool:
        """Authenticate the app's opening `hello` against the shared secret+code."""
        if not isinstance(raw, str):
            return False
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return False
        if not isinstance(msg, dict) or msg.get("type") != "hello":
            return False
        if msg.get("role") != "ipad" or msg.get("room") != self.room:
            return False
        exp = msg.get("exp")
        sig = msg.get("sig")
        if not isinstance(exp, int) or isinstance(exp, bool):
            return False
        if not isinstance(sig, str) or not sig:
            return False
        now = int(time.time())
        if exp <= now or exp > now + _MAX_EXP_AHEAD:
            return False
        expected = pairing_signature(self._secret, self.room, self._code, exp)
        return hmac.compare_digest(sig, expected)

    # ---- event loop / server --------------------------------------------

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
            from aiohttp import web       # noqa: PLC0415 - lazy by design
        except ImportError as exc:
            raise RuntimeError("iPad LAN source needs: pip install -r "
                               "requirements-ipad.txt") from exc

        self._stop_event = asyncio.Event()
        app = web.Application()
        app.router.add_get("/ws", self._ws_handler)
        app.router.add_get("/healthz", self._healthz)
        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner
        site = web.TCPSite(runner, self.listen_host, self.listen_port)
        try:
            await site.start()
        except OSError as exc:
            # A missing hotspot interface is the common first-run failure; make it
            # legible rather than a bare traceback. IPadCamera.open() surfaces it.
            self._set(relay="failed",
                      error=f"cannot bind {self.listen_host}:{self.listen_port}: "
                            f"{exc} (is the Windows hotspot up? try "
                            f"--ipad-listen-host 0.0.0.0)")
            raise
        # "connected" here means the server is listening and awaiting the app —
        # the same semantics IPadCamera.open() waits on for the WebRTC relay.
        self._set(relay="connected", error=None)
        print(f"[ipad-lan] listening on ws://{self.listen_host}:{self.listen_port}/ws "
              f"(room {self.room!r}); pair the app with the printed code")
        await self._stop_event.wait()

    async def _healthz(self, _request):
        from aiohttp import web           # noqa: PLC0415
        return web.Response(text="ok")

    async def _ws_handler(self, request):
        from aiohttp import web           # noqa: PLC0415
        ws = web.WebSocketResponse(max_msg_size=_MAX_MSG_SIZE, heartbeat=30.0)
        await ws.prepare(request)

        first = await ws.receive()
        if first.type is not web.WSMsgType.TEXT or not self._verify_hello(first.data):
            await ws.close(code=4401)
            return ws

        # Single-flight, last-connection-wins: a reconnecting app must not be
        # rejected by its own not-yet-cleaned-up socket. Evict any prior peer.
        prior, self._ws = self._ws, ws
        if prior is not None and not prior.closed:
            try:
                await prior.close(code=4409)
            except Exception:  # noqa: BLE001 - a dead socket must not block the new one
                pass
        self._reset_peer_state()
        self._set(dc="open", ice="connected", error=None)
        print("[ipad-lan] app paired")

        try:
            async for msg in ws:
                if self._ws is not ws:
                    break               # superseded by a newer connection
                if msg.type is web.WSMsgType.BINARY:
                    completed = self._ingest_frame(msg.data)
                    if completed is not None:
                        await self._send_frame_ack(ws, completed)
                elif msg.type is web.WSMsgType.TEXT:
                    self._ingest_text(msg.data)
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING,
                                  web.WSMsgType.ERROR):
                    break
        finally:
            if self._ws is ws:
                self._ws = None
                self._reset_peer_state()
                self._set(dc="closed", ice="new", audio_mic=False, audio_out=False)
                print("[ipad-lan] app disconnected")
        return ws

    def _send_text_now(self, data: str) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return
        # Fire-and-forget on the loop; a dead socket must not kill the loop.
        asyncio.ensure_future(self._safe_send(ws, data))

    async def _safe_send(self, ws, data: str) -> None:
        try:
            await ws.send_str(data)
        except Exception:  # noqa: BLE001 - a dropped control message is recoverable
            pass

    async def _send_frame_ack(self, ws, seq: int) -> None:
        """Acknowledge one admitted JPEG so the app's windowed sender advances."""
        if ws.closed:
            return
        data = json.dumps({"type": "frame_ack", "seq": int(seq) & 0xFFFFFFFF},
                          separators=(",", ":"))
        try:
            await ws.send_str(data)
        except Exception:  # noqa: BLE001 - ACK loss must not stop frame receive
            self._bump("frame_ack_errors")
            return
        self._bump("frame_acks_tx")

    async def _teardown(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None and not ws.closed:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
        runner, self._runner = self._runner, None
        if runner is not None:
            try:
                await runner.cleanup()
            except Exception:  # noqa: BLE001
                pass
        self._reset_peer_state()
        self._set(dc="closed", relay="disconnected",
                  audio_mic=False, audio_out=False)
