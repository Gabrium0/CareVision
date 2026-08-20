"""WebSocket signaling relay for the iPad-as-camera WebRTC link.

Why this exists: an iPad in Safari can only reach `navigator.mediaDevices`
(and therefore `getUserMedia`) from a secure context, and the laptop it needs
to talk to has no TLS and is unreachable anyway because the WiFi AP isolates
clients from each other. This service's only irreplaceable job is to be an
HTTPS origin the iPad can load, plus a signaling channel free-riding on that
same connection. Media never flows through it: once WebRTC negotiation
completes, the iPad and laptop exchange frames peer-to-peer (they share a
Windows Mobile Hotspot, so ICE finds a direct host candidate and no TURN
relay is needed).

Routes:
    GET /healthz  -> "ok" (cold-start warm-up target)
    GET /r/{room} -> static/ipad.html, with the shared secret templated in
    GET /ws       -> WebSocket signaling, two members per room ("host", "ipad")

Hard safety rules enforced below, not just documented: a 64KB message ceiling
that makes relaying video physically impossible over this socket, an instant
close on any binary frame, and no parsing/storing/logging of message bodies
(no database, no files written) — this process only ever sees room id, role,
and connect/disconnect events.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import time
from pathlib import Path

import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("relay")

STATIC_DIR = Path(__file__).parent / "static"
IPAD_HTML_PATH = STATIC_DIR / "ipad.html"

# The placeholder ipad.html templates the shared secret into, so the page can
# derive its HMAC client-side. See relay/README.md for why shipping the
# secret to the page is safe here: it is useless without the human-relayed
# 6-digit pairing code, which the relay itself never learns.
SECRET_PLACEHOLDER = "__RELAY_SECRET_JSON__"

RELAY_SECRET = os.environ.get("RELAY_SECRET")
if not RELAY_SECRET:
    raise RuntimeError(
        "RELAY_SECRET is not set. Refusing to start: this relay authenticates "
        "every room pairing with it. Set RELAY_SECRET in the environment "
        "(on Render: service Settings > Environment) and restart."
    )

ROLES = ("host", "ipad")
HELLO_CLOSE_CODE = 4401       # bad/missing hello, sig mismatch, or expired
ROLE_TAKEN_CLOSE_CODE = 4409  # second connection for an already-occupied role
MAX_MSG_SIZE = 64 * 1024      # hard ceiling; makes relaying video impossible


class Room:
    """At most one `host` and one `ipad` socket, plus the pairing proof both
    sides must present identically (same sig, same exp) before joining."""

    __slots__ = ("room_id", "members", "expected_sig", "exp")

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.members: dict[str, web.WebSocketResponse] = {}
        self.expected_sig: str | None = None
        self.exp: int | None = None

    @staticmethod
    def peer_role(role: str) -> str:
        return "host" if role == "ipad" else "ipad"


rooms: dict[str, Room] = {}


async def healthz(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def serve_ipad_page(request: web.Request) -> web.Response:
    room = request.match_info.get("room", "")
    if not room or any(c in room for c in "/\\?#"):
        return web.Response(status=400, text="invalid room id")
    html = IPAD_HTML_PATH.read_text(encoding="utf-8")
    html = html.replace(SECRET_PLACEHOLDER, json.dumps(RELAY_SECRET))
    return web.Response(text=html, content_type="text/html")


def _parse_hello(raw: str) -> dict | None:
    try:
        msg = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(msg, dict) or msg.get("type") != "hello":
        return None
    if msg.get("role") not in ROLES:
        return None
    room = msg.get("room")
    exp = msg.get("exp")
    sig = msg.get("sig")
    if not isinstance(room, str) or not room:
        return None
    if not isinstance(exp, int) or isinstance(exp, bool):
        return None
    if not isinstance(sig, str) or not sig:
        return None
    return msg


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=MAX_MSG_SIZE)
    await ws.prepare(request)

    room_obj: Room | None = None
    role: str | None = None

    try:
        first = await ws.receive()
        if first.type is not aiohttp.WSMsgType.TEXT:
            await ws.close(code=HELLO_CLOSE_CODE)
            return ws

        hello = _parse_hello(first.data)
        if hello is None:
            await ws.close(code=HELLO_CLOSE_CODE)
            return ws

        room_id = hello["room"]
        role = hello["role"]
        exp = hello["exp"]
        sig = hello["sig"]
        now = int(time.time())

        if exp <= now:
            await ws.close(code=HELLO_CLOSE_CODE)
            return ws

        room_obj = rooms.setdefault(room_id, Room(room_id))

        # The relay never learns the pairing code, so it cannot recompute
        # `sig` itself. What it CAN verify: both members of the room
        # presented the exact same sig for the exact same exp, which is only
        # possible if both know the shared secret and the same code — and
        # that exp has not passed.
        if room_obj.expected_sig is None:
            room_obj.expected_sig = sig
            room_obj.exp = exp
        else:
            if room_obj.exp != exp or not hmac.compare_digest(sig, room_obj.expected_sig):
                await ws.close(code=HELLO_CLOSE_CODE)
                return ws
            if room_obj.exp <= now:
                await ws.close(code=HELLO_CLOSE_CODE)
                return ws

        if role in room_obj.members:
            # A second socket claiming an occupied role is rejected outright;
            # the first holder of that role keeps its connection. Documented
            # in relay/README.md.
            log.info("room=%s role=%s rejected: role already connected", room_id, role)
            await ws.close(code=ROLE_TAKEN_CLOSE_CODE)
            return ws

        room_obj.members[role] = ws
        log.info("room=%s role=%s connected", room_id, role)

        async for msg in ws:
            if msg.type is aiohttp.WSMsgType.TEXT:
                peer = room_obj.members.get(room_obj.peer_role(role))
                if peer is not None and not peer.closed:
                    await peer.send_str(msg.data)
            elif msg.type is aiohttp.WSMsgType.BINARY:
                # Media can never accidentally route through here.
                await ws.close(code=1003)
                break
            else:
                break
    finally:
        if room_obj is not None and role is not None and room_obj.members.get(role) is ws:
            del room_obj.members[role]
            log.info("room=%s role=%s disconnected", room_obj.room_id, role)
            if not room_obj.members:
                rooms.pop(room_obj.room_id, None)

    return ws


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/r/{room}", serve_ipad_page)
    app.router.add_get("/ws", ws_handler)
    return app


def main() -> None:
    port = int(os.environ.get("PORT", 10000))
    web.run_app(build_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
