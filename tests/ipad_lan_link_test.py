"""Tests for core/ipad_lan_link.py — the direct-LAN WebSocket transport the
native iPad app uses instead of WebRTC + relay.

Two layers:
  * Pure-logic tests construct an IPadLanLink and call its ingest/verify helpers
    directly (no server, no network) — these assert reassembly parity with
    IPadLink and the pairing HMAC checks.
  * One end-to-end test starts the real aiohttp server on loopback, connects an
    aiohttp WebSocket client, and exercises hello -> frame -> frame_ack, an
    inbound control message routed to on_control, and an outbound send_control.

Run standalone:  python tests/ipad_lan_link_test.py
"""
import asyncio
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ipad_camera import pack_frame_header
from core.ipad_lan_link import IPadLanLink
from core.ipad_link import pairing_expiry, pairing_signature

ROOM = "lan-room"
SECRET = "shared-secret-value"
CODE = "424242"


def _link(on_control=None, **kw):
    return IPadLanLink(room=ROOM, secret=SECRET, code=CODE,
                       on_control=on_control, **kw)


def _hello(room=ROOM, code=CODE, role="ipad", exp=None, sig=None):
    import json
    exp = pairing_expiry() if exp is None else exp
    sig = pairing_signature(SECRET, room, code, exp) if sig is None else sig
    return json.dumps({"type": "hello", "role": role, "room": room,
                       "exp": exp, "sig": sig})


def test_verify_hello_accepts_valid_and_rejects_forgeries():
    link = _link()
    assert link._verify_hello(_hello()) is True

    # Wrong code -> wrong signature.
    assert link._verify_hello(_hello(code="000000")) is False
    # Wrong room.
    assert link._verify_hello(_hello(room="other")) is False
    # Wrong role.
    assert link._verify_hello(_hello(role="host")) is False
    # Tampered signature.
    assert link._verify_hello(_hello(sig="deadbeef")) is False
    # Expired exp (in the past).
    past = int(time.time()) - 10
    assert link._verify_hello(
        _hello(exp=past, sig=pairing_signature(SECRET, ROOM, CODE, past))) is False
    # Absurd far-future exp, even with a matching signature, is refused so a
    # captured hello cannot be replayed indefinitely.
    far = int(time.time()) + 10 * 3600
    assert link._verify_hello(
        _hello(exp=far, sig=pairing_signature(SECRET, ROOM, CODE, far))) is False
    # Garbage / non-hello.
    assert link._verify_hello("not json") is False
    assert link._verify_hello('{"type":"offer"}') is False
    print("[ipad-lan-test] hello verification accepts valid, rejects forgeries OK")


def test_frame_reassembly_parity_with_webrtc_link():
    link = _link()

    # Single-chunk frame is admitted and its seq returned (so it gets ACKed).
    single = pack_frame_header(10, 0.5, 640, 480) + b"single-jpeg"
    assert link._ingest_frame(single) == 10

    # Multi-chunk frame: partial returns None, completion returns the seq.
    first = pack_frame_header(11, 0.55, 640, 480, 0, 2) + b"chunk-a"
    second = pack_frame_header(11, 0.55, 640, 480, 1, 2) + b"chunk-b"
    assert link._ingest_frame(first) is None
    assert link._ingest_frame(second) == 11

    # An older, already-superseded sequence is dropped (newest-wins), no ACK.
    older = pack_frame_header(9, 0.45, 640, 480) + b"older"
    assert link._ingest_frame(older) is None

    status = link.status()
    assert status["frames_rx"] == 2, status["frames_rx"]
    assert status["out_of_order"] == 1, status["out_of_order"]
    print("[ipad-lan-test] frame reassembly + newest-wins parity OK")


def test_seq_gap_accounting_matches_webrtc():
    link = _link()
    for seq in (1, 2, 4, 5):
        link._ingest_frame(pack_frame_header(seq, seq * 0.05, 640, 480) + b"d")
    status = link.status()
    assert status["seq_gaps"] == 1, status["seq_gaps"]
    assert status["frames_rx"] == 4, status["frames_rx"]
    print("[ipad-lan-test] seq-gap accounting parity OK")


def test_take_frame_newest_wins_and_coalesces():
    link = _link()
    link._ingest_frame(pack_frame_header(1, 0.1, 640, 480) + b"a")
    link._ingest_frame(pack_frame_header(2, 0.2, 640, 480) + b"b")
    link._ingest_frame(pack_frame_header(3, 0.3, 640, 480) + b"c")  # depth-2 drop
    header, payload = link.take_frame()
    assert header["seq"] == 3 and payload == b"c", (header["seq"], payload)
    assert link.take_frame() is None
    assert link.status()["coalesced"] >= 1
    print("[ipad-lan-test] take_frame newest-wins + coalesce OK")


def test_capture_profile_advances_generation():
    link = _link()
    import json
    prof = json.dumps({"type": "capture_profile", "width": 960, "height": 720,
                       "tier": 0, "quality_tier": 0, "jpeg_quality": 0.92})
    link._ingest_text(prof)
    gen1 = link.status()["capture_profile_generation"]
    assert gen1 == 1, gen1
    link._ingest_text(prof)                       # unchanged -> no advance
    assert link.status()["capture_profile_generation"] == 1
    prof2 = json.dumps({"type": "capture_profile", "width": 800, "height": 600,
                        "tier": 1, "quality_tier": 1, "jpeg_quality": 0.84})
    link._ingest_text(prof2)
    assert link.status()["capture_profile_generation"] == 2
    print("[ipad-lan-test] capture_profile generation ordering OK")


def test_control_routing_builtin_and_passthrough():
    seen = []
    link = _link(on_control=lambda payload, send: seen.append(payload))
    import json

    # Built-in: sender_stats is validated and stored, not routed.
    link._ingest_text(json.dumps({"type": "sender_stats", "sent_fps": 19.5,
                                  "width": 960, "height": 720}))
    stats = link.status()["sender_stats"]
    assert stats and stats["sent_fps"] == 19.5 and stats["width"] == 960

    # Built-in: client_version caps are sanitised into diagnostics.
    link._ingest_text(json.dumps({"type": "client_version", "build": "abc123",
                                  "ua": "CareVisioniPad/1.0",
                                  "caps": {"native": True, "ios": 13}}))
    assert link.status()["client_build"] == "abc123"
    assert link.status()["client_caps"] == {"native": True, "ios": 13}

    # Passthrough: unknown types (module toggles, asr_text) reach on_control.
    link._ingest_text(json.dumps({"type": "module", "action": "enable",
                                  "target": "fall"}))
    link._ingest_text(json.dumps({"type": "asr_text", "text": "I feel dizzy"}))
    kinds = [p.get("type") for p in seen]
    assert kinds == ["module", "asr_text"], kinds
    assert seen[1]["text"] == "I feel dizzy"
    print("[ipad-lan-test] control routing (built-in vs passthrough) OK")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_end_to_end_server_pairing_frame_ack_and_control():
    import json
    import aiohttp

    port = _free_port()
    seen = []
    link = _link(on_control=lambda payload, send: seen.append(payload),
                 listen_host="127.0.0.1", listen_port=port)
    link.start()
    # Wait for the server to bind (IPadCamera.open() waits on this same signal).
    deadline = time.time() + 5.0
    while time.time() < deadline and link.status().get("relay") != "connected":
        if link.status().get("error"):
            link.stop()
            raise AssertionError(link.status()["error"])
        time.sleep(0.02)
    assert link.status().get("relay") == "connected", link.status()

    async def client():
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
                await ws.send_str(_hello())
                # Push one frame; expect a frame_ack for its sequence.
                await ws.send_bytes(
                    pack_frame_header(7, 1.0, 640, 480) + b"jpeg-bytes")
                ack = json.loads((await asyncio.wait_for(ws.receive(), 3.0)).data)
                assert ack == {"type": "frame_ack", "seq": 7}, ack
                # Inbound control routed to on_control.
                await ws.send_str(json.dumps({"type": "asr_text",
                                              "text": "hello there"}))
                # Outbound control from the host reaches the app.
                await asyncio.sleep(0.05)
                link.send_control({"type": "telemetry", "v": 1})
                pushed = json.loads((await asyncio.wait_for(ws.receive(), 3.0)).data)
                assert pushed == {"type": "telemetry", "v": 1}, pushed

    try:
        asyncio.run(client())
        # on_control saw the asr_text (give the loop a beat to run it).
        deadline = time.time() + 2.0
        while time.time() < deadline and not seen:
            time.sleep(0.02)
        assert any(p.get("type") == "asr_text" for p in seen), seen
        assert link.status()["frames_rx"] == 1
        assert link.status()["frame_acks_tx"] == 1
    finally:
        link.stop()
    print("[ipad-lan-test] end-to-end pairing + frame_ack + control OK")


def main():
    test_verify_hello_accepts_valid_and_rejects_forgeries()
    test_frame_reassembly_parity_with_webrtc_link()
    test_seq_gap_accounting_matches_webrtc()
    test_take_frame_newest_wins_and_coalesces()
    test_capture_profile_advances_generation()
    test_control_routing_builtin_and_passthrough()
    test_end_to_end_server_pairing_frame_ack_and_control()
    print("[ipad-lan-test] OK")


if __name__ == "__main__":
    main()
