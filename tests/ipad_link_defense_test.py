"""Focused tests for stale-page media rejection and bounded sender telemetry.

No iPad, network, or aiortc installation is required.  A small fake peer
connection exercises the lazy-imported offer path and reproduces aiortc's
unbounded RemoteStreamTrack queue shape.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ipad_link import IPadLink, _validated_sender_stats
from core.ipad_camera import pack_frame_header


class _Description:
    def __init__(self, sdp: str, type: str):
        self.sdp = sdp
        self.type = type


class _IceServer:
    def __init__(self, urls):
        self.urls = urls


class _Configuration:
    def __init__(self, iceServers):
        self.iceServers = iceServers


class _Track:
    kind = "video"

    def __init__(self):
        # Matches aiortc.RemoteStreamTrack's private, unbounded output queue.
        self._queue = asyncio.Queue()
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class _Transceiver:
    def __init__(self):
        self.direction = "recvonly"


class _Channel:
    def __init__(self, label: str):
        self.label = label
        self.handlers = {}
        self.readyState = "open"
        self.sent = []
        self.fail_send = False

    def on(self, name: str):
        def register(handler):
            self.handlers[name] = handler
            return handler
        return register

    def send(self, message):
        if self.fail_send:
            raise RuntimeError("simulated closed SCTP transport")
        self.sent.append(message)


class _PeerConnection:
    instances = []

    def __init__(self, configuration):
        self.configuration = configuration
        self.handlers = {}
        self.connectionState = "new"
        self.iceConnectionState = "new"
        self.transceiver = _Transceiver()
        self.track = _Track()
        self.direction_at_answer = None
        self.localDescription = None
        self.closed = False
        self.__class__.instances.append(self)

    def on(self, name: str):
        def register(handler):
            self.handlers[name] = handler
            return handler
        return register

    async def setRemoteDescription(self, description):
        self.remoteDescription = description
        self.handlers["track"](self.track)
        # Model frames decoded during the pre-answer window. The defensive
        # drain must consume them even though track.stop() was already called.
        self.track._queue.put_nowait(object())
        self.track._queue.put_nowait(object())

    def getTransceivers(self):
        return [self.transceiver]

    async def createAnswer(self):
        self.direction_at_answer = self.transceiver.direction
        return _Description("v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
                            "answer")

    async def setLocalDescription(self, description):
        self.localDescription = description

    async def close(self):
        self.closed = True


class _WebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


def _install_fake_aiortc(monkeypatch):
    module = types.ModuleType("aiortc")
    module.RTCConfiguration = _Configuration
    module.RTCIceServer = _IceServer
    module.RTCPeerConnection = _PeerConnection
    module.RTCSessionDescription = _Description
    monkeypatch.setitem(sys.modules, "aiortc", module)


def _link(on_control=None):
    return IPadLink(relay_url="https://relay.invalid", room="room",
                    secret="secret", code="123456", on_control=on_control)


def test_stale_offer_rejects_media_and_preserves_datachannels(monkeypatch):
    _PeerConnection.instances.clear()
    _install_fake_aiortc(monkeypatch)
    link = _link()
    ws = _WebSocket()

    async def scenario():
        await link._answer_offer(ws, {"sdp": "stale-page-offer"})
        peer = _PeerConnection.instances[-1]

        assert peer.direction_at_answer == "inactive"
        assert peer.transceiver.direction == "inactive"
        assert peer.track.stop_calls >= 1
        assert ws.messages == [{
            "type": "answer",
            "sdp": peer.localDescription.sdp,
        }]

        # Give the discard task a turn: queued frames must not accumulate even
        # if a version begins decoding before the inactive answer is applied.
        await asyncio.sleep(0)
        assert peer.track._queue.empty()

        frames = _Channel("frames")
        control = _Channel("control")
        peer.handlers["datachannel"](frames)
        peer.handlers["datachannel"](control)
        assert link.status()["dc"] == "open"
        assert link._control_channel is control
        assert "message" in frames.handlers
        assert "message" in control.handlers

        # A fully admitted JPEG gets one bounded application ACK on the same
        # frames channel. ACK send failure must not escape the receive callback
        # or undo admission of the next completed frame.
        frames.handlers["message"](
            pack_frame_header(7, 0.35, 640, 480) + b"jpeg-seven")
        assert frames.sent == ['{"type":"frame_ack","seq":7}']
        assert link.status()["frame_acks_tx"] == 1

        frames.fail_send = True
        frames.handlers["message"](
            pack_frame_header(8, 0.40, 640, 480) + b"jpeg-eight")
        assert link.status()["frames_rx"] == 2
        assert link.status()["frame_ack_errors"] == 1

        await link._teardown()
        assert peer.closed is True

    asyncio.run(scenario())


def test_media_defense_handles_optional_api_variants():
    class StopFallback:
        stopped = False

        @property
        def direction(self):
            return "recvonly"

        @direction.setter
        def direction(self, _value):
            raise AttributeError("read-only on this version")

        async def stop(self):
            self.stopped = True

    class Peer:
        def __init__(self, transceiver):
            self.transceiver = transceiver

        def getTransceivers(self):
            return [self.transceiver]

    async def scenario():
        link = _link()
        fallback = StopFallback()
        await link._deactivate_media_transceivers(Peer(fallback))
        assert fallback.stopped is True

        # A still older compatible peer may expose no transceiver API. The
        # track-level queue drain remains the bounded-memory safety net.
        track = types.SimpleNamespace(kind="audio", _queue=asyncio.Queue())
        link._reject_unexpected_track(track)
        track._queue.put_nowait(object())
        await asyncio.sleep(0)
        assert track._queue.empty()
        await link._deactivate_media_transceivers(object())
        await link._clear_unexpected_tracks()

    asyncio.run(scenario())


def test_real_aiortc_answer_is_inactive_and_keeps_sctp_when_available():
    aiortc = pytest.importorskip("aiortc")

    async def scenario():
        config = aiortc.RTCConfiguration(iceServers=[])
        offerer = aiortc.RTCPeerConnection(configuration=config)
        answerer = aiortc.RTCPeerConnection(configuration=config)
        link = _link()
        offerer.addTrack(aiortc.VideoStreamTrack())
        offerer.createDataChannel("frames")
        offerer.createDataChannel("control")

        @answerer.on("track")
        def reject(track):
            link._reject_unexpected_track(track)

        try:
            offer = await offerer.createOffer()
            await answerer.setRemoteDescription(offer)
            await link._deactivate_media_transceivers(answerer)
            answer = await answerer.createAnswer()

            media_sections = ["m=" + section for section in answer.sdp.split("m=")[1:]]
            video = next(section for section in media_sections
                         if section.startswith("m=video "))
            application = next(section for section in media_sections
                               if section.startswith("m=application "))
            assert "\r\na=inactive\r\n" in video
            assert "UDP/DTLS/SCTP" in application
        finally:
            await answerer.close()
            await offerer.close()
            await link._clear_unexpected_tracks()

    asyncio.run(scenario())


def test_sender_stats_are_bounded_copied_and_not_forwarded():
    forwarded = []
    link = _link(on_control=lambda payload, _send: forwarded.append(payload))
    report = {
        "type": "sender_stats",
        "sent_fps": 19.876,
        "target_fps": 20,
        "width": 960.0,
        "height": 720,
        "tier": 2,
        "quality_tier": 3,
        "jpeg_quality": 0.801,
        "encode_p90_ms": 41.237,
        "ack_p90_ms": 52.345,
        "jpeg_bytes": 245_000,
        "buffered_bytes": 65_536,
        "encode_busy_skips": 3,
        "backpressure_skips": 1,
        "transport_wait_skips": 14,
        "ack_timeouts": 0,
        "constrained": True,
        "extra": {"raw": "must not be retained"},
    }
    link._ingest_control(json.dumps(report))

    expected = {
        "sent_fps": 19.88,
        "target_fps": 20.0,
        "width": 960,
        "height": 720,
        "tier": 2,
        "quality_tier": 3,
        "jpeg_quality": 0.8,
        "encode_p90_ms": 41.24,
        "ack_p90_ms": 52.34,
        "jpeg_bytes": 245_000,
        "buffered_bytes": 65_536,
        "encode_busy_skips": 3,
        "backpressure_skips": 1,
        "transport_wait_skips": 14,
        "ack_timeouts": 0,
        "constrained": True,
    }
    first = link.status()
    assert first["sender_stats"] == expected
    assert forwarded == []

    # status() must not expose the retained nested dict by reference.
    first["sender_stats"]["width"] = 2
    assert link.status()["sender_stats"]["width"] == 960

    # A report with no valid whitelisted fields neither erases the last good
    # sample nor escapes into the module-control callback.
    malformed = {"type": "sender_stats", "sent_fps": True, "width": 1,
                 "tier": 17, "encode_p90_ms": float("nan"),
                 "constrained": "yes", "unexpected": "ignored"}
    link._ingest_control(json.dumps(malformed))
    assert link.status()["sender_stats"] == expected
    assert forwarded == []

    assert _validated_sender_stats({
        "sent_fps": 12.5, "width": 640.5, "height": 480,
        "jpeg_bytes": 33_554_433, "constrained": False,
    }) == {"sent_fps": 12.5, "height": 480, "constrained": False}


def test_teardown_clears_partial_and_complete_frames_before_sequence_reuse():
    link = _link()
    link._partial[0] = {0: b"old-first-chunk"}
    link._frames.append(({"seq": 99}, b"old-complete-frame"))
    link._last_seq = 99

    asyncio.run(link._teardown())

    assert link._partial == {}
    assert list(link._frames) == []
    assert link._last_seq is None
