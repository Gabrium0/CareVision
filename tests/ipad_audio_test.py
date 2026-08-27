"""Focused tests for iPad two-way audio: PCM conversion, paced buffer,
TTS remote-sink routing, and the link's audio-aware answer path.

No iPad, network, microphone, or real WebRTC stack is required.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.tts import Speaker
from core.ipad_audio import PacedPcmSource, pcm_to_mono, to_mono_16k
from core.ipad_camera import IPadCamera
from core.ipad_link import IPadLink
from core.camera_factory import SwitchableCamera


# --------------------------------------------------------------- conversion

def test_packed_stereo_s16_48k_downmixes_to_mono_16k():
    left = np.full(960, 8000, dtype=np.int16)
    right = np.full(960, -4000, dtype=np.int16)
    interleaved = np.empty(1920, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    out = to_mono_16k(interleaved.reshape(1, -1), rate=48000, channels=2)

    assert out.shape == (320,)
    expected = np.float32(2000 / 32768)
    assert np.allclose(out, expected, atol=1e-6)


def test_planar_stereo_frame_layout_is_supported():
    planar = np.vstack([
        np.full(480, 16000, dtype=np.int16),
        np.full(480, 16000, dtype=np.int16),
    ])

    out = to_mono_16k(planar, rate=48000, channels=2)

    assert out.shape == (160,)
    assert np.allclose(out, 16000 / 32768, atol=1e-6)


def test_float32_16k_passthrough_and_odd_rate_interpolation():
    block = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    assert to_mono_16k(block, rate=16000, channels=1) is not None
    assert np.allclose(to_mono_16k(block, rate=16000), block)

    resampled = to_mono_16k(np.ones(441, dtype=np.float32), rate=44100)
    assert resampled.shape == (160,)
    assert np.allclose(resampled, 1.0)


def test_pcm_to_mono_degrades_on_unknown_shape():
    junk = np.zeros((3, 7), dtype=np.float32)
    assert pcm_to_mono(junk, channels=5).shape == (21,)
    # Integer scaling uses the s16 range for 16-bit types.
    scaled = pcm_to_mono(np.array([32767], dtype=np.int16), channels=1)
    assert 0.99 < float(scaled[0]) <= 1.0


# ------------------------------------------------------------ paced buffer

def test_paced_source_is_fifo_with_partial_reads():
    source = PacedPcmSource(block_samples=4)
    first = np.arange(10, dtype=np.float32)
    second = np.arange(100, 105, dtype=np.float32)
    assert source.write(first) == 10
    assert source.write(second) == 5
    assert source.pending() == 15

    assert source.read().tolist() == [0, 1, 2, 3]
    assert source.read().tolist() == [4, 5, 6, 7]
    tail = source.read(64)               # explicit cap above block size drains all
    assert tail.tolist() == [8, 9, 100, 101, 102, 103, 104]
    assert source.read() is None
    assert source.pending() == 0


def test_paced_source_rejects_nonfinite_and_clears():
    source = PacedPcmSource()
    nan_block = np.array([0.0, np.nan], dtype=np.float32)
    assert source.write(nan_block) == 0
    assert source.write(np.ones(4, dtype=np.float32)) == 4
    source.clear()
    assert source.read() is None


# ------------------------------------------------------------------ speaker

class _StubPiperBackend:
    """Two-chunk synthesis stand-in matching PiperBackend.synthesize."""

    def __init__(self):
        self.loaded = False

    def load(self):
        self.loaded = True
        return object()

    def synthesize(self, voice, text):
        yield np.full(160, 0.5, dtype=np.float32), 22050
        yield np.full(80, 0.25, dtype=np.float32), 22050


class _RecordingPlayer:
    def __init__(self):
        self.played = []

    def play(self, samples, rate):
        self.played.append((np.asarray(samples).copy(), rate))

    def wait(self):
        pass


def _speaker_with_backend(monkeypatch) -> Speaker:
    monkeypatch.setattr("audio.tts._SENTENCE_PAUSE", 0.0)
    speaker = Speaker(enabled=False)
    speaker.engine_name = "piper"
    speaker._backend = _StubPiperBackend()
    speaker._engine = None
    return speaker


def test_speaker_routes_chunks_to_remote_sink_and_mutes_local(monkeypatch):
    speaker = _speaker_with_backend(monkeypatch)
    # Isolate chunk routing from the Bluetooth-route lead pad (own test in
    # tests/echo_turn_taking_test.py).
    monkeypatch.setattr("audio.tts._REMOTE_LEAD_SECONDS", 0.0)
    routed = []
    speaker.remote_sink = lambda samples, rate: routed.append((samples, rate))
    speaker.local_playback = False

    speaker._emit("Hello there. How are you?", player=None)

    assert len(routed) == 2          # one call per synthesized sentence chunk
    samples, rate = routed[0]
    assert rate == 22050
    assert samples.dtype == np.float32
    assert speaker.status()["remote_sink"] is True


def test_speaker_keeps_local_audio_when_not_muted(monkeypatch):
    speaker = _speaker_with_backend(monkeypatch)
    routed = []
    speaker.remote_sink = lambda samples, rate: routed.append(len(samples))
    player = _RecordingPlayer()

    speaker._emit("Hello there. How are you?", player=player)

    assert len(routed) == 2          # sink still receives a copy
    assert len(player.played) == 2   # laptop speakers stayed on


def test_speaker_falls_back_to_local_when_sink_fails(monkeypatch):
    speaker = _speaker_with_backend(monkeypatch)

    def broken_sink(samples, rate):
        raise RuntimeError("peer gone")

    speaker.remote_sink = broken_sink
    speaker.local_playback = False
    player = _RecordingPlayer()

    speaker._emit("Still audible?", player=player)

    assert speaker.local_playback is True
    assert len(player.played) == 1   # the chunk failed over to local speakers


# -------------------------------------------------------------------- link

def _link():
    return IPadLink(relay_url="https://relay.invalid", room="room",
                    secret="secret", code="123456")


def test_send_audio_without_peer_is_a_safe_noop():
    link = _link()
    link.send_audio(np.ones(320, dtype=np.float32), 22050)
    assert link.status()["audio_tx_chunks"] == 0


def test_send_audio_buffers_into_live_source_and_teardown_clears_it():
    link = _link()
    source = PacedPcmSource()
    link._audio_source = source

    link.send_audio(np.full(960, 8000, dtype=np.int16).reshape(1, -1),
                    rate=48000, channels=1)

    assert source.pending() > 0
    assert link.status()["audio_tx_chunks"] == 1

    asyncio.run(link._teardown())
    assert link._audio_source is None
    assert source.pending() == 0
    assert link.status()["audio_out"] is False


# --------------------------------------------- negotiation with a fake peer

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


class _Sender:
    def __init__(self):
        self.replaced = []

    def replaceTrack(self, track):
        self.replaced.append(track)


class _AudioTransceiver:
    kind = "audio"

    def __init__(self):
        self.direction = "recvonly"
        self.sender = _Sender()


class _VideoTransceiver:
    kind = "video"

    def __init__(self):
        self.direction = "sendrecv"


class _RemoteMicTrack:
    kind = "audio"

    def __init__(self):
        self._release = asyncio.Event()

    async def recv(self):
        await self._release.wait()


class _VideoTrack:
    kind = "video"

    def __init__(self):
        self._queue = asyncio.Queue()
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class _PeerConnection:
    instances = []

    def __init__(self, configuration):
        self.configuration = configuration
        self.handlers = {}
        self.audio = _AudioTransceiver()
        self.video = _VideoTransceiver()
        self.mic_track = _RemoteMicTrack()
        self.video_track = _VideoTrack()
        self.closed = False
        self.__class__.instances.append(self)

    def on(self, name):
        def register(handler):
            self.handlers[name] = handler
            return handler
        return register

    async def setRemoteDescription(self, description):
        self.handlers["track"](self.mic_track)
        self.handlers["track"](self.video_track)

    def getTransceivers(self):
        return [self.audio, self.video]

    async def createAnswer(self):
        return _Description("v=0\r\n", "answer")

    async def setLocalDescription(self, description):
        self.localDescription = description

    async def close(self):
        self.closed = True


class _WebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


def _install_fake_webrtc(monkeypatch):
    aiortc = types.ModuleType("aiortc")
    aiortc.RTCConfiguration = _Configuration
    aiortc.RTCIceServer = _IceServer
    aiortc.RTCPeerConnection = _PeerConnection
    aiortc.RTCSessionDescription = _Description
    mediastreams = types.ModuleType("aiortc.mediastreams")

    class AudioStreamTrack:
        kind = "audio"

        def __init__(self):
            pass

    mediastreams.AudioStreamTrack = AudioStreamTrack
    aiortc.mediastreams = mediastreams
    monkeypatch.setitem(sys.modules, "aiortc", aiortc)
    monkeypatch.setitem(sys.modules, "aiortc.mediastreams", mediastreams)
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))


def test_answer_negotiates_sendrecv_audio_and_rejects_video(monkeypatch):
    _PeerConnection.instances.clear()
    _install_fake_webrtc(monkeypatch)
    link = _link()
    ws = _WebSocket()

    async def scenario():
        await link._answer_offer(ws, {"sdp": "offer"})
        peer = _PeerConnection.instances[-1]

        assert peer.audio.direction == "sendrecv"
        assert len(peer.audio.sender.replaced) == 1
        assert peer.video.direction == "inactive"
        assert peer.video_track.stop_calls == 1
        assert ws.messages[0]["type"] == "answer"

        status = link.status()
        assert status["audio_out"] is True
        assert status["audio_mic"] is True

        # TTS written at piper's native rate lands converted in the buffer.
        link.send_audio(np.full(220, 0.5, dtype=np.float32), rate=22050)
        source = link._audio_source
        assert source is not None and source.pending() == 160   # 22050 -> 16k

        await asyncio.sleep(0)      # let the mic consumer task take its turn
        await link._teardown()
        assert peer.closed is True
        assert link.status()["audio_out"] is False
        assert link.status()["audio_mic"] is False

    asyncio.run(scenario())


class _TwoAudioLinePeer(_PeerConnection):
    """The real browser offer: mic m-line (sendonly) THEN a dedicated agent-voice
    m-line (recvonly), in that order — what the single-line fake above misses."""

    def __init__(self, configuration):
        super().__init__(configuration)
        self.mic = _AudioTransceiver()
        self.mic.direction = "sendonly"
        self.agent = _AudioTransceiver()
        self.agent.direction = "recvonly"

    def getTransceivers(self):
        return [self.mic, self.agent, self.video]


_TWO_LINE_OFFER = (
    "v=0\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:0\r\na=sendonly\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:1\r\na=recvonly\r\n"
    "m=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:2\r\na=sendrecv\r\n"
)


def test_answer_attaches_agent_track_to_the_recvonly_slot(monkeypatch):
    # Regression: the agent voice must land on the m-line the device offered
    # recvonly (answered sendonly), NOT on the sendonly mic line — Safari has no
    # receiver there and drops the audio, so nothing plays.
    _PeerConnection.instances.clear()
    _install_fake_webrtc(monkeypatch)
    sys.modules["aiortc"].RTCPeerConnection = _TwoAudioLinePeer
    link = _link()
    ws = _WebSocket()

    async def scenario():
        await link._answer_offer(ws, {"sdp": _TWO_LINE_OFFER})
        peer = _PeerConnection.instances[-1]
        assert isinstance(peer, _TwoAudioLinePeer)
        # Agent voice out on the recvonly slot, answered sendonly + track attached.
        assert peer.agent.direction == "sendonly"
        assert len(peer.agent.sender.replaced) == 1
        # Mic line receives only; the agent track is never put on it.
        assert peer.mic.direction == "recvonly"
        assert peer.mic.sender.replaced == []
        assert link.status()["audio_out"] is True
        assert link.status()["audio_mic"] is True
        await asyncio.sleep(0)
        await link._teardown()

    asyncio.run(scenario())


def test_camera_facade_late_binds_audio_bus_and_survives_rebuild_paths():
    camera = IPadCamera(source="ipad")
    bus = object()
    camera.attach_audio_bus(bus)
    assert camera._link_opts["audio_bus"] is bus

    facade = SwitchableCamera("ipad", {})
    facade.ipad_attach_audio_bus(bus)
    assert isinstance(facade.inner, IPadCamera)
    assert facade.inner._link_opts["audio_bus"] is bus
    facade.ipad_send_agent_audio(np.zeros(320, dtype=np.float32))  # no-op safe


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
