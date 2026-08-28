# Native iPad app (iPadOS 13.3.1)

A native iPadOS app (`ios/CareVisioniPad`) that replaces the Safari page in
`relay/static/ipad.html` as the iPad front end. It connects **directly to the
laptop over the Windows Mobile Hotspot** — no cloud relay, no WebRTC — and does
on-device what the iPad does better, leaving heavy perception on the backend.

## Why native, and why direct-LAN

The browser page existed for two reasons stated in `relay/README.md`: Safari only
exposes `getUserMedia` from an HTTPS origin, and ordinary Wi-Fi isolates AP
clients so the iPad could not reach the laptop's LAN IP. **Both vanish on the
hotspot with a native app** — the iPad reaches the laptop directly at
`192.168.137.x`, and AVFoundation needs no secure context. So the native app drops
the relay, WebRTC, STUN/ICE, and SCTP chunking, and speaks the same wire contract
over one plain WebSocket.

The iPad is kept on iPadOS 13.3.1 deliberately, which rules out iOS-14 APIs
(notably native body pose). ~40 of the ~49 detectors consume MediaPipe's dense
478-point face mesh (+iris) or the 33-point body pose, which iPadOS 13.3.1's
Vision cannot reproduce (only ~76 coarse face points, no iris, no body pose). So
**accurate perception keeps receiving camera frames on the backend**; the real
native wins are raw camera capture with precise timestamps, on-device TTS/ASR, and
a native control-surface UI.

## What runs where

| Native on the iPad | Backend (unchanged) |
|---|---|
| Camera capture (AVFoundation), JPEG encode, precise `CMSampleBuffer` PTS timestamps (clock source `avfoundation-pts`) | MediaPipe face-mesh + pose extractors, all detectors, aggregator, `HistoryStore`, `alerts/` |
| UIKit control surface: vitals, demo tiles, signal feed, module toggles, agent pill | rPPG, SpO₂, VLMs, torch/onnx detectors |
| TTS via `AVSpeechSynthesizer` (speaks the backend's guarded lines) | LLM phrasing + its deterministic guards/fallbacks |
| ASR via `SFSpeechRecognizer` on-device → sent as `asr_text` | Deterministic caregiver alerts |
| Pairing HMAC, windowed-ACK flow control, diagnostics | — |

Deferred (not v1): on-device face-ROI rPPG sampling, ROI-crop bandwidth cuts, and
raw-PCM audio streaming for YAMNet cough/smoke.

## Transport / protocol

`core/ipad_lan_link.py::IPadLanLink` is a drop-in for `IPadLink`, selected by
`IPadCamera._build_link()` when `--ipad-transport lan`. It binds an aiohttp
WebSocket server on the hotspot interface and speaks the existing contract:

- **Pairing**: the app's first message is the same `hello`
  (`{room, role:"ipad", exp, sig}`) the browser computed; the laptop verifies
  `sig = HMAC-SHA256(secret, "room|code|exp")[:32]` against the printed 6-digit
  code. No relay mediates, so the laptop checks the HMAC itself.
- **Frames**: binary WebSocket messages are IPF1 frames (24-byte little-endian
  header `<4sIdHHHH>` + JPEG), reassembled with the same newest-wins + windowed
  ACK logic as the WebRTC path.
- **Control**: text JSON both ways — iPad→laptop `client_version`, `clock_source`,
  `camera_tuning`, `sender_stats`, `capture_profile`, `module`, `asr_text`,
  `speaking`; laptop→iPad `telemetry`, `state`, `module_result`, `frame_ack`,
  `speak`.

Voice is text-only (no audio media on the wire): the backend sends the guarded
`voice_agent.public_line()` as a `speak` message and keeps the laptop silent; the
app transcribes on-device and returns `asr_text`, fed through the same
`TypedListener` contract so the agent's self-echo / cadence / no-repeat guards
apply. **Use `--tts piper`** so the laptop Speaker stays muted while the iPad
voices the line.

## Running it

Backend (Windows laptop, system Python):

```powershell
python main.py --source ipad --ipad-transport lan --ipad-room care --tts piper --listen --webui
```

- `IPAD_ROOM` and `RELAY_SECRET` must be set (in `.env` or via `--ipad-room`); no
  `IPAD_RELAY_URL` is needed for LAN.
- The server binds `192.168.137.1:8788` by default (`--ipad-listen-host` /
  `--ipad-listen-port` to override; use `0.0.0.0` if the hotspot gateway differs).
- Allow the port inbound on the **Private** firewall profile:

  ```powershell
  New-NetFirewallRule -DisplayName "CareVision LAN" -Direction Inbound `
    -Protocol TCP -LocalPort 8788 -Profile Private -Action Allow
  ```

On the iPad: launch the app, enter the laptop address, port, room, shared secret,
and the printed 6-digit code, then Connect.

## Build & deploy

- The Xcode project is `ios/CareVisioniPad.xcodeproj` (Swift + UIKit, Apple
  frameworks only — no CocoaPods/WebRTC). Deployment target **iOS 13.3**;
  landscape-only.
- Simulator build/UI smoke:

  ```bash
  xcodebuild -project ios/CareVisioniPad.xcodeproj -target CareVisioniPad \
    -sdk iphonesimulator -configuration Debug build
  ```

- **Installing on the real 13.3.1 iPad is a device-side step.** Xcode 26 can set
  the 13.3 deployment target and compile, but it ships no iOS 13.3 DeviceSupport
  files and no iOS 13 simulator. To run on the device, add iOS 13.3 DeviceSupport
  files to Xcode (or sideload a dev-signed `.ipa`) and sign with an Apple ID /
  developer account (free = 7-day resign; paid = 1 year). Camera, ASR/TTS, and the
  final install are verified on the physical iPad.

## Verification status

- `tests/ipad_lan_link_test.py` — pairing accept/reject, IPF1 reassembly parity
  with `IPadLink`, control routing, and an end-to-end aiohttp server round trip
  (hello → frame → `frame_ack`, `asr_text` → `on_control`, `send_control` → app).
- Cross-language interop checked byte-for-byte: the Swift `Pairing.signature`
  matches `core/ipad_link.pairing_signature`, and the Swift `FrameProtocol.header`
  matches `core/ipad_camera.unpack_frame_header`.
- iOS app compiles for the 13.3 target and launches in the simulator (pairing
  screen renders; no crash). On-device camera/voice are the user's device check.
