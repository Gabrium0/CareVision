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

## On-device features (round 2)

The app pushes perception onto the iPad where doing so improves accuracy,
bandwidth, or responsiveness — using **Apple Vision** (`VNDetectFaceLandmarksRequest`,
hardware-accelerated) plus the near-free `AVCaptureMetadataOutput` face detector,
chosen over bundling MediaPipe FaceMesh so the A10 (iPad 6th gen) never lags.

- **On-device rPPG colour streaming (flagship).** `Capture/RPPGSampler.swift` samples
  forehead/cheek skin colour from the **raw** BGRA pixel buffer (no JPEG chroma loss)
  with the frame's precise PTS, applies the same mouth-motion talk-guard as the
  backend, and streams a combined RGB mean as `rppg_samples`
  (`{type:"rppg_samples", s:[[t,r,g,b,n]]}`). The backend feeds these straight into
  the classical HR / SpO₂ `TimedBuffer`s (`modules/rppg_backends/classical.py`,
  `modules/spo2.py`) via `ctx.extras["rppg_samples"]`, bypassing `roi_patch` — no
  change to `compute()`. This removes the two dominant rPPG error sources (JPEG 4:2:0
  chroma + timestamp jitter) and lets the full frame stay a normal 4:2:0 JPEG for the
  backend's MediaPipe/appearance detectors. Auto-enabled when the app's
  `client_version` caps report `ondevice_rppg`; active mode shows in `camera`
  diagnostics (`ondevice_rppg`, `ondevice_rppg_samples`).
- **Presence + framing coaching.** `AVCaptureMetadataOutput` drives a centered
  "move closer / center yourself / no one in view" banner — directly addressing the
  `showcase.min_face_px` block that otherwise stops HR from ever starting.
- **On-device quick signals.** Vision landmarks yield EAR (blink), MAR (yawn), and
  head yaw, shown as an on-screen chip for instant feedback. **UI only** — the
  backend stays authoritative for the real detections.
- **ROI-crop bandwidth cut (scoped).** On 13.3.1 the backend still needs full frames
  for MediaPipe, so crops cannot replace frames; the on-device face ROIs (already
  computed for rPPG) and the framing gate are the enablers. Full ROI-crop routing to
  the appearance detectors + full-frame reduction is a backend-compositing follow-on.

**Perf discipline (no lag):** metadata face box is always-on and near-free; the Vision
landmark pass is throttled to ~12 fps, one request in flight on its own queue, and
never blocks the capture queue; rPPG averaging touches only a few hundred pixels.

Backend seam: `main.py` `ipad_control` routes `rppg_samples` →
`IPadCamera.ingest_rppg_samples`; `drain_rppg_samples` maps the app's PTS to the
capture clock; `core/pipeline.py::_fast_hook` forwards the batch onto `fast_ctx`.

## App polish & experience (round 3)

- **Effortless pairing.** The laptop prints a scannable QR of a
  `carevision://pair?h=&p=&r=&c=&s=` URL (`build_lan_pairing_url` in `main.py`,
  rendered with `qrcode` like `scripts/show_qr.py`). The app's "📷 Scan QR to
  connect" button opens `QRScannerViewController` (`AVCaptureMetadataOutput` `.qr`),
  parses it (`PairingURL`), and connects — no typing. The shared secret is stored in
  the **Keychain** (`KeychainStore`), not `UserDefaults` (old values migrate on
  launch).
- **Stable on the A10.** An ACK-latency-driven `AdaptiveLadder` steps resolution
  `[960,800,640]` / quality `[0.90→0.72]` / fps `[20→10]` down as the link's rolling
  ACK p90 rises or the sender window backs up, and back up on recovery (with
  hysteresis), announcing each change as a `capture_profile`. `ProcessInfo`
  thermal state pins a minimum tier and pauses the Vision pass under pressure. The
  app keeps the screen awake, pauses/resumes camera + mic around backgrounding, and
  the status pill is tap-to-reconnect.
- **Voice & alerts.** Barge-in: input voice processing (hardware AEC) plus a light
  energy VAD lets the person interrupt the agent mid-sentence
  (`synth.stopSpeaking`). Caregiver alerts get a distinct chime + a dominant red
  banner (no haptics — iPads have none). A settings sheet controls TTS voice, rate,
  and volume/mute (`SettingsStore`).

Honest scope: iOS has no clean hardware-JPEG API, so the CPU/heat wins come from the
adaptive ladder + thermal frame-skip rather than a faster encoder; an H.264 transport
via VideoToolbox is the larger future lever (now viable since rPPG left the frame).
Bonjour auto-discovery and battery-bias are noted as optional follow-ons.

## Verification status

- `tests/ipad_lan_link_test.py` — pairing accept/reject, IPF1 reassembly parity
  with `IPadLink`, control routing, and an end-to-end aiohttp server round trip
  (hello → frame → `frame_ack`, `asr_text` → `on_control`, `send_control` → app).
- Cross-language interop checked byte-for-byte: the Swift `Pairing.signature`
  matches `core/ipad_link.pairing_signature`, and the Swift `FrameProtocol.header`
  matches `core/ipad_camera.unpack_frame_header`.
- iOS app compiles for the 13.3 target and launches in the simulator (pairing
  screen renders; no crash). On-device camera/voice are the user's device check.
- `tests/ipad_rppg_samples_test.py` — `rppg_samples` validation, camera
  ingest/drain (clock-mapped, bounded), and the device-first branch: it bypasses
  `roi_patch`, and a synthetic 72 bpm colour sinusoid recovers ~72 bpm through the
  unchanged `compute()`. The Swift BGRA patch-mean and EAR/MAR arithmetic are
  cross-checked standalone. Vision throughput and rPPG A/B are device-verified.
- Round 3: the Python `carevision://pair` URL is cross-checked byte-for-byte against
  the Swift `PairingURL` parser (special chars in the secret included), and the pure
  `AdaptiveLadder` (degrade / cooldown / recover / thermal-floor) is verified
  standalone. The app builds for 13.3 and the QR-first pairing screen renders in the
  simulator. Barge-in, thermal throttling, and alert chime are device-verified.
