# Operations

This guide is the canonical operator reference for installing and running
CareVision. Command-line defaults remain authoritative in `main.py --help` and
`dev.py --help`; replay contents remain authoritative in
`config/replay_scenarios.json`.

## Environment setup

The repository's known Windows environment is `.venv`:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py --help
```

If the intended environment is already active, use `python` in place of the
explicit interpreter. Do not install every optional ML stack by default. Add
only the capability group required for the run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-agent.txt
.venv\Scripts\python.exe -m pip install -r requirements-asr.txt
.venv\Scripts\python.exe -m pip install -r requirements-audio-events.txt
.venv\Scripts\python.exe -m pip install -r requirements-realsense.txt
.venv\Scripts\python.exe -m pip install -r requirements-skin.txt
.venv\Scripts\python.exe -m pip install -r requirements-clothing.txt
.venv\Scripts\python.exe -m pip install -r requirements-openrppg.txt
.venv\Scripts\python.exe -m pip install -r requirements-sensors.txt
```

Some optional GPU stacks have platform-specific binary compatibility
requirements. Follow the comments in their requirement file and the relevant
feature guide instead of mixing arbitrary Torch, CUDA, JAX, or CTranslate2
builds.

Copy `.env.example` to `.env` only for integrations that need credentials.
Leaving optional credentials unset is supported. Never print `.env`, place
secrets in YAML, or copy provider responses into diagnostics.

## Safe hardware-free run

Use a deterministic replay for development and verification:

```powershell
.venv\Scripts\python.exe main.py --source replay:dev_hot_reload --headless --debug-endpoint --dev-mode --no-voice --no-moondream --max-frames 600
```

This mode needs no camera, disables heavyweight/network-backed optional paths,
and exits after its frame budget. `--no-voice` disables spoken output;
`--no-moondream` prevents language-provider calls. The replay source itself
does not authorize microphone, camera upload, or other cloud features.

Other scenario names are defined in `config/replay_scenarios.json`. Useful
operator examples include:

```powershell
.venv\Scripts\python.exe main.py --source replay:client_demo --webui --no-moondream
.venv\Scripts\python.exe main.py --source replay:kitchen_spill --headless --no-voice --no-moondream
.venv\Scripts\python.exe main.py --source replay:topic_coverage --headless --dev-mode --no-voice --no-moondream --max-frames 600
```

A replay process exits when its scripted scenario ends, taking its web servers
with it.

## Camera and video sources

Run the default webcam, a video file, or RealSense explicitly:

```powershell
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe main.py --source clip.mp4
.venv\Scripts\python.exe main.py --source realsense
```

List available conventional cameras or configure a live alternate source:

```powershell
.venv\Scripts\python.exe main.py --list-cameras
.venv\Scripts\python.exe main.py --source 1 --alt-source 0
```

Press `c` in a windowed run to request the alternate source. Camera resolution
defaults to automatic probing; use `--resolution WIDTHxHEIGHT` to request a
specific size and `--min-fps` to set the automatic-selection floor. Inspect
`main.py --help` for the current defaults and quality-profile choices.

RealSense needs `requirements-realsense.txt` and physical hardware. Use the
[RealSense guide](REALSENSE_D435I.md) for depth/IMU expectations and the
hardware validation checklist. Never claim that path was verified from replay.

### iPad as the camera and main interface

`--source ipad` takes frames from an iPad browser over WebRTC instead of a local
device, and lets the same page drive the detector toggles. Needs
`requirements-ipad.txt` on the laptop and the `relay/` service deployed
somewhere with HTTPS (see `relay/README.md`).

Two constraints force this shape, and neither is optional:

- `getUserMedia` only exists in a **secure context**. On a plain `http://` LAN
  address Safari leaves `navigator.mediaDevices` undefined — not
  permission-denied, absent. That is why the page is served from an HTTPS relay
  rather than from `webui/server.py`.
- From that HTTPS page, `http://` and `ws://` back to the non-TLS laptop are
  mixed-content blocked. WebRTC is exempt, because DTLS-SRTP encrypts it
  unconditionally. So WebRTC carries both the video and the control messages,
  and the relay only ever sees SDP/ICE.

The relay can be hosted (see `relay/render.yaml`) or simply **run locally behind
a tunnel**, which is the lower-friction option: nothing to deploy, no cold
start, and the shared secret never leaves the machine.

**Fastest path — one launcher script.** `scripts/start-showcase.ps1` starts all
three processes (relay, `cloudflared`, the app) itself — as tabs in one
Windows Terminal window when `wt` is installed, otherwise as separate pwsh
windows — captures that run's fresh `trycloudflare.com` URL straight out of
the `cloudflared` log and passes it to `main.py --ipad-relay-url` — so
`IPAD_RELAY_URL` in `.env` never needs manual updating — and adds a fourth
tab/window with a scannable QR code (`scripts/show_qr.py`) for the pairing
URL:

```powershell
pwsh -File scripts\start-showcase.ps1
```

Extra flags are forwarded to `main.py`, e.g. `pwsh -File scripts\start-showcase.ps1
--no-ipad-toggle`. It reads `RELAY_SECRET` and `IPAD_ROOM` out of `.env` into its
own session only (never printed, never on a command line, `.env` itself is
never modified), and requires `qrcode` (`requirements-ipad.txt`) for the QR —
without it, `show_qr.py` degrades to printing the plain URL.

**Manual path**, in three terminals, is equivalent to what the script above
automates:

```powershell
$env:RELAY_SECRET=(Select-String '^RELAY_SECRET=' .env).Line.Split('=',2)[1]
python relay\server.py

cloudflared tunnel --url http://localhost:10000
```

`cloudflared` prints an `https://<random>.trycloudflare.com` URL — that is the
HTTPS origin, and the only thing the relay was ever needed for. Quick-tunnel
URLs change on every restart, so the iPad bookmark breaks each session; a named
Cloudflare tunnel (free, needs an account and a domain) gives a stable one.
Media still goes peer-to-peer over WebRTC either way — the tunnel carries only
the page load and the SDP/ICE handshake.

```powershell
python main.py --source ipad --webui --enable-multi-person --ipad-relay-url https://<random>.trycloudflare.com
```

`--ipad-relay-url` overrides `IPAD_RELAY_URL` from `.env` for this run, which is
how the launcher script above avoids ever touching `.env`; pass it explicitly
(or edit `.env`) with whatever URL `cloudflared` printed this time. `IPAD_ROOM`
and `RELAY_SECRET` still come from `.env` — never put them on the command line,
where they would land in shell history and the process list.

Run this and `relay/server.py` on **system Python**, not `.venv`:
`requirements-ipad.txt` documents that `.venv` on a typical dev machine is a
stale, unused environment missing `aiortc`/`cv2`/`mediapipe`/`torch`, while
system Python (on `PATH`) has the actual working set. Confirm with
`python -c "import aiortc, cv2, mediapipe, torch; print('ok')"` before relying
on either interpreter.

The run prints a bookmarkable URL and a fresh 6-digit pairing code. **Start the
laptop first**: the relay's free tier can take ~60s to wake, and whoever
connects first waits for it.

Once paired, the iPad is the **main interface**, not just a camera. The laptop
pushes a live telemetry snapshot (`output.dashboard.ipad_payload`) over the same
WebRTC control channel at ~1 Hz — the iPad cannot reach `webui/server.py`'s
`/data` endpoint because the hotspot AP isolates clients, so this is its only
feed. The page (`relay/static/ipad.html`) renders it two ways, switched from the
app bar and remembered per device:

- **Resident** — big live-vitals tiles (heart rate, breathing, SpO2 trend, mood)
  with reliability tiers and a calm one-line headline. A vital greys to
  "measuring…" once its `fresh_for` window lapses, so a stalled reading is never
  shown as current.
- **Provider** — a severity-ranked signal feed, fatigue/clothing/weather stats,
  and the connection diagnostics (WS/ICE/DC/FPS) that used to be the whole page.

The ⚙ settings drawer holds the view default, per-tile visibility, a
**presentation mode** (larger type, hides diagnostics and the preview for a
showcase), keep-awake, a **Run guided demo circuit** button, and the detector
pause/resume grid (honours `--no-ipad-toggle`). The telemetry stays a strict
subset of `to_payload` and is size-gated below the 64 KB control-channel ceiling
by `tests/ipad_payload_test.py` — grow it and that test fails rather than the
iPad silently going blank. Editing `relay/static/ipad.html` only reaches the
device once the relay serving it is restarted/redeployed (it is `no-store`, so a
reload then picks it up).

Two-way voice rides the same peer connection as RTP audio. When an iPad pairs,
its microphone (`--ipad-audio device`, the default) publishes onto the shared
16 kHz bus, so `--listen` transcription and cough detection hear the person at
the iPad instead of whatever the laptop's own mic picks up — no extra flag
beyond what those features already need. Piper speech routes to the iPad's
speaker instead of the laptop's; `--ipad-audio both` keeps the laptop audible
too, and `--ipad-audio laptop` opts out entirely (mic included). Only the Piper
engine produces PCM a remote speaker can carry: if TTS falls back to the system
voice (`pyttsx3`), speech stays on the laptop and the run prints a one-line
notice. Safari's echo cancellation keeps the agent's own voice out of the mic
feed, and turn-taking still mutes transcription while the agent speaks.
`IPadLink.status()` carries `audio_mic` / `audio_out` negotiation state plus
`audio_rx_blocks` / `audio_tx_chunks` counters for diagnostics.

Network: put both devices on a Windows Mobile Hotspot (share from Ethernet so
the laptop keeps internet for cloud features). ICE then settles on host
candidates on the same subnet and no TURN server is needed, which also makes
access-point client isolation irrelevant. Add an inbound UDP firewall rule for
the Python executable on the Private profile — without it ICE stalls at
`checking` forever, and that is the most common first-run failure.

Expect vitals to be **worse than the locked USB path**. iOS Safari exposes no
way to lock exposure or white balance, and `core/camera.py` calls auto-exposure
the single biggest accuracy killer for rPPG. Check `heart_rate_block_reason`
first when heart rate never starts: `no_face` usually means the iPad is too far
away for `showcase.min_face_px`. SpO2 on this path is uncalibrated and
indicative only — see the comments on `modules.spo2` in `config/modules.yaml`.

## Local web surfaces

The servers are independent and opt-in:

| Surface | Flag | Default address | Visibility |
|---|---|---|---|
| Companion and telemetry UI | `--webui` | `http://127.0.0.1:8770/` | local server; follow startup output |
| Private debug dashboard/state | `--debug-endpoint` | `http://127.0.0.1:8771/debug` | loopback only |
| Caregiver review portal | `--caregiver-portal` | `http://127.0.0.1:8772/caregiver` | loopback only |

The companion server also exposes `/data`, `/demo`, and `/modules` (the
detector roster and toggle console — see below). The private JSON health
payload is `/debug/state`. Use `--webui-port`, `--debug-port`, or
`--caregiver-port` when the default port is occupied.

Do not expose the debug or caregiver servers on a LAN. Their payloads are
designed to omit credentials, transcripts, raw frames/audio, binary values,
data URLs, and raw provider responses; preserve that redaction contract.

## Multi-person tracking and detector toggles

`--enable-multi-person` turns on short-lived anonymous tracking with a stable
primary subject. Secondary (non-primary) tracked people get their own
detector instances from a small, configurable set
(`tracking.secondary_modules` in `config/modules.yaml`) so their readings
never share state with the primary's. A secondary subject's fall or
unresponsive result still appears in the live dashboard and camera overlay,
but by deterministic rule in `alerts/` it never reaches a caregiver
notification channel — only the primary subject's alerts escalate.

Any currently-loaded detector can be paused or resumed at runtime from
`/modules`, per subject scope (primary/secondary). This never loads or
unloads a model — it only pauses a warm instance's cadence, so re-enabling a
heavy detector is instant — and it never writes to `config/modules.yaml`; a
restart always returns to the file's declared set. A module that
`config/modules.yaml` disabled at startup was never instantiated and cannot
be toggled on this way. Toggle mutations (not viewing) require a loopback
client by default; pass `--allow-remote-toggle` to permit them from the LAN.

`--start-blank` seeds the toggle state empty instead of mirroring
`config/modules.yaml`: every loaded detector starts paused, so a fresh run
shows only the camera feed and the person-outline boxes until you enable
things from `/modules`. This is a demo mode, not for unattended or
production use — no detector, and therefore no caregiver alert, runs until
it is manually toggled on.

## Voice, typed input, microphone, and audio events

Templated companion behavior works without a language-provider key. Relevant
run controls are:

- `--no-voice` disables spoken audio while retaining printed lines.
- `--tts {auto,piper,pyttsx3}` picks the spoken voice. `piper` is the local
  neural voice (offline; fetch a model once with
  `python -m audio.tts_piper --download`, stored in `assets/tts/`). `auto`
  prefers piper and falls back to the pyttsx3 system voice, then to printed
  lines, so a demo never depends on the venue network.
- `--showcase` starts the narrated tour on launch: a spoken introduction, the
  guided demo circuit (facial movement, arm drift, balance), and a closing
  wrap-up. Press `s` during a run instead.
- `--no-moondream` starts with Moondream calls disabled.
- `--type-input` accepts answers from the companion page without loading a
  microphone or speech-recognition model.
- `--listen` opts into microphone speech recognition and requires
  `requirements-asr.txt`.
- `--detect-cough` opts into local microphone audio-event detection and requires
  `requirements-audio-events.txt` for the optional model backend.

`--listen` and `--detect-cough` may share microphone infrastructure, but neither
is enabled by default. On Windows, Faster-Whisper runs in its established spawned
worker to isolate native runtime libraries; do not move that import into the
camera process.

## Cloud consent

Credentials alone never constitute consent. Each cloud camera path requires its
own per-run flag:

- `--enable-agent-vision` permits bounded conversational camera frames to the
  configured language provider while conversation is active.
- `--enable-cloud-skin` permits sampled stills for the configured skin-screening
  provider.
- `--enable-cloud-scene` separately permits infrequent room-scene stills.

Omit these flags for local-only operation. `--no-moondream` controls language
calls but is not a replacement for the separate upload consent boundaries.
Read [skin screening](SKIN_DETECTION.md) before evaluating that flow.

## Guided assessments and demos

Use `--assessment NAME` for one guided protocol or `--demo` for the short live
demo circuit. The supported names are authoritative in `main.py --help`.
Assessment output is non-diagnostic. Positioning failure, timeout, and skipped
steps must be represented honestly rather than narrated as successful capture.

The deterministic showcase and consent model are documented in
[MULTIMODAL_SHOWCASE.md](MULTIMODAL_SHOWCASE.md).

## Private health acceptance

For runtime-facing verification, add `--debug-endpoint` and inspect
`http://127.0.0.1:8771/debug/state` while the bounded process is alive. Read the
fields in this order:

1. `health.status`
2. `health.components`
3. `health.reasons` and `health.actions`
4. `performance`
5. credential-safe `system` lifecycle entries
6. serialized `results` relevant to the behavior under test

Require a timestamp newer than the process start so an orphaned server cannot
be mistaken for the current run. Full acceptance and failure recipes live in
[AI_DEVELOPMENT.md](AI_DEVELOPMENT.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| `python` is not found | Use `.venv\Scripts\python.exe` from the repository root. |
| Port 8770 refuses connections | Start with `--webui`; replay may already have ended. |
| Port 8771 refuses connections | Start with `--debug-endpoint` and wait for its startup marker. |
| Port is already occupied | Stop the prior bounded run/supervisor or choose the matching custom port flag. |
| Optional model is unavailable | Install only its requirement group and inspect credential-safe lifecycle status. |
| RealSense does not open | Confirm `pyrealsense2`, cable/device availability, and the hardware checklist. |
| Runtime is degraded | Follow `health.reasons` and `health.actions`, then inspect slow stages and queue pressure. |
| Cloud path remains disabled | Confirm both the specific per-run consent flag and its credential outside logs. |

Never leave `main.py`, `dev.py`, or a web server running after a development or
agent task. The hot-reload supervisor is a manual convenience, not the default
agent workflow.
