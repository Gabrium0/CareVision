# CareVision elderly-care detection pipeline

CareVision is a modular computer-vision pipeline for an elderly person living
alone. It combines live or replayed camera input with non-diagnostic health and
safety observations, a spoken companion, deterministic caregiver alerts, and
local dashboards.

> [!CAUTION]
> CareVision is not a medical device. Its health signals are screening prompts,
> never diagnoses. Urgent caregiver alerts follow deterministic rules in
> `alerts/`; they never depend on an LLM or VLM. With `--enable-multi-person`,
> only the primary subject's alerts escalate — a secondary tracked person's
> fall or unresponsive result stays visible in the dashboard but never pages
> a caregiver, since the system makes no identity claim about a visitor.

## Quick start

The checked-in virtual environment is the known development setup. On Windows,
use its interpreter explicitly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py --source replay:dev_hot_reload --headless --dev-mode --no-voice --no-moondream --max-frames 600
```

On systems where the project environment is already active, replace
`.venv\Scripts\python.exe` with `python`.

The replay command is hardware-free, makes no cloud or microphone requests,
and exits after its frame budget. For an interactive local dashboard:

```powershell
.venv\Scripts\python.exe main.py --source replay:client_demo --webui --no-moondream
```

Open the URL printed by the application. Replay runs exit when their scripted
scenario ends.

## Common run modes

| Goal | Command or option |
|---|---|
| Default webcam | `python main.py` |
| Video file | `python main.py --source clip.mp4` |
| Deterministic replay | `python main.py --source replay:SCENARIO --headless` |
| Intel RealSense | `python main.py --source realsense` |
| iPad as camera + control surface | `python main.py --source ipad` — see [iPad demo run](#ipad-demo-run) below |
| Companion and telemetry web UI | add `--webui` |
| Multi-person tracking + per-detector toggles | add `--enable-multi-person` (toggle console at `/modules` with `--webui`) |
| Live demo starting with everything paused | add `--start-blank` (camera + person outline only, until you enable detectors from `/modules`) |
| Private localhost diagnostics | add `--debug-endpoint` |
| Caregiver review portal | add `--caregiver-portal` |
| Disable spoken audio | add `--no-voice` |
| Natural neural voice (offline) | add `--tts piper` (fetch once: `python -m audio.tts_piper --download`) |
| Narrated showcase tour | add `--showcase` (or press `s`; guided face/arm/balance checks with intro and wrap-up) |
| Disable Moondream calls | add `--no-moondream` |

Cloud camera uploads and microphone processing are separately opt-in. See the
[operations guide](docs/OPERATIONS.md) before enabling them.

## iPad demo run

This is the full showcase invocation — iPad camera and control surface, web
dashboard, multi-person tracking, spoken listening, and local diagnostics:

```powershell
python main.py --source ipad --webui --enable-multi-person --ipad-relay-url https://mercury-upper-processes-seemed.trycloudflare.com --ipad-room an6d-bvdd-rxyr --groq-stt --debug-endpoint
```

What each flag does:

| Flag | Purpose |
|---|---|
| `--source ipad` | Take video from the iPad's browser over WebRTC; that page also becomes the control surface |
| `--webui` | Companion + telemetry dashboard at `http://127.0.0.1:8770/` (detector toggles at `/modules`) |
| `--enable-multi-person` | Track a secondary person alongside the primary subject |
| `--ipad-relay-url` | HTTPS signaling relay for this run; overrides `IPAD_RELAY_URL` from `.env`. Quick-tunnel `trycloudflare.com` URLs change every time `cloudflared` restarts, so pass whatever URL it currently prints |
| `--ipad-room` | Relay room name; must match the `/r/<room>` URL opened on the iPad |
| `--groq-stt` | Opt in to fast Groq Whisper transcription (`GROQ_API_KEY`); implies listening and automatically uses only the paired iPad microphone, with local Whisper fallback |
| `--debug-endpoint` | Private loopback-only diagnostics at `http://127.0.0.1:8771/debug` |

Then on the iPad, open the pairing page — for the example above,
`https://mercury-upper-processes-seemed.trycloudflare.com/r/an6d-bvdd-rxyr` —
and type the 6-digit code the app printed at startup. **Start the laptop
first**: the relay may need ~60 s to wake from a cold start, and whoever
connects first waits for it.

### Why a Windows Mobile Hotspot is required

The relay only carries signaling (SDP/ICE). The actual video flows **directly
between the iPad and the laptop** over WebRTC, which only works when the two
devices can reach each other on the network. Ordinary home, office, or campus
Wi-Fi isolates wireless clients from each other (AP isolation) or splits them
across subnets, so connection setup stalls at `checking` or demands a TURN
server. A Windows Mobile Hotspot solves this in one move: both devices land on
a single `192.168.137.x` subnet with client-to-client traffic allowed, so ICE
settles immediately with no TURN server.

Setup:

1. Open **Settings → Network & internet → Mobile hotspot** and turn it **On**.
2. Set **Share my internet connection from** to **Ethernet** — the laptop keeps
   its wired internet, so cloud features (VLM calls, tunnel, relay) still work.
3. Connect the iPad to the hotspot's Wi-Fi name and password shown on that page.
4. From an **Administrator** PowerShell, allow inbound WebRTC traffic for
   Python on the Private profile — without this rule ICE stalls at `checking`
   forever, which is the most common first-run failure:

   ```powershell
   New-NetFirewallRule -DisplayName "CareVision WebRTC" -Direction Inbound `
     -Protocol UDP -Program (Get-Command python).Source -Profile Private -Action Allow
   ```

Run the app and `relay/server.py` with **system Python**, not `.venv` —
`requirements-ipad.txt` explains why. The one-command launcher
(`pwsh -File scripts\start-showcase.ps1`) automates relay + tunnel + app + QR,
and the [operations guide](docs/OPERATIONS.md#ipad-as-the-camera-and-main-interface)
covers manual relay/tunnel setup and troubleshooting.

## Optional capabilities

Install only the requirement groups needed for a run:

| Capability | Requirements |
|---|---|
| Voice and caregiver integrations | `requirements-agent.txt` |
| Speech recognition | `requirements-asr.txt` |
| Audio-event detection | `requirements-audio-events.txt` |
| Intel RealSense | `requirements-realsense.txt` |
| iPad camera over WebRTC | `requirements-ipad.txt` (downgrades `av`; read the file first) |
| Local skin model | `requirements-skin.txt` |
| Clothing models | `requirements-clothing.txt` |
| Neural rPPG | `requirements-openrppg.txt` |
| Simulated or external sensors | `requirements-sensors.txt` |

Copy `.env.example` to `.env` only when an optional integration needs a secret.
Never commit `.env` or expose its contents in logs, diagnostics, screenshots, or
task summaries.

## Documentation

- [Documentation map](docs/README.md) — start here and choose the guide for the
  task.
- [Operations](docs/OPERATIONS.md) — setup, run modes, ports, opt-ins, and
  troubleshooting.
- [Architecture](docs/ARCHITECTURE.md) — data flow, contracts, threading, and
  package ownership.
- [Extending the system](docs/EXTENDING.md) — recipes for detectors, agent
  topics, alerts, configuration, and dashboards.
- [AI development workflow](docs/AI_DEVELOPMENT.md) — bounded verification and
  private runtime health checks.
- [Multimodal showcase](docs/MULTIMODAL_SHOWCASE.md),
  [RealSense](docs/REALSENSE_D435I.md), and
  [skin screening](docs/SKIN_DETECTION.md) — feature-specific behavior.

Autonomous coding agents must begin with [AGENTS.md](AGENTS.md). More specific
`AGENTS.md` files inside packages refine those instructions for their subtree.

## Development checks

Use focused tests for the area changed. The camera-free pipeline smoke check is:

```powershell
.venv\Scripts\python.exe tests/smoke_test.py
```

Developer tooling is installed from `requirements-dev.txt`. Public Python
symbols are held to the docstring policy in `pyproject.toml`.

## Privacy defaults

The default pipeline stores numeric summaries rather than raw video. Features
that may send an in-memory camera still to a configured cloud provider require
an explicit per-run consent flag. Debug endpoints bind to loopback and redact
credentials, raw media, data URLs, and provider responses. Preserve those
boundaries when extending the system.
