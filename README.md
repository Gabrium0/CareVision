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
| iPad as camera + control surface | `python main.py --source ipad` (needs the `relay/` HTTPS service; see [operations guide](docs/OPERATIONS.md)) |
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
