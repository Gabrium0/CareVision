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

## Local web surfaces

The servers are independent and opt-in:

| Surface | Flag | Default address | Visibility |
|---|---|---|---|
| Companion and telemetry UI | `--webui` | `http://127.0.0.1:8770/` | local server; follow startup output |
| Private debug dashboard/state | `--debug-endpoint` | `http://127.0.0.1:8771/debug` | loopback only |
| Caregiver review portal | `--caregiver-portal` | `http://127.0.0.1:8772/caregiver` | loopback only |

The companion server also exposes `/data` and `/demo`. The private JSON health
payload is `/debug/state`. Use `--webui-port`, `--debug-port`, or
`--caregiver-port` when the default port is occupied.

Do not expose the debug or caregiver servers on a LAN. Their payloads are
designed to omit credentials, transcripts, raw frames/audio, binary values,
data URLs, and raw provider responses; preserve that redaction contract.

## Voice, typed input, microphone, and audio events

Templated companion behavior works without a language-provider key. Relevant
run controls are:

- `--no-voice` disables spoken audio while retaining printed lines.
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
