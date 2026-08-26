# Documentation map

This directory separates operator guidance, coding-agent guidance, architecture,
and feature design so readers can load only the context needed for a task.

## Choose a path

### Run or troubleshoot CareVision

Start with [Operations](OPERATIONS.md) for environment setup, run modes, replay,
ports, consent flags, optional dependencies, and common failures. For a bounded
agent verification loop and the private health payload, continue with
[AI development workflow](AI_DEVELOPMENT.md).

### Change the code

Autonomous coding agents start at the repository [AGENTS.md](../AGENTS.md), then
read the nearest package-level `AGENTS.md`.

- [Architecture](ARCHITECTURE.md) defines data flow, shared contracts, package
  ownership, and threading constraints.
- [Extending the system](EXTENDING.md) gives recipes for detectors, model
  backends, caregiver channels, conversation topics, dashboard fields, and
  configuration.
- [AI development workflow](AI_DEVELOPMENT.md) defines bounded runtime
  verification and health acceptance criteria.

### Understand or demonstrate a feature

- [Multimodal showcase](MULTIMODAL_SHOWCASE.md) — consent flags, assessments,
  deterministic replay, demo surfaces, audio, routines, and optional sensors.
- [Intel RealSense D435i](REALSENSE_D435I.md) — depth/IMU capability and hardware
  validation.
- [NVIDIA-assisted skin detection](SKIN_DETECTION.md) — consent, local and cloud
  flow, result visibility, and failure behavior.
- [Detection research](RESEARCH.md) — methods, feasibility, limitations, and
  model reproduction.
- [rPPG and small-camera accuracy notes](../bpmupdate.md) contain specialized
  evaluation guidance; they do not override runtime configuration or tests.

## Sources of truth

Documentation explains intent, but executable contracts win when facts drift:

| Question | Authoritative source |
|---|---|
| Supported CLI flags and defaults | `main.py --help`, `dev.py --help` |
| Enabled detector configuration | `config/modules.yaml` and `core.registry` |
| Alert channels and policy | `config/alerts.yaml`, then `alerts/` tests |
| Replay scenario names and events | `config/replay_scenarios.json` |
| Web payload shape | `output/dashboard.py` and focused tests |
| Private debug payload | `webui/debug_server.py` and focused tests |
| Environment variable names | `.env.example` and the consuming code |

Generated files under `graphify-out/` help navigation and relationship queries;
they are not a substitute for current source, configuration, or tests.

## Documentation conventions

- Use repository-relative links and commands that work from the repository root.
- Show `.venv\Scripts\python.exe` for the known Windows setup and note that an
  activated environment may use `python`.
- Avoid volatile counts and copied flag catalogues. Link to the source of truth
  or derive values mechanically.
- Mark camera, microphone, network, credential, and paid-provider steps as
  explicit opt-ins.
- Keep health-facing language non-diagnostic and state verification limits.
- Put durable knowledge in the appropriate guide. Keep branch state, temporary
  failures, and uncommitted-file inventories in task summaries rather than an
  evergreen root handoff file.

## Maintenance check

After documentation changes, run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/documentation_test.py
.venv\Scripts\python.exe main.py --help
.venv\Scripts\python.exe dev.py --help
git diff --check
```

Documentation-only changes do not require a camera replay.
