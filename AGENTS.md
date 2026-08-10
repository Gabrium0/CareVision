# Coding-agent guide

This is the canonical instruction file for autonomous coding agents working in
this repository. Package-level `AGENTS.md` files add narrower guidance for their
subtrees; they do not replace the safety and verification rules below.

## Start every task

1. Read the user request, `git status --short`, and the nearest applicable
   `AGENTS.md` files. Existing changes belong to the user unless proven
   otherwise; never clean, reset, overwrite, or reformat unrelated work.
2. Query the knowledge graph before broad source searches when
   `graphify-out/graph.json` exists:

   ```text
   graphify query "<focused question>"
   graphify path "<concept A>" "<concept B>"
   graphify explain "<concept>"
   ```

   If the native CLI is unavailable, use the installed Graphify skill's
   read-only helper. Dirty `graphify-out/` files are expected and are not a
   reason to skip the graph. Use `graphify-out/wiki/index.md` for broad
   navigation when it exists; read `GRAPH_REPORT.md` only when scoped graph
   queries are insufficient.
3. Inspect the source of truth for the behavior being changed. Do not infer
   runtime contracts from an old handoff, screenshot, generated graph report,
   or prose summary when code, configuration, or tests can answer the question.
4. Keep the change bounded. Preserve runtime APIs, schemas, privacy boundaries,
   and unrelated configuration unless the user explicitly asks to change them.

## Sources of truth

Use these in descending order when documentation and implementation disagree:

1. Executable contracts and tests: types, public methods, CLI parsers, payload
   serializers, and focused tests.
2. Runtime configuration: `config/modules.yaml`, `config/alerts.yaml`, and
   `config/replay_scenarios.json`.
3. Package-level `AGENTS.md` files and [architecture](docs/ARCHITECTURE.md).
4. Feature and operations guides linked from [docs/README.md](docs/README.md).
5. Generated Graphify reports and historical planning material.

For command-line flags, `main.py --help` and `dev.py --help` are authoritative.
For registered detectors, query `core.registry`; do not hard-code a module count
in evergreen documentation or UI copy unless a test derives it from the
registry.

## Repository routing

| Change | Start with | Then read |
|---|---|---|
| Runtime spine, event/context schema, scheduler | `core/AGENTS.md` | `docs/ARCHITECTURE.md` |
| Detector or backend | `modules/AGENTS.md` | `docs/EXTENDING.md` recipes 1–2 |
| Conversation or voice behavior | `agent/AGENTS.md` | `docs/EXTENDING.md` recipe 4 |
| Caregiver alert delivery | `alerts/AGENTS.md` | `docs/EXTENDING.md` recipe 3 |
| Aggregation or dashboard payload | `output/AGENTS.md` | `docs/EXTENDING.md` recipe 5 |
| Web routes or pages | `webui/AGENTS.md` | architecture threading section |
| Setup, run mode, or diagnostics | `docs/OPERATIONS.md` | `docs/AI_DEVELOPMENT.md` |

The alert path in `alerts/` is deterministic and must never depend on generated
language. The conversational agent in `agent/` may phrase non-diagnostic
check-ins but does not own urgent escalation.

## Implementation rules

- The main video loop must not block. Heavy inference, network calls, database
  reads, and speech run through the existing bounded worker patterns.
- Optional dependencies and providers degrade gracefully. Import lazily where
  the package already does so, expose credential-safe lifecycle state, and keep
  offline paths functional.
- Secrets come from `.env` or the environment, never YAML or source. Never log
  `.env` values, credentials, raw media, data URLs, biometric material, or raw
  provider responses.
- Camera, microphone, and cloud upload paths remain explicitly opt-in. Do not
  weaken loopback binding or consent checks for convenience.
- Health-facing text is observational and non-diagnostic. Images or model
  hypotheses alone must not become caregiver alerts.
- Add or update focused tests with behavior changes. Keep public symbols within
  the docstring policy configured in `pyproject.toml`.
- Use `graphify update .` after code changes so the knowledge graph remains
  current. Documentation-only edits do not require rewriting already-dirty
  Graphify artifacts.

## Verification

Choose checks in proportion to the change:

- Documentation only: documentation tests, CLI help for referenced commands,
  link/path validation, and `git diff --check`.
- Isolated pure function: focused unit tests.
- Runtime, UI, configuration, integration, health, or performance: focused
  tests plus a bounded, self-terminating runtime check.
- Hardware or paid provider: run only when explicitly authorized and report
  what could not be verified otherwise.

Use the checked-in interpreter on Windows:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/<focused_test>.py
```

For runtime-facing changes, prefer this camera-free run:

```powershell
.venv\Scripts\python.exe main.py --source replay:dev_hot_reload --headless --debug-endpoint --dev-mode --no-voice --no-moondream --max-frames 600
```

The process must exit within its frame budget. If live debug state is needed,
query `http://127.0.0.1:8771/debug/state` while that process runs and stop it in
the same session. Never leave `main.py`, `dev.py`, a supervisor, or a web server
running after the task. See [AI_DEVELOPMENT.md](docs/AI_DEVELOPMENT.md) for the
acceptance fields and failure recipes.

## Handoff standard

Before finishing, inspect the diff, confirm unrelated user changes remain, and
state exactly what was verified. Distinguish passing automated checks from live,
hardware, or provider behavior that was not exercised. Do not create root-level
status or handoff files whose branch and worktree claims will become stale;
record durable knowledge in the relevant guide and transient state in the task
summary.
