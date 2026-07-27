# AI development workflow

This guide describes how to test live application behavior. For runtime, UI,
configuration, integration, health, and performance changes, an agent should
prefer a **bounded, self-terminating run** (see "Bounded verification" below)
over the long-lived supervisor. Documentation-only changes and isolated
pure-function work may use focused checks instead.

> **Mandatory cleanup.** Whatever you start, stop before the task ends. Never
> leave a `python dev.py` supervisor or an app process running past your
> session — an orphaned run keeps holding the debug/web ports and can shadow a
> later live run with stale data. Run the app in the foreground with a frame
> budget, or stop any backgrounded process explicitly (Ctrl+C / kill) when done.

## Bounded verification (preferred for agents)

Run a capped headless replay that exits on its own after the frame budget, so
nothing can outlive the task:

```bash
python main.py --source replay:dev_hot_reload --headless --debug-endpoint --dev-mode --no-voice --no-moondream --max-frames 600
```

`--max-frames` (wired through `main.py` to `pipeline.run`) stops the run after N
frames; at the fixture's ~12 fps, 600 frames is ~50 s — long enough to reach
steady state and query `http://127.0.0.1:8771/debug/state` while it runs. If you
need that live reading, start the process, query it, and let it exit (or stop it
in the same session).

## Stable development interfaces

- Supervisor CLI: `python dev.py [--watch PATH] [--interval SECONDS]
  [--debounce SECONDS] [-- COMMAND...]`
- Supervisor and child logs: inherited terminal output with `[reload]`, `[main]`,
  and `[debug]` prefixes
- Machine-readable health: `GET http://127.0.0.1:8771/debug/state`
- Readable private dashboard: <http://127.0.0.1:8771/debug>

The default command is deliberately safe for unattended development. It runs a
long deterministic replay headlessly, binds diagnostics to loopback, disables
voice and Moondream requests, and avoids heavyweight/network-backed optional
workers. It does not require a camera. The project root is watched by default;
polling defaults to 0.25 seconds and restart debounce to 0.2 seconds. Repeat
`--watch PATH` to narrow monitoring to selected files or trees.

## Optional: the hot-reload supervisor (manual use)

`python dev.py` is an optional convenience for a human developer iterating
locally — it restarts the app on file changes. It is **not** the agent default;
an agent that starts it must stop it (Ctrl+C) before the session ends (see
"Mandatory cleanup" above). Use the active project interpreter:

```powershell
# Windows virtual environment
.venv\Scripts\python.exe dev.py

# Or when python already resolves to the project environment
python dev.py
```

A healthy startup reaches these markers in order:

```text
[reload] watching ...
[reload] starting: ... main.py ...
[debug] private JSON state: http://127.0.0.1:8771/debug/state
[main] starting; ...
```

The supervisor sets unbuffered Python output, so its own messages and all child
logs appear in the same terminal. Do not start a second default supervisor on
the same debug port, and stop it (Ctrl+C) when you are done.

After saving a watched Python, YAML, JSON, HTML, CSS, or JavaScript file, wait for
both markers before testing the new process:

```text
[reload] change detected: path\to\file.py
[reload] starting: ...
```

Then wait for the new child's `[debug]` and `[main]` startup markers. A message
such as `[reload] child exited with code 1; waiting for a file change` means the
child crashed or completed; the supervisor is still alive and will retry after
the next watched edit. Read the traceback or preceding child logs before making
another change.

## Use debug state as the live acceptance gate

Query the private endpoint only after the run's `[main] starting` marker (and,
when using the supervisor, the post-reload markers). Record the start time and
require the returned `timestamp` to be newer so an old or orphaned process, or a
cached response, cannot be mistaken for the current build.

PowerShell:

```powershell
$state = Invoke-RestMethod -Uri 'http://127.0.0.1:8771/debug/state' -TimeoutSec 5
$state.health | ConvertTo-Json -Depth 6
```

Portable shell:

```bash
curl --fail --silent --show-error http://127.0.0.1:8771/debug/state
```

Evaluate fields in this order:

1. `health.status` is the overall result: `healthy`, `degraded`, or `failed`.
2. `health.components` identifies the subsystem responsible.
3. `health.reasons` provides stable reason codes; `health.actions` provides the
   corresponding next diagnostic step.
4. `performance` contains rates, latency distributions, scheduler state, queue
   pressure, and slow stages for runtime problems.
5. `system` contains credential-safe component lifecycle diagnostics.
6. `results` contains the latest serialized detector outputs for behavioral
   assertions.

An optional component reported as `unconfigured` is an intentional opt-out, not
a failure. Do not relabel or ignore `degraded` or `failed`; investigate its reason
and component state, reload, and query again. Do not claim live verification when
the endpoint is unreachable, the payload predates the reload, or the relevant
component never reached its expected lifecycle state.

## Run focused tests alongside the app

Use a second terminal for one-off tests so the application supervisor keeps its
logs and debug endpoint available:

```bash
python -m pytest -q tests/runtime_performance_test.py
python -m pytest -q tests/dev_reload_test.py
```

To rerun a focused test command after every watched edit instead of running the
application, pass the command after `--`:

```bash
python dev.py -- python -m pytest -q tests/runtime_performance_test.py
python dev.py --watch core --watch tests -- python -m pytest -q tests/pipeline_staleness_test.py
```

A test child that exits remains idle until a watched file changes. Test success
does not replace live `/debug/state` verification when the requested behavior is
runtime-facing.

## Failure recipes

| Symptom | Action |
|---|---|
| `/debug/state` is unavailable | Wait for the child's `[debug]` marker. If it never appears, inspect startup logs and the child exit code. |
| Payload appears stale | Confirm a post-edit `[reload] starting`, wait for the new `[main] starting`, and require `timestamp` to be later than that restart. |
| Port 8771 is already in use | Stop the older supervisor, or use a custom child command with `--debug-port PORT` and query that port. |
| Child exits or loops on reload | Read the first traceback after `[reload] starting`; fix that root error before treating later failures as independent. |
| `runtime` is degraded | Follow `health.reasons` and `health.actions`, then inspect `performance.slow_stages_ms`, latency distributions, sampler rates, and queue-drop fields. |
| A provider is degraded | Inspect only its credential-safe entry under `system` for status, HTTP class, retryability, and circuit state; verify configuration outside logs when necessary. |

For a real camera or an explicitly authorized cloud test, replace the default
child command. These paths can use hardware, credentials, network access, and
money, so keep them opt-in:

```bash
python dev.py -- python main.py --source 0 --headless --debug-endpoint --no-moondream
python dev.py -- python main.py --source replay:dev_hot_reload --headless --debug-endpoint --dev-mode --no-voice
```

Stop the supervisor cleanly with Ctrl+C when verification is complete.

## Privacy rules

The debug server is private because it binds to `127.0.0.1`; do not change an
agent workflow to expose it on a LAN. Never print or copy `.env` values,
credentials, raw frames/audio, data URLs, or raw provider responses into logs,
tests, debug JSON, or task summaries. Diagnose integrations from redacted status,
reason codes, HTTP status classes, retry state, and bounded metadata only.
