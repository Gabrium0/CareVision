## Working in `core/`

- Query graphify first: `graphify query "<a question scoped to core>"`.
- This package: camera capture, `FrameContext`, `Result`/`Severity`, the module registry, scheduler, and per-frame pipeline — the runtime spine every other package depends on.
- Contracts defined here (the god nodes — changes are cross-cutting, affect every module):
  - `Result`, `Severity` — `core/events.py`
  - `FrameContext`, `FaceData`, `PoseData` — `core/context.py`
  - `@register`, `discover`, `build_enabled` — `core/registry.py`
  - `Scheduler` — `core/scheduler.py`
  - `Pipeline.process_frame` — `core/pipeline.py`
- Common edit paths: adding a new field to `FrameContext.extras`, changing `Result`'s shape, or adjusting scheduler timing logic. There is no EXTENDING.md recipe for editing `core/` itself — treat any change here as needing a check of every consumer (`modules/`, `output/`, `agent/`, `alerts/`).
- Read `docs/ARCHITECTURE.md`'s "Core abstractions" table only if graphify + this file don't answer the question — don't load the whole repo for a change scoped to `core/`.
