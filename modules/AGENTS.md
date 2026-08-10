## Working in `modules/`

- Query graphify first: `graphify query "<a question scoped to modules>"`.
- This package: registry-discovered detectors for vitals, face/skin, neuro/motor, fatigue, falls/safety, behavioral, and demographic signals. The flat filesystem is not an authoritative catalogue; registration and `config/modules.yaml` determine the current set.
- Contracts it depends on:
  - `DetectionModule` — `modules/base.py` (declares `interval`, `requires`, implements `process(ctx)`)
  - `Result`, `Severity` — `core/events.py`
  - `register`, `discover` — `core/registry.py`
  - `Backend` — `modules/backends/base.py` (multi-backend "show both" pattern)
  - Shared signal helpers (`TimedBuffer`, `bandpass`, FFT) — `modules/_util.py`
- Common edit paths: `docs/EXTENDING.md` recipe 1 (add a detection module) and recipe 2 (add a backend). Multi-backend detectors have their implementations under `modules/backends/`, `modules/rppg_backends/`, `modules/emotion_backends/`.
- New modules are auto-discovered via `pkgutil.walk_packages` — no registry changes needed, just add the file and list it in `config/modules.yaml`.
- Read `docs/EXTENDING.md` recipes 1-2 only if graphify + this file don't answer the question — don't load the whole repo for a change scoped to one detector.
