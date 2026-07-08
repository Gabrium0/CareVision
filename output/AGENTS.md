## Working in `output/`

- Query graphify first: `graphify query "<a question scoped to output>"`.
- This package: aggregation and presentation — turns raw per-module `Result`s into the single "person state" and renders it (window + web).
- Contracts it depends on:
  - `Result`, `Severity` — `core/events.py`
  - `Aggregator` — `output/aggregator.py` (keeps latest non-expired `Result` per `(module, key)`, smooths numeric values with a rolling median; `snapshot()` is what `agent/`, `alerts/`, and the dashboard all read)
- Files: `dashboard.py` (single source of truth for both the on-screen window and the web `/data` payload via `to_payload()`), `overlay.py`, legacy `greeting_engine.py`.
- Common edit paths: `docs/EXTENDING.md` recipe 5 (add a dashboard / `/data` field). Per-metric backend comparisons are picked up automatically by `_parse_comparisons`; stat-grid entries need a `(module, key, "Label")` tuple added to `_FATIGUE_STATS` or `_CLOTHING_WEATHER_STATS`. `render()` and `to_payload()` share these lists — keep window and web in sync.
- Read `docs/EXTENDING.md` recipe 5 only if graphify + this file don't answer the question.
