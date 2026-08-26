## Working in `webui/`

- Query graphify first: `graphify query "<a question scoped to webui>"`.
- This package: stdlib local web surfaces for companion text, `/data` telemetry, `/demo`, private diagnostics, and caregiver review.
- Contracts it depends on:
  - `Aggregator.snapshot()` / `output/dashboard.py: to_payload()` — the payload `/data` serves
- Files: `server.py` (companion/data/demo server and thread-safe publish buses), `debug_server.py`, `caregiver_server.py`, their HTML/JS/CSS pages, and `_demo.py` for demo payload helpers.
- Common edit paths: `docs/EXTENDING.md` recipe 5's last step (add the new row's markup in `data.html` when a new dashboard field needs a web-side counterpart).
- Threading note: the server runs off the main video loop on a daemon thread — never add blocking work to a request handler.
- Read `docs/ARCHITECTURE.md`'s threading-model section and `docs/EXTENDING.md` recipe 5 only if graphify + this file don't answer the question.
