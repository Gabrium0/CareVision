## Working in `webui/`

- Query graphify first: `graphify query "<a question scoped to webui>"`.
- This package: stdlib web server exposing the companion UI and telemetry — `/` companion text, `/data` full telemetry via SSE.
- Contracts it depends on:
  - `Aggregator.snapshot()` / `output/dashboard.py: to_payload()` — the payload `/data` serves
- Files: `server.py` (`ThreadingHTTPServer` on a daemon thread; `publish`/`publish_data` push to thread-safe buses read by SSE clients), `page.html` (companion view), `data.html` (telemetry view), `_demo.py`.
- Common edit paths: `docs/EXTENDING.md` recipe 5's last step (add the new row's markup in `data.html` when a new dashboard field needs a web-side counterpart).
- Threading note: the server runs off the main video loop on a daemon thread — never add blocking work to a request handler.
- Read `docs/ARCHITECTURE.md`'s threading-model section and `docs/EXTENDING.md` recipe 5 only if graphify + this file don't answer the question.
