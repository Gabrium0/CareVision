## Runtime verification

For runtime, UI, configuration, integration, health, or performance changes,
verify with a **bounded, self-terminating** run so nothing outlives the task —
never leave a supervisor or app process running after you finish. Prefer a
capped headless replay that exits on its own after the frame budget:

```bash
python main.py --source replay:dev_hot_reload --headless --debug-endpoint --dev-mode --no-voice --no-moondream --max-frames 600
```

If you need a live `http://127.0.0.1:8771/debug/state` reading, start the
process, query it while it runs, then stop it in the same session. The `python
dev.py` hot-reload supervisor (see [docs/AI_DEVELOPMENT.md](docs/AI_DEVELOPMENT.md))
is an optional manual tool; if you start it, stop it (Ctrl+C) before ending the
session. Focused checks are sufficient for documentation-only and isolated
pure-function work. Never expose `.env` values, credentials, raw media, or
provider responses in logs.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
