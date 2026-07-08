## Working in `agent/`

- Query graphify first: `graphify query "<a question scoped to agent>"`.
- This package: the Gemini voice companion — decides what to say and speaks it. Never handles caregiver alerts (that's `alerts/`, deterministic, no LLM).
- Contracts it depends on:
  - `Aggregator.snapshot()` — `output/aggregator.py` (the "person state" it reads)
  - `Result`, `Severity` — `core/events.py`
  - `ObservationMemory` — `agent/state.py`
- Files: `policy.py` (`Policy._candidates` picks conversation topics), `advisor_engine.py` (post-aggregation rules that emit `Result`s back into the aggregator), `gemini_client.py`, `voice_agent.py`, `env.py`.
- Common edit paths: `docs/EXTENDING.md` recipe 4 (add a voice-agent conversation topic). Severity.ALERT items are **not** handled here — they flow through `alerts/` instead.
- Read `docs/ARCHITECTURE.md`'s safety principle note and `docs/EXTENDING.md` recipe 4 only if graphify + this file don't answer the question.
