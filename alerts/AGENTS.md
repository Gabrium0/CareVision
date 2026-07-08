## Working in `alerts/`

- Query graphify first: `graphify query "<a question scoped to alerts>"`.
- This package: deterministic caregiver notifications. Safety principle: the alert path never depends on the LLM — it must stay predictable.
- Contracts it depends on:
  - `Result`, `Severity` — `core/events.py` (anything the alert path acts on arrives as one of these)
- Files: `notifier.py` (channel implementations: email/SMS/webhook/console), `manager.py` (confirm/dedupe/escalate/quiet-hours logic).
- Common edit paths: `docs/EXTENDING.md` recipe 3 (add a caregiver alert channel) — copy an existing `Channel` class in `notifier.py`, register it in `_REGISTRY`, list it under `channels:` in `config/alerts.yaml`. Missing creds should self-disable the channel gracefully; `console` always stays enabled so alerts are never silently lost.
- Read `docs/ARCHITECTURE.md`'s safety principle and `docs/EXTENDING.md` recipe 3 only if graphify + this file don't answer the question.
