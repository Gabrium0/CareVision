# Implementation Plan: LLM-steered topic selection + broadened corroboration coverage

> Generated inline. The `ccg-workflow` runtime (`codeagent-wrapper`, `~/.claude/.ccg/prompts/*`)
> is **not installed**, so no Codex/Gemini drafts were produced. Provision with
> `npx ccg-workflow` if you want the dual-model pass re-run.

## Task Type
- [x] Backend (agent/conversation logic)
- [ ] Frontend
- [ ] Fullstack

## Context (what already exists)
- `agent/corroboration.py` — `CorroborationEngine`: `observe()` flags low-confidence
  detector cues → `next_question(now)` returns the **oldest flagged** `(topic, rule)`
  deterministically → `mark_asked` / `hear` / `pending_conclusions`. 6 `FollowUpRule`s
  in `DEFAULT_RULES` (rash, pallor, dry_lips, sweating, cold_symptoms, expressivity).
- `safe_check_in()` airlock (added this session) already validates the **phrasing** of
  any LLM-generated check-in and falls back to `rule.question` on a clinical/accusatory
  line. `CorroborationEngine.funnel()` exposes the flagged→asked→answered counts on
  `/debug/state`.
- `agent/voice_agent.py:404` calls `next_question(now)`; builds an `Intent` whose
  `fallback` is the deterministic `rule.question`; the LLM (`MoondreamClient`) phrases it.
- `MoondreamClient.classify_answer()` is the existing one-word constrained-LLM call
  pattern to mirror for topic selection.

## Technical Solution
Two independent changes, both preserving the **"LLM proposes, deterministic disposes,
validator guarantees"** invariant:

### Part A — LLM steers *which* flagged topic to raise
The LLM chooses among **already-flagged, already-safe** topics; it can never invent a
topic or a question. The deterministic controller keeps every hard gate (one-at-a-time
cadence, cooldowns, `max_asks`, denial suppression). Selection is validated to be a member
of the offered set; anything else (hallucinated id, empty, offline) → deterministic
oldest-flagged fallback.

### Part B — broaden coverage
Add `FollowUpRule`s for uncovered low-confidence detectors (discomfort/pain, facial
puffiness, possible bruise, drowsiness/fatigue, restlessness), each grounded in the
detector's **verified** `(module, key, severity)` and phrased as a gentle, non-accusatory
check-in with a confirm-only conclusion. No engine changes — pure rule data + the airlock
already covers phrasing.

## Implementation Steps

### Part A: LLM topic steering
1. **`flagged_topics(now)` on `CorroborationEngine`** — return `[(topic, rule)]` currently
   in `flagged` state (respecting cooldowns), oldest-first. This is the neutral "ledger"
   the selector sees. *Deliverable: pure method + test.*
2. **`select_topic()` on `MoondreamClient`** (mirror `classify_answer`) — given a list of
   candidate topic ids + neutral one-line intents + recent conversation, return exactly one
   id or `none`. Constrained one-word-ish output; membership-validated by the caller.
   *Deliverable: method + offline no-op (returns None when unavailable).*
3. **`next_question_steered(now, selector)` on `CorroborationEngine`** — if `selector`
   yields a valid flagged topic id, return that `(topic, rule)`; else fall back to
   `next_question(now)` (oldest flagged). Selector output validated against
   `flagged_topics` ids. *Deliverable: method + tests (valid pick / invalid id → fallback /
   None → fallback).*
4. **Wire into `voice_agent.py:404`** — call `next_question_steered(now, self.moondream)`
   when `self.moondream` is available/enabled, else `next_question(now)`. Build the neutral
   candidate context from `flagged_topics` (topic → a non-clinical intent string, NOT the
   raw detector label). *Deliverable: one-line swap + context builder.*

**Safety invariant (must hold):** the selector receives only topic ids the engine already
flagged from real detector hits, described in neutral terms; its output is membership-checked;
phrasing still passes `safe_check_in`. The LLM reorders; it never creates.

### Part B: broaden coverage
1. **Verify detector signals** — for each candidate module (pain, facial_swelling, bruise,
   drowsiness, yawn, agitation), grep its `process()`/emit calls to record the exact
   `(module, key, severity, typical confidence)`. *Deliverable: a short verified table;
   no guessed keys.*
2. **Add `FollowUpRule`s to `DEFAULT_RULES`** using the verified keys, e.g.:
   - discomfort → "Are you feeling any aches or discomfort at the moment?"
   - facial puffiness → "Have you noticed any puffiness or swelling lately?"
   - possible bruise/mark → "Have you bumped or knocked yourself recently?"
   - tiredness → "You seem a little tired — did you manage to rest well?"
   - restlessness → "Feeling a bit restless? Anything on your mind?"
   Each with `max_confidence` (~0.6) and `min_severity=NOTICE`, matching existing rules,
   plus a supportive confirm-only `conclusion`. *Deliverable: rule entries.*
3. **Optional `voice_agent` support cues** — mirror the existing `cold_symptoms`/`hydration`
   `relevant`-cue enrichment (voice_agent.py:409-422) for the new topics if the matching
   `facial_cues()` keys exist. *Deliverable: optional phrasing hints.*

### Tests
- `flagged_topics` returns only flagged, cooldown-respecting, oldest-first.
- `next_question_steered`: valid selector pick honored; invalid/hallucinated id → deterministic
  fallback; `None`/offline → deterministic fallback.
- `select_topic` returns None when the client is unavailable (offline safety).
- Each new `FollowUpRule` flags on a synthetic low-confidence NOTICE snapshot for its verified
  key, and does **not** flag above `max_confidence`.
- `funnel()` reflects the new topics; new phrasings still pass `safe_check_in`.
- All offline — no network/LLM.

## Key Files
| File | Operation | Description |
|------|-----------|-------------|
| `agent/corroboration.py` | Modify | `flagged_topics()`, `next_question_steered()`; new `FollowUpRule`s in `DEFAULT_RULES` |
| `agent/moondream_client.py` | Modify | `select_topic()` constrained-choice call (mirror `classify_answer`) |
| `agent/voice_agent.py:404` | Modify | Use steered selection when LLM available; build neutral candidate context; optional support cues for new topics |
| `tests/corroboration_steering_test.py` | Create | Steering fallback + flagged_topics + new-rule flagging tests |
| `tests/corroboration_airlock_test.py` | Modify | Extend funnel/phrasing assertions to new topics |

## Risks and Mitigation
| Risk | Mitigation |
|------|------------|
| LLM invents a topic / picks unsafe | Selector output membership-checked against `flagged_topics`; invalid → deterministic fallback; phrasing still airlocked |
| Over-asking as coverage grows | Controller unchanged: one-at-a-time (`next_question`), `max_asks`, denial cooldown; new rules `NOTICE`+ and `max_confidence`-capped |
| Rule keys don't match real signals | Part B Step 1 verifies each `(module, key, severity)` before writing a rule — no guessed keys |
| Offline / no key regression | `next_question_steered` and `select_topic` both no-op to the deterministic path; fully offline-safe |
| Latency from an extra LLM call per ask | Selection reuses the existing async client; gate to when a check-in is already due, not per-frame |

## Complexity: Medium
Part A ~3–4 h (selector + steering + wiring + tests), Part B ~2 h (verify keys + rules + tests).
Engine core and airlock unchanged.

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (ccg-workflow runtime not installed)
- GEMINI_SESSION: N/A (ccg-workflow runtime not installed)
