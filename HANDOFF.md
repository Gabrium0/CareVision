# Handoff: guest-facing demo legibility work

**Date:** 2026-07-22
**Branch:** `showcase` (HEAD `dd90183` "Showcase")
**Status:** all changes are **uncommitted** in the working tree. Nothing was pushed.

Goal that drove this work: *make the running pipeline legible and impressive to
someone watching, in under two minutes, without a presenter having to explain
JSON.*

---

## ▶ START HERE — next task

**§4.1 (arm-drift silent failure) is DONE** — see §4.1 below for what shipped.
The next highest-value leftover is **§4.2 (Vitals / Skin beats never tick on
`client_demo`)**.

To ask a fresh agent to continue, say something like:

> Read HANDOFF.md and start on the §4.2 vitals/skin beats task.

Full leftover list is in **§4**, ordered by value. Everything shipped so far is
in §2; what is proven vs unproven is in §3.

---

## 1. Current working-tree state

| File | State | What changed |
|---|---|---|
| `output/dashboard.py` | modified | module labels, `INTERNAL_MODULES`, `GUEST_CONFIDENCE_FLOOR`, payload additions |
| `webui/demo.html` | modified | rewritten guest view (roster, stat line, event stream, beats, moment card) |
| `webui/_http.py` | **new** | `QuietThreadingHTTPServer` |
| `webui/server.py` | modified | quiet base + SSE teardown |
| `webui/debug_server.py` | modified | quiet base |
| `webui/caregiver_server.py` | modified | quiet base |
| `main.py` | modified | one line: `modules_enabled` in `system_snapshot()` |
| `tests/demo_labels_test.py` | **new** | label-coverage / anti-jargon gate |
| `tests/http_quiet_test.py` | **new** | disconnect-silencing tests |
| `tests/dashboard_signals_test.py` | modified | payload contract + promotion tests |
| `README.md`, `docs/MULTIMODAL_SHOWCASE.md` | modified | demo-mode docs |
| `.claude/launch.json` | modified | added `client-demo` entry |

The pre-rewrite `webui/demo.html` is recoverable from commit `dd90183`.

---

## 2. What was built

### Payload contract (`output/dashboard.py :: to_payload()`)

Each entry of `payload["signals"]` now carries, in addition to the pre-existing
fields:

```
label     str   guest-facing module name    ("Facial symmetry")
blurb     str   one-line description        ("Watches for one-sided droop")
internal  bool  True for plumbing modules   (showcase, replay_events)
promote   bool  may become headline/moment  (see confidence floor below)
```

Two new top-level keys:

```
modules        [{module, label, blurb, running}, ...]   all 49 registered, sorted by label
module_counts  {registered: 49, running: 46}
```

`running` is derived from `system["modules_enabled"]`, which `main.py` supplies
from `pipeline.scheduler.modules`.

### Guest view (`webui/demo.html`, served at `/demo`)

Self-contained vanilla JS, no build step, no external requests. Renders: a stat
line (`49 detectors · 46 running · one camera · 30 fps`), the full detector
roster (dim = running-but-quiet, lit = fired within 4s), an append-only event
stream driven by `system.timeline` and deduped on entry `id`, confidence as
*high / moderate / tentative* (never a raw float), a replay progress bar with
countdown, a beat checklist of signal families, and a moment card that holds
notable events for ~6s.

### Confidence floor

`GUEST_CONFIDENCE_FLOOR = 0.45` in `output/dashboard.py`. A signal may become the
headline or a moment card only if severity is `warning`/`alert` (safety-critical
always shows, at any confidence) **or** confidence >= 0.45.

This exists because a real run put this on the TV as the largest text on screen:

```
masked_face  confidence 0.277  "Reduced facial expressiveness (flat affect / masking — screening)"
grooming     confidence 0.157  "Hair looks more textured/unkempt than the usual weekly average"
```

`notice` outranks `info` in the severity sort, so a 28%-confidence clinical-
sounding guess about an identifiable person beat every confident observation.

**It is a promotion filter, not a suppression filter.** Low-confidence results
still appear in `payload["signals"]`, the caregiver `/data` view, and the `/demo`
scrolling feed. Do not "simplify" this into a filter that drops them.

### Quiet servers

`webui/_http.py` defines `QuietThreadingHTTPServer`, whose `handle_error()`
silently returns for `ConnectionAbortedError` / `ConnectionResetError` /
`BrokenPipeError` and defers to `super()` otherwise. All three servers use it.

Benign browser disconnects were dumping a full traceback into the terminal
mid-demo. **Catch only those three classes** — never bare `OSError`. There is a
test asserting a `ValueError` still surfaces; keep it.

---

## 3. Verification status — read before trusting anything

### Verified

- `pytest tests/dashboard_signals_test.py tests/demo_labels_test.py -q` → 12 passed
- `pytest tests/http_quiet_test.py -q` → 4 passed
- `python tests/smoke_test.py` → end-to-end OK (49 registered, 46 built)
- Live `replay:client_demo` run: payload contract intact, `promote` present on
  all signals, headline resolves to a real observation, roster shows zero
  developer jargon, a real SSE disconnect produced no traceback

### NOT verified

- **The entire RealSense path.** No camera was available. Everything above was
  verified against `replay:client_demo` only.
- **The original masked_face/grooming case.** The floor is proven on synthetic
  data; the real 0.277-confidence result comes from live hardware. Confirm with:
  ```
  python main.py --webui --demo --source realsense
  ```
  The headline should no longer read "Reduced facial expressiveness…".

---

## 4. Open issues, highest value first

### 4.1 Demo circuit's arm-drift step fails silently  ✅ DONE (2026-07-23)

**Fixed.** The demo circuit no longer narrates a failed step as a success. A step
that ends in `CANCELLED`/`TIMED_OUT` (i.e. never captured) is now: (1) announced
honestly with an actionable reposition instruction and the task restated in one
spoken line, then retried once at the circuit level; (2) if it fails again, an
honest skip is announced ("Skipping the arm check — I couldn't capture it this
time.") before moving on.

Changes:
- `core/workflows.py` — added `WorkflowEngine.get(correlation_id)` so the circuit
  can read the terminal stage of a workflow that already left `_active_by_subject`.
- `agent/voice_agent.py` — `_advance_demo_circuit()` now tracks the running step
  (`_demo_active_cid`/`_demo_active_protocol`), enforces a per-step retry budget
  (`_DEMO_MAX_RETRIES = 1`, on top of the positioner's own internal reposition
  retry), and speaks via `_say_demo()`/`_start_demo_step()` so `last_utterance`
  reflects circuit narration. Guest-safe labels/guidance in `_DEMO_STEP_LABELS`
  and `_DEMO_STEP_GUIDANCE` (framing only — no clinical claims, per §7).
- `tests/demo_circuit_test.py` — added cases for retry-with-guidance, skip after
  budget exhausted, concluded-step-no-skip, timed-out announcement, and failure
  on the final step. 10 passed; `assessment_library`, `dashboard_signals`,
  `demo_labels`, `http_quiet`, `phase_infrastructure`, `multimodal_expansion` and
  `tests/smoke_test.py` all still green.

**Still unverified:** the live RealSense path (no camera here). The fix is proven
on unit tests that drive `cancel()`/`tick()` deterministically; confirm on
hardware that a real positioning give-up now triggers the retry/skip narration
rather than a silent advance. The `_advance_demo_circuit()` outcome branch now
distinguishes `CONCLUSION` (success, silent continue) from the failed stages.

<details><summary>Original symptom (kept for reference)</summary>

**Symptom.** On a live RealSense run the `--demo` circuit runs three steps in
order: `facial_movement` → `arm_drift` → `balance`. The middle step never
captured, yet the agent narrated the next step and moved on. Observed output:

```
[agent speaks] Facial movement measurements captured.
[agent speaks] Demo: Raise both arms forward with palms up and hold them still. (1 more to go)
[agent speaks] Raise both arms forward with palms up and hold them still.
[agent speaks] Please reposition so the requested body area is fully visible.   ← capture failed
[agent speaks] Demo: Stand still near safe support, without using it unless needed. (last one)  ← advanced anyway
```

and in the debug state:

```
guided_assessments.positioning  confidence 0.25
"Please reposition so the requested body area is fully visible."
```

**Why it matters.** A demo step that fails to capture while narration continues
is worse than any cosmetic issue — a client watches the circuit appear to work
while it is actually broken. This is why it is first.

**Reproduce.**
```
python main.py --webui --demo --source realsense
```
(Needs the camera. There may be a way to reproduce the positioning failure on a
replay source — worth checking before assuming hardware is required.)

**Where to look first.**
- `agent/voice_agent.py` — `_advance_demo_circuit()` / `_DEMO_CIRCUIT` and the
  `_actions`/result-handling path that decides a step is "done." The circuit
  currently advances whenever `workflows.active(subject)` becomes `None`; if a
  workflow *concludes as failed/aborted* the same way it concludes as
  *succeeded*, the circuit can't tell the difference.
- `core/workflows.py` (`WorkflowEngine`) — how `arm_drift` concludes, and
  whether a positioning timeout ends the session vs. keeps it active. Check what
  distinguishes a captured result from a give-up.
- The `arm_drift` protocol in `assessments/protocols.py` — its positioning gate
  and timeout.

**Definition of done.** When a demo step fails to capture, the circuit should
either (a) retry that step, or (b) announce the skip honestly ("skipping the arm
check — I couldn't see your arms") rather than narrating success and moving on.
Decide which with the user; (b) is the smaller, safer change and reads fine in a
live demo. Add a test alongside `tests/demo_circuit_test.py` for whichever path.

**Unknowns to resolve before coding.** Is the arm not visible a camera-framing
issue (person too close — the run showed `~0.7 m away`) or a pose-landmark
issue? If it is purely framing, the honest-skip path (b) is clearly right and no
capture logic needs to change.

_Resolution (shipped): chose path (a)+(b) combined per the user — announce +
coach + retry once, then honest skip. Framing vs landmark was left undecided; the
retry gives the person a genuine second attempt with reposition guidance either
way._

</details>

### 4.2 Vitals / Skin beats never tick on `client_demo`  ← START HERE

`config/replay_scenarios.json` defines `heart_rate` (t=2), `facial_asymmetry`
(t=10) and `skin_color` (t=26) in the `client_demo` scenario, but across three
independent live captures those modules appeared in **neither** `payload.signals`
**nor** `system.timeline`. So the beat checklist shows `○ Vitals ○ Skin`
permanently on the flagship reel.

`○ Motor` is *correct* — the reel genuinely has no motor events.

Root cause is upstream of the demo page: either the scenario events for those
modules are not emitted, or they are filtered before reaching the payload. Note
`to_payload()` excludes rows whose `(module, key)` is in `compared_keys` (the
multi-backend comparison table) and rows with no `message`. Start there.

### 4.3 `module_counts.running` overcounts

`age_estimation` logs `no model at models/age_googlenet.onnx; disabled` yet still
appears in the enabled list, so the stat line counts it. Debug output confirms
it is inert (`module:age_estimation` p50/p95/max all `0`). Same for `hsemotion`.
The docs currently state 46 — update them if the real count changes.

### 4.4 Quality profile silently downgrades

Running `--quality-profile maximum` yielded `effective: "reduced_geometry"`,
health `degraded`, `geometry_age 1238ms` against a 250ms target, analysis at
13.3 fps, 16,909 throttled module runs. Top costs: `module:rash` (p95 121ms),
`extractor:PoseExtractor` (p50 34ms), `module:dry_lips` (p95 32.6ms).

---

## 5. Things that will waste your time if you don't know them

- **`modules` is not in scope in `system_snapshot()`.** `modules = build_enabled(config)`
  lives in `build_pipeline()`; `system_snapshot()` is nested in `main()`. Use
  `pipeline.scheduler.modules` (as lines ~420 and ~432 already do).
- **`system.timeline` holds only *persisted* events** (`EventStore.recent(60)`,
  filtered by persistence policy). Vitals and most skin results have
  `persistence: none` and never appear there. Anything that must react to those
  has to read `payload.signals` instead.
- **Timeline entry `id` is a hex string**, not an integer. Set-based dedup works;
  numeric assumptions don't.
- **`payload.signals` is a current-frame snapshot**, already sorted severity-desc
  then confidence-desc. It is *not* an event log — rendering it directly makes a
  UI that looks frozen.
- **`--webui` serves port 8770; `--debug-endpoint` serves 8771.** They are
  independent. `ERR_CONNECTION_REFUSED` on 8770 usually means `--webui` was
  omitted, not that anything is broken.
- **A `replay:` run exits when the reel ends** (~50s for `client_demo`), taking
  the web server with it.
- **A "Fact-Forcing Gate" hook blocks the first edit of every file** until you
  state importers/callers, affected API, data schemas, and the user's verbatim
  instruction — then retry the same edit. Budget a turn per file.
- **Module docstrings are written for developers.** `_MODULE_LABELS` covers all
  49 registered modules precisely so the docstring fallback ("MediaPipe
  GestureRecognizer", "deprojected pose landmarks") never reaches a guest screen.
  `tests/demo_labels_test.py` fails if a new detector has no curated label.

---

## 6. Running it

```bash
python main.py --source replay:client_demo --webui      # scripted 50s reel
python main.py --webui --demo --source realsense        # live camera + guided circuit
```

`/` = companion (iPad) · `/data` = caregiver · `/demo` = big-screen guest view.
Debug state, localhost only: `http://127.0.0.1:8771/debug/state`.

```bash
python -m pytest tests/dashboard_signals_test.py tests/demo_labels_test.py tests/http_quiet_test.py -q
```

---

## 7. Non-negotiables

- **The system is non-diagnostic.** Labels and blurbs are observational
  ("Watches for one-sided droop"), never clinical claims. `tests/demo_labels_test.py`
  enforces this mechanically — making the demo "impressive" is exactly the
  pressure that produces overclaiming copy.
- **Never log or display `.env` values, credentials, raw media, or provider
  responses** (see `CLAUDE.md`).
- The guest view shows observations about a **real, identifiable person in the
  room**. Err toward saying less.
