# CareVision timeline

This is a human-readable summary of the Git history. Times are commit-author
times in IST (`+05:30`); commit hashes link the notes back to the exact change.
Generated from `git log --all` on 13 July 2026.

## 7 July 2026 - Foundation and first product features

- **11:00 - `87ecf1e` - Initial CareVision setup.**
  Established the application entry point, camera/pipeline/scheduler core,
  face, pose, and motion extractors, a broad set of care-observation modules,
  output aggregation/overlay, configuration, storage, research notes, and
  initial smoke/pose tests.

- **12:53 - `51e4ba2` - Pluggable models/backends.**
  Added backend interfaces and implementations for emotion detection and
  remote photoplethysmography (rPPG), including DeepFace, FER+, HSEmotion,
  heuristic emotion, classical rPPG, and Open-rPPG. Added the dashboard and
  supporting dependency/test configuration.

- **14:56 - `40836a6` - Statistics display.**
  Extended the dashboard and heart-rate/emotion paths to present statistics;
  added rPPG model benchmarking and backend tests.

- **16:24 - `21a41b5` - Clothing and weather features.**
  Added clothing detection, clothing advice, weather support, debug utilities,
  configuration and dashboard integration, and clothing-specific requirements.

- **16:59 - `6e2fa21` - Voice/advisory agent and alerts.**
  Added environment configuration, Gemini client integration, policy/state and
  voice-agent components, alert management/notification, text-to-speech,
  alert settings, and agent-alert tests.

- **17:28 - `b6a3f09` - iPad-oriented web interface.**
  Added the `webui` package, demo server, and page; adjusted the main runtime
  and clothing integration to support the browser UI.

## 8 July 2026 - Web data, documentation, and video work

- **10:39 - `e853f78` - Advisory data on the web.**
  Added the advisor engine, a web data page, dashboard data output, server/demo
  updates, pipeline integration, and tests for the advisor engine.

- **11:07 - `41636c4` - Architecture documentation.**
  Added `docs/ARCHITECTURE.md` and `docs/EXTENDING.md` while aligning the core,
  extractors, alerts, agent, and audio components with the documented design.

- **11:16 - `a0d7ad2` - Code comments and cleanup.**
  Improved inline documentation around the advisor engine and selected module
  helpers/backends.

- **15:07 - `6131664` - Directory-level agent guidance.**
  Added `AGENTS.md` instructions for the agent, alerts, core, modules, output,
  and web UI areas; added a showcase video asset.

- **17:28-17:40 - `05b1006`, `475d339`, `162e616` - Video showcase iteration.**
  Tuned camera and pipeline behavior, rPPG processing, and tests for camera
  tuning, smoothing, fast paths, staleness, and rPPG signals. Added and then
  renamed/updated showcase video assets.

## 9 July 2026 - Camera switching

- **11:23 - `e89d74a` - Camera support expansion.**
  Added a camera test utility and camera-switch tests; updated the camera core,
  runtime/configuration, and classical rPPG talking gate.

## 13 July 2026 - G1 camera and expanded observation set

- **09:53 - `05ba105` - G1/RealSense camera features.**
  Added a camera factory, RealSense D435i implementation and setup guide,
  speech-to-text, elicitation and corroboration support, and substantially
  expanded observation modules (attention, expressivity, face touch, gesture,
  grooming, height/distance, sneeze, and related refinements). Added tests for
  the new camera/depth, elicitation/corroboration, grooming, facial-region, and
  heart-rate behavior.

## Current uncommitted work (as of 13 July 2026)

The following changes are present in the working tree but are **not yet part of
Git history**: updates to the pipeline, main runtime, voice agent, dashboard,
web data/server, face/pose extraction, and heart-rate module; plus new
`core/showcase.py` and `tests/showcase_gate_test.py`. This section should be
updated after those changes are committed.

## 14 July 2026 - NVIDIA skin screening and multimodal Phases 1-7

This session implemented the NVIDIA-assisted skin-screening plan and then
expanded CareVision through all seven phases of the multimodal showcase
roadmap. These changes are currently working-tree changes rather than a single
Git commit.

### NVIDIA-assisted skin screening

- Added a shared OpenAI-compatible NVIDIA NIM transport in
  `integrations/nvidia_vlm.py`. It resizes frames, JPEG-encodes them in memory,
  submits them with a configurable endpoint/model/timeout, sanitizes remote
  failures, and never logs credentials or encoded media.
- Added the opt-in `skin_vision` module, defaulting to
  `meta/llama-3.2-11b-vision-instruct`. A configured `NVIDIA_API_KEY` is not
  sufficient by itself: `--enable-cloud-skin` is required for each run.
- Implemented background skin inference with one request in flight, periodic
  sampling, request timeouts, failure handling, and exponential backoff.
- Added strict skin-response validation for image quality, visible features,
  body location, confidence, possible conditions, and follow-up topics.
- Split skin output into public observations and agent-only hypotheses. Public
  consumers see only neutral features, body location, quality, and confidence.
- Added the close-up workflow: a preliminary finding prompts the person to
  bring the area closer, the sharpest frame from a short in-memory window is
  selected, and that close-up is analyzed again.
- Added `agent/skin_dialogue.py` to ask at most three neutral questions about
  symptoms, progression, exposure, and relevant warning signs.
- Added strict prompt instructions and a speech guard that rejects condition
  names, aliases, diagnostic wording, or disclosure-prone generated text and
  substitutes a reviewed deterministic question or conclusion.
- Preserved the local facial-rash heuristic as the offline fallback and reused
  the existing `skin_changes` cooldown to prevent duplicate conversations.
- Ensured image-only findings cannot produce caregiver or urgent alerts.
- Added tests for authentication/encoding, consent gating, validation, HTTP
  failures, timeouts, privacy, close-up behavior, prompt construction, and
  speech filtering.

### Phase 1 - Shared observations, workflows, replay, and dashboard

- Extended `Result` with backward-compatible metadata: `subject_id`, `source`,
  `quality`, `location`, `evidence_window`, `correlation_id`,
  `conversation_tags`, and `PersistencePolicy`.
- Added `Visibility.PUBLIC` and `Visibility.AGENT_ONLY`, and changed the
  aggregator to isolate state by subject while keeping private data out of
  normal snapshots and severity queries.
- Added `storage/event_store.py`, a thread-safe SQLite timeline for public,
  JSON-safe observations, assessments, questions, answer classifications, and
  recommendations. Event and baseline retention policies have separate
  expiration periods and can be purged deterministically.
- Hardened persistence against raw arrays, frames, images, JPEG/base64 media,
  audio, video, waveforms, embeddings, biometric data, private hypotheses,
  oversized sequences, and oversized strings. Agent-only results are ignored.
- Retained the numeric `HistoryStore` for longitudinal baselines, migrated it
  to subject-aware storage, added locking, and added recent/mean/last/count
  queries.
- Added a capability registry with ready, degraded, and unavailable states for
  cameras, microphones, cloud paths, models, and optional sensors.
- Added a generalized workflow engine implementing instruction, positioning,
  sampling, scoring, questions, and conclusion, plus cancellation and timeout.
  It enforces one retry, at most three questions, denial suppression, one
  unclear-response rephrase, per-subject concurrency, and a one-minute default
  unsolicited-health-prompt budget.
- Recorded causal chains with a shared `correlation_id` from observation or
  assessment request through scoring, question, answer, and recommendation.
- Added deterministic replay using the same camera, listener, audio bus,
  sensor, workflow, aggregator, and agent interfaces as live operation.
- Replay can use generated pose/face trajectories, recorded video, WAV audio,
  scripted sound/sensor events, and scripted answers. It supports pause,
  resume, restart, seek, and speed controls with synchronized virtual time.
- Expanded the browser dashboard with a chronological event/conversation
  timeline, capabilities, cloud consent indicators, active workflow progress,
  confidence versus measurement quality, replay controls, and track status.
- Added defensive privacy filtering inside dashboard rendering and JSON
  generation, even when an internal snapshot is passed accidentally.

### Phase 2 - Agent-led assessment library

- Added the reusable `AssessmentProtocol` and `AssessmentScore` contracts with
  instructions, required extractors, positioning checks, sampling duration,
  deterministic scorers, quality gates, public-safe summaries, and approved
  follow-up topics.
- Implemented five-times sit-to-stand measurements: repetitions, total time,
  partial/failed attempts, support use, movement consistency, and knee-angle
  range.
- Implemented Timed Up and Go measurements: stand, walk out, turn, return walk,
  sit, per-phase timings, total time, and completed phases.
- Implemented arm-drift measurements: left/right relative height, downward
  drift, symmetry, and compliance.
- Implemented finger-tapping measurements: left/right rates, rhythm
  variability, and side-to-side difference.
- Implemented hold-still balance measurements: body sway, corrective steps,
  and support use.
- Implemented guided gait measurements: cadence, step symmetry, turning
  stability, and a shuffling indicator.
- Implemented facial movement measurements for smile, eyebrow raise, eye
  closure, regional symmetry, and sequence compliance.
- Implemented read-aloud timing for completion, duration, rate, pauses, and
  change from the person's own baseline.
- Implemented guided-breathing observations for compliance, visible cycles,
  and rhythm consistency, explicitly avoiding claims about lung function.
- Changed assessment execution so the instruction must actually be delivered
  before positioning and sampling can begin. Poor-quality attempts receive a
  neutral explanation and no more than one retry.
- Added no-microphone behavior: a measured assessment concludes neutrally
  without waiting for verbal follow-up answers.
- Fixed a replay/conversation race so a question is reserved only after it is
  spoken; early speech can no longer answer a question that was never asked.
- Suppressed unrelated greeting/small-talk candidates while an assessment is
  active, while still allowing the deterministic assessment sequence.

### Phase 3 - Shared audio and sound intelligence

- Added `audio/bus.py` so speech recognition and sound-event detection consume
  one shared microphone stream instead of opening the device twice.
- Updated Faster-Whisper transcription to use VAD and word timestamps.
- Added public-safe speech timing summaries for speech duration, word rate,
  pause count/frequency, response latency, turn gap, interruptions, quality,
  and change from the person's vocal baseline.
- Stored only numeric timing baselines in `HistoryStore`; transcripts remain in
  the existing short in-memory conversation history and audio is never saved.
- Added the optional background YAMNet classifier and canonical event taxonomy
  for cough, sneeze, throat clearing, laughter, crying, calls for help,
  impacts/crashes, alarms, smoke alarms, glass breaking, running water, and
  door sounds.
- Required repeated evidence for ordinary health-related sounds. Only a smoke
  alarm or explicit recognized call for help uses the deterministic urgent
  non-medical path.
- Added replay listener/audio producers that follow the live interfaces and
  publish synchronized synthetic or WAV samples through the shared audio bus.

### Phase 4 - NVIDIA scene and activity understanding

- Added the independent `--enable-cloud-scene` consent flag. Skin and scene
  consent are separate even though they share the NVIDIA transport.
- Added `modules/scene_vision.py` with its own prompt, strict schema, rate
  limit, timeout, quality/confidence fields, private state, and exponential
  backoff.
- Added infrequent multi-frame scene windows for eating, drinking, reading,
  exercising, resting, cooking, and preparing to leave.
- Added object context for cups, meals, medication containers, mobility aids,
  glasses, phones, blankets, and commonly misplaced objects.
- Added visible-context checks for clutter, spills, poor lighting, blocked
  paths, open exterior doors, and unattended-cooking context.
- Scene media remains in memory, only one request can be in flight, and
  capability state changes to degraded during sanitized backoff failures.
- A one-off VLM hazard remains informational. Repeated temporal evidence may
  become a notice, but a VLM-only result can never become an urgent alert.
- Added tests for strict schema validation, independent consent, sanitized
  failure handling, and backoff timing.

### Phase 5 - Routines, temporal safety, and fusion

- Added camera-location labels such as living room, kitchen, entryway, and
  bedroom, and propagated location metadata through results and persistence.
- Added subject-specific routine learning for observed wake/sleep windows,
  room occupancy, meal/drink/medication opportunities, activity/inactivity,
  leaving/returning, conversation frequency, mobility-aid visibility, and
  sit-to-stand timing.
- Expanded routine baselines with seven-day room occupancy, mean activity, and
  inactive-observation counts.
- Added temporal events for recovered stumble/near-fall, repeated difficulty
  standing, slower sit-to-stand versus personal baseline, prolonged post-fall
  immobility, repetitive pacing, unusual night activity, missed normal meal or
  drink opportunity, and possible medication opportunity without claiming
  adherence.
- Added daily public-safe summaries containing only meaningful changes.
- Added conservative deterministic fusion for cough plus reduced activity plus
  user-confirmed symptoms; near-fall plus support use plus reported dizziness;
  repeated stand attempts plus a slower baseline; and a visible drink after a
  long interval.
- Fixed fusion to operate strictly within one `subject_id`, preventing a
  visitor's answer or observation from corroborating the primary person.
- Kept urgent post-fall handling dependent on deterministic fall and prolonged
  immobility signals rather than an LLM, VLM, emotion label, or hypothesis.

### Phase 6 - Anonymous multi-person support

- Replaced largest-person-only assignment with short-lived anonymous track IDs
  using pose/face geometry, centroid association, temporal stability, TTLs,
  and ambiguity reporting.
- Added an opt-in `--enable-multi-person` runtime flag and matching configuration.
- The first stable, unambiguous track becomes primary and remains primary even
  if a visitor appears larger or moves closer. An operator can explicitly
  select a current anonymous track from the dashboard.
- When the primary track is absent or assignment is ambiguous, health modules
  pause rather than allowing visitor data to contaminate the primary profile.
- Kept aggregator state, numeric history, workflow sessions, routine state,
  and dialogue memory isolated by subject.
- Added public debug results and dashboard controls for track ID, assignment,
  stable-frame count, primary status, and ambiguity.
- Added a two-person replay fixture and tests proving that a larger visitor
  cannot take over the primary track or affect primary baselines/fusion.
- Did not add persistent face recognition, names, embeddings, or biometric
  identity storage.

### Phase 7 - Optional sensors and simulated hardware

- Added a common `SensorAdapter`/`SensorReading` contract with capability
  status, timestamp, quality, subject association, source labeling, polling,
  and graceful shutdown.
- Added a non-blocking `SensorManager` that converts trusted measurements into
  public results and numeric baselines while rejecting sensitive measurement
  claims from untrusted RGB-derived sources.
- Added RealSense depth/IMU context for floor-plane distance, nearest obstacle,
  camera motion, hip height above the floor, fall-height context, stance width,
  and metric step width without retaining depth frames.
- Added reconnecting BLE GATT adapters and parsers for heart rate, pulse
  oximetry, smart scale, and blood pressure.
- Added an MLX90640 thermal adapter that immediately reduces thermal frames to
  skin/ambient temperature summaries and does not persist thermal images.
- Added a serial JSON adapter for temperature, humidity, CO2, air quality,
  pollen, ambient light, and other configured sensor hubs.
- Added GPIO adapters for door, bed-pressure, and appliance state changes.
- Added deterministic simulated and replay adapters for every optional sensor
  class so the showcase can run without hardware.
- Ensured blood pressure, SpO2, temperature, and weight are presented as
  measurements only when their source is a real sensor, explicit simulation,
  or replay sensor. RGB body estimates remain categorical proxies.
- Added `requirements-sensors.txt` and documented setup/configuration examples.

### Attention, safety, privacy, and agent behavior

- Added a deterministic attention planner that ranks candidates by priority,
  severity, novelty, confidence, measurement quality, denial state, recent
  speech, and interruption budget.
- Kept test selection, alerts, follow-up topics, retries, and conclusions in
  deterministic workflow code. Generative models receive curated context and
  only phrase approved communication.
- Added public-safe answer-classification events for deterministic fusion while
  keeping raw speech and transcripts out of persistent stores.
- A denial suppresses its topic; an unclear response is rephrased at most once.
- Private hypotheses remain excluded from general conversation memory,
  dashboards, alerts, public snapshots, and SQLite. The skin dialogue receives
  them only as explicitly private prompt context with non-disclosure rules.
- Added defense-in-depth filtering at the event store, aggregator, dashboard,
  web payload, alert input, and speech-output layers.
- Corrected capability reporting so consent alone cannot make a cloud service
  appear ready when its key, model, or dependency is unavailable.

### Replay scenarios and documentation

- Added deterministic scenarios for a healthy greeting, all nine guided
  assessments, cough with follow-up, near-fall recovery, skin close-up,
  meal/hydration context, kitchen spill, anonymous visitor, and wearable plus
  thermal sensor fusion in `config/replay_scenarios.json`.
- Extended replay-generated pose/face profiles for sit-to-stand, Timed Up and
  Go, arm drift/tremor, finger tapping, balance, gait, facial movement,
  breathing, and near-fall recovery.
- Added `docs/MULTIMODAL_SHOWCASE.md` and linked it from the README. It documents
  consent flags, assessment launching, replay, audio behavior, routines,
  anonymous tracking, sensor dependencies, and configuration examples.
- Added focused contract tests in `tests/assessment_library_test.py`,
  `tests/multimodal_expansion_test.py`, and
  `tests/phase_infrastructure_test.py`.

### Validation and acceptance evidence

- Ran the complete automated suite after the final changes: **99 tests passed**.
  The only warning was that pytest could not write its optional `.pytest_cache`
  directory; this did not affect test execution or results.
- Ran Python compilation across the changed application packages with no
  syntax errors.
- Ran `interrogate -c pyproject.toml .`: **98.9% docstring coverage**, above the
  required 95% threshold.
- Ran `git diff --check` successfully; only expected Windows LF-to-CRLF notices
  were printed.
- Ran the production sit-to-stand replay end to end. It delivered the
  instruction, sampled and scored the movement, asked all three approved
  questions, consumed the scripted answers, and produced a neutral conclusion.
- Ran the production anonymous-visitor replay with multi-person support enabled
  and verified it completed without contaminating primary state.
- Ran an explicitly local live-camera smoke test using camera 0 at 640x480. The
  camera opened and locked exposure/white balance, the capture loop remained
  responsive, five frames completed, optional unavailable models self-disabled,
  and the agent greeted normally.
- No cloud-consent flag was supplied during replay or live-camera acceptance,
  so neither the skin nor scene path uploaded any frame despite the configured
  API key. Automated cloud tests used mocks.
- Refreshed the project graph with `graphify update .`: **2,054 nodes, 4,479
  edges, and 184 communities**. The warning about six configuration/JSON files
  producing no AST nodes is expected for non-code files.
- Final goal accounting reported 561,474 tokens and approximately 44 minutes
  of active implementation time.

## Notes

- The history also contains generated `graphify-out/` artifacts. They were not
  treated as product milestones unless paired with source, documentation, or
  test changes.
- Git commit messages are short, so descriptions above are inferred from the
  source files changed by each commit.
