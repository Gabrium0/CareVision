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

## Notes

- The history also contains generated `graphify-out/` artifacts. They were not
  treated as product milestones unless paired with source, documentation, or
  test changes.
- Git commit messages are short, so descriptions above are inferred from the
  source files changed by each commit.
