# iPad on-device offload — implementation plan (Phases 1–3)

Status: **planned, not started.** Blocked only by the demo iPad's OS version.
Continue from here once the iPad is on iPadOS 16.4+ (17 for WebGPU).

## Why

The laptop pipeline is CPU-saturated (analysis was frame-starved at ~3.6 fps;
`PoseExtractor` ~31 ms/frame, face worker ~18 ms). That saturation is also what
inflates the iPad's per-frame ACK latency (~130–170 ms), which is the current
throughput ceiling even after the windowed-ACK fix (commit `7b00d29`, ~20 fps).

Moving the heaviest per-frame work — MediaPipe **pose** and **face** landmarking —
onto the iPad's GPU lets it send a few KB of landmarks (and later small ROI crops)
instead of full JPEGs. That:

- **offloads the laptop** → lower ACK latency → higher end-to-end fps (helps every
  time-based signal, including rPPG);
- **improves geometry accuracy** modestly — landmarking runs on the *uncompressed,
  higher-res, evenly-timed* camera frame instead of a lossy 640×480 q0.7x JPEG;
- with Phase 3, **improves rPPG (heart-rate/SpO₂)** meaningfully — the skin-ROI
  color is sampled from raw pixels with clean timestamps, removing JPEG chroma
  compression and timestamp jitter (the dominant remaining error sources).

## Prerequisite (device) — already measurable

On-device MediaPipe **hard-requires WASM SIMD and WebGL2**; WebGPU is optional but
wanted. The page already probes these and reports them; the laptop surfaces them at
`IPadLink.status().client_caps` (see commit `0654a7c`). The current demo iPad
(6th gen, still on Safari 13) reports `wasm_simd:false, webgl2:false, webgpu:false,
rvfc:false` → **must update to iPadOS 17 first**. iPad 6th gen supports iPadOS 17.

**Free win from the update alone (do before/independent of this plan):** iPadOS 15.4+
gives `requestVideoFrameCallback`, which replaces the `performance-now` clock
fallback (we measured 65 clock resyncs). Even-spaced capture timestamps directly
improve rPPG input quality with the *current* server-side pipeline — verify via
`camera.clock_source` becoming `rvfc-mediaTime` and `camera.clock.resyncs` dropping.

**Gate for all phases:** only enable on-device paths when
`client_caps.wasm_simd && client_caps.webgl2` (pose/face). Fall back seamlessly to
the current server-side path otherwise. Report the active mode in diagnostics.

## MediaPipe asset delivery (Phase 1 prerequisite)

`@mediapipe/tasks-vision` (WASM + `.task` models) must be reachable by the page.
The relay (`relay/server.py`) currently serves only 3 routes and has **no general
static server** and **no CSP**.

- **Recommended: vendor** the tasks-vision runtime (`vision_wasm_internal.{js,wasm}`,
  the ESM entry) into `relay/static/vendor/tasks-vision/`, and reuse the existing
  `models/*.task` files. Add **one bounded static route** to `relay/server.py`
  (allowlist exact filenames; `Cache-Control: public, max-age=...` for the large
  immutable assets — unlike the page, these can cache). Vendoring keeps it working
  on the isolated hotspot and avoids a CDN runtime dependency.
- Alternative (fastest prototype): load from `cdn.jsdelivr.net` + Google model
  storage. Works (no CSP) but needs iPad internet and adds an external dep.

Pin the tasks-vision version and keep the `.task` model files identical to the
laptop's (`models/pose_landmarker_lite.task`, `models/face_landmarker.task`) so
geometry is reproducible across the two MediaPipe builds (WASM vs desktop Python).

## Phase 1 — on-device PoseLandmarker

**iPad (`relay/static/ipad.html`)**
- Behind the capability gate, load `PoseLandmarker` (`runningMode: "VIDEO"`,
  `numPoses: 2`, delegate GPU/WebGL2). It runs directly on the `<video>` element:
  `poseLandmarker.detectForVideo(previewEl, mediaTimeMs)`.
- Call it in the existing capture callback (`startCaptureLoop` → rVFC `onFrame`)
  and send `{type:'pose_landmarks', media_time, seq, poses:[[[x,y,z,v]×33], …]}`
  on the **control** channel (reliable). Round floats (~4 dp) to bound size
  (~1–2 KB/pose). Reuse the same `mediaTime` stamped on frames.
- Add `ondevice_pose` to the `client_version` `caps`/report so the laptop knows a
  landmark stream is active.

**Laptop**
- `core/ipad_link.py`: in `_ingest_control`, handle `type == "pose_landmarks"` —
  validate (list ≤4 subjects, each exactly 33 × 4 finite floats in bounds), store
  the latest as `{poses, media_time, recv_wall}`. Add `latest_device_pose(max_age)`
  returning the poses only if fresh (e.g. ≤0.5 s). Follow the existing bounded-
  validation style (`_validated_sender_stats`, `_validated_capture_profile`).
- `core/ipad_camera.py`: in `frames()`, before `yield`, attach
  `ctx.extras["device_pose"] = self._link.latest_device_pose()` (or `None`). The
  same `ctx` reaches the extractors (`core/pipeline.py:986–997`).
- `extractors/pose.py`: factor the landmark→context logic (current lines ~60–85:
  `ctx.extras["poses"]`, largest-pose `ctx.pose = PoseData(...)`, `person_present`)
  into a shared `apply_pose_landmarks(ctx, poses)` where `poses` is a list of
  `(33,4)` float32 arrays. `PoseExtractor.extract` then:
  - if `ctx.extras.get("device_pose")` is usable → `apply_pose_landmarks(...)`,
    return (skip MediaPipe / the `detect_for_video` call);
  - else run MediaPipe as today and feed the same helper.
- Correspondence: **latest-wins by `media_time`** for v1 (pose barely moves in one
  frame). Refinement: buffer a small ring keyed by `seq`/`media_time` and match the
  decoded frame exactly.
- Flag/telemetry: auto-enable when caps allow (or `--ipad-ondevice-pose`); surface
  the active mode + a device-vs-local counter in `camera` diagnostics.

**Tests**
- `apply_pose_landmarks` parity: identical `ctx` from a device payload vs the
  MediaPipe-shaped landmark arrays.
- `ipad_link` pose-landmarks validation (reject wrong shape / non-finite / oversize;
  freshness expiry).
- Capability gating (no device path when `wasm_simd`/`webgl2` false).

## Phase 2 — on-device FaceLandmarker + ROI crops

- iPad: add `FaceLandmarker` (478 landmarks) on-device; send landmarks like Phase 1.
  478×(x,y,z) is larger — send a compact subset if only ROIs are needed downstream,
  or full mesh at a reduced rate.
- Laptop: mirror the Phase 1 adapter for the face path (`extractors/face.py`,
  `extractors/face_landmarks.py`, `core/fast_face_tracker.py`) — accept device face
  landmarks, skip local inference when present.
- Once **both** pose and face run on-device, full-frame JPEGs become optional: send
  landmarks + **ROI-cropped** JPEGs (small) for the pixel-appearance detectors
  (skin color, rash, bruise, eye redness, dry lips, sweating, facial swelling,
  emotion, `skin_vision`). ROIs are derived from the landmarks. This is where the
  bandwidth/latency win compounds.

## Phase 3 — on-device rPPG ROI color sampling

- With on-device face landmarks, compute the mean skin-ROI color per frame on the
  **raw** (uncompressed) frame on the iPad, and send a compact time series
  `{type:'rppg_samples', media_time, roi:{r,g,b,n}, …}` on the control channel.
- Laptop: `modules/heart_rate.py` + `modules/rppg_backends/*` consume the color
  series instead of sampling from the decoded JPEG. Keep the ROI definition
  (forehead/cheeks) consistent with the current server-side ROI logic.
- Biggest rPPG accuracy win: removes JPEG chroma quantization (recall the green-
  channel-under-4:2:0 workaround in `config/modules.yaml`) and timestamp jitter.
- Keep the server-side rPPG path as the fallback when on-device is unavailable.

## Cross-cutting

- **Fallback discipline:** any on-device load failure (model fetch, WebGL context,
  OOM) must fall back to the current server-side path without dropping the session.
- **Model version pinning** for reproducibility across MediaPipe builds.
- **Thermal/perf:** on-device landmarking adds iPad GPU load; watch for thermal
  throttling of the on-device fps (the iPad currently only encodes JPEGs, so there
  is headroom, but Phase 1+2+3 together are heavier).
- **Bandwidth:** landmarks + ROI crops ≪ full frames; expect the ACK-latency
  ceiling to relax as the laptop sheds work.

## Verification / A-B (numbers, not promises)

Everything needed is already in diagnostics:
- **Feasibility:** `camera.link.client_caps` shows `wasm_simd/webgl2/webgpu/rvfc`.
- **Refresh proof:** `camera.link.client_build` vs `sha256(ipad.html + policy.js)[:12]`
  off-disk (see `relay/server.py:_client_build_id`).
- **Throughput:** `camera.link.sender_stats.{sent_fps, ack_p90_ms, buffered_bytes}`
  and `performance.{capture_fps, analysis_fps}` — expect fps up, ack latency down as
  the laptop offloads.
- **rPPG quality:** `camera.clock_source` (`rvfc-mediaTime` after update),
  `camera.clock.resyncs`, and the `rppg_input_quality` signal; A/B device-landmarks
  + on-device sampling vs the server-side path on the same subject.
- **Geometry:** landmark stability / jitter of `ctx.pose` across frames.

## Concrete file touch-list

- `relay/static/ipad.html` — MediaPipe load (gated), per-frame landmark/rPPG sends,
  `caps.ondevice_*` reporting.
- `relay/static/vendor/tasks-vision/*` (new) + `relay/server.py` — one bounded
  static route for the WASM runtime and `.task` models.
- `core/ipad_link.py` — `_ingest_control` handlers + validators + freshness getters
  for `pose_landmarks`, `face_landmarks`, `rppg_samples`.
- `core/ipad_camera.py` — attach `ctx.extras["device_pose"|"device_face"|"device_rppg"]`
  in `frames()`; diagnostics for active on-device modes.
- `extractors/pose.py`, `extractors/face.py`, `extractors/face_landmarks.py` — shared
  `apply_*` helpers + device-first branch.
- `modules/heart_rate.py`, `modules/rppg_backends/*` — accept an external color series.
- Tests alongside each (parity, validation, gating).

## Risks

- MediaPipe WASM vs desktop Python builds are close but not bit-identical — pin
  versions; validate parity on a real subject before trusting geometry.
- iOS Safari WASM/WebGL quirks (memory, context loss on backgrounding) — the page
  already survives backgrounding for capture; extend the same generation-guard
  discipline to the landmarker lifecycle.
- Multi-person: device landmarks must carry stable per-subject ordering or the
  laptop tracker (`core/pipeline.py` tracks/bboxes) must re-associate.
