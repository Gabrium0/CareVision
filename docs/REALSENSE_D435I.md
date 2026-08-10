# Extension: Intel RealSense D435i (Unitree G1)

Status: **implemented, hardware verification required.** The `realsense` source
is wired through `main.py` and `core/realsense_camera.py`; claims about physical
stream delivery, depth alignment, or IMU behavior still require a D435i.
Use [OPERATIONS.md](OPERATIONS.md) for general setup and current run modes;
`main.py --help` remains authoritative for capture flags and defaults.

This document describes what becomes possible on the target deployment platform — a **Unitree G1**
humanoid carrying a **RealSense D435i** — and how it integrates with the
existing architecture. Everything here is additive: when depth is absent
(the current OV2735/laptop webcam paths), nothing changes.

See [RESEARCH.md](RESEARCH.md) for the method/feasibility rating behind
2D modules, and [ARCHITECTURE.md](ARCHITECTURE.md) /
[EXTENDING.md](EXTENDING.md) for the module system this source uses.

## Why the D435i is a different capability class

The primary camera today is a 2 MP USB 2.0 UVC webcam (OV2735). USB 2.0
forces MJPEG with 4:2:0 chroma subsampling — half the color resolution is
discarded before any code runs — so color and eye-based signals are capped
while luma-based landmark geometry is largely unaffected. The D435i provides
three streams the OV2735 cannot:

- **Uncompressed RGB** (USB 3, no MJPEG chroma penalty).
- **Stereo depth** (2 IR imagers + IR projector, ~0.3–3 m range, mm–cm
  accuracy at the 1–1.5 m operating distance).
- **A 6-DoF IMU** (accelerometer + gyroscope).

Depth, metric scale, and ego-motion are the game-changers here — they unlock
signals 2D RGB structurally cannot measure, not just better versions of the
existing ones.

## Architecture (do this once; everything else builds on it)

- **New source backend `core/realsense_camera.py`** (lazy `import
  pyrealsense2 as rs`; optional dependency in a new
  `requirements-realsense.txt`, matching the repo's existing optional-extra
  convention alongside `requirements-openrppg.txt` /
  `requirements-clothing.txt`). It mirrors the `Camera` interface
  (`core/camera.py`) — `frames()` yields `FrameContext`, plus
  `register_fast_hook`, `switch_to` (so the `c`-key camera toggle works
  *between* the OV2735 and the D435i too), and `release()`. Internally: an
  `rs.pipeline` with color + depth enabled, `rs.align(rs.stream.color)` so
  depth is pixel-aligned to color, and the motion streams (accel/gyro) for
  the IMU.
- **Camera factory** in `main.py`: `--source realsense` (or auto-detect a
  connected D435i) constructs `RealSenseCamera` instead of `Camera`.
  `build_pipeline` changes from `Camera(source=...)` to a small
  `make_camera(source, opts)` selector — one indirection, no changes to the
  rest of the pipeline.
- **Extend `FrameContext`** (`core/context.py`) with optional, default-`None`
  fields so nothing else breaks:
  - `depth: Optional[np.ndarray]` — uint16 depth (mm) aligned to `frame`.
  - `intrinsics` and `depth_scale` — for metric math.
  - Helpers `depth_m(px)` → meters at a pixel, and `deproject(px)` → 3D point
    (`rs.rs2_deproject_pixel_to_point`), plus `mm_per_px()` at the subject.
  - `ego_motion: float` — IMU-derived motion magnitude for gating.
- **New `requires` token `"depth"`**: the scheduler skips depth-dependent
  modules when `ctx.depth is None`, so the same module can run 2D on the
  OV2735 and 3D on the D435i.

## What the D435i unlocks, and how

Highest value first — these are the reasons to use it:

1. **IMU ego-motion gating (build first).** On a *walking* G1, camera motion
   destroys every frequency-domain vital. Read accel/gyro, compute a motion
   magnitude into `ctx.ego_motion`, and add a shared confidence multiplier
   (mirror the existing `low_light_factor` pattern in `modules/_util.py`)
   that **deweights or suspends** rPPG/respiration/tremor when the robot
   moves. This is the single biggest real-world robustness win, and it
   benefits modules you already have without touching their core logic.

2. **Robust respiration (easy, high value).** `modules/respiration.py`
   already buffers a 1D signal and bandpasses 0.1–0.5 Hz. Swap the input:
   instead of `shoulder_y`, feed **mean depth over a chest ROI** (below the
   pose shoulders). Sub-cm chest-wall motion, works clothed, robust to
   lighting. Keep the exact `TimedBuffer`/`dominant_frequency` pipeline —
   only the sampled value changes.

3. **Volumetric facial edema.** Sample `ctx.depth` over the existing
   `face_skin_mask`, and track a coarse **volume/surface proxy** (integrated
   depth deviation over periorbital + cheek ROIs) vs. a long baseline — a
   true 3D upgrade to `facial_swelling`'s 2D contour drift. Depth resolves
   real puffiness from expression far better than 2D geometry can.

4. **3D facial asymmetry / droop.** Deproject the symmetry landmark pairs
   (`extractors/face_landmarks.py::SYMMETRY_PAIRS`) to 3D, mirror across the
   3D mid-sagittal plane (from nose/chin/face-edge 3D points), and measure
   the 3D residual. Removes the head-yaw confound that inflates 2D
   asymmetry — a direct precision boost to `facial_asymmetry`, today's
   highest-value screening signal.

5. **Metric scale normalization (foundational).** With depth + intrinsics,
   express every ROI radius and threshold in **mm regardless of distance**,
   and **distance-gate** (skip when the subject is too far/near for reliable
   pixels). Add `ctx.mm_per_px()`; retrofit the color/texture modules to size
   skin patches in mm. Makes thresholds stable as the robot or subject move.

6. **Metric fall / posture / gait / balance.** Deproject pose joints to
   metric 3D (or read depth at joints): body height, center-of-mass height,
   subject distance, stride length in **meters**, postural sway in **mm**.
   Far more reliable than the current 2D `fall`/`gait`/`balance` proxies.

7. **Better color signals (free with the RGB stream).** USB 3 uncompressed
   RGB removes the MJPEG 4:2:0 penalty, so pallor/jaundice/flushing/rash
   regain real headroom vs. the OV2735. Eye-based signals
   (conjunctivitis/scleral jaundice) remain resolution-limited at 1–1.5 m
   unless you crop closer, but the color *trend* signals get materially
   cleaner.

### Still NOT possible even with the D435i

- **SpO₂ / cyanosis as a measurement** — the D435i's IR is structured/stereo
  depth, not two-wavelength oximetry. Unchanged verdict: color-only
  screening prompt, never a number.
- **True fever / thermal** — no thermal sensor; fever stays an indirect
  inference (flush + sweat gloss).

### Caveats

Depth is noisy on hair, edges, and specular/oily skin; the IR **projector can
contaminate RGB skin pixels** in some modes (you can disable the emitter for
clean color capture, trading depth quality — consider alternating frames);
and subject-relative motion on a moving robot still challenges rPPG even with
IMU gating.

## Suggested phasing

- **Phase A — plumbing:** `RealSenseCamera` + `FrameContext.depth /
  intrinsics / ego_motion` + camera factory + `"depth"` requires token.
  Verify color-path parity with the OV2735 (no regressions when depth is
  unused).
- **Phase B — robustness:** IMU ego-motion gate + metric-scale/distance-gate
  helpers. Biggest wins, reused everywhere.
- **Phase C — depth vitals:** respiration (depth ROI), facial_swelling
  (volumetric), facial_asymmetry (3D). Highest-value clinical signals.
- **Phase D — metric body:** fall/gait/balance in real units.

## Hardware validation checklist (run with a physical D435i)

1. **Bring-up**: `pip install -r requirements-realsense.txt`, then
   `python main.py --source realsense`. Confirm the face box on the color
   image and sensible `distance_m` values (depth is aligned to color).
2. **Parity**: same scene via `--source 0` and `--source realsense` — the
   RGB-only modules should behave equivalently. Press `c` to cross-switch
   both directions; the stream must survive both.
3. **Depth respiration**: sit still 60 s, count breaths, compare with
   `respiration.breaths_per_min` (published depth benchmark: <1 brpm RMSD).
4. **rPPG vs reference**: compare `heart_rate.bpm` with a pulse
   oximeter/fitness watch, once with the IR emitter on and once with
   `emitter: false` — quantify projector contamination of skin pixels and
   pick the default from data.
5. **Metric checks**: `height_m` vs a tape measure; `distance_m` at
   1.0 / 1.5 / 2.0 m marks.
6. **Elicited tremor**: press `t` (or say "check my hands" with
   `--listen`); do one run holding still and one deliberately shaking
   ~5 Hz — expect "steady" vs a ~5 Hz `tremor_test`.
7. **Corroboration loop end-to-end** (`--listen`): provoke a low-confidence
   flag, wait for the follow-up question, answer aloud, and confirm the
   topic state change and the gated conclusion (or suppression on "no").

## Verification

Bring up with `pyrealsense2` examples to confirm aligned depth + IMU
streams; run `--source realsense` and confirm the color path matches OV2735
output (parity); assert every depth-aware module returns identical results
to its 2D fallback when `ctx.depth` is forced to `None` (a graceful-degrade
test, stubbed like `tests/camera_switch_test.py`); validate the respiration
depth-ROI against a counted breath rate; validate the ego-motion gate by
walking the robot while watching vitals confidence drop.

## Sources

- Stroke/palsy landmarks: [IEEE](https://ieeexplore.ieee.org/document/10924203/),
  [MDPI Computers 13/8/200](https://www.mdpi.com/2073-431X/13/8/200),
  [DeepStroke](https://arxiv.org/pdf/2109.12065),
  [JeFaPaTo](https://github.com/cvjena/JeFaPaTo)
- Skin-tone fairness: [Beyond Fitzpatrick (npj Digital Med)](https://www.nature.com/articles/s41746-025-01770-4),
  [DermDiff](https://arxiv.org/pdf/2503.17536)
- Depth vitals context (respiration/edema via RGB-D): general RealSense +
  depth-PPG literature; validate on-device before trusting any figure.
- Out of scope on the OV2735, revisit with the D435i or a close/tele crop:
  [SCIN](https://github.com/google-research-datasets/scin),
  [ScleraSegNet](https://github.com/xiamenwcy/ScleraSegNet),
  [RITnet](https://arxiv.org/pdf/1910.03274),
  [BiliScreen](https://ubicomplab.cs.washington.edu/publications/biliscreen/),
  [CP-AnemiC](https://www.sciencedirect.com/science/article/pii/S2590093523000395)
