# Detection Research & Methods

How each item from [detectionList.md](../detectionList.md) is implemented, the
technique behind it, and an honest feasibility rating from a single RGB camera.

**Ratings:** 🟢 reliable · 🟡 works, condition-sensitive · 🔴 experimental / screening-only

> **Not a medical device.** Every health-related signal is a *screening prompt*
> to trigger a gentle check-in or caregiver notification — never a diagnosis.
> Modules emit a confidence and a severity; the greeting engine gates on both.

## Shared foundation
All modules read pre-computed features from a per-frame `FrameContext` produced
once by three extractors, so nothing runs a detector twice:
- **FaceLandmarker** (MediaPipe Tasks) — 478 landmarks incl. iris → `ctx.face`
- **PoseLandmarker** (MediaPipe Tasks) — 33 body landmarks → `ctx.pose`
- **Motion** — frame-difference energy → `ctx.motion_energy`

| # | Detection | Module | Method | Feasibility |
|---|-----------|--------|--------|-------------|
| **Vitals** |
| 1 | Heart rate | `heart_rate` | Two selectable backends run side by side: **classical** (forehead green-channel, bandpass 0.7–3 Hz, FFT peak) and **open-rppg** (neural: FacePhys/PhysMamba/PhysFormer via JAX). | 🟡 classical / 🟢 open-rppg |
| 2 | HRV (RMSSD, SDNN) | `heart_rate` | classical: inter-beat intervals of the filtered waveform (RMSSD + SDNN). open-rppg: RMSSD/SDNN/pNN50/LF-HF from its BVP. | 🔴 classical / 🟡 open-rppg |
| 3 | Respiratory rate | `respiration` | shoulder vertical oscillation, bandpass 0.1–0.5 Hz; **with depth: mean chest-ROI distance** (sub-cm chest wall, works clothed) | 🟡 needs still torso / 🟢 with depth |
| **Skin & face** |
| 4 | Pallor | `skin_color` | cheek normalized-chroma redness below personal baseline | 🔴 white-balance dependent |
| 5 | Flushing / redness | `skin_color` | redness above baseline | 🔴 same |
| 6 | Cyanosis | `skin_color` | bluish lip chroma (B−R) | 🔴 color-calibration dependent |
| 7 | Jaundice tint | `skin_color` | yellow axis (R+G vs B) above baseline | 🔴 same |
| 8 | Rash / eruption | `rash` | red (LAB-a) + high local texture fraction on skin mask | 🔴 screening only |
| 9 | Bruise / discoloration | `bruise` | purple/blue + darker-than-skin connected regions | 🔴 screening only |
| 10 | Eye redness | `eye_redness` | sclera LAB-a redness inside eye rings | 🟡 needs resolution |
| 11 | Sweating | `sweating` | forehead specular-highlight fraction vs baseline | 🔴 confounded by oil/light |
| 12 | Dry / cracked lips | `dry_lips` | lip texture energy + low saturation | 🔴 hydration reminder |
| 13 | Facial asymmetry / droop | `facial_asymmetry` | mirror landmarks across face axis; flag change vs baseline (stroke FAST-F); **with depth: 3D mirror across the mid-sagittal plane** (removes head-yaw confound) | 🟡 screen, not dx |
| 14 | Facial / eyelid swelling | `facial_swelling` | slow eye-aperture / cheek-fullness drift vs long baseline; **with depth: + volumetric cheek/periorbital protrusion** | 🔴 longitudinal / 🟡 with depth |
| **Motor / neurological** |
| 15 | Hand/limb tremor | `tremor` | wrist position FFT, 3–12 Hz band (4–6 Hz = Parkinsonian) | 🟡 fps-limited |
| 16 | Gait abnormality | `gait` | ankle-oscillation cadence + L/R amplitude asymmetry | 🟡 needs legs in view |
| 17 | Balance / sway | `balance` | mid-hip horizontal sway 0.1–1 Hz while standing | 🟡 |
| 18 | Bradykinesia (slowness) | `bradykinesia` | median limb velocity low during active periods | 🔴 coarse |
| 19 | Masked face / flat affect | `masked_face` | low variance of expression distances over 20 s | 🟡 longitudinal |
| 20 | Abnormal eye movement | `eye_movement` | iris-vs-corner gaze offset; 3–10 Hz oscillation = nystagmus | gaze 🟡 / nystagmus 🔴 |
| **Fatigue & consciousness** |
| 21 | Drowsiness (PERCLOS) | `drowsiness` | eye-aspect-ratio vs baseline, % closed over 60 s | 🟢 |
| 22 | Blink rate | `drowsiness` | closure onsets per minute | 🟢 |
| 23 | Microsleep | `drowsiness` | continuous closure > 1.2 s | 🟢 |
| 24 | Yawning | `yawn` | mouth-aspect-ratio wide & sustained > 1.5 s | 🟡 |
| 25 | Head nodding / droop | `head_nod` | head-pitch oscillation / sustained downward pitch | 🟡 |
| **Safety** |
| 26 | Fall | `fall` | torso→horizontal + rapid center drop + stays low | 🟡 clear falls |
| 27 | Prolonged immobility | `unresponsive` | motion below threshold for N minutes while present | 🟡 |
| 28 | Wandering / pacing | `wandering` | repetitive horizontal traversal + direction reversals | 🟡 |
| 29 | Hazard-zone entry | `hazard_zones` | foot point-in-polygon for calibrated zones | 🟡 needs calibration (off by default) |
| **Emotion & cognition** |
| 30 | Emotion | `emotion` | ONNX FER+ if present, else landmark geometry heuristic | 🟡 |
| 31 | Pain / grimacing | `pain` | action-unit proxies (brow lower, eye tighten, lip raise) vs baseline | 🔴 screening |
| 32 | Agitation / restlessness | `agitation` | high + erratic wrist-speed variance without locomotion | 🔴 |
| **Behavior (longitudinal, SQLite)** |
| 33 | Activity level / inactivity | `activity_level` | 10-min vs 1-hr mean motion ratio, persisted | 🟡 needs history |
| 34 | Presence / arrival | `presence` | face/pose presence with absence timeout; drives greetings | 🟢 |
| **Demographic / context** |
| 35 | Age estimation | `age_estimation` | optional ONNX age net (Levi-Hassner buckets); self-disables if model absent | 🟡 coarse |
| 36 | Body build proxy | `body_estimate` | shoulder/hip width-to-height ratio (NOT BMI) | 🔴 uncalibrated |
| **Interaction & symptoms (showcase additions)** |
| 37 | Hand gesture | `gesture` | MediaPipe GestureRecognizer (thumbs up/down, open palm, victory, pointing, fist) + waving via palm-center oscillation; self-disables without `models/gesture_recognizer.task` | 🟢 static labels / 🟡 waving |
| 38 | Eye contact / attention | `attention` | gaze offset (iris vs corners) + head-yaw proxy; sustained-contact timer for engagement | 🟡 |
| 39 | Sneeze | `sneeze` | rapid head-pitch jerk + reflex eye closure in the same sub-second window; 10-min rolling count | 🟡 face in view only |
| 40 | Nose-wipe / face touch | `face_touch` | pose wrist/index-tip within nose radius, edge-triggered; behavioral proxy for a runny nose | 🟡 |
| 41 | Height & distance | `height_distance` | **depth only**: deprojected nose→ankle span + torso median depth | 🟢 distance / 🟡 height (±5 cm) |
| 42 | Expressivity / flat affect | `expressivity` | smile level + expression variance + blink rate vs 30-day personal history (GDS-15-adjacent cues); check-in fuel, never a mood label | 🟡 longitudinal |
| 43 | Elicited tremor test | `tremor` (`tremor_test`) | agent-scripted hold-still window (core/elicitation.py, 't' hotkey or "check my hands"); elicited protocol is what the ~0.98 video-vs-accelerometer validation used | 🟢 elicited / 🟡 passive |

## The corroboration loop (CDSS pattern)
Low-confidence cues never surface directly. `agent/corroboration.py` turns
them into gentle follow-up questions ("have you noticed any skin changes
lately?"); with the microphone listener (`--listen`, `audio/stt.py`,
faster-whisper offline) the person's answer is classified
(confirmed/denied/unclear — Gemini when available, keywords offline) and
only **confirmed** topics produce a spoken suggestion; denials suppress the
topic for hours. This is the boundary that keeps camera inference on the
"conversation steering" side rather than the "health screening device" side
— worth stating explicitly in any showcase material.

The cold-symptom composite (`advice.cold` in `config/modules.yaml`) combines
#39 + #40 + flushing (#5) + drowsiness (#21) into a gentle check-in — at
least two independent signals required, never a diagnosis.

## D435i depth capabilities (Unitree G1)
The rows marked "with depth" activate automatically on `--source realsense`
(scheduler `"depth"` requires-token; RGB sources are unaffected). Design and
phasing: [REALSENSE_D435I.md](REALSENSE_D435I.md). Still impossible even
with the D435i: SpO₂/cyanosis as a measurement, true fever (no thermal).

## Items intentionally deferred (need extra hardware / sensors)
These appear in detectionList.md's notes but are out of scope for pure RGB:
- **SpO₂, blood pressure** — RGB-only estimates are not trustworthy; would need
  calibrated multi-wavelength or a contact sensor.
- **Skin temperature / fever confirmation** — needs a thermal/IR camera.
- **Cough, speech slurring, fall sound** — need the audio channel; the module
  interface can host an `AudioContext` extractor the same way.
- **Grooming/hygiene, clothing appropriateness, weight change, sleep quality,
  medication/eating tracking** — all longitudinal + scene-context heavy; the
  `storage/history_store.py` + a room/object detector are the hooks to add them.
- **Person identity / re-ID (greet returning people by name)** — face
  embeddings are proven tech, but storing biometric identity is a deliberate
  privacy decision; deferred until explicitly wanted.
- **Multi-person handling** — extractors currently track one subject
  (largest face); several people in frame get merged into whoever dominates.
- **Speaking detection as its own signal** — a mouth-motion heuristic already
  gates rPPG inside `heart_rate`; promoting it to a module output is cheap.
- **Open-vocabulary rash detection** (OWLv2/`clothing.py` infrastructure,
  prompted with "skin rash") — tried-and-true infra, but zero-shot medical
  skin detection is unreliable; would stay 🔴 screening-only next to `rash`.

## Reproducing the models
`models/` holds MediaPipe `.task` bundles (downloaded on setup). Optional ONNX
models (`emotion.onnx`, `age_googlenet.onnx`) enable the higher-accuracy paths;
without them those modules fall back to heuristics or self-disable.
