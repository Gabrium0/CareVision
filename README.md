# Humanoid Camera — Elderly Care Detection Pipeline

A modular computer-vision pipeline that watches live camera footage and detects
health, safety, and emotional signals to power **customized greetings and
recommendations** for elderly people living alone.

It implements every item in [detectionList.md](detectionList.md) as an
independent, hot-swappable module. See [docs/RESEARCH.md](docs/RESEARCH.md) for
the method and feasibility rating behind each one, and
[docs/REALSENSE_D435I.md](docs/REALSENSE_D435I.md) for a design doc on what a
depth+IMU camera (e.g. the Unitree G1's RealSense D435i) unlocks beyond a 2D
webcam.

> ⚠️ **Not a medical device.** All health signals are screening prompts to
> trigger a gentle check-in or caregiver alert — never diagnoses.

## Quick start
```bash
pip install -r requirements.txt
pip install -r requirements-openrppg.txt   # optional: neural rPPG backend
pip install -r requirements-clothing.txt   # optional: OWL-ViT clothing detection
# MediaPipe .task models are downloaded into models/ (see docs/RESEARCH.md)
python main.py                       # default webcam; Camera + Data windows
python main.py --source clip.mp4     # run on a video file
python main.py --combined            # old single-window overlay instead
python main.py --headless --name Margaret   # no window; prints greetings/alerts
python main.py --enable-cloud-skin    # opt in to NVIDIA skin screening
python main.py --source realsense --webui --debug-endpoint  # smooth preview + local raw state
```
Windowed controls: `q` quit · `g` force a greeting · `c` switch camera (see below).

By default two windows open: **Camera** (video + face/pose boxes only) and
**Detections — Data**, a readable dashboard with all signals. The vitals
section shows both heart-rate backends side by side:

![data window](docs/dashboard_preview.png)

### Camera source: switching and auto resolution
Run with two cameras configured and swap between them live, without
restarting — useful when an external webcam (e.g. a low-cost USB 2.0 UVC
camera) needs to be compared against or replaced by the built-in laptop
camera on the fly:
```bash
python main.py --list-cameras                       # print index + resolution for each device
python main.py --source 1 --alt-source 0             # start on index 1; 'c' toggles to 0 (laptop)
```
Pressing `c` requests the swap on the camera's own frame loop (never from the
keypress handler directly, so it can't race the reader thread) and re-runs
that device's exposure/white-balance/gain lock. If the new device fails to
open (unplugged, busy), it **automatically reverts to the previous camera**
and keeps running rather than crashing.

By default (`--resolution auto`) the camera probes candidate resolutions
(1920x1080 → 1280x720 → 960x540 → 640x480) at startup and locks in the
**largest one that still delivers `--min-fps` (default 25)** — the floor
below which the FFT-based vitals (`heart_rate`, `respiration`, `tremor`)
start to alias. This also sets the MJPG FOURCC first, since many UVC
webcams — especially budget USB 2.0 sensors — silently cap out around
640x480 on their default uncompressed capture mode otherwise. Pass an
explicit `--resolution 1280x720` to skip probing. Startup prints each
candidate's measured fps and the final choice, e.g.:
```
[camera] probe 1920x1080 -> delivered 1920x1080 @ 4.3fps
[camera] probe 1280x720 -> delivered 1280x720 @ 4.3fps
[camera] probe 960x540 -> delivered 640x480 @ 29.7fps
[camera] auto-selected 640x480 @ ~29.7fps (min_fps=25)
```
A low-resolution auto-selection like this usually means the camera/USB link
can't sustain higher-resolution frame delivery (common on USB 2.0 with a
2 MP sensor) — pass `--min-fps` lower to trade frame rate for spatial detail
if the vitals modules aren't a priority, or use a USB 3 camera for both.

For RealSense, auto mode probes actual stream delivery and chooses the highest
profile that sustains `max(--min-fps, 95% of --fps)`. Depth-capable profiles
are preferred over a higher-resolution color-only profile. The local preview
and full-rate rPPG sampler consume the capture stream independently from the
heavier analysis loop; the overlay reports capture (`C`), preview (`P`), and
analysis (`A`) FPS separately. `--analysis-width 960` controls only the
MediaPipe input size—landmarks and rPPG crops still use capture coordinates and
original pixels.

### Private debugging endpoint

Pass `--debug-endpoint` to serve the complete live result objects, including
agent-only hypotheses, at `http://127.0.0.1:8771/debug/state`. Use
`--debug-port` to change the port. This is a separate localhost-only server;
the LAN-facing `/data` dashboard remains public-only. Raw image/audio arrays,
binary values, and data URLs are always redacted. The payload also includes
capture/preview/analysis rates, latency, skipped analysis frames, selected
camera profile, and rPPG fast-path counters.

### Heart rate: two backends, compared live
The `heart_rate` module runs one or more rPPG backends and reports each one's
numbers so you can compare them:
- **classical** — forehead+cheek multi-ROI CHROM/POS chrominance + FFT
  (numpy/scipy only; illumination-robust, `classical_method: chrom|pos|green`
  in config). Reported bpm is median-smoothed over the last
  `classical_smoothing_window` raw picks (default 5), same as open-rppg below.
- **open-rppg** — neural models ([KegangWangCCNU/open-rppg](https://github.com/KegangWangCCNU/open-rppg),
  JAX); gives HR, RMSSD, SDNN, and breathing rate. Auto-disables if not installed.

**Accuracy vs. a wearable (Apple Watch, chest strap, pulse oximeter):** camera
rPPG measures a ~0.5-2% skin-color change from a webcam and is inherently
noisier than a wearable's direct-contact PPG sensor — this is true of any
rPPG algorithm, including trained neural models like open-rppg, not just the
classical backend here. To get close to wearable accuracy: sit ~40-60cm from
the camera, face it directly, use steady indirect light (avoid backlighting
or flicker), and hold still without talking for the ~10-30s the backend needs
to fill its buffer — the same movement that trips `facial_asymmetry` or
`drowsiness` alerts is exactly the kind of motion that degrades rPPG signal
quality, independent of algorithm choice. A single live reading compared
against a watch during normal activity is not a reliable accuracy check; use
`tests/benchmark_rppg_models.py` (below) against a recorded clip with a known
reference bpm to actually measure MAE.

If readings still look wrong (jumping between backends, low confidence) on a
live webcam specifically, the vitals "fast path" that feeds the buffers from
the camera's reader thread can be starved if the heavy per-frame detection
loop falls behind — run with `--debug-modules pipeline` to see a periodic
`[pipeline/debug]` summary of how often fresh face geometry is published vs.
how many fast-path frames were fed/rejected as stale.

Select in `config/modules.yaml`:
```yaml
heart_rate:
  backends: [classical, openrppg]   # drop openrppg to run classical only
  openrppg_model: null              # null = default; try physformer or efficientphys
```
open-rppg loads its model once (~15–20s) at startup, then runs batched
inference on a rolling buffer of face crops. Live inference runs in a background
thread so CPU-only Open-RPPG does not freeze camera frames; by default it updates
about every 5 seconds from the same 30-second signal window.
The backend sets `KERAS_BACKEND=jax` before importing Open-RPPG to avoid
TensorFlow/JAX tensor mismatches. On startup it logs the selected model,
dependency versions, and JAX devices so you can confirm whether the A2000 GPU is
being used. If JAX only prints `cpu:0`, install a CUDA-enabled JAX wheel; on
Windows this is most reliable inside WSL2/Linux.
The dashboard always shows Open-RPPG rows for HR, RMSSD, SDNN, respiration, and
status; unavailable metrics display `...` until enough clean signal is available.
The terminal also prints periodic vitals summaries; tune with
`--vitals-log-every` or disable with `--vitals-log-every 0`.
For focused diagnostics, use throttled module debug logs, e.g.
`python main.py --debug-modules pipeline,openrppg,clothing,drowsiness,deepface,weather`.

For accuracy testing, record the same clean clip while wearing a pulse oximeter,
watch, or chest strap, then benchmark candidate models:
```bash
python tests/benchmark_rppg_models.py --source clip.mp4 --reference-bpm 72 \
  --models default physformer efficientphys --csv rppg_benchmark.csv
```
Pick the model with the lowest MAE, acceptable confidence coverage, and usable
latency. HRV and BVP-derived breathing are only emitted after a longer clean
window, and may show lower confidence, because they are less reliable than HR
from webcam video.

DeepFace emotion runs in a separate TensorFlow subprocess so it can coexist with
Open-RPPG's required JAX backend in the main process.

### Clothing + weather recommendations
The `weather`, `clothing`, and `clothing_advice` modules combine Open-Meteo
weather data with visible upper-body clothing detection. Weather uses Open-Meteo
and needs no API key; set `latitude`/`longitude` in `config/modules.yaml` or set
`WEATHER_LAT` / `WEATHER_LON`. Clothing detection uses optional OWL-ViT
zero-shot detection (`requirements-clothing.txt`) with labels such as `hoodie`,
`t-shirt`, `tank top`, `jacket`, and `coat`. Without those optional packages,
the dashboard shows a clear install/status message instead of guessing. OWL-ViT
loads and runs in a background worker so first-time model download/inference does
not freeze the camera loop.

## Multimodal showcase

The showcase also supports guided assessments, shared microphone intelligence,
anonymous multi-person tracks, optional NVIDIA room understanding, routine
correlation, optional/simulated sensors, and deterministic replay. Cloud paths
are independent and opt-in for each run:

```bash
python main.py --enable-cloud-skin                 # skin stills only
python main.py --enable-cloud-scene                # room stills only
python main.py --assessment arm_drift              # one guided protocol
python main.py --source replay:kitchen_spill --headless
pip install -r requirements-audio-events.txt       # optional YAMNet backend
```

Available assessment names are `sit_to_stand`, `timed_up_and_go`, `arm_drift`,
`finger_tapping`, `balance`, `guided_gait`, `facial_movement`, `read_aloud`, and
`guided_breathing`. Replay scenario names are listed in
`config/replay_scenarios.json`. The web dashboard includes capability/consent
status, workflow progress, confidence and quality, and a privacy-filtered event
timeline. Raw frames, audio, embeddings, and private hypotheses are never stored.
See [docs/MULTIMODAL_SHOWCASE.md](docs/MULTIMODAL_SHOWCASE.md) for complete
Phase 1â€“7 consent, replay, assessment, tracking, routine, and hardware setup.

RGB video never supplies reliable blood pressure, SpO2, temperature, or weight;
those values are presented as measurements only when an appropriate sensor
adapter provides them. All outputs remain non-diagnostic, and urgent alerts are
deterministic rather than generated by an LLM or VLM.

## Architecture
```
Camera ─▶ Extractors ─▶ Scheduler(modules) ─▶ Aggregator ─▶ Greeting / Overlay
          (face, pose,     runs each module      latest signal
           motion — once   at its own cadence    per (module,key)
           per frame)      when inputs present
```
- **`core/`** — camera, per-frame `FrameContext`, `Result` schema, module
  registry, cadence-aware scheduler, pipeline orchestrator.
- **`extractors/`** — expensive shared steps (MediaPipe FaceLandmarker /
  PoseLandmarker / motion) run **once** per frame and cached in the context.
- **`modules/`** — one file per detection; each is a `DetectionModule` that
  reads the context and returns `Result`s. Self-registered via `@register`.
- **`output/`** — aggregator, rule-based greeting engine, debug overlay.
- **`storage/`** — SQLite history for longitudinal signals (activity, presence,
  grooming).
- **`config/modules.yaml`** — enable/disable and tune every module.

### Opt-in NVIDIA skin screening

Set `NVIDIA_API_KEY` in `.env`, then add `--enable-cloud-skin` for each run in
which the person has consented to upload occasional camera stills. Without that
flag, the `skin_vision` module never makes a network request even when a key is
configured. The model and endpoint are configurable under `skin_vision` in
`config/modules.yaml`.

Screening runs in the background about once per minute. A possible whole-frame
observation causes the companion to request a closer view; only the close-up
can create a public, non-diagnostic skin-change result. Possible condition
names remain private agent context used to phrase follow-up questions. A
speech guard replaces any generated line that names or implies a condition
with a reviewed neutral fallback. Image results alone never trigger caregiver
alerts.

## Add or change a module
1. Drop `modules/my_thing.py`:
   ```python
   from core.registry import register
   from modules.base import DetectionModule

   @register("my_thing")
   class MyThing(DetectionModule):
       interval = 1.0            # run at most once a second
       requires = ("face",)      # skipped when no face in frame

       def process(self, ctx):
           return self.result("my_key", value=..., message="...")
   ```
2. Add `my_thing: { enabled: true }` to `config/modules.yaml`.

No edits to the pipeline, scheduler, or output layer are ever needed — that's
the point of the design.

## Test without a webcam
```bash
python tests/smoke_test.py          # full pipeline on synthetic frames
python tests/make_clip.py           # write a synthetic video
python main.py --source tests/synthetic_clip.mp4 --headless --max-frames 80
```

## Module catalogue (35)
Vitals: `heart_rate` (+HRV), `respiration`.
Clothing/weather: `weather`, `clothing`, `clothing_advice`.
Skin/face: `skin_color` (pallor/flushing/cyanosis/jaundice; chroma samples are
temporally pooled — see below), `rash`, `bruise`, `eye_redness`, `sweating`,
`dry_lips`, `skin_vision` (opt-in NVIDIA whole-body screen + guided close-up),
`facial_asymmetry` (regional: mouth/eye/brow/cheek-edge, each with
its own baseline — only mouth/eye can escalate to ALERT), `facial_swelling`
(features pooled the same way to damp small-face landmark jitter).
Motor/neuro: `tremor`, `gait`, `balance`, `bradykinesia`, `masked_face`,
`eye_movement`.
Fatigue/safety: `drowsiness` (PERCLOS/blink/microsleep), `yawn`, `head_nod`,
`fall`, `unresponsive`, `wandering`, `hazard_zones`.
Emotion/cognition: `emotion`, `pain`, `agitation`.
Behavior/demographic: `activity_level`, `presence`, `grooming` (hair/facial-hair
texture drift vs. a trailing weekly baseline — longitudinal, needs history to
be meaningful), `age_estimation`, `body_estimate`.

### Accuracy notes: small/compressed cameras
Two changes specifically target low-cost or distant cameras (e.g. a 2 MP USB
2.0 webcam at 1–1.5 m), where MJPEG chroma subsampling and a small face in
frame otherwise degrade color and geometry signals:
- **Temporal color pooling** (`modules/_util.py::pooled_skin_sample`, used by
  `skin_color` and `facial_swelling`) averages a near-stationary subject's
  sample over a short rolling window before any threshold check, recovering
  signal-to-noise lost to chroma subsampling / landmark jitter at the cost of
  a few seconds of lag.
- **Regional facial asymmetry** (`facial_asymmetry`) scores mouth, eye, brow,
  and cheek/face-edge symmetry independently instead of averaging them into
  one number, so a droop isolated to one FAST-relevant region (mouth/eye)
  isn't diluted by an unrelated symmetric region.

## Privacy note
The default pipeline and SQLite store keep only numeric signal summaries, not
video. `skin_vision` is the explicit exception: with `--enable-cloud-skin`, it
sends sampled in-memory JPEG stills to the configured NVIDIA endpoint and does
not save them locally. Obtain informed consent before enabling it; camera
imagery and inferred health information are sensitive data.
