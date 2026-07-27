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
pip install -r requirements-skin.txt       # local ViT skin classifier + ONNX export
pip install -r requirements-realsense.txt # optional: Intel RealSense D435i source
# NVIDIA CUDA 12.4 clothing backend (keep all PyTorch binaries matched):
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-clothing.txt   # optional: FashionCLIP / OWL-ViT clothing detection
# MediaPipe .task models are downloaded into models/ (see docs/RESEARCH.md)
python main.py                       # default webcam; Camera + Data windows
python main.py --source clip.mp4     # run on a video file
python main.py --combined            # old single-window overlay instead
python main.py --headless --name Margaret   # no window; prints greetings/alerts
python main.py --enable-cloud-skin    # opt in to NVIDIA skin screening
python main.py --source realsense --webui --debug-endpoint --no-moondream  # private debug, no Moondream calls initially
python main.py --source realsense --quality-profile maximum --webui --debug-endpoint  # native-detail quality profile
```
Windowed controls: `q` quit · `g` force a greeting · `m` toggle Moondream API calls ·
`t` start a tremor test · `c` switch camera (see below). Use `--no-moondream` to
guarantee that testing starts with Moondream disabled; templated speech remains active.
Set `X-Moondream-Auth` in `.env` to your Moondream Cloud API key (the conventional
`MOONDREAM_API_KEY` name is also accepted). The credential is sent only to the
Moondream API and is never included in logs or debug diagnostics.

### Hot reload for autonomous development

Run the dependency-free development supervisor to restart the application when
Python, YAML, JSON, HTML, CSS, or JavaScript files change:

```bash
python dev.py
```

The default is hardware-free, headless, private, and safe for unattended testing.
It exposes a readable dashboard at `http://127.0.0.1:8771/debug` and
machine-readable health at `http://127.0.0.1:8771/debug/state`. Stop it with
Ctrl+C when you're done — a supervisor left running keeps holding those ports.
See
[AI development workflow](docs/AI_DEVELOPMENT.md) for the log markers, health
acceptance gate, failure recipes, privacy rules, and real-camera/cloud opt-ins.

Pass any command after `--` to reload a custom application or focused test command:

```bash
python dev.py -- python main.py --source 0 --headless --debug-endpoint --no-moondream
python dev.py -- python -m pytest -q tests/runtime_performance_test.py
python dev.py --watch core --watch tests -- python -m pytest -q tests/pipeline_staleness_test.py
```

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
analysis (`A`) FPS separately. `--face-analysis-width 640` caps authoritative
face detection, while `--analysis-width 960` caps pose and passive analysis.
Landmarks and rPPG crops still use capture coordinates and original pixels.

### Private debugging endpoint

Pass `--debug-endpoint` for a readable, auto-refreshing private dashboard at
`http://127.0.0.1:8771/debug`. Its programmatic JSON is available at
`http://127.0.0.1:8771/debug/state`. Both include complete live result objects,
including agent-only hypotheses and credential-free cloud request counters. Raw
transcripts, credentials, and provider responses are omitted. Use `--debug-port`
to change the port. This is a separate localhost-only server;
the LAN-facing `/data` dashboard remains public-only. Raw image/audio arrays,
binary values, and data URLs are always redacted. The payload also includes
capture/preview/analysis rates, rolling p50/p95/max latency, queue-drop rate,
adaptive scheduler state, selected camera profile, physical capture FPS, bounded
vitals-sampler FPS/latency/drops, rPPG fast-path acceptance, lifecycle-backed
capabilities, and the Whisper worker PID/path-free native-runtime inventory.
The top-level health object reports `healthy`, `degraded`, or `failed` with
reason codes and actions. Its always-visible Vitals panel
shows current BPM when valid, or explains whether positioning is blocked,
samples are warming up, inference is running, or a backend is unavailable.

### Skin screening flow and local-only operation

Skin screening is a guided, close-up workflow rather than a continuous model
call. With `--source realsense`, `core.realsense_camera.RealSenseCamera` yields
color frames with depth aligned to color plus best-effort IMU motion. The normal
pipeline keeps capture, safety modules, and the fast vitals sampler responsive.

At startup, `skin_vision` asynchronously preloads the pinned
`LaurianeMD/vit-skin-disease` model from `runtime-models/huggingface`. The
configured `backend: auto` prefers a device-built TensorRT FP16 engine on an
ARM64 Jetson and falls back to the PyTorch reference backend; on the A2000 it
normally uses PyTorch. A missing cache or failed optional backend degrades skin
screening without stopping the rest of CareVision.

The local classifier does **not** run on passive camera frames. It runs once on
the sharpest frame after either:

- a normal NVIDIA close-up request created by an explicitly consented cloud
  skin analysis, or
- a user-requested arm check (`a` key or “check my arm”), which does not need
  cloud consent.

The close-up is always analyzed locally. When cloud consent is enabled, the
same frame is independently sent to the NVIDIA VLM; a cloud failure does not
discard the local result. The private local result contains the vitiligo target
probability (calibrated only after a validated calibration file) plus all other
Model 1 `label_scores`. Those non-vitiligo scores are raw experimental softmax
values, explicitly uncalibrated, and never become diagnoses, alerts, speech,
history, caregiver data, or public dashboard fields. In the current
`mode: debug`, local results remain agent-only and an uncalibrated model
abstains; `Unknown Normal` is never converted into “normal.”

For an A2000 RealSense run with no paid cloud calls, use:

```powershell
python main.py --source realsense --quality-profile maximum --debug-endpoint --no-moondream --dev-mode
```

Open <http://127.0.0.1:8771/debug>, press `a`, and hold the forearm close and
steady for about ten seconds. Inspect `results[].value.local_classifier` and
its `label_scores` in the private dashboard or `/debug/state`. `--dev-mode`
disables weather and clothing network/model extras; omitting
`--enable-cloud-skin` and `--enable-cloud-scene` keeps both NVIDIA endpoints
off. Do not press `m`, because that explicitly toggles Moondream back on.

The one-time `python tools/cache_skin_model.py` setup command downloads the
pinned weights from Hugging Face. Runtime inference is cache-only. To enable
NVIDIA corroboration later, add `--enable-cloud-skin` only after placing the
key in `.env`; sampled stills may then consume provider quota.

### Local caregiver review portal

Pass `--caregiver-portal` to start the separate, loopback-only review portal at
`http://127.0.0.1:8772/caregiver` (`--caregiver-port` changes the port). It is
never exposed on the LAN and does not require `--webui` or `--debug-endpoint`:

```bash
python main.py --caregiver-portal
```

The portal provides privacy-safe 24-hour, 7-day, and 30-day trends, anonymous
subject filters, durable alert cases, acknowledgement/resolution with a short
caregiver note, JSON/CSV export, and expired-data cleanup. Acknowledgement
pauses ordinary reminders while preserving deterministic escalation for an
active unresolved signal. Raw media, private hypotheses, and biometric
identity are excluded; the portal remains non-diagnostic.

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

See [bpmupdate.md](bpmupdate.md) for the live RealSense sampling fix, fast-path
A/B modes, wearable calibration procedure, public-dataset adapters, and the
current acceptance record.

If readings still look wrong (jumping between backends, low confidence) on a
live webcam specifically, the bounded vitals sampler that feeds the buffers can
be starved if the heavy per-frame detection
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
inference on a rolling buffer of face crops. Live inference runs in a separate
worker process so CPU-only Open-RPPG does not freeze camera frames. In automatic
CPU mode the worker leaves half the logical cores for capture and MediaPipe,
waits about five seconds after an inference completes before starting another,
and stops publishing a neural result after it becomes stale. Confidence reflects
the estimator; the separate `quality` field reflects lighting, effective sample
rate, timing regularity, and accepted-frame coverage.
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
watch, or chest strap. A wearable CSV can contain `seconds,bpm` (also accepted:
`timestamp`/`time` and `heart_rate`/`hr`). The benchmark interpolates the wearable
value at each estimate, runs Classical CHROM/POS and neural model variants
sequentially, and reports MAE plus accepted-window coverage:
```bash
python tests/benchmark_rppg_models.py --source clip.mp4 \
  --reference-csv wearable.csv --models default physformer efficientphys \
  --csv rppg_benchmark.csv
```
The default evaluation targets are at most 5 BPM MAE and at least 70% accepted
coverage; treat those as measurement-session acceptance criteria, not a medical
accuracy claim. HRV and BVP-derived breathing are only emitted after a longer clean
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
```

For opt-in local microphone cough detection, including setup and debug
verification, see [Microphone cough detection](#microphone-cough-detection)
below.

Available assessment names are `sit_to_stand`, `timed_up_and_go`, `arm_drift`,
`finger_tapping`, `balance`, `guided_gait`, `facial_movement`, `read_aloud`, and
`guided_breathing`. Replay scenario names are listed in
`config/replay_scenarios.json`. The web dashboard includes capability/consent
status, workflow progress, confidence and quality, and a privacy-filtered event
timeline. Raw frames, audio, embeddings, and private hypotheses are never stored.
See [docs/MULTIMODAL_SHOWCASE.md](docs/MULTIMODAL_SHOWCASE.md) for complete
Phase 1–7 consent, replay, assessment, tracking, routine, and hardware setup.

For showing the pipeline live to a guest or client:

```bash
python main.py --source replay:client_demo --webui
```

That runs a ~50-second scripted reel and serves the big-screen `/demo` page
— the full detector roster, a `49 detectors · 46 running · ...` stat line,
and a live event stream — alongside the caregiver `/data` view. `--demo`
(or the `'d'` hotkey at any point during a run) instead drives a live
guided circuit through facial movement, arm drift, and balance on a live
camera; the two compose. See
[Guest/client demo mode](docs/MULTIMODAL_SHOWCASE.md#guestclient-demo-mode)
for details.

### Microphone cough detection

Install the optional local audio-event dependencies, then opt in to cough
detection and the localhost debug dashboard:

```bash
pip install -r requirements-audio-events.txt
python main.py --detect-cough --debug-endpoint
```

YAMNet may download its model on first use, so that first launch can require an
internet connection and take longer. Cough detection does not require Whisper
or `faster-whisper`. If `--listen` is also supplied (after installing
`requirements-asr.txt`), speech recognition and the broader sound-event
taxonomy share the same physical 16 kHz mono microphone stream; the device is
not opened twice. Faster-Whisper runs in a spawned CPU worker so CTranslate2's
native libraries stay isolated from CUDA-enabled PyTorch models such as
FashionCLIP. The child installs a Torch import blocker before Faster-Whisper is
loaded because CTranslate2 otherwise probes Torch for optional model-spec
support; do not move that import or model construction back into the camera
process on Windows.

The live pipeline separates resolution by workload: capture and rPPG keep the
original camera pixels, while passive background detectors receive a synchronized
context capped by `--analysis-width` (960 px by default). If that lane falls
behind, the scheduler slows passive skin/clothing/scene work with hysteresis;
fall, unresponsive, vitals, and active assessments are never throttled.
`--quality-profile maximum` is the default: it retains 640 px authoritative face
inference and crops skin, eye, lip, arm, and clothing detail from the original
capture. It reduces cadence and distributes expensive modules across frames
before allowing the guarded 480 px geometry fallback. `balanced` and `realtime`
select progressively smaller detail budgets for constrained machines.
Authoritative face extraction owns a latest-frame worker (640 px normally,
480 px under sustained load), classical rPPG calculations are one-flight, and
busy background frames are reported as intentional coalescing rather than
queue loss. Original capture pixels still feed rPPG.
Moondream generation and advisor queries run off the camera thread, with bounded
timeouts, local speech fallback, backoff, and credential-free circuit diagnostics.
Moondream reports `configured` until its first successful authenticated request.
HTTP 401/403 is latched as `authorization_failed` with no automatic retry; toggle
Moondream off and on after replacing/authorizing the credential. Timeouts, 429,
and server failures retain exponential retry/backoff.

Native OpenCV, TensorFlow, Torch-CPU, BLAS, and tokenizer thread pools are bounded
before model imports so their workers cannot collectively starve MediaPipe or the
vitals sampler. Set `APP_RESPECT_NATIVE_THREAD_ENV=1` only when intentionally
supplying your own thread limits. FashionCLIP explicitly selects `cuda:0` when
CUDA is available; its actual device, load deadline, and first-inference state
appear under `system.clothing` in the private debug payload.

Longitudinal detectors never query SQLite from a detector callback. The history
service bootstraps rolling windows on a separate read connection and exposes
constant-time snapshots; writes remain batched. Aggregate freshness/failures are
reported under `system.history_writer.aggregates`.

Open `http://127.0.0.1:8771/debug` for the readable private dashboard or
`http://127.0.0.1:8771/debug/state` for JSON. A healthy **Audio & cough
detection** panel reports the microphone and YAMNet as `ready`, the detector
worker as alive, and a `windows_processed` count that continues to increase.
The latest and peak cough confidence update as audio windows are classified.
A completed `sound_event.cough_episode` appears in Results and Timeline only
after consecutive qualifying windows confirm a burst and approximately three
seconds of quiet close the episode; nearby bursts are counted together.

Only the completed episode summary—count, timestamps, confidence, quality, and
evidence window—is eligible for event persistence. Raw microphone samples,
embeddings, and spectrograms are neither persisted nor included in debug
telemetry. YAMNet is a non-diagnostic wellness/showcase classifier; a detected
cough-like sound does not identify an illness.

The checked-in real-audio integration benchmark exercises the production
YAMNet path against 24 privacy-minimized Coswara cough and non-cough fixtures:

```bash
python -m pytest -s tests/cough_audio_integration_test.py
```

The test prints per-category detections, maximum cough confidence, episode
counts, recall, and false-positive rate. It skips with an explicit reason when
the optional backend or model is unavailable. Source revision, CC BY 4.0
license, citation, and transformations are documented in the
[fixture attribution notes](tests/fixtures/cough/README.md).

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

See [NVIDIA-Assisted Skin Detection](docs/SKIN_DETECTION.md) for the complete
two-stage flow, structured result schema, customized agent questions, normal
versus debug visibility, live demonstration commands, and failure behavior.

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

## Module catalogue (36)
Vitals: `heart_rate` (+HRV), `respiration`.
Clothing/weather: `weather`, `clothing`, `clothing_advice`.
Skin/face: `skin_color` (pallor/flushing/cyanosis/jaundice; chroma samples are
temporally pooled — see below), `rash`, `bruise`, `arm_skin` (rash/bruise/
dryness/dark-spot screening on bare arms via pose ROIs; depth adds mm sizing
and an `arm_check` guided window — 'a' hotkey or "check my arm"),
`eye_redness`, `sweating`,
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
