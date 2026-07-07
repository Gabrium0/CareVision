# Humanoid Camera — Elderly Care Detection Pipeline

A modular computer-vision pipeline that watches live camera footage and detects
health, safety, and emotional signals to power **customized greetings and
recommendations** for elderly people living alone.

It implements every item in [detectionList.md](detectionList.md) as an
independent, hot-swappable module. See [docs/RESEARCH.md](docs/RESEARCH.md) for
the method and feasibility rating behind each one.

> ⚠️ **Not a medical device.** All health signals are screening prompts to
> trigger a gentle check-in or caregiver alert — never diagnoses.

## Quick start
```bash
pip install -r requirements.txt
# MediaPipe .task models are downloaded into models/ (see docs/RESEARCH.md)
python main.py                       # default webcam, live window
python main.py --source clip.mp4     # run on a video file
python main.py --headless --name Margaret   # no window; prints greetings/alerts
```
Windowed controls: `q` quit · `g` force a greeting.

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
- **`storage/`** — SQLite history for longitudinal signals (activity, presence).
- **`config/modules.yaml`** — enable/disable and tune every module.

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

## Module catalogue (30)
Vitals: `heart_rate` (+HRV), `respiration`.
Skin/face: `skin_color` (pallor/flushing/cyanosis/jaundice), `rash`, `bruise`,
`eye_redness`, `sweating`, `dry_lips`, `facial_asymmetry`, `facial_swelling`.
Motor/neuro: `tremor`, `gait`, `balance`, `bradykinesia`, `masked_face`,
`eye_movement`.
Fatigue/safety: `drowsiness` (PERCLOS/blink/microsleep), `yawn`, `head_nod`,
`fall`, `unresponsive`, `wandering`, `hazard_zones`.
Emotion/cognition: `emotion`, `pain`, `agitation`.
Behavior/demographic: `activity_level`, `presence`, `age_estimation`,
`body_estimate`.

## Privacy note
Designed to run fully on-device. The SQLite store keeps only numeric signal
summaries, not video. If you add cloud features, get informed consent — this is
sensitive health data about a vulnerable person.
