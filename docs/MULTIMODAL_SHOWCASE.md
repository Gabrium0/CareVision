# Multimodal showcase

This guide describes showcase behavior. Use [OPERATIONS.md](OPERATIONS.md) for
environment setup and ports, `main.py --help` for current flags, and
`config/replay_scenarios.json` for the authoritative replay catalogue.

This project is a non-diagnostic demonstration. Generative models phrase
approved questions; deterministic code selects assessments and alerts. Raw
camera frames, microphone samples, depth maps, thermal frames, embeddings, and
agent-only hypotheses are never written to the event database or dashboard.

## Consent and launch options

Skin and room analysis have separate, per-run upload consent:

```powershell
python main.py --enable-cloud-skin
python main.py --enable-cloud-scene
python main.py --enable-cloud-skin --enable-cloud-scene
```

`NVIDIA_API_KEY` may remain configured when both flags are absent; no cloud
request is scheduled. Scene analysis uses three spaced frames by default and a
single in-flight request. Change `window_frames`, `window_spacing`, model,
endpoint, or intervals in `config/modules.yaml`.

Anonymous multi-person support is opt-in with `--enable-multi-person`. The first
unambiguous track stable for three frames becomes the primary subject and stays
primary while visitors move closer or appear larger. If it is lost, health
modules pause until another track becomes stably assignable. There is no face
recognition or identity database.

## Assessments

Start a protocol with `--assessment NAME` or a replay scenario. Available
protocols are sit-to-stand, Timed Up and Go, arm drift, finger tapping, balance,
guided gait, facial movement, read aloud, and guided breathing. Each follows
instruction, positioning, in-memory sampling, deterministic scoring, up to three
approved questions, and a neutral conclusion. One quality retry is allowed. A
denial suppresses the topic and an unclear answer is rephrased once. Without a
microphone, the measured result concludes without questions.

## Deterministic replay

Scenarios are declared in `config/replay_scenarios.json`. Assessment scenarios
generate pose or face trajectories and pass them through the same modules,
workflow engine, aggregator, and agent used live. Sensor records use a
`ReplaySensorAdapter`; answers use the listener interface; optional WAV fixtures
are published on the shared `AudioBus`.

```powershell
python main.py --source replay:sit_to_stand --headless --fps 5
python main.py --source replay:visitor --enable-multi-person
python main.py --source replay:wearable_thermal --headless
```

## Guest/client demo mode

The recommended way to show the pipeline to a guest or client:

```powershell
python main.py --source replay:client_demo --webui
```

That runs a ~50-second deterministic scenario spanning vitals, skin, motor,
fatigue, and a cough episode, and serves the demo page — a reliable fallback
when a live camera isn't available or lighting is poor. `--demo` (or the
`'d'` hotkey at any point during a run) instead queues a live guided circuit
— facial movement, arm drift, then balance — narrated as each step starts,
using the same `WorkflowEngine`/`request_test` machinery as the manual
`'t'`/`'a'` hotkeys; it is ignored if that subject already has an assessment
running. The two compose: reach for `--demo` on a live camera, the replay
reel above otherwise.

`--webui` serves `/demo`: a big-screen page for a TV or projector rather
than a caregiver. Its header derives registered and running module counts from
the current runtime rather than embedding a catalogue number. Below it sits the
full detector roster, always on
screen: each entry shows a plain-English name and one-line description
instead of a raw module slug, dim while it runs quietly and lit for a
moment when it fires. A live event stream, driven by the same event
timeline as `/data`, appends observations as they arrive and shows
confidence as high, moderate, or tentative rather than a raw number. A
progress bar tracks replay runs, a beat checklist marks off signal families
as they're touched, and a moment card holds each notable event on screen
for a few seconds so a viewer glancing over doesn't miss it.

For an interactive live-camera showcase, `/demo` also offers a **"Try a guided
check" picker**: a touch-friendly button for each guided assessment (plus a
full-circuit button) that starts it on the current subject via the same
`request_test`/`start_demo_circuit` machinery as the `'t'`/`'a'`/`'d'` hotkeys.
The buttons POST to a narrow local `/assessment-control` endpoint and are
disabled while a check is already running. Drive them from a phone or tablet on
the same network pointed at `/demo` while a projector mirrors the big view — the
projector itself need not be a touchscreen. While a check runs, a large
**coaching banner** shows the current framing/positioning guidance ("Line
yourself up — please step back so both arms are fully in view") so a person can
follow the circuit self-serve without a presenter reading JSON. Picker labels
are observational activity names only, enforced by `tests/demo_labels_test.py`.

The web dashboard exposes pause, resume, restart, and speed controls for replay.
It also shows capabilities, consent, assessment progress, confidence versus
quality, subject assignment, ambiguity, and a privacy-filtered causal timeline.

## Audio, routines, and reasoning

The live microphone is opened once. Faster-whisper and YAMNet consume the same
bounded in-memory bus. Timing summaries cover duration, rate, pauses, response
latency, turn gaps, interruptions, and change from the person's own baseline.
Health-related sounds require repetition or corroboration. Smoke alarms and an
explicit recognized call for help are deterministic non-medical safety paths.

Use `python main.py --detect-cough` for opt-in cough-episode detection without
installing Whisper. Install `requirements-audio-events.txt` first; YAMNet may
download its model on first use. `--listen` and `--detect-cough` share one
16 kHz mono microphone stream when enabled together. Only counted event
summaries are persisted; raw microphone samples remain in bounded memory.

Set `camera.location` to `living_room`, `kitchen`, `entryway`, or `bedroom`.
Subject-specific history learns observed presence, occupancy, meal/drink
opportunities, activity, leaving/returning, conversation, mobility-aid
visibility, and sit-to-stand timing. It never claims eating, medication
adherence, or a diagnosis. Daily summaries contain meaningful changes only.

## Optional sensors

```powershell
pip install -r requirements-sensors.txt
pip install -r requirements-realsense.txt
```

Supported adapters include RealSense depth/IMU summaries; BLE heart-rate,
pulse-oximeter, smart-scale, and blood-pressure GATT characteristics; MLX90640
thermal summaries; serial JSON temperature, humidity, CO2, air quality, pollen,
and ambient-light hubs; GPIO door, bed-pressure, and appliance states; and a
deterministic simulation/replay adapter for each class.

Example simulated sensor:

```yaml
sensors:
  pulse_oximeter:
    simulated: true
    interval: 5
    readings:
      - {key: spo2_pct, value: 97, unit: "%", quality: 1.0}
```

Example serial environment hub:

```yaml
sensors:
  environment:
    enabled: true
    port: COM5
    fields:
      temperature: {key: ambient_temperature_c, unit: C}
      humidity: {key: humidity_pct, unit: "%"}
      co2: {key: co2_ppm, unit: ppm}
      pollen: {key: pollen_index, unit: index}
      light: {key: ambient_light_lux, unit: lux}
```

Blood pressure, SpO2, temperature, and weight are measurements only when their
source is an enabled sensor, an explicitly labeled simulation, or replay sensor.
RGB body build remains a categorical proxy and is never weight or BMI.
