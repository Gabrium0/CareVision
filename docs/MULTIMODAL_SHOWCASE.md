# Multimodal showcase: Phases 1â€“7

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
