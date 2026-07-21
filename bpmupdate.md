# BPM Update: Live rPPG Sampling and Wearable Validation

This document records the heart-rate sampling fix, its safety constraints, and
the validation procedure for the Classical and Open-rPPG backends. Camera heart
rate remains a non-diagnostic wellness estimate.

## Observed baseline

On 2026-07-17, a seated, silent subject reported **61 BPM** on a wearable while
the live RealSense pipeline produced:

| Mode | Wearable | Classical | Open-rPPG | Classical quality | Effective input | Result |
|---|---:|---:|---:|---:|---:|---|
| Strict, 250 ms | 61 | 49–69 | ~50 | 0.00 | 1–2 Hz | Open-rPPG rejected; Classical unstable |

The camera itself delivered approximately 30 FPS. The bottleneck was the
250 ms face-geometry lease: MediaPipe refreshed geometry only around 1.6 FPS,
so most camera frames were rejected as stale before reaching either rPPG
buffer. At 1–2 Hz, the 0.7–3 Hz pulse band is undersampled and FFT/model tuning
would fit aliasing rather than pulse. No personal additive BPM correction is
used.

## Fast-path modes

Select one mode at startup:

```bash
python main.py --debug-endpoint --vitals-fast-path-mode strict
python main.py --debug-endpoint --vitals-fast-path-mode extended
python main.py --debug-endpoint --vitals-fast-path-mode tracked
```

- `strict` retains the existing 250 ms geometry lease. This remains the default
  until tracked mode passes the local wearable and movement-safety trials.
- `extended` permits a stationary geometry lease up to 750 ms. Current frame
  motion must remain below the showcase safety threshold.
- `tracked` uses low-resolution optical flow between authoritative MediaPipe
  detections. It validates feature count, affine inliers, translation, scale,
  motion, and a maximum 1.5-second anchor age. Failure pauses sampling until a
  fresh MediaPipe anchor arrives.

`http://127.0.0.1:8771/debug/state` exposes the selected mode, geometry lease,
tracker status, anchor age, tracked sample rate, failure count, and last
rejection reason. It does not expose frames, tracking points, landmarks, raw
rPPG arrays, embeddings, or spectrograms.

## Measurement acceptance

Classical consumes the full safe capture cadence, with a target of at least
20 samples/second. Open-rPPG uniformly selects frames from the clean timestamped
buffer instead of treating irregular frames as evenly spaced:

- CPU worker: approximately 8 inference frames/second.
- GPU worker: up to 20 inference frames/second.

The debug state reports capture rate and Open-rPPG inference rate separately.
Confidence describes the backend estimate; quality describes lighting,
effective cadence, timing regularity, and accepted-frame coverage. A backend
candidate with `quality < 0.35` cannot become canonical, even if it reports a
numeric BPM. Rejected candidates remain visible on the localhost debug page.
Open-rPPG keeps its SQI threshold, completion-based throttle, and stale-result
expiry; a failing model remains comparison-only rather than being made trusted
by lowering its threshold.

## Live wearable calibration

Restart the application for each mode, remain seated and silent, and record the
wearable at the beginning, 30 seconds, and 60 seconds. The calibration tool
reads only JSON summaries from localhost.

Constant reference example:

```bash
python tests/live_rppg_calibration.py --duration 60 --reference-bpm 61 --output strict.csv
```

Interpolated checkpoints example:

```bash
python tests/live_rppg_calibration.py --duration 60 \
  --checkpoint 0:61 --checkpoint 30:62 --checkpoint 60:61 \
  --output tracked.csv
```

Run strict, extended, and tracked trials, then briefly turn the head during an
additional tracked trial. The movement trial must pause or reject sampling; it
must not publish a trusted value from an unsafe ROI.

| Mode | Wearable BPM | Backend | Mean BPM | MAE | Confidence | Quality | Coverage | Sample Hz | Latency ms | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Strict | 61 | Classical | pending | pending | pending | pending | pending | 1–2 baseline | 0 | Baseline |
| Strict | 61 | Open-rPPG | ~50 baseline | ~11 | 0.13–0.19 | 0.05–0.06 | 0% accepted | 1–2 baseline | ~6500 | Rejected |
| Extended | pending | Classical | pending | pending | pending | pending | pending | pending | 0 | Pending trial |
| Extended | pending | Open-rPPG | pending | pending | pending | pending | pending | pending | pending | Pending trial |
| Tracked | pending | Classical | pending | pending | pending | pending | pending | target ≥20 | 0 | Pending trial |
| Tracked | pending | Open-rPPG | pending | pending | pending | pending | pending | CPU target ~8 | pending | Pending trial |

Acceptance requires no more than **5 BPM MAE** and at least **70% accepted
coverage** in clean trials. Tracked mode becomes the default only after passing
those targets and the deliberate-movement safety trial.

## Public dataset benchmark

Public datasets test general behavior but cannot reproduce this RealSense
sensor, exposure, distance, room lighting, or live scheduling. Download them
under a local directory outside Git; the adapters never download, copy, or
commit biometric video.

UBFC-rPPG:

```bash
python tests/benchmark_rppg_models.py --dataset ubfc \
  --dataset-root C:/datasets/UBFC-rPPG --max-recordings 5 \
  --models default physformer efficientphys --csv ubfc-rppg.csv
```

PURE:

```bash
python tests/benchmark_rppg_models.py --dataset pure \
  --dataset-root C:/datasets/PURE --max-recordings 5 \
  --models default physformer efficientphys --csv pure-rppg.csv
```

Generic synchronized video and wearable CSV remain supported:

```bash
python tests/benchmark_rppg_models.py --source clip.mp4 \
  --reference-csv wearable.csv --csv local-rppg.csv
```

Model variants run sequentially so they do not compete for CPU/GPU resources.
Dataset licensing and access terms remain the responsibility of the person
downloading the data.

## Troubleshooting

- **Low effective sample rate:** inspect `system.vitals.fast_path`. In tracked
  mode, check anchor age and the tracker rejection reason. Reposition the face
  and keep it well inside the frame.
- **Quality near zero:** confirm at least 20 Hz for Classical, stable timestamps,
  brightness within the showcase range, and few rejected frames.
- **Open-rPPG low SQI:** wait for a complete clean window; check inference sample
  rate, model state, device, and latency. Do not lower SQI merely to show BPM.
- **Motion rejection:** remain still and silent. Tracking deliberately stops on
  unsafe motion and resumes after MediaPipe provides a new anchor.
- **CPU inference is slow:** automatic mode reserves half the logical cores for
  capture. The CPU tensor is uniformly limited to about 8 Hz. A CUDA-enabled
  JAX environment can use the higher GPU target.
- **Model unavailable:** install `requirements-openrppg.txt` and inspect the
  worker error in `/debug`; Classical remains available independently.
- **Old BPM remains visible:** check measurement age and freshness. Expired or
  low-quality values must not become canonical.

## Selected default

`strict` remains the production default pending the three local A/B trials.
Update this section and the results table after tracked mode meets accuracy,
coverage, and movement-safety acceptance.
