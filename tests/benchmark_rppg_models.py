"""Benchmark Open-RPPG models on the same clip against a reference BPM.

Examples:
    python tests/benchmark_rppg_models.py --source clip.mp4 --reference-bpm 72
    python tests/benchmark_rppg_models.py --source clip.mp4 --models default physformer efficientphys --csv out.csv

Use a clean, still, well-lit clip. For HRV/breathing, collect at least 30s.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext
from extractors.face import FaceExtractor
from modules.rppg_backends.openrppg import OpenRPPGBackend


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _pct(values: list[float], threshold: float) -> float:
    return 100.0 * sum(v >= threshold for v in values) / len(values) if values else 0.0


def _load_reference(args) -> float | None:
    if args.reference_bpm is not None:
        return float(args.reference_bpm)
    if args.reference_csv is None:
        return None
    bpms: list[float] = []
    with open(args.reference_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("bpm") or row.get("heart_rate") or row.get("hr")
            if value:
                bpms.append(float(value))
    return _mean(bpms)


def _run_model(args, model_name: str) -> dict:
    model_arg = None if model_name == "default" else model_name
    backend = OpenRPPGBackend(
        window_seconds=args.window_seconds,
        model=model_arg,
        infer_every=args.infer_every,
        min_seconds=args.min_seconds,
        hrv_min_seconds=args.hrv_min_seconds,
        min_confidence=args.min_confidence,
        smoothing_window=args.smoothing_window,
        motion_threshold=args.motion_threshold,
        face_jitter_threshold=args.face_jitter_threshold,
        async_inference=False,
    )
    if not backend.available:
        return {"model": model_name, "available": False}

    face = FaceExtractor()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps or 30.0)

    readings: list[dict] = []
    latencies: list[float] = []
    prev_gray = None
    frame_index = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break

            ts = t0 + frame_index / fps
            ctx = FrameContext(frame=frame, timestamp=ts, frame_index=frame_index, fps=fps)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                ctx.motion_energy = float(np.mean(cv2.absdiff(gray, prev_gray)))
            prev_gray = gray

            face.extract(ctx)
            backend.update(ctx)
            start = time.perf_counter()
            reading = backend.compute()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if reading:
                latencies.append(elapsed_ms)
                readings.append(reading)
            frame_index += 1
    finally:
        cap.release()
        face.close()
        backend.close()

    bpms = [float(r["bpm"]) for r in readings if "bpm" in r]
    confs = [float(r.get("confidence", 0.0)) for r in readings]
    valid_bpms = [float(r["bpm"]) for r in readings
                  if "bpm" in r and float(r.get("confidence", 0.0)) >= args.min_confidence]
    reference = _load_reference(args)
    errors = [abs(v - reference) for v in valid_bpms] if reference is not None else []

    return {
        "model": model_name,
        "available": True,
        "frames": frame_index,
        "readings": len(readings),
        "valid_readings": len(valid_bpms),
        "mean_bpm": _mean(valid_bpms),
        "median_bpm": _median(valid_bpms),
        "reference_bpm": reference,
        "mae_bpm": _mean(errors),
        "median_abs_error_bpm": _median(errors),
        "mean_confidence": _mean(confs),
        "usable_confidence_pct": _pct(confs, args.min_confidence),
        "mean_compute_ms": _mean(latencies),
        "median_compute_ms": _median(latencies),
        "hrv_outputs": sum("hrv_rmssd_ms" in r or "hrv_sdnn_ms" in r for r in readings),
        "breathing_outputs": sum("breaths_per_min" in r for r in readings),
        "raw_bpm_count": len(bpms),
    }


def _write_csv(path: str, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video clip path")
    parser.add_argument("--models", nargs="+", default=["default", "physformer", "efficientphys"])
    parser.add_argument("--reference-bpm", type=float, help="Reference BPM for the clip")
    parser.add_argument("--reference-csv", help="CSV with bpm/hr/heart_rate column; mean is used")
    parser.add_argument("--csv", help="Optional output CSV summary")
    parser.add_argument("--fps", type=float, help="Fallback FPS if the video does not report one")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--infer-every", type=float, default=2.0)
    parser.add_argument("--min-seconds", type=float, default=10.0)
    parser.add_argument("--hrv-min-seconds", type=float, default=30.0)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--motion-threshold", type=float, default=18.0)
    parser.add_argument("--face-jitter-threshold", type=float, default=0.12)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        print(f"\n[benchmark] running {model}")
        row = _run_model(args, model)
        rows.append(row)
        if not row.get("available"):
            print(f"  {model}: unavailable")
            continue
        print(
            f"  valid={row['valid_readings']}/{row['readings']} "
            f"median_bpm={row['median_bpm']} mae={row['mae_bpm']} "
            f"conf={row['mean_confidence']} latency_ms={row['median_compute_ms']}"
        )

    if args.csv:
        _write_csv(args.csv, rows)
        print(f"\n[benchmark] wrote {args.csv}")


if __name__ == "__main__":
    main()
