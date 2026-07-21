"""Sequentially benchmark production rPPG backends against wearable BPM.

Examples:
    python tests/benchmark_rppg_models.py --source clip.mp4 --reference-bpm 72
    python tests/benchmark_rppg_models.py --source clip.mp4 --reference-csv wearable.csv --csv out.csv

The reference CSV accepts ``timestamp``/``time``/``seconds`` plus
``bpm``/``heart_rate``/``hr``. Timestamped values are linearly interpolated
at each backend reading; a BPM-only CSV falls back to its mean. Model variants
run one after another so they do not compete for camera or accelerator resources.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from modules.rppg_backends.classical import ClassicalBackend
from modules.rppg_backends.openrppg import OpenRPPGBackend


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _load_reference(args) -> list[tuple[float | None, float]]:
    injected = getattr(args, "reference_points", None)
    if injected is not None:
        return list(injected)
    if args.reference_bpm is not None:
        return [(None, float(args.reference_bpm))]
    if args.reference_csv is None:
        return []
    points: list[tuple[float | None, float]] = []
    with open(args.reference_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("bpm") or row.get("heart_rate") or row.get("hr")
            if not value:
                continue
            stamp = row.get("timestamp") or row.get("time") or row.get("seconds")
            points.append((float(stamp) if stamp not in (None, "") else None,
                           float(value)))
    timed = [(stamp, bpm) for stamp, bpm in points if stamp is not None]
    return sorted(timed) if timed and len(timed) == len(points) else points


def _ubfc_reference(path: Path) -> list[tuple[float, float]]:
    rows = [[float(value) for value in line.split()]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 3:
        raise ValueError(f"UBFC ground truth must have three numeric rows: {path}")
    heart_rate, timestamps = rows[1], rows[2]
    count = min(len(heart_rate), len(timestamps))
    if count < 2:
        raise ValueError(f"UBFC ground truth is too short: {path}")
    origin = timestamps[0]
    points = [(timestamps[index] - origin, heart_rate[index]) for index in range(count)
              if 35.0 <= heart_rate[index] <= 220.0]
    if len(points) < 2:
        raise ValueError(f"UBFC ground truth has no usable heart-rate row: {path}")
    return points


def _pure_reference(path: Path) -> list[tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    packages = payload.get("/FullPackage") or payload.get("FullPackage") or []
    raw: list[tuple[float, float]] = []
    for package in packages:
        value = package.get("Value") or package.get("value") or {}
        bpm = (value.get("pulseRate") or value.get("pulse_rate") or
               value.get("heartRate") or value.get("heart_rate"))
        stamp = package.get("Timestamp", package.get("timestamp"))
        if bpm is not None and stamp is not None and 35.0 <= float(bpm) <= 220.0:
            raw.append((float(stamp), float(bpm)))
    if len(raw) < 2:
        raise ValueError(f"PURE JSON has no timestamped pulseRate values: {path}")
    raw.sort()
    origin = raw[0][0]
    # PURE timestamps may be nanoseconds; infer a scale from the median step.
    steps = np.diff([stamp for stamp, _ in raw])
    median_step = float(np.median(steps)) if len(steps) else 1.0
    scale = 1e-9 if median_step > 1e6 else (1e-3 if median_step > 10 else 1.0)
    return [((stamp - origin) * scale, bpm) for stamp, bpm in raw]


def _dataset_items(name: str, root: str, max_recordings: int = 0) -> list[dict]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    items: list[dict] = []
    if name == "ubfc":
        for video in sorted(base.rglob("vid.avi")):
            ground_truth = video.with_name("ground_truth.txt")
            if ground_truth.exists():
                items.append({"recording": video.parent.name, "source": str(video),
                              "reference_points": _ubfc_reference(ground_truth)})
    elif name == "pure":
        for metadata_path in sorted(base.rglob("*.json")):
            recording = metadata_path.stem
            image_dir = metadata_path.parent
            if any(image_dir.glob("*.png")) or any(image_dir.glob("*.jpg")):
                items.append({"recording": recording, "source": str(image_dir),
                              "reference_points": _pure_reference(metadata_path)})
    if max_recordings:
        items = items[:max_recordings]
    if not items:
        raise ValueError(f"No supported {name.upper()} recordings found under {base}")
    return items


def _reference_at(points: list[tuple[float | None, float]], seconds: float) -> float | None:
    if not points:
        return None
    if any(stamp is None for stamp, _ in points):
        return _mean([bpm for _, bpm in points])
    times = np.asarray([stamp for stamp, _ in points], dtype=np.float64)
    bpms = np.asarray([bpm for _, bpm in points], dtype=np.float64)
    return float(np.interp(seconds, times, bpms))


def _wait_openrppg(backend: OpenRPPGBackend, timeout: float,
                   require_result: bool = False) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        diag = backend.diagnostics()
        if require_result and not diag.get("inference_pending"):
            reading = backend.latest_reading()
            if reading is not None:
                return reading
        if not require_result and (diag.get("worker_state") == "ready" or
                                   not diag.get("available", True)):
            return backend.latest_reading()
        time.sleep(0.05)
    return None


def _make_backend(args, backend_name: str, variant: str):
    if backend_name == "classical":
        return ClassicalBackend(window_seconds=args.window_seconds, method=variant,
                                smoothing_window=args.smoothing_window)
    return OpenRPPGBackend(
        window_seconds=args.window_seconds,
        model=None if variant == "default" else variant,
        infer_every=0.0,
        min_seconds=args.min_seconds,
        hrv_min_seconds=args.hrv_min_seconds,
        min_confidence=args.min_confidence,
        smoothing_window=args.smoothing_window,
        motion_threshold=args.motion_threshold,
        face_jitter_threshold=args.face_jitter_threshold,
        cpu_reserved_cores=args.cpu_reserved_cores,
        result_fresh_seconds=args.result_fresh_seconds,
        async_inference=True,
    )


def _run_backend(args, backend_name: str, variant: str) -> dict:
    backend = _make_backend(args, backend_name, variant)
    label = f"{backend_name}:{variant}"
    if not backend.available:
        return {"backend": backend_name, "variant": variant, "available": False}
    if backend_name == "openrppg":
        _wait_openrppg(backend, args.worker_timeout)
        if not backend.available:
            backend.close()
            return {"backend": backend_name, "variant": variant, "available": False}

    face = FaceExtractor()
    source_path = Path(args.source)
    cap = None
    image_files: list[Path] = []
    if source_path.is_dir():
        image_files = sorted([*source_path.glob("*.png"), *source_path.glob("*.jpg"),
                              *source_path.glob("*.jpeg")])
        if not image_files:
            backend.close()
            raise RuntimeError(f"No images found in {args.source}")
        fps = float(args.fps or 30.0)
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            backend.close()
            raise RuntimeError(f"Could not open {args.source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps or 30.0)
    reference = _load_reference(args)
    readings: list[tuple[float, dict]] = []
    compute_ms: list[float] = []
    prev_gray = None
    frame_index = 0
    next_evaluation = args.min_seconds
    eligible_windows = 0
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
            elif frame_index < len(image_files):
                frame = cv2.imread(str(image_files[frame_index]), cv2.IMREAD_COLOR)
                ok = frame is not None
            else:
                ok, frame = False, None
            if not ok or (args.max_frames and frame_index >= args.max_frames):
                break
            elapsed = frame_index / fps
            ctx = FrameContext(frame=frame, timestamp=elapsed,
                               frame_index=frame_index, fps=fps)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                ctx.motion_energy = float(np.mean(cv2.absdiff(gray, prev_gray)))
            prev_gray = gray
            face.extract(ctx)
            backend.update(ctx)
            if elapsed + 1e-9 >= next_evaluation:
                eligible_windows += 1
                start = time.perf_counter()
                reading = backend.compute()
                if backend_name == "openrppg":
                    reading = _wait_openrppg(backend, args.worker_timeout,
                                             require_result=True)
                compute_ms.append((time.perf_counter() - start) * 1000.0)
                if reading:
                    readings.append((elapsed, dict(reading)))
                next_evaluation += args.evaluation_stride_seconds
            frame_index += 1
    finally:
        if cap is not None:
            cap.release()
        face.close()
        backend.close()

    accepted = [(seconds, reading) for seconds, reading in readings
                if reading.get("bpm") is not None and
                float(reading.get("confidence", 0.0)) >= args.min_confidence]
    errors = [abs(float(reading["bpm"]) - ref)
              for seconds, reading in accepted
              if (ref := _reference_at(reference, seconds)) is not None]
    confidences = [float(reading.get("confidence", reading.get("raw_confidence", 0.0)))
                   for _, reading in readings]
    qualities = [float(reading["quality"]) for _, reading in readings
                 if reading.get("quality") is not None]
    valid_bpms = [float(reading["bpm"]) for _, reading in accepted]
    coverage = 100.0 * len(accepted) / eligible_windows if eligible_windows else 0.0
    return {
        "backend": backend_name, "variant": variant, "label": label,
        "dataset": getattr(args, "dataset", None),
        "recording": getattr(args, "recording", Path(args.source).stem),
        "available": True, "frames": frame_index,
        "eligible_windows": eligible_windows, "accepted_readings": len(accepted),
        "accepted_coverage_pct": round(coverage, 1),
        "mean_bpm": _mean(valid_bpms), "median_bpm": _median(valid_bpms),
        "mae_bpm": _mean(errors), "median_abs_error_bpm": _median(errors),
        "meets_mae_target": (_mean(errors) is not None and
                             _mean(errors) <= args.target_mae_bpm),
        "meets_coverage_target": coverage >= args.target_coverage_pct,
        "mean_confidence": _mean(confidences), "mean_quality": _mean(qualities),
        "mean_compute_ms": _mean(compute_ms), "median_compute_ms": _median(compute_ms),
    }


def _write_csv(path: str, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict], backend: str, variant: str,
                    target_mae: float = 5.0, target_coverage: float = 70.0) -> dict:
    usable = [row for row in rows if row.get("available") and
              row.get("backend") == backend and row.get("variant") == variant]
    accepted = sum(int(row.get("accepted_readings") or 0) for row in usable)
    eligible = sum(int(row.get("eligible_windows") or 0) for row in usable)
    error_weight = sum((float(row["mae_bpm"]) * int(row["accepted_readings"]))
                       for row in usable if row.get("mae_bpm") is not None)
    error_count = sum(int(row["accepted_readings"]) for row in usable
                      if row.get("mae_bpm") is not None)
    coverage = 100.0 * accepted / eligible if eligible else 0.0
    return {"backend": backend, "variant": variant, "label": f"{backend}:{variant}",
            "dataset": usable[0].get("dataset") if usable else None,
            "recording": "ALL", "available": bool(usable),
            "eligible_windows": eligible, "accepted_readings": accepted,
            "accepted_coverage_pct": round(coverage, 1),
            "mae_bpm": error_weight / error_count if error_count else None,
            "meets_mae_target": bool(error_count and error_weight / error_count <= target_mae),
            "meets_coverage_target": coverage >= target_coverage}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", help="Video clip or image-sequence directory")
    inputs.add_argument("--dataset", choices=("ubfc", "pure"))
    parser.add_argument("--dataset-root", help="Local dataset root; never modified")
    parser.add_argument("--max-recordings", type=int, default=0)
    parser.add_argument("--backends", nargs="+", choices=("classical", "openrppg"),
                        default=["classical", "openrppg"])
    parser.add_argument("--classical-methods", nargs="+", default=["chrom", "pos"])
    parser.add_argument("--models", nargs="+", default=["default", "physformer", "efficientphys"])
    parser.add_argument("--reference-bpm", type=float)
    parser.add_argument("--reference-csv",
                        help="CSV with timestamp/time/seconds and bpm/hr/heart_rate")
    parser.add_argument("--csv", help="Optional output CSV summary")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--evaluation-stride-seconds", type=float, default=5.0)
    parser.add_argument("--min-seconds", type=float, default=10.0)
    parser.add_argument("--hrv-min-seconds", type=float, default=30.0)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--motion-threshold", type=float, default=18.0)
    parser.add_argument("--face-jitter-threshold", type=float, default=0.12)
    parser.add_argument("--cpu-reserved-cores", default="auto")
    parser.add_argument("--result-fresh-seconds", type=float, default=15.0)
    parser.add_argument("--worker-timeout", type=float, default=120.0)
    parser.add_argument("--target-mae-bpm", type=float, default=5.0)
    parser.add_argument("--target-coverage-pct", type=float, default=70.0)
    args = parser.parse_args()

    if args.dataset:
        if not args.dataset_root:
            parser.error("--dataset-root is required with --dataset")
        sources = _dataset_items(args.dataset, args.dataset_root, args.max_recordings)
    else:
        sources = [{"recording": Path(args.source).stem, "source": args.source,
                    "reference_points": None}]

    rows = []
    variants = []
    if "classical" in args.backends:
        variants.extend(("classical", method) for method in args.classical_methods)
    if "openrppg" in args.backends:
        variants.extend(("openrppg", model) for model in args.models)
    for backend_name, variant in variants:
        for item in sources:
            run_args = argparse.Namespace(**vars(args))
            run_args.source = item["source"]
            run_args.recording = item["recording"]
            run_args.reference_points = item["reference_points"]
            print(f"\n[benchmark] running {backend_name}:{variant} / {item['recording']}")
            row = _run_backend(run_args, backend_name, variant)
            rows.append(row)
            if not row.get("available"):
                print("  unavailable")
                continue
            print(f"  accepted={row['accepted_readings']}/{row['eligible_windows']} "
                  f"coverage={row['accepted_coverage_pct']}% mae={row['mae_bpm']} "
                  f"quality={row['mean_quality']} confidence={row['mean_confidence']}")
    if args.dataset:
        summaries = [_aggregate_rows(rows, backend, variant,
                                     args.target_mae_bpm, args.target_coverage_pct)
                     for backend, variant in variants]
        rows.extend(summaries)
        for summary in summaries:
            print(f"\n[benchmark] summary {summary['label']}: "
                  f"coverage={summary['accepted_coverage_pct']}% mae={summary['mae_bpm']}")
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"\n[benchmark] wrote {args.csv}")


if __name__ == "__main__":
    main()
