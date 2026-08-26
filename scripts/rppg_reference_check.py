"""Friendly heart-rate accuracy check: contactless rPPG vs. an Apple Watch.

Point it at a recorded clip of a face and give it the heart rate the wearable
showed at the same time. It runs the production classical rPPG backend over the
clip and prints the detected BPM alongside the reference and the error -- the
one number for the "how accurate is it?" slide.

    python scripts/rppg_reference_check.py --video clip.mp4 --reference-bpm 72

This reuses tests/benchmark_rppg_models.py (the full multi-backend benchmark);
run that directly for neural backends, image sequences, or timestamped CSV
ground truth. Needs the camera/vision stack (cv2, numpy) -- use system python.
"""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.benchmark_rppg_models import _run_backend


def _run_args(video: str, reference_bpm: float | None, fps: float | None,
              window_seconds: float, min_seconds: float) -> Namespace:
    """Assemble the argument bag _run_backend expects for a classical run."""
    return Namespace(
        source=video, reference_bpm=reference_bpm, reference_csv=None,
        reference_points=None, fps=fps, max_frames=0,
        window_seconds=window_seconds, evaluation_stride_seconds=5.0,
        min_seconds=min_seconds, min_confidence=0.35, smoothing_window=5,
        target_mae_bpm=5.0, target_coverage_pct=70.0,
        recording=Path(video).stem, dataset=None)


def main() -> int:
    """Measure rPPG heart rate on a clip and compare it to a wearable reading."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True,
                    help="path to a recorded clip of the person's face")
    ap.add_argument("--reference-bpm", type=float,
                    help="the Apple Watch reading during the clip (ground truth)")
    ap.add_argument("--method", default="chrom", choices=("chrom", "pos", "green"),
                    help="classical rPPG method (default: chrom)")
    ap.add_argument("--fps", type=float,
                    help="override clip FPS if its metadata is wrong")
    ap.add_argument("--window-seconds", type=float, default=30.0)
    ap.add_argument("--min-seconds", type=float, default=10.0)
    args = ap.parse_args()

    if not Path(args.video).exists():
        print(f"[rppg] video not found: {args.video}")
        return 2

    print(f"[rppg] analysing {args.video} with classical:{args.method} ...")
    row = _run_backend(
        _run_args(args.video, args.reference_bpm, args.fps,
                  args.window_seconds, args.min_seconds),
        "classical", args.method)

    if not row.get("available"):
        print("[rppg] backend unavailable (missing rPPG dependencies?)")
        return 1

    mean_bpm = row.get("mean_bpm")
    median_bpm = row.get("median_bpm")
    coverage = row.get("accepted_coverage_pct")
    print("\n  Heart-rate check")
    print("  ----------------")
    print("  detected BPM (mean) : "
          + (f"{mean_bpm:.1f}" if mean_bpm is not None else "n/a"))
    if median_bpm is not None:
        print(f"  detected BPM (median): {median_bpm:.1f}")
    print(f"  readings accepted   : {row.get('accepted_readings')}"
          f"/{row.get('eligible_windows')} windows ({coverage}%)")

    if args.reference_bpm is not None:
        print(f"  Apple Watch (ref)   : {args.reference_bpm:.1f}")
        mae = row.get("mae_bpm")
        if mae is not None:
            print(f"  mean abs. error     : {mae:.1f} BPM"
                  f"  ({'within' if row.get('meets_mae_target') else 'above'} "
                  "the 5 BPM target)")
        elif mean_bpm is not None:
            print(f"  error (mean)        : {abs(mean_bpm - args.reference_bpm):.1f} BPM")
    else:
        print("  (pass --reference-bpm to compare against the watch reading)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
