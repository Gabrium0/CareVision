"""Measure live localhost rPPG output against manual wearable checkpoints.

Examples:
    python tests/live_rppg_calibration.py --duration 60 --reference-bpm 61
    python tests/live_rppg_calibration.py --duration 60 --checkpoint 0:61 --checkpoint 30:62 --checkpoint 60:61 --output tracked.csv

Only JSON summaries from /debug/state are read. No image, identity, or raw
physiological waveform is requested or persisted.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.request
from urllib.parse import urlparse


def _parse_checkpoint(value: str) -> tuple[float, float]:
    try:
        seconds, bpm = (float(part) for part in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("checkpoint must be SECOND:BPM") from exc
    if seconds < 0 or not 35.0 <= bpm <= 220.0:
        raise argparse.ArgumentTypeError("checkpoint requires seconds >= 0 and BPM 35..220")
    return seconds, bpm


def _reference_at(checkpoints: list[tuple[float, float]], elapsed: float) -> float:
    ordered = sorted(checkpoints)
    if elapsed <= ordered[0][0]:
        return ordered[0][1]
    if elapsed >= ordered[-1][0]:
        return ordered[-1][1]
    for (left_t, left_bpm), (right_t, right_bpm) in zip(ordered, ordered[1:]):
        if left_t <= elapsed <= right_t:
            weight = (elapsed - left_t) / max(right_t - left_t, 1e-9)
            return left_bpm + weight * (right_bpm - left_bpm)
    return ordered[-1][1]


def _fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - localhost CLI
        return json.load(response)


def _localhost_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise argparse.ArgumentTypeError("debug URL must be an HTTP loopback address")
    return value


def _sample(payload: dict, elapsed: float,
            checkpoints: list[tuple[float, float]]) -> list[dict]:
    vitals = (payload.get("system") or {}).get("vitals") or payload.get("vitals") or {}
    mode = (vitals.get("fast_path") or {}).get("mode")
    rows = []
    for backend in vitals.get("backends") or []:
        measurement = backend.get("measurement") or {}
        bpm = measurement.get("bpm")
        reference = _reference_at(checkpoints, elapsed)
        accepted = bool(measurement.get("accepted") and bpm is not None)
        rows.append({
            "elapsed_seconds": round(elapsed, 3), "mode": mode,
            "backend": backend.get("name"), "wearable_bpm": round(reference, 2),
            "bpm": bpm, "absolute_error_bpm": (round(abs(float(bpm) - reference), 2)
                                                if accepted else None),
            "accepted": accepted, "confidence": measurement.get("confidence"),
            "quality": measurement.get("quality"),
            "sample_hz": backend.get("effective_sample_hz"),
            "inference_sample_hz": backend.get("inference_sample_hz"),
            "inference_latency_ms": backend.get("inference_latency_ms"),
            "state": measurement.get("state"),
            "rejection_reason": measurement.get("rejection_reason"),
        })
    return rows


def _summaries(rows: list[dict]) -> list[dict]:
    summaries = []
    for backend in sorted({row["backend"] for row in rows if row.get("backend")}):
        selected = [row for row in rows if row["backend"] == backend]
        accepted = [row for row in selected if row["accepted"]]
        errors = [row["absolute_error_bpm"] for row in accepted]
        coverage = 100.0 * len(accepted) / len(selected) if selected else 0.0
        summaries.append({
            "backend": backend, "samples": len(selected), "accepted": len(accepted),
            "coverage_pct": round(coverage, 1),
            "mae_bpm": round(statistics.fmean(errors), 2) if errors else None,
            "passes_mae": bool(errors and statistics.fmean(errors) <= 5.0),
            "passes_coverage": coverage >= 70.0,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-url", type=_localhost_url,
                        default="http://127.0.0.1:8771/debug/state")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--reference-bpm", type=float)
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, default=[])
    parser.add_argument("--output", help="Optional normalized CSV output")
    args = parser.parse_args()
    checkpoints = list(args.checkpoint)
    if args.reference_bpm is not None:
        checkpoints = [(0.0, float(args.reference_bpm)),
                       (float(args.duration), float(args.reference_bpm))]
    if not checkpoints:
        parser.error("provide --reference-bpm or one or more --checkpoint SECOND:BPM")
    if len(checkpoints) == 1:
        checkpoints.append((float(args.duration), checkpoints[0][1]))

    rows: list[dict] = []
    started = time.monotonic()
    next_sample = started
    while True:
        now = time.monotonic()
        elapsed = now - started
        if elapsed > args.duration:
            break
        if now >= next_sample:
            rows.extend(_sample(_fetch(args.debug_url), elapsed, checkpoints))
            next_sample += max(0.1, args.interval)
        time.sleep(min(0.05, max(0.0, next_sample - time.monotonic())))

    summaries = _summaries(rows)
    print(json.dumps({"checkpoints": checkpoints, "summaries": summaries}, indent=2))
    if args.output:
        fields = list(rows[0]) if rows else []
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
