#!/usr/bin/env python3
"""Verify PyTorch/TensorRT parity, latency, and Jetson memory headroom."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.local_skin_classifier import (  # noqa: E402
    DEFAULT_MODEL, DEFAULT_REVISION, TensorRTSkinClassifier,
    TorchSkinClassifier, _softmax,
)


def _memory_headroom() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if value:
            values[key] = int(value.strip().split()[0])
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    return available / total if total and available is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path,
                        help="directory containing representative close-up images")
    parser.add_argument("--engine", type=Path,
                        default=Path("runtime-models/vit_skin_orin_fp16.engine"))
    parser.add_argument("--calibration", type=Path,
                        default=Path("assets/skin_models/vit_skin_calibration.json"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("runtime-models/huggingface"))
    parser.add_argument("--max-probability-delta", type=float, default=0.02)
    parser.add_argument("--max-p95-ms", type=float, default=500.0)
    parser.add_argument("--min-memory-headroom", type=float, default=0.25)
    args = parser.parse_args()
    paths = sorted(path for path in args.images.rglob("*")
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise SystemExit("no representative images found")
    common = {
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
        "target_labels": ["vitiligo"], "calibration_file": str(args.calibration),
        "cache_dir": str(args.cache_dir),
        "min_sharpness": 0.0, "min_brightness": 0.0,
        "max_brightness": 255.0, "max_clipped_fraction": 1.0,
    }
    reference = TorchSkinClassifier(device="cpu", download_if_missing=False, **common)
    deployed = TensorRTSkinClassifier(tensorrt_engine=str(args.engine), **common)
    reference.preload()
    deployed.preload()
    failures, latencies, records = [], [], []
    if reference._labels != deployed._labels:
        failures.append("label_mapping_mismatch")
    headroom = None
    try:
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                failures.append(f"unreadable:{path}")
                continue
            reference_pixels = reference._preprocess(image)
            deployed_pixels = deployed._preprocess(image)
            if not np.array_equal(reference_pixels, deployed_pixels):
                failures.append(f"preprocessing_mismatch:{path}")
                continue
            reference_logits = reference._infer_logits(reference_pixels)
            started = time.perf_counter()
            deployed_logits = deployed._infer_logits(deployed_pixels)
            latencies.append((time.perf_counter() - started) * 1000.0)
            temperature = reference._calibration.temperature
            reference_probs = _softmax(reference_logits, temperature)
            deployed_probs = _softmax(deployed_logits, temperature)
            if reference_probs.shape != deployed_probs.shape:
                failures.append(f"output_count_mismatch:{path}")
                continue
            reference_top, deployed_top = int(reference_probs.argmax()), int(deployed_probs.argmax())
            target = int(reference._target_index)
            class_deltas = np.abs(reference_probs - deployed_probs)
            delta = abs(float(reference_probs[target]) - float(deployed_probs[target]))
            max_class_delta = float(class_deltas.max())
            records.append({"image": str(path), "reference_top": reference_top,
                            "deployed_top": deployed_top,
                            "target_probability_delta": round(delta, 8),
                            "max_class_probability_delta": round(max_class_delta, 8)})
            if reference_top != deployed_top:
                failures.append(f"top_label_mismatch:{path}")
            if max_class_delta > args.max_probability_delta:
                failures.append(
                    f"class_probability_delta:{path}:{max_class_delta:.6f}")
        headroom = _memory_headroom()
    finally:
        reference.close()
        deployed.close()
    p95 = float(np.percentile(latencies, 95)) if latencies else float("inf")
    if p95 > args.max_p95_ms:
        failures.append(f"p95_latency:{p95:.3f}ms")
    if headroom is not None and headroom < args.min_memory_headroom:
        failures.append(f"memory_headroom:{headroom:.3f}")
    report = {"images": len(records), "p95_inference_ms": round(p95, 3),
              "memory_headroom": headroom, "failures": failures, "records": records}
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
