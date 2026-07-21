#!/usr/bin/env python3
"""Calibrate and evaluate the pinned vitiligo signal on a person-disjoint CSV.

Required columns: path,label,person_id,split.  Labels are ``vitiligo`` or
``negative`` and splits are ``calibration`` or ``test``. Optional subgroup
columns (skin_tone,camera,lighting,body_region) are reported independently.
The tool writes a screening-enabled calibration only when --approve is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.local_skin_classifier import (  # noqa: E402
    DEFAULT_MODEL, DEFAULT_REVISION, TorchSkinClassifier,
    _normalized_entropy, _softmax,
)


GROUPS = ("skin_tone", "camera", "lighting", "body_region")


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = float(ranks[positive].sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    auc = ((pos_ranks - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))
    bins = []
    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        high = low + 0.1
        mask = (scores >= low) & (scores <= high if high >= 1.0 else scores < high)
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        prevalence = float(labels[mask].mean())
        count = int(mask.sum())
        ece += count / len(scores) * abs(confidence - prevalence)
        bins.append({"low": round(float(low), 2), "high": round(float(high), 2),
                     "count": count, "mean_probability": round(confidence, 6),
                     "prevalence": round(prevalence, 6)})
    return {
        "count": len(labels), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sensitivity, 6),
        "specificity": round(specificity, 6),
        "precision": round(precision, 6), "npv": round(npv, 6),
        "auroc": round(float(auc), 6),
        "expected_calibration_error": round(float(ece), 6),
        "reliability_bins": bins,
        "false_reassurance_rate": round(fn / max(1, tp + fn), 6),
    }


def _target_probabilities(logits: np.ndarray, target_index: int,
                          temperature: float) -> np.ndarray:
    return np.asarray([_softmax(row, temperature)[target_index] for row in logits])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("assets/skin_models/vit_skin_calibration.json"))
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--approve", action="store_true",
                        help="mark calibration validated after human review")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("runtime-models/huggingface"))
    args = parser.parse_args()
    rows = list(csv.DictReader(args.dataset.open("r", encoding="utf-8", newline="")))
    required = {"path", "label", "person_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"dataset must contain columns: {sorted(required)}")
    for row in rows:
        if row["label"] not in {"vitiligo", "negative"}:
            raise SystemExit("labels must be vitiligo or negative")
        if row["split"] not in {"calibration", "test"}:
            raise SystemExit("splits must be calibration or test")
    calibration_people = {row["person_id"] for row in rows if row["split"] == "calibration"}
    test_people = {row["person_id"] for row in rows if row["split"] == "test"}
    overlap = sorted(calibration_people & test_people)
    if overlap:
        raise SystemExit(f"person-disjoint split violated by {len(overlap)} person(s)")

    classifier = TorchSkinClassifier(
        model=DEFAULT_MODEL, revision=DEFAULT_REVISION,
        target_labels=["vitiligo"], calibration_file=None,
        download_if_missing=args.download_if_missing,
        cache_dir=str(args.cache_dir),
        min_sharpness=0.0, min_brightness=0.0, max_brightness=255.0,
        max_clipped_fraction=1.0)
    classifier.preload()
    logits, usable_rows = [], []
    try:
        for row in rows:
            path = (args.dataset.parent / row["path"]).resolve()
            image = cv2.imread(str(path))
            if image is None:
                raise SystemExit(f"cannot read {path}")
            reason = classifier._quality_reason(image)
            if reason:
                print(f"abstain,{path},{reason}")
                continue
            logits.append(classifier._infer_logits(classifier._preprocess(image)))
            usable_rows.append(row)
    finally:
        classifier.close()
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray([row["label"] == "vitiligo" for row in usable_rows], dtype=np.int8)
    splits = np.asarray([row["split"] for row in usable_rows])
    calibration_mask = splits == "calibration"
    test_mask = splits == "test"
    if not calibration_mask.any() or not test_mask.any():
        raise SystemExit("usable calibration and test samples are both required")
    target_index = int(classifier._target_index)

    def loss(temperature: float) -> float:
        probabilities = _target_probabilities(values[calibration_mask], target_index, temperature)
        y = labels[calibration_mask].astype(np.float64)
        return -float(np.mean(y * np.log(np.maximum(probabilities, 1e-9))
                              + (1 - y) * np.log(np.maximum(1 - probabilities, 1e-9))))

    fitted = minimize_scalar(loss, bounds=(0.05, 10.0), method="bounded")
    temperature = float(fitted.x)
    probabilities = _target_probabilities(values, target_index, temperature)
    calibration_scores = probabilities[calibration_mask]
    calibration_labels = labels[calibration_mask]
    candidates = sorted(set(float(score) for score in calibration_scores), reverse=True)
    eligible = [threshold for threshold in candidates
                if _metrics(calibration_labels, calibration_scores, threshold)["sensitivity"]
                >= args.min_sensitivity]
    threshold = max(eligible, key=lambda item: (
        _metrics(calibration_labels, calibration_scores, item)["specificity"], item),
        default=0.0)
    calibrated_probs = [_softmax(row, temperature) for row in values[calibration_mask]]
    positive_probs = [probs for probs, label in zip(calibrated_probs,
                                                     calibration_labels) if label]
    entropies = [_normalized_entropy(probs) for probs in positive_probs] or [1.0]
    unknown_indices = classifier._unknown_indices
    unknown_scores = [max((float(probs[index]) for index in unknown_indices), default=0.0)
                      for probs in positive_probs] or [1.0]
    output = {
        "version": 1,
        "validated": bool(args.approve),
        "model": DEFAULT_MODEL,
        "model_revision": DEFAULT_REVISION,
        "targets": {"vitiligo": {
            "temperature": round(temperature, 8),
            "accept_threshold": round(float(threshold), 8),
            "max_normalized_entropy": round(float(np.quantile(entropies, 0.95)), 8),
            "max_unknown_probability": round(float(np.quantile(unknown_scores, 0.95)), 8),
        }},
        "calibration_metrics": _metrics(
            calibration_labels, calibration_scores, threshold),
        "test_metrics": _metrics(labels[test_mask], probabilities[test_mask], threshold),
        "person_disjoint": True,
        "approved": bool(args.approve),
    }
    subgroup = {}
    for column in GROUPS:
        names = sorted({row.get(column, "").strip() for row in usable_rows if row.get(column, "").strip()})
        if names:
            subgroup[column] = {}
            for name in names:
                mask = test_mask & np.asarray([row.get(column, "").strip() == name
                                               for row in usable_rows])
                if mask.any():
                    subgroup[column][name] = _metrics(labels[mask], probabilities[mask], threshold)
    output["test_subgroups"] = subgroup
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not args.approve:
        print("Calibration remains debug-only; rerun with --approve only after clinical review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
