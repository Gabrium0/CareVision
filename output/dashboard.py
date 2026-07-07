"""Standalone data window: renders all detections as readable text on a
dark panel, separate from the camera feed.

Three sections:
- COMPARISON: every detector that runs multiple backends (heart rate,
  emotion, fall, pain, ...) shown with each backend's value side by side, so
  the original heuristic and the tested models line up in columns.
- SIGNALS: every other live detection, colored by severity, most-severe first.
- greeting footer.

Multi-backend results are keyed "<metric>_<backend>" (e.g. bpm_classical,
bpm_open_rppg, emotion_heuristic, emotion_deepface). Backend labels can contain
underscores and model suffixes (open_rppg_physformer), so we split against a
known-label set rather than on the last underscore.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.events import Severity

_W, _H = 700, 820
_BG = (28, 28, 30)
_COLORS = {
    Severity.INFO: (200, 200, 200),
    Severity.NOTICE: (0, 200, 255),
    Severity.WARNING: (0, 140, 255),
    Severity.ALERT: (60, 60, 255),
}
_SEV_ORDER = {Severity.ALERT: 3, Severity.WARNING: 2, Severity.NOTICE: 1, Severity.INFO: 0}
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Known backend labels (normalized to result-key form). Longest-first so
# "open_rppg" matches before a hypothetical "rppg".
_BACKENDS = ["open_rppg", "open_rppg_physformer", "open_rppg_efficientphys",
             "open_rppg_mamba", "tandon_lstm", "hsemotion", "classical",
             "heuristic", "deepface", "ferplus", "pyfeat", "yolo"]

# Pretty names + column order for known metrics
_METRIC_NAMES = {
    "bpm": "HR (bpm)", "hrv_rmssd_ms": "RMSSD (ms)", "hrv_sdnn_ms": "SDNN (ms)",
    "breaths_per_min": "Resp (/min)", "status": "rPPG Status",
    "backend_status": "Backend Status",
    "emotion": "Emotion", "valence": "Mood",
    "fall": "Fall", "pain": "Pain", "age": "Age",
}
_METRIC_ORDER = ["bpm", "hrv_rmssd_ms", "hrv_sdnn_ms", "breaths_per_min",
                 "status", "backend_status", "emotion", "valence", "fall", "pain", "age"]


def _text(img, s, x, y, scale=0.5, color=(230, 230, 230), thick=1):
    cv2.putText(img, s, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def _split_backend(key: str):
    for b in _BACKENDS:
        if key.endswith("_" + b):
            return key[: -(len(b) + 1)], b
    marker = "_open_rppg_"
    if marker in key:
        metric, model = key.split(marker, 1)
        return metric, "open_rppg_" + model
    return None, None


def _parse_comparisons(snapshot):
    """-> {(module, metric): {backend: Result}} for multi-backend keys."""
    groups: dict[tuple, dict] = {}
    for r in snapshot:
        metric, backend = _split_backend(r.key)
        if metric is None:
            continue
        groups.setdefault((r.module, metric), {})[backend] = r
    return groups


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
    return str(v)


def render(snapshot, fps: float, greeting: str | None = None):
    img = np.full((_H, _W, 3), _BG, np.uint8)
    _text(img, "ELDERLY CARE MONITOR — LIVE DATA", 16, 30, 0.62, (255, 255, 255), 2)
    _text(img, f"{fps:4.1f} fps   {len(snapshot)} live signals", 16, 52, 0.45, (150, 150, 150))
    cv2.line(img, (16, 64), (_W - 16, 64), (70, 70, 72), 1)

    groups = _parse_comparisons(snapshot)
    compared_keys = {(m, f"{metric}_{b}")
                     for (m, metric), bes in groups.items() for b in bes}

    y = 90
    _text(img, "COMPARISON  (original heuristic vs tested models)", 16, y, 0.52, (120, 220, 255))
    y += 24
    if not groups:
        _text(img, "(warming up...)", 28, y, 0.45, (150, 150, 150))
        y += 22
    else:
        # order: heart_rate metrics first, then others; metrics in _METRIC_ORDER
        def gkey(item):
            (module, metric) = item[0]
            mo = _METRIC_ORDER.index(metric) if metric in _METRIC_ORDER else 99
            return (module != "heart_rate", module, mo, metric)
        for (module, metric), bes in sorted(groups.items(), key=gkey):
            name = _METRIC_NAMES.get(metric, metric)
            _text(img, name, 24, y, 0.48, (235, 235, 235))
            x = 200
            for b in sorted(bes, key=lambda k: _BACKENDS.index(k) if k in _BACKENDS else 99):
                r = bes[b]
                col = _COLORS[r.severity]
                conf = r.confidence
                shade = col if conf >= 0.4 else tuple(int(c * 0.6) for c in col)
                _text(img, f"{b}:", x, y, 0.38, (150, 150, 150))
                _text(img, _fmt(r.value), x + 78, y, 0.46, shade, 1)
                _text(img, f"{conf:.2f}", x + 78, y + 14, 0.34, (130, 130, 130))
                x += 150
                if x > _W - 120:
                    break
            y += 40

    cv2.line(img, (16, y), (_W - 16, y), (70, 70, 72), 1)
    y += 24
    _text(img, "SIGNALS", 16, y, 0.52, (120, 220, 255))
    y += 22
    rows = [r for r in snapshot
            if r.message and (r.module, r.key) not in compared_keys]
    rows.sort(key=lambda r: (_SEV_ORDER[r.severity], r.confidence), reverse=True)
    for r in rows:
        if y > _H - 60:
            _text(img, f"... (+{len(rows) - rows.index(r)} more)", 28, y, 0.44, (150, 150, 150))
            break
        _text(img, f"[{r.confidence:.2f}]", 20, y, 0.42, (150, 150, 150))
        _text(img, r.message, 92, y, 0.45, _COLORS[r.severity])
        y += 21

    if greeting:
        cv2.line(img, (16, _H - 44), (_W - 16, _H - 44), (70, 70, 72), 1)
        _text(img, greeting.split("\n")[0][:78], 16, _H - 20, 0.5, (180, 230, 180))
    return img
