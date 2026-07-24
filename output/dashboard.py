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

from core.events import Severity, Visibility

_W = 760
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

# Guest-facing (label, blurb) for the /demo webui page. Non-diagnostic by
# design: observational phrasing only ("Watches for...", never "Detects
# stroke" or "Diagnoses..."). Curated for the ~30 most demo-visible modules;
# everything else falls back to the module's docstring, then the slug
# itself — see module_label().
_MODULE_LABELS = {
    "presence": ("Presence check", "Notices when someone comes into view"),
    "heart_rate": ("Heart rate", "Tracks pulse from subtle skin color shifts"),
    "respiration": ("Breathing rate", "Tracks breaths from shoulder motion"),
    "spo2": ("Oxygen trend", "Tracks a contactless blood-oxygen trend"),
    "emotion": ("Mood cues", "Reads facial expression for mood cues"),
    "facial_asymmetry": ("Facial symmetry", "Watches for one-sided droop"),
    "activity_level": ("Activity level", "Tracks how much movement happens over time"),
    "drowsiness": ("Drowsiness signs", "Watches for heavy eyelids and slow blinks"),
    "skin_color": ("Skin tone", "Screens skin tone changes like paleness or flushing"),
    "clothing_advice": ("Clothing tips", "Suggests weather-appropriate outfit changes"),
    "grooming": ("Grooming trend", "Tracks grooming appearance drift over days"),
    "tremor": ("Hand tremor", "Watches for rhythmic shaking in the hands"),
    "balance": ("Standing balance", "Tracks postural sway while standing"),
    "gait": ("Walking pattern", "Tracks stride rhythm and left-right symmetry"),
    "fall": ("Fall alert", "Watches for a sudden fall to the ground"),
    "near_fall": ("Stumble recovery", "Watches for a stumble followed by recovery"),
    "pain": ("Discomfort cues", "Watches for grimacing and discomfort expressions"),
    "attention": ("Eye contact", "Tracks attention toward the camera"),
    "eye_movement": ("Eye movement", "Watches gaze direction and unusual eye motion"),
    "eye_redness": ("Eye redness", "Screens for redness in the whites of the eyes"),
    "bruise": ("Skin marks", "Screens for bruising or discoloration on the face"),
    "rash": ("Skin rash", "Screens for a rash or eruption on facial skin"),
    "sweating": ("Sweating signs", "Watches for sweating on the forehead"),
    "yawn": ("Yawn frequency", "Tracks yawn frequency as a fatigue cue"),
    "sneeze": ("Sneeze watch", "Watches for sneeze-like head jerks"),
    "agitation": ("Restlessness signs", "Watches for restless or agitated movement"),
    "multi_person": ("People nearby", "Tracks how many people are in view"),
    "scene_vision": ("Room scene", "Describes the general scene and setting"),
    "hazard_zones": ("Hazard zones", "Watches entries into marked hazard areas"),
    "weather": ("Local weather", "Tracks current outdoor weather conditions"),
    # The rest of the registered detectors. Curated rather than left to the
    # docstring fallback because the docstrings are written for developers
    # ("deprojected pose landmarks", "MediaPipe GestureRecognizer") and this
    # roster is shown to guests on a TV.
    "age_estimation": ("Age estimate", "Estimates an approximate age range"),
    "arm_skin": ("Arm skin", "Screens arm skin when an arm is in view"),
    "body_estimate": ("Body build", "Estimates rough build from shoulder and hip width"),
    "bradykinesia": ("Movement speed", "Watches for unusually slow movement"),
    "clothing": ("Clothing", "Recognizes what upper-body clothing is worn"),
    "dry_lips": ("Lip dryness", "Watches for dry or cracked lips"),
    "expressivity": ("Expressiveness", "Tracks how much the face moves while talking"),
    "face_touch": ("Face touching", "Counts how often a hand touches the face"),
    "facial_swelling": ("Facial puffiness", "Watches for gradual puffiness around the eyes"),
    "gesture": ("Hand gestures", "Recognizes waves, thumbs-up, and similar gestures"),
    "guided_assessments": ("Guided checks", "Runs short guided activities and scores them"),
    "head_nod": ("Head nodding", "Watches for nodding and head drooping"),
    "height_distance": ("Height and distance", "Estimates height and how far away someone is"),
    "masked_face": ("Face movement", "Tracks reduced facial movement over time"),
    "replay_events": ("Scripted reel", "Plays back a fixed scenario for demos"),
    "routine": ("Daily routine", "Tracks the shape of the day without judging it"),
    "skin_vision": ("Skin close-up", "Takes a closer look at skin when asked"),
    "unresponsive": ("Stillness watch", "Watches for unusually long stillness"),
    "wandering": ("Pacing", "Watches for repeated pacing or wandering"),
}

# Guest-facing (label, blurb) for the guided assessments offered in the /demo
# "try this" picker. Observational framing only: each names an activity the
# person performs, never a clinical test or its result (§7 of HANDOFF). The
# blurb doubles as the on-screen "what you'll do" hint. Any protocol without a
# curated entry falls back to its humanized slug via _launchable_assessments().
_PROTOCOL_LABELS = {
    "facial_movement":  ("Face check", "Smile, raise your eyebrows, then close both eyes"),
    "arm_drift":        ("Arm hold", "Hold both arms out in front and keep them still"),
    "balance":          ("Balance", "Stand still for a few seconds near safe support"),
    "sit_to_stand":     ("Sit to stand", "Stand up and sit down a few times at your own pace"),
    "timed_up_and_go":  ("Up and go", "Stand, walk to the marker, turn, and come back"),
    "finger_tapping":   ("Finger taps", "Tap finger and thumb together as evenly as you can"),
    "guided_gait":      ("Walk across", "Walk across the view, turn, and return"),
    "read_aloud":       ("Read aloud", "Read a short sentence aloud at your normal pace"),
    "guided_breathing": ("Breathing", "Breathe normally while the rhythm is observed"),
}


def _launchable_assessments() -> list[dict]:
    """Guest-safe roster of protocols the /demo picker can start, sorted by
    label. Mirrors assessments.PROTOCOLS so the picker never drifts out of
    sync with what the WorkflowEngine can actually run."""
    try:
        from assessments import PROTOCOLS
    except Exception:
        return []
    out = []
    for name in PROTOCOLS:
        label, blurb = _PROTOCOL_LABELS.get(name, (_humanize_slug(name), ""))
        out.append({"protocol": name, "label": label, "blurb": blurb})
    out.sort(key=lambda a: a["label"])
    return out


# Modules that exist for plumbing rather than observation. They still appear
# in the roster, but must never win the guest-facing headline or moment card.
INTERNAL_MODULES = frozenset({"showcase", "replay_events"})

# Minimum confidence for a signal to be promoted to the guest-facing headline
# or a moment card on /demo. Without this floor, a low-confidence "notice"
# (e.g. masked_face at 0.28 confidence) can outrank a high-confidence "info"
# purely on severity and become the largest text on a guest's screen, reading
# a clinical-sounding guess about a real, identifiable person out loud. Safety
# -critical severities (warning/alert) always bypass this floor.
GUEST_CONFIDENCE_FLOOR = 0.45


def _promotable(severity: Severity, confidence: float) -> bool:
    """Whether a signal may become the guest headline / a moment card.

    This is a promotion filter, not a suppression filter: everything still
    appears in payload["signals"] / the timeline and the /demo scrolling
    feed regardless of this result.
    """
    if _SEV_ORDER[severity] >= _SEV_ORDER[Severity.WARNING]:
        return True
    return confidence >= GUEST_CONFIDENCE_FLOOR


def _humanize_slug(name: str) -> str:
    words = name.replace("_", " ").strip()
    if not words:
        return name
    return words[0].upper() + words[1:]


def module_label(name: str) -> tuple[str, str]:
    """Resolve a module slug to a guest-facing (label, blurb).

    Falls through: curated _MODULE_LABELS -> first sentence of the
    registered class's docstring (label derived from the slug) -> the
    slug itself, humanized. Never raises; never hands back the bare
    underscored slug as the label.
    """
    curated = _MODULE_LABELS.get(name)
    if curated is not None:
        return curated
    try:
        from core.registry import all_registered
        cls = all_registered().get(name)
        doc = (cls.__doc__ or "").strip() if cls is not None else ""
    except Exception:
        doc = ""
    if doc:
        blurb = doc.split(". ")[0].split("\n")[0].strip().rstrip(".")
        if blurb:
            return _humanize_slug(name), blurb
    return _humanize_slug(name), ""

_FATIGUE_STATS = [
    ("drowsiness", "ear", "EAR"),
    ("drowsiness", "ear_baseline", "EAR base"),
    ("drowsiness", "ear_closed_threshold", "EAR shut"),
    ("drowsiness", "eye_closed", "Eyes closed"),
    ("drowsiness", "perclos", "PERCLOS"),
    ("drowsiness", "perclos_samples", "PERCLOS n"),
    ("drowsiness", "blink_rate", "Blink/min"),
    ("drowsiness", "microsleep_duration", "Closed sec"),
    ("drowsiness", "perclos_status", "PERCLOS status"),
    ("yawn", "mar", "MAR"),
    ("yawn", "mouth_open", "Mouth open"),
    ("yawn", "mouth_open_duration", "Mouth sec"),
    ("yawn", "yawn_count_3min", "Yawns/3m"),
    ("yawn", "yawn_rate_per_min", "Yawns/min"),
    ("head_nod", "head_pitch", "Pitch"),
    ("head_nod", "head_drop", "Pitch drop"),
    ("head_nod", "nodding_score", "Nod score"),
    ("head_nod", "nod_count", "Nods"),
]

_CLOTHING_WEATHER_STATS = [
    ("weather", "feels_like_c", "Feels like"),
    ("weather", "temperature_c", "Temp"),
    ("weather", "rain_mm", "Rain mm"),
    ("weather", "wind_kph", "Wind kph"),
    ("weather", "uv_index", "UV"),
    ("weather", "status", "Weather"),
    ("clothing", "upper_body", "Clothing"),
    ("clothing", "warmth_score", "Warmth"),
    ("clothing", "status", "Clothing status"),
    ("clothing_advice", "recommendation", "Advice"),
]


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
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
    return str(v)


def _render_fatigue_stats(img, snapshot, y: int) -> int:
    by_key = {(r.module, r.key): r for r in snapshot}
    _text(img, "DROWSINESS / FATIGUE STATS", 16, y, 0.52, (120, 220, 255))
    y += 23
    col_w = 185
    row_h = 20
    x0 = 24
    rows_used = 0
    for i, (module, key, label) in enumerate(_FATIGUE_STATS):
        col = i % 4
        row = i // 4
        rows_used = max(rows_used, row + 1)
        x = x0 + col * col_w
        yy = y + row * row_h
        r = by_key.get((module, key))
        value = _fmt(r.value) if r else "..."
        color = _COLORS[r.severity] if r else (120, 120, 120)
        _text(img, f"{label}:", x, yy, 0.38, (150, 150, 150))
        _text(img, value, x + 88, yy, 0.42, color)
    return y + rows_used * row_h + 6


def _render_clothing_weather(img, snapshot, y: int) -> int:
    by_key = {(r.module, r.key): r for r in snapshot}
    _text(img, "CLOTHING / WEATHER", 16, y, 0.52, (120, 220, 255))
    y += 23
    col_w = 185
    row_h = 20
    x0 = 24
    for i, (module, key, label) in enumerate(_CLOTHING_WEATHER_STATS):
        if key == "recommendation":
            continue
        col = i % 4
        row = i // 4
        x = x0 + col * col_w
        yy = y + row * row_h
        r = by_key.get((module, key))
        value = _fmt(r.value) if r else "..."
        color = _COLORS[r.severity] if r else (120, 120, 120)
        _text(img, f"{label}:", x, yy, 0.38, (150, 150, 150))
        _text(img, value[:18], x + 88, yy, 0.42, color)
    y += 3 * row_h + 4
    advice = by_key.get(("clothing_advice", "recommendation"))
    if advice:
        _text(img, "Advice:", 24, y, 0.38, (150, 150, 150))
        _text(img, str(advice.value)[:95], 88, y, 0.40, _COLORS[advice.severity])
        y += 22
    return y + 4


def render(snapshot, fps: float, greeting: str | None = None, reasoning: dict | None = None,
           performance: dict | None = None, moondream: dict | None = None):
    """Render the data window image for the snapshot."""
    snapshot = [r for r in snapshot if r.visibility == Visibility.PUBLIC]
    MAX_H = 920
    img = np.full((MAX_H, _W, 3), _BG, np.uint8)
    _text(img, "ELDERLY CARE MONITOR — LIVE DATA", 16, 30, 0.62, (255, 255, 255), 2)
    perf = performance or {}
    if perf:
        fps_line = (f"capture {perf.get('capture_fps', fps):4.1f}  "
                    f"preview {perf.get('preview_fps', 0):4.1f}  "
                    f"analysis {perf.get('analysis_fps', 0):4.1f} fps")
    else:
        fps_line = f"{fps:4.1f} fps"
    if moondream is not None:
        fps_line += f"   Moondream {'ON' if moondream.get('active') else 'OFF'}"
    _text(img, f"{fps_line}   {len(snapshot)} live signals", 16, 52, 0.45, (150, 150, 150))
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
    y = _render_clothing_weather(img, snapshot, y)

    cv2.line(img, (16, y), (_W - 16, y), (70, 70, 72), 1)
    y += 24
    y = _render_fatigue_stats(img, snapshot, y)

    cv2.line(img, (16, y), (_W - 16, y), (70, 70, 72), 1)
    y += 24
    _text(img, "SIGNALS", 16, y, 0.52, (120, 220, 255))
    y += 22
    fixed_end = y

    rows = [r for r in snapshot
            if r.message and (r.module, r.key) not in compared_keys]
    rows.sort(key=lambda r: (_SEV_ORDER[r.severity], r.confidence), reverse=True)
    greeting_h = 44 if greeting else 0
    avail = MAX_H - fixed_end - greeting_h - 16
    max_rows = max(0, int(avail / 21))
    for i, r in enumerate(rows):
        if i >= max_rows:
            _text(img, f"... (+{len(rows) - i} more)", 28, y, 0.44, (150, 150, 150))
            y += 22
            break
        _text(img, f"[{r.confidence:.2f}]", 20, y, 0.42, (150, 150, 150))
        _text(img, r.message, 92, y, 0.45, _COLORS[r.severity])
        y += 21

    if greeting:
        y += 4
        cv2.line(img, (16, y), (_W - 16, y), (70, 70, 72), 1)
        y += 24
        _text(img, greeting.split("\n")[0][:78], 16, y, 0.5, (180, 230, 180))
        y += 24

    if reasoning:
        _text(img, f"Observed: {reasoning['observed']}", 16, y, 0.42, (150, 210, 255))
        y += 18
        _text(img, f"Question: {reasoning['question'][:78]}", 16, y, 0.40, (220, 220, 220))
        y += 18
        _text(img, f"Answer: {reasoning['answer']}", 16, y, 0.40, (180, 230, 180))
        y += 18
        if reasoning.get("suggestion"):
            _text(img, f"Suggestion: {reasoning['suggestion'][:72]}", 16, y, 0.40, (180, 230, 180))
            y += 18

    return img[:y + 8, :]


def to_payload(snapshot, fps: float = 0.0, greeting: str | None = None,
               reasoning: dict | None = None, system: dict | None = None,
               performance: dict | None = None) -> dict:
    """JSON-safe dict mirroring the rendered window, for the /data web endpoint.

    Reuses the same grouping helpers as render() so the web view matches the
    window exactly (single source of truth)."""
    snapshot = [r for r in snapshot if r.visibility == Visibility.PUBLIC]
    groups = _parse_comparisons(snapshot)
    compared_keys = {(m, f"{metric}_{b}")
                     for (m, metric), bes in groups.items() for b in bes}

    def gkey(item):
        (module, metric) = item[0]
        mo = _METRIC_ORDER.index(metric) if metric in _METRIC_ORDER else 99
        return (module != "heart_rate", module, mo, metric)

    comparison = []
    for (module, metric), bes in sorted(groups.items(), key=gkey):
        cells = []
        for b in sorted(bes, key=lambda k: _BACKENDS.index(k) if k in _BACKENDS else 99):
            r = bes[b]
            cells.append({"backend": b, "value": _fmt(r.value),
                          "conf": round(float(r.confidence), 2),
                          "severity": r.severity.value})
        comparison.append({"metric": metric,
                           "name": _METRIC_NAMES.get(metric, metric),
                           "backends": cells})

    by_key = {(r.module, r.key): r for r in snapshot}

    def stat_rows(defs):
        out = []
        for module, key, label in defs:
            r = by_key.get((module, key))
            out.append({"label": label,
                        "value": _fmt(r.value) if r else "...",
                        "severity": r.severity.value if r else "info"})
        return out

    clothing_weather = stat_rows([(m, k, l) for (m, k, l) in _CLOTHING_WEATHER_STATS
                                  if k != "recommendation"])
    advice_r = by_key.get(("clothing_advice", "recommendation"))
    fatigue = stat_rows(_FATIGUE_STATS)

    rows = [r for r in snapshot
            if r.message and (r.module, r.key) not in compared_keys]
    rows.sort(key=lambda r: (_SEV_ORDER[r.severity], r.confidence), reverse=True)
    signals = []
    for r in rows:
        label, blurb = module_label(r.module)
        signals.append({"conf": round(float(r.confidence), 2),
                        "quality": (round(float(r.quality), 2) if r.quality is not None else None),
                        "message": r.message, "severity": r.severity.value,
                        "subject_id": r.subject_id, "source": r.source,
                        "location": r.location, "module": r.module, "key": r.key,
                        "label": label, "blurb": blurb,
                        "internal": r.module in INTERNAL_MODULES,
                        "promote": _promotable(r.severity, r.confidence)})
    tracks_result = next((r for r in snapshot
                          if r.module == "multi_person" and r.key == "tracks"), None)

    try:
        from core.registry import all_registered
        registered = all_registered()
    except Exception:
        registered = {}
    enabled = set((system or {}).get("modules_enabled", []) if system else [])
    modules_list = []
    for slug in registered:
        label, blurb = module_label(slug)
        modules_list.append({"module": slug, "label": label, "blurb": blurb,
                             "running": slug in enabled})
    modules_list.sort(key=lambda m: m["label"])
    module_counts = {"registered": len(registered),
                     "running": sum(1 for m in modules_list if m["running"])}

    return {
        "fps": round(float(fps), 1),
        "performance": performance or {"capture_fps": round(float(fps), 1)},
        "count": len(snapshot),
        "greeting": (greeting.split("\n")[0] if greeting else None),
        "comparison": comparison,
        "clothing_weather": clothing_weather,
        "advice": (str(advice_r.value) if advice_r else None),
        "fatigue": fatigue,
        "signals": signals,
        "tracks": tracks_result.value if tracks_result is not None else [],
        "reasoning": reasoning,
        "system": system or {},
        "modules": modules_list,
        "module_counts": module_counts,
        "assessments": _launchable_assessments(),
    }
