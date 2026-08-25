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

import json
import math
import time
from enum import Enum

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
    "arm_check": "Arm check", "vlm_arm_check": "Cloud photo check",
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


# Detectors with a genuine on-demand action, mirroring the main.py hotkeys
# ('t' -> request_test(), 'a' -> request_test("arm_check"), 'd' -> circuit).
# Everything absent here is passive: it fires when the frame allows, and the
# /modules console must not offer a button that would do nothing.
_MODULE_TRIGGERS = {
    "tremor":       {"action": "test", "target": "hold_still",
                     "label": "Hold still"},
    "arm_skin":     {"action": "test", "target": "arm_check",
                     "label": "Arm skin check"},
    "skin_vision":  {"action": "test", "target": "arm_check",
                     "label": "Arm skin check"},
    "guided_assessments": {"action": "circuit", "target": None,
                           "label": "Run demo circuit"},
    # Detectors the vision model also comments on. Its passive cadence is 60s,
    # which reads as the system doing nothing in front of an audience, so each
    # card can ask for a reading now. Kept in step with skin_vision._CUE_ROUTES
    # by tests/vlm_cues_test.py.
    **{module: {"action": "vlm_scan", "target": None,
                "label": "Ask the vision model now"}
       for module in ("dry_lips", "facial_swelling", "skin_color",
                      "drowsiness", "sweating", "eye_redness", "rash")},
}

# Honest expectation-setting for a client audience: how far a given signal can
# actually be trusted, and why. Sourced from each module's own documented
# limits, not from wishful thinking. Absent means unrated, not good.
_MODULE_RELIABILITY = {
    "presence":         ("HIGH", "Straightforward person-in-view detection"),
    "weather":          ("HIGH", "API passthrough, not a vision heuristic"),
    "gesture":          ("HIGH", "Mature hand-landmark model for static labels"),
    "drowsiness":       ("HIGH", "Eye closure and PERCLOS with a frontal face"),
    "heart_rate":       ("MEDIUM", "Sensitive to lighting, motion and skin tone"),
    "respiration":      ("MEDIUM", "Shoulder motion; stronger with depth"),
    "clothing":         ("MEDIUM", "Angle and occlusion sensitive"),
    "clothing_advice":  ("MEDIUM", "Only as good as the clothing and weather inputs"),
    "facial_asymmetry": ("MEDIUM", "Landmark baselines; stronger with depth"),
    "tremor":           ("MEDIUM", "Passive is fair; the hold-still test is far stronger"),
    "gait":             ("MEDIUM", "Needs legs visible in a side or front view"),
    "balance":          ("MEDIUM", "Sway from pose; needs a still stance"),
    "eye_movement":     ("MEDIUM", "Gaze is solid; nystagmus is frame-rate limited"),
    "yawn":             ("MEDIUM", "Dependable for the gesture itself"),
    "head_nod":         ("MEDIUM", "Head pitch against a personal baseline"),
    "fall":             ("MEDIUM", "Clear falls within view"),
    "near_fall":        ("MEDIUM", "Stumble-and-recover cue; never alerts alone"),
    "unresponsive":     ("MEDIUM", "A very still but well person can trigger it"),
    "wandering":        ("MEDIUM", "Within a single camera view"),
    "attention":        ("MEDIUM", "Eye contact and sustained engagement"),
    "sneeze":           ("MEDIUM", "Misses sneezes turned away; counts are a lower bound"),
    "face_touch":       ("MEDIUM", "Hand-to-face contact as a behavioural proxy"),
    "activity_level":   ("MEDIUM", "A relative trend; needs history"),
    "height_distance":  ("MEDIUM", "Distance is strong; height within a few cm"),
    "age_estimation":   ("MEDIUM", "A coarse eight-bucket estimate at best"),
    "emotion":          ("MEDIUM", "Backends vary; not validated for elder care"),
    "scene_vision":     ("MEDIUM", "Opt-in cloud model; shape-validated only"),
    "skin_color":       ("LOW", "Chroma heuristic; very lighting dependent"),
    "rash":             ("LOW", "Colour and texture threshold heuristic"),
    "arm_skin":         ("LOW", "Colour and texture heuristics on arm regions"),
    "skin_vision":      ("LOW", "Off unless consent and an API key are given"),
    "bruise":           ("LOW", "Colour patch heuristic"),
    "eye_redness":      ("LOW", "Sclera colour heuristic"),
    "sweating":         ("LOW", "Confounded by oily skin and lighting"),
    "dry_lips":         ("LOW", "Lip texture heuristic"),
    "facial_swelling":  ("LOW", "Weak on colour alone; stronger with depth"),
    "bradykinesia":     ("LOW", "Arm speed during otherwise active periods"),
    "masked_face":      ("LOW", "Longitudinal; needs minutes of history"),
    "expressivity":     ("LOW", "The longitudinal comparison is the real signal"),
    "pain":             ("LOW", "Grimace cues; a screening prompt only"),
    "agitation":        ("LOW", "Erratic upper-body movement"),
    "grooming":         ("LOW", "Needs weeks of history; a haircut also trips it"),
    "body_estimate":    ("LOW", "Monocular and uncalibrated; explicitly not BMI"),
    "spo2":             ("LOW", "Trend only, uncalibrated by default; plain-webcam "
                                "mode is opt-in and unvalidated against a reference oximeter"),
    "hazard_zones":     ("MEDIUM", "Only as good as the zones configured for the room"),
    "multi_person":     ("MEDIUM", "Anonymous track assignment; carries no identity"),
    "routine":          ("LOW", "An occupancy trend; makes no adherence claim"),
    "guided_assessments": ("N/A", "Orchestrates guided protocols rather than observing"),
    "replay_events":    ("N/A", "Replays scripted fixtures; not a live sensor"),
}


def _reliability(name: str) -> dict | None:
    """Reliability tier and reason for one module, or None when unrated."""
    entry = _MODULE_RELIABILITY.get(name)
    if entry is None:
        return None
    tier, note = entry
    return {"tier": tier, "note": note}


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
    ("yawn", "yawn_count_total", "Yawns total"),
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


_DISPLAY_MAX_CHARS = 500
_DISPLAY_MAX_ITEMS = 32
_MEDIA_KEY_PARTS = (
    "audio", "base64", "bytes", "crop", "embedding", "frame", "image",
    "media", "pixels", "samples", "thumbnail", "video",
)


def _safe_display_data(value, *, key_hint: str = "", depth: int = 0):
    """Return bounded JSON-like data while redacting raw media."""
    if key_hint and any(part in key_hint.lower() for part in _MEDIA_KEY_PARTS):
        return "<redacted-media>"
    if depth > 4:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return _safe_display_data(value.value, depth=depth + 1)
    if isinstance(value, np.generic):
        return _safe_display_data(value.item(), depth=depth + 1)
    if isinstance(value, (bytes, bytearray, memoryview, np.ndarray)):
        return "<redacted-media>"
    if isinstance(value, str):
        if value.lstrip().lower().startswith("data:"):
            return "<redacted-media>"
        return value if len(value) <= _DISPLAY_MAX_CHARS else "<oversized-value>"
    if isinstance(value, dict):
        items = list(value.items())
        out = {
            str(key): _safe_display_data(item, key_hint=str(key), depth=depth + 1)
            for key, item in items[:_DISPLAY_MAX_ITEMS]
        }
        if len(items) > _DISPLAY_MAX_ITEMS:
            out["..."] = f"{len(items) - _DISPLAY_MAX_ITEMS} more fields"
        return out
    if isinstance(value, (tuple, list, set)):
        items = list(value)
        out = [_safe_display_data(item, depth=depth + 1)
               for item in items[:_DISPLAY_MAX_ITEMS]]
        if len(items) > _DISPLAY_MAX_ITEMS:
            out.append(f"<{len(items) - _DISPLAY_MAX_ITEMS} more items>")
        return out
    return f"<unsupported:{type(value).__name__}>"


def _display_value(value) -> str:
    """Format one Result value for a compact, privacy-safe browser card."""
    safe = _safe_display_data(value)
    if isinstance(safe, (dict, list)):
        rendered = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
    elif safe is None:
        rendered = "—"
    else:
        rendered = _fmt(safe)
    if len(rendered) > _DISPLAY_MAX_CHARS:
        return "<oversized-value>"
    return rendered


def _module_reading(r, *, now: float | None = None) -> dict:
    """Serialize one public Result for the module console.

    Unlike ``signals``, this feed includes message-less measurements and
    backend-comparison rows.  Values are display-formatted strings so complex
    module payloads remain JSON-safe and the browser never has to guess how to
    format a detector-specific type.
    """
    metric, backend = _split_backend(r.key)
    display_metric = metric or r.key
    now = time.time() if now is None else now
    expires_at = float(r.timestamp + r.ttl)
    return {
        "module": r.module,
        "key": r.key,
        "metric": display_metric,
        "name": _METRIC_NAMES.get(display_metric, _humanize_slug(display_metric)),
        "backend": backend,
        "value": _display_value(r.value),
        "conf": round(float(r.confidence), 2),
        "quality": (round(float(r.quality), 2) if r.quality is not None else None),
        "severity": r.severity.value,
        "message": r.message,
        "subject_id": r.subject_id,
        "source": r.source,
        "timestamp": round(float(r.timestamp), 3),
        "expires_at": round(expires_at, 3),
        # Browsers establish a local deadline from this duration, avoiding
        # incorrect freshness when a LAN client's wall clock is skewed.
        "fresh_for": round(max(0.0, expires_at - now), 3),
    }


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
            cells.append({"backend": b, "value": _display_value(r.value),
                          "conf": round(float(r.confidence), 2),
                          "severity": r.severity.value})
        comparison.append({"module": module,
                           "metric": metric,
                           "name": _METRIC_NAMES.get(metric, metric),
                           "backends": cells})

    by_key = {(r.module, r.key): r for r in snapshot}
    readings_now = time.time()
    module_readings = [_module_reading(r, now=readings_now) for r in sorted(
        snapshot, key=lambda r: (r.module, r.subject_id, r.key))]

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
    # ModuleGate (core/module_gate.py) pause/resume overlay for the /modules
    # console. Absent (None) for a caller/test that predates the gate -- fall
    # back to the existing `enabled` set so old payloads still render.
    gate_snapshot = (system or {}).get("module_gate") if system else None
    # What was actually instantiated at startup -- independent of live gate
    # state. modules_enabled above is gate-filtered (empty under
    # --start-blank), so it cannot answer "is this module loaded" without
    # falsely claiming a merely-paused module needs a restart to toggle on.
    # A caller/test that predates this field falls back to `enabled`.
    loaded = set((system or {}).get("modules_loaded", enabled)) if system else enabled
    secondary_eligible = set((system or {}).get("secondary_modules", []) if system else [])
    modules_list = []
    cloud_vision = (system or {}).get("cloud_vision")
    for slug, cls in registered.items():
        label, blurb = module_label(slug)
        # requires/interval/consent come straight off the registered class, so
        # the /modules console can never drift from what the scheduler enforces.
        trigger = _MODULE_TRIGGERS.get(slug)
        loaded_primary = slug in loaded
        if gate_snapshot is not None:
            primary_on = slug in gate_snapshot.get("primary", [])
        else:
            primary_on = slug in enabled
        eligible_secondary = slug in secondary_eligible
        if not eligible_secondary:
            secondary_on = None
        elif gate_snapshot is not None:
            secondary_on = slug in gate_snapshot.get("secondary", [])
        else:
            secondary_on = False
        # Toggleable mirrors what main.py's module_handler will actually accept:
        # primary requires the module to already be instantiated (loaded from
        # config/modules.yaml at startup); secondary requires it to be listed
        # under tracking.secondary_modules. Neither can be granted by a UI click
        # -- both need a restart or a config change, so the reason text below
        # must never imply the switch alone can flip them on.
        toggleable_primary = loaded_primary
        toggleable_secondary = eligible_secondary
        reason_primary = None if toggleable_primary else (
            "Not loaded — enable in config/modules.yaml and restart.")
        reason_secondary = None if toggleable_secondary else (
            "Not configured for secondary subjects — add to "
            "tracking.secondary_modules in config/modules.yaml.")
        entry = {"module": slug, "label": label, "blurb": blurb,
                 "running": slug in enabled,
                 "requires": list(getattr(cls, "requires", ()) or ()),
                 "interval": round(float(getattr(cls, "interval", 0.0) or 0.0), 2),
                 "consent": bool(getattr(cls, "consent", False)),
                 "internal": slug in INTERNAL_MODULES,
                 "reliability": _reliability(slug),
                 "trigger": trigger,
                 "enabled": {"primary": bool(primary_on), "secondary": secondary_on},
                 "toggleable": {"primary": bool(toggleable_primary),
                                "secondary": bool(toggleable_secondary)},
                 "toggle_reason": {"primary": reason_primary,
                                   "secondary": reason_secondary}}
        # Only the cards whose button actually calls the cloud carry provider
        # health, so an outage explains itself where the button is.
        if cloud_vision and trigger and trigger.get("action") == "vlm_scan":
            entry["provider"] = cloud_vision
        modules_list.append(entry)
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
        "module_readings": module_readings,
        "module_counts": module_counts,
        "assessments": _launchable_assessments(),
    }


# --- iPad telemetry -------------------------------------------------------
#
# The paired iPad cannot reach the laptop's /data HTTP endpoint: the hotspot
# AP isolates clients from each other (the whole reason the signaling relay
# exists). Its dashboard is therefore fed over the WebRTC *control* data
# channel instead of HTTP. aiortc advertises a=max-message-size:65536 and
# Safari throws on send() past it, so one telemetry message must serialize
# below this ceiling -- tests/ipad_payload_test.py asserts it with a full
# module roster loaded.
IPAD_PAYLOAD_MAX_BYTES = 60000

# Most-severe-first signals shown on the iPad, trimmed so one message stays
# well under the ceiling no matter how busy the scene gets.
_IPAD_MAX_SIGNALS = 18

# Vitals promoted to the iPad's "live vitals" tiles -- the showcase's visual
# star. Each names the canonical Result its module emits (heart_rate->"bpm",
# respiration->"breaths_per_min", spo2->"spo2", drowsiness->"blink_rate");
# emotion has no canonical key, so the best-confidence "emotion_<backend>"
# wins.
# (id, label, unit, module, metric, kind)
_IPAD_VITALS = [
    ("hr",    "Heart rate", "bpm",  "heart_rate",  "bpm",             "number"),
    ("resp",  "Breathing",  "/min", "respiration", "breaths_per_min", "number"),
    ("spo2",  "SpO₂",       "%",    "spo2",        "spo2",            "number"),
    ("blink", "Blinks",     "/min", "drowsiness",  "blink_rate",      "number"),
    ("mood",  "Mood",       "",     "emotion",     "emotion",         "label"),
]


def _best_vital(snapshot, module: str, metric: str):
    """Best Result for one vital: an exact canonical key wins; otherwise the
    highest-confidence backend-suffixed key (``<metric>_<backend>``)."""
    exact = [r for r in snapshot if r.module == module and r.key == metric]
    if exact:
        return max(exact, key=lambda r: r.confidence)
    prefix = metric + "_"
    backend = [r for r in snapshot if r.module == module and r.key.startswith(prefix)]
    if not backend:
        return None
    return max(backend, key=lambda r: r.confidence)


def _ipad_vital(snapshot, spec, now: float) -> dict:
    vid, label, unit, module, metric, kind = spec
    reliability = _reliability(module)
    r = _best_vital(snapshot, module, metric)
    if r is None:
        # A tile the page still renders as "measuring…" rather than dropping,
        # so the layout is stable and the audience sees what is being tracked.
        return {"id": vid, "label": label, "unit": unit, "value": None,
                "present": False, "severity": "info", "conf": 0.0,
                "reliability": reliability}
    value = str(r.value) if kind == "label" else _fmt(r.value)
    return {
        "id": vid, "label": label, "unit": unit, "value": value,
        "present": True, "severity": r.severity.value,
        "conf": round(float(r.confidence), 2),
        "quality": (round(float(r.quality), 2) if r.quality is not None else None),
        "message": r.message,
        # A local deadline the page derives without trusting a shared clock,
        # mirroring _module_reading's fresh_for (LAN wall clocks drift).
        "fresh_for": round(max(0.0, float(r.timestamp + r.ttl) - now), 3),
        "reliability": reliability,
    }


# Big, one-action-verifiable tiles for the iPad Resident view "live demo" strip.
# Each is a counter that ticks up on a single action, or a label that flips when
# the person does something, so a showcase can perform the action and point at
# the tile reacting on screen. Counters read their module's monotonic *_total /
# *_count keys (never the decaying rolling-window ones). (id, label, icon,
# module, metric, kind); kind is "count" (integer that only rises) or "label".
_IPAD_DEMO_TILES = [
    ("yawns",   "Yawns",     "🥱", "yawn",       "yawn_count_total",  "count"),
    ("blinks",  "Blinks",    "👁", "drowsiness", "blink_count_total", "count"),
    ("nods",    "Head nods", "🙂", "head_nod",   "nod_count",         "count"),
    ("mood",    "Mood",      "😊", "emotion",    "emotion",           "label"),
    ("gesture", "Gesture",   "✋", "gesture",    "gesture",           "label"),
]


def _ipad_demo_tile(snapshot, spec, now: float) -> dict:
    """One live-demo tile: current counter/label plus freshness, or absent.

    A count tile stays ``present`` at 0 (its module emits the total every frame),
    so the tile exists on screen before the first action; a label tile is absent
    until the module has something to say (e.g. no gesture recognized yet).
    """
    tid, label, icon, module, metric, kind = spec
    r = _best_vital(snapshot, module, metric)
    if r is None:
        return {"id": tid, "label": label, "icon": icon, "kind": kind,
                "value": None, "present": False}
    if kind == "count":
        value = int(r.value) if isinstance(r.value, (int, float)) else r.value
    else:
        value = str(r.value).replace("_", " ")
    return {"id": tid, "label": label, "icon": icon, "kind": kind,
            "value": value, "present": True,
            "severity": r.severity.value,
            "fresh_for": round(max(0.0, float(r.timestamp + r.ttl) - now), 3)}


def ipad_payload(snapshot, fps: float = 0.0, greeting: str | None = None,
                 reasoning: dict | None = None, system: dict | None = None,
                 performance: dict | None = None) -> dict:
    """Compact telemetry projection pushed to the paired iPad.

    A strict subset of :func:`to_payload`: it reuses the same signal, module,
    and stat builders (single source of truth for promotion floors, reliability
    tiers, and gate state) and then drops the heavy fields -- module_readings,
    per-backend comparison rows, requires/interval -- and caps the signal feed,
    so one JSON message serializes below ``IPAD_PAYLOAD_MAX_BYTES``. Adds a
    clean ``vitals`` list scanned straight from the snapshot, because
    to_payload exposes vitals only as prose signals, not as tile-ready values.
    """
    public = [r for r in snapshot if r.visibility == Visibility.PUBLIC]
    now = time.time()
    full = to_payload(snapshot, fps=fps, greeting=greeting, reasoning=reasoning,
                      system=system, performance=performance)

    vitals = [_ipad_vital(public, spec, now) for spec in _IPAD_VITALS]
    demo_tiles = [_ipad_demo_tile(public, spec, now) for spec in _IPAD_DEMO_TILES]

    signals = []
    for s in full["signals"][:_IPAD_MAX_SIGNALS]:
        rel = _MODULE_RELIABILITY.get(s["module"])
        signals.append({
            "label": s["label"], "message": s["message"],
            "severity": s["severity"], "conf": s["conf"],
            "module": s["module"], "promote": s["promote"],
            "internal": s["internal"], "subject_id": s["subject_id"],
            "tier": rel[0] if rel else None,
        })

    modules = []
    for m in full["modules"]:
        rel = m.get("reliability")
        modules.append({
            "module": m["module"], "label": m["label"], "blurb": m["blurb"],
            "running": m["running"], "enabled": m["enabled"],
            "toggleable": m["toggleable"], "toggle_reason": m["toggle_reason"],
            "tier": rel["tier"] if rel else None,
            "consent": m["consent"], "internal": m["internal"],
            "trigger": m["trigger"],
        })

    perf = performance or {}
    performance_slim = {
        "capture_fps": round(float(perf.get("capture_fps", fps) or 0.0), 1),
        "preview_fps": round(float(perf.get("preview_fps", 0.0) or 0.0), 1),
        "analysis_fps": round(float(perf.get("analysis_fps", 0.0) or 0.0), 1),
    }
    features = (system or {}).get("features") or {}
    tracks = full.get("tracks")

    return {
        "type": "telemetry",
        "v": 1,
        "ts": round(now, 3),
        "fps": round(float(fps), 1),
        "performance": performance_slim,
        "greeting": full["greeting"],
        "person_present": bool(features.get("person")),
        "vitals": vitals,
        "demo_tiles": demo_tiles,
        "signals": signals,
        "stats": {"fatigue": full["fatigue"],
                  "clothing_weather": full["clothing_weather"],
                  "advice": full["advice"]},
        "modules": modules,
        "module_counts": full["module_counts"],
        "reasoning": full["reasoning"],
        "assessments": full["assessments"],
        "subjects": len(tracks) if isinstance(tracks, list) else 0,
    }
