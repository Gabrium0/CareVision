"""Reusable, non-diagnostic guided movement and speech assessments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class AssessmentScore:
    """Public-safe score plus neutral measurement quality."""
    measurements: dict[str, float | int | bool | str]
    quality: float
    confidence: float
    summary: str


@dataclass(frozen=True)
class AssessmentProtocol:
    """Declarative contract used by live and replay assessment workflows."""
    name: str
    instruction: str
    required_extractors: tuple[str, ...]
    sampling_seconds: float
    positioner: Callable[[dict], tuple[bool, float, str]]
    scorer: Callable[[list[dict]], AssessmentScore]
    quality_gate: float = 0.45
    follow_up_topics: tuple[str, ...] = ("symptoms", "progression", "warning_signs")


def _visible(pose: np.ndarray, indices: tuple[int, ...]) -> float:
    return float(np.mean(pose[list(indices), 3] >= 0.5))


def _pose_position(indices: tuple[int, ...], message: str):
    def check(sample: dict) -> tuple[bool, float, str]:
        pose = sample.get("pose")
        if pose is None:
            return False, 0.0, "I cannot see the required body area. Please step into view."
        pose = np.asarray(pose)
        quality = _visible(pose, indices)
        inside = float(np.mean((pose[list(indices), :2] > .02) &
                               (pose[list(indices), :2] < .98)))
        quality = min(quality, inside)
        return quality >= .8, quality, message if quality >= .8 else \
            "Please reposition so the requested body area is fully visible."
    return check


def _face_position(sample: dict) -> tuple[bool, float, str]:
    face = sample.get("face")
    if face is None:
        return False, 0.0, "Please face the camera in even light."
    arr = np.asarray(face)
    inside = float(np.mean((arr[:, :2] > .02) & (arr[:, :2] < .98)))
    return inside >= .9, inside, "Your face position is usable."


def _audio_position(sample: dict) -> tuple[bool, float, str]:
    ready = bool(sample.get("microphone_ready"))
    return ready, 1.0 if ready else 0.0, \
        "Microphone ready." if ready else "The microphone is unavailable for this task."


def _duration(samples: list[dict]) -> float:
    return max(0.0, float(samples[-1]["timestamp"] - samples[0]["timestamp"])) if len(samples) > 1 else 0.0


def _series(samples: list[dict], index: int, axis: int = 1) -> np.ndarray:
    return np.asarray([float(s["pose"][index][axis]) for s in samples
                       if s.get("pose") is not None and s["pose"][index][3] >= .5])


def _joint_angle(pose: np.ndarray, a: int, b: int, c: int) -> float:
    u, v = pose[a, :2] - pose[b, :2], pose[c, :2] - pose[b, :2]
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / max(denom, 1e-8), -1, 1))))


def _quality(samples: list[dict], indices: tuple[int, ...]) -> float:
    valid = [min(_visible(np.asarray(s["pose"]), indices), 1.0)
             for s in samples if s.get("pose") is not None]
    return round(float(np.mean(valid)) * min(1.0, len(valid) / 12), 3) if valid else 0.0


def _sit_to_stand(samples: list[dict]) -> AssessmentScore:
    duration = _duration(samples)
    hip = np.asarray([np.mean(np.asarray(s["pose"])[[23, 24], 1]) for s in samples if s.get("pose") is not None])
    knee = np.asarray([np.mean([_joint_angle(np.asarray(s["pose"]), 23, 25, 27),
                               _joint_angle(np.asarray(s["pose"]), 24, 26, 28)])
                       for s in samples if s.get("pose") is not None])
    prominence = max(.025, float(np.ptp(hip)) * .2) if len(hip) else .025
    stands, _ = find_peaks(-hip, prominence=prominence, distance=max(2, len(hip)//12))
    partial, _ = find_peaks(-hip, prominence=prominence * .4, distance=max(2, len(hip)//16))
    reps = min(5, len(stands))
    intervals = np.diff([samples[min(i, len(samples)-1)]["timestamp"] for i in stands]) if len(stands) > 1 else np.array([])
    consistency = 1 / (1 + float(np.std(intervals))) if len(intervals) else 0.0
    inferred_support = []
    for sample in samples:
        if sample.get("pose") is None:
            continue
        p = np.asarray(sample["pose"])
        inferred_support.append(min(np.linalg.norm(p[15,:2]-p[25,:2]),
                                    np.linalg.norm(p[16,:2]-p[26,:2])) < .08)
    support = any(bool(s.get("support_used")) for s in samples) or \
        (bool(inferred_support) and float(np.mean(inferred_support)) > .2)
    quality = _quality(samples, (11, 12, 23, 24, 25, 26, 27, 28))
    return AssessmentScore({"repetitions": reps, "total_time_seconds": round(duration, 2),
        "failed_attempts": max(0, len(partial) - reps), "support_used": support,
        "movement_consistency": round(consistency, 3),
        "knee_angle_range_deg": round(float(np.ptp(knee)), 1) if len(knee) else 0.0},
        quality, min(.95, quality), "Sit-to-stand movement measurements captured.")


def _timed_up_and_go(samples: list[dict]) -> AssessmentScore:
    duration = _duration(samples)
    poses = [np.asarray(s["pose"]) for s in samples if s.get("pose") is not None]
    hip_y = np.asarray([np.mean(p[[23, 24], 1]) for p in poses])
    ankle_x = np.asarray([np.mean(p[[27, 28], 0]) for p in poses])
    shoulder_width = np.asarray([abs(p[11, 0] - p[12, 0]) for p in poses])
    standing = bool(len(hip_y) and np.ptp(hip_y) > .08)
    walking = bool(len(ankle_x) and np.ptp(ankle_x) > .12)
    turning = bool(len(shoulder_width) and np.min(shoulder_width) < .6 * np.max(shoulder_width))
    returned = bool(walking and abs(ankle_x[-1] - ankle_x[0]) < .12)
    seated = bool(standing and hip_y[-1] > np.min(hip_y) + .05)
    phases = [standing, walking, turning, returned, seated]
    timestamps = np.asarray([float(s["timestamp"]) for s in samples if s.get("pose") is not None])
    stand_index = int(np.argmin(hip_y)) if len(hip_y) else 0
    turn_index = int(np.argmin(shoulder_width)) if len(shoulder_width) else stand_index
    narrow = np.flatnonzero(shoulder_width < .8*np.max(shoulder_width)) if len(shoulder_width) else np.array([])
    turn_start = int(narrow[0]) if len(narrow) else turn_index
    turn_end = int(narrow[-1]) if len(narrow) else turn_index
    return_candidates = np.flatnonzero((np.arange(len(ankle_x)) > turn_end)
                                       & (np.abs(ankle_x-ankle_x[0]) < .05)) if len(ankle_x) else np.array([])
    return_index = int(return_candidates[0]) if len(return_candidates) else max(turn_end, len(timestamps)-2)
    def span(a, b):
        return round(max(0.0, float(timestamps[min(b,len(timestamps)-1)]-
                                    timestamps[min(a,len(timestamps)-1)])), 2) if len(timestamps) else 0.0
    quality = _quality(samples, (11, 12, 23, 24, 25, 26, 27, 28))
    return AssessmentScore({"stand": standing, "walk": walking, "turn": turning,
        "return": returned, "sit": seated, "phases_completed": sum(phases),
        "stand_seconds": span(0, stand_index), "walk_out_seconds": span(stand_index, turn_start),
        "turn_seconds": span(turn_start, turn_end), "return_walk_seconds": span(turn_end, return_index),
        "sit_seconds": span(return_index, len(timestamps)-1),
        "total_time_seconds": round(duration, 2)}, quality, min(.9, quality),
        "Timed Up and Go phase timings captured.")


def _arm_drift(samples: list[dict]) -> AssessmentScore:
    poses = [np.asarray(s["pose"]) for s in samples if s.get("pose") is not None]
    left = np.asarray([p[15, 1] - p[11, 1] for p in poses])
    right = np.asarray([p[16, 1] - p[12, 1] for p in poses])
    drift_l = float(np.median(left[-max(1, len(left)//4):]) - np.median(left[:max(1, len(left)//4)])) if len(left) else 0
    drift_r = float(np.median(right[-max(1, len(right)//4):]) - np.median(right[:max(1, len(right)//4)])) if len(right) else 0
    quality = _quality(samples, (11, 12, 13, 14, 15, 16))
    return AssessmentScore({"left_relative_height": round(float(np.median(left)), 3) if len(left) else 0,
        "right_relative_height": round(float(np.median(right)), 3) if len(right) else 0,
        "left_downward_drift": round(max(0.0, drift_l), 3),
        "right_downward_drift": round(max(0.0, drift_r), 3),
        "symmetry": round(max(0.0, 1 - abs(drift_l - drift_r) * 5), 3),
        "compliance": bool(len(left) and np.median(np.abs(left)) < .2)},
        quality, min(.95, quality), "Arm height and drift measurements captured.")


def _finger_tapping(samples: list[dict]) -> AssessmentScore:
    poses = [np.asarray(s["pose"]) for s in samples if s.get("pose") is not None]
    duration = _duration(samples)
    left = np.asarray([np.linalg.norm(p[19, :2] - p[21, :2]) for p in poses])
    right = np.asarray([np.linalg.norm(p[20, :2] - p[22, :2]) for p in poses])
    def taps(signal):
        peaks, _ = find_peaks(-signal, prominence=max(.002, float(np.ptp(signal)) * .2)) if len(signal) > 2 else ([], {})
        return np.asarray(peaks)
    lp, rp = taps(left), taps(right)
    def variability(peaks):
        return float(np.std(np.diff(peaks))) if len(peaks) > 2 else 0.0
    quality = _quality(samples, (15, 16, 19, 20, 21, 22))
    lr, rr = len(lp) / max(duration, .1), len(rp) / max(duration, .1)
    return AssessmentScore({"left_tapping_rate_hz": round(lr, 2),
        "right_tapping_rate_hz": round(rr, 2), "left_rhythm_variability": round(variability(lp), 2),
        "right_rhythm_variability": round(variability(rp), 2),
        "left_right_difference_hz": round(abs(lr-rr), 2)}, quality, min(.85, quality),
        "Finger-tapping rhythm measurements captured.")


def _balance(samples: list[dict]) -> AssessmentScore:
    poses = [np.asarray(s["pose"]) for s in samples if s.get("pose") is not None]
    hip = np.asarray([np.mean(p[[23, 24], :2], axis=0) for p in poses])
    ankles = np.asarray([np.linalg.norm(p[27, :2] - p[28, :2]) for p in poses])
    steps = int(np.sum(np.abs(np.diff(ankles)) > .04)) if len(ankles) > 1 else 0
    sway = float(np.sqrt(np.mean(np.sum((hip - np.mean(hip, axis=0)) ** 2, axis=1)))) if len(hip) else 0
    quality = _quality(samples, (11, 12, 23, 24, 27, 28))
    return AssessmentScore({"body_sway": round(sway, 4), "corrective_steps": steps,
        "support_used": any(bool(s.get("support_used")) for s in samples)},
        quality, min(.9, quality), "Hold-still balance measurements captured.")


def _gait(samples: list[dict]) -> AssessmentScore:
    duration = _duration(samples)
    left, right = _series(samples, 27, 0), _series(samples, 28, 0)
    lp, _ = find_peaks(np.abs(np.diff(left)), prominence=.005) if len(left) > 3 else ([], {})
    rp, _ = find_peaks(np.abs(np.diff(right)), prominence=.005) if len(right) > 3 else ([], {})
    cadence = (len(lp) + len(rp)) / max(duration, .1) * 60
    symmetry = min(len(lp), len(rp)) / max(1, max(len(lp), len(rp)))
    shoulder = np.asarray([abs(np.asarray(s["pose"])[11, 0] - np.asarray(s["pose"])[12, 0])
                           for s in samples if s.get("pose") is not None])
    turn_stability = 1 / (1 + float(np.std(shoulder))) if len(shoulder) else 0
    stride = np.concatenate([np.abs(np.diff(left)), np.abs(np.diff(right))]) if len(left)>1 and len(right)>1 else np.array([])
    quality = _quality(samples, (11, 12, 23, 24, 25, 26, 27, 28))
    return AssessmentScore({"cadence_spm": round(cadence, 1), "step_symmetry": round(symmetry, 3),
        "turning_stability": round(turn_stability, 3),
        "shuffling_indicator": bool(len(stride) and np.median(stride) < .004)},
        quality, min(.9, quality), "Guided gait measurements captured.")


def _facial(samples: list[dict]) -> AssessmentScore:
    faces = [np.asarray(s["face"]) for s in samples if s.get("face") is not None]
    mouth = np.asarray([[f[61, 1], f[291, 1]] for f in faces])
    brows = np.asarray([[np.mean(f[[70, 63, 105, 66, 107], 1]),
                         np.mean(f[[336, 296, 334, 293, 300], 1])] for f in faces])
    eyes = np.asarray([[abs(f[159, 1]-f[145, 1]), abs(f[386, 1]-f[374, 1])] for f in faces])
    def symmetry(values):
        return max(0.0, 1 - float(np.max(np.abs(values[:, 0]-values[:, 1]))) * 10) if len(values) else 0
    quality = min(1.0, len(faces) / 12)
    return AssessmentScore({"smile_symmetry": round(symmetry(mouth), 3),
        "eyebrow_raise_symmetry": round(symmetry(brows), 3),
        "eye_closure_symmetry": round(symmetry(eyes), 3),
        "sequence_compliance": bool(len(mouth) and np.ptp(np.mean(mouth, axis=1)) > .005)},
        quality, min(.9, quality), "Facial movement measurements captured.")


def _speech(samples: list[dict]) -> AssessmentScore:
    metrics = next((s.get("speech_metrics") for s in reversed(samples) if s.get("speech_metrics")), {})
    quality = min(1.0, float(metrics.get("quality", .9))) if metrics else 0.0
    return AssessmentScore({"completion": bool(metrics), "speech_duration": float(metrics.get("duration", 0)),
        "words_per_minute": float(metrics.get("words_per_minute", 0)),
        "pauses": int(metrics.get("pauses", 0)),
        "baseline_change": float(metrics.get("baseline_change", 0))}, quality, quality,
        "Read-aloud timing captured." if metrics else "No usable speech captured.")


def _breathing(samples: list[dict]) -> AssessmentScore:
    shoulder = np.asarray([np.mean(np.asarray(s["pose"])[[11, 12], 1])
                           for s in samples if s.get("pose") is not None])
    if len(shoulder) > 3:
        peaks, _ = find_peaks(shoulder, prominence=max(.001, float(np.ptp(shoulder))*.15))
        intervals = np.diff(peaks)
        consistency = 1 / (1 + float(np.std(intervals))) if len(intervals) else 0.0
    else:
        peaks, consistency = [], 0.0
    quality = _quality(samples, (11, 12, 23, 24))
    return AssessmentScore({"instruction_compliance": bool(len(peaks)),
        "observed_cycles": len(peaks), "respiration_consistency": round(consistency, 3)},
        quality, min(.8, quality), "Breathing rhythm consistency captured; lung function was not assessed.")


_FULL = (11, 12, 23, 24, 25, 26, 27, 28)
PROTOCOLS = {
    "sit_to_stand": AssessmentProtocol("sit_to_stand", "Please sit safely, then stand and sit five times at a comfortable pace.", ("pose",), 20, _pose_position(_FULL, "Your full body is visible."), _sit_to_stand),
    "timed_up_and_go": AssessmentProtocol("timed_up_and_go", "When safe, stand, walk to the marker, turn, return, and sit.", ("pose",), 30, _pose_position(_FULL, "The walking area and your full body are visible."), _timed_up_and_go),
    "arm_drift": AssessmentProtocol("arm_drift", "Raise both arms forward with palms up and hold them still.", ("pose",), 10, _pose_position((11,12,13,14,15,16), "Both arms are visible."), _arm_drift),
    "finger_tapping": AssessmentProtocol("finger_tapping", "Tap each index finger and thumb as evenly as you can.", ("pose",), 10, _pose_position((15,16,19,20,21,22), "Both hands are visible."), _finger_tapping, .5),
    "balance": AssessmentProtocol("balance", "Stand still near safe support, without using it unless needed.", ("pose",), 15, _pose_position(_FULL, "Your stance is visible."), _balance),
    "guided_gait": AssessmentProtocol("guided_gait", "Walk across view, turn carefully, and return.", ("pose",), 20, _pose_position(_FULL, "The walking area and your full body are visible."), _gait),
    "facial_movement": AssessmentProtocol("facial_movement", "Smile, raise your eyebrows, then close both eyes.", ("face",), 10, _face_position, _facial),
    "read_aloud": AssessmentProtocol("read_aloud", "Please read the displayed sentence aloud at your normal pace.", ("microphone",), 20, _audio_position, _speech),
    "guided_breathing": AssessmentProtocol("guided_breathing", "Breathe normally and comfortably while I observe the rhythm.", ("pose",), 20, _pose_position((11,12,23,24), "Your shoulders and torso are visible."), _breathing),
}
