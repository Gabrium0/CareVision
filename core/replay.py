"""Deterministic camera-compatible replay source for showcase scenarios."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import cv2

from .context import FaceData, FrameContext, PoseData


def _base_pose() -> np.ndarray:
    """Return a visible neutral 33-landmark pose fixture."""
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[:, :2] = .5
    coords = {0:(.5,.12), 11:(.42,.28), 12:(.58,.28), 13:(.36,.42), 14:(.64,.42),
              15:(.32,.55), 16:(.68,.55), 17:(.30,.56), 18:(.70,.56),
              19:(.31,.55), 20:(.69,.55), 21:(.33,.55), 22:(.67,.55),
              23:(.45,.55), 24:(.55,.55), 25:(.45,.72), 26:(.55,.72),
              27:(.44,.90), 28:(.56,.90)}
    for index, xy in coords.items():
        pose[index, :2] = xy
    return pose


def synthetic_pose(profile: str, offset: float, duration: float) -> np.ndarray:
    """Generate deterministic pose trajectories for assessment replay."""
    pose = _base_pose()
    phase = offset / max(duration, .1)
    if profile == "sit_to_stand":
        wave = .5 * (1 - np.cos(2*np.pi*5*phase))
        pose[[23,24], 1] = .65 - .16*wave
        pose[[25,26], 1] = .73 - .05*wave
    elif profile == "timed_up_and_go":
        if phase < .15:
            pose[[23,24], 1] = .65 - phase/.15*.16
        elif phase < .75:
            travel = (phase-.15)/.6
            pose[:, 0] += .22*np.sin(np.pi*travel)
            pose[27, 1] += .025*np.sin(24*np.pi*travel)
            pose[28, 1] -= .025*np.sin(24*np.pi*travel)
            if .42 < phase < .55:
                pose[11,0], pose[12,0] = .48, .52
        else:
            pose[[23,24], 1] = .49 + min(1,(phase-.75)/.2)*.16
    elif profile == "arm_drift":
        pose[15, :2] = (.40 + .008*np.sin(2*np.pi*5*offset), .30 + .12*phase)
        pose[16, :2] = (.60, .30 + .02*phase)
    elif profile == "finger_tapping":
        gap_l = .015 + .02*(.5+.5*np.sin(2*np.pi*3*offset))
        gap_r = .015 + .02*(.5+.5*np.sin(2*np.pi*2.7*offset))
        pose[19,:2], pose[21,:2] = (.31-gap_l/2,.5), (.31+gap_l/2,.5)
        pose[20,:2], pose[22,:2] = (.69-gap_r/2,.5), (.69+gap_r/2,.5)
    elif profile == "balance":
        sway = .018*np.sin(2*np.pi*.35*offset)
        pose[[11,12,23,24], 0] += sway
        if int(offset) in (5, 11):
            pose[27,0] -= .05
    elif profile == "guided_gait":
        pose[:,0] += .2*np.sin(np.pi*phase)
        pose[27,1] += .03*np.sin(2*np.pi*1.6*offset)
        pose[28,1] -= .03*np.sin(2*np.pi*1.6*offset)
        if .45 < phase < .55:
            pose[11,0], pose[12,0] = .48, .52
    elif profile == "guided_breathing":
        pose[[11,12],1] += .012*np.sin(2*np.pi*.22*offset)
    elif profile == "near_fall":
        if phase < .25:
            drop = 0.0
        elif phase < .4:
            drop = (phase-.25)/.15*.35
        elif phase < .55:
            drop = .35-(phase-.4)/.15*.35
        else:
            drop = 0.0
        pose[[11,12,23,24,25,26,27,28],1] += drop
    return pose


def synthetic_face(profile: str, offset: float) -> np.ndarray:
    """Generate a deterministic 478-landmark facial movement fixture."""
    face = np.zeros((478, 3), dtype=np.float32)
    face[:, :2] = (.5, .5)
    face[61,:2], face[291,:2] = (.42,.56),(.58,.56)
    face[[70,63,105,66,107],1] = .40
    face[[336,296,334,293,300],1] = .40
    face[159,:2], face[145,:2] = (.46,.47),(.46,.51)
    face[386,:2], face[374,:2] = (.54,.47),(.54,.51)
    phase = offset % 9
    if profile == "facial_movement":
        if phase < 3:
            face[[61,291],1] -= .03*np.sin(np.pi*phase/3)
        elif phase < 6:
            face[[70,63,105,66,107,336,296,334,293,300],1] -= .025*np.sin(np.pi*(phase-3)/3)
        else:
            close = .02*np.sin(np.pi*(phase-6)/3)
            face[[159,386],1] += close
            face[[145,374],1] -= close
    return face


class ReplayCamera:
    """Feed scripted events through the same FrameContext interface as live cameras."""
    def __init__(self, source: str, **opts):
        name = source.split(":", 1)[1] if ":" in source else source
        path = Path(opts.pop("scenario_path", "config/replay_scenarios.json"))
        scenarios = json.loads(path.read_text(encoding="utf-8"))
        if name not in scenarios:
            raise ValueError(f"unknown replay scenario {name!r}; choose {', '.join(scenarios)}")
        spec = scenarios[name]
        self.source, self.name = source, name
        self.events = list(spec.get("events", [])) if isinstance(spec, dict) else list(spec)
        self.current_fps = float(spec.get("fps", opts.get("request_fps", 10.0))
                                 if isinstance(spec, dict) else opts.get("request_fps", 10.0))
        size = opts.get("request_size", (640, 480))
        self.width, self.height = int(size[0]), int(size[1])
        self._hooks = []
        self._condition = threading.Condition()
        self._paused = False
        self._speed = 1.0
        self._seek_to: float | None = None
        self._position = 0.0
        self._generation = 0
        self._video_path = None
        if isinstance(spec, dict) and spec.get("video"):
            self._video_path = (path.parent / str(spec["video"])).resolve()
        self._duration = float(spec.get("duration", 0)) if isinstance(spec, dict) else 0.0
        self._realtime = bool(spec.get("realtime", False)) if isinstance(spec, dict) else False
        self._pose_profile = str(spec.get("pose_profile", "")) if isinstance(spec, dict) else ""
        self._face_profile = str(spec.get("face_profile", "")) if isinstance(spec, dict) else ""
        self._people = int(spec.get("people", 1)) if isinstance(spec, dict) else 1
        if self._duration <= 0:
            self._duration = max((float(e.get("at", 0)) for e in self.events), default=0) + 1.0

    def open(self) -> None:
        """Camera parity hook; replay data is already loaded."""

    def register_fast_hook(self, hook) -> None:
        """Register a fast-path consumer just like a live camera."""
        self._hooks.append(hook)

    def frames(self):
        """Yield recorded or synthetic frames with synchronized replay channels."""
        while True:
            epoch = time.time() - self._position
            generation = self._generation
            emitted: set[int] = set()
            states: dict[str, object] = {}
            capture = cv2.VideoCapture(str(self._video_path)) if self._video_path else None
            index = max(0, int(self._position * self.current_fps))
            if capture is not None:
                capture.set(cv2.CAP_PROP_POS_MSEC, self._position * 1000)
            while self._position < self._duration and generation == self._generation:
                with self._condition:
                    while self._paused and generation == self._generation:
                        self._condition.wait(timeout=0.25)
                    if self._seek_to is not None:
                        self._position = max(0.0, min(self._duration, self._seek_to))
                        self._seek_to = None
                        self._generation += 1
                        break
                offset = self._position
                if capture is not None:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame = cv2.resize(frame, (self.width, self.height))
                else:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                due, channels = [], {"audio": [], "sensor": [], "answer": []}
                for event_index, event in enumerate(self.events):
                    at = float(event.get("at", 0))
                    if event_index in emitted or at > offset:
                        continue
                    emitted.add(event_index)
                    kind = str(event.get("type", "observation"))
                    if kind in channels:
                        channels[kind].append(event)
                    elif kind in ("pose", "face"):
                        states[kind] = event.get("landmarks")
                    else:
                        due.append(event)
                ts = epoch + offset
                if self._realtime:
                    delay = ts - time.time()
                    if delay > 0:
                        time.sleep(delay)
                for hook in self._hooks:
                    hook(frame, ts)
                ctx = FrameContext(frame=frame, timestamp=ts, frame_index=index,
                                   fps=self.current_fps, extras={"replay_events": due,
                                       "replay_channels": channels, "replay": self.source,
                                       "replay_position": offset,
                                       "replay_states": dict(states)})
                if self._pose_profile:
                    pose = synthetic_pose(self._pose_profile, offset, self._duration)
                    pose_items = []
                    for person_index in range(self._people):
                        person_pose = pose.copy()
                        if self._people > 1:
                            person_pose[:,0] += (-.18 if person_index == 0 else .18)
                            if person_index > 0:
                                person_pose[:, :2] = .5 + (person_pose[:, :2]-.5) * .75
                        visible = person_pose[:,3] >= .5
                        px = person_pose[visible,:2] * np.array([self.width,self.height])
                        x1,y1 = px.min(axis=0).astype(int); x2,y2 = px.max(axis=0).astype(int)
                        pose_items.append({"landmarks": person_pose, "bbox": (x1,y1,x2,y2)})
                    ctx.extras["poses"] = pose_items
                    ctx.extras["pose_count"] = len(pose_items)
                    ctx.pose = PoseData(pose_items[0]["landmarks"], pose_items[0]["bbox"])
                    ctx.person_present = True
                if self._face_profile:
                    face = synthetic_face(self._face_profile, offset)
                    x1,y1,x2,y2 = int(.35*self.width),int(.3*self.height),int(.65*self.width),int(.65*self.height)
                    ctx.face = FaceData(face, (x1,y1,x2,y2), frame[y1:y2,x1:x2], True)
                    ctx.person_present = True
                yield ctx
                index += 1
                self._position += (1.0 / self.current_fps) * self._speed
            if capture is not None:
                capture.release()
            if generation == self._generation:
                return

    def release(self) -> None:
        """No resources are held by replay sources."""

    def control(self, action: str, value: float | None = None) -> dict:
        """Pause, resume, restart, seek, or change deterministic replay speed."""
        with self._condition:
            if action == "pause":
                self._paused = True
            elif action == "resume":
                self._paused = False
            elif action == "restart":
                self._seek_to = 0.0
                self._paused = False
            elif action == "seek" and value is not None:
                self._seek_to = float(value)
            elif action == "speed" and value is not None:
                self._speed = max(0.1, min(8.0, float(value)))
            else:
                raise ValueError(f"unsupported replay action: {action}")
            self._condition.notify_all()
        return self.status()

    def status(self) -> dict:
        """Return public replay state for dashboard controls."""
        return {"scenario": self.name, "paused": self._paused,
                "position": round(self._position, 2), "duration": self._duration,
                "speed": self._speed, "recorded_video": bool(self._video_path)}
