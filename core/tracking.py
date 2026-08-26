"""Short-lived anonymous tracking based only on pose/face geometry and time."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class _Track:
    x: float
    y: float
    timestamp: float
    first_seen: float
    frames: int = 1


class AnonymousTracker:
    """Associate geometry without recognition and keep a stable primary track."""
    def __init__(self, max_distance: float = 0.25, ttl: float = 3.0,
                 primary_min_frames: int = 3):
        self.max_distance = max_distance
        self.ttl = ttl
        self.primary_min_frames = primary_min_frames
        self._tracks: dict[str, _Track] = {}
        self._next = 1
        self.primary_track_id: str | None = None

    def set_primary(self, track_id: str) -> bool:
        """Explicitly select one current anonymous track as primary."""
        if track_id not in self._tracks:
            return False
        self.primary_track_id = track_id
        return True

    def update(self, boxes: list[tuple], width: int, height: int,
               now: float | None = None) -> list[dict]:
        """Return stable anonymous IDs, assignment, and geometry ambiguity."""
        now = time.time() if now is None else now
        self._tracks = {k: v for k, v in self._tracks.items()
                        if now - v.timestamp <= self.ttl}
        if self.primary_track_id not in self._tracks:
            self.primary_track_id = None
        out, used = [], set()
        ordered = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        for box in ordered:
            cx = (box[0] + box[2]) / (2 * max(width, 1))
            cy = (box[1] + box[3]) / (2 * max(height, 1))
            candidates = sorted((math.hypot(cx - track.x, cy - track.y), tid)
                                for tid, track in self._tracks.items() if tid not in used)
            ambiguous = len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 0.04
            if candidates and candidates[0][0] <= self.max_distance:
                tid = candidates[0][1]
                previous = self._tracks[tid]
                self._tracks[tid] = _Track(cx, cy, now, previous.first_seen,
                                           previous.frames + 1)
            else:
                tid = f"track-{self._next}"
                self._next += 1
                self._tracks[tid] = _Track(cx, cy, now, now)
            used.add(tid)
            out.append({"track_id": tid, "subject_id": tid, "bbox": box,
                        "ambiguous": ambiguous,
                        "stable_frames": self._tracks[tid].frames})
        if self.primary_track_id is None:
            eligible = [item for item in out
                        if item["stable_frames"] >= self.primary_min_frames
                        and not item["ambiguous"]]
            if eligible:
                self.primary_track_id = min(
                    eligible, key=lambda item: self._tracks[item["track_id"]].first_seen)["track_id"]
        for item in out:
            if item["track_id"] == self.primary_track_id:
                item["subject_id"] = "primary"
                item["primary"] = True
            else:
                item["primary"] = False
        return out

    def snapshot(self) -> dict:
        """Return public debug state without persistent identity information."""
        return {"primary_track_id": self.primary_track_id,
                "active_track_ids": sorted(self._tracks)}
