"""Per-subject module pool: independent detector instances and rolling
state for every anonymously-tracked person, so a second visitor's readings
never contaminate the primary's buffers (heart_rate windows, longitudinal
baselines, etc.). See core/tracking.py for how track IDs are assigned and
core/module_gate.py for how a subject's active module set is toggled.
"""
from __future__ import annotations

from core.context import FaceData, FrameContext, PoseData, SubjectView
from core.registry import all_registered
from core.scheduler import Scheduler


def _overlap(a: tuple, b: tuple) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def build_subject_views(ctx: FrameContext, tracks: list[dict]) -> list[SubjectView]:
    """Match each anonymous track this frame to its best-overlapping face
    and pose, for every tracked person (primary included). Mirrors the
    single-primary matching that used to live inline in Pipeline.process_frame."""
    faces = ctx.extras.get("faces", [])
    poses = ctx.extras.get("poses", [])
    views = []
    for track in tracks:
        bbox = track["bbox"]
        pose_item = max(poses, key=lambda p: _overlap(bbox, p["bbox"]), default=None)
        pose = None
        if pose_item is not None and _overlap(bbox, pose_item["bbox"]) > 0:
            pose = PoseData(pose_item["landmarks"], pose_item["bbox"])
        face_item = max(faces, key=lambda f: _overlap(bbox, f["bbox"]), default=None)
        face = None
        if face_item is not None and _overlap(bbox, face_item["bbox"]) > 0:
            x1, y1, x2, y2 = face_item["bbox"]
            face = FaceData(face_item["landmarks"], face_item["bbox"],
                            ctx.frame[y1:y2, x1:x2],
                            face_item["landmarks"].shape[0] >= 478)
        views.append(SubjectView(
            subject_id=track["subject_id"], track_id=track["track_id"],
            face=face, pose=pose, bbox=bbox, primary=track["primary"],
            ambiguous=track["ambiguous"], stable_frames=track["stable_frames"]))
    return views


class SubjectModulePool:
    """Lazily builds and evicts per-subject detector instances + a private
    Scheduler for every secondary (non-primary) tracked person.

    A module's rolling state (rPPG buffers, backend instances, baselines)
    lives on the instance, so each secondary subject gets wholly separate
    instances -- reusing the primary's would blend two people's readings.
    Instances stay warm across frames; only a track disappearing for
    `ttl` seconds evicts (and closes) its pool.
    """
    def __init__(self, module_names: list[str], module_params: dict, gate,
                 ttl: float = 3.0):
        self.module_names = list(module_names)
        self.module_params = module_params or {}
        self.gate = gate
        self.ttl = ttl
        self._pools: dict[str, Scheduler] = {}
        self._last_seen: dict[str, float] = {}

    def _build_modules(self) -> list:
        registry = all_registered()
        modules = []
        for name in self.module_names:
            cls = registry.get(name)
            if cls is None:
                continue
            params = dict(self.module_params.get(name) or {})
            params.pop("enabled", None)
            modules.append(cls(**params))
        return modules

    def get(self, track_id: str, now: float) -> Scheduler:
        """Return this track's scheduler, building it on first sight."""
        self._last_seen[track_id] = now
        scheduler = self._pools.get(track_id)
        if scheduler is None:
            scheduler = Scheduler(self._build_modules(), gate=self.gate, scope="secondary")
            self._pools[track_id] = scheduler
        return scheduler

    def _close(self, scheduler: Scheduler) -> None:
        for module in scheduler.modules:
            close = getattr(module, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:  # noqa: BLE001
                print(f"[subjects] module '{module.name}' raised on close")

    def evict_stale(self, now: float) -> None:
        """Drop and close pools for tracks unseen within the TTL."""
        stale = [tid for tid, seen in self._last_seen.items() if now - seen > self.ttl]
        for tid in stale:
            scheduler = self._pools.pop(tid, None)
            self._last_seen.pop(tid, None)
            if scheduler is not None:
                self._close(scheduler)

    def close(self) -> None:
        """Close every pool (shutdown path)."""
        for scheduler in self._pools.values():
            self._close(scheduler)
        self._pools.clear()
        self._last_seen.clear()
