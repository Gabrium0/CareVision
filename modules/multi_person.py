"""Public debug view of short-lived anonymous geometry tracks."""
from core.events import Result, Severity
from core.registry import register
from modules.base import DetectionModule


@register("multi_person")
class MultiPerson(DetectionModule):
    """Expose track assignment and ambiguity without biometric identity."""
    name = "multi_person"
    interval = 0.5

    def process(self, ctx):
        """Summarize current anonymous tracks for the debug dashboard."""
        tracks = ctx.extras.get("tracks", [])
        if not tracks:
            return None
        safe = [{"track_id": t["track_id"], "subject_id": t["subject_id"],
                 "primary": t["primary"], "ambiguous": t["ambiguous"],
                 "stable_frames": t["stable_frames"]} for t in tracks]
        results = [Result(self.name, "tracks", safe, .9, Severity.INFO,
                          f"{len(safe)} anonymous track(s); primary assignment is temporally stable",
                          ttl=2, quality=.9)]
        for track in safe:
            results.append(Result(self.name, "assignment", track, .9, Severity.INFO,
                                  f"{track['track_id']} assigned as {track['subject_id']}"
                                  + (" (ambiguous)" if track["ambiguous"] else ""),
                                  ttl=2, subject_id=track["subject_id"], quality=.9))
        return results
