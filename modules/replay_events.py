"""Convert scripted replay summaries into ordinary production Results."""
from core.events import PersistencePolicy, Result, Severity
from modules.base import DetectionModule
from core.registry import register


@register("replay_events")
class ReplayEvents(DetectionModule):
    """Emit deterministic fixture observations without special downstream paths."""
    name = "replay_events"

    def process(self, ctx):
        """Convert due replay records to standard public results."""
        out = []
        for event in ctx.extras.get("replay_events", []):
            severity = Severity(str(event.get("severity", "info")))
            out.append(Result(module=str(event.get("module", self.name)),
                              key=str(event.get("key", "event")), value=event.get("value"),
                              confidence=float(event.get("confidence", 1.0)), severity=severity,
                              message=str(event.get("message", "Scripted replay observation")),
                              source="replay", quality=float(event.get("quality", 1.0)),
                              subject_id=str(event.get("subject_id", "primary")),
                              location=event.get("location"),
                              correlation_id=event.get("correlation_id"),
                              persistence=PersistencePolicy.EVENT))
        return out
