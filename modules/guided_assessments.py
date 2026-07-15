"""Camera-side executor for active AssessmentProtocol sessions."""
from __future__ import annotations

from assessments import PROTOCOLS
from core.events import PersistencePolicy, Result, Severity
from core.registry import register
from core.workflows import WorkflowEngine, WorkflowStage
from modules.base import DetectionModule
from core.shared_signals import SharedSignals


@register("guided_assessments")
class GuidedAssessments(DetectionModule):
    """Verify positioning, sample in memory, score, and offer one quality retry."""
    name = "guided_assessments"
    interval = 0.0

    def __init__(self, **params):
        super().__init__(**params)
        self.engine = WorkflowEngine.instance()
        self._samples: dict[str, list[dict]] = {}
        self.shared_signals = SharedSignals.instance()

    def process(self, ctx):
        """Advance the primary subject's active protocol without blocking capture."""
        session = self.engine.active("primary")
        if session is None or session.protocol not in PROTOCOLS:
            return None
        protocol = PROTOCOLS[session.protocol]
        sample = {"timestamp": ctx.timestamp,
                  "pose": ctx.pose.landmarks.copy() if ctx.pose is not None else None,
                  "face": ctx.face.landmarks.copy() if ctx.face is not None else None,
                  "support_used": bool(ctx.extras.get("support_used", False)),
                  "respiration": ctx.extras.get("respiration"),
                  "speech_metrics": self.shared_signals.get("speech_metrics", max_age=30,
                                                             now=ctx.timestamp),
                  "microphone_ready": bool(self.shared_signals.get("microphone_ready", False)),
                  "depth": ctx.depth,
                  "frame_size": (ctx.w, ctx.h)}
        if session.stage == WorkflowStage.INSTRUCTION:
            # The voice agent advances this stage only after the instruction is
            # actually delivered, so replay speed or another higher-priority
            # utterance cannot start sampling before the person is prompted.
            return None
        if session.stage == WorkflowStage.POSITIONING:
            ready, quality, message = protocol.positioner(sample)
            if not ready:
                if ctx.timestamp - session.stage_started_at > 5 and not self.engine.retry(message):
                    self.engine.transition(WorkflowStage.CANCELLED, message=message, quality=quality)
                return Result(self.name, "positioning", False, quality, Severity.INFO,
                              message, quality=quality, correlation_id=session.correlation_id)
            self._samples[session.correlation_id] = []
            self.engine.transition(WorkflowStage.SAMPLING, message="Sampling", quality=quality)
        if session.stage == WorkflowStage.SAMPLING:
            samples = self._samples.setdefault(session.correlation_id, [])
            samples.append(sample)
            elapsed = ctx.timestamp - session.stage_started_at
            self.engine.transition(WorkflowStage.SAMPLING, progress=elapsed / protocol.sampling_seconds)
            if elapsed < protocol.sampling_seconds:
                return None
            self.engine.transition(WorkflowStage.SCORING, message="Scoring")
            score = protocol.scorer(samples)
            if score.quality < protocol.quality_gate and self.engine.retry("That attempt was hard to measure. Let's reposition once."):
                self._samples.pop(session.correlation_id, None)
                return None
            self.engine.set_score(score.summary, score.measurements, score.quality,
                                  protocol.follow_up_topics)
            self._samples.pop(session.correlation_id, None)
            return Result(self.name, session.protocol, score.measurements,
                          score.confidence, Severity.INFO, score.summary, ttl=60,
                          quality=score.quality, correlation_id=session.correlation_id,
                          conversation_tags=protocol.follow_up_topics,
                          persistence=PersistencePolicy.EVENT)
        return None
