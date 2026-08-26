"""General agent-led assessment workflow state machine."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum

from storage.event_store import EventStore


class WorkflowStage(Enum):
    """Ordered stages shared by all guided assessments."""
    INSTRUCTION = "instruction"
    POSITIONING = "positioning"
    SAMPLING = "sampling"
    SCORING = "scoring"
    QUESTIONS = "questions"
    CONCLUSION = "conclusion"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class WorkflowSession:
    """Live public-safe workflow state; samples remain memory-only."""
    protocol: str
    subject_id: str = "primary"
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stage: WorkflowStage = WorkflowStage.INSTRUCTION
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    stage_started_at: float = field(default_factory=time.time)
    retry_count: int = 0
    questions_asked: int = 0
    quality: float | None = None
    progress: float = 0.0
    message: str = ""
    deadline: float | None = None
    current_topic: str | None = None
    unclear_rephrased: bool = False
    denied_topics: tuple[str, ...] = ()
    answers: tuple[str, ...] = ()
    question_topics: tuple[str, ...] = ()
    asked_topics: tuple[str, ...] = ()
    score_summary: str = ""
    score_measurements: dict = field(default_factory=dict)


class WorkflowEngine:
    """Singleton enforcing one retry, three questions, and prompt cadence."""
    _instance = None

    def __init__(self, unsolicited_gap: float = 60.0, event_store: EventStore | None = None):
        self.unsolicited_gap = unsolicited_gap
        self.event_store = event_store or EventStore.instance()
        self._sessions: dict[str, WorkflowSession] = {}
        self._active_by_subject: dict[str, str] = {}
        self._last_unsolicited: dict[str, float] = {}
        self._lock = threading.RLock()

    @classmethod
    def instance(cls) -> "WorkflowEngine":
        """Return the shared workflow engine."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, protocol: str, subject_id: str = "primary", *,
              unsolicited: bool = False, now: float | None = None,
              timeout: float = 120.0,
              correlation_id: str | None = None) -> WorkflowSession | None:
        """Start a workflow unless the subject or interruption budget is busy."""
        now = time.time() if now is None else now
        with self._lock:
            if subject_id in self._active_by_subject:
                return None
            if unsolicited and now - self._last_unsolicited.get(subject_id, -1e9) < self.unsolicited_gap:
                return None
            session = WorkflowSession(protocol=protocol, subject_id=subject_id,
                                      started_at=now, updated_at=now,
                                      stage_started_at=now, deadline=now + timeout)
            if correlation_id:
                session.correlation_id = correlation_id
            self._sessions[session.correlation_id] = session
            self._active_by_subject[subject_id] = session.correlation_id
            if unsolicited:
                self._last_unsolicited[subject_id] = now
            self.event_store.record_assessment("started", protocol,
                subject_id=subject_id, source="workflow", correlation_id=session.correlation_id,
                timestamp=now)
            return session

    def active(self, subject_id: str = "primary") -> WorkflowSession | None:
        """Return a subject's active workflow, if any."""
        with self._lock:
            cid = self._active_by_subject.get(subject_id)
            return self._sessions.get(cid) if cid else None

    def get(self, correlation_id: str) -> WorkflowSession | None:
        """Return any session by correlation id, active or terminal.

        Sessions are retained after they leave `_active_by_subject`, so this
        lets a caller inspect the final stage of a workflow that already ended
        (concluded, cancelled, or timed out)."""
        with self._lock:
            return self._sessions.get(correlation_id)

    def transition(self, stage: WorkflowStage, *, subject_id: str = "primary",
                   message: str = "", quality: float | None = None,
                   progress: float | None = None) -> WorkflowSession | None:
        """Advance a session and publish only its safe state."""
        with self._lock:
            session = self.active(subject_id)
            if session is None:
                return None
            previous_stage, previous_progress = session.stage, session.progress
            now = time.time()
            if stage != session.stage:
                session.stage_started_at = now
            session.stage, session.message, session.updated_at = stage, message, now
            if quality is not None:
                session.quality = max(0.0, min(1.0, quality))
            if progress is not None:
                session.progress = max(0.0, min(1.0, progress))
            if stage != previous_stage or session.progress - previous_progress >= 0.1:
                self._event(session, "assessment", {"action": "stage", "stage": stage.value,
                                                     "message": message, "quality": session.quality,
                                                     "progress": session.progress})
            if stage in (WorkflowStage.CONCLUSION, WorkflowStage.CANCELLED,
                         WorkflowStage.TIMED_OUT):
                self._active_by_subject.pop(subject_id, None)
            return session

    def retry(self, reason: str, subject_id: str = "primary") -> bool:
        """Permit one neutral quality retry per workflow."""
        with self._lock:
            session = self.active(subject_id)
            if session is None or session.retry_count >= 1:
                return False
            session.retry_count += 1
            session.stage = WorkflowStage.POSITIONING
            session.stage_started_at = time.time()
            session.message = reason
            session.updated_at = time.time()
            return True

    def ask(self, topic: str, subject_id: str = "primary") -> bool:
        """Reserve one of at most three follow-up questions."""
        with self._lock:
            session = self.active(subject_id)
            if session is None or session.questions_asked >= 3:
                return False
            session.questions_asked += 1
            session.stage = WorkflowStage.QUESTIONS
            session.current_topic = topic
            session.unclear_rephrased = False
            session.updated_at = time.time()
            self.event_store.record_question(topic, session.questions_asked,
                subject_id=subject_id, source="workflow",
                correlation_id=session.correlation_id)
            return True

    def set_score(self, summary: str, measurements: dict, quality: float,
                  topics: tuple[str, ...], subject_id: str = "primary") -> bool:
        """Attach public-safe scoring output and enter the question stage."""
        with self._lock:
            session = self.active(subject_id)
            if session is None:
                return False
            session.score_summary = summary
            session.score_measurements = dict(measurements)
            session.quality = max(0.0, min(1.0, quality))
            session.question_topics = tuple(dict.fromkeys(topics))[:3]
            session.stage = WorkflowStage.QUESTIONS
            session.stage_started_at = session.updated_at = time.time()
            self._event(session, "assessment", {"action": "scored",
                "protocol": session.protocol, "summary": summary, "quality": session.quality})
            return True

    def next_question(self, subject_id: str = "primary") -> str | None:
        """Reserve the next non-denied topic, respecting the three-question cap."""
        with self._lock:
            topic = self.peek_question(subject_id)
            if topic is None:
                return None
            session = self.active(subject_id)
            session.asked_topics = (*session.asked_topics, topic)
            if not self.ask(topic, subject_id):
                return None
            return topic

    def peek_question(self, subject_id: str = "primary") -> str | None:
        """Return the next eligible topic without consuming the question budget."""
        with self._lock:
            session = self.active(subject_id)
            if session is None or session.stage != WorkflowStage.QUESTIONS \
                    or session.current_topic is not None or session.questions_asked >= 3:
                return None
            return next((topic for topic in session.question_topics
                         if topic not in session.asked_topics
                         and topic not in session.denied_topics), None)

    def conclude(self, subject_id: str = "primary") -> WorkflowSession | None:
        """Finish a scored workflow with its public-safe summary."""
        session = self.active(subject_id)
        if session is None:
            return None
        return self.transition(WorkflowStage.CONCLUSION, subject_id=subject_id,
                               message=session.score_summary, quality=session.quality,
                               progress=1.0)

    def answer(self, classification: str, subject_id: str = "primary") -> str:
        """Apply denial/unclear semantics and return the deterministic next action."""
        with self._lock:
            session = self.active(subject_id)
            if session is None or session.stage != WorkflowStage.QUESTIONS:
                return "ignored"
            value = classification.strip().lower()
            if value not in ("affirmed", "denied", "unclear"):
                value = "unclear"
            self.event_store.record_answer(value, subject_id=subject_id,
                source="workflow", correlation_id=session.correlation_id)
            session.answers = (*session.answers, value)
            if value == "denied" and session.current_topic:
                session.denied_topics = (*session.denied_topics, session.current_topic)
                session.current_topic = None
                return "suppressed"
            if value == "unclear" and not session.unclear_rephrased:
                session.unclear_rephrased = True
                return "rephrase"
            session.current_topic = None
            return "continue"

    def tick(self, now: float | None = None) -> list[WorkflowSession]:
        """Expire overdue workflows and return the sessions that timed out."""
        now = time.time() if now is None else now
        expired = []
        with self._lock:
            for subject_id, cid in list(self._active_by_subject.items()):
                session = self._sessions[cid]
                if session.deadline is not None and now >= session.deadline:
                    session.stage = WorkflowStage.TIMED_OUT
                    session.message = "Assessment timed out without enough usable information."
                    session.updated_at = now
                    self._active_by_subject.pop(subject_id, None)
                    self._event(session, "assessment", {"action": "timed_out",
                                                         "protocol": session.protocol})
                    expired.append(session)
        return expired

    def cancel(self, subject_id: str = "primary", reason: str = "cancelled") -> bool:
        """Cancel a workflow deterministically and retain only its safe reason."""
        return self.transition(WorkflowStage.CANCELLED, subject_id=subject_id,
                               message=reason) is not None

    def snapshot(self) -> list[dict]:
        """Return active workflow progress for the dashboard."""
        with self._lock:
            out = []
            for cid in self._active_by_subject.values():
                data = asdict(self._sessions[cid])
                data["stage"] = self._sessions[cid].stage.value
                out.append(data)
            return out

    def _event(self, session: WorkflowSession, kind: str, payload: dict) -> None:
        self.event_store.record(kind, payload, subject_id=session.subject_id,
                                source="workflow", correlation_id=session.correlation_id)
