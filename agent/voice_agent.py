"""VoiceAgent: orchestrates memory -> policy -> Moondream -> speech, plus ears.

Call `tick(snapshot)` once per frame (cheap; rate-limited internally). The agent
updates its memory, lets the policy pick at most one thing to say, phrases it
with Moondream (falling back to a templated line offline), speaks it via TTS, and
records it so it won't repeat. The last utterance is exposed for the dashboard.

With a `Listener` (audio/stt.py) attached, the loop closes: heard utterances
enter the dialogue memory, answer pending corroboration questions
(agent/corroboration.py — flags only become conclusions when the person
confirms them), can trigger scripted tests ("check my hands" ->
core/elicitation.py window that modules/tremor.py samples), and otherwise get
a warm contextual reply. Replies and confirmed conclusions bypass the normal
speaking gap so conversation feels responsive.
"""
from __future__ import annotations

import time
import uuid
import re
from dataclasses import replace

from agent.answers import (
    CLASSIFICATION_DEADLINE, PendingClassification, interpret_answer,
)
from agent.state import ObservationMemory
from agent.policy import Intent, Policy
from agent.corroboration import (CorroborationEngine, enforce_second_person,
                                  safe_check_in, strip_prior_disclosure)
from agent.moondream_client import MoondreamClient
from agent.conversation import (
    AgentContextBroker, AgentResponse, ConversationTurn, ProposedAction,
    TopicCandidate, TopicQueue, VisionCadence, guard_agent_only_speech,
)
from agent.skin_dialogue import SkinDialogue
from agent.topics import (
    SUPPORT_TRAILING_CHECK_IN, covered_keys, support_clause,
)
from audio.tts import Speaker
from core.elicitation import ElicitationState
from core.workflows import WorkflowEngine, WorkflowStage
from assessments import PROTOCOLS
from storage.event_store import EventStore
from core.events import PersistencePolicy, Result, Severity

_TEST_TRIGGERS = ("check my hand", "check my tremor", "test my hand",
                  "test my tremor", "am i shaking", "tremor test",
                  "check my hands", "test my hands")
_ARM_TRIGGERS = ("check my arm", "check my arms", "look at my arm",
                 "check my skin", "look at my skin", "arm check")
_HOLD_STILL_SECONDS = 10.0     # ~2 s to comply + 8 s of sampling
_ARM_CHECK_SECONDS = 12.0      # ~2 s positioning + 10 s of arm sampling
_ARM_CHECK_SESSION_TIMEOUT = _ARM_CHECK_SECONDS + 65.0
_MISSING_ACTION = object()
# Appearance cues that may be named alongside one corroboration check-in, in
# the order they are listed. Only cues currently visible are mentioned.
_CHECK_IN_CUES = {
    "cold_symptoms": ("nose_redness", "cheek_redness", "nasal_discharge_visible"),
    "hydration": ("lip_dryness",),
    "tiredness_pallor": ("under_eye_darkness", "under_eye_puffiness"),
}
_ASSESSMENT_QUESTIONS = {
    "symptoms": "Did you notice any discomfort, weakness, dizziness, or other symptoms during that?",
    "progression": "Has this movement or task changed recently compared with what is normal for you?",
    "warning_signs": "Are you feeling suddenly unwell, faint, confused, or having trouble speaking right now?",
}
# Short, low-risk guided assessments chained back-to-back for a live guest/
# client demo ('d' hotkey or --demo): each is brief and needs no equipment.
_DEMO_CIRCUIT = ("facial_movement", "arm_drift", "balance")
# A demo step ending in one of these never captured a usable result; the
# circuit must not narrate the next step as if it had succeeded.
_DEMO_FAILED_STAGES = (WorkflowStage.CANCELLED, WorkflowStage.TIMED_OUT)
_DEMO_MAX_RETRIES = 1          # circuit-level retries per step, beyond the
                               # positioner's own one internal reposition retry
# Guest-facing names and corrective framing for a step that failed to capture.
# Framing only (where to stand / what to face) — never a clinical claim (§7).
_DEMO_STEP_LABELS = {
    "facial_movement": "face check",
    "arm_drift": "arm check",
    "balance": "balance check",
}
# Vocabulary the classify lane is allowed to return. `interpret_answer`'s
# "affirmed" is spelled "confirmed" by the corroboration state machine and by
# MoondreamClient.classify_answer; the action/workflow lanes map it back.
_CLASSIFY_VERDICTS = ("confirmed", "denied", "unclear")
# Intent kinds whose lines are politely phrased as questions ("Could you hold a
# hand out flat...?") but are answered by MOVING, not by speaking: the capture
# window and the module's result close them. Waiting for a spoken reply after
# one would suppress the corrective re-prompt the capture may need next.
_PHYSICAL_PROMPT_KINDS = ("elicit_test", "instruction")
_HELP_PHRASES = ("help me", "call for help", "call emergency")
_HESITATION_WORDS = frozenset({"um", "uh", "hmm", "hm", "er", "ah", "well"})
_DEMO_STEP_GUIDANCE = {
    "facial_movement": "Please face the camera in even light, then we'll try again.",
    "arm_drift": "Please step back so your whole upper body and both arms are in view, "
                 "then we'll try again.",
    "balance": "Please step back so your full body is in view, then we'll try again.",
}


class VoiceAgent:
    """Orchestrates memory -> policy -> Moondream -> speech (and listening)."""
    def __init__(self, name: str = "there", speak: bool = True,
                 model: str | None = None, listener=None,
                 moondream_enabled: bool = True,
                 vision_enabled: bool = False,
                 tts_engine: str = "auto",
                 **policy_kwargs):
        self.memory = ObservationMemory(name=name)
        self.memories: dict[str, ObservationMemory] = {"primary": self.memory}
        self.policy = Policy(**policy_kwargs)
        # Turn-taking: how long a question the agent asked on an *untracked*
        # lane (small talk, a topic fallback ending in "?") holds the floor.
        # Tuned from the same `conversation:` block that configures Policy.
        conversation_cfg = dict(policy_kwargs.get("conversation") or {})
        self.reply_window = float(conversation_cfg.get("reply_window", 30.0))
        self._awaiting_reply_until = 0.0
        # Emergency episodes are intentionally separate from AlertManager.
        # These are the two narrow cases that may speak over an answer window;
        # all other alerting stays on its caregiver/external path.
        self._active_emergencies: set[str] = set()
        self._emergency_episodes: dict[str, int] = {}
        self._pending_help_alert = False
        # A "gemini*" voice model routes to Gemini's OpenAI-compatible endpoint
        # (GeminiClient duck-types MoondreamClient exactly). moondream3-preview is
        # a small vision model whose open-conversation output degenerates into
        # echoing the controller/context; gemini-2.5-flash produces coherent,
        # instruction-following speech. Falls back to Moondream for any other id.
        _m = str(model or "")
        if _m.lower().startswith("gemini"):
            from agent.gemini_client import GeminiClient  # noqa: PLC0415 - optional path
            self.moondream = GeminiClient(model=_m, enabled=moondream_enabled)
        elif _m.lower().startswith("nvidia:"):
            # "nvidia:<model-id>" routes to NVIDIA's hosted OpenAI-compatible
            # endpoint (higher free limits than Gemini). e.g. the id after the
            # colon: "nvidia:meta/llama-3.3-70b-instruct".
            from agent.nvidia_client import NvidiaClient  # noqa: PLC0415 - optional path
            self.moondream = NvidiaClient(model=(_m.split(":", 1)[1] or None),
                                          enabled=moondream_enabled)
        else:
            self.moondream = MoondreamClient(model=model, enabled=moondream_enabled)
        self.speaker = Speaker(enabled=speak, engine=tts_engine)
        self.listener = listener
        # Answer interpretation runs synchronously inside tick(), so keep it
        # deterministic/local; only natural-language generation uses cloud.
        self.corroboration = CorroborationEngine(language_model=None)
        self.skin_dialogue = SkinDialogue(language_model=None)
        self.elicitation = ElicitationState.instance()
        self.workflows = WorkflowEngine.instance()
        self.events = EventStore.instance()
        self.last_utterance = ""
        self._test_requested = False
        self._arm_check_requested = False
        self._arm_check_session_id: str | None = None
        self._arm_check_attempt = 0
        self._arm_check_in_flight = False
        self._arm_check_in_flight_until = 0.0
        self._arm_check_followup_requested = False
        self._demo_queue: list[str] = []
        self._demo_subject: str = "primary"
        self._demo_active_cid: str | None = None    # correlation id of the running step
        self._demo_active_protocol: str | None = None
        self._demo_retries: dict[str, int] = {}     # per-protocol circuit-level retries used
        self._showcase_pending_close = False        # narrated tour awaiting wrap-up
        self._actions: dict = {}       # intent signature -> post-speech callback
        self._safety_results: list[Result] = []
        self._conversation_results: list[Result] = []
        self._pending_speech: tuple[str, object, float, float, object] | None = None
        self._pending_response: tuple[str, object, float, float, object,
                                      TopicCandidate | None, list] | None = None
        self._pending_vision: str | None = None
        self.context_broker = AgentContextBroker()
        self.topic_queue = TopicQueue()
        self.vision = VisionCadence()
        self.vision_enabled = bool(vision_enabled)
        self._capability_source = None
        self._history_source = None
        self._pending_action: ProposedAction | None = None
        # One bounded re-ask on the action lane, named after the equivalents
        # already in WorkflowSession.unclear_rephrased and TopicState.asks.
        self._pending_action_unclear_rephrased = False
        # Exactly ONE in-flight answer classification, matching the provider's
        # single classify-lane future so the two can never disagree.
        self._pending_classification: PendingClassification | None = None
        self._last_heard_at = 0.0
        self._classification_dropped_stale = 0
        self._classification_dropped_deadline = 0
        self._classification_superseded = 0
        # (request_id, flagged-topic signature) for the async steer selection.
        self._pending_topic_selection: tuple[str, tuple[str, ...]] | None = None
        self._last_steered_topic: TopicCandidate | None = None
        self._action_ack: tuple[str, float] | None = None
        self._last_conversation_latency_ms: float | None = None
        self._last_fallback_reason: str | None = None
        self._cloud_speech_deadline = 0.75
        # Structured-response deadline (seconds of wall-clock before the
        # templated fallback is spoken). Kept an instance attribute so a demo
        # or test can give a slow provider more time to phrase a real answer.
        self._response_deadline = 4.0
        moondream = self.moondream.status()
        mode = ("Moondream" if moondream["active"] else
                "Moondream disabled (templated)" if moondream["available"] else "templated")
        ears = "listening" if (listener is not None and
                               getattr(listener, "available", False)) else "speak-only"
        print(f"[agent] voice agent ready (name={name}, speech={mode}, {ears})")

    def set_moondream_enabled(self, enabled: bool) -> dict:
        """Control future Moondream calls without disabling templated speech."""
        self.moondream.set_enabled(enabled)
        return self.moondream.status()

    def toggle_moondream(self) -> dict:
        """Toggle future Moondream calls and return the diagnostic state."""
        self.moondream.toggle_enabled()
        return self.moondream.status()

    def moondream_status(self) -> dict:
        """Return safe Moondream transport diagnostics."""
        return self.moondream.status()

    def set_context_sources(self, *, capabilities=None, history=None) -> None:
        """Attach public-safe runtime sources used only when composing a turn."""
        self._capability_source = capabilities
        self._history_source = history

    def observe_frame(self, frame, snapshot: list[Result],
                      now: float | None = None) -> None:
        """Offer one ephemeral frame to the change-driven vision cadence."""
        now = time.time() if now is None else now
        if not self.vision_enabled:
            return
        present = any(
            result.subject_id == "primary" and result.module == "presence"
            and (result.key == "arrival" or bool(result.value))
            for result in snapshot)
        conversation_active = bool(self.listener is not None or self.last_utterance
                                   or self.memory.dialogue)
        self.vision.observe(frame, active=present and conversation_active, now=now)

    def public_line(self) -> dict:
        """The agent's own last spoken line for demo captions.

        Agent-authored text only — never a heard transcript, question id, or
        private value — so it is safe on every public surface.
        """
        return {"text": self.last_utterance,
                "speaking": bool(self.speaker.speaking),
                "listening": self._can_hear()}

    def conversation_diagnostics(self) -> dict:
        """Return private-safe orchestration state without text, values, or media."""
        pending = None
        if self._pending_action is not None:
            pending = {"action": self._pending_action.action,
                       "target": self._pending_action.target,
                       "expires_at": self._pending_action.expires_at}
        return {"context": self.context_broker.diagnostics(),
                "topics": self.topic_queue.diagnostics(),
                "vision": {"enabled": self.vision_enabled,
                           **self.vision.diagnostics()},
                "pending_action": pending,
                # Counters and the lane name ONLY: never the question, never
                # the transcript, never a question id (they embed free text).
                "classification": {
                    "pending": self._pending_classification is not None,
                    "lane": (self._pending_classification.lane
                             if self._pending_classification is not None else None),
                    "dropped_stale": self._classification_dropped_stale,
                    "dropped_deadline": self._classification_dropped_deadline,
                    "superseded": self._classification_superseded},
                "conversation_latency_ms": self._last_conversation_latency_ms,
                "fallback_reason": self._last_fallback_reason,
                "session_turns": len(self.memory.dialogue)}

    def request_test(self, test: str = "hold_still") -> None:
        """Queue a scripted test (the 't'/'a' hotkey path)."""
        if test in PROTOCOLS:
            self.workflows.start(test)
        elif test == "arm_check":
            self._queue_arm_check()
        else:
            self._test_requested = True

    def _queue_arm_check(self) -> None:
        """Queue one new manual arm-check session."""
        if self._arm_check_requested:
            return
        if (self._arm_check_in_flight
                or self.elicitation.active("arm_check")):
            # Preserve one explicit follow-up instead of opening a capture
            # window that SkinVision cannot sample while the prior analysis
            # still owns the manual lane.
            self._arm_check_followup_requested = True
            if not self._arm_check_in_flight:
                self._arm_check_in_flight = True
                self._arm_check_in_flight_until = (
                    time.monotonic() + _ARM_CHECK_SESSION_TIMEOUT)
            return
        self._arm_check_session_id = uuid.uuid4().hex
        self._arm_check_attempt = 0
        self._arm_check_requested = True

    def _finish_arm_check_session(self) -> None:
        """Release one terminal session and schedule one deferred request."""
        if not self._arm_check_in_flight:
            return
        if (self.elicitation.test == "arm_check"
                and self.elicitation.correlation_id
                == self._arm_check_session_id):
            self.elicitation.clear()
        self._arm_check_in_flight = False
        self._arm_check_in_flight_until = 0.0
        if self._arm_check_followup_requested:
            self._arm_check_followup_requested = False
            self._queue_arm_check()

    def _expire_arm_check_session(self) -> None:
        """Recover if a module never publishes the terminal manual result."""
        if (self._arm_check_in_flight
                and time.monotonic() >= self._arm_check_in_flight_until):
            self._finish_arm_check_session()

    def start_demo_circuit(self, subject_id: str = "primary") -> bool:
        """Queue a short scripted tour of guided assessments (the 'd' hotkey /
        --demo path). Ignored if that subject already has an assessment
        running or a circuit already queued."""
        if self._demo_queue or self.workflows.active(subject_id) is not None:
            return False
        self._demo_subject = subject_id
        self._demo_queue = list(_DEMO_CIRCUIT)
        self._demo_active_cid = None
        self._demo_active_protocol = None
        self._demo_retries = {}
        self._showcase_pending_close = False
        self._advance_demo_circuit()
        return True

    def start_showcase(self, subject_id: str = "primary") -> bool:
        """Start the narrated showcase tour (the 's' hotkey / --showcase path).

        A spoken introduction, then the guided demo circuit, then a closing
        wrap-up once the last step has ended. Returns False if a circuit is
        already running or an assessment is active for that subject.
        """
        if not self.start_demo_circuit(subject_id):
            return False
        self._say_demo(
            "Hi there! Let me show you what I can help with around here. "
            "We'll try three quick checks together: face, arms, and balance.")
        self._showcase_pending_close = True
        return True

    def _say_demo(self, text: str) -> None:
        """Speak a circuit line and expose it as the last utterance.

        Mirrors the main speak path so the dashboard and tests can read the
        latest circuit narration off `last_utterance`."""
        self.last_utterance = text
        self.speaker.say(text)

    def _start_demo_step(self, protocol: str, prompt: str) -> None:
        """Begin one demo protocol and speak the given prompt, or clear the
        circuit if the workflow engine refuses to start it."""
        session = self.workflows.start(protocol, subject_id=self._demo_subject)
        if session is None:
            self._demo_queue.clear()
            self._demo_active_cid = None
            self._demo_active_protocol = None
            return
        self._demo_active_cid = session.correlation_id
        self._demo_active_protocol = protocol
        self._say_demo(prompt)

    def _advance_demo_circuit(self) -> None:
        """Start the next queued demo step once the previous one has ended.

        A step that ends without capturing (cancelled or timed out) is not
        narrated as a success: the circuit announces the miss, tells the person
        how to reposition, and retries that step once (in a single spoken line
        that also restates the instruction). If it fails again the step is
        honestly skipped rather than silently passed over."""
        if self.workflows.active(self._demo_subject) is not None:
            return
        # Handle the outcome of the step that just ended before the empty-queue
        # check, so even the final step's failure is announced.
        if self._demo_active_cid is not None:
            ended = self.workflows.get(self._demo_active_cid)
            protocol = self._demo_active_protocol
            self._demo_active_cid = None
            self._demo_active_protocol = None
            if ended is not None and protocol is not None and ended.stage in _DEMO_FAILED_STAGES:
                label = _DEMO_STEP_LABELS.get(protocol, "that step")
                if self._demo_retries.get(protocol, 0) < _DEMO_MAX_RETRIES:
                    self._demo_retries[protocol] = self._demo_retries.get(protocol, 0) + 1
                    guidance = _DEMO_STEP_GUIDANCE.get(
                        protocol, "Please step fully into view, then we'll try again.")
                    self._start_demo_step(protocol,
                        f"I couldn't capture the {label}. {guidance} "
                        f"{PROTOCOLS[protocol].instruction}")
                    return
                # Announce the skip on its own; the next tick starts the next
                # step, keeping each spoken line distinct.
                self._say_demo(f"Skipping the {label} — I couldn't capture it this time.")
                return
        if not self._demo_queue:
            if (self._showcase_pending_close
                    and self.workflows.active(self._demo_subject) is None):
                # Tour finished (any mix of captured and honestly-skipped
                # steps): close it warmly, exactly once.
                self._showcase_pending_close = False
                self._say_demo(
                    "And that's the little tour! I'll keep watching quietly "
                    "and check in now and then. Thank you for trying these "
                    "with me.")
            return
        protocol = self._demo_queue.pop(0)
        remaining = len(self._demo_queue)
        tail = f" ({remaining} more to go)" if remaining else " (last one)"
        self._start_demo_step(protocol, f"Demo: {PROTOCOLS[protocol].instruction}{tail}")

    # ---------------------------------------------------------------- ears

    @staticmethod
    def _affirmation(text: str) -> str:
        """Classify a reply as affirmed/denied/unclear (shared classifier)."""
        return interpret_answer(text)

    def _proposed_action_for(self, text: str, now: float) -> ProposedAction | None:
        """Map speech to an allowlisted inert proposal; never execute here."""
        low = text.lower()
        if any(trigger in low for trigger in _ARM_TRIGGERS):
            return ProposedAction("arm_check", None,
                "The person asked for a closer arm or skin check.", now, now + 30.0)
        if any(trigger in low for trigger in _TEST_TRIGGERS):
            return ProposedAction("assessment", "hold_still",
                "The person asked for a hand movement check.", now, now + 30.0)
        if any(phrase in low for phrase in ("look again", "take another look",
                                             "refresh the camera", "check the camera")):
            return ProposedAction("vision_refresh", None,
                "The person requested a fresh camera observation.", now, now + 30.0)
        for protocol in PROTOCOLS:
            label = protocol.replace("_", " ")
            if label in low and any(term in low for term in ("check", "test", "assess")):
                return ProposedAction("assessment", protocol,
                    f"The person asked about the {label} assessment.", now, now + 30.0)
        return None

    def _execute_confirmed_action(self, proposal: ProposedAction) -> None:
        """Execute only locally allowlisted actions after explicit confirmation."""
        if proposal.action == "arm_check":
            self._queue_arm_check()
        elif proposal.action == "vision_refresh":
            self.vision.force_refresh()
        elif proposal.action == "assessment":
            target = proposal.target or "hold_still"
            if target in PROTOCOLS:
                self.workflows.start(target)
            elif target == "hold_still":
                self._test_requested = True

    def _dialogue_turns(self) -> list[ConversationTurn]:
        return [ConversationTurn("user" if who == "them" else "assistant",
                                 text, ts)
                for who, text, ts in self.memory.dialogue[-20:]]

    def _turn_context(self, query: str):
        capabilities = (self._capability_source.snapshot()
                        if self._capability_source is not None else [])
        try:
            recent_events = self.events.recent(12, subject_id="primary")
        except Exception:
            recent_events = []
        trends = []
        if self._history_source is not None:
            query_terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
            history_items = sorted(
                self.context_broker.items(),
                key=lambda item: len(query_terms & set(re.findall(
                    r"[a-z0-9_]+", f"{item.module} {item.key} {item.message}".lower()))),
                reverse=True)
            for item in history_items:
                if len(trends) >= 12 or not isinstance(item.value, (int, float)) \
                        or isinstance(item.value, bool):
                    continue
                try:
                    mean_day = self._history_source.mean_since(
                        item.module, item.key, 86400, item.subject_id)
                    mean_week = self._history_source.mean_since(
                        item.module, item.key, 7 * 86400, item.subject_id)
                except Exception:
                    continue
                if mean_day is None and mean_week is None:
                    continue
                trends.append({"subject_id": item.subject_id, "module": item.module,
                               "key": item.key, "mean_24h": mean_day,
                               "mean_7d": mean_week,
                               "note": "Numeric observational history; not a diagnosis."})
        return self.context_broker.build(
            query, self._dialogue_turns(), workflows=self.workflows.snapshot(),
            capabilities=capabilities, recent_events=recent_events,
            history_trends=trends)

    @staticmethod
    def _turn_messages(intent: Intent, context, topic: TopicCandidate | None) -> list[dict]:
        messages = [{"role": turn.role, "content": turn.text}
                    for turn in context.turns]
        instruction = intent.llm_intent
        if intent.detail:
            instruction += " Supporting detail: " + intent.detail
        if intent.signature.startswith(("ask:", "conclude:")):
            # Corroboration is "ask, don't announce the prior": the low-confidence
            # visual cue is in the observation context, but the person must never
            # hear it named. Phrase only the gentle line; never describe, announce,
            # or hint at any observation, sensor reading, camera, or location.
            instruction += (
                " Rephrase the supporting detail as one warm, natural spoken line "
                "addressed to the person as 'you'. Do NOT describe, announce, quote, "
                "or hint at any observation, sensor reading, camera, or room/location "
                "— only say the line itself.")
        if topic is not None:
            instruction += (" First answer the person's current question completely. "
                            "Only afterward, if natural, transition with 'By the way' and "
                            f"{topic.prompt} Context item: {topic.context_id}.")
        messages.append({"role": "user", "content": (
            "Conversation controller instruction (not spoken verbatim): " + instruction)})
        return messages

    # Meta/echo phrases a spoken line must never contain. A weak local model
    # sometimes parrots the controller scaffolding ("Conversation controller
    # instruction...", "The controller should respond with...") or emits a list
    # of stage directions instead of the line itself; speaking that reads the
    # status out loud. Reject those and let the hand-authored fallback speak.
    _META_MARKERS = (
        "conversation controller", "not spoken verbatim", "controller should",
        "controller instruction", "audio controller", "controller asked",
        "the controller", "supporting detail", "context item",
        "rephrase", "spoken line", "addressed to the person", "as 'you'",
        "sensor reading", "do not describe", "do not announce",
        # Narrating the session/context instead of speaking to the person.
        "the conversation start", "the conversation beg", "this conversation",
        "person arrived", "has arrived", "just arrived", "entered the room",
        "was detected", "the system", "recent event", "observation:",
        "would you like to talk about", "like to discuss this",
    )

    # The pipeline's capitalized subject labels. If one appears mid-line the
    # model is reading an event/status ("By the way, Person arrived"), not speech.
    _SUBJECT_LABELS = ("Person", "Subject", "Resident", "User", "Track", "Primary")

    @classmethod
    def _clean_spoken_line(cls, generated: str | None) -> str:
        """Return the line only if it reads like speech, else "" to force the
        deterministic fallback. Deterministic-disposes over the LLM proposal."""
        if not generated:
            return ""
        text = " ".join(str(generated).split()).strip()
        # Some models wrap the spoken line in quotes ("Good morning, ...");
        # strip a wrapping pair so the person doesn't hear stray quote marks.
        wrap_quotes = "\"'" + "".join(chr(c) for c in (0x201c, 0x201d, 0x2018, 0x2019))
        text = text.strip(wrap_quotes).strip()
        if not text:
            return ""
        low = text.lower()
        if any(marker in low for marker in cls._META_MARKERS):
            return ""
        # A conversational turn is one or two sentences. Runaway length or a
        # long repeated clause is model degeneration, not something to speak.
        if len(text) > 320:
            return ""
        words = low.split()
        for span in (6, 5, 4):
            if len(words) >= span * 3:
                phrase = " ".join(words[:span])
                if low.count(phrase) >= 3:
                    return ""
        # A list of stage directions to the controller ("Ask about... Describe
        # a... Offer...") is not a spoken line. Two or more sentences opening
        # with a directive verb is the degenerate pattern.
        directive_leads = {"ask", "describe", "offer", "invite", "make", "share",
                           "mention", "discuss", "suggest", "provide", "gently",
                           "respond", "acknowledge", "encourage", "prompt"}
        sentences = [s.strip() for s in
                     text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        lead_hits = sum(1 for s in sentences
                        if s.split() and s.split()[0].lower().strip(",") in directive_leads)
        # A spoken line addresses the person as "you"; a stage direction talks
        # ABOUT them ("ask how their day is going", "invite them to chat"). A
        # directive-led sentence that references the person in third person is
        # the controller's instruction leaking through, not speech.
        third_person = any(t in f" {low} " for t in
                           (" their ", " them ", " they ", "the person",
                            "the user", "the resident"))
        if lead_hits >= 2 or (lead_hits >= 1 and third_person):
            return ""
        # A capitalized internal subject label ("Person", "Resident", ...) is an
        # event/status echo, never speech. Case-sensitive so "a lovely person"
        # (lowercase) still passes.
        tokens = [w.strip(".,!?;:'\"()") for w in text.split()]
        if any(tok in cls._SUBJECT_LABELS for tok in tokens):
            return ""
        # Deterministic backstop for raw sensor readings the persona forbids: a
        # number next to a unit/metric term ("9.6 breaths per minute", "tint of
        # 0.0", "62 bpm", "95%"). Adjacency keeps ordinary numbers ("see you at
        # 3", "take 3 deep breaths") passing.
        import re  # noqa: PLC0415
        if (re.search(r"\d[\d.,]*\s*(?:%|bpm|beats?\b|breaths?\b|per\s*minute|"
                      r"percent|degrees?|celsius|fahrenheit)", low)
                or re.search(r"(?:tint|saturation|spo2|heart rate|breathing rate|"
                             r"respiration)\s*(?:of|is|at|:|=)?\s*\d", low)):
            return ""
        return text

    def _can_hear(self) -> bool:
        """True when an ASR listener exists and is currently able to hear.

        `available` defaults to True: TypedListener/ReplayListener do not
        publish the attribute, and only an explicit False (audio/stt.py flips
        it at runtime) should be read as "no ears".
        """
        return (self.listener is not None
                and bool(getattr(self.listener, "available", True)))

    def _awaiting_answer(self, now: float) -> bool:
        """True while a question the agent asked is still within its answer window.

        Every lane that owns an outstanding question is consulted here, so the
        agent holds the floor for the person instead of talking over them. Each
        lane must expire on its own — a status that never lapses (corroboration
        keeps `status == "asked"` forever) would mute the agent permanently,
        which is why this asks `pending_question`, not the status.
        """
        if not self._can_hear():
            return False          # no ears: waiting would mute us forever
        if self.corroboration.pending_question(now) is not None:
            return True
        # NOTE: `_pending_action` is deliberately NOT consulted. A proposal is
        # created the moment the request is *heard*, one tick before the
        # confirmation question is spoken, so waiting on it would silence that
        # very question. The action lane is covered by the generic wait below:
        # its confirmation line and its one bounded re-ask both end in "?".
        if self.skin_dialogue.awaiting_answer(now):
            return True
        workflow = self.workflows.active("primary")
        # `unclear_rephrased` means the agent still owes this person a plainer
        # re-ask of the SAME question; that re-ask must not be gated, exactly as
        # an unclear corroboration answer reverts the topic to "flagged".
        if (workflow is not None and workflow.current_topic is not None
                and not workflow.unclear_rephrased):
            return True
        return self._awaiting_reply_until > now

    def _low_information_utterance(self, text: str, now: float) -> bool:
        """Whether speech is just a thinking sound, not a conversational turn.

        Clear yes/no answers, safety calls, and allowlisted requests always
        count, even when short. Only actual filler keeps the current answer
        window open; brevity alone never does, because "my back hurts" is three
        words and is a turn the person is entitled to have heard.
        """
        low = text.lower()
        words = re.findall(r"[a-z']+", low)
        # "okay" may be a genuine brief confirmation, but the trailing
        # thinking filler in "okay then" is not an answer to a health prompt.
        if len(words) <= 3 and (all(w in _HESITATION_WORDS for w in words)
                                or words == ["okay", "then"]):
            return True
        if any(phrase in low for phrase in _HELP_PHRASES):
            return False
        if interpret_answer(text) != "unclear":
            return False
        if self._proposed_action_for(text, now) is not None:
            return False
        # An empty transcript (words == []) satisfies this vacuously and stays
        # low-information, which is what a dropped ASR segment should be.
        return len(words) <= 4 and all(w in _HESITATION_WORDS for w in words)

    def _emergency_intents(self, snapshot) -> list[Intent]:
        """Return only confirmed emergency speech candidates for this tick.

        A positive fall result starts one episode until the signal clears. An
        explicit spoken help request is queued once by `_consume_heard`.
        Generic ALERT results deliberately do not enter this path.
        """
        active = set()
        candidates: list[Intent] = []
        if any(result.subject_id == "primary" and result.module == "fall"
               and result.key == "fall" and bool(result.value)
               and result.severity == Severity.ALERT for result in snapshot):
            active.add("fall")
        if self._pending_help_alert:
            active.add("explicit_help")
        for name in active:
            if name not in self._active_emergencies:
                self._emergency_episodes[name] = self._emergency_episodes.get(name, 0) + 1
            episode = self._emergency_episodes[name]
            if name == "fall":
                fallback = "I detected a fall. Please call for help now."
            else:
                fallback = "I heard you ask for help. Please call for help now."
            candidates.append(Intent(
                "urgent_alert", f"urgent:{name}:{episode}",
                "State this confirmed emergency clearly and briefly.", "", fallback,
                1000, health_prompt=False))
        self._active_emergencies = active
        return candidates

    def _consume_heard(self, now: float) -> tuple[list, bool]:
        """Drain the listener into memory/corroboration.

        Returns (heard utterances, whether one confirmed a pending topic —
        in which case the conclusion intent is the reply, not a generic one).
        """
        if self.listener is None:
            return [], False
        handled = False
        heard = self.listener.pop_utterances()
        for text, ts in heard:
            self.memory.person_said(text, ts)
            self._last_heard_at = max(self._last_heard_at, float(ts))
            if self._low_information_utterance(text, now):
                # Do not turn "um" into an unclear answer/re-ask. The person
                # retains the floor and the original deadline keeps running.
                continue
            # A meaningful utterance, including a direct request, ends only
            # the generic conversational wait. Tracked lanes update below.
            self._awaiting_reply_until = 0.0
            if (self._pending_classification is not None
                    and interpret_answer(text) != "unclear"):
                # A clear human yes/no beats a speculative model verdict
                # outright: deterministic wins, and the slot is freed now.
                self._cancel_pending_classification(superseded=True)
            low = text.lower()
            if any(phrase in low for phrase in _HELP_PHRASES):
                self._safety_results.append(Result(
                    "explicit_help", "call_for_help", True, .95, Severity.ALERT,
                    "The person explicitly called for help", ttl=20,
                    source="speech_recognition", quality=.95,
                    persistence=PersistencePolicy.EVENT))
                self._pending_help_alert = True
            action_unclear = False
            if self._pending_action is not None:
                if now > self._pending_action.expires_at:
                    self._pending_action = None
                    self._pending_action_unclear_rephrased = False
                else:
                    answer = self._affirmation(text)
                    if answer == "affirmed":
                        self._confirm_pending_action(now)
                        handled = True
                        continue
                    if answer == "denied":
                        self._pending_action = None
                        self._pending_action_unclear_rephrased = False
                        self._action_ack = ("Okay, I won't start it.", now)
                        handled = True
                        continue
                    # Neither yes nor no. Resolved below, AFTER the chance to
                    # recognize a differently-worded new request.
                    action_unclear = True
            if self._last_steered_topic is not None \
                    and now <= self._last_steered_topic.expires_at \
                    and self._affirmation(text) == "denied":
                self.topic_queue.suppress(self._last_steered_topic, "user_denied")
                self._last_steered_topic = None
                handled = True
                continue
            proposal = self._proposed_action_for(text, now)
            if proposal is not None:
                self._pending_action = proposal
                self._pending_action_unclear_rephrased = False
                handled = True
                continue
            if action_unclear:
                if not self._submit_classification(
                        "action", self._action_question_id(self._pending_action),
                        self._pending_action.target or self._pending_action.action,
                        self._action_question(self._pending_action), text, now, ts):
                    self._action_unclear(now)
                handled = True
                continue
            active_workflow = self.workflows.active("primary")
            if active_workflow is not None:
                response = interpret_answer(text)
                if (active_workflow.stage == WorkflowStage.QUESTIONS
                        and active_workflow.current_topic is not None):
                    answered_topic = active_workflow.current_topic
                    if response == "unclear" and self._submit_classification(
                            "workflow",
                            f"workflow-question:{active_workflow.correlation_id}:"
                            f"{answered_topic}", answered_topic,
                            _ASSESSMENT_QUESTIONS.get(
                                answered_topic, "How did that task feel for you?"),
                            text, now, ts):
                        handled = True
                    else:
                        self._apply_workflow_answer(
                            active_workflow, answered_topic, response)
                        handled = True
            skin_answered = self.skin_dialogue.hear(text, now)
            answered = None
            if not skin_answered:
                pending_question = self.corroboration.pending_question(now)
                if pending_question is not None:
                    topic, rule = pending_question
                    verdict = self.corroboration.classify(topic, text)
                    asks = self.corroboration.topics[topic].asks
                    if verdict == "unclear" and self._submit_classification(
                            "corroboration", f"ask:{topic}:{asks}", topic,
                            rule.question, text, now, ts):
                        handled = True      # the state machine waits for it
                    else:
                        answered = self.corroboration.apply_verdict(
                            topic, verdict, now)
            if answered is not None and answered[1] in ("confirmed", "denied"):
                self._record_corroboration_answer(*answered)
            if skin_answered or answered is not None:
                handled = True
        return heard, handled

    # ------------------------------------------------ async classification

    def _confirm_pending_action(self, now: float) -> None:
        """Start the confirmed proposal (allowlist unchanged) and clear state."""
        proposal, self._pending_action = self._pending_action, None
        self._pending_action_unclear_rephrased = False
        self._execute_confirmed_action(proposal)
        if proposal.action == "vision_refresh":
            self._action_ack = ("Okay, I'll take a fresh look.", now)

    def _action_unclear(self, now: float) -> None:
        """One bounded re-ask on the action lane, then give up.

        Mirrors WorkflowSession.unclear_rephrased and TopicState.asks: exactly
        one gentle retry, never a loop. `ProposedAction` is frozen, so the
        confirmation window is extended by replacing the proposal — `created_at`
        stays put, which keeps the question id (and so the async staleness
        guard) pointing at the same question.
        """
        proposal = self._pending_action
        if proposal is None:
            return
        if not self._pending_action_unclear_rephrased:
            self._pending_action_unclear_rephrased = True
            self._pending_action = replace(proposal, expires_at=now + 30.0)
            return
        self._pending_action = None
        self._pending_action_unclear_rephrased = False
        self._action_ack = ("Okay, I'll leave it for now.", now)

    @staticmethod
    def _action_question_id(proposal: ProposedAction) -> str:
        return (f"confirm-action:{proposal.action}:{proposal.target}:"
                f"{int(proposal.created_at)}")

    @staticmethod
    def _action_question(proposal: ProposedAction) -> str:
        target = (proposal.target or proposal.action).replace("_", " ")
        return f"Would you like me to start the {target} now?"

    def _record_corroboration_answer(self, topic: str, verdict: str) -> None:
        self._conversation_results.append(Result(
            "conversation", f"{topic}_{verdict}", True, 1.0, Severity.INFO,
            f"User {verdict} {topic.replace('_', ' ')}", ttl=120,
            source="user_answer", persistence=PersistencePolicy.EVENT))
        # Deliberately NO spoken acknowledgment here, even for a "no": the very
        # next tick may owe this person a new tailored check-in, and any
        # interjected pleasantry would steal its turn (the adaptive-question
        # contract pins each cue's FIRST spoken line to its question).

    def _apply_workflow_answer(self, workflow, answered_topic: str,
                               response: str) -> None:
        """Hand one assessment answer to the workflow engine (single path)."""
        action = self.workflows.answer(response, workflow.subject_id)
        if response in ("affirmed", "denied") and answered_topic:
            label = "confirmed" if response == "affirmed" else "denied"
            self._conversation_results.append(Result(
                "conversation", f"{answered_topic}_{label}",
                True, 1.0, Severity.INFO,
                f"User {label} {answered_topic.replace('_', ' ')}",
                ttl=120, subject_id=workflow.subject_id,
                source="user_answer", correlation_id=workflow.correlation_id,
                persistence=PersistencePolicy.EVENT))
        if action == "suppressed":
            self.policy.attention.deny(workflow.current_topic or "assessment")

    def _cancel_pending_classification(self, *, superseded: bool = False) -> None:
        """Drop the pending slot, reaping the provider future rather than leaking it."""
        pending, self._pending_classification = self._pending_classification, None
        if pending is None:
            return
        poll = getattr(self.moondream, "poll_classification", None)
        if poll is not None:
            poll(pending.request_id)
        if superseded:
            self._classification_superseded += 1

    def _submit_classification(self, lane: str, question_id: str, target: str,
                               question: str, text: str, now: float,
                               heard_at: float) -> bool:
        """Ask the model to refine ONE ambiguous answer on a later tick.

        Returns whether the request was accepted. The caller falls back to the
        deterministic keyword verdict when it was not — closed provider,
        disabled, unavailable, auth-failed, circuit-open, or the shared
        classify lane already busy all read the same way here.

        Only ever called for a keyword "unclear": a confident keyword match is
        acted on immediately. That ordering is what makes denied -> affirmed
        structurally unreachable, so a person's clear "no" can never be
        overturned by a model, and it keeps offline behavior byte-identical.
        """
        submit = getattr(self.moondream, "submit_classification", None)
        if submit is None:
            return False
        if (self._pending_classification is not None
                and self._pending_classification.heard_at == float(heard_at)):
            # An earlier lane already claimed the single slot for THIS
            # utterance. Later lanes take their deterministic keyword verdict
            # rather than cannibalizing it, which would lose both answers.
            return False
        # One slot, one future: reap the outgoing request before overwriting.
        self._cancel_pending_classification(superseded=True)
        request_id = submit(question, text)
        if request_id is None:
            return False
        self._pending_classification = PendingClassification(
            request_id=request_id, lane=lane, question_id=question_id,
            target=target, question=question, text=text, submitted_at=now,
            deadline=now + CLASSIFICATION_DEADLINE, heard_at=float(heard_at))
        return True

    def _current_question_id(self, pending: PendingClassification) -> str | None:
        """Recompute the lane's question id from LIVE state, or None if gone.

        The primary staleness guard: one comparison covers every way the
        question can have moved on — topic re-asked, answered by keyword in the
        meantime, cooled down, proposal expired or replaced, workflow advanced.
        """
        if pending.lane == "action":
            proposal = self._pending_action
            return None if proposal is None else self._action_question_id(proposal)
        if pending.lane == "corroboration":
            state = self.corroboration.topics.get(pending.target)
            if state is None or state.status != "asked":
                return None
            return f"ask:{pending.target}:{state.asks}"
        if pending.lane == "workflow":
            workflow = self.workflows.active("primary")
            if (workflow is None or workflow.stage != WorkflowStage.QUESTIONS
                    or not workflow.current_topic):
                return None
            return (f"workflow-question:{workflow.correlation_id}:"
                    f"{workflow.current_topic}")
        return None

    def _poll_classification(self, now: float) -> None:
        """Apply or drop one refined answer verdict; never blocks."""
        pending = self._pending_classification
        if pending is None:
            return
        poll = getattr(self.moondream, "poll_classification", None)
        if poll is None:
            self._pending_classification = None
            return
        done, verdict = poll(pending.request_id)
        if not done:
            if now > pending.deadline:
                # Give up on the slot; the provider prunes its own finished
                # future on the next submit, so nothing accumulates.
                self._pending_classification = None
                self._classification_dropped_deadline += 1
            return
        self._pending_classification = None
        if verdict not in _CLASSIFY_VERDICTS:                       # (1)
            return
        if now > pending.deadline:                                  # (2)
            self._classification_dropped_deadline += 1
            return
        if self._last_heard_at > pending.heard_at:                  # (3)
            self._classification_superseded += 1
            return
        if self._current_question_id(pending) != pending.question_id:   # (4)
            self._classification_dropped_stale += 1
            return
        if pending.lane == "action" and (self._pending_action is None
                                         or now > self._pending_action.expires_at):
            self._classification_dropped_stale += 1                 # (5)
            return
        self._apply_classification(pending, verdict, now)

    def _apply_classification(self, pending: PendingClassification,
                              verdict: str, now: float) -> None:
        """Route a surviving verdict through the very path a keyword hit takes."""
        if pending.lane == "corroboration":
            answered = self.corroboration.apply_verdict(pending.target, verdict, now)
            if answered is not None and answered[1] in ("confirmed", "denied"):
                self._record_corroboration_answer(*answered)
            return
        answer = "affirmed" if verdict == "confirmed" else verdict
        if pending.lane == "action":
            if answer == "affirmed":
                self._confirm_pending_action(now)
            elif answer == "denied":
                self._pending_action = None
                self._pending_action_unclear_rephrased = False
                self._action_ack = ("Okay, I won't start it.", now)
            else:
                self._action_unclear(now)
            return
        if pending.lane == "workflow":
            workflow = self.workflows.active("primary")
            if workflow is not None:
                self._apply_workflow_answer(workflow, pending.target, answer)

    def _poll_topic_selection(self, now: float) -> None:
        """Land an async steer choice in the engine's memo cache; never blocks."""
        if self._pending_topic_selection is None:
            return
        poll = getattr(self.moondream, "poll_topic_selection", None)
        if poll is None:
            self._pending_topic_selection = None
            return
        request_id, signature = self._pending_topic_selection
        done, choice = poll(request_id)
        if not done:
            return
        self._pending_topic_selection = None
        # Keyed on the flagged set it was asked about, so a late answer can
        # only ever be read back for that same set; the engine's membership
        # check remains the authority over whether it is honored at all.
        self.corroboration.cache_selection(signature, choice)

    def pop_safety_results(self) -> list[Result]:
        """Drain deterministic non-medical safety events recognized from speech."""
        out, self._safety_results = self._safety_results, []
        return out

    def pop_conversation_results(self) -> list[Result]:
        """Drain public-safe answer classifications for deterministic fusion."""
        out, self._conversation_results = self._conversation_results, []
        return out

    # -------------------------------------------------------------- intents

    def _extra_intents(self, now: float, heard: list, handled: bool) -> list:
        """Candidates from the corroboration/elicitation layers this tick."""
        extra: list[Intent] = []
        self._actions.clear()
        self._expire_arm_check_session()

        if self._action_ack is not None:
            text, created = self._action_ack
            sig = f"action-ack:{int(created * 10)}"
            extra.append(Intent(
                "reply", sig,
                "Briefly acknowledge the confirmed or cancelled action.",
                "", text, 125, health_prompt=False))
            self._actions[sig] = lambda: setattr(self, "_action_ack", None)

        if self._pending_action is not None:
            if now > self._pending_action.expires_at:
                self._pending_action = None
                self._pending_action_unclear_rephrased = False
            else:
                action = self._pending_action
                target = (action.target or action.action).replace("_", " ")
                sig = self._action_question_id(action)
                if self._pending_action_unclear_rephrased:
                    # The single bounded retry: a distinct signature so the
                    # no-repeat bookkeeping lets it be spoken once, one notch
                    # below the original ask so the two can never compete.
                    extra.append(Intent(
                        "question", f"{sig}:rephrase",
                        "Ask once more, plainly, for a yes or no before starting "
                        "the proposed action.", action.reason,
                        f"Sorry — should I start the {target}? Just yes or no.", 119,
                        health_prompt=False, topic=action.target or action.action))
                else:
                    extra.append(Intent(
                        "question", sig,
                        "Ask for explicit confirmation before starting the proposed action.",
                        action.reason,
                        f"Would you like me to start the {target} now?", 120,
                        health_prompt=False, topic=action.target or action.action))

        workflow = self.workflows.active("primary")
        if workflow is not None and workflow.stage == WorkflowStage.INSTRUCTION:
            protocol = PROTOCOLS[workflow.protocol]
            sig = f"workflow:{workflow.correlation_id}:instruction"
            extra.append(Intent("instruction", sig,
                "Briefly explain this is a non-diagnostic measurement, then give the instruction.",
                f"Protocol: {workflow.protocol}. Instruction: {protocol.instruction}",
                protocol.instruction, 110, health_prompt=False, topic=workflow.protocol))
            self._actions[sig] = lambda p=protocol: self.workflows.transition(
                WorkflowStage.POSITIONING, message=p.instruction)
        if workflow is not None and workflow.message and workflow.stage.value in ("positioning", "sampling"):
            sig = f"workflow:{workflow.correlation_id}:{workflow.stage.value}"
            extra.append(Intent("instruction", sig,
                "Explain briefly why this non-diagnostic assessment needs this position, then give the instruction.",
                f"Protocol: {workflow.protocol}. Stage: {workflow.stage.value}. Instruction: {workflow.message}",
                workflow.message, 92, health_prompt=False, topic=workflow.protocol))
        if workflow is not None and workflow.stage == WorkflowStage.QUESTIONS:
            if self.listener is None:
                sig = f"workflow-conclusion:{workflow.correlation_id}"
                extra.append(Intent("conclusion", sig,
                    "State the neutral assessment summary and explain that no diagnosis was made.",
                    workflow.score_summary, workflow.score_summary, 94,
                    health_prompt=False, topic=workflow.protocol))
                self._actions[sig] = lambda: self.workflows.conclude("primary")
            elif workflow.current_topic and workflow.unclear_rephrased:
                topic = workflow.current_topic
                question = _ASSESSMENT_QUESTIONS.get(topic, "Could you tell me a little more about how that felt?")
                sig = f"workflow-rephrase:{workflow.correlation_id}:{topic}"
                extra.append(Intent("question", sig,
                    "Rephrase this once in plain neutral language: " + question, "", question, 95,
                    health_prompt=False, topic=topic))
            elif workflow.current_topic is None:
                topic = self.workflows.peek_question("primary")
                if topic is not None:
                    question = _ASSESSMENT_QUESTIONS.get(topic, "How did that task feel for you?")
                    sig = f"workflow-question:{workflow.correlation_id}:{topic}"
                    extra.append(Intent("question", sig,
                        "Ask this approved neutral follow-up exactly in meaning: " + question,
                        "", question, 95, health_prompt=False, topic=topic))
                    self._actions[sig] = lambda: self.workflows.next_question("primary")
                else:
                    sig = f"workflow-conclusion:{workflow.correlation_id}"
                    extra.append(Intent("conclusion", sig,
                        "State the neutral assessment summary and explain that no diagnosis was made.",
                        workflow.score_summary, workflow.score_summary, 94,
                        health_prompt=False, topic=workflow.protocol))
                    self._actions[sig] = lambda: self.workflows.conclude("primary")

        # 1) Scripted tremor test: speak the instruction, then open the window.
        if self._test_requested and not self.elicitation.active(now=now):
            sig = f"elicit:{int(now)}"
            extra.append(Intent(
                "elicit_test", sig,
                "Ask them, warmly, to hold one hand out flat toward you and "
                "keep it as still as they can for about eight seconds.", "",
                "Could you hold a hand out flat toward me and keep it as "
                "still as you can for about eight seconds?", 90))
            self._actions[sig] = self._begin_hold_still

        # 1b) Scripted arm check: speak the instruction, then open the window.
        if self._arm_check_requested and not self.elicitation.active(now=now):
            sig = f"elicit-arm:{self._arm_check_session_id or 'pending'}:0"
            extra.append(Intent(
                "elicit_test", sig,
                "Ask them, warmly, to show either a bare forearm or a phone "
                "displaying a close-up skin photo, centered toward the camera "
                "and steady for about ten seconds.", "",
                "Could you center either your bare forearm or a phone showing "
                "the skin photo close to the camera and hold it steady for "
                "about ten seconds?",
                90))
            self._actions[sig] = self._begin_arm_check

        # 2) Report a fresh scripted-test result.
        res = self.memory.get("tremor", "tremor_test")
        if res is not None:
            first = self.memory.first_seen.get(
                ("tremor", "tremor_test", str(res.value)))
            if first is not None and now - first <= 12.0:
                sig = f"tremor_result:{int(first)}"
                extra.append(Intent(
                    "conclusion", sig,
                    "Tell them the result of the little hold-still exercise "
                    "in one kind sentence; do not diagnose.",
                    str(res.message), str(res.message), 85))

        # 2b) Report a fresh arm-check result. When both the on-device screening
        # and the cloud photo check have concluded for this session, say what
        # EACH one thought in a single attributed statement; otherwise report
        # whichever one is available.
        local_fresh = None
        res = self.memory.get("arm_skin", "arm_check")
        if res is not None:
            first = self.memory.first_seen.get(
                ("arm_skin", "arm_check", str(res.value)))
            value = res.value if isinstance(res.value, dict) else {}
            status = str(value.get("status") or "succeeded")
            if first is not None and now - first <= 12.0 and status == "succeeded":
                local_fresh = (first, res)

        vlm_fresh = None
        vlm_reposition = False
        vlm_arm = self.memory.get("skin_vision", "arm_check")
        if vlm_arm is not None:
            first = self.memory.first_seen.get(
                ("skin_vision", "arm_check", str(vlm_arm.value)))
            if first is not None and now - first <= 12.0:
                value = vlm_arm.value if isinstance(vlm_arm.value, dict) else {}
                status = str(value.get("status") or "succeeded")
                same_session = (not self._arm_check_session_id
                                or vlm_arm.correlation_id == self._arm_check_session_id)
                if (same_session and status != "reposition_required"
                        and self._arm_check_in_flight):
                    self._finish_arm_check_session()
                if status == "reposition_required" and same_session:
                    vlm_reposition = True
                    phone = (str(value.get("visual_source")) == "displayed_photo"
                             or str(value.get("reason", "")).startswith(
                                 "displayed_photo"))
                    sig = f"elicit-arm:{self._arm_check_session_id}:retry"
                    extra.append(Intent(
                        "elicit_test", sig,
                        ("Explain that the phone photo was not clear enough, then ask "
                         "them once to bring it closer, reduce glare, and hold still."
                         if phone else
                         "Explain that the view was not clear enough, then ask them once "
                         "to center a bare forearm or phone photo and hold still."),
                        "",
                        ("I couldn't get a clear view of the photo. Please bring the "
                         "phone closer, reduce glare, and hold it steady once more."
                         if phone else
                         "I couldn't get a clear skin view. Please center a bare "
                         "forearm or phone photo close to the camera and hold it "
                         "steady once more."), 91))
                    self._actions[sig] = self._retry_arm_check
                else:
                    vlm_fresh = (first, vlm_arm)

        if not vlm_reposition and (local_fresh or vlm_fresh):
            if local_fresh and vlm_fresh:
                lfirst, lres = local_fresh
                vfirst, vres = vlm_fresh
                local_line = str(lres.message)
                for pre in ("Local camera arm check: ", "Local camera arm check "):
                    if local_line.startswith(pre):
                        local_line = local_line[len(pre):]
                        break
                combined = ("Two quick skin checks just finished. "
                            f"The on-device camera screening thought: {local_line}. "
                            f"The cloud photo check thought: {vres.message}")
                sig = f"arm_check_both:{int(lfirst)}:{int(vfirst)}"
                extra.append(Intent(
                    "conclusion", sig,
                    "Tell them what BOTH skin checks found in one or two kind, "
                    "non-diagnostic sentences, clearly attributing the on-device "
                    "camera screening and the cloud photo check separately.",
                    combined, combined, 86))
            elif local_fresh:
                lfirst, lres = local_fresh
                sig = f"arm_check_result:{int(lfirst)}"
                extra.append(Intent(
                    "conclusion", sig,
                    "Tell them what the quick arm skin check showed in one "
                    "kind, non-diagnostic sentence.",
                    str(lres.message), str(lres.message), 85))
            else:
                vfirst, vres = vlm_fresh
                sig = f"vlm_arm_check_result:{int(vfirst)}"
                extra.append(Intent(
                    "conclusion", sig,
                    "Report the arm-check outcome in one cautious, "
                    "non-diagnostic sentence.",
                    str(vres.message), str(vres.message), 86))

        # 3) Guided skin close-up, questions, or safe conclusion.
        skin_prompt = self.skin_dialogue.next_prompt(now)
        if skin_prompt is not None:
            extra.append(Intent(
                skin_prompt.kind, skin_prompt.signature,
                skin_prompt.instruction, skin_prompt.private_detail,
                skin_prompt.fallback, skin_prompt.priority))
            self._actions[skin_prompt.signature] = (
                lambda p=skin_prompt: self.skin_dialogue.mark_spoken(p, now))

        # 4) Corroboration follow-up question for a flagged low-confidence cue.
        # The LLM may steer *which* already-flagged topic to raise (offered only
        # the vetted questions, choice membership-checked in the engine); it
        # falls back to deterministic oldest-flagged when offline or unsure.
        # The engine only calls the selector on a memo MISS, so reaching here
        # means no choice is resolved for this flagged set yet. Returning None
        # takes the deterministic oldest-flagged topic THIS tick — the ask is
        # never delayed — while the request is submitted for a later tick.
        # tick() must never block on the network: the synchronous select_topic
        # reaches urlopen(timeout=8.0) and would stall the frame loop for every
        # new flagged set.
        def _steer(cands):
            items = [(topic, rule.question) for topic, rule in cands]
            signature = tuple(topic for topic, _rule in cands)
            submit = getattr(self.moondream, "submit_topic_selection", None)
            if submit is not None:
                if self._pending_topic_selection is None:
                    request_id = submit(items, self.memory.context_text())
                    if request_id is not None:
                        self._pending_topic_selection = (request_id, signature)
                return None
            select = getattr(self.moondream, "select_topic", None)
            # No async lane on this provider: only a provider that offers the
            # synchronous call at all is asked, and never one that offers both.
            return None if select is None else select(items,
                                                      self.memory.context_text())
        nq = (None if self.skin_dialogue.suppresses_local_skin(now)
              else self.corroboration.next_question_steered(now, _steer))
        if nq is not None:
            topic, rule = nq
            asks = self.corroboration.topics[topic].asks
            sig = f"ask:{topic}:{asks}"
            support = support_clause(self.memory, _CHECK_IN_CUES.get(topic, ()),
                                     SUPPORT_TRAILING_CHECK_IN)
            extra.append(Intent(
                "follow_up", sig,
                "Ask this gentle check-in question, naturally and without "
                "alarm: " + rule.question + support,
                support.strip(), rule.question, 45))
            self._actions[sig] = (lambda t=topic:
                                  self.corroboration.mark_asked(t, now))

        # 5) Conclusions for topics the person confirmed.
        for topic, rule in self.corroboration.pending_conclusions():
            sig = f"conclude:{topic}"
            extra.append(Intent(
                "conclusion", sig,
                "They confirmed your gentle check-in. Say this supportive "
                "suggestion in your own words; do not diagnose.",
                rule.conclusion, rule.conclusion, 68))
            self._actions[sig] = (lambda t=topic:
                                  self.corroboration.mark_concluded(t))

        # 6) A warm reply to free speech (unless a health flow handled it).
        if heard and not handled:
            text, ts = heard[-1]
            extra.append(Intent(
                "reply", f"reply:{int(ts * 10)}",
                "Reply briefly and warmly to what the person just said.",
                f'They said: "{text}"', "I'm glad you told me that.", 80))
        elif not heard and not handled and workflow is None:
            topic = self.topic_queue.next(now)
            if topic is not None:
                extra.append(Intent(
                    "observation", f"steer:{topic.id}", topic.prompt,
                    f"Approved context item: {topic.context_id}", topic.fallback, 42,
                    confidence=min(1.0, topic.score / 12.0),
                    health_prompt=False, topic=topic.id))
        return extra

    def _begin_hold_still(self) -> None:
        self.elicitation.begin("hold_still", _HOLD_STILL_SECONDS)
        self._test_requested = False

    def _begin_arm_check(self) -> None:
        self.elicitation.begin(
            "arm_check", _ARM_CHECK_SECONDS,
            correlation_id=self._arm_check_session_id,
            attempt=self._arm_check_attempt)
        self._arm_check_requested = False
        self._arm_check_in_flight = True
        self._arm_check_in_flight_until = (
            time.monotonic() + _ARM_CHECK_SESSION_TIMEOUT)

    def _retry_arm_check(self) -> None:
        """Open the single permitted corrected capture window."""
        self._arm_check_attempt = 1
        self.elicitation.begin(
            "arm_check", _ARM_CHECK_SECONDS,
            correlation_id=self._arm_check_session_id,
            attempt=self._arm_check_attempt)
        self._arm_check_in_flight = True
        self._arm_check_in_flight_until = (
            time.monotonic() + _ARM_CHECK_SESSION_TIMEOUT)

    def reasoning_card(self) -> dict | None:
        """A compact, non-diagnostic explanation for the showcase dashboard."""
        skin = self.skin_dialogue.reasoning_card()
        if skin is not None:
            return skin
        active = [(topic, state) for topic, state in self.corroboration.topics.items()
                  if state.status in ("flagged", "asked", "confirmed", "denied")]
        if not active:
            workflow = self.workflows.active("primary")
            if workflow is None:
                return None
            return {"observed": f"assessment requested: {workflow.protocol}",
                    "question": workflow.message or "Waiting for the next assessment stage",
                    "answer": workflow.stage.value,
                    "suggestion": "Positioning and quality are verified before scoring."}
        topic, state = max(active, key=lambda item: item[1].flagged_at)
        rule = self.corroboration.rules[topic]
        suggestion = rule.conclusion if state.status == "confirmed" else None
        return {"observed": topic.replace("_", " "), "question": rule.question,
                "answer": state.status, "suggestion": suggestion}

    # ----------------------------------------------------------------- tick

    def _poll_periodic_vision(self, now: float) -> None:
        if self._pending_vision is None or not hasattr(self.moondream, "poll_response"):
            return
        done, response = self.moondream.poll_response(self._pending_vision)
        if not done:
            return
        self._pending_vision = None
        self.vision.complete(now)
        if self._capability_source is not None and self.vision_enabled:
            from core.capabilities import CapabilityStatus
            self._capability_source.set(
                "agent_vision", "cloud",
                CapabilityStatus.READY if response is not None else CapabilityStatus.DEGRADED,
                "periodic vision ready" if response is not None
                else "periodic vision request failed; retry is bounded")
        if response is not None and response.text:
            self.context_broker.add_vision_observation(response.text, now)

    def _submit_periodic_vision(self, now: float) -> bool:
        if not self.vision_enabled or not hasattr(self.moondream, "submit_response"):
            return False
        status = self.moondream.status()
        if not status.get("active"):
            return False
        frame = self.vision.take_due()
        if frame is None:
            return False
        context = self._turn_context("What new, neutral, conversationally useful details are visible?")
        messages = [{"role": "user", "content": (
            "Privately summarize only new neutral visible details useful for a future "
            "conversation. Do not diagnose, identify the person, infer sensitive traits, "
            "or address the person directly.")}]
        request_id = self.moondream.submit_response(messages, context.items, image=frame)
        # The provider worker now owns the sole ephemeral copy; this scope drops it.
        if request_id is None:
            self.vision.complete(now)
            return False
        self._pending_vision = request_id
        return True

    def tick(self, snapshot, now: float | None = None) -> str | None:
        """Advance one step: hear, update state, and act if warranted."""
        now = time.time() if now is None else now
        self._poll_periodic_vision(now)
        # Before _consume_heard: an arriving verdict must be applied to the
        # state its question was asked against, not to state a new utterance
        # has already mutated.
        self._poll_classification(now)
        self._poll_topic_selection(now)
        heard, handled = self._consume_heard(now)
        self.memory.ingest(snapshot, now)
        self.context_broker.ingest(snapshot, now)
        # Signals the topic table or a corroboration rule already speaks for
        # must not also be promoted as an unauthored "By the way, ..." line.
        self.topic_queue.observe(self.context_broker.items(), now,
                                 excluded=covered_keys())
        grouped: dict[str, list] = {}
        for result in snapshot:
            grouped.setdefault(result.subject_id, []).append(result)
        for subject_id, subject_results in grouped.items():
            if subject_id == "primary":
                continue
            memory = self.memories.setdefault(subject_id, ObservationMemory(name="anonymous visitor"))
            memory.ingest(subject_results, now)
        for result in snapshot:
            if result.module == "assessment_request" and result.key == "protocol" \
                    and str(result.value) in PROTOCOLS:
                self.workflows.start(str(result.value), subject_id=result.subject_id,
                                     correlation_id=result.correlation_id)
        self._advance_demo_circuit()
        self.skin_dialogue.observe(snapshot, now)
        corroboration_snapshot = snapshot
        if self.skin_dialogue.suppresses_local_skin(now):
            self.corroboration.suppress("skin_changes", now)
            corroboration_snapshot = [
                r for r in snapshot
                if not (r.module == "rash" and r.key.startswith("rash"))]
        self.corroboration.observe(corroboration_snapshot, now)
        extra = self._extra_intents(now, heard, handled)
        emergency = self._emergency_intents(snapshot)
        if emergency:
            # Urgent wording is deterministic: it must not wait for a model
            # response while a confirmed emergency is active.
            intent = self.policy.next_intent(
                self.memory, now, extra=emergency, suppress_routine=True,
                awaiting_answer=True)
            if intent is not None:
                if intent.signature.startswith("urgent:explicit_help:"):
                    self._pending_help_alert = False
                    self._active_emergencies.discard("explicit_help")
                return self._speak_intent(intent, intent.fallback, now)
        if self._pending_response is not None:
            (request_id, pending_intent, requested_mono, requested_at,
             pending_action, pending_topic, pending_items) = self._pending_response
            done, response = self.moondream.poll_response(request_id)
            if done or time.monotonic() - requested_mono >= self._response_deadline:
                self._pending_response = None
                self._last_conversation_latency_ms = round(
                    (time.monotonic() - requested_mono) * 1000.0, 1)
                if response is None:
                    self._last_fallback_reason = "provider_timeout_or_unavailable"
                    text = pending_intent.fallback
                else:
                    self._last_fallback_reason = response.fallback_reason
                    text = response.text or pending_intent.fallback
                    text, guard_reason = guard_agent_only_speech(
                        text, pending_items, pending_intent.fallback)
                    if guard_reason is not None:
                        self._last_fallback_reason = guard_reason
                if pending_topic is not None:
                    self.topic_queue.mark_raised(pending_topic, requested_at)
                    self._last_steered_topic = pending_topic
                return self._speak_intent(
                    pending_intent, text, requested_at, action=pending_action)
            return None
        if self._pending_speech is not None:
            (request_id, pending_intent, requested_mono, requested_at,
             pending_action) = self._pending_speech
            done, generated = self.moondream.poll_generation(request_id)
            if done or time.monotonic() - requested_mono >= self._cloud_speech_deadline:
                self._pending_speech = None
                text = generated or pending_intent.fallback
                return self._speak_intent(
                    pending_intent, text, requested_at, action=pending_action)
            return None
        intent = self.policy.next_intent(
            self.memory, now, extra=extra,
            suppress_routine=self.workflows.active("primary") is not None,
            corroboration=self.corroboration,
            awaiting_answer=self._awaiting_answer(now))
        if intent is None:
            if not heard and self._pending_vision is None:
                self._submit_periodic_vision(now)
            return None
        topic = self.topic_queue.next(now)
        if not (intent.kind == "reply" or intent.signature.startswith("steer:")):
            topic = None
        if hasattr(self.moondream, "submit_response"):
            query = (heard[-1][0] if heard and intent.kind == "reply"
                     else intent.detail or intent.llm_intent)
            context = self._turn_context(query)
            request_id = self.moondream.submit_response(
                self._turn_messages(intent, context, topic), context.items)
            if request_id is not None:
                action = self._actions.get(intent.signature)
                self._pending_response = (
                    request_id, intent, time.monotonic(), now, action, topic,
                    list(context.items))
                return None
        request_id = self.moondream.submit_generation(
            intent.llm_intent, self.memory.context_text(), intent.detail)
        if request_id is not None:
            action = self._actions.get(intent.signature)
            self._pending_speech = (
                request_id, intent, time.monotonic(), now, action)
            return None
        if topic is not None and intent.signature.startswith("steer:"):
            self.topic_queue.mark_raised(topic, now)
            self._last_steered_topic = topic
        self._last_fallback_reason = "provider_unavailable"
        return self._speak_intent(intent, intent.fallback, now)

    def _speak_intent(self, intent, candidate: str, now: float,
                      action=_MISSING_ACTION) -> str:
        """Apply safety wording and commit one selected intent after generation."""
        generated = candidate
        if intent.signature.startswith("skin:"):
            text = self.skin_dialogue.safe_speech(generated, intent.fallback)
        elif intent.signature.startswith(("ask:", "conclude:")):
            # Health check-in lines are LLM-phrased from low-confidence visual
            # priors; the airlock discards any generation that asserts a finding
            # or accuses, falling back to the hand-authored rule text. A second
            # deterministic guard keeps the line addressed to the person ('you')
            # rather than a third-person report about them.
            text = safe_check_in(generated, intent.fallback)
            text = enforce_second_person(text, intent.fallback)
            text = strip_prior_disclosure(text, intent.fallback)
        else:
            # LLM-proposes / deterministic-disposes: only speak the generation
            # when it reads like a spoken line, else the hand-authored fallback.
            text = self._clean_spoken_line(generated) or intent.fallback
        # Small talk and most topic fallbacks end in a question but belong to no
        # lane that tracks an answer. Judge the FINAL text (after the airlocks),
        # since a fallback substitution can turn a question into a statement.
        if (self._can_hear() and text.rstrip().endswith("?")
                and intent.kind not in _PHYSICAL_PROMPT_KINDS):
            self._awaiting_reply_until = now + self.reply_window
        self.policy.mark_spoken(intent, now)
        active_workflow = self.workflows.active("primary")
        if action is _MISSING_ACTION:
            action = self._actions.get(intent.signature)
        if action is not None:
            action()
        if active_workflow is not None and intent.kind == "conclusion":
            self.events.record_recommendation(
                active_workflow.protocol, text,
                subject_id=active_workflow.subject_id,
                correlation_id=active_workflow.correlation_id)
        self.memory.agent_said(text, now)
        self.last_utterance = text
        self.speaker.say(text)
        mark_spoke = getattr(self.listener, "mark_agent_spoke", None)
        if mark_spoke is not None:
            mark_spoke(now)
        return text

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self.moondream.close()
        self.speaker.close()
        if self.listener is not None:
            self.listener.close()
