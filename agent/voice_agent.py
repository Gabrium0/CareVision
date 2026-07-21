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

from agent.state import ObservationMemory
from agent.policy import Intent, Policy
from agent.corroboration import CorroborationEngine
from agent.moondream_client import MoondreamClient
from agent.skin_dialogue import SkinDialogue
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
_ASSESSMENT_QUESTIONS = {
    "symptoms": "Did you notice any discomfort, weakness, dizziness, or other symptoms during that?",
    "progression": "Has this movement or task changed recently compared with what is normal for you?",
    "warning_signs": "Are you feeling suddenly unwell, faint, confused, or having trouble speaking right now?",
}


class VoiceAgent:
    """Orchestrates memory -> policy -> Moondream -> speech (and listening)."""
    def __init__(self, name: str = "there", speak: bool = True,
                 model: str = "moondream3.1-9B-A2B", listener=None,
                 moondream_enabled: bool = True,
                 **policy_kwargs):
        self.memory = ObservationMemory(name=name)
        self.memories: dict[str, ObservationMemory] = {"primary": self.memory}
        self.policy = Policy(**policy_kwargs)
        self.moondream = MoondreamClient(model=model, enabled=moondream_enabled)
        self.speaker = Speaker(enabled=speak)
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
        self._actions: dict = {}       # intent signature -> post-speech callback
        self._safety_results: list[Result] = []
        self._conversation_results: list[Result] = []
        self._pending_speech: tuple[str, object, float, float] | None = None
        self._cloud_speech_deadline = 0.75
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

    def request_test(self, test: str = "hold_still") -> None:
        """Queue a scripted test (the 't'/'a' hotkey path)."""
        if test in PROTOCOLS:
            self.workflows.start(test)
        elif test == "arm_check":
            now = time.time()
            if (self._arm_check_requested or
                    (self.elicitation.test == "arm_check"
                     and now < self.elicitation.until + 50.0)):
                return
            self._arm_check_requested = True
        else:
            self._test_requested = True

    # ---------------------------------------------------------------- ears

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
            active_workflow = self.workflows.active("primary")
            if active_workflow is not None:
                low_answer = text.lower().strip()
                response = ("denied" if any(word in low_answer.split() for word in ("no", "nope", "none"))
                            else "affirmed" if any(word in low_answer.split() for word in ("yes", "yeah", "yep"))
                            else "unclear")
                if (active_workflow.stage == WorkflowStage.QUESTIONS
                        and active_workflow.current_topic is not None):
                    answered_topic = active_workflow.current_topic
                    action = self.workflows.answer(response, active_workflow.subject_id)
                    handled = True
                    if response in ("affirmed", "denied") and answered_topic:
                        self._conversation_results.append(Result(
                            "conversation", f"{answered_topic}_{'confirmed' if response == 'affirmed' else 'denied'}",
                            True, 1.0, Severity.INFO,
                            f"User {'confirmed' if response == 'affirmed' else 'denied'} {answered_topic.replace('_', ' ')}",
                            ttl=120, subject_id=active_workflow.subject_id,
                            source="user_answer", correlation_id=active_workflow.correlation_id,
                            persistence=PersistencePolicy.EVENT))
                    if action == "suppressed":
                        self.policy.attention.deny(active_workflow.current_topic or "assessment")
            skin_answered = self.skin_dialogue.hear(text, now)
            answered = None if skin_answered else self.corroboration.hear(text, now)
            if answered is not None and answered[1] in ("confirmed", "denied"):
                topic, verdict = answered
                self._conversation_results.append(Result(
                    "conversation", f"{topic}_{verdict}", True, 1.0, Severity.INFO,
                    f"User {verdict} {topic.replace('_', ' ')}", ttl=120,
                    source="user_answer", persistence=PersistencePolicy.EVENT))
            if skin_answered or answered is not None:
                handled = True
            low = text.lower()
            if any(phrase in low for phrase in ("help me", "call for help", "call emergency")):
                self._safety_results.append(Result(
                    "explicit_help", "call_for_help", True, .95, Severity.ALERT,
                    "The person explicitly called for help", ttl=20,
                    source="speech_recognition", quality=.95,
                    persistence=PersistencePolicy.EVENT))
            if any(t in low for t in _TEST_TRIGGERS):
                self._test_requested = True
            if any(t in low for t in _ARM_TRIGGERS):
                self._arm_check_requested = True
        return heard, handled

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
            sig = f"elicit-arm:{int(now)}"
            extra.append(Intent(
                "elicit_test", sig,
                "Ask them, warmly, to hold a forearm up toward the camera "
                "with the skin facing the lens and keep it steady for about "
                "ten seconds.", "",
                "Could you hold your forearm up toward the camera, skin "
                "facing the lens, and keep it steady for about ten seconds?",
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

        # 2b) Report a fresh arm-check result.
        res = self.memory.get("arm_skin", "arm_check")
        if res is not None:
            first = self.memory.first_seen.get(
                ("arm_skin", "arm_check", str(res.value)))
            if first is not None and now - first <= 12.0:
                sig = f"arm_check_result:{int(first)}"
                extra.append(Intent(
                    "conclusion", sig,
                    "Tell them what the quick arm skin check showed in one "
                    "kind, non-diagnostic sentence.",
                    str(res.message), str(res.message), 85))

        vlm_arm = self.memory.get("skin_vision", "arm_check")
        if vlm_arm is not None:
            first = self.memory.first_seen.get(
                ("skin_vision", "arm_check", str(vlm_arm.value)))
            if first is not None and now - first <= 12.0:
                sig = f"vlm_arm_check_result:{int(first)}"
                extra.append(Intent(
                    "conclusion", sig,
                    "Report the NVIDIA arm VLM result in one cautious sentence; "
                    "keep it separate from the local camera screening and do not diagnose.",
                    str(vlm_arm.message), str(vlm_arm.message), 86))

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
        nq = (None if self.skin_dialogue.suppresses_local_skin(now)
              else self.corroboration.next_question(now))
        if nq is not None:
            topic, rule = nq
            asks = self.corroboration.topics[topic].asks
            sig = f"ask:{topic}:{asks}"
            cues = self.memory.facial_cues()
            relevant = []
            if topic == "cold_symptoms":
                relevant = [key for key in ("nose_redness", "cheek_redness",
                                             "nasal_discharge_visible") if key in cues]
            elif topic == "hydration" and "lip_dryness" in cues:
                relevant = ["lip_dryness"]
            elif topic == "tiredness_pallor":
                relevant = [key for key in ("under_eye_darkness", "under_eye_puffiness")
                            if key in cues]
            support = (" Supporting visible appearance cues: " +
                       ", ".join(key.replace("_", " ") for key in relevant) +
                       ". Use them only to phrase the check-in; do not state a cause or diagnosis."
                       if relevant else "")
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
        return extra

    def _begin_hold_still(self) -> None:
        self.elicitation.begin("hold_still", _HOLD_STILL_SECONDS)
        self._test_requested = False

    def _begin_arm_check(self) -> None:
        self.elicitation.begin("arm_check", _ARM_CHECK_SECONDS)
        self._arm_check_requested = False

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

    def tick(self, snapshot, now: float | None = None) -> str | None:
        """Advance one step: hear, update state, and act if warranted."""
        now = time.time() if now is None else now
        heard, handled = self._consume_heard(now)
        self.memory.ingest(snapshot, now)
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
        self.skin_dialogue.observe(snapshot, now)
        corroboration_snapshot = snapshot
        if self.skin_dialogue.suppresses_local_skin(now):
            self.corroboration.suppress("skin_changes", now)
            corroboration_snapshot = [
                r for r in snapshot
                if not (r.module == "rash" and r.key.startswith("rash"))]
        self.corroboration.observe(corroboration_snapshot, now)
        extra = self._extra_intents(now, heard, handled)
        if self._pending_speech is not None:
            request_id, pending_intent, requested_mono, requested_at = self._pending_speech
            done, generated = self.moondream.poll_generation(request_id)
            if done or time.monotonic() - requested_mono >= self._cloud_speech_deadline:
                self._pending_speech = None
                text = generated or pending_intent.fallback
                return self._speak_intent(pending_intent, text, requested_at)
            return None
        intent = self.policy.next_intent(
            self.memory, now, extra=extra,
            suppress_routine=self.workflows.active("primary") is not None)
        if intent is None:
            return None
        request_id = self.moondream.submit_generation(
            intent.llm_intent, self.memory.context_text(), intent.detail)
        if request_id is not None:
            self._pending_speech = (request_id, intent, time.monotonic(), now)
            return None
        return self._speak_intent(intent, intent.fallback, now)

    def _speak_intent(self, intent, candidate: str, now: float) -> str:
        """Apply safety wording and commit one selected intent after generation."""
        generated = candidate
        if intent.signature.startswith("skin:"):
            text = self.skin_dialogue.safe_speech(generated, intent.fallback)
        else:
            text = generated or intent.fallback
        self.policy.mark_spoken(intent, now)
        active_workflow = self.workflows.active("primary")
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
