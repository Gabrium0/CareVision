"""VoiceAgent: orchestrates memory -> policy -> Gemini -> speech, plus ears.

Call `tick(snapshot)` once per frame (cheap; rate-limited internally). The agent
updates its memory, lets the policy pick at most one thing to say, phrases it
with Gemini (falling back to a templated line offline), speaks it via TTS, and
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
from agent.gemini_client import GeminiClient
from audio.tts import Speaker
from core.elicitation import ElicitationState

_TEST_TRIGGERS = ("check my hand", "check my tremor", "test my hand",
                  "test my tremor", "am i shaking", "tremor test",
                  "check my hands", "test my hands")
_HOLD_STILL_SECONDS = 10.0     # ~2 s to comply + 8 s of sampling


class VoiceAgent:
    """Orchestrates memory -> policy -> Gemini -> speech (and listening)."""
    def __init__(self, name: str = "there", speak: bool = True,
                 model: str = "gemini-2.5-flash", listener=None,
                 **policy_kwargs):
        self.memory = ObservationMemory(name=name)
        self.policy = Policy(**policy_kwargs)
        self.gemini = GeminiClient(model=model)
        self.speaker = Speaker(enabled=speak)
        self.listener = listener
        self.corroboration = CorroborationEngine(gemini=self.gemini)
        self.elicitation = ElicitationState.instance()
        self.last_utterance = ""
        self._test_requested = False
        self._actions: dict = {}       # intent signature -> post-speech callback
        mode = "Gemini" if self.gemini.available else "templated"
        ears = "listening" if (listener is not None and
                               getattr(listener, "available", False)) else "speak-only"
        print(f"[agent] voice agent ready (name={name}, speech={mode}, {ears})")

    def request_test(self, test: str = "hold_still") -> None:
        """Queue a scripted test (the 't' hotkey path)."""
        self._test_requested = True

    # ---------------------------------------------------------------- ears

    def _consume_heard(self, now: float) -> tuple[list, bool]:
        """Drain the listener into memory/corroboration.

        Returns (heard utterances, whether one confirmed a pending topic —
        in which case the conclusion intent is the reply, not a generic one).
        """
        if self.listener is None:
            return [], False
        confirmed = False
        heard = self.listener.pop_utterances()
        for text, ts in heard:
            self.memory.person_said(text, ts)
            answered = self.corroboration.hear(text, now)
            if answered is not None and answered[1] == "confirmed":
                confirmed = True
            low = text.lower()
            if any(t in low for t in _TEST_TRIGGERS):
                self._test_requested = True
        return heard, confirmed

    # -------------------------------------------------------------- intents

    def _extra_intents(self, now: float, heard: list, confirmed: bool) -> list:
        """Candidates from the corroboration/elicitation layers this tick."""
        extra: list[Intent] = []
        self._actions.clear()

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

        # 3) Corroboration follow-up question for a flagged low-confidence cue.
        nq = self.corroboration.next_question(now)
        if nq is not None:
            topic, rule = nq
            asks = self.corroboration.topics[topic].asks
            sig = f"ask:{topic}:{asks}"
            extra.append(Intent(
                "follow_up", sig,
                "Ask this gentle check-in question, naturally and without "
                "alarm: " + rule.question, "", rule.question, 45))
            self._actions[sig] = (lambda t=topic:
                                  self.corroboration.mark_asked(t, now))

        # 4) Conclusions for topics the person confirmed.
        for topic, rule in self.corroboration.pending_conclusions():
            sig = f"conclude:{topic}"
            extra.append(Intent(
                "conclusion", sig,
                "They confirmed your gentle check-in. Say this supportive "
                "suggestion in your own words; do not diagnose.",
                rule.conclusion, rule.conclusion, 68))
            self._actions[sig] = (lambda t=topic:
                                  self.corroboration.mark_concluded(t))

        # 5) A warm reply to free speech (unless a conclusion IS the reply).
        if heard and not confirmed:
            text, ts = heard[-1]
            extra.append(Intent(
                "reply", f"reply:{int(ts * 10)}",
                "Reply briefly and warmly to what the person just said.",
                f'They said: "{text}"', "I'm glad you told me that.", 80))
        return extra

    def _begin_hold_still(self) -> None:
        self.elicitation.begin("hold_still", _HOLD_STILL_SECONDS)
        self._test_requested = False

    def reasoning_card(self) -> dict | None:
        """A compact, non-diagnostic explanation for the showcase dashboard."""
        active = [(topic, state) for topic, state in self.corroboration.topics.items()
                  if state.status in ("flagged", "asked", "confirmed", "denied")]
        if not active:
            return None
        topic, state = max(active, key=lambda item: item[1].flagged_at)
        rule = self.corroboration.rules[topic]
        suggestion = rule.conclusion if state.status == "confirmed" else None
        return {"observed": topic.replace("_", " "), "question": rule.question,
                "answer": state.status, "suggestion": suggestion}

    # ----------------------------------------------------------------- tick

    def tick(self, snapshot, now: float | None = None) -> str | None:
        """Advance one step: hear, update state, and act if warranted."""
        now = time.time() if now is None else now
        heard, confirmed = self._consume_heard(now)
        self.memory.ingest(snapshot, now)
        self.corroboration.observe(snapshot, now)
        extra = self._extra_intents(now, heard, confirmed)
        intent = self.policy.next_intent(self.memory, now, extra=extra)
        if intent is None:
            return None
        text = self.gemini.generate(intent.llm_intent, self.memory.context_text(),
                                    intent.detail) or intent.fallback
        self.policy.mark_spoken(intent, now)
        action = self._actions.get(intent.signature)
        if action is not None:
            action()
        self.memory.agent_said(text, now)
        self.last_utterance = text
        self.speaker.say(text)
        return text

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        self.speaker.close()
        if self.listener is not None:
            self.listener.close()
