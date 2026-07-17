"""Private skin-screening dialogue and speech-disclosure guard."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.corroboration import interpret_answer_keywords
from core.elicitation import ElicitationState
from core.events import Result, Visibility


_DIAGNOSTIC_LANGUAGE = re.compile(
    r"\b(you have|you've got|this is|that is|looks like|appears to be|"
    r"might be|could be|possibly|diagnos(?:e|ed|is|tic)|suffering from)\b", re.I)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def _hypothesis_terms(hypotheses: list[str] | tuple[str, ...]) -> set[str]:
    terms = set()
    for hypothesis in hypotheses:
        whole = _normalize(hypothesis)
        if len(whole) >= 4:
            terms.add(whole)
        for part in re.split(r"[/,(]", hypothesis):
            part = _normalize(part)
            if len(part) >= 4:
                terms.add(part)
        for token in whole.split():
            if len(token) >= 6 and token.endswith(
                    ("itis", "osis", "oma", "pox", "zoster", "eczema")):
                terms.add(token)
            if token in {"rash", "eczema", "psoriasis", "melanoma", "shingles",
                         "hives", "ringworm", "cancer"}:
                terms.add(token)
    return terms


def speech_mentions_hypothesis(text: str,
                               hypotheses: list[str] | tuple[str, ...]) -> bool:
    """Return whether speech discloses a private name or diagnostic claim."""
    normalized = _normalize(text)
    return bool(_DIAGNOSTIC_LANGUAGE.search(text) or any(
        term in normalized for term in _hypothesis_terms(hypotheses)))


@dataclass
class SkinPrompt:
    """A safe agent intent plus its private prompt-only context."""

    kind: str
    signature: str
    instruction: str
    private_detail: str
    fallback: str
    priority: int


@dataclass
class SkinDialogue:
    """Coordinates close-up capture and at most three symptom questions."""

    language_model: Any = None
    answer_window: float = 45.0
    cooldown_seconds: float = 4 * 3600.0
    closeup_wait: float = 45.0
    status: str = "idle"
    region: str = "the area"
    features: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    questions: list[tuple[str, str]] = field(default_factory=list)
    question_index: int = 0
    answers: list[tuple[str, str]] = field(default_factory=list)
    started_at: float = 0.0
    asked_at: float = 0.0
    cooldown_until: float = 0.0
    last_result_timestamp: float = 0.0
    conclusion: str | None = None
    closeup_seconds: float = 10.0
    _elicitation: ElicitationState = field(default_factory=ElicitationState.instance)

    def _private_result(self, snapshot: list[Result], key: str) -> Result | None:
        matches = [r for r in snapshot if r.module == "skin_vision" and r.key == key
                   and r.visibility == Visibility.AGENT_ONLY]
        return max(matches, key=lambda r: r.timestamp, default=None)

    @staticmethod
    def _value(result: Result) -> dict:
        return result.value if isinstance(result.value, dict) else {}

    def _load_context(self, value: dict) -> None:
        self.region = str(value.get("body_region") or "the visible area")[:60]
        self.features = [str(v) for v in value.get("visible_features", [])[:5]]
        self.hypotheses = [str(v) for v in value.get("possible_conditions", [])[:3]]
        self.topics = [str(v) for v in value.get("follow_up_topics", [])[:5]]
        try:
            self.closeup_seconds = max(3.0, min(20.0,
                float(value.get("closeup_seconds", self.closeup_seconds))))
        except (TypeError, ValueError):
            self.closeup_seconds = 10.0

    def _build_questions(self) -> list[tuple[str, str]]:
        questions: list[tuple[str, str]] = []
        symptom_bits = []
        if "itching" in self.topics:
            symptom_bits.append("itchy")
        if "pain" in self.topics:
            symptom_bits.append("painful")
        if symptom_bits:
            questions.append((
                f"Does the skin on {self.region} feel {' or '.join(symptom_bits)}?",
                "symptoms"))
        else:
            questions.append((
                f"Have you noticed any discomfort or irritation on {self.region}?",
                "symptoms"))
        questions.append((
            "Is this new, or has it been spreading or changing quickly?",
            "progression"))
        if "fever_unwell" in self.topics or "blisters" in self.topics:
            questions.append((
                "Are you feeling feverish or unwell, or is the area severely blistered?",
                "red_flag"))
        elif "new_medication" in self.topics or "new_product_exposure" in self.topics:
            questions.append((
                "Did it start after a new medicine, cream, soap, or other product?",
                "exposure"))
        return questions[:3]

    def observe(self, snapshot: list[Result], now: float) -> None:
        """Consume fresh agent-only results and expire stalled workflows."""
        if self.status == "cooldown" and now >= self.cooldown_until:
            self.status = "idle"
        if self.status == "waiting_closeup" and now - self.started_at > self.closeup_wait:
            self._finish(now, short=True)
        if self.status == "awaiting_answer" and now - self.asked_at > self.answer_window:
            self._finish(now, short=True)

        analysis = self._private_result(snapshot, "analysis")
        request = self._private_result(snapshot, "closeup_request")
        fresh = analysis if (analysis and analysis.timestamp > self.last_result_timestamp) else None
        if fresh is not None:
            self.last_result_timestamp = fresh.timestamp
            self._load_context(self._value(fresh))
            self.questions = self._build_questions()
            self.question_index = 0
            self.answers.clear()
            self.conclusion = None
            self.status = "questions"
            self.started_at = now
            return
        if (self.status == "idle" and request is not None
                and request.timestamp > self.last_result_timestamp
                and now >= self.cooldown_until):
            self.last_result_timestamp = request.timestamp
            self._load_context(self._value(request))
            self.status = "request_closeup"
            self.started_at = now

    def suppresses_local_skin(self, now: float) -> bool:
        """Prevent the local rash heuristic from starting a duplicate dialogue."""
        return self.status != "idle" or now < self.cooldown_until

    def _private_detail(self) -> str:
        names = ", ".join(self.hypotheses) if self.hypotheses else "none"
        return ("PRIVATE UNVERIFIED CONDITION HYPOTHESES: " + names + ". "
                "Use them only to understand why the question is relevant. "
                "Never say, paraphrase, or imply any condition name or diagnosis.")

    def next_prompt(self, now: float) -> SkinPrompt | None:
        """Return the next close-up/question/conclusion prompt, if ready."""
        if self.status == "request_closeup":
            if self._elicitation.active(now=now):
                return None
            fallback = (f"I noticed a possible skin change on {self.region}. "
                        "Could you show that area a little closer to the camera?")
            return SkinPrompt(
                "elicit_test", f"skin:closeup:{int(self.last_result_timestamp)}",
                "Ask for a closer camera view of the named area. Mention only a "
                "possible skin change, never a condition or diagnosis.",
                self._private_detail(), fallback, 88)
        if self.status == "questions" and self.question_index < len(self.questions):
            question, _tag = self.questions[self.question_index]
            return SkinPrompt(
                "follow_up", f"skin:question:{int(self.last_result_timestamp)}:"
                f"{self.question_index}",
                "Ask exactly one gentle symptom question. Do not name or imply a "
                "medical condition.", self._private_detail(), question, 72)
        if self.status == "conclusion" and self.conclusion:
            return SkinPrompt(
                "conclusion", f"skin:conclusion:{int(self.last_result_timestamp)}",
                "Give the neutral safety suggestion without naming or implying a "
                "condition.", self._private_detail(), self.conclusion, 76)
        return None

    def mark_spoken(self, prompt: SkinPrompt, now: float) -> None:
        """Advance state only after the selected prompt was actually spoken."""
        if prompt.signature.startswith("skin:closeup:"):
            self._elicitation.begin("skin_closeup", self.closeup_seconds, now=now)
            self.status = "waiting_closeup"
            self.started_at = now
        elif prompt.signature.startswith("skin:question:"):
            self.status = "awaiting_answer"
            self.asked_at = now
        elif prompt.signature.startswith("skin:conclusion:"):
            self._finish(now)

    def hear(self, text: str, now: float) -> bool:
        """Consume an answer to the current skin question; return if handled."""
        if self.status != "awaiting_answer" or now - self.asked_at > self.answer_window:
            return False
        question, tag = self.questions[self.question_index]
        verdict = None
        if (self.language_model is not None
                and getattr(self.language_model, "available", False)):
            verdict = self.language_model.classify_answer(question, text)
        if verdict not in ("confirmed", "denied", "unclear"):
            verdict = interpret_answer_keywords(text)
        self.answers.append((tag, verdict))
        self.question_index += 1
        if verdict == "unclear" or self.question_index >= len(self.questions):
            self._prepare_conclusion()
        else:
            self.status = "questions"
        return True

    def _prepare_conclusion(self) -> None:
        red_flag = any(verdict == "confirmed" and tag in ("red_flag", "progression")
                       for tag, verdict in self.answers)
        symptoms = any(verdict == "confirmed" for _tag, verdict in self.answers)
        if red_flag:
            self.conclusion = ("Because it is changing quickly or you feel unwell, "
                               "please consider contacting a healthcare professional promptly.")
        elif symptoms:
            self.conclusion = ("Thanks for telling me. Please keep an eye on the area, "
                               "and consider asking a healthcare professional if it persists.")
        else:
            self.conclusion = "Thanks. Please keep an eye on the area in case it changes."
        self.status = "conclusion"

    def _finish(self, now: float, short: bool = False) -> None:
        self.status = "cooldown"
        self.cooldown_until = now + (600.0 if short else self.cooldown_seconds)
        self.conclusion = None
        self.hypotheses.clear()
        self.topics.clear()

    def safe_speech(self, generated: str | None, fallback: str) -> str:
        """Replace disclosure-prone generated text with the reviewed fallback."""
        text = (generated or "").strip()
        if not text or speech_mentions_hypothesis(text, self.hypotheses):
            return fallback
        return text

    def reasoning_card(self) -> dict | None:
        """Return public-safe dialogue status without private hypotheses."""
        if self.status in ("idle", "cooldown"):
            return None
        prompt = self.next_prompt(self.started_at)
        return {
            "observed": f"possible skin change on {self.region}",
            "question": prompt.fallback if prompt else "Waiting for a response",
            "answer": self.status,
            "suggestion": self.conclusion,
        }
