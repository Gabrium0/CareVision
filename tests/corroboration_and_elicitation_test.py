"""Tests for the round-2 agent stack: corroboration loop, elicited tremor
test, expressivity screening, and cross-session asymmetry baselines.

Everything runs offline and synthetic: a fake Listener stands in for the
microphone, Moondream is unavailable (keyword interpretation path), and
HistoryStore-backed modules get temp-file stores (never data/history.db),
following tests/grooming_test.py.

Run standalone:  python -m pytest tests/corroboration_and_elicitation_test.py
"""
import sys
import tempfile
import time as _time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FaceData, FrameContext, PoseData
from core.elicitation import ElicitationState
from core.events import Result, Severity
from agent.corroboration import (CorroborationEngine,
                                 interpret_answer_keywords)
from storage.history_store import HistoryStore
from extractors import face_landmarks as FL

FRAME = 200


def _res(module, key, value, conf, severity=Severity.NOTICE, message="x"):
    return Result(module=module, key=key, value=value, confidence=conf,
                  severity=severity, message=message, ttl=10.0)


# ------------------------------------------------------------ corroboration

def test_keyword_interpretation():
    assert interpret_answer_keywords("No, not really.") == "denied"
    assert interpret_answer_keywords("Yes, a little actually") == "confirmed"
    assert interpret_answer_keywords("The weather is nice") == "unclear"
    assert interpret_answer_keywords("I'm fine, thanks") == "denied"


def test_flag_ask_confirm_flow():
    eng = CorroborationEngine()
    snap = [_res("rash", "rash", 0.3, 0.4)]
    eng.observe(snap, now=100.0)
    topic, rule = eng.next_question(100.0)
    assert topic == "skin_changes" and "skin" in rule.question
    eng.mark_asked(topic, 101.0)
    assert eng.status(topic) == "asked"
    assert eng.hear("yes, I noticed a spot lately", 105.0) == (topic, "confirmed")
    assert eng.pending_conclusions()[0][0] == topic
    eng.mark_concluded(topic)
    assert eng.pending_conclusions() == []


def test_nvidia_facial_cues_never_flag_questions_without_local_trigger():
    appearance = _res(
        "skin_vision", "facial_appearance",
        {"cues": {"under_eye_darkness": "mild", "lip_dryness": "marked"}},
        0.8, severity=Severity.INFO)
    eng = CorroborationEngine()
    eng.observe([appearance], now=100.0)
    assert eng.next_question(100.0) is None

    eng.observe([appearance, _res("dry_lips", "lip_dryness", 1.2, 0.5)],
                now=101.0)
    topic, rule = eng.next_question(101.0)
    assert topic == "hydration"
    assert "drink" in rule.question.lower()


def test_tiredness_policy_uses_nvidia_eye_cues_only_after_local_perclos():
    from agent.policy import Policy
    from agent.state import ObservationMemory

    appearance = _res(
        "skin_vision", "facial_appearance",
        {"cues": {"under_eye_darkness": "mild"}},
        0.8, severity=Severity.INFO)
    memory = ObservationMemory()
    memory.ingest([appearance], now=100.0)
    policy = Policy(small_talk_interval=1e9)
    assert all(intent.signature != "tired"
               for intent in policy._candidates(memory, 100.0))

    perclos = _res("drowsiness", "perclos", 0.25, 0.7,
                   severity=Severity.NOTICE,
                   message="Drowsiness: eyes closed 25% of the time")
    memory.ingest([perclos], now=101.0)
    tired = next(intent for intent in policy._candidates(memory, 101.0)
                 if intent.signature == "tired")
    assert "supporting visible appearance cues" in tired.llm_intent.lower()
    assert "under eye darkness" in tired.llm_intent.lower()


def test_denied_topic_cools_down_and_is_not_reflagged():
    eng = CorroborationEngine()
    snap = [_res("rash", "rash", 0.3, 0.4)]
    eng.observe(snap, now=100.0)
    eng.mark_asked("skin_changes", 101.0)
    assert eng.hear("no, nothing like that", 105.0) == ("skin_changes", "denied")
    assert eng.pending_conclusions() == []
    eng.observe(snap, now=200.0)                 # within denied cooldown
    assert eng.status("skin_changes") == "denied"
    assert eng.next_question(200.0) is None


def test_unclear_answer_allows_one_reask():
    eng = CorroborationEngine()
    eng.observe([_res("rash", "rash", 0.3, 0.4)], now=100.0)
    eng.mark_asked("skin_changes", 101.0)
    assert eng.hear("what a lovely day", 105.0) == ("skin_changes", "unclear")
    assert eng.status("skin_changes") == "flagged"   # re-ask allowed
    eng.mark_asked("skin_changes", 110.0)
    assert eng.hear("hmm the birds are singing", 112.0)[1] == "unclear"
    assert eng.status("skin_changes") == "unclear"   # terminal after max asks


def test_answers_outside_window_are_free_speech():
    eng = CorroborationEngine(answer_window=5.0)
    eng.observe([_res("rash", "rash", 0.3, 0.4)], now=100.0)
    eng.mark_asked("skin_changes", 100.0)
    assert eng.hear("yes", 200.0) is None            # way past the window


# --------------------------------------------------------------- elicitation

def test_elicitation_window_lifecycle():
    es = ElicitationState.instance()
    es.begin("hold_still", 8.0, now=100.0)
    assert es.active("hold_still", now=104.0)
    assert not es.active("other_test", now=104.0)
    assert not es.active("hold_still", now=109.0)    # expired
    es.clear()


def _pose_ctx(t, wrist_x):
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[11] = [0.40, 0.45, 0.0, 1.0]
    lm[12] = [0.60, 0.45, 0.0, 1.0]
    lm[15] = [wrist_x, 0.70, 0.0, 1.0]               # left wrist (tracked)
    lm[16] = [0.80, 0.70, 0.0, 0.0]                  # right wrist invisible
    pose = PoseData(landmarks=lm, bbox=(0, 0, FRAME - 1, FRAME - 1))
    frame = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    return FrameContext(frame=frame, timestamp=t, frame_index=0, fps=30.0,
                        pose=pose, person_present=True)


def test_tremor_test_reports_oscillation_then_steady():
    from modules.tremor import Tremor
    es = ElicitationState.instance()

    # Run 1: a deliberate ~5 Hz shake during the window.
    mod = Tremor()
    es.begin("hold_still", 4.0, now=0.0)
    for i in range(120):                              # 4 s at 30 fps
        t = i / 30.0
        mod.process(_pose_ctx(t, 0.30 + 0.02 * np.sin(2 * np.pi * 5.0 * t)))
    out = mod.process(_pose_ctx(4.2, 0.30)) or []     # window closed: report
    test = next(r for r in out if r.key == "tremor_test")
    assert 4.0 <= float(test.value) <= 6.0
    assert test.severity == Severity.NOTICE

    # Run 2: genuinely still hand -> "steady".
    mod = Tremor()
    es.begin("hold_still", 4.0, now=100.0)
    for i in range(120):
        mod.process(_pose_ctx(100.0 + i / 30.0, 0.30))
    out = mod.process(_pose_ctx(104.2, 0.30)) or []
    test = next(r for r in out if r.key == "tremor_test")
    assert test.value == "steady"
    es.clear()


# -------------------------------------------------------------- voice agent

class _FakeListener:
    available = True

    def __init__(self):
        self.queue = []

    def pop_utterances(self):
        out, self.queue = self.queue, []
        return out

    def close(self):
        pass


def _agent():
    from agent.voice_agent import VoiceAgent
    # Corroboration tests exercise deterministic local wording, not cloud timing.
    return VoiceAgent(name="Ada", speak=False, listener=_FakeListener(),
                      moondream_enabled=False)


def test_voice_agent_full_corroboration_loop():
    ag = _agent()
    snap = [_res("rash", "rash", 0.3, 0.4)]
    said = ag.tick(snap, now=1000.0)                 # asks the follow-up
    assert said is not None and "skin" in said.lower()
    assert ag.corroboration.status("skin_changes") == "asked"
    ag.listener.queue.append(("yes, a bit itchy actually", 1005.0))
    said = ag.tick(snap, now=1012.0)                 # confirmed -> conclusion
    assert ag.corroboration.status("skin_changes") == "confirmed"
    assert said is not None and "doctor" in said.lower()
    assert ("them", "yes, a bit itchy actually", 1005.0) in ag.memory.dialogue


def test_voice_agent_denial_suppresses_conclusion():
    ag = _agent()
    snap = [_res("rash", "rash", 0.3, 0.4)]
    ag.tick(snap, now=1000.0)
    ag.listener.queue.append(("no, nothing like that", 1005.0))
    said = ag.tick(snap, now=1012.0)                 # replies, no conclusion
    assert ag.corroboration.status("skin_changes") == "denied"
    assert ag.corroboration.pending_conclusions() == []
    if said is not None:
        assert "doctor" not in said.lower()


def test_voice_agent_reply_bypasses_min_gap():
    ag = _agent()
    ag.tick([], now=1000.0)                          # small talk
    ag.listener.queue.append(("hello robot", 1001.0))
    said = ag.tick([], now=1002.0)                   # within min_gap (8 s)
    assert said is not None                          # reply interjects anyway


def test_voice_agent_scripted_test_opens_window():
    ag = _agent()
    ag.elicitation.clear()
    ag.request_test()
    said = ag.tick([], now=2000.0)
    assert said is not None and "eight seconds" in said.lower()
    assert ag.elicitation.active("hold_still")
    ag.elicitation.clear()


# ------------------------------------------------------------- expressivity

def _face_ctx(t, smile=0.20):
    lm = np.zeros((478, 3), dtype=np.float32)
    lm[FL.NOSE_TIP] = [0.50, 0.55, 0.0]
    lm[FL.CHIN] = [0.50, 0.85, 0.0]
    lm[FL.FOREHEAD_TOP] = [0.50, 0.20, 0.0]
    lm[FL.LEFT_FACE_EDGE] = [0.20, 0.55, 0.0]
    lm[FL.RIGHT_FACE_EDGE] = [0.80, 0.55, 0.0]
    half = smile / 2.0
    lm[FL.MOUTH_LEFT] = [0.50 - half, 0.65, 0.0]
    lm[FL.MOUTH_RIGHT] = [0.50 + half, 0.65, 0.0]
    lm[FL.MOUTH_TOP_INNER] = [0.50, 0.645, 0.0]
    lm[FL.MOUTH_BOTTOM_INNER] = [0.50, 0.655, 0.0]   # nearly closed mouth
    # left eye: corners 33/133, lids above/below
    lm[33] = [0.40, 0.50, 0.0]
    lm[133] = [0.46, 0.50, 0.0]
    for i in (160, 158, 159):
        lm[i] = [0.43, 0.49, 0.0]
    for i in (153, 144, 145):
        lm[i] = [0.43, 0.51, 0.0]
    # right eye: corners 362/263
    lm[362] = [0.54, 0.50, 0.0]
    lm[263] = [0.60, 0.50, 0.0]
    for i in (385, 387, 386):
        lm[i] = [0.57, 0.49, 0.0]
    for i in (373, 380, 374):
        lm[i] = [0.57, 0.51, 0.0]
    face = FaceData(landmarks=lm, bbox=(0, 0, FRAME - 1, FRAME - 1),
                    crop=np.zeros((10, 10, 3), dtype=np.uint8), has_iris=True)
    frame = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    return FrameContext(frame=frame, timestamp=t, frame_index=0, fps=30.0,
                        face=face, person_present=True)


def test_expressivity_flags_flat_affect_vs_history():
    from modules.expressivity import Expressivity
    with tempfile.TemporaryDirectory() as tmp:
        mod = Expressivity(store_every=5.0, low_hits_needed=2)
        mod.store = HistoryStore(path=Path(tmp) / "expr.db")
        try:
            day_seconds = mod.history_days * 86400.0
            mod.store.rolling_mean("expressivity", "smile_mean", day_seconds)
            mod.store.rolling_mean("expressivity", "expr_std", day_seconds)
            assert mod.store.wait_aggregates()
            now = _time.time()
            # History: a lively month. Today's constant smile=0.18 span is a
            # 0.30 smile-level metric (span / 0.6 face width), so history
            # sits well above it.
            for d in range(5):
                mod.store.add("expressivity", "smile_mean", 0.45,
                              ts=now - 86400.0 * (d + 1))
                mod.store.add("expressivity", "expr_std", 0.05,
                              ts=now - 86400.0 * (d + 1))
            # Today: a flat session (constant small smile).
            out = []
            for i in range(0, 900):                  # 30 s at 30 fps
                r = mod.process(_face_ctx(now + i / 30.0, smile=0.18))
                if r:
                    out.extend(r)
            keys = {r.key for r in out}
            assert "smile_level" in keys and "expression_variance" in keys
            assert "expressivity_low" in keys
        finally:
            mod.store.close()


# -------------------------------------------- cross-session asymmetry trend

def _asym_ctx(t, mouth_dy=0.0):
    """Mirror-symmetric face with an optional one-sided mouth droop (the
    same construction tests/facial_asymmetry_regions_test.py uses)."""
    ctx = _face_ctx(t)
    lm = ctx.face.landmarks
    lm[FL.MOUTH_RIGHT][1] += mouth_dy
    lm[FL.LEFT_CHEEK] = [0.35, 0.60, 0.0]
    lm[FL.RIGHT_CHEEK] = [0.65, 0.60, 0.0]
    return ctx


def test_asymmetry_trend_vs_stored_history():
    from modules.facial_asymmetry import FacialAsymmetry
    with tempfile.TemporaryDirectory() as tmp:
        store = HistoryStore(path=Path(tmp) / "asym.db")
        try:
            store.rolling_mean("facial_asymmetry", "base_mouth_2d", 30 * 86400.0)
            assert store.wait_aggregates()
            now = _time.time()
            for d in range(3):                       # symmetric history
                store.add("facial_asymmetry", "base_mouth_2d", 0.0,
                          ts=now - 86400.0 * (d + 1))
            mod = FacialAsymmetry()
            mod.store = store
            # learn a *drooped* mouth as today's baseline (real-clock
            # timestamps so mean_since windows behave)
            assert mod.process(_asym_ctx(now, mouth_dy=0.06)) is None
            out = mod.process(_asym_ctx(now + 25.0, mouth_dy=0.06)) or []
            trend = [r for r in out if r.key == "asymmetry_trend_mouth"]
            assert trend and trend[0].severity == Severity.NOTICE
            # and today's baseline was persisted for future visits
            assert store.mean_since("facial_asymmetry", "base_mouth_2d",
                                    3600.0) is not None
        finally:
            store.close()
