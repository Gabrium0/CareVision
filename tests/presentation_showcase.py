"""Narrated, offline feature showcase for the CareVision final presentation.

Drives the *real* engine, orchestrator, alert manager, TTS speaker, iPad wire
protocol, and skin-vision JSON rescue -- no staging, no network, no camera --
and prints each scenario as a readable transcript:

    CUE     what a detector reported
    AGENT   the question / spoken line the agent produced
    YOU     what the (simulated) person answered
    RESULT  the disposition the engine reached

Every scenario also asserts, so this file doubles as a test (see the thin
tests/presentation_showcase_test.py wrapper) and exits non-zero if anything
regresses. Pass ``--html PATH`` to render the same runs as a slide-ready report.

Run standalone:
    python tests/presentation_showcase.py
    python tests/presentation_showcase.py --html carevision_showcase.html
"""
from __future__ import annotations

import argparse
import contextlib
import html
import io
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ----------------------------------------------------------------- data model

ROLE_LABEL = {
    "cue": "CUE", "agent": "AGENT", "you": "YOU", "result": "RESULT",
    "guard": "GUARD", "data": "DATA", "note": "NOTE",
}


@dataclass
class Line:
    """One narrated transcript line (role badge + spoken/observed text)."""
    role: str
    text: str


@dataclass
class Check:
    """One asserted claim within a scenario."""
    desc: str
    ok: bool


@dataclass
class Scenario:
    """A single demonstrated behavior: narration lines plus asserted checks."""
    key: str
    area: str
    title: str
    blurb: str
    lines: list[Line] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    skipped: str | None = None

    def say(self, role: str, text: str) -> "Scenario":
        """Append a narration line."""
        self.lines.append(Line(role, str(text)))
        return self

    def expect(self, desc: str, ok: bool) -> "Scenario":
        """Record an asserted claim and whether it held."""
        self.checks.append(Check(desc, bool(ok)))
        return self

    @property
    def passed(self) -> bool:
        """True when the scenario ran and every check held."""
        return (self.skipped is None and bool(self.checks)
                and all(c.ok for c in self.checks))


# --------------------------------------------------------------- helpers

@contextlib.contextmanager
def _quiet():
    """Silence a subsystem's construction chatter during a demo build."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _isolate_storage(tmp: Path) -> None:
    """Point the process-wide EventStore/WorkflowEngine singletons at temp files
    so a demo never touches data/ (mirrors tests/tts_and_showcase_test.py)."""
    from storage.event_store import EventStore
    from core.workflows import WorkflowEngine
    EventStore._instance = EventStore(tmp / f"events_{uuid.uuid4().hex}.sqlite3")
    WorkflowEngine._instance = WorkflowEngine(event_store=EventStore._instance)


class _FakeListener:
    """Stand-in microphone: the demo feeds it the person's replies by hand."""
    available = True

    def __init__(self):
        self.queue: list[tuple[str, float]] = []

    def pop_utterances(self):
        """Drain and return queued (text, timestamp) utterances."""
        out, self.queue = self.queue, []
        return out

    def close(self):
        """No-op close to satisfy the agent's listener contract."""


def _res(module, key, value, conf, severity=None, message="x"):
    """Build a detector Result the way the existing tests do."""
    from core.events import Result, Severity
    return Result(module=module, key=key, value=value, confidence=conf,
                  severity=severity or Severity.NOTICE, message=message, ttl=10.0)


# ============================================================ SCENARIOS
# Each returns a Scenario. Grouped by area for the report's sections.

def scn_agent_confirm(tmp: Path) -> Scenario:
    """A1 -- low-confidence skin cue, confirmed, gentle conclusion (full agent)."""
    from core.events import Severity
    s = Scenario(
        "A1", "Conversational agent",
        "Skin cue -> asks -> confirmed -> gentle nudge",
        "A camera cue is only a prior. The agent doesn't announce it -- it "
        "asks, and only a corroborating answer earns a gentle, non-diagnostic "
        "suggestion. This runs the real VoiceAgent end to end, offline.")
    with _quiet():
        _isolate_storage(tmp)
        from agent.voice_agent import VoiceAgent
        ag = VoiceAgent(name="Ada", speak=False, listener=_FakeListener(),
                        moondream_enabled=False)
    try:
        snap = [_res("rash", "rash", 0.3, 0.4, Severity.NOTICE, "possible rash")]
        with _quiet():
            asked = ag.tick(snap, now=1000.0)
        s.say("cue", "rash detector: possible skin change (confidence 0.40, notice)")
        s.say("agent", asked)
        s.expect("agent asks about skin (not announcing the cue)",
                 asked is not None and "skin" in asked.lower())
        s.expect("topic 'skin_changes' now awaiting an answer",
                 ag.corroboration.status("skin_changes") == "asked")

        ag.listener.queue.append(("yes, a bit itchy actually", 1005.0))
        with _quiet():
            replied = ag.tick(snap, now=1012.0)
        s.say("you", "yes, a bit itchy actually")
        s.say("agent", replied)
        s.say("result", "topic 'skin_changes' -> confirmed")
        s.expect("answer interpreted as confirmed",
                 ag.corroboration.status("skin_changes") == "confirmed")
        s.expect("conclusion gently suggests a doctor",
                 replied is not None and "doctor" in replied.lower())
    finally:
        with _quiet():
            ag.close()
    return s


def scn_agent_deny(_tmp: Path) -> Scenario:
    """A2 -- a denial suppresses the topic instead of arguing with the person."""
    from agent.corroboration import CorroborationEngine
    from core.events import Severity
    s = Scenario(
        "A2", "Conversational agent",
        "Hydration cue -> denied -> cooled down",
        "A 'no' is respected: no conclusion is drawn and the topic goes quiet "
        "for hours rather than being re-raised on the next frame.")
    eng = CorroborationEngine()
    cue = _res("dry_lips", "lip_dryness", 1.2, 0.5, Severity.NOTICE)
    eng.observe([cue], now=100.0)
    topic, rule = eng.next_question(100.0)
    s.say("cue", "dry_lips detector: lip dryness (confidence 0.50, notice)")
    s.say("agent", rule.question)
    s.expect("cue routed to the hydration follow-up", topic == "hydration")
    s.expect("question is about drinking", "drink" in rule.question.lower())
    eng.mark_asked(topic, 101.0)
    verdict = eng.hear("no, I'm fine thanks", 105.0)
    s.say("you", "no, I'm fine thanks")
    s.say("result", "topic 'hydration' -> denied (no conclusion)")
    s.expect("answer interpreted as denied", verdict == ("hydration", "denied"))
    s.expect("no conclusion is queued", eng.pending_conclusions() == [])
    eng.observe([cue], now=200.0)     # same cue, still inside the denied cooldown
    s.say("note", "same cue re-appears seconds later -- stays quiet (cooldown)")
    s.expect("denied topic is not re-asked during cooldown",
             eng.next_question(200.0) is None
             and eng.status("hydration") == "denied")
    return s


def scn_agent_unclear(_tmp: Path) -> Scenario:
    """A3 -- an unclear reply buys exactly one gentle re-ask, then rests."""
    from agent.corroboration import CorroborationEngine
    s = Scenario(
        "A3", "Conversational agent",
        "Unclear answer -> one gentle re-ask",
        "Off-topic chatter isn't forced into yes/no. The agent re-asks once, "
        "then lets it go rather than nagging.")
    eng = CorroborationEngine()
    eng.observe([_res("rash", "rash", 0.3, 0.4)], now=100.0)
    _t, rule = eng.next_question(100.0)
    s.say("cue", "rash detector: possible skin change (confidence 0.40)")
    s.say("agent", rule.question)
    eng.mark_asked("skin_changes", 101.0)
    v1 = eng.hear("what a lovely day", 105.0)
    s.say("you", "what a lovely day")
    s.expect("first reply read as unclear", v1 == ("skin_changes", "unclear"))
    s.expect("topic re-opens for one gentle re-ask",
             eng.status("skin_changes") == "flagged")
    eng.mark_asked("skin_changes", 110.0)
    v2 = eng.hear("the birds are singing", 112.0)
    s.say("agent", rule.question + "  (gentle re-ask)")
    s.say("you", "the birds are singing")
    s.say("result", "topic 'skin_changes' -> unclear (rests, no nagging)")
    s.expect("second unclear reply is terminal", v2[1] == "unclear")
    s.expect("no further re-ask after the cap",
             eng.status("skin_changes") == "unclear")
    return s


def scn_agent_evidence_gate(_tmp: Path) -> Scenario:
    """A4 -- a cloud appearance cue never asks on its own; a local signal does."""
    from agent.corroboration import CorroborationEngine
    from core.events import Severity
    s = Scenario(
        "A4", "Conversational agent",
        "Evidence-gating: local trigger required",
        "A cloud vision opinion about appearance is context, not a trigger. It "
        "cannot start a health question by itself -- a local detector signal "
        "must corroborate first.")
    eng = CorroborationEngine()
    appearance = _res("skin_vision", "facial_appearance",
                      {"cues": {"under_eye_darkness": "mild"}}, 0.8, Severity.INFO)
    eng.observe([appearance], now=100.0)
    s.say("cue", "skin_vision (cloud): mild under-eye darkness -- appearance only")
    s.say("result", "no question raised (cloud cue is not a local trigger)")
    s.expect("cloud appearance cue alone raises no question",
             eng.next_question(100.0) is None)
    perclos = _res("drowsiness", "perclos", 0.25, 0.5, Severity.NOTICE,
                   "eyes closed 25% of the time")
    eng.observe([appearance, perclos], now=101.0)
    topic, rule = eng.next_question(101.0)
    s.say("cue", "drowsiness detector (local): PERCLOS 0.25 (notice)")
    s.say("agent", rule.question)
    s.say("result", "now a tiredness question is warranted")
    s.expect("local PERCLOS signal opens the tiredness question",
             topic == "tiredness")
    return s


def scn_agent_steering(_tmp: Path) -> Scenario:
    """A5 -- the LLM may reorder which flagged topic to raise, never invent one."""
    from agent.corroboration import CorroborationEngine
    from core.events import Severity
    s = Scenario(
        "A5", "Conversational agent",
        "LLM proposes, deterministic disposes",
        "With several topics flagged, an LLM selector may choose which to raise "
        "first -- but only from topics real detectors flagged, and a bad or "
        "crashing selector always falls back to the safe deterministic order.")

    def _two_cues(eng):
        eng.observe([_res("rash", "rash", 0.3, 0.4, Severity.NOTICE),
                     _res("drowsiness", "perclos", 0.25, 0.5, Severity.NOTICE)],
                    now=100.0)

    base = CorroborationEngine()
    _two_cues(base)
    default_topic, _ = base.next_question(100.0)
    s.say("cue", "two cues flagged: skin_changes (older) and tiredness")
    s.say("result", f"deterministic order would raise: {default_topic}")
    s.expect("default order raises the oldest flag (skin_changes)",
             default_topic == "skin_changes")

    steered = CorroborationEngine()
    _two_cues(steered)
    choice, _ = steered.next_question_steered(100.0, selector=lambda c: "tiredness")
    s.say("agent", "LLM selector reorders to: tiredness")
    s.expect("valid selector choice is honored", choice == "tiredness")

    crashy = CorroborationEngine()
    _two_cues(crashy)
    fb, _ = crashy.next_question_steered(
        100.0, selector=lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    s.say("guard", "a selector that crashes -> falls back to deterministic order")
    s.expect("crashing selector falls back to oldest flag",
             fb == "skin_changes")

    invent = CorroborationEngine()
    _two_cues(invent)
    inv, _ = invent.next_question_steered(100.0, selector=lambda c: "cancer_scare")
    s.say("guard", "a selector that invents a topic -> rejected, falls back")
    s.expect("selector cannot invent a topic outside the flagged set",
             inv == "skin_changes")
    return s


def scn_agent_validators(_tmp: Path) -> Scenario:
    """A6 -- pure airlocks discard unsafe/off-address/leaky generated text."""
    from agent.corroboration import (safe_check_in, enforce_second_person,
                                     strip_prior_disclosure)
    s = Scenario(
        "A6", "Conversational agent",
        "Validator-guarantees: the safety airlock",
        "Whatever an LLM phrases passes deterministic airlocks before anyone "
        "hears it. Diagnostic claims, third-person drift, and disclosure of the "
        "camera's prior are all discarded for hand-authored fallback text.")
    fallback = "How are you feeling today?"

    diagnostic = "This looks like a symptom of a stroke."
    out1 = safe_check_in(diagnostic, fallback)
    s.say("guard", f"generated: {diagnostic}")
    s.say("result", f"spoken: {out1}")
    s.expect("diagnostic language is blocked -> safe fallback", out1 == fallback)

    third = "They seem tired today and should rest."
    out2 = enforce_second_person(third, fallback)
    s.say("guard", f"generated: {third}")
    s.say("result", f"spoken: {out2}")
    s.expect("third-person drift is blocked -> safe fallback", out2 == fallback)

    leak = "I noticed a rash in the living room - have you felt itchy?"
    out3 = strip_prior_disclosure(leak, fallback)
    s.say("guard", f"generated: {leak}")
    s.say("result", f"spoken: {out3}")
    s.expect("the camera's prior is stripped, the real question kept",
             "itchy" in out3.lower() and "noticed" not in out3.lower())

    good = "Have you had enough to drink today?"
    out4 = safe_check_in(good, fallback)
    s.say("agent", f"generated: {good}")
    s.say("result", f"spoken: {out4} (clean phrasing survives)")
    s.expect("safe, natural phrasing passes through unchanged", out4 == good)
    return s


def scn_alerts_primary_only(_tmp: Path) -> Scenario:
    """B -- only the primary person's ALERT escalates to a caregiver."""
    from alerts.manager import AlertManager
    from alerts.notifier import Channel
    from core.events import Result, Severity
    s = Scenario(
        "B1", "Caregiver alerts",
        "Primary-only escalation (deterministic)",
        "In a multi-person room the deterministic alert path pages a caregiver "
        "only for the primary resident. A visitor's fall stays on the dashboard "
        "but never triggers an external page.")

    class _Recorder(Channel):
        name = "recorder"

        def __init__(self):
            self.sent = []

        def send(self, subject, body):
            """Record which subject a page was sent for."""
            self.sent.append(subject)
            return True

    def _fall(subject_id):
        return Result(module="fall", key="fall", value=True, confidence=0.9,
                      severity=Severity.ALERT, message="FALL DETECTED",
                      subject_id=subject_id)

    rec_p = _Recorder()
    mgr_p = AlertManager(channels=[rec_p], confirm_seconds=1.0,
                         cooldown_seconds=60.0, escalate_after=5.0)
    for step in range(1, 20):
        mgr_p.evaluate([_fall("primary")], now=float(step))
    s.say("cue", "sustained FALL alert on subject 'primary'")
    s.say("result", f"caregiver paged: {bool(rec_p.sent)}")
    s.expect("primary ALERT reaches the caregiver", bool(rec_p.sent))

    rec_s = _Recorder()
    mgr_s = AlertManager(channels=[rec_s], confirm_seconds=1.0,
                         cooldown_seconds=60.0, escalate_after=5.0)
    for step in range(1, 20):
        mgr_s.evaluate([_fall("track-2")], now=float(step))
    s.say("cue", "sustained FALL alert on a secondary subject 'track-2'")
    s.say("result", f"caregiver paged: {bool(rec_s.sent)}")
    s.expect("secondary ALERT never escalates", rec_s.sent == [])
    return s


def scn_tts_and_showcase(tmp: Path) -> Scenario:
    """C -- TTS engine selection and the narrated showcase circuit."""
    from audio.tts import Speaker
    from audio import tts_piper
    from agent.voice_agent import VoiceAgent, _DEMO_CIRCUIT
    from storage.event_store import EventStore
    from core.workflows import WorkflowEngine
    s = Scenario(
        "C1", "Voice & narrated tour",
        "Neural TTS chain and the guided showcase",
        "Speech degrades gracefully (piper -> pyttsx3 -> print) and the "
        "narrated tour speaks one intro, runs three quick checks, and closes "
        "exactly once -- the flow that plays on the iPad's speaker.")

    speaker = Speaker(enabled=False)
    status = speaker.status()
    s.say("data", f"disabled speaker resolves to engine: {status['engine']}")
    s.expect("speaker always resolves to a working backend",
             status["engine"] == "print")
    voices = 0
    with contextlib.suppress(Exception):
        voices = len(tts_piper.available_voices())
    s.say("data", f"piper available={tts_piper.dependency_available()}, "
                  f"bundled voices={voices} (neural voice used when present)")

    with _quiet():
        _isolate_storage(tmp)
        agent = VoiceAgent(speak=False, moondream_enabled=False)
        agent.workflows = WorkflowEngine(
            event_store=EventStore(tmp / f"wf_{uuid.uuid4().hex}.sqlite3"))
    try:
        with _quiet():
            started = agent.start_showcase()
        intro = (agent.last_utterance or "").lower()
        s.say("agent", agent.last_utterance)
        s.expect("showcase starts", started is True)
        s.expect("intro previews three quick checks incl. balance",
                 "three quick checks" in intro and "balance" in intro)
        s.expect(f"first step is the circuit's opener ({_DEMO_CIRCUIT[0]})",
                 agent.workflows.active("primary").protocol == _DEMO_CIRCUIT[0])
        closing_seen = 0
        for _step in range(len(_DEMO_CIRCUIT)):
            with _quiet():
                agent.workflows.conclude("primary")
                agent._advance_demo_circuit()
            if "little tour" in (agent.last_utterance or "").lower():
                closing_seen += 1
        s.say("result", f"ran {len(_DEMO_CIRCUIT)} checks, closed the tour "
                        f"{closing_seen} time(s)")
        s.expect("the tour closes exactly once", closing_seen == 1)
        s.expect("no step is left running", agent.workflows.active("primary") is None)
    finally:
        with _quiet():
            agent.close()
    return s


def scn_ipad_protocol(_tmp: Path) -> Scenario:
    """D -- iPad frame wire-format round-trips and the link's gap accounting."""
    from core.ipad_camera import (pack_frame_header, unpack_frame_header,
                                  FRAME_MAGIC, FRAME_HEADER_SIZE)
    from core.ipad_link import IPadLink
    s = Scenario(
        "D1", "iPad capture link",
        "Frame protocol round-trip & loss accounting",
        "The 24-byte frame header packs and unpacks losslessly, rejects "
        "corrupt/short buffers instead of crashing, and the receiver correctly "
        "counts dropped frames and reassembles chunked ones.")

    combos = [(0, 0.0, 0, 0), (1, 1.234, 640, 480),
              (2 ** 32 - 1, 123456.789, 1920, 1080), (42, -0.0, 960, 540)]
    ok_roundtrip = True
    for seq, mt, w, h in combos:
        parsed = unpack_frame_header(pack_frame_header(seq, mt, w, h))
        if not (parsed and parsed["seq"] == seq and parsed["media_time"] == mt
                and parsed["width"] == w and parsed["height"] == h):
            ok_roundtrip = False
    s.say("cue", f"packed/unpacked {len(combos)} headers (seq, time, w, h)")
    s.say("result", "all fields round-tripped exactly")
    s.expect("header packs and unpacks losslessly", ok_roundtrip)

    bad_magic = b"XXXX" + b"\x00" * (FRAME_HEADER_SIZE - 4)
    s.expect("a corrupt magic is rejected (returns None, no crash)",
             unpack_frame_header(bad_magic) is None)
    s.expect("a truncated buffer is rejected (returns None, no crash)",
             unpack_frame_header((FRAME_MAGIC + b"\x00" * 20)[:12]) is None)

    link = IPadLink(relay_url="https://x", room="r", secret="s", code="123456")
    try:
        for seq in (1, 2, 4, 5):                     # seq 3 dropped
            link._ingest_frame(pack_frame_header(seq, seq * 0.05, 640, 480) + b"d")
        st = link.status()
        s.say("cue", "received frames 1, 2, 4, 5 (frame 3 lost in transit)")
        s.say("result", f"frames_rx={st['frames_rx']}, seq_gaps={st['seq_gaps']}")
        s.expect("receiver counts 4 frames and 1 gap",
                 st["frames_rx"] == 4 and st["seq_gaps"] == 1)

        first = pack_frame_header(11, 0.55, 640, 480, 0, 2) + b"chunk-a"
        second = pack_frame_header(11, 0.55, 640, 480, 1, 2) + b"chunk-b"
        r1 = link._ingest_frame(first)
        r2 = link._ingest_frame(second)
        s.say("cue", "a frame split into 2 chunks arrives")
        s.say("result", "completes only once the last chunk lands")
        s.expect("a chunked frame completes only when whole",
                 r1 is None and r2 == 11)
    finally:
        with contextlib.suppress(Exception):
            close = getattr(link, "close", None)
            if close:
                close()
    return s


def scn_skin_vision_rescue(_tmp: Path) -> Scenario:
    """E -- prose replies are salvaged, but a rescue can only be less alarming."""
    from modules.skin_vision import _extract_json, _normalize_rescued
    s = Scenario(
        "E1", "Cloud skin screen",
        "Robust JSON rescue (fail-safe by design)",
        "The vision endpoint sometimes replies in prose. The parser salvages "
        "the JSON it can, drops chatter, and -- crucially -- never invents "
        "the fields that decide whether something was seen, so a rescue can only "
        "ever be less alarming than the model's own words.")

    prose = ("Sure! Here is the analysis you asked for:\n```json\n"
             '{"image_quality": "good", "chatty_note": "looks fine to me"}\n'
             "```\nHope that helps!")
    parsed = _extract_json(prose)
    s.say("cue", "model replies with JSON wrapped in prose + a markdown fence")
    s.say("result", f"salvaged object with keys: {sorted(parsed)}")
    s.expect("JSON is salvaged from the surrounding prose",
             parsed.get("image_quality") == "good")

    rescued = _normalize_rescued(parsed, "preliminary")
    s.say("result", "normalized: chatter dropped, decision fields NOT invented")
    s.expect("unknown chatter key is dropped", "chatty_note" not in rescued)
    s.expect("known field is kept", rescued.get("image_quality") == "good")
    s.expect("absent facial cues fill to neutral 'unclear'",
             rescued.get("lip_dryness") == "unclear")
    s.expect("'finding_present' is never fabricated",
             "finding_present" not in rescued)
    s.expect("'confidence' / 'visible_features' are never fabricated",
             "confidence" not in rescued and "visible_features" not in rescued)

    raised = False
    try:
        _extract_json("there is no json object here at all")
    except ValueError:
        raised = True
    s.say("guard", "a reply with no object at all -> rejected (json_parse)")
    s.expect("a reply with no JSON object is rejected, not guessed", raised)
    return s


def scn_vitals_note(_tmp: Path) -> Scenario:
    """F -- how vitals accuracy is demonstrated (Apple Watch video)."""
    s = Scenario(
        "F1", "Vitals",
        "Heart rate: measured against a wearable",
        "Contactless heart rate (rPPG) accuracy is shown live from a recorded "
        "clip with an Apple Watch reading as ground truth -- the automated "
        "backends already pass their signal tests in the suite.")
    s.say("note", "run: python scripts/rppg_reference_check.py "
                  "--video clip.mp4 --reference-bpm 72")
    s.say("note", "reports detected BPM and the error against the watch reading")
    s.say("note", "deterministic rPPG signal tests: tests/rppg_signal_test.py, "
                  "tests/heart_rate_canonical_test.py")
    s.expect("informational -- no automated assertion here", True)
    return s


SCENARIOS = [
    scn_agent_confirm, scn_agent_deny, scn_agent_unclear, scn_agent_evidence_gate,
    scn_agent_steering, scn_agent_validators, scn_alerts_primary_only,
    scn_tts_and_showcase, scn_ipad_protocol, scn_skin_vision_rescue,
    scn_vitals_note,
]


def run_all() -> list[Scenario]:
    """Run every scenario; a missing dependency skips, any other error fails."""
    results: list[Scenario] = []
    # ignore_cleanup_errors: the isolated SQLite stores keep an open handle on
    # Windows, so let the OS reap the temp files rather than crash the demo.
    with tempfile.TemporaryDirectory(prefix="carevision_showcase_",
                                     ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        for fn in SCENARIOS:
            try:
                results.append(fn(tmp))
            except ImportError as exc:
                sc = Scenario(fn.__name__, "(skipped)", fn.__name__, "")
                sc.skipped = f"dependency unavailable: {exc}"
                results.append(sc)
            except Exception as exc:  # noqa: BLE001 - surface a real regression
                sc = Scenario(fn.__name__, "(error)", fn.__name__, "")
                sc.expect(f"scenario raised: {type(exc).__name__}: {exc}", False)
                results.append(sc)
    return results


# ------------------------------------------------------------- console report

def print_console(results: list[Scenario]) -> None:
    """Render the narrated transcript to the terminal."""
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    bar = "-" * 74
    print("\n" + bar)
    print(" CareVision - feature showcase (offline, deterministic)")
    print(bar)
    current_area = None
    for s in results:
        if s.area != current_area:
            current_area = s.area
            print(f"\n# {current_area}")
        if s.skipped:
            print(f"\n  [{s.key}] {s.title}\n    SKIPPED - {s.skipped}")
            continue
        print(f"\n  [{s.key}] {s.title}")
        if s.blurb:
            print("    " + s.blurb)
        for ln in s.lines:
            print(f"      {ROLE_LABEL.get(ln.role, ln.role.upper()):<7} {ln.text}")
        for c in s.checks:
            print(f"      {'[PASS]' if c.ok else '[FAIL]'} {c.desc}")
        print(f"      => {'PASS' if s.passed else 'FAIL'}")
    ran = [s for s in results if not s.skipped]
    passed = [s for s in ran if s.passed]
    skipped = [s for s in results if s.skipped]
    print("\n" + bar)
    print(f" {len(passed)}/{len(ran)} scenarios passed"
          + (f", {len(skipped)} skipped" if skipped else ""))
    print(bar + "\n")


# ---------------------------------------------------------------- HTML report

_HTML_HEAD = """<title>CareVision Feature Proof</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#F4F7F7; --surface:#FFFFFF; --surface-2:#EEF3F2; --ink:#13201F;
  --muted:#5C6C6B; --line:#DCE5E3; --teal:#0E7C7B;
  --pass:#1E8E5A; --fail:#C0392B; --cue:#8A6D3B;
  --shadow:0 1px 2px rgba(19,32,31,.06),0 8px 24px rgba(19,32,31,.05);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E1514; --surface:#17221F; --surface-2:#1E2C29; --ink:#E7EDEB;
  --muted:#9BB0AD; --line:#2A3A37; --teal:#54B7B3;
  --pass:#4CC585; --fail:#E77A6E; --cue:#C9A24B;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --ground:#0E1514; --surface:#17221F; --surface-2:#1E2C29; --ink:#E7EDEB;
  --muted:#9BB0AD; --line:#2A3A37; --teal:#54B7B3;
  --pass:#4CC585; --fail:#E77A6E; --cue:#C9A24B;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased;}
.wrap{max-width:980px;margin:0 auto;padding:56px 24px 72px;}
header.top{border-bottom:1px solid var(--line);padding-bottom:28px;}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--teal);
  font-weight:600;margin:0 0 10px;}
h1{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:clamp(30px,5vw,46px);
  line-height:1.05;margin:0 0 12px;text-wrap:balance;letter-spacing:-.01em;}
.lede{color:var(--muted);max-width:62ch;margin:0;font-size:16px;}
.stats{display:flex;flex-wrap:wrap;gap:14px;margin-top:26px;}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:14px 18px;box-shadow:var(--shadow);min-width:120px;}
.stat .n{font-family:"Fraunces",Georgia,serif;font-size:30px;font-weight:600;
  font-variant-numeric:tabular-nums;line-height:1;}
.stat .l{font-size:12px;color:var(--muted);letter-spacing:.05em;text-transform:uppercase;margin-top:6px;}
.stat.ok .n{color:var(--pass)} .stat.bad .n{color:var(--fail)}
section.area{margin-top:44px;}
h2{font-family:"IBM Plex Sans",sans-serif;font-size:14px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  margin:0 0 16px;padding-bottom:8px;border-bottom:1px solid var(--line);}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;
  padding:22px 24px;margin-bottom:16px;box-shadow:var(--shadow);}
.card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;}
.card h3{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:21px;
  margin:0 0 6px;letter-spacing:-.01em;}
.key{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--teal);
  font-weight:500;letter-spacing:.05em;}
.blurb{color:var(--muted);margin:2px 0 18px;font-size:14.5px;max-width:68ch;}
.pill{flex:none;font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:500;
  padding:5px 12px;border-radius:999px;letter-spacing:.06em;}
.pill.pass{background:color-mix(in srgb,var(--pass) 15%,transparent);color:var(--pass);}
.pill.fail{background:color-mix(in srgb,var(--fail) 15%,transparent);color:var(--fail);}
.pill.skip{background:var(--surface-2);color:var(--muted);}
.transcript{background:var(--surface-2);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;margin-bottom:16px;overflow-x:auto;}
.tline{display:grid;grid-template-columns:76px 1fr;gap:12px;align-items:baseline;
  padding:4px 0;font-family:"IBM Plex Mono",monospace;font-size:13.5px;}
.tline .role{font-size:11px;font-weight:500;letter-spacing:.08em;text-align:right;padding-top:1px;}
.role.cue{color:var(--cue)}.role.agent{color:var(--teal)}.role.you{color:#7B6FC0}
.role.result{color:var(--ink)}.role.guard{color:var(--fail)}
.role.data,.role.note{color:var(--muted)}
.checks{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:7px;}
.checks li{display:grid;grid-template-columns:20px 1fr;gap:8px;font-size:14px;align-items:baseline;}
.mark{font-weight:700;font-family:"IBM Plex Mono",monospace;}
.mark.ok{color:var(--pass)} .mark.no{color:var(--fail)}
footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);
  color:var(--muted);font-size:13px;}
footer code{font-family:"IBM Plex Mono",monospace;background:var(--surface-2);
  padding:2px 6px;border-radius:5px;}
</style>
"""


def _role_class(role: str) -> str:
    """Map a line role to its transcript CSS class."""
    return role if role in ("cue", "agent", "you", "result", "guard", "data",
                            "note") else "note"


def build_html(results: list[Scenario]) -> str:
    """Render the scenarios as a self-contained, theme-aware HTML fragment."""
    ran = [s for s in results if not s.skipped]
    passed = sum(1 for s in ran if s.passed)
    skipped = sum(1 for s in results if s.skipped)
    checks = sum(len(s.checks) for s in ran)
    stamp = time.strftime("%Y-%m-%d %H:%M")

    out = [_HTML_HEAD, '<div class="wrap">']
    out.append('<header class="top">')
    out.append('<p class="eyebrow">CareVision &middot; elderly-care AI</p>')
    out.append("<h1>Every feature, demonstrated end&#8209;to&#8209;end</h1>")
    out.append('<p class="lede">Each card runs the real system offline -- the '
               "conversational agent, caregiver alerts, voice tour, iPad capture "
               "link, and cloud skin screen -- showing the actual inputs it saw "
               "and the outputs it produced.</p>")
    allok = passed == len(ran) and skipped == 0
    out.append('<div class="stats">')
    out.append(f'<div class="stat {"ok" if allok else "bad"}"><div class="n">'
               f'{passed}/{len(ran)}</div><div class="l">Scenarios&nbsp;passed</div></div>')
    out.append(f'<div class="stat"><div class="n">{checks}</div>'
               '<div class="l">Assertions</div></div>')
    out.append(f'<div class="stat"><div class="n">{len({s.area for s in ran})}</div>'
               '<div class="l">Feature&nbsp;areas</div></div>')
    if skipped:
        out.append(f'<div class="stat"><div class="n">{skipped}</div>'
                   '<div class="l">Skipped</div></div>')
    out.append("</div></header>")

    current = None
    for s in results:
        if s.area != current:
            if current is not None:
                out.append("</section>")
            current = s.area
            out.append(f'<section class="area"><h2>{html.escape(current)}</h2>')
        out.append('<div class="card"><div class="card-head"><div>')
        out.append(f'<div class="key">{html.escape(s.key)}</div>')
        out.append(f"<h3>{html.escape(s.title)}</h3></div>")
        if s.skipped:
            out.append('<span class="pill skip">SKIPPED</span></div>')
            out.append(f'<p class="blurb">{html.escape(s.skipped)}</p></div>')
            continue
        pill = "pass" if s.passed else "fail"
        out.append(f'<span class="pill {pill}">{pill.upper()}</span></div>')
        if s.blurb:
            out.append(f'<p class="blurb">{html.escape(s.blurb)}</p>')
        if s.lines:
            out.append('<div class="transcript">')
            for ln in s.lines:
                rc = _role_class(ln.role)
                out.append(
                    f'<div class="tline"><span class="role {rc}">'
                    f'{html.escape(ROLE_LABEL.get(ln.role, ln.role.upper()))}</span>'
                    f"<span>{html.escape(ln.text)}</span></div>")
            out.append("</div>")
        out.append('<ul class="checks">')
        for c in s.checks:
            m = "ok" if c.ok else "no"
            g = "&#10003;" if c.ok else "&#10007;"
            out.append(f'<li><span class="mark {m}">{g}</span>'
                       f"<span>{html.escape(c.desc)}</span></li>")
        out.append("</ul></div>")
    if current is not None:
        out.append("</section>")

    out.append(f'<footer>Generated {stamp} &middot; offline &amp; deterministic '
               "(no camera, no network) &middot; reproduce with "
               "<code>python tests/presentation_showcase.py</code></footer>")
    out.append("</div>")
    return "\n".join(out)


def main() -> int:
    """Run the showcase; print to console and optionally write the HTML report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", metavar="PATH",
                    help="also write a slide-ready HTML report to PATH")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the console transcript (use with --html)")
    args = ap.parse_args()

    results = run_all()
    if not args.quiet:
        print_console(results)
    if args.html:
        path = Path(args.html)
        path.write_text(build_html(results), encoding="utf-8")
        print(f"[showcase] wrote HTML report -> {path}")

    ran = [s for s in results if not s.skipped]
    return 0 if all(s.passed for s in ran) else 1


if __name__ == "__main__":
    raise SystemExit(main())
