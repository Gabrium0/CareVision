"""Voice agent + caregiver alerting test (no camera, offline-deterministic).

Forces the templated fallback (no LLM key) so assertions are stable, then feeds
a scripted timeline and checks:
- the agent greets on arrival,
- it raises a salient clothing observation once and does not repeat it,
- a fall ALERT dispatches exactly one caregiver notification after confirmation.
"""
import os
import sys
from pathlib import Path

# Force templated fallback: hide keys AND stop .env from reloading them.
os.environ.pop("X-Moondream-Auth", None)
os.environ.pop("MOONDREAM_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.env as agent_env
agent_env.load_env = lambda: None            # neutralize .env loading for the test

from core.events import Result, Severity
from agent.voice_agent import VoiceAgent
from alerts.manager import AlertManager
from alerts.notifier import Channel


class Recorder(Channel):
    name = "recorder"

    def __init__(self):
        self.sent = []

    def send(self, subject, body):
        self.sent.append(subject)
        return True


def R(module, key, value, sev=Severity.INFO, msg="", conf=0.6):
    return Result(module=module, key=key, value=value, confidence=conf,
                  severity=sev, message=msg)


def main():
    agent = VoiceAgent(name="Margaret", speak=False)
    assert not agent.moondream.available, "test must run in templated (offline) mode"
    rec = Recorder()
    mgr = AlertManager(channels=[rec], confirm_seconds=0.1,
                       cooldown_seconds=120, escalate_after=300)

    present = R("presence", "present", True)
    arrival = R("presence", "arrival", True, Severity.NOTICE, "Person arrived")
    clothing = R("clothing_advice", "recommendation",
                 "It feels cool (14C). A sweater or light jacket could help.",
                 Severity.NOTICE, "cool")
    fall = R("fall", "fall", True, Severity.ALERT,
             "FALL DETECTED — person is down and horizontal")

    timeline = [
        (0.0,  [arrival, present]),
        (9.0,  [present, clothing]),
        (10.0, [present, clothing]),
        (18.0, [present, clothing]),
        (20.0, [present, fall]),
        (20.5, [present, fall]),
        (21.0, [present, fall]),
    ]
    said = {}
    for now, snap in timeline:
        said[now] = agent.tick(snap, now=now)
        mgr.evaluate(snap, now=now)

    agent.close()

    assert said[0.0], "expected a greeting on arrival"
    assert said[9.0], "expected a clothing remark once clothing is known"
    assert said[10.0] is None and said[18.0] is None, "must not repeat the clothing topic"
    assert rec.sent == ["FALL DETECTED"], f"expected one fall alert, got {rec.sent}"
    print("greeting:", said[0.0])
    print("clothing:", said[9.0])
    print("alerts  :", rec.sent)
    print("[agent-alert-test] OK")


if __name__ == "__main__":
    main()
