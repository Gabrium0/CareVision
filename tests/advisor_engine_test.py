"""Advisor engine unit checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.advisor_engine import AdvisorEngine
from core.events import Result, Severity


def R(module, key, value, conf=0.6, sev=Severity.INFO):
    return Result(module=module, key=key, value=value, confidence=conf, severity=sev)


def main():
    engine = AdvisorEngine.from_config({
        "enabled": True,
        "vitals": {"enabled": True, "interval": 0.0, "bpm_high": 100.0},
    })

    normal = engine.evaluate([R("heart_rate", "bpm_classical", 72)], now=0.0)
    assert normal == [], f"normal vitals should not advise, got {normal}"

    elevated = engine.evaluate([R("heart_rate", "bpm_classical", 108)], now=1.0)
    assert len(elevated) == 1
    assert elevated[0].module == "vitals_advice"
    assert elevated[0].key == "recommendation"
    assert elevated[0].severity == Severity.NOTICE

    combined = engine.evaluate([
        R("heart_rate", "bpm_classical", 108),
        R("pain", "pain", 0.5, sev=Severity.WARNING),
    ], now=2.0)
    assert len(combined) == 1
    assert combined[0].severity == Severity.WARNING

    loop_input = [R("vitals_advice", "recommendation", "old advice", sev=Severity.WARNING)]
    assert engine.evaluate(loop_input, now=3.0) == []

    print("[advisor-engine-test] OK")


if __name__ == "__main__":
    main()
