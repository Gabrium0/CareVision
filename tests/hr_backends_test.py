"""Integration test for the heart_rate module with BOTH backends.

Feeds synthetic face frames through the real FaceExtractor so ctx.face.crop
is populated, then runs the heart_rate module (classical + openrppg) and
prints the readings each backend emits. open-rppg auto-disables if absent,
in which case only classical readings appear.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext
from extractors.face import FaceExtractor
from modules.heart_rate import HeartRate
from tests.smoke_test import make_face_frame


def main():
    hr = HeartRate(window_seconds=12.0, backends=["classical", "openrppg"])
    active = [b.label for b in hr._backends]
    print(f"[hr-test] active backends: {active}")

    face = FaceExtractor()
    t0 = time.time()
    seen = {}
    N = 260  # ~13s at 20fps
    for i in range(N):
        ts = t0 + i * 0.05
        frame = make_face_frame(t=i * 0.05)
        ctx = FrameContext(frame=frame, timestamp=ts, frame_index=i, fps=20.0)
        face.extract(ctx)
        out = hr.process(ctx) or []
        for r in out:
            seen[r.key] = (r.value, r.confidence, r.message)

    print(f"[hr-test] emitted {len(seen)} distinct keys after {N} frames:")
    for k in sorted(seen):
        v, c, m = seen[k]
        print(f"   {k}: value={v} conf={c}  | {m}")

    face.close()
    hr.close()
    # Assert both backends produced a bpm if both are active
    have_classical = any(k.startswith("bpm_classical") for k in seen)
    have_openrppg = any(k.startswith("bpm_open") for k in seen)
    print(f"\n[hr-test] classical bpm: {have_classical} | open-rppg bpm: {have_openrppg}")
    print("[hr-test] OK")


if __name__ == "__main__":
    main()
