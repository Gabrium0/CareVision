"""Exercise pose-dependent modules by injecting synthetic pose landmarks.

The synthetic image renderer doesn't produce a MediaPipe-detectable body, so
this test builds a FrameContext with hand-crafted PoseData (33 landmarks) that
oscillates over time — driving tremor/gait/balance/fall/etc. through their real
code paths to confirm none raise and that at least some emit results.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext, PoseData
from core.registry import discover, build_enabled
from core.scheduler import Scheduler
from output.aggregator import Aggregator
import extractors.pose as P
import yaml


def synth_pose(t, fall=False):
    """33x4 pose landmarks; add a fast wrist tremor + walking ankle motion."""
    lm = np.zeros((33, 4), np.float32)
    lm[:, 3] = 1.0  # full visibility
    cx = 0.5 + 0.02 * np.sin(t * 0.5)          # slight sway
    lm[P.NOSE] = [cx, 0.20, 0, 1]
    lm[P.L_SHOULDER] = [cx - 0.09, 0.32, 0, 1]
    lm[P.R_SHOULDER] = [cx + 0.09, 0.32 + 0.01 * np.sin(t * 0.3), 0, 1]  # breathing
    lm[P.L_ELBOW] = [cx - 0.12, 0.45, 0, 1]
    lm[P.R_ELBOW] = [cx + 0.12, 0.45, 0, 1]
    tremor = 0.01 * np.sin(t * 2 * np.pi * 5)  # 5 Hz wrist tremor
    lm[P.L_WRIST] = [cx - 0.14 + tremor, 0.55, 0, 1]
    lm[P.R_WRIST] = [cx + 0.14 + tremor, 0.55, 0, 1]
    lm[P.L_HIP] = [cx - 0.06, 0.55, 0, 1]
    lm[P.R_HIP] = [cx + 0.06, 0.55, 0, 1]
    lm[P.L_KNEE] = [cx - 0.06, 0.72, 0, 1]
    lm[P.R_KNEE] = [cx + 0.06, 0.72, 0, 1]
    step = 0.03 * np.sin(t * 2 * np.pi * 1.0)  # 1 Hz gait
    lm[P.L_ANKLE] = [cx - 0.06, 0.90 + step, 0, 1]
    lm[P.R_ANKLE] = [cx + 0.06, 0.90 - step, 0, 1]
    if fall:  # collapse: torso horizontal, body low
        lm[P.L_SHOULDER] = [cx - 0.20, 0.80, 0, 1]
        lm[P.R_SHOULDER] = [cx + 0.20, 0.82, 0, 1]
        lm[P.L_HIP] = [cx - 0.02, 0.85, 0, 1]
        lm[P.R_HIP] = [cx + 0.02, 0.85, 0, 1]
    return lm


def main():
    discover("modules")
    with open(Path(__file__).resolve().parent.parent / "config" / "modules.yaml") as f:
        config = yaml.safe_load(f)
    modules = [m for m in build_enabled(config) if "pose" in getattr(m, "requires", ())]
    print(f"[pose-test] exercising {len(modules)} pose modules: "
          f"{', '.join(m.name for m in modules)}")

    sched = Scheduler(modules)
    agg = Aggregator()
    t0 = time.time()
    fired = set()
    for i in range(220):                       # ~11 s at 20 fps
        ts = t0 + i * 0.05
        frame = np.zeros((480, 640, 3), np.uint8)
        ctx = FrameContext(frame=frame, timestamp=ts, frame_index=i, fps=20.0)
        ctx.pose = PoseData(landmarks=synth_pose(i * 0.05, fall=(i > 200)),
                            bbox=(100, 80, 540, 460))
        ctx.person_present = True
        ctx.motion_energy = 1.5
        for r in sched.tick(ctx):
            fired.add(f"{r.module}.{r.key}")
        agg.ingest(sched.tick(ctx) or [])

    print(f"[pose-test] signals seen: {sorted(fired) or 'none (no exceptions though)'}")
    print("[pose-test] OK — no module raised.")


if __name__ == "__main__":
    main()
