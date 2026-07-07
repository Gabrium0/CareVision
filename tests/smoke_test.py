"""Headless smoke test: no camera required.

Verifies that (1) all modules import & self-register, (2) every enabled
module is constructible, (3) extractors + scheduler + aggregator run without
raising on synthetic frames including a real face image, and (4) the
greeting engine composes a message.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import FrameContext
from core.registry import discover, build_enabled, all_registered
from core.scheduler import Scheduler
from extractors.face import FaceExtractor
from extractors.pose import PoseExtractor
from extractors.motion import MotionExtractor
from output.aggregator import Aggregator
from output.greeting_engine import GreetingEngine
import yaml


def make_face_frame(w=640, h=480, t=0.0):
    """Very rough synthetic 'face + body' so extractors have something to find.
    MediaPipe may or may not lock on; the point is that nothing crashes."""
    img = np.full((h, w, 3), 60, np.uint8)
    import cv2
    # head
    cv2.ellipse(img, (w // 2, h // 3), (70, 90), 0, 0, 360, (170, 150, 140), -1)
    # eyes
    for dx in (-25, 25):
        cv2.circle(img, (w // 2 + dx, h // 3 - 10), 8, (255, 255, 255), -1)
        cv2.circle(img, (w // 2 + dx, h // 3 - 10), 4, (40, 40, 40), -1)
    # mouth (animate a little)
    open_px = int(6 + 4 * np.sin(t * 3))
    cv2.ellipse(img, (w // 2, h // 3 + 45), (25, open_px), 0, 0, 360, (90, 70, 70), -1)
    # torso
    cv2.rectangle(img, (w // 2 - 60, h // 2), (w // 2 + 60, h), (120, 110, 100), -1)
    return img


def main():
    discover("modules")
    reg = all_registered()
    print(f"[smoke] registered {len(reg)} modules: {', '.join(sorted(reg))}")

    with open(Path(__file__).resolve().parent.parent / "config" / "modules.yaml") as f:
        config = yaml.safe_load(f)
    modules = build_enabled(config)
    print(f"[smoke] built {len(modules)} enabled modules")

    extractors = [FaceExtractor(), PoseExtractor(), MotionExtractor()]
    scheduler = Scheduler(modules)
    agg = Aggregator()

    t0 = time.time()
    total_results = 0
    N = 40
    for i in range(N):
        # simulate ~20 fps timeline so time-based buffers advance
        ts = t0 + i * 0.05
        frame = make_face_frame(t=i * 0.05)
        ctx = FrameContext(frame=frame, timestamp=ts, frame_index=i, fps=20.0)
        for ex in extractors:
            ex.extract(ctx)
        results = scheduler.tick(ctx)
        agg.ingest(results)
        total_results += len(results)

    snap = agg.snapshot()
    print(f"[smoke] ran {N} frames, {total_results} results, "
          f"{len(snap)} live signals, face_found={ctx.face is not None}, "
          f"pose_found={ctx.pose is not None}")
    for r in snap:
        print(f"   - {r.module}.{r.key} = {r.value} (conf {r.confidence}, {r.severity.value})")

    greeter = GreetingEngine(name="Test")
    print("\n[smoke] sample greeting:\n" + greeter.compose(snap))

    for ex in extractors:
        close = getattr(ex, "close", None)
        if close:
            close()
    print("\n[smoke] OK — pipeline ran end-to-end without errors.")


if __name__ == "__main__":
    main()
