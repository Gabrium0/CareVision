"""Live humanoid-camera detection pipeline.

Usage:
  python main.py                        # default webcam (index 0), live window
  python main.py --source 1             # a different camera
  python main.py --source clip.mp4      # a video file
  python main.py --headless             # no window; prints greetings/alerts
  python main.py --name Margaret        # personalize greetings
  python main.py --max-frames 100       # process N frames then stop (testing)

Controls (windowed mode): 'q' quits, 'g' forces a greeting.

Architecture:
  Camera -> [FaceExtractor, PoseExtractor, MotionExtractor] -> Scheduler(modules)
         -> Aggregator -> GreetingEngine / Overlay
Modules are enabled/tuned in config/modules.yaml and auto-discovered from
modules/. Add a detector by dropping a file there and listing it in the yaml.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from core.camera import Camera
from core.pipeline import Pipeline
from core.registry import discover, build_enabled
from core.scheduler import Scheduler
from extractors.face import FaceExtractor
from extractors.pose import PoseExtractor
from extractors.motion import MotionExtractor
from output.aggregator import Aggregator
from output.greeting_engine import GreetingEngine

CONFIG = Path(__file__).resolve().parent / "config" / "modules.yaml"


def load_config():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(source, config):
    discover("modules")
    modules = build_enabled(config)
    print(f"[main] enabled modules: {', '.join(m.name for m in modules)}")
    extractors = [FaceExtractor(), PoseExtractor(), MotionExtractor()]
    scheduler = Scheduler(modules)
    aggregator = Aggregator()
    camera = Camera(source=source)
    pipeline = Pipeline(camera, extractors, scheduler, aggregator)
    return pipeline, aggregator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index, file path, or URL")
    ap.add_argument("--headless", action="store_true", help="no display window")
    ap.add_argument("--name", default="there", help="person's name for greetings")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    config = load_config()
    pipeline, aggregator = build_pipeline(args.source, config)
    greeter = GreetingEngine(name=args.name)

    display = not args.headless
    cv2 = None
    if display:
        import cv2 as _cv2
        cv2 = _cv2
        from output import overlay

    force_greet = {"v": False}
    last_alert_print = {"t": 0.0}

    def on_frame(ctx, results):
        greeting = greeter.maybe_greet(aggregator, force=force_greet["v"])
        force_greet["v"] = False
        if greeting:
            print("\n" + "=" * 50 + f"\n{greeting}\n" + "=" * 50)

        # surface alerts promptly even without an arrival
        alerts = greeter.alerts(aggregator.snapshot())
        if alerts and time.time() - last_alert_print["t"] > 5.0:
            for a in alerts:
                print(f"[ALERT] {a}")
            last_alert_print["t"] = time.time()

        if display:
            frame = overlay.draw(ctx.frame.copy(), ctx, aggregator.snapshot(), ctx.fps)
            cv2.imshow("Humanoid Camera — detections", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord("g"):
                force_greet["v"] = True
        return True

    print("[main] starting; press 'q' in the window to quit (Ctrl+C in headless).")
    try:
        pipeline.run(on_frame=on_frame, max_frames=args.max_frames)
    except KeyboardInterrupt:
        print("\n[main] stopped.")
    finally:
        if display and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
