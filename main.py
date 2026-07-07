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


def build_pipeline(source, config, camera_opts=None):
    discover("modules")
    modules = build_enabled(config)
    print(f"[main] enabled modules: {', '.join(m.name for m in modules)}")
    extractors = [FaceExtractor(), PoseExtractor(), MotionExtractor()]
    scheduler = Scheduler(modules)
    aggregator = Aggregator()
    camera = Camera(source=source, **(camera_opts or {}))
    pipeline = Pipeline(camera, extractors, scheduler, aggregator)
    return pipeline, aggregator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index, file path, or URL")
    ap.add_argument("--headless", action="store_true", help="no display window")
    ap.add_argument("--name", default="there", help="person's name for greetings")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--combined", action="store_true",
                    help="draw all text over the camera feed instead of a "
                         "separate data window")
    ap.add_argument("--no-lock", action="store_true",
                    help="keep camera auto-exposure/white-balance on "
                         "(default locks them for stable rPPG/skin color)")
    ap.add_argument("--exposure", type=float, default=None,
                    help="manual exposure value (camera-specific; try -6 to -4)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="requested capture fps (default 30)")
    args = ap.parse_args()

    camera_opts = {"lock": not args.no_lock, "exposure": args.exposure,
                   "request_fps": args.fps}

    config = load_config()
    pipeline, aggregator = build_pipeline(args.source, config, camera_opts)
    greeter = GreetingEngine(name=args.name)

    display = not args.headless
    cv2 = None
    if display:
        import cv2 as _cv2
        cv2 = _cv2
        from output import overlay, dashboard

    force_greet = {"v": False}
    last_alert_print = {"t": 0.0}
    last_greeting = {"text": None}

    def on_frame(ctx, results):
        greeting = greeter.maybe_greet(aggregator, force=force_greet["v"])
        force_greet["v"] = False
        if greeting:
            last_greeting["text"] = greeting
            print("\n" + "=" * 50 + f"\n{greeting}\n" + "=" * 50)

        # surface alerts promptly even without an arrival
        alerts = greeter.alerts(aggregator.snapshot())
        if alerts and time.time() - last_alert_print["t"] > 5.0:
            for a in alerts:
                print(f"[ALERT] {a}")
            last_alert_print["t"] = time.time()

        if display:
            snapshot = aggregator.snapshot()
            if args.combined:
                frame = overlay.draw(ctx.frame.copy(), ctx, snapshot, ctx.fps)
                cv2.imshow("Humanoid Camera — detections", frame)
            else:
                # camera window: video + boxes only
                cam = overlay.draw_boxes(ctx.frame.copy(), ctx, ctx.fps)
                cv2.imshow("Camera", cam)
                # separate, readable data window
                panel = dashboard.render(snapshot, ctx.fps, last_greeting["text"])
                cv2.imshow("Detections — Data", panel)
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
