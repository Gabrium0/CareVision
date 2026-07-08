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
import os
import time
from pathlib import Path

if os.environ.get("DEEPFACE_WORKER") != "1":
    os.environ.setdefault("KERAS_BACKEND", "jax")

import yaml

from core.camera import Camera
from core.pipeline import Pipeline
from core.registry import discover, build_enabled
from core.scheduler import Scheduler
from extractors.face import FaceExtractor
from extractors.pose import PoseExtractor
from extractors.motion import MotionExtractor
from output.aggregator import Aggregator
from alerts.manager import AlertManager
from agent.advisor_engine import AdvisorEngine
from agent.voice_agent import VoiceAgent
from agent.env import load_env
from webui.server import CompanionServer

CONFIG = Path(__file__).resolve().parent / "config" / "modules.yaml"
ALERTS_CONFIG = Path(__file__).resolve().parent / "config" / "alerts.yaml"


def load_config():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_alerts_config():
    if ALERTS_CONFIG.exists():
        with open(ALERTS_CONFIG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_pipeline(source, config, camera_opts=None):
    discover("modules")
    modules = build_enabled(config)
    print(f"[main] enabled modules: {', '.join(m.name for m in modules)}")
    extractors = [FaceExtractor(), PoseExtractor(), MotionExtractor()]
    scheduler = Scheduler(modules)
    aggregator = Aggregator()
    advisor_engine = AdvisorEngine.from_config(config.get("advice"))
    camera = Camera(source=source, **(camera_opts or {}))
    pipeline = Pipeline(camera, extractors, scheduler, aggregator, advisor_engine)
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
    ap.add_argument("--width", type=int, default=640,
                    help="requested/display processing width (default 640)")
    ap.add_argument("--height", type=int, default=480,
                    help="requested capture height (default 480)")
    ap.add_argument("--vitals-log-every", type=float, default=10.0,
                    help="seconds between terminal vitals summaries; 0 disables")
    ap.add_argument("--alert-cooldown", type=float, default=30.0,
                    help="seconds before repeating the same terminal alert")
    ap.add_argument("--no-deepface", action="store_true",
                    help="disable DeepFace subprocess backend for this run")
    ap.add_argument("--debug-modules", default="",
                    help="comma-separated debug logs: openrppg,clothing,deepface,drowsiness,weather or all")
    ap.add_argument("--no-voice", action="store_true",
                    help="disable the spoken voice agent (still prints its lines)")
    ap.add_argument("--voice-model", default="gemini-2.5-flash",
                    help="Gemini model for the voice agent (key from .env)")
    ap.add_argument("--webui", action="store_true",
                    help="serve the agent's text on a web page for an iPad "
                         "(prints the URL to open)")
    ap.add_argument("--webui-port", type=int, default=8770,
                    help="port for the companion web display (default 8770)")
    args = ap.parse_args()
    if args.debug_modules:
        os.environ["APP_DEBUG_MODULES"] = args.debug_modules
    load_env()   # make .env keys (GEMINI_API_KEY, alert creds) available

    camera_opts = {"lock": not args.no_lock, "exposure": args.exposure,
                   "request_fps": args.fps, "request_size": (args.width, args.height),
                   "target_width": args.width}

    config = load_config()
    if args.no_deepface:
        emotion = config.get("modules", {}).get("emotion", {})
        emotion["backends"] = [b for b in emotion.get("backends", []) if b != "deepface"]
    pipeline, aggregator = build_pipeline(args.source, config, camera_opts)

    alerts_cfg = load_alerts_config()
    alert_mgr = AlertManager.from_config(alerts_cfg) if alerts_cfg.get("enabled", True) else None
    voice_agent = VoiceAgent(name=args.name, speak=not args.no_voice,
                             model=args.voice_model)

    web = None
    if args.webui:
        web = CompanionServer(port=args.webui_port)
        web.start()

    display = not args.headless
    cv2 = None
    if display:
        import cv2 as _cv2
        cv2 = _cv2
        from output import overlay, dashboard

    force_greet = {"v": False}
    last_alert_print: dict[str, float] = {}
    last_vitals_print = {"t": 0.0}
    last_greeting = {"text": None}

    def print_vitals(snapshot):
        if args.vitals_log_every <= 0:
            return
        now = time.time()
        if now - last_vitals_print["t"] < args.vitals_log_every:
            return
        rows = [r for r in snapshot if r.module == "heart_rate"]
        if not rows:
            return
        last_vitals_print["t"] = now
        parts = [f"{r.key}={r.value} ({r.confidence:.2f})" for r in sorted(rows, key=lambda r: r.key)]
        print("[vitals] " + " | ".join(parts))

    def on_frame(ctx, results):
        snapshot = aggregator.snapshot()

        # deterministic caregiver alerting (independent of the LLM)
        if alert_mgr is not None:
            alert_mgr.evaluate(snapshot)

        # conversational voice agent: greets, small-talks, raises salient things
        if force_greet["v"]:
            voice_agent.policy._last_spoken = 0.0   # let 'g' force a line now
            force_greet["v"] = False
        utterance = voice_agent.tick(snapshot)
        if utterance:
            last_greeting["text"] = utterance
            if web is not None:
                web.publish(utterance)

        # mirror the full detections window to the /data web endpoint
        if web is not None:
            web.publish_data(snapshot, ctx.fps, last_greeting["text"])

        print_vitals(snapshot)

        if display:
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
        voice_agent.close()
        if display and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
