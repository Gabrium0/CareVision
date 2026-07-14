"""Live humanoid-camera detection pipeline.

Usage:
  python main.py                        # default webcam (index 0), live window
  python main.py --source 1             # a different camera
  python main.py --source clip.mp4      # a video file
  python main.py --source realsense     # RealSense D435i (depth + IMU)
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
from core.camera_factory import make_camera
from core.pipeline import Pipeline
from core.showcase import ShowcaseGate
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
    """Load the module configuration YAML."""
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_alerts_config():
    """Load the caregiver-alerts YAML (empty dict if absent)."""
    if ALERTS_CONFIG.exists():
        with open(ALERTS_CONFIG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_pipeline(source, config, camera_opts=None, max_staleness: float = 0.25):
    """Discover modules and assemble the full processing pipeline."""
    discover("modules")
    modules = build_enabled(config)
    print(f"[main] enabled modules: {', '.join(m.name for m in modules)}")
    extractors = [FaceExtractor(), PoseExtractor(), MotionExtractor()]
    scheduler = Scheduler(modules)
    aggregator = Aggregator()
    advisor_engine = AdvisorEngine.from_config(config.get("advice"))
    camera = make_camera(source, camera_opts or {})
    pipeline = Pipeline(camera, extractors, scheduler, aggregator, advisor_engine,
                        max_staleness=max_staleness,
                        showcase_gate=ShowcaseGate.from_config(config))
    return pipeline, aggregator


def main():
    """Parse CLI args and run the live detection pipeline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="camera index, file path, URL, or 'realsense' for a "
                         "RealSense D435i (depth + IMU; needs pyrealsense2)")
    ap.add_argument("--alt-source", default="0",
                    help="the other camera to toggle to with 'c' (default 0 = "
                         "laptop cam; 'realsense' also works here)")
    ap.add_argument("--list-cameras", action="store_true",
                    help="probe camera indices, print index+resolution, and exit")
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
    ap.add_argument("--target-brightness", type=float, default=90.0,
                    help="mean luminance (0-255) to reach by raising exposure/gain "
                         "before locking; higher = brighter (default 90)")
    ap.add_argument("--no-gain-boost", action="store_true",
                    help="only raise exposure (not sensor gain) when brightening a "
                         "dark scene; gain adds noise")
    ap.add_argument("--vitals-max-staleness", type=float, default=0.25,
                    help="max seconds the vitals fast-path may reuse a detected face "
                         "bbox before pausing rather than sampling a stale ROI "
                         "(default 0.25)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="requested capture fps (default 30)")
    ap.add_argument("--resolution", default="auto",
                    help="'auto' probes candidate resolutions on startup and picks the "
                         "largest that holds --min-fps (default), or an explicit WxH "
                         "e.g. 1280x720 to skip probing")
    ap.add_argument("--min-fps", type=float, default=25.0,
                    help="fps floor for --resolution auto, below which the FFT-based "
                         "vitals (heart rate/respiration/tremor) start to alias (default 25)")
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
    ap.add_argument("--listen", action="store_true",
                    help="enable the microphone listener (speech-to-text via "
                         "faster-whisper; pip install -r requirements-asr.txt)")
    ap.add_argument("--whisper-model", default="base",
                    help="faster-whisper model size for --listen (default base)")
    ap.add_argument("--voice-model", default="gemini-2.5-flash",
                    help="Gemini model for the voice agent (key from .env)")
    ap.add_argument("--enable-cloud-skin", action="store_true",
                    help="consent to upload sampled camera stills to the configured "
                         "NVIDIA skin-screening model for this run")
    ap.add_argument("--webui", action="store_true",
                    help="serve the agent's text on a web page for an iPad "
                         "(prints the URL to open)")
    ap.add_argument("--webui-port", type=int, default=8770,
                    help="port for the companion web display (default 8770)")
    args = ap.parse_args()
    if args.list_cameras:
        Camera.list_devices()
        return
    if args.debug_modules:
        os.environ["APP_DEBUG_MODULES"] = args.debug_modules
    load_env()   # make .env keys (GEMINI_API_KEY, alert creds) available

    auto_resolution = args.resolution.strip().lower() == "auto"
    if auto_resolution:
        request_size = Camera._RESOLUTION_CANDIDATES[0]   # probe overrides this on open()
    else:
        try:
            rw, rh = (int(v) for v in args.resolution.lower().split("x", 1))
        except ValueError:
            ap.error(f"--resolution must be 'auto' or WxH (e.g. 1280x720), got "
                      f"{args.resolution!r}")
        request_size = (rw, rh)
    camera_opts = {"lock": not args.no_lock, "exposure": args.exposure,
                   "request_fps": args.fps, "request_size": request_size,
                   "target_width": request_size[0],
                   "target_brightness": args.target_brightness,
                   "allow_gain_boost": not args.no_gain_boost,
                   "auto_resolution": auto_resolution,
                   "min_fps": args.min_fps}

    config = load_config()
    skin_cfg = config.get("modules", {}).get("skin_vision", {})
    skin_cfg["consent"] = bool(args.enable_cloud_skin)
    if args.no_deepface:
        emotion = config.get("modules", {}).get("emotion", {})
        emotion["backends"] = [b for b in emotion.get("backends", []) if b != "deepface"]
    pipeline, aggregator = build_pipeline(args.source, config, camera_opts,
                                          max_staleness=args.vitals_max_staleness)
    # Runtime camera toggle ('c'): start on --source, swap to --alt-source and
    # back. Both use the same camera_opts so the laptop view is unchanged.
    primary_source, alt_source = args.source, args.alt_source
    cam_state = {"current": primary_source}

    alerts_cfg = load_alerts_config()
    alert_mgr = AlertManager.from_config(alerts_cfg) if alerts_cfg.get("enabled", True) else None
    voice_agent = VoiceAgent(name=args.name, speak=not args.no_voice,
                             model=args.voice_model)
    if args.listen:
        from audio.stt import Listener
        voice_agent.listener = Listener(model_size=args.whisper_model,
                                        speaker=voice_agent.speaker)
        if voice_agent.listener.available:
            print("[agent] listener attached — the agent can hear replies")

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
        agent_snapshot = aggregator.agent_snapshot()

        # deterministic caregiver alerting (independent of the LLM)
        if alert_mgr is not None:
            alert_mgr.evaluate(snapshot)

        # conversational voice agent: greets, small-talks, raises salient things
        if force_greet["v"]:
            voice_agent.policy._last_spoken = 0.0   # let 'g' force a line now
            force_greet["v"] = False
        utterance = voice_agent.tick(agent_snapshot)
        if utterance:
            last_greeting["text"] = utterance
            if web is not None:
                web.publish(utterance)

        # mirror the full detections window to the /data web endpoint
        if web is not None:
            web.publish_data(snapshot, ctx.fps, last_greeting["text"],
                             reasoning=voice_agent.reasoning_card())

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
                panel = dashboard.render(snapshot, ctx.fps, last_greeting["text"],
                                         reasoning=voice_agent.reasoning_card())
                cv2.imshow("Detections — Data", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord("g"):
                force_greet["v"] = True
            if key == ord("t"):
                print("[main] tremor test requested ('t')")
                voice_agent.request_test()
            if key == ord("c") and primary_source != alt_source:
                nxt = alt_source if cam_state["current"] == primary_source else primary_source
                print(f"[camera] switching -> {nxt}")
                pipeline.camera.switch_to(nxt, camera_opts)
                cam_state["current"] = nxt
        return True

    print("[main] starting; press 'q' to quit, 'g' to greet, 't' for a tremor "
          "test, 'c' to switch camera (Ctrl+C in headless).")
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
