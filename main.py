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
import threading
import time
import traceback
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
from webui.debug_server import DebugServer, build_debug_payload
from core.capabilities import CapabilityRegistry, CapabilityStatus
from core.workflows import WorkflowEngine
from storage.event_store import EventStore
from sensors import SensorManager
from core.shared_signals import SharedSignals
from core.events import PersistencePolicy, Result, Severity

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


def build_pipeline(source, config, camera_opts=None, max_staleness: float = 0.25,
                   analysis_width: int = 960):
    """Discover modules and assemble the full processing pipeline."""
    discover("modules")
    modules = build_enabled(config)
    registry = CapabilityRegistry.instance()
    for module in modules:
        available = getattr(module, "available", True)
        registry.set(module.name, "model", CapabilityStatus.READY if available
                     else CapabilityStatus.UNAVAILABLE,
                     "enabled" if available else "optional dependency, model, credential, or consent unavailable")
    print(f"[main] enabled modules: {', '.join(m.name for m in modules)}")
    extractors = [FaceExtractor(input_width=analysis_width),
                  PoseExtractor(input_width=analysis_width), MotionExtractor()]
    scheduler = Scheduler(modules)
    aggregator = Aggregator()
    advisor_engine = AdvisorEngine.from_config(config.get("advice"))
    camera = make_camera(source, camera_opts or {})
    pipeline = Pipeline(camera, extractors, scheduler, aggregator, advisor_engine,
                        max_staleness=max_staleness,
                        showcase_gate=ShowcaseGate.from_config(config),
                        camera_location=(config.get("camera") or {}).get("location"),
                        tracking_enabled=bool((config.get("tracking") or {}).get("enabled", False)))
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
    ap.add_argument("--analysis-width", type=int, default=960,
                    help="maximum width passed to MediaPipe; coordinates and rPPG crops "
                         "remain at capture resolution (default 960)")
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
    ap.add_argument("--no-gemini", action="store_true",
                    help="start with Gemini API calls disabled; press M to toggle")
    ap.add_argument("--enable-cloud-skin", action="store_true",
                    help="consent to upload sampled camera stills to the configured "
                         "NVIDIA skin-screening model for this run")
    ap.add_argument("--enable-cloud-scene", action="store_true",
                    help="separate consent to upload infrequent room stills to NVIDIA")
    ap.add_argument("--assessment", choices=["sit_to_stand", "timed_up_and_go",
                    "arm_drift", "finger_tapping", "balance", "guided_gait",
                    "facial_movement", "read_aloud", "guided_breathing"],
                    help="start one guided assessment when the run begins")
    ap.add_argument("--enable-multi-person", action="store_true",
                    help="enable short-lived anonymous tracks with a stable primary subject")
    ap.add_argument("--webui", action="store_true",
                    help="serve the agent's text on a web page for an iPad "
                         "(prints the URL to open)")
    ap.add_argument("--webui-port", type=int, default=8770,
                    help="port for the companion web display (default 8770)")
    ap.add_argument("--debug-endpoint", action="store_true",
                    help="serve private raw diagnostics on localhost only")
    ap.add_argument("--debug-port", type=int, default=8771,
                    help="localhost private diagnostics port (default 8771)")
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
    scene_cfg = config.get("modules", {}).get("scene_vision", {})
    scene_cfg["consent"] = bool(args.enable_cloud_scene)
    if args.enable_multi_person:
        config.setdefault("tracking", {})["enabled"] = True
        config.get("modules", {}).setdefault("multi_person", {})["enabled"] = True
    if args.no_deepface:
        emotion = config.get("modules", {}).get("emotion", {})
        emotion["backends"] = [b for b in emotion.get("backends", []) if b != "deepface"]
    pipeline, aggregator = build_pipeline(args.source, config, camera_opts,
                                          max_staleness=args.vitals_max_staleness,
                                          analysis_width=args.analysis_width)
    print(f"[main] MediaPipe analysis width <= {args.analysis_width}px; "
          "rPPG uses original capture pixels")
    is_replay = str(args.source).startswith("replay:")
    # Runtime camera toggle ('c'): start on --source, swap to --alt-source and
    # back. Both use the same camera_opts so the laptop view is unchanged.
    primary_source, alt_source = args.source, args.alt_source
    cam_state = {"current": primary_source}

    alerts_cfg = load_alerts_config()
    alert_mgr = AlertManager.from_config(alerts_cfg) if alerts_cfg.get("enabled", True) else None
    voice_agent = VoiceAgent(name=args.name, speak=not args.no_voice,
                             model=args.voice_model,
                             gemini_enabled=not args.no_gemini)
    capabilities = CapabilityRegistry.instance()
    capabilities.set("camera", "hardware", CapabilityStatus.READY,
                     "replay" if str(args.source).startswith("replay:") else "live source")
    if capabilities.get("nvidia_skin") is None:
        capabilities.set("nvidia_skin", "cloud", CapabilityStatus.UNAVAILABLE,
                         "skin module is not enabled")
    sensor_manager = SensorManager.from_config(config.get("sensors"), replay=is_replay)
    shared_signals = SharedSignals.instance()
    sound_detector = None
    replay_audio = None
    if args.listen or is_replay:
        from audio.bus import AudioBus
        from audio.intelligence import SoundEventDetector
        audio_bus = AudioBus()
        if args.listen:
            from audio.stt import Listener
            voice_agent.listener = Listener(model_size=args.whisper_model,
                                            speaker=voice_agent.speaker, audio_bus=audio_bus)
        else:
            from audio.replay import ReplayAudioProducer, ReplayListener
            voice_agent.listener = ReplayListener()
            replay_audio = ReplayAudioProducer(audio_bus)
        sound_detector = SoundEventDetector(audio_bus)
        if voice_agent.listener.available:
            capabilities.set("microphone", "hardware", CapabilityStatus.READY,
                             "replay" if is_replay else "shared audio bus")
            shared_signals.set("microphone_ready", True)
            print("[agent] listener attached — the agent can hear replies")

    else:
        capabilities.set("microphone", "hardware", CapabilityStatus.UNAVAILABLE, "disabled")
        shared_signals.set("microphone_ready", False)
    if args.assessment:
        WorkflowEngine.instance().start(args.assessment)

    web = None
    if args.webui:
        control = pipeline.camera.replay_control if is_replay else None
        primary_handler = pipeline.tracker.set_primary if args.enable_multi_person else None
        web = CompanionServer(port=args.webui_port, control_handler=control,
                              primary_handler=primary_handler)
        web.start()

    display = not args.headless
    cv2 = None
    if display:
        import cv2 as _cv2
        cv2 = _cv2
        from output import overlay, dashboard

    source_text = str(args.source).strip().lower()
    decoupled_display = bool(display and not is_replay and
                             (source_text.isdigit() or source_text in ("realsense", "rs", "d435i")))

    force_greet = {"v": False}
    last_alert_print: dict[str, float] = {}
    last_vitals_print = {"t": 0.0}
    last_greeting = {"text": None}
    analysis_state_lock = threading.Lock()
    analysis_state = {"ctx": None, "snapshot": [], "reasoning": None}

    def system_snapshot():
        return {
            "timeline": EventStore.instance().recent(60),
            "capabilities": capabilities.snapshot(),
            "workflows": WorkflowEngine.instance().snapshot(),
            "consent": {"cloud_skin": args.enable_cloud_skin,
                        "cloud_scene": args.enable_cloud_scene},
            "replay": pipeline.camera.replay_status(),
            "gemini": voice_agent.gemini_status(),
        }

    debug_server = None
    if args.debug_endpoint:
        def debug_provider():
            with analysis_state_lock:
                reasoning = analysis_state["reasoning"]
            system = system_snapshot()
            system["reasoning"] = reasoning
            return build_debug_payload(aggregator.agent_snapshot(),
                                       pipeline.runtime_metrics.snapshot(), system)
        debug_server = DebugServer(debug_provider, port=args.debug_port)
        debug_server.start()

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

    def toggle_gemini() -> None:
        status = voice_agent.toggle_gemini()
        state = "ON" if status["enabled"] else "OFF"
        note = "" if status["available"] else " (API unavailable; templates remain active)"
        print(f"[agent/gemini] Gemini {state}{note}")

    def on_frame(ctx, results):
        pipeline.runtime_metrics.note_capture(ctx.fps, ctx.timestamp,
                                              pipeline.camera.diagnostics())
        WorkflowEngine.instance().tick(ctx.timestamp)
        sensor_manager.feed_context(ctx)
        feed_listener = getattr(voice_agent.listener, "feed_context", None)
        if feed_listener is not None:
            feed_listener(ctx)
        if replay_audio is not None:
            replay_audio.feed_context(ctx)
        auxiliary = sensor_manager.poll(ctx.timestamp)
        if sound_detector is not None:
            auxiliary.extend(sound_detector.pop_results())
        if voice_agent.listener is not None:
            metrics = voice_agent.listener.pop_metrics()
            if metrics:
                ctx.extras["speech_metrics"] = metrics[-1]
                shared_signals.set("speech_metrics", metrics[-1], ctx.timestamp)
                for metric in metrics:
                    auxiliary.append(Result("speech_timing", "turn", metric,
                        float(metric.get("quality", .5)), Severity.INFO,
                        "Speech timing summary captured", ttl=30, source="microphone",
                        quality=float(metric.get("quality", .5)),
                        persistence=PersistencePolicy.EVENT))
        if auxiliary:
            aggregator.ingest(auxiliary)
            for result in auxiliary:
                EventStore.instance().record_result(result)
        snapshot = aggregator.snapshot()
        agent_snapshot = aggregator.agent_snapshot()

        # deterministic caregiver alerting (independent of the LLM)
        if alert_mgr is not None:
            alert_mgr.evaluate(snapshot)

        # conversational voice agent: greets, small-talks, raises salient things
        if force_greet["v"]:
            voice_agent.policy._last_spoken = 0.0   # let 'g' force a line now
            force_greet["v"] = False
        utterance = voice_agent.tick(agent_snapshot, now=ctx.timestamp)
        safety_results = (voice_agent.pop_safety_results()
                          + voice_agent.pop_conversation_results())
        if safety_results:
            aggregator.ingest(safety_results)
            for result in safety_results:
                EventStore.instance().record_result(result)
            if alert_mgr is not None:
                alert_mgr.evaluate(aggregator.snapshot())
        if utterance:
            last_greeting["text"] = utterance
            if web is not None:
                web.publish(utterance)

        # mirror the full detections window to the /data web endpoint
        if web is not None:
            performance = pipeline.runtime_metrics.snapshot()
            web.publish_data(snapshot, ctx.fps, last_greeting["text"],
                             reasoning=voice_agent.reasoning_card(),
                             system=system_snapshot(), performance=performance)

        print_vitals(snapshot)

        with analysis_state_lock:
            analysis_state["ctx"] = ctx
            analysis_state["snapshot"] = snapshot
            analysis_state["reasoning"] = voice_agent.reasoning_card()

        if display and not decoupled_display:
            if args.combined:
                frame = overlay.draw(ctx.frame.copy(), ctx, snapshot, ctx.fps)
                cv2.imshow("Humanoid Camera — detections", frame)
            else:
                # camera window: video + boxes only
                cam = overlay.draw_boxes(ctx.frame.copy(), ctx, ctx.fps)
                cv2.imshow("Camera", cam)
                # separate, readable data window
                panel = dashboard.render(snapshot, ctx.fps, last_greeting["text"],
                                         reasoning=voice_agent.reasoning_card(),
                                         performance=pipeline.runtime_metrics.snapshot(),
                                         gemini=voice_agent.gemini_status())
                cv2.imshow("Detections — Data", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord("g"):
                force_greet["v"] = True
            if key == ord("m"):
                toggle_gemini()
            if key == ord("t"):
                print("[main] tremor test requested ('t')")
                voice_agent.request_test()
            if key == ord("c") and primary_source != alt_source:
                nxt = alt_source if cam_state["current"] == primary_source else primary_source
                print(f"[camera] switching -> {nxt}")
                pipeline.camera.switch_to(nxt, camera_opts)
                cam_state["current"] = nxt
        return True

    print("[main] starting; press 'q' to quit, 'g' to greet, 'm' to toggle "
          "Gemini, 't' for a tremor test, 'c' to switch camera "
          "(Ctrl+C in headless).")
    try:
        if not decoupled_display:
            pipeline.run(on_frame=on_frame, max_frames=args.max_frames)
        else:
            worker_error = []

            def analysis_worker():
                try:
                    pipeline.run(on_frame=on_frame, max_frames=args.max_frames)
                except BaseException as exc:  # propagate to the GUI/main thread
                    worker_error.append((exc, traceback.format_exc()))

            worker = threading.Thread(target=analysis_worker, daemon=True,
                                      name="analysis-loop")
            worker.start()
            last_seq = -1
            last_panel_at = 0.0
            while worker.is_alive():
                packet = pipeline.camera.latest_frame()
                if packet is not None:
                    seq, raw_frame, captured_at = packet
                    pipeline.runtime_metrics.note_capture(
                        pipeline.camera.current_fps, captured_at,
                        pipeline.camera.diagnostics())
                    if seq != last_seq:
                        last_seq = seq
                        with analysis_state_lock:
                            analyzed_ctx = analysis_state["ctx"]
                            snapshot = list(analysis_state["snapshot"])
                            reasoning = analysis_state["reasoning"]
                        performance = pipeline.runtime_metrics.snapshot()
                        frame = raw_frame.copy()
                        if args.combined:
                            frame = overlay.draw(frame, analyzed_ctx, snapshot,
                                                 performance["capture_fps"]) \
                                if analyzed_ctx is not None else frame
                            cv2.imshow("Humanoid Camera — detections", frame)
                        else:
                            if analyzed_ctx is not None:
                                frame = overlay.draw_boxes(frame, analyzed_ctx,
                                                           performance["capture_fps"],
                                                           performance)
                            cv2.imshow("Camera", frame)
                            now = time.time()
                            if now - last_panel_at >= 0.5:
                                panel = dashboard.render(
                                    snapshot, performance["capture_fps"],
                                    last_greeting["text"], reasoning=reasoning,
                                    performance=performance,
                                    gemini=voice_agent.gemini_status())
                                cv2.imshow("Detections — Data", panel)
                                last_panel_at = now
                        pipeline.runtime_metrics.note_preview()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    pipeline.request_stop()
                    break
                if key == ord("g"):
                    force_greet["v"] = True
                if key == ord("m"):
                    toggle_gemini()
                if key == ord("t"):
                    print("[main] tremor test requested ('t')")
                    voice_agent.request_test()
                if key == ord("c") and primary_source != alt_source:
                    nxt = alt_source if cam_state["current"] == primary_source else primary_source
                    print(f"[camera] switching -> {nxt}")
                    with analysis_state_lock:
                        analysis_state["ctx"] = None
                    pipeline.camera.switch_to(nxt, camera_opts)
                    cam_state["current"] = nxt
                if packet is None:
                    time.sleep(0.002)
            pipeline.request_stop()
            worker.join(timeout=5.0)
            if worker_error:
                exc, tb = worker_error[0]
                print(tb, end="")
                raise exc
    except KeyboardInterrupt:
        print("\n[main] stopped.")
    finally:
        voice_agent.close()
        sensor_manager.close()
        if sound_detector is not None:
            sound_detector.close()
        if debug_server is not None:
            debug_server.stop()
        if display and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
