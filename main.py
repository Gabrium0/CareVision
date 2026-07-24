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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from core.runtime_resources import (apply_loaded_limits, configure_environment,
                                    diagnostics as runtime_resource_diagnostics)

configure_environment("maximum")

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
from webui.debug_server import (DebugServer, build_audio_debug_state,
                                build_debug_payload)
from core.capabilities import CapabilityRegistry, CapabilityStatus
from core.workflows import WorkflowEngine
from storage.event_store import EventStore
from storage.history_store import HistoryStore
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
                   analysis_width: int = 960, face_analysis_width: int = 640,
                   fast_path_mode: str = "tracked",
                   quality_profile: str = "maximum"):
    """Discover modules and assemble the full processing pipeline."""
    discover("modules")
    modules = build_enabled(config)
    for module in modules:
        if module.name == "gesture":
            module.input_width = analysis_width
    registry = CapabilityRegistry.instance()
    for module in modules:
        lifecycle = getattr(module, "capability_status", None)
        if lifecycle is not None:
            status, detail = lifecycle()
        elif hasattr(module, "available") and not getattr(module, "available"):
            status, detail = CapabilityStatus.UNCONFIGURED, "optional capability is not configured"
        elif callable(getattr(module, "start", None)):
            status, detail = CapabilityStatus.LOADING, "model startup pending"
        else:
            status, detail = CapabilityStatus.READY, "enabled"
        registry.set(module.name, "model", status, detail)
    print(f"[main] enabled modules: {', '.join(m.name for m in modules)}")
    extractors = [FaceExtractor(input_width=face_analysis_width),
                  PoseExtractor(input_width=analysis_width), MotionExtractor()]
    scheduler = Scheduler(modules)
    aggregator = Aggregator()
    advisor_engine = AdvisorEngine.from_config(config.get("advice"))
    camera = make_camera(source, camera_opts or {})
    pipeline = Pipeline(camera, extractors, scheduler, aggregator, advisor_engine,
                        max_staleness=max_staleness,
                        showcase_gate=ShowcaseGate.from_config(config),
                        camera_location=(config.get("camera") or {}).get("location"),
                        tracking_enabled=bool((config.get("tracking") or {}).get("enabled", False)),
                        background_analysis=not str(source).startswith("replay:"),
                        fast_path_mode=fast_path_mode,
                        analysis_width=analysis_width,
                        quality_profile=quality_profile,
                        runtime_config=config.get("runtime"))
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
    ap.add_argument("--dev-mode", action="store_true",
                    help="disable heavyweight/network optional backends for local "
                         "debug replay (used by dev.py)")
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
    ap.add_argument("--vitals-fast-path-mode",
                    choices=("strict", "extended", "tracked"), default="tracked",
                    help="vitals ROI reuse mode: strict 250ms lease, extended 750ms "
                         "stationary lease, or guarded optical-flow tracking "
                         "(default tracked)")
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
    ap.add_argument("--face-analysis-width", type=int, default=640,
                    help="maximum width for authoritative face detection; rPPG still "
                         "uses original capture pixels (default 640)")
    ap.add_argument("--quality-profile", choices=("maximum", "balanced", "realtime"),
                    default="maximum",
                    help="analysis policy: maximum preserves high-detail ROIs and "
                         "reduces cadence first; balanced/realtime trade spatial "
                         "detail for latency (default maximum)")
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
    ap.add_argument("--type-input", action="store_true",
                    help="answer the agent by typing on the companion page "
                         "instead of speaking; loads no microphone or whisper "
                         "and takes precedence over --listen")
    ap.add_argument("--detect-cough", action="store_true",
                    help="enable local microphone cough-episode detection without "
                         "requiring speech recognition")
    ap.add_argument("--whisper-model", default="base",
                    help="faster-whisper model size for --listen (default base)")
    ap.add_argument("--voice-model", default="moondream3.1-9B-A2B",
                    help="Moondream model for the voice agent (key from .env)")
    ap.add_argument("--no-moondream", action="store_true",
                    help="start with Moondream API calls disabled; press M to toggle")
    ap.add_argument("--enable-cloud-skin", action="store_true",
                    help="consent to upload sampled camera stills to the configured "
                         "NVIDIA skin-screening model for this run")
    ap.add_argument("--enable-cloud-scene", action="store_true",
                    help="separate consent to upload infrequent room stills to NVIDIA")
    ap.add_argument("--assessment", choices=["sit_to_stand", "timed_up_and_go",
                    "arm_drift", "finger_tapping", "balance", "guided_gait",
                    "facial_movement", "read_aloud", "guided_breathing"],
                    help="start one guided assessment when the run begins")
    ap.add_argument("--demo", action="store_true",
                    help="start a short guided demo circuit (facial movement, "
                         "arm drift, balance) shortly after launch — press 'd' "
                         "to trigger it instead any time; good for a quick "
                         "live tour with a guest or client")
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
    ap.add_argument("--caregiver-portal", action="store_true",
                    help="serve the local caregiver review portal on loopback only")
    ap.add_argument("--caregiver-port", type=int, default=8772,
                    help="localhost caregiver portal port (default 8772)")
    args = ap.parse_args()
    apply_loaded_limits(args.quality_profile)
    if args.list_cameras:
        Camera.list_devices()
        return
    if args.debug_modules:
        os.environ["APP_DEBUG_MODULES"] = args.debug_modules
    load_env()   # make .env keys (X-Moondream-Auth, alert creds) available

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
    if args.dev_mode:
        modules = config.get("modules", {})
        for name in ("clothing", "weather"):
            modules.get(name, {})["enabled"] = False
        heart_rate = modules.get("heart_rate", {})
        heart_rate["backends"] = [backend for backend in heart_rate.get("backends", [])
                                  if backend != "openrppg"]
        emotion = modules.get("emotion", {})
        emotion["backends"] = [backend for backend in emotion.get("backends", [])
                               if backend != "deepface"]
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
                                          analysis_width=args.analysis_width,
                                          face_analysis_width=args.face_analysis_width,
                                          fast_path_mode=args.vitals_fast_path_mode,
                                          quality_profile=args.quality_profile)
    # Start FashionCLIP before audio/native workers compete for CPU and import
    # bandwidth. Its loader is asynchronous, so CLI and camera startup remain
    # responsive and Pipeline.run's repeated preload is idempotent.
    pipeline.preload_modules()
    print(f"[main] face analysis width <= {args.face_analysis_width}px; "
          f"background analysis width <= {args.analysis_width}px; "
          f"quality={args.quality_profile}; rPPG uses original capture pixels")
    is_replay = str(args.source).startswith("replay:")
    # Runtime camera toggle ('c'): start on --source, swap to --alt-source and
    # back. Both use the same camera_opts so the laptop view is unchanged.
    primary_source, alt_source = args.source, args.alt_source
    cam_state = {"current": primary_source}

    alerts_cfg = load_alerts_config()
    event_store = EventStore.instance()
    if args.caregiver_portal:
        event_store.interrupt_active_cases()
    alert_mgr = AlertManager.from_config(alerts_cfg) if alerts_cfg.get("enabled", True) else None
    if alert_mgr is not None and args.caregiver_portal:
        alert_mgr.case_store = event_store
    voice_agent = VoiceAgent(name=args.name, speak=not args.no_voice,
                             model=args.voice_model,
                             moondream_enabled=not args.no_moondream)
    capabilities = CapabilityRegistry.instance()
    capabilities.set("camera", "hardware", CapabilityStatus.READY,
                     "replay" if str(args.source).startswith("replay:") else "live source")
    if capabilities.get("nvidia_skin") is None:
        capabilities.set("nvidia_skin", "cloud", CapabilityStatus.UNCONFIGURED,
                         "skin module is not enabled")
    sensor_manager = SensorManager.from_config(config.get("sensors"), replay=is_replay)
    shared_signals = SharedSignals.instance()
    sound_detector = None
    microphone = None
    replay_audio = None
    if args.listen or args.detect_cough or is_replay:
        from audio.bus import AudioBus
        from audio.intelligence import SoundEventDetector
        audio_bus = AudioBus()
        if not is_replay:
            from audio.microphone import MicrophoneProducer
            microphone = MicrophoneProducer(audio_bus)
        if args.listen and not is_replay and not args.type_input:
            from audio.stt import Listener
            voice_agent.listener = Listener(model_size=args.whisper_model,
                                            speaker=voice_agent.speaker, audio_bus=audio_bus)
        elif is_replay:
            from audio.replay import ReplayAudioProducer, ReplayListener
            voice_agent.listener = ReplayListener()
            replay_audio = ReplayAudioProducer(audio_bus)
        allowed_events = {"cough"} if args.detect_cough and not args.listen else None
        sound_detector = SoundEventDetector(audio_bus, allowed_events=allowed_events)
        microphone_ready = is_replay or bool(microphone and microphone.available)
        if microphone_ready:
            if is_replay:
                capabilities.set("microphone", "hardware", CapabilityStatus.READY,
                                 "replay")
            shared_signals.set("microphone_ready", True)
            if voice_agent.listener is not None:
                print("[agent] listener attached — the agent can hear replies")
            if args.detect_cough:
                print("[audio] cough episode detection enabled")
        else:
            shared_signals.set("microphone_ready", False)

    else:
        capabilities.set("microphone", "hardware", CapabilityStatus.UNCONFIGURED, "disabled")
        shared_signals.set("microphone_ready", False)

    # A typed listener replaces any audio one: in a noisy room whisper invents
    # utterances, and a demo that answers phantom speech is worse than mute.
    typed_listener = None
    if args.type_input:
        from audio.typed import TypedListener
        typed_listener = TypedListener()
        voice_agent.listener = typed_listener
        print("[agent] typed input attached — answer from the companion page")

    if args.assessment:
        WorkflowEngine.instance().start(args.assessment)
    elif args.demo:
        voice_agent.start_demo_circuit()

    web = None
    web_publish_executor = None
    web_publish_state = {"future": None, "last": -1e9}
    if args.webui:
        control = pipeline.camera.replay_control if is_replay else None
        primary_handler = pipeline.tracker.set_primary if args.enable_multi_person else None

        def assessment_handler(action, protocol=None):
            """Web hook mirroring the 't'/'a'/'d' hotkeys so the big-screen
            /demo picker can start a guided assessment or the full circuit.
            Raises ValueError on an unknown protocol (rendered as HTTP 400)."""
            if action == "circuit":
                return {"action": "circuit",
                        "started": bool(voice_agent.start_demo_circuit())}
            from assessments import PROTOCOLS
            if not protocol or protocol not in PROTOCOLS:
                raise ValueError("unknown protocol")
            voice_agent.request_test(protocol)
            return {"action": "start", "protocol": protocol, "started": True}

        def say_handler(text):
            """Web hook feeding a typed reply into the same corroboration path
            as heard speech. Raises RuntimeError when the run has no typed
            listener, ValueError on empty text (both rendered as HTTP 400)."""
            if typed_listener is None:
                raise RuntimeError("typed input is not enabled (use --type-input)")
            if not typed_listener.push(text):
                raise ValueError("empty reply")
            return {"text": " ".join(str(text).split())[:400]}

        web = CompanionServer(port=args.webui_port, control_handler=control,
                              primary_handler=primary_handler,
                              assessment_handler=assessment_handler,
                              say_handler=say_handler)
        web.start()
        web_publish_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="web-publish")

    caregiver_server = None
    if args.caregiver_portal:
        from webui.caregiver_server import CaregiverServer
        caregiver_server = CaregiverServer(event_store, HistoryStore.instance(),
                                            port=args.caregiver_port)
        caregiver_server.start()

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

    def system_snapshot(private: bool = False, results=None, performance=None):
        system = {
            "timeline": EventStore.instance().recent(60),
            "capabilities": capabilities.snapshot(),
            "workflows": WorkflowEngine.instance().snapshot(),
            "consent": {"cloud_skin": args.enable_cloud_skin,
                        "cloud_scene": args.enable_cloud_scene},
            "replay": pipeline.camera.replay_status(),
            "moondream": voice_agent.moondream_status(),
            "modules_enabled": sorted(m.name for m in pipeline.scheduler.modules),
        }
        if private:
            audio_enabled = bool(args.listen or args.detect_cough or is_replay)
            audio_mode = ("replay" if is_replay else
                          "cough_only" if args.detect_cough and not args.listen else
                          "broad_listening")
            system["audio"] = build_audio_debug_state(
                capabilities, sound_detector, audio_enabled, audio_mode,
                voice_agent.listener)
            system["vitals"] = pipeline.vitals_diagnostics(
                list(results or []), now=time.time(), performance=performance)
            skin_vision = next((module for module in pipeline.scheduler.modules
                                if module.name == "skin_vision"), None)
            system["nvidia_skin"] = (
                skin_vision.diagnostics()
                if skin_vision is not None and hasattr(skin_vision, "diagnostics")
                else {"available": False, "status": "unavailable",
                      "last_attempt": None}
            )
            system["history_writer"] = HistoryStore.instance().diagnostics()
            system["runtime_resources"] = runtime_resource_diagnostics()
            # Research telemetry: the flagged->asked->confirmed/denied funnel for
            # the visual-prior -> gentle-question corroboration loop.
            system["corroboration"] = voice_agent.corroboration.funnel()
            system["model_workers"] = {
                module.name: module.diagnostics()
                for module in pipeline.scheduler.modules
                if module.name == "clothing" and hasattr(module, "diagnostics")}
            system["clothing"] = system["model_workers"].get(
                "clothing", {"status": "unconfigured", "ready": False})
        return system

    debug_server = None
    if args.debug_endpoint:
        def debug_provider():
            with analysis_state_lock:
                reasoning = analysis_state["reasoning"]
            results = aggregator.agent_snapshot()
            performance = pipeline.runtime_metrics.snapshot()
            system = system_snapshot(private=True, results=results,
                                     performance=performance)
            system["reasoning"] = reasoning
            return build_debug_payload(results, performance, system)
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

    def toggle_moondream() -> None:
        status = voice_agent.toggle_moondream()
        state = "ON" if status["enabled"] else "OFF"
        note = "" if status["available"] else " (API unavailable; templates remain active)"
        print(f"[agent/moondream] Moondream {state}{note}")

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
                event_store.record_result(result)
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
                event_store.record_result(result)
            if alert_mgr is not None:
                alert_mgr.evaluate(aggregator.snapshot())
        if utterance:
            last_greeting["text"] = utterance
            if web is not None:
                web.publish(utterance)

        # mirror the full detections window to the /data web endpoint
        if web is not None and web_publish_executor is not None:
            pending: Future | None = web_publish_state["future"]
            if pending is not None and pending.done():
                try:
                    pending.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[webui] data publication failed ({type(exc).__name__})")
                web_publish_state["future"] = None
                pending = None
            if pending is None and ctx.timestamp - web_publish_state["last"] >= 0.5:
                stable_snapshot = list(snapshot)
                greeting = last_greeting["text"]
                reasoning = voice_agent.reasoning_card()
                performance = pipeline.runtime_metrics.snapshot()
                def publish_data_snapshot():
                    web.publish_data(
                        stable_snapshot, ctx.fps, greeting, reasoning=reasoning,
                        system=system_snapshot(), performance=performance)
                web_publish_state["future"] = web_publish_executor.submit(
                    publish_data_snapshot)
                web_publish_state["last"] = ctx.timestamp

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
                                         moondream=voice_agent.moondream_status())
                cv2.imshow("Detections — Data", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord("g"):
                force_greet["v"] = True
            if key == ord("m"):
                toggle_moondream()
            if key == ord("t"):
                print("[main] tremor test requested ('t')")
                voice_agent.request_test()
            if key == ord("a"):
                print("[main] arm skin check requested ('a')")
                voice_agent.request_test("arm_check")
            if key == ord("d"):
                if voice_agent.start_demo_circuit():
                    print("[main] guest/client demo circuit requested ('d')")
            if key == ord("c") and primary_source != alt_source:
                nxt = alt_source if cam_state["current"] == primary_source else primary_source
                print(f"[camera] switching -> {nxt}")
                pipeline.reset_capture_state()
                pipeline.camera.switch_to(nxt, camera_opts)
                cam_state["current"] = nxt
        return True

    print("[main] starting; press 'q' to quit, 'g' to greet, 'm' to toggle "
          "Moondream, 't' for a tremor test, 'a' for an arm skin check, "
          "'d' for a guest/client demo circuit, 'c' to switch camera "
          "(Ctrl+C in headless).")
    worker = None
    interrupted = False
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
                                    moondream=voice_agent.moondream_status())
                                cv2.imshow("Detections — Data", panel)
                                last_panel_at = now
                        pipeline.runtime_metrics.note_preview(
                            overlay_timestamp=(analyzed_ctx.timestamp
                                               if analyzed_ctx is not None else None))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    pipeline.request_stop()
                    break
                if key == ord("g"):
                    force_greet["v"] = True
                if key == ord("m"):
                    toggle_moondream()
                if key == ord("t"):
                    print("[main] tremor test requested ('t')")
                    voice_agent.request_test()
                if key == ord("a"):
                    print("[main] arm skin check requested ('a')")
                    voice_agent.request_test("arm_check")
                if key == ord("d"):
                    if voice_agent.start_demo_circuit():
                        print("[main] guest/client demo circuit requested ('d')")
                if key == ord("c") and primary_source != alt_source:
                    nxt = alt_source if cam_state["current"] == primary_source else primary_source
                    print(f"[camera] switching -> {nxt}")
                    with analysis_state_lock:
                        analysis_state["ctx"] = None
                    pipeline.reset_capture_state()
                    pipeline.camera.switch_to(nxt, camera_opts)
                    cam_state["current"] = nxt
                if packet is None:
                    time.sleep(0.002)
            pipeline.request_stop()
            worker.join()
            if worker_error:
                exc, tb = worker_error[0]
                print(tb, end="")
                raise exc
    except KeyboardInterrupt:
        pipeline.request_stop()
        if worker is not None and worker.is_alive():
            worker.join()
        interrupted = True
    finally:
        if web_publish_executor is not None:
            pending = web_publish_state.get("future")
            if pending is not None:
                pending.cancel()
            web_publish_executor.shutdown(wait=False, cancel_futures=True)
        voice_agent.close()
        sensor_manager.close()
        if sound_detector is not None:
            sound_detector.close()
        if microphone is not None:
            microphone.close()
        if debug_server is not None:
            debug_server.stop()
        if caregiver_server is not None:
            caregiver_server.stop()
        event_store.flush()
        if display and cv2 is not None:
            cv2.destroyAllWindows()
    if interrupted:
        print("\n[main] stopped.")


if __name__ == "__main__":
    main()
