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
import math
import os
import secrets
import sys
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
from core.ipad_camera import is_ipad_source
from core.module_gate import ModuleGate, seed_start_paused
from core.pipeline import Pipeline
from core.showcase import ShowcaseGate
from core.registry import discover, build_enabled, all_registered
from core.scheduler import Scheduler
from core.subjects import SubjectModulePool
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

_IPAD_CAPTURE_FPS_BOUNDS = (1.0, 30.0)
_IPAD_CAPTURE_DIMENSION_BOUNDS = (160, 4096)


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


def _build_ipad_capture_config(request_fps: float,
                               request_size: tuple[int, int] | None) -> dict:
    """Build the bounded camera request sent to the paired browser.

    ``None`` keeps resolution selection adaptive on the iPad.  An explicit
    size is fixed so ``--resolution WxH`` has the same meaning at capture as
    it does for the laptop backends.
    """
    min_fps, max_fps = _IPAD_CAPTURE_FPS_BOUNDS
    fps = float(request_fps)
    if not math.isfinite(fps) or not min_fps <= fps <= max_fps:
        raise ValueError(
            f"--ipad-fps must be between {min_fps:g} and {max_fps:g}, got "
            f"{request_fps!r}")
    if request_size is None:
        return {"fps": fps, "resolution": {"mode": "auto"}}

    min_dimension, max_dimension = _IPAD_CAPTURE_DIMENSION_BOUNDS
    width, height = (int(request_size[0]), int(request_size[1]))
    if not (min_dimension <= width <= max_dimension
            and min_dimension <= height <= max_dimension):
        raise ValueError(
            "iPad --resolution dimensions must each be between "
            f"{min_dimension} and {max_dimension}, got {width}x{height}")
    return {"fps": fps,
            "resolution": {"mode": "fixed", "width": width, "height": height}}


def _uses_decoupled_display(display: bool, *sources) -> bool:
    """Whether this run can use non-consuming preview for either live source."""
    def supports_latest_frame(source) -> bool:
        source_text = str(source).strip().lower()
        return (source_text.isdigit()
                or source_text in ("realsense", "rs", "d435i")
                or is_ipad_source(source))

    return bool(display
                and any(supports_latest_frame(source) for source in sources))


def build_pipeline(source, config, camera_opts=None, max_staleness: float = 0.25,
                   analysis_width: int = 960, face_analysis_width: int = 640,
                   fast_path_mode: str = "tracked",
                   quality_profile: str = "maximum",
                   start_blank: bool = False):
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
    tracking_cfg = config.get("tracking") or {}
    tracking_enabled = bool(tracking_cfg.get("enabled", False))
    max_subjects = int(tracking_cfg.get("max_subjects", 3)) if tracking_enabled else 2
    extractors = [FaceExtractor(input_width=face_analysis_width, max_subjects=max_subjects),
                  PoseExtractor(input_width=analysis_width, max_subjects=max_subjects),
                  MotionExtractor()]
    # Runtime pause/resume overlay (core/module_gate.py): by default starts
    # mirroring what config/modules.yaml already enabled, so a fresh run
    # behaves identically to before toggling existed -- alerts work from
    # frame one. --start-blank seeds it empty instead (camera + person
    # outline only, everything else off until enabled from /modules); it is
    # a demo mode, not for unattended/production use, since no module -- and
    # therefore no alert -- runs until manually toggled on.
    # `runtime.start_paused` picks the middle ground: listed modules boot
    # warm but gated off (heavy detectors stay loaded yet never run) so a
    # showcase starts lean without losing one-command recovery from the
    # /modules console. The config file only chooses the starting point --
    # from here on, toggle state belongs to the running process alone.
    secondary_names = [name for name in tracking_cfg.get("secondary_modules", [])
                       if name in all_registered()]
    enabled_names = {m.name for m in modules}
    module_gate = ModuleGate(
        primary_enabled=(set() if start_blank else set(enabled_names)),
        secondary_enabled=(set() if start_blank else set(secondary_names)))
    paused_cfg = (config.get("runtime") or {}).get("start_paused") or []
    applied, unknown = seed_start_paused(module_gate, paused_cfg, enabled_names)
    if unknown:
        print(f"[main] ignoring unknown start_paused names: {', '.join(unknown)}")
    if applied:
        print(f"[main] start-paused (toggle back via /modules): {', '.join(applied)}")
    subject_pool = (SubjectModulePool(secondary_names, config.get("modules", {}),
                                      gate=module_gate)
                    if tracking_enabled and secondary_names else None)
    scheduler = Scheduler(modules)
    aggregator = Aggregator()
    advisor_engine = AdvisorEngine.from_config(config.get("advice"))
    camera = make_camera(source, camera_opts or {})
    pipeline = Pipeline(camera, extractors, scheduler, aggregator, advisor_engine,
                        max_staleness=max_staleness,
                        showcase_gate=ShowcaseGate.from_config(config),
                        camera_location=(config.get("camera") or {}).get("location"),
                        tracking_enabled=tracking_enabled,
                        background_analysis=not str(source).startswith("replay:"),
                        fast_path_mode=fast_path_mode,
                        analysis_width=analysis_width,
                        quality_profile=quality_profile,
                        runtime_config=config.get("runtime"),
                        module_gate=module_gate,
                        subject_pool=subject_pool,
                        max_subjects=max_subjects)
    return pipeline, aggregator


def main():
    """Parse CLI args and run the live detection pipeline."""
    # A redirected or piped stdout on Windows is a cp1252 stream, and one
    # emoji in any printed line (agent speech does this) would raise
    # UnicodeEncodeError and kill the process mid-run. Reconfigure first so
    # logging can never be fatal; replace, don't fail, on stray characters.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="camera index, file path, URL, 'realsense' for a "
                         "RealSense D435i (depth + IMU; needs pyrealsense2), or "
                         "'ipad' to take frames from an iPad browser over WebRTC "
                         "(needs requirements-ipad.txt and a signaling relay)")
    ap.add_argument("--alt-source", default="0",
                    help="the other camera to toggle to with 'c' (default 0 = "
                         "laptop cam; 'realsense' and 'ipad' also work here)")
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
                    help="'auto' probes local-camera resolutions (and lets an iPad "
                         "adapt capture size) by default; an explicit WxH such as "
                         "1280x720 skips probing and fixes the iPad capture size")
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
    ap.add_argument("--tts", choices=("auto", "piper", "pyttsx3"), default="auto",
                    help="spoken-voice engine: piper is the local neural voice "
                         "(python -m audio.tts_piper --download), pyttsx3 is the "
                         "system voice, auto prefers piper and falls back "
                         "(default auto)")
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
    ap.add_argument("--voice-model", default="moondream/moondream3-preview",
                    help="Moondream model for the voice agent (key from .env; "
                         "override via MOONDREAM_MODEL env var)")
    ap.add_argument("--no-moondream", action="store_true",
                    help="start with Moondream API calls disabled; press M to toggle")
    ap.add_argument("--enable-agent-vision", action="store_true",
                    help="consent to periodic bounded camera frames being sent to "
                         "Moondream while a person is present and conversation is active")
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
    ap.add_argument("--showcase", action="store_true",
                    help="start the full narrated tour: a spoken introduction, "
                         "the guided demo circuit, and a closing wrap-up "
                         "(press 's' to trigger instead)")
    ap.add_argument("--enable-multi-person", action="store_true",
                    help="enable short-lived anonymous tracks with a stable primary subject")
    ap.add_argument("--webui", action="store_true",
                    help="serve the agent's text on a web page for an iPad "
                         "(prints the URL to open)")
    ap.add_argument("--webui-port", type=int, default=8770,
                    help="port for the companion web display (default 8770)")
    ap.add_argument("--allow-remote-toggle", action="store_true",
                    help="allow non-loopback clients to pause/resume detectors "
                         "from /modules (default: loopback only)")
    ap.add_argument("--start-blank", action="store_true",
                    help="start with every detector paused (camera + person "
                         "outline only); enable them from /modules as you go. "
                         "For live demos -- NOT for unattended/production use, "
                         "since no alert can fire until manually enabled")
    ap.add_argument("--debug-endpoint", action="store_true",
                    help="serve private raw diagnostics on localhost only")
    ap.add_argument("--debug-port", type=int, default=8771,
                    help="localhost private diagnostics port (default 8771)")
    ap.add_argument("--caregiver-portal", action="store_true",
                    help="serve the local caregiver review portal on loopback only")
    ap.add_argument("--caregiver-port", type=int, default=8772,
                    help="localhost caregiver portal port (default 8772)")
    ap.add_argument("--ipad-relay-url", default=None,
                    help="https URL of the signaling relay for --source ipad "
                         "(default: $IPAD_RELAY_URL). The relay only carries "
                         "SDP/ICE; video goes peer-to-peer")
    ap.add_argument("--ipad-room", default=None,
                    help="relay room name for --source ipad (default: $IPAD_ROOM); "
                         "stable so the iPad can keep a bookmark")
    ap.add_argument("--ipad-fps", type=float, default=20.0,
                    help="frame rate to request from the iPad camera, 1-30 "
                         "(default 20; "
                         "below ~20 the rPPG quality score is scaled down)")
    ap.add_argument("--ipad-transport", default="datachannel",
                    choices=("datachannel", "video"),
                    help="datachannel = JPEG frames (no temporal compression, "
                         "better for rPPG); video is reserved but not implemented")
    ap.add_argument("--ipad-stun", default="",
                    help="comma-separated STUN URLs; unused on a laptop hotspot "
                         "where ICE settles on host candidates")
    ap.add_argument("--no-ipad-toggle", action="store_true",
                    help="refuse module enable/disable from the paired iPad "
                         "(default: allowed, since control is the point)")
    args = ap.parse_args()
    if args.ipad_transport == "video":
        ap.error("--ipad-transport video is not implemented; use "
                 "--ipad-transport datachannel")
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
        if rw <= 0 or rh <= 0:
            ap.error("--resolution dimensions must be positive, got "
                     f"{rw}x{rh}")
        request_size = (rw, rh)
    camera_opts = {"lock": not args.no_lock, "exposure": args.exposure,
                   "request_fps": args.fps, "request_size": request_size,
                   "target_width": request_size[0],
                   "target_brightness": args.target_brightness,
                   "allow_gain_boost": not args.no_gain_boost,
                   "auto_resolution": auto_resolution,
                   "min_fps": args.min_fps}

    # An iPad source pairs over a signaling relay. The pairing code is generated
    # per run and printed once; the shared secret only ever comes from .env, so
    # it never lands in shell history or the process list.
    ipad_source = is_ipad_source(args.source) or is_ipad_source(args.alt_source)
    ipad_code = None
    ipad_capture_config = None
    if ipad_source:
        try:
            ipad_capture_config = _build_ipad_capture_config(
                args.ipad_fps, None if auto_resolution else request_size)
        except ValueError as exc:
            ap.error(str(exc))
        # A fresh random code per run is the secure default. IPAD_PAIR_CODE in
        # .env pins it instead, so restarting the pipeline reuses the same code
        # and an already-paired iPad reconnects on its own — a dev convenience,
        # not for production, since a stable code is easier to guess.
        fixed_code = os.environ.get("IPAD_PAIR_CODE", "").strip()
        if fixed_code:
            if not (fixed_code.isdigit() and len(fixed_code) == 6):
                ap.error("IPAD_PAIR_CODE must be exactly 6 digits")
            ipad_code = fixed_code
        else:
            ipad_code = f"{secrets.randbelow(1_000_000):06d}"
        ipad_relay = args.ipad_relay_url or os.environ.get("IPAD_RELAY_URL")
        ipad_room = args.ipad_room or os.environ.get("IPAD_ROOM")
        ipad_secret = os.environ.get("RELAY_SECRET")
        missing = [name for name, value in (("IPAD_RELAY_URL", ipad_relay),
                                            ("IPAD_ROOM", ipad_room),
                                            ("RELAY_SECRET", ipad_secret))
                   if not value]
        if missing:
            ap.error(f"--source ipad needs {', '.join(missing)}; set them in .env "
                     f"(or pass --ipad-relay-url/--ipad-room)")
        camera_opts.update({
            "request_fps": args.ipad_fps,
            "ipad_relay_url": ipad_relay, "ipad_room": ipad_room,
            "ipad_secret": ipad_secret, "ipad_code": ipad_code,
            "ipad_stun": tuple(s.strip() for s in args.ipad_stun.split(",") if s.strip()),
            "ipad_transport": args.ipad_transport})
        print(f"\n[ipad] open  {ipad_relay.rstrip('/')}/r/{ipad_room}"
              f"\n[ipad] pairing code: {ipad_code}"
              f"\n[ipad] start the laptop first — the relay's free tier can take "
              f"~60s to wake, and whoever connects first waits for it\n")

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
                                          quality_profile=args.quality_profile,
                                          start_blank=args.start_blank)
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
                             moondream_enabled=not args.no_moondream,
                             vision_enabled=args.enable_agent_vision,
                             tts_engine=args.tts,
                             conversation=config.get("conversation"))
    capabilities = CapabilityRegistry.instance()
    voice_agent.set_context_sources(capabilities=capabilities,
                                    history=HistoryStore.instance())
    tts_state = voice_agent.speaker.status()
    capabilities.set(
        "speech_output", "voice",
        CapabilityStatus.UNCONFIGURED if args.no_voice else
        CapabilityStatus.READY if tts_state["engine"] != "print"
        else CapabilityStatus.DEGRADED,
        "--no-voice" if args.no_voice else
        f"{tts_state['engine']}: {tts_state['model'] or tts_state['error']}")
    capabilities.set("camera", "hardware", CapabilityStatus.READY,
                     "replay" if str(args.source).startswith("replay:") else "live source")
    _moondream_ready = voice_agent.moondream_status().get("available", False)
    capabilities.set(
        "agent_vision", "cloud",
        (CapabilityStatus.LOADING if args.enable_agent_vision and _moondream_ready
         else CapabilityStatus.FAILED if args.enable_agent_vision
         else CapabilityStatus.UNCONFIGURED),
        ("consented; provider authorization pending" if args.enable_agent_vision
         and _moondream_ready else
         "Moondream credential unavailable" if args.enable_agent_vision else
         "disabled; use --enable-agent-vision"))
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
            listener_ready = bool(voice_agent.listener is not None and
                                  getattr(voice_agent.listener, "available", False))
            if listener_ready:
                print("[agent] listener attached — the agent can hear replies")
                capabilities.set("speech_recognition", "model", CapabilityStatus.READY,
                                 "listener ready")
            elif args.listen:
                capabilities.set("speech_recognition", "model", CapabilityStatus.FAILED,
                                 "listener unavailable; install requirements-asr.txt")
            if args.detect_cough:
                print("[audio] cough episode detection enabled")
        else:
            shared_signals.set("microphone_ready", False)
            capabilities.set(
                "speech_recognition", "model",
                CapabilityStatus.FAILED if args.listen else CapabilityStatus.UNCONFIGURED,
                "microphone or speech listener unavailable" if args.listen else "disabled")

    else:
        capabilities.set("microphone", "hardware", CapabilityStatus.UNCONFIGURED, "disabled")
        capabilities.set("speech_recognition", "model", CapabilityStatus.UNCONFIGURED,
                         "disabled")
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
    elif args.showcase:
        voice_agent.start_showcase()
    elif args.demo:
        voice_agent.start_demo_circuit()

    web = None
    web_publish_executor = None
    web_publish_state = {"future": None, "last": -1e9}
    ipad_executor = None
    ipad_publish_executor = None
    ipad_publish_state = {"future": None, "last": -1e9, "last_tele": -1e9}
    # The control handlers below serve both surfaces, so they are built whenever
    # either one is active. Keeping them in one place matters: module toggles
    # validate against the loaded scheduler/subject-pool names, and that
    # validation must not be duplicated per transport.
    if args.webui or ipad_source:
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

        def module_handler(action, target=None, scope=None):
            """Web hook for the /modules console, mirroring the 't'/'a'/'d'
            hotkeys so a detector's on-demand action can be started from a
            phone. Raises ValueError on an unknown action or target (rendered
            as HTTP 400); passive modules simply expose no trigger.

            'enable'/'disable' pause or resume an already-instantiated
            module's cadence via ModuleGate -- they never load or unload a
            model, so re-enabling a heavy detector is instant. A module
            config/modules.yaml disabled at startup was never instantiated
            and cannot be toggled on this way; that still needs a restart."""
            if action == "circuit":
                return {"action": "circuit",
                        "started": bool(voice_agent.start_demo_circuit())}
            if action in ("enable", "disable"):
                effective_scope = scope or "primary"
                if effective_scope not in ("primary", "secondary"):
                    raise ValueError("unknown scope")
                if effective_scope == "primary":
                    loaded = {m.name for m in pipeline.scheduler.modules}
                else:
                    if pipeline.subject_pool is None:
                        raise RuntimeError(
                            "multi-person tracking has no secondary modules configured")
                    loaded = set(pipeline.subject_pool.module_names)
                if target not in loaded:
                    raise ValueError(
                        "module is not loaded for this scope; enable it in "
                        "config/modules.yaml and restart")
                pipeline.module_gate.set(target, action == "enable", scope=effective_scope)
                return {"action": action, "target": target, "scope": effective_scope,
                        "enabled": pipeline.module_gate.enabled(target, effective_scope)}
            if action == "vlm_scan":
                screening = next((m for m in pipeline.scheduler.modules
                                  if m.name == "skin_vision"), None)
                if screening is None or not screening.request_scan():
                    raise ValueError("cloud skin screening is not enabled")
                return {"action": "vlm_scan", "requested": True}
            if action != "test":
                raise ValueError("unsupported action")
            from assessments import PROTOCOLS
            if target not in set(PROTOCOLS) | {"hold_still", "arm_check"}:
                raise ValueError("unknown test")
            voice_agent.request_test(target)
            return {"action": "test", "target": target, "started": True}

        def say_handler(text):
            """Web hook feeding a typed reply into the same corroboration path
            as heard speech. Raises RuntimeError when the run has no typed
            listener, ValueError on empty text (both rendered as HTTP 400)."""
            if typed_listener is None:
                raise RuntimeError("typed input is not enabled (use --type-input)")
            if not typed_listener.push(text):
                raise ValueError("empty reply")
            return {"text": " ".join(str(text).split())[:400]}

        if args.webui:
            web = CompanionServer(port=args.webui_port, control_handler=control,
                                  primary_handler=primary_handler,
                                  assessment_handler=assessment_handler,
                                  say_handler=say_handler,
                                  module_handler=module_handler,
                                  allow_remote_module_toggle=args.allow_remote_toggle)
            web.start()
            web_publish_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="web-publish")

        if ipad_source:
            ipad_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ipad-control")
            # Separate from ipad_executor (which services inbound control
            # replies): the outbound telemetry projection calls to_payload(),
            # which is non-trivial, so it is built here off the capture loop and
            # pushed at ~1 Hz, mirroring the /data web-publish executor above.
            ipad_publish_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ipad-telemetry")

            def ipad_control(payload, reply):
                """Service a control message from the paired iPad.

                Returns immediately: the work runs on a worker thread because
                this is called from the link's asyncio loop, and blocking there
                would stall frame receive for every module toggle.
                """
                action = str(payload.get("action", ""))
                target = payload.get("target")
                scope = payload.get("scope")

                def run():
                    # `target` is echoed at the top level of both replies: the
                    # page keys its per-button busy flag off msg.target, so an
                    # error without it would leave that button stuck spinning.
                    try:
                        if payload.get("type") != "module":
                            raise ValueError("unsupported control message")
                        if action in ("enable", "disable") and args.no_ipad_toggle:
                            raise RuntimeError("module toggles from the iPad are "
                                               "disabled (--no-ipad-toggle)")
                        result = module_handler(
                            action, str(target) if target is not None else None,
                            str(scope) if scope is not None else None)
                        reply({"type": "module_result", "ok": True,
                               "target": target, "module": result})
                    except (ValueError, TypeError, RuntimeError) as exc:
                        # Same shape webui/server.py returns, so the page's error
                        # ladder works unchanged across both transports.
                        reply({"type": "module_result", "ok": False,
                               "target": target, "error": str(exc)})
                ipad_executor.submit(run)

            pipeline.camera.set_ipad_control_handler(ipad_control)
            if not args.no_ipad_toggle:
                print("[ipad] module toggles from the paired iPad are ENABLED "
                      "(--no-ipad-toggle to refuse them)")

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
    elif ipad_source:
        # The paired page still needs its compact telemetry payload in headless
        # mode; only the OpenCV overlay renderer is display-dependent.
        from output import dashboard

    decoupled_display = _uses_decoupled_display(
        display, args.source, args.alt_source)

    force_greet = {"v": False}
    last_alert_print: dict[str, float] = {}
    last_vitals_print = {"t": 0.0}
    last_greeting = {"text": None}
    analysis_state_lock = threading.Lock()
    analysis_state = {"ctx": None, "snapshot": [], "reasoning": None}

    def system_snapshot(private: bool = False, results=None, performance=None,
                        features=None):
        system = {
            # Which frame features the scheduler could satisfy this frame, so
            # /modules can say *why* a detector is idle rather than showing
            # an empty panel. Mirrors Scheduler._requirements_met.
            "features": features or {},
            "timeline": EventStore.instance().recent(60),
            "capabilities": capabilities.snapshot(),
            "workflows": WorkflowEngine.instance().snapshot(),
            "consent": {"cloud_skin": args.enable_cloud_skin,
                        "cloud_scene": args.enable_cloud_scene,
                        "agent_vision": args.enable_agent_vision},
            "replay": pipeline.camera.replay_status(),
            "moondream": voice_agent.moondream_status(),
            # Live spoken-voice engine state (may downgrade after async load,
            # e.g. piper model missing -> pyttsx3 -> print).
            "tts": voice_agent.speaker.status(),            "modules_enabled": sorted(
                m.name for m in pipeline.scheduler.modules
                if pipeline.module_gate is None
                or pipeline.module_gate.enabled(m.name, "primary")),
            # Unlike modules_enabled above (which reflects live gate state --
            # empty under --start-blank), this reflects what was actually
            # instantiated at startup and never changes with the gate. /modules
            # needs this to know a paused module can still be toggled back on,
            # instead of wrongly claiming it needs a restart.
            "modules_loaded": sorted(m.name for m in pipeline.scheduler.modules),
            "module_gate": (pipeline.module_gate.snapshot()
                           if pipeline.module_gate is not None else None),
            "secondary_modules": (sorted(pipeline.subject_pool.module_names)
                                  if pipeline.subject_pool is not None else []),
        }
        # Bounded provider health for every consumer, not just the private
        # debug port: without it a cloud outage is indistinguishable from a
        # dead "Ask the vision model now" button on /modules.
        _screening = next((m for m in pipeline.scheduler.modules
                           if m.name == "skin_vision"), None)
        if _screening is not None and hasattr(_screening, "provider_health"):
            system["cloud_vision"] = _screening.provider_health()
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
            system["conversation_agent"] = voice_agent.conversation_diagnostics()
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
        voice_agent.observe_frame(ctx.frame, agent_snapshot, now=ctx.timestamp)
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
                # Read off the settled context here, not in the scheduler's
                # hot loop; these are exactly the tokens `requires` gates on.
                frame_features = {"face": ctx.face is not None,
                                  "pose": ctx.pose is not None,
                                  "person": bool(ctx.person_present),
                                  "depth": ctx.depth is not None}
                def publish_data_snapshot():
                    web.publish_data(
                        stable_snapshot, ctx.fps, greeting, reasoning=reasoning,
                        system=system_snapshot(features=frame_features),
                        performance=performance)
                web_publish_state["future"] = web_publish_executor.submit(
                    publish_data_snapshot)
                web_publish_state["last"] = ctx.timestamp

        # mirror module state to the paired iPad so its toggle grid stays honest.
        # send_control hands off to the link's loop and returns, so this stays a
        # cheap call on the heavy loop.
        if ipad_source and ctx.timestamp - ipad_publish_state["last"] >= 0.5:
            ipad_publish_state["last"] = ctx.timestamp
            gate = getattr(pipeline, "module_gate", None)
            modules_state = [
                {"name": m.name, "scope": "primary",
                 "enabled": bool(gate.enabled(m.name, "primary")) if gate else True}
                for m in pipeline.scheduler.modules]
            pipeline.camera.ipad_control({
                "type": "state", "modules": modules_state,
                "fps": round(ctx.fps, 1),
                "capture": pipeline.camera.diagnostics(),
                # Desired browser capture policy is deliberately separate from
                # delivered capture diagnostics and analysis-loop performance.
                "capture_config": ipad_capture_config,
                # The agent's own last line for a caption bar (never a
                # transcript of the person).
                "agent": voice_agent.public_line()})

        # push the live dashboard to the paired iPad over that same control
        # channel -- its only feed, since the hotspot AP isolates it from the
        # /data HTTP endpoint. Built on a worker thread (ipad_payload wraps the
        # non-trivial to_payload) so the capture loop stays cheap; mirrors the
        # /data web-publish future pattern above.
        if ipad_source and ipad_publish_executor is not None:
            pending: Future | None = ipad_publish_state["future"]
            if pending is not None and pending.done():
                try:
                    pending.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[ipad] telemetry publication failed "
                          f"({type(exc).__name__}: {exc})")
                ipad_publish_state["future"] = None
                pending = None
            if pending is None and ctx.timestamp - ipad_publish_state["last_tele"] >= 1.0:
                tele_snapshot = list(snapshot)
                tele_fps = ctx.fps
                tele_greeting = last_greeting["text"]
                tele_reasoning = voice_agent.reasoning_card()
                tele_performance = pipeline.runtime_metrics.snapshot()
                tele_features = {"face": ctx.face is not None,
                                 "pose": ctx.pose is not None,
                                 "person": bool(ctx.person_present),
                                 "depth": ctx.depth is not None}
                def publish_ipad_telemetry():
                    payload = dashboard.ipad_payload(
                        tele_snapshot, tele_fps, tele_greeting,
                        reasoning=tele_reasoning,
                        system=system_snapshot(features=tele_features),
                        performance=tele_performance)
                    pipeline.camera.ipad_control(payload)
                ipad_publish_state["future"] = ipad_publish_executor.submit(
                    publish_ipad_telemetry)
                ipad_publish_state["last_tele"] = ctx.timestamp

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
            if key == ord("s"):
                if voice_agent.start_showcase():
                    print("[main] narrated showcase tour requested ('s')")
            if key == ord("c") and primary_source != alt_source:
                nxt = alt_source if cam_state["current"] == primary_source else primary_source
                print(f"[camera] switching -> {nxt}")
                pipeline.reset_capture_state()
                pipeline.camera.switch_to(nxt, camera_opts)
                cam_state["current"] = nxt
        return True

    print("[main] starting; press 'q' to quit, 'g' to greet, 'm' to toggle "
          "Moondream, 't' for a tremor test, 'a' for an arm skin check, "
          "'d' for a guest/client demo circuit, 's' for the narrated showcase, "
          "'c' to switch camera (Ctrl+C in headless).")
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
            last_preview_token = None
            last_panel_at = 0.0
            while worker.is_alive():
                packet = pipeline.camera.latest_frame()
                preview_frame = None
                preview_token = None
                if packet is not None:
                    seq, raw_frame, captured_at = packet
                    pipeline.runtime_metrics.note_capture(
                        pipeline.camera.current_fps, captured_at,
                        pipeline.camera.diagnostics())
                    preview_frame = raw_frame
                    preview_token = ("capture", cam_state["current"], seq)
                else:
                    # Files and replay-style backends do not expose a separate
                    # latest-frame slot. If one is selected via the hot switch,
                    # keep the GUI live at analysis cadence instead of freezing
                    # the decoupled loop that an iPad/local source enabled.
                    with analysis_state_lock:
                        fallback_ctx = analysis_state["ctx"]
                    if fallback_ctx is not None:
                        preview_frame = fallback_ctx.frame
                        preview_token = (
                            "analysis", cam_state["current"],
                            fallback_ctx.frame_index, fallback_ctx.timestamp)
                if preview_frame is not None and preview_token != last_preview_token:
                    last_preview_token = preview_token
                    with analysis_state_lock:
                        analyzed_ctx = analysis_state["ctx"]
                        snapshot = list(analysis_state["snapshot"])
                        reasoning = analysis_state["reasoning"]
                    performance = pipeline.runtime_metrics.snapshot()
                    frame = preview_frame.copy()
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
                if key == ord("s"):
                    if voice_agent.start_showcase():
                        print("[main] narrated showcase tour requested ('s')")
                if key == ord("c") and primary_source != alt_source:
                    nxt = alt_source if cam_state["current"] == primary_source else primary_source
                    print(f"[camera] switching -> {nxt}")
                    with analysis_state_lock:
                        analysis_state["ctx"] = None
                    last_preview_token = None
                    pipeline.reset_capture_state()
                    pipeline.camera.switch_to(nxt, camera_opts)
                    cam_state["current"] = nxt
                if preview_frame is None:
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
        if ipad_executor is not None:
            ipad_executor.shutdown(wait=False, cancel_futures=True)
        if ipad_publish_executor is not None:
            pending = ipad_publish_state.get("future")
            if pending is not None:
                pending.cancel()
            ipad_publish_executor.shutdown(wait=False, cancel_futures=True)
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
