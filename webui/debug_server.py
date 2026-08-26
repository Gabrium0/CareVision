"""Opt-in localhost-only raw diagnostic state endpoint."""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse, parse_qs

import numpy as np

from core.events import Result
from storage.event_store import EventStore
from webui._http import QuietThreadingHTTPServer

_DEBUG_PAGE = Path(__file__).resolve().parent / "debug.html"

# Modules of interest for the export endpoint
_EXPORT_MODULES = {
    "heart_rate", "respiration", "clothing", "clothing_advice", "weather",
    "skin_color", "rash", "skin_vision", "scene_vision", "arm_skin",
    "facial_swelling", "facial_asymmetry", "bruise", "eye_redness",
    "dry_lips", "sweating", "age_estimation", "body_estimate",
    "emotion", "pain", "agitation", "drowsiness", "yawn", "head_nod",
    "tremor", "gait", "balance", "bradykinesia", "masked_face", "expressivity",
    "fall", "near_fall", "unresponsive", "wandering", "attention", "gesture",
    "sneeze", "face_touch", "activity_level", "presence", "grooming",
    "height_distance", "multi_person", "routine", "hazard_zones",
    "guided_assessments", "replay_events",
}


def safe_json(value: Any) -> Any:
    """Convert diagnostic values without ever serializing raw media."""
    if value is None or isinstance(value, (bool, str, int)):
        if isinstance(value, str) and value.strip().lower().startswith("data:"):
            return "<redacted-media>"
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, (bytes, bytearray, memoryview, np.ndarray)):
        return "<redacted-media>"
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [safe_json(v) for v in value]
    return f"<unsupported:{type(value).__name__}>"


def serialize_result(result: Result) -> dict:
    return safe_json({
        "module": result.module, "key": result.key, "value": result.value,
        "confidence": result.confidence, "severity": result.severity,
        "message": result.message, "ttl": result.ttl,
        "visibility": result.visibility, "timestamp": result.timestamp,
        "subject_id": result.subject_id, "source": result.source,
        "quality": result.quality, "location": result.location,
        "evidence_window": result.evidence_window,
        "correlation_id": result.correlation_id,
        "conversation_tags": result.conversation_tags,
        "persistence": result.persistence,
    })


def build_debug_payload(results: list[Result], performance: dict,
                        system: dict | None = None) -> dict:
    system = system or {}
    components = {
        "runtime": (performance.get("health") or {}).get("status", "healthy"),
        "moondream": _component_state(system.get("moondream"), optional=True),
        "nvidia_skin": _component_state(system.get("nvidia_skin"), optional=True),
        "vitals": _component_state(system.get("vitals")),
        "audio": _component_state(system.get("audio")),
        "history_writer": _component_state(system.get("history_writer")),
        "runtime_resources": _component_state(system.get("runtime_resources")),
        "clothing": _component_state(system.get("clothing"), optional=True),
    }
    reasons = list((performance.get("health") or {}).get("reasons", []))
    reasons.extend(name + "_degraded" for name, state in components.items()
                   if state == "degraded" and name != "runtime")
    overall = "failed" if "failed" in components.values() else (
        "degraded" if reasons else "healthy")
    component_actions = {
        "moondream_degraded": "check Moondream credential, quota, and provider status",
        "nvidia_skin_degraded": "inspect NVIDIA HTTP or schema validation diagnostics",
        "audio_degraded": "inspect microphone and Whisper worker state",
        "history_writer_degraded": "inspect SQLite writer failures or queue pressure",
        "runtime_resources_degraded": "inspect native thread-pool configuration",
        "clothing_degraded": "inspect FashionCLIP CUDA model lifecycle",
    }
    health = {"status": overall, "reasons": reasons,
              "components": components,
              "actions": list((performance.get("health") or {}).get("actions", []))
                         + [component_actions[r] for r in reasons if r in component_actions]}
    return safe_json({"timestamp": time.time(), "health": health,
                      "performance": performance,
                      "results": [serialize_result(r) for r in results],
                      "system": system})


def _component_state(value: Any, optional: bool = False) -> str:
    """Normalize heterogeneous diagnostics without treating opt-outs as faults."""
    if not isinstance(value, dict):
        return "unconfigured"
    if value.get("enabled") is False or value.get("consent") is False:
        return "unconfigured"
    if value.get("available") is False:
        return "unconfigured"
    if "alive" in value and value.get("alive") is False:
        return "failed"
    if (value.get("failures", 0) or value.get("aggregate_failures", 0)
            or value.get("dropped", 0)):
        return "degraded"
    stt = value.get("stt")
    if value.get("enabled") and isinstance(stt, dict) \
            and stt.get("status") in ("failed", "unavailable"):
        return "degraded"
    status = str(value.get("status") or value.get("state") or "").lower()
    if status == "degraded":
        return "degraded"
    if status in ("failed", "error"):
        return "degraded" if optional else "failed"
    if status in ("unavailable", "invalid_response"):
        return "degraded"
    # A person being outside the capture gate is actionable guidance, not a
    # software/component failure.
    if status == "blocked":
        return "healthy"
    if (value.get("authorization_failed")
            or value.get("circuit_state") in ("open", "authorization_failed")
            or value.get("failures", 0) >= 3):
        return "degraded"
    return "healthy"


def build_audio_debug_state(capabilities, detector, enabled: bool,
                            mode: str, listener=None) -> dict:
    """Compose safe microphone/YAMNet telemetry for the private dashboard."""
    def capability(name: str) -> dict:
        item = capabilities.get(name)
        if item is None:
            return {"status": "unavailable", "detail": "not initialized",
                    "updated_at": None}
        status = item.status.value if isinstance(item.status, Enum) else item.status
        return {"status": status, "detail": item.detail,
                "updated_at": item.updated_at}

    if detector is None:
        diagnostics = {
            "available": False,
            "status": "disabled" if not enabled else "unavailable",
            "mode": mode,
            "worker_alive": False,
            "windows_processed": 0,
            "last_inference_at": None,
            "latest_cough_confidence": 0.0,
            "peak_cough_confidence": 0.0,
            "max_scores": {},
            "threshold": None,
            "pending_cough_episode": False,
            "pending_burst_count": 0,
        }
    else:
        diagnostics = detector.diagnostics()
    listener_state = (listener.diagnostics() if listener is not None
                      and hasattr(listener, "diagnostics") else
                      {"status": ("not_applicable" if mode == "replay" else
                                  "disabled" if not enabled else "unavailable"),
                       "available": False, "worker_alive": False,
                       "worker_ready": False})
    resolved_mode = mode if mode == "replay" else diagnostics.get("mode", mode)
    return {"enabled": bool(enabled), "mode": resolved_mode,
            "microphone": capability("microphone"),
            "yamnet": capability("yamnet"), "detector": diagnostics,
            "stt": listener_state}


class DebugServer:
    """Serve a fresh provider snapshot on 127.0.0.1 only."""

    def __init__(self, provider: Callable[[], dict], port: int = 8771):
        self.provider = provider
        self.port = int(port)
        self._httpd: QuietThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        provider = self.provider

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/debug":
                    try:
                        body = _DEBUG_PAGE.read_bytes()
                        self.send_response(200)
                    except OSError:
                        body = b"<h1>debug.html missing</h1>"
                        self.send_response(500)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/debug/export":
                    self._handle_export(parsed.query)
                    return
                if path != "/debug/state":
                    self.send_error(404)
                    return
                try:
                    body = json.dumps(safe_json(provider())).encode("utf-8")
                    self.send_response(200)
                except Exception as exc:  # noqa: BLE001
                    body = json.dumps({"error": type(exc).__name__}).encode("utf-8")
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle_export(self, query: str):
                qs = parse_qs(query)
                now = time.time()
                # Default: last 60 seconds if no params given
                since = float(qs.get("since", [60.0])[0])
                # Optional comma-separated module filter
                modules_filter = None
                if "modules" in qs:
                    modules_filter = set(qs["modules"][0].split(","))
                # Optional include_events flag to include EventStore persisted events
                include_events = qs.get("include_events", ["0"])[0] in ("1", "true", "yes")

                # 1) Get live results from provider (Aggregator snapshot)
                try:
                    debug_payload = provider()
                except Exception:
                    debug_payload = {}
                live_results_raw = debug_payload.get("results", [])

                # 2) Optionally include EventStore persisted events
                event_results = []
                if include_events:
                    start = now - since
                    end = now
                    events = EventStore.instance().query(start=start, end=end, limit=5000)
                    event_results = [e for e in events if e.get("module") in _EXPORT_MODULES]

                # 3) Build grouped module results
                modules: dict[str, list[dict]] = {}
                for r in live_results_raw:
                    mod = r.get("module", "")
                    if modules_filter and mod not in modules_filter:
                        continue
                    # Format timestamp as ISO
                    ts = r.get("timestamp")
                    if isinstance(ts, (int, float)):
                        ts = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                    entry = {
                        "key": r.get("key", ""),
                        "value": r.get("value"),
                        "confidence": r.get("confidence"),
                        "severity": r.get("severity"),
                        "message": r.get("message", ""),
                        "timestamp": ts,
                    }
                    modules.setdefault(mod, []).append(entry)

                # 4) Build summary counts per module
                summary = {mod: len(entries) for mod, entries in modules.items()}

                payload = {
                    "query_time": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                    "session_lookback_seconds": since,
                    "count_live": sum(len(v) for v in modules.values()),
                    "count_events": len(event_results),
                    "modules_seen": sorted(modules.keys()),
                    "summary": summary,
                    "modules": modules,
                }
                if include_events:
                    payload["events"] = event_results

                body = json.dumps(safe_json(payload), indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = QuietThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="debug-http")
        self._thread.start()
        print(f"[debug] readable private dashboard: http://127.0.0.1:{self.port}/debug")
        print(f"[debug] private JSON state: http://127.0.0.1:{self.port}/debug/state")
        print(f"[debug] session export (live results): http://127.0.0.1:{self.port}/debug/export?since=60")
        print(f"[debug] session export (with events): http://127.0.0.1:{self.port}/debug/export?since=60&include_events=1")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None