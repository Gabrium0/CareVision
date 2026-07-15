"""Opt-in localhost-only raw diagnostic state endpoint."""
from __future__ import annotations

import json
import math
import threading
import time
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np

from core.events import Result


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
    return safe_json({"timestamp": time.time(), "performance": performance,
                      "results": [serialize_result(r) for r in results],
                      "system": system or {}})


class DebugServer:
    """Serve a fresh provider snapshot on 127.0.0.1 only."""

    def __init__(self, provider: Callable[[], dict], port: int = 8771):
        self.provider = provider
        self.port = int(port)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        provider = self.provider

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path.split("?", 1)[0] != "/debug/state":
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

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="debug-http")
        self._thread.start()
        print(f"[debug] private state: http://127.0.0.1:{self.port}/debug/state")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
