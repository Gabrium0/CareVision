"""Tiny stdlib web server that broadcasts the voice agent's utterances.

No third-party dependency: uses http.server on a daemon thread so it never
blocks the video loop. Clients get lines instantly via Server-Sent Events
(/events), with a JSON polling fallback (/latest). The page is webui/page.html.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_PAGE = Path(__file__).resolve().parent / "page.html"
_DATA_PAGE = Path(__file__).resolve().parent / "data.html"
_DEMO_PAGE = Path(__file__).resolve().parent / "demo.html"


class DataBus:
    """Holds the latest full-telemetry payload (one snapshot) for /data."""

    def __init__(self):
        self._cond = threading.Condition()
        self._seq = 0
        self._payload: dict | None = None

    def publish(self, payload: dict) -> None:
        """Publish a new item to connected subscribers."""
        with self._cond:
            self._seq += 1
            self._payload = payload
            self._cond.notify_all()

    def current(self) -> tuple[int, dict | None]:
        """Return the latest (seq, payload) pair."""
        with self._cond:
            return self._seq, self._payload

    def wait(self, last_seq: int, timeout: float) -> tuple[int, dict | None]:
        """Block until newer items arrive or the timeout elapses."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            return self._seq, self._payload


class UtteranceBus:
    """Thread-safe pub/sub of agent lines with a small replay buffer."""

    def __init__(self, history: int = 20):
        self._cond = threading.Condition()
        self._seq = 0
        self._items: deque[dict] = deque(maxlen=history)

    def publish(self, text: str) -> None:
        """Publish a new item to connected subscribers."""
        text = (text or "").strip()
        if not text:
            return
        with self._cond:
            self._seq += 1
            self._items.append({"seq": self._seq, "text": text, "ts": time.time()})
            self._cond.notify_all()

    def since(self, seq: int) -> list[dict]:
        """Return items newer than the given sequence number."""
        with self._cond:
            return [it for it in self._items if it["seq"] > seq]

    def current_seq(self) -> int:
        """Return the latest sequence number."""
        with self._cond:
            return self._seq

    def wait(self, last_seq: int, timeout: float) -> list[dict]:
        """Block until there are items newer than last_seq (or timeout)."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            return [it for it in self._items if it["seq"] > last_seq]


def _make_handler(bus: UtteranceBus, data_bus: DataBus, control_handler=None,
                  primary_handler=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):    # silence per-request console spam
            pass

        def _send(self, code=200, ctype="text/html; charset=utf-8", body=b"",
                  extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                try:
                    html = _PAGE.read_bytes()
                except OSError:
                    html = b"<h1>page.html missing</h1>"
                self._send(body=html)
            elif path == "/latest":
                qs = parse_qs(urlparse(self.path).query)
                since = int((qs.get("since", ["0"])[0]) or 0)
                payload = json.dumps({"items": bus.since(since),
                                      "seq": bus.current_seq()}).encode()
                self._send(ctype="application/json", body=payload,
                           extra={"Cache-Control": "no-store"})
            elif path == "/events":
                self._stream_events()
            elif path == "/data":
                try:
                    html = _DATA_PAGE.read_bytes()
                except OSError:
                    html = b"<h1>data.html missing</h1>"
                self._send(body=html)
            elif path == "/demo":
                try:
                    html = _DEMO_PAGE.read_bytes()
                except OSError:
                    html = b"<h1>demo.html missing</h1>"
                self._send(body=html)
            elif path == "/data-latest":
                seq, payload = data_bus.current()
                self._send(ctype="application/json",
                           body=json.dumps({"seq": seq, "payload": payload}).encode(),
                           extra={"Cache-Control": "no-store"})
            elif path == "/data-events":
                self._stream_data()
            else:
                self._send(code=404, body=b"not found")

        def do_POST(self):
            """Accept a narrow local replay-control API; no other mutation is exposed."""
            path = urlparse(self.path).path
            if path not in ("/replay-control", "/primary-control"):
                self._send(code=404, body=b"not found")
                return
            try:
                length = min(2048, int(self.headers.get("Content-Length", "0")))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if path == "/primary-control":
                    if primary_handler is None:
                        raise RuntimeError("anonymous tracking is not enabled")
                    track_id = str(request.get("track_id", ""))
                    if not track_id or not primary_handler(track_id):
                        raise ValueError("track is not currently assignable")
                    self._send(ctype="application/json",
                               body=json.dumps({"ok": True, "track_id": track_id}).encode())
                    return
                if control_handler is None:
                    raise RuntimeError("active source is not a replay")
                action = str(request.get("action", ""))
                if action not in ("pause", "resume", "restart", "seek", "speed"):
                    raise ValueError("unsupported action")
                value = request.get("value")
                result = control_handler(action, None if value is None else float(value))
                body = json.dumps({"ok": True, "replay": result}).encode()
                self._send(ctype="application/json", body=body)
            except (ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
                self._send(code=400, ctype="application/json",
                           body=json.dumps({"ok": False, "error": str(exc)}).encode())

        def _stream_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last = 0
            try:
                # replay recent history so a freshly-opened iPad isn't blank
                for it in bus.since(0):
                    self._write_event(it)
                    last = it["seq"]
                while True:
                    items = bus.wait(last, timeout=15.0)
                    if items:
                        for it in items:
                            self._write_event(it)
                            last = it["seq"]
                    else:
                        self.wfile.write(b": keep-alive\n\n")  # heartbeat
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return   # client (iPad) disconnected

        def _write_event(self, item: dict):
            self.wfile.write(f"data: {json.dumps(item)}\n\n".encode())

        def _stream_data(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last = -1
            try:
                while True:
                    seq, payload = data_bus.wait(last, timeout=15.0)
                    if seq != last and payload is not None:
                        last = seq
                        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    else:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    return Handler


def _lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))       # no packets sent; just picks the iface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


class CompanionServer:
    """Local web server: companion text at / and full telemetry at /data."""
    def __init__(self, port: int = 8770, host: str = "0.0.0.0",
                 data_hz: float = 2.0, control_handler=None,
                 primary_handler=None):
        self.port = port
        self.host = host
        self.bus = UtteranceBus()
        self.data_bus = DataBus()
        self._data_min_gap = 1.0 / data_hz if data_hz > 0 else 0.0
        self._last_data_push = 0.0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.control_handler = control_handler
        self.primary_handler = primary_handler

    def start(self) -> None:
        """Start the HTTP server on a background daemon thread."""
        self._httpd = ThreadingHTTPServer((self.host, self.port),
                                          _make_handler(self.bus, self.data_bus,
                                                        self.control_handler,
                                                        self.primary_handler))
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        urls = [f"http://{ip}:{self.port}" for ip in _lan_ips()] or \
               [f"http://localhost:{self.port}"]
        print("[webui] companion display running:")
        for u in urls:
            print(f"[webui]   {u}         (agent text — for the iPad)")
            print(f"[webui]   {u}/data    (all detections — for a caregiver)")
            print(f"[webui]   {u}/demo    (big-screen guest/client demo view)")

    def publish(self, text: str) -> None:
        """Publish a new item to connected subscribers."""
        self.bus.publish(text)

    def publish_data(self, snapshot, fps: float = 0.0, greeting: str | None = None,
                     reasoning: dict | None = None, system: dict | None = None,
                     performance: dict | None = None) -> None:
        """Push the full telemetry payload to /data, throttled to data_hz."""
        now = time.time()
        if now - self._last_data_push < self._data_min_gap:
            return
        self._last_data_push = now
        from output import dashboard
        self.data_bus.publish(dashboard.to_payload(snapshot, fps, greeting, reasoning,
                                                   system, performance))

    def stop(self) -> None:
        """Shut the server down and release its socket."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
