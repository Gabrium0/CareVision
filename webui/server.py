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


class UtteranceBus:
    """Thread-safe pub/sub of agent lines with a small replay buffer."""

    def __init__(self, history: int = 20):
        self._cond = threading.Condition()
        self._seq = 0
        self._items: deque[dict] = deque(maxlen=history)

    def publish(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._cond:
            self._seq += 1
            self._items.append({"seq": self._seq, "text": text, "ts": time.time()})
            self._cond.notify_all()

    def since(self, seq: int) -> list[dict]:
        with self._cond:
            return [it for it in self._items if it["seq"] > seq]

    def current_seq(self) -> int:
        with self._cond:
            return self._seq

    def wait(self, last_seq: int, timeout: float) -> list[dict]:
        """Block until there are items newer than last_seq (or timeout)."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            return [it for it in self._items if it["seq"] > last_seq]


def _make_handler(bus: UtteranceBus):
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
            else:
                self._send(code=404, body=b"not found")

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
    def __init__(self, port: int = 8770, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.bus = UtteranceBus()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port),
                                          _make_handler(self.bus))
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        urls = [f"http://{ip}:{self.port}" for ip in _lan_ips()] or \
               [f"http://localhost:{self.port}"]
        print("[webui] companion display running — open on the iPad:")
        for u in urls:
            print(f"[webui]   {u}")

    def publish(self, text: str) -> None:
        self.bus.publish(text)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
