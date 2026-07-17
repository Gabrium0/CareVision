"""Loopback-only caregiver review portal and privacy-safe JSON API."""
from __future__ import annotations

import csv
import io
import json
import math
import secrets
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from storage.event_store import (CaseConflictError, CaseNotFoundError,
                                 CaseValidationError)


_ROOT = Path(__file__).resolve().parent
_ASSETS = {
    "/caregiver": (_ROOT / "caregiver.html", "text/html; charset=utf-8"),
    "/caregiver/": (_ROOT / "caregiver.html", "text/html; charset=utf-8"),
    "/caregiver/portal.js": (_ROOT / "caregiver.js", "text/javascript; charset=utf-8"),
    "/caregiver/portal.css": (_ROOT / "caregiver.css", "text/css; charset=utf-8"),
}
_RANGES = {"24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}
_MAX_BODY = 4096


def _numeric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _csv_safe(value):
    """Prevent spreadsheet formula execution in explicit CSV exports."""
    if value is None:
        return ""
    text = str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


def _bucket_series(events: list[dict], start: float, end: float,
                   max_points: int = 300) -> list[dict]:
    """Downsample selected numeric observations into min/mean/max buckets."""
    selected = {"heart_rate", "respiration", "sensor", "activity_level", "routine"}
    grouped = defaultdict(list)
    for event in events:
        module = str(event.get("module") or "")
        if module not in selected:
            continue
        value = _numeric((event.get("payload") or {}).get("value"))
        if value is None:
            continue
        grouped[(event["subject_id"], module, str(event.get("key") or "value"))].append(
            (float(event["timestamp"]), value))
    span = max(1.0, end - start)
    bucket_count = max(1, min(int(max_points), 300))
    width = span / bucket_count
    out = []
    for (subject, module, key), points in sorted(grouped.items()):
        buckets = defaultdict(list)
        for timestamp, value in points:
            index = min(bucket_count - 1, max(0, int((timestamp - start) / width)))
            buckets[index].append(value)
        compact = []
        for index, values in sorted(buckets.items()):
            compact.append({"timestamp": start + (index + .5) * width,
                            "mean": sum(values) / len(values),
                            "min": min(values), "max": max(values),
                            "count": len(values)})
        out.append({"subject_id": subject, "module": module, "key": key,
                    "points": compact})
    return out


def _event_counts(events: list[dict]) -> dict:
    counts = {"alerts": 0, "falls_near_falls": 0, "cough_episodes": 0,
              "assessments": 0, "routine_deviations": 0}
    for event in events:
        module, key = str(event.get("module") or ""), str(event.get("key") or "")
        if event.get("severity") == "alert":
            counts["alerts"] += 1
        if module in ("fall", "near_fall") or "fall" in key:
            counts["falls_near_falls"] += 1
        if module == "sound_event" and key == "cough_episode":
            counts["cough_episodes"] += 1
        if event.get("kind") == "assessment":
            counts["assessments"] += 1
        if module == "routine" and key not in ("daily_summary",):
            counts["routine_deviations"] += 1
    return counts


class CaregiverService:
    """Bounded caregiver queries and mutations over existing safe stores."""

    def __init__(self, event_store, history_store, now=time.time):
        self.events = event_store
        self.history = history_store
        self.now = now

    def subjects(self) -> list[str]:
        """Return privacy-safe retained subject identifiers."""
        values = set(self.events.subjects()) | set(self.history.subjects())
        values.add("primary")
        return sorted(values, key=lambda value: (value != "primary", value))

    def _selection(self, query: dict) -> tuple[str, str, float, float]:
        subject = str((query.get("subject_id") or ["primary"])[0])
        range_name = str((query.get("range") or ["7d"])[0])
        if subject != "all" and subject not in self.subjects():
            raise CaseValidationError("unknown subject")
        if range_name not in _RANGES:
            raise CaseValidationError("range must be 24h, 7d, or 30d")
        end = float(self.now())
        return subject, range_name, end - _RANGES[range_name], end

    def summary(self, query: dict) -> dict:
        """Build one bounded case, count, trend, and retention snapshot."""
        subject, range_name, start, end = self._selection(query)
        events = self.events.query(start=start, end=end, subject_id=subject)
        series = [item for item in self.history.portal_series(subject, start, end)
                  if item["module"] in ("sensor", "routine", "activity_level")]
        series.extend(_bucket_series(events, start, end))
        status = str((query.get("status") or ["all"])[0])
        cases = self.events.list_cases(subject_id=subject, status=status,
                                       start=start, limit=500)
        retention = self.events.retention_status(end)
        retention.update(self.history.retention_status(end - 365 * 86400))
        return {"subject_id": subject, "range": range_name,
                "start": start, "end": end, "subjects": self.subjects(),
                "counts": _event_counts(events), "series": series,
                "cases": cases, "retention": retention}

    def export(self, query: dict, format_name: str) -> tuple[bytes, str, str]:
        """Create an explicit JSON or spreadsheet-safe CSV download."""
        subject, range_name, start, end = self._selection(query)
        events = self.events.query(start=start, end=end, subject_id=subject)
        cases = [self.events.case(item["id"]) for item in
                 self.events.list_cases(subject_id=subject, start=start, limit=1000)
                 if item["opened_at"] >= start]
        series = self.history.portal_series(subject, start, end)
        if format_name == "json":
            body = json.dumps({"subject_id": subject, "range": range_name,
                               "events": events, "cases": cases,
                               "history_series": series}, indent=2).encode("utf-8")
            return body, "application/json", "caregiver-export.json"
        if format_name != "csv":
            raise CaseValidationError("format must be json or csv")
        target = io.StringIO(newline="")
        fields = ["record_type", "id", "timestamp", "subject_id", "module",
                  "key", "status", "severity", "value", "action", "note"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({"record_type": "event", "id": event["id"],
                             "timestamp": event["timestamp"],
                             "subject_id": event["subject_id"],
                             "module": event["module"], "key": event["key"],
                             "severity": event["severity"],
                             "value": json.dumps(event["payload"], separators=(",", ":"))})
        for case in cases:
            writer.writerow({"record_type": "case", "id": case["id"],
                             "timestamp": case["opened_at"],
                             "subject_id": case["subject_id"],
                             "module": case["module"], "key": case["key"],
                             "status": case["status"], "severity": case["severity"],
                             "value": _csv_safe(case["summary"])})
            for action in case["actions"]:
                writer.writerow({"record_type": "case_action", "id": action["id"],
                                 "timestamp": action["timestamp"],
                                 "subject_id": case["subject_id"],
                                 "module": case["module"], "key": case["key"],
                                 "status": case["status"],
                                 "action": action["action"],
                                 "note": _csv_safe(action["note"])})
        return target.getvalue().encode("utf-8"), "text/csv; charset=utf-8", \
            "caregiver-export.csv"

    def purge(self, confirmation: str) -> dict:
        """Purge only rows whose configured retention period has expired."""
        if confirmation != "purge-expired":
            raise CaseValidationError("confirmation must be purge-expired")
        now = float(self.now())
        return {"events_and_cases": self.events.purge_expired(now),
                "history": self.history.purge_before(now - 365 * 86400)}


def _make_handler(service: CaregiverService, csrf_token: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def _send(self, code=200, ctype="application/json", body=b"", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json(self, value, code=200):
            self._send(code, body=json.dumps(value, separators=(",", ":")).encode("utf-8"))

        def _error(self, exc):
            if isinstance(exc, CaseNotFoundError):
                self._json({"ok": False, "error": "case not found"}, 404)
            elif isinstance(exc, CaseConflictError):
                self._json({"ok": False, "error": str(exc)}, 409)
            elif isinstance(exc, (CaseValidationError, ValueError, TypeError,
                                  json.JSONDecodeError)):
                self._json({"ok": False, "error": str(exc)}, 400)
            else:
                self._json({"ok": False, "error": type(exc).__name__}, 500)

        def do_GET(self):
            parsed = urlparse(self.path)
            path, query = parsed.path, parse_qs(parsed.query)
            if path in _ASSETS:
                file, ctype = _ASSETS[path]
                try:
                    self._send(ctype=ctype, body=file.read_bytes())
                except OSError:
                    self._send(500, "text/plain", b"caregiver portal asset missing")
                return
            try:
                if path == "/caregiver/api/bootstrap":
                    self._json({"csrf_token": csrf_token,
                                "subjects": service.subjects(),
                                "ranges": list(_RANGES)})
                elif path == "/caregiver/api/summary":
                    self._json(service.summary(query))
                elif path == "/caregiver/api/cases":
                    subject = str((query.get("subject_id") or ["primary"])[0])
                    status = str((query.get("status") or ["all"])[0])
                    if subject != "all" and subject not in service.subjects():
                        raise CaseValidationError("unknown subject")
                    self._json({"cases": service.events.list_cases(
                        subject_id=subject, status=status)})
                elif path.startswith("/caregiver/api/cases/"):
                    case_id = path.rsplit("/", 1)[-1]
                    case = service.events.case(case_id)
                    if case is None:
                        raise CaseNotFoundError(case_id)
                    self._json(case)
                elif path == "/caregiver/api/export":
                    format_name = str((query.get("format") or ["json"])[0])
                    body, ctype, filename = service.export(query, format_name)
                    self._send(ctype=ctype, body=body,
                               extra={"Content-Disposition": f'attachment; filename="{filename}"'})
                else:
                    self._json({"ok": False, "error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                self._error(exc)

        def do_POST(self):
            parsed = urlparse(self.path)
            if self.headers.get("X-Caregiver-CSRF") != csrf_token:
                self._json({"ok": False, "error": "invalid CSRF token"}, 403)
                return
            if self.headers.get_content_type() != "application/json":
                self._json({"ok": False, "error": "application/json required"}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > _MAX_BODY:
                    self._json({"ok": False, "error": "request body too large"}, 413)
                    return
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise CaseValidationError("JSON object required")
                path = parsed.path
                if path == "/caregiver/api/retention/purge-expired":
                    if set(body) != {"confirm"}:
                        raise CaseValidationError("unknown request fields")
                    self._json({"ok": True, "purged": service.purge(body["confirm"])})
                    return
                prefix = "/caregiver/api/cases/"
                if not path.startswith(prefix):
                    self._json({"ok": False, "error": "not found"}, 404)
                    return
                suffix = path[len(prefix):].split("/")
                if len(suffix) != 2 or suffix[1] not in ("acknowledge", "resolve", "note"):
                    self._json({"ok": False, "error": "not found"}, 404)
                    return
                allowed = {"version", "note"}
                if not set(body) <= allowed or "version" not in body:
                    raise CaseValidationError("version is required; unknown fields refused")
                case = service.events.mutate_case(
                    suffix[0], suffix[1], int(body["version"]), body.get("note"))
                self._json({"ok": True, "case": case})
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                self._error(exc)

    return Handler


class CaregiverServer:
    """Serve the caregiver portal on 127.0.0.1 only."""

    def __init__(self, event_store, history_store, port: int = 8772):
        self.port = int(port)
        self.service = CaregiverService(event_store, history_store)
        self.csrf_token = secrets.token_urlsafe(32)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind the portal to loopback and start its daemon HTTP thread."""
        self._httpd = ThreadingHTTPServer(
            ("127.0.0.1", self.port), _make_handler(self.service, self.csrf_token))
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="caregiver-http")
        self._thread.start()
        host, port = self._httpd.server_address
        print(f"[caregiver] local review portal: http://{host}:{port}/caregiver")

    def stop(self) -> None:
        """Stop the HTTP server and join its background thread."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
