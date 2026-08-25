"""Focused contract for the browser capture-policy asset route."""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("RELAY_SECRET", "test-only-relay-secret")

from relay import server


def test_capture_policy_route_is_exact_and_never_cached():
    routes = {
        route.resource.canonical
        for route in server.build_app().router.routes()
        if route.method == "GET"
    }

    assert "/static/ipad_capture_policy.js" in routes
    assert "/static" not in routes

    response = asyncio.run(server.serve_capture_policy(None))
    assert response.status == 200
    assert response.content_type == "application/javascript"
    assert response.headers["Cache-Control"] == "no-store, must-revalidate"
    assert "IPadCapturePolicy" in response.text


def test_signalling_websocket_has_proxy_keepalive():
    assert server.SIGNAL_HEARTBEAT_SECONDS == 30.0
