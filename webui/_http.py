"""Shared HTTP server base that keeps benign client disconnects off the console.

Live demos run with the terminal visible. Closing, refreshing, or navigating
away from a browser tab while a server is mid-write raises a connection error
on the socket (ConnectionAbortedError on Windows, BrokenPipeError or
ConnectionResetError on Linux/macOS) purely because the peer went away — it is
not a bug in the app. socketserver.BaseServer.handle_error() defaults to
printing the full traceback for every request-handling exception, so that
harmless event would otherwise dump a stack trace mid-demo. This base class
swallows only those three disconnect exceptions and defers to the normal
traceback-printing behavior for everything else, so real errors still surface.
"""
from __future__ import annotations

import sys
from http.server import ThreadingHTTPServer

_DISCONNECT_ERRORS = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that silences benign client-disconnect tracebacks."""

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, _DISCONNECT_ERRORS):
            return
        super().handle_error(request, client_address)
