"""Unit tests for QuietThreadingHTTPServer (webui/_http.py).

Live demos run with the terminal visible, and the stdlib's default
handle_error() prints a full traceback for every exception raised while
handling a request -- including the benign case where a browser tab was
closed mid-write. These tests prove the quiet override swallows exactly the
three disconnect exceptions (ConnectionAbortedError on Windows,
BrokenPipeError/ConnectionResetError on Linux/macOS) and nothing else, so a
genuine bug (e.g. ValueError) still surfaces on stderr instead of vanishing.
"""
from __future__ import annotations

import sys

import pytest

from webui._http import QuietThreadingHTTPServer


def _make_server() -> QuietThreadingHTTPServer:
    # __new__ so the constructor (which binds a real socket) never runs;
    # handle_error() only touches sys.exc_info(), not any server state.
    return QuietThreadingHTTPServer.__new__(QuietThreadingHTTPServer)


@pytest.mark.parametrize("exc_type", [ConnectionAbortedError, ConnectionResetError,
                                      BrokenPipeError])
def test_handle_error_is_silent_for_disconnects(capsys, exc_type):
    server = _make_server()
    try:
        raise exc_type("client went away")
    except exc_type:
        server.handle_error(("127.0.0.1", 0), ("127.0.0.1", 0))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_handle_error_still_reports_genuine_errors(capsys):
    server = _make_server()
    try:
        raise ValueError("something actually broke")
    except ValueError:
        server.handle_error(("127.0.0.1", 0), ("127.0.0.1", 0))
    captured = capsys.readouterr()
    assert "ValueError" in captured.err
    assert "something actually broke" in captured.err
