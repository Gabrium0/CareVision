"""Demo launcher for the companion display (used for previewing the page).

    python -m webui._demo
Starts the server on port 8770 and publishes a few sample agent lines so the
page has content to render, then serves forever.
"""
import time

from webui.server import CompanionServer

srv = CompanionServer(port=8770)
srv.start()
for t in [
    "Good afternoon, Margaret! Lovely to see you.",
    "You're looking well today.",
    "It feels a little cool today — a cozy sweater might keep you comfortable.",
]:
    srv.publish(t)
    time.sleep(0.5)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    srv.stop()
