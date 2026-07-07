"""Companion web display: shows the voice agent's spoken lines as large text on
a browser (e.g. an iPad mounted in front of the humanoid) so the person can read
what the agent says. Broadcast-only; no new detection logic.
"""
from webui.server import CompanionServer, UtteranceBus

__all__ = ["CompanionServer", "UtteranceBus"]
