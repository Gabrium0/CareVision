"""rPPG heart-rate backend interface.

Thin specialization of the shared `modules.backends.base.Backend`. compute()
returns a dict with any of:

    bpm             float   heart rate, beats per minute
    confidence      float   0..1 self-assessed quality
    hrv_rmssd_ms    float   HRV RMSSD in ms            (optional)
    hrv_sdnn_ms     float   HRV SDNN in ms             (optional)
    breaths_per_min float   respiration rate           (optional)
"""
from __future__ import annotations

from modules.backends.base import Backend


class RPPGBackend(Backend):
    """Alias kept for the rPPG backends; identical contract to Backend."""
    pass
