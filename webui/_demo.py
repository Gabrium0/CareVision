"""Demo launcher for the companion + data web pages (for previewing).

    python -m webui._demo
Starts the server on port 8770, publishes a few sample agent lines and a
sample detections snapshot, then serves forever so `/` and `/data` render.
"""
import time

from core.events import Result, Severity
from webui.server import CompanionServer


def R(m, k, v, sev=Severity.INFO, msg="", c=0.6):
    return Result(m, k, v, c, sev, msg)


SNAP = [
    R("heart_rate", "bpm_classical", 72.0, c=0.68), R("heart_rate", "bpm_open_rppg", 75.0, c=0.82),
    R("heart_rate", "hrv_rmssd_ms_open_rppg", 38.0, c=0.6), R("heart_rate", "hrv_sdnn_ms_open_rppg", 52.0, c=0.6),
    R("emotion", "emotion_heuristic", "happy", c=0.55), R("emotion", "emotion_hsemotion", "happiness", c=0.88),
    R("emotion", "emotion_deepface", "happy", c=0.79), R("emotion", "valence_hsemotion", 0.62, c=0.88),
    R("weather", "feels_like_c", 14.0), R("weather", "temperature_c", 15.0),
    R("weather", "rain_mm", 0.0), R("weather", "wind_kph", 12.0), R("weather", "status", "ready"),
    R("clothing", "upper_body", "t-shirt", c=0.41), R("clothing", "warmth_score", 1),
    R("clothing", "status", "ready (bare arm)"),
    R("clothing_advice", "recommendation", "It feels cool (14C). A sweater or light jacket could help.",
      Severity.NOTICE, "It feels cool (14C). A sweater or light jacket could help.", 0.6),
    R("drowsiness", "ear", 0.30), R("drowsiness", "perclos", 0.22, Severity.NOTICE),
    R("drowsiness", "blink_rate", 18), R("drowsiness", "perclos_status", "ready"),
    R("yawn", "mar", 0.20), R("head_nod", "head_pitch", 0.10), R("head_nod", "nod_count", 1),
    R("skin_color", "pallor", 0.03, Severity.WARNING, "Skin looks paler than baseline (possible pallor)", 0.4),
    R("tremor", "tremor_right", 5.2, Severity.WARNING, "Right hand tremor ~5.2 Hz detected", 0.4),
    R("gait", "cadence_spm", 96, msg="Walking cadence ~96 steps/min"),
]

srv = CompanionServer(port=8770)
srv.start()
for t in ["Good afternoon, Margaret! Lovely to see you.",
          "It feels a little cool today — a cozy sweater might keep you comfortable."]:
    srv.publish(t)
    time.sleep(0.4)

try:
    while True:
        srv.publish_data(SNAP, fps=22.4, greeting="Good afternoon, Margaret!")
        time.sleep(1.0)
except KeyboardInterrupt:
    srv.stop()
