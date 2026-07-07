"""Generate a short synthetic clip with a moving face+body to exercise the
real Camera -> Pipeline path in main.py (no webcam needed)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test import make_face_frame  # reuse the synthetic renderer

out = str(Path(__file__).resolve().parent / "synthetic_clip.mp4")
w, h, fps, n = 640, 480, 20, 80
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(n):
    frame = make_face_frame(w, h, t=i / fps)
    # gentle full-body sway so motion/pose-ish signals have something
    M = np.float32([[1, 0, 8 * np.sin(i / 6)], [0, 1, 0]])
    frame = cv2.warpAffine(frame, M, (w, h))
    vw.write(frame)
vw.release()
print(f"wrote {out} ({n} frames @ {fps}fps)")
