"""Thread-safety + correctness test for the vitals "fast path" added in
core/pipeline.py and modules/heart_rate.py: HeartRate.fast_update() feeding
a backend from one thread while HeartRate.process() (via backend.compute())
reads it from another, as happens when core.camera.Camera's reader thread
drives a live webcam.

Run standalone:  python tests/fast_path_test.py
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.context import FrameContext
from modules.heart_rate import HeartRate
from extractors.face import FaceExtractor
from smoke_test import make_face_frame

FPS = 30.0
BPM = 72.0


def _pulsing_frame(t: float) -> np.ndarray:
    frame = make_face_frame(t=t).astype(np.int16)
    frame[:, :, 1] = np.clip(frame[:, :, 1] + 6.0 * np.sin(2 * np.pi * (BPM / 60.0) * t), 0, 255)
    return frame.astype(np.uint8)


def test_fast_update_flag_and_no_double_feed():
    """process() must stop calling backend.update() once fast_update() has
    fed the backend at least once (else samples get double-counted)."""
    hr = HeartRate(window_seconds=12.0, backends=["classical"])
    be = hr._backends[0]
    face = FaceExtractor()

    ctx = FrameContext(frame=_pulsing_frame(0.0), timestamp=0.0, frame_index=0, fps=FPS)
    face.extract(ctx)
    assert ctx.face is not None, "synthetic frame must produce a detected face"

    assert hr._fast_fed is False
    hr.process(ctx)                      # should have called be.update() itself
    n_after_process = len(be.buf)
    assert n_after_process == 1, f"expected 1 buffered sample, got {n_after_process}"

    hr.fast_update(ctx)                  # simulates the reader thread feeding directly
    assert hr._fast_fed is True
    n_after_fast = len(be.buf)
    assert n_after_fast == 2

    hr.process(ctx)                      # must NOT call update() again now
    n_after_process2 = len(be.buf)
    assert n_after_process2 == 2, (
        f"process() double-fed the backend after fast path engaged: {n_after_process2}")
    face.close()
    hr.close()
    print("[fast-path-test] fast_update flag + no-double-feed OK")


def test_concurrent_update_and_compute():
    """Hammer backend.update() from a writer thread while compute() runs
    concurrently on the main thread, mirroring the reader-thread/heavy-loop
    split; must not raise and must eventually recover ~72 bpm."""
    hr = HeartRate(window_seconds=30.0, backends=["classical"])
    be = hr._backends[0]
    face = FaceExtractor()

    stop = threading.Event()
    errors: list[Exception] = []
    t0 = time.time()

    def writer():
        i = 0
        try:
            while not stop.is_set():
                t = i / FPS
                ctx = FrameContext(frame=_pulsing_frame(t), timestamp=t, frame_index=i, fps=FPS)
                face.extract(ctx)
                if ctx.face is not None:
                    be.update(ctx)
                i += 1
                time.sleep(1.0 / FPS)
                if t > 15.0:
                    stop.set()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()

    last = None
    reads = 0
    while thread.is_alive() or reads < 5:
        try:
            r = be.compute()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            break
        if r:
            last = r
        reads += 1
        time.sleep(0.05)
        if time.time() - t0 > 25.0:
            break
    stop.set()
    thread.join(timeout=5.0)
    face.close()
    hr.close()

    assert not errors, f"concurrent update/compute raised: {errors}"
    assert last is not None, "no reading was ever produced"
    assert abs(last["bpm"] - BPM) < 8.0, f"recovered {last['bpm']} bpm, expected ~{BPM}"
    print(f"[fast-path-test] concurrent update/compute OK -> {last}")


def test_classical_diagnostics_are_safe_during_updates():
    hr = HeartRate(window_seconds=12.0, backends=["classical"])
    backend = hr._backends[0]
    face = FaceExtractor()
    ctx = FrameContext(frame=_pulsing_frame(0.0), timestamp=0.0,
                       frame_index=0, fps=FPS)
    face.extract(ctx)
    assert ctx.face is not None
    errors: list[Exception] = []

    def update_many():
        try:
            for i in range(100):
                ctx.timestamp = i / FPS
                backend.update(ctx)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=update_many)
    worker.start()
    while worker.is_alive():
        snapshot = backend.diagnostics()
        assert 0.0 <= snapshot["progress"] <= 1.0
        assert "latest" in snapshot
        time.sleep(0.001)  # yield so this lock-contention test cannot starve its writer
    worker.join()
    face.close()
    hr.close()
    assert not errors


def main():
    """Run all fast-path tests."""
    test_fast_update_flag_and_no_double_feed()
    test_concurrent_update_and_compute()
    test_classical_diagnostics_are_safe_during_updates()
    print("[fast-path-test] OK")


if __name__ == "__main__":
    main()
