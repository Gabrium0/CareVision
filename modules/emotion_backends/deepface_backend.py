"""DeepFace emotion (+age/gender) backend (tested; from research.md).

Uses the `deepface` package. Runs on the MediaPipe face crop with detection
skipped (we already have the face). Emotion is the headline; age/gender are
exposed too so the age module can reuse this backend. Self-disables if deepface
(TensorFlow) isn't installed.

DeepFace is comparatively heavy, so inference is throttled and cached.
It runs in a subprocess so the main process can use Keras/JAX for Open-RPPG
while DeepFace uses Keras/TensorFlow.
"""
from __future__ import annotations

import os
import queue
import sys
import time
from multiprocessing import get_context

from core.context import FrameContext
from core.debug import log as debug_log
from modules.backends.base import Backend


def _worker(in_q, out_q, actions):
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["DEEPFACE_WORKER"] = "1"
    for name in list(sys.modules):
        if name == "keras" or name.startswith("keras."):
            sys.modules.pop(name, None)
    try:
        from deepface import DeepFace
        try:
            import keras
            backend = keras.backend.backend()
        except Exception:  # noqa: BLE001
            backend = "unknown"
        out_q.put({"event": "ready", "backend": backend})
    except Exception as e:  # noqa: BLE001
        out_q.put({"event": "error", "error": f"load failed: {type(e).__name__}: {e}"})
        return
    while True:
        try:
            item = in_q.get()
        except (KeyboardInterrupt, EOFError):
            return
        if item is None:
            return
        try:
            t0 = time.time()
            res = DeepFace.analyze(item, actions=actions, enforce_detection=False,
                                   detector_backend="skip", silent=True)
            latency_ms = (time.time() - t0) * 1000.0
            if isinstance(res, list):
                res = res[0] if res else {}
            out = {}
            if "dominant_emotion" in res:
                out["emotion"] = str(res["dominant_emotion"]).lower()
                scores = res.get("emotion", {})
                top = max(scores.values()) if scores else 100.0
                out["confidence"] = round(float(top) / 100.0, 2)
            if "age" in res:
                out["age"] = int(res["age"])
            if "dominant_gender" in res:
                out["gender"] = str(res["dominant_gender"]).lower()
            out_q.put({"event": "result", "result": out, "latency_ms": latency_ms})
        except Exception as e:  # noqa: BLE001
            out_q.put({"event": "error", "error": f"inference failed: {e}"})


class DeepFaceBackend(Backend):
    """DeepFace emotion (+age/gender) backend in a TensorFlow subprocess."""
    label = "deepface"

    def __init__(self, actions=("emotion",), infer_every: float = 1.5):
        self.available = False
        self._df = None
        self.actions = list(actions)
        self.infer_every = infer_every
        self._crop = None
        self._last = 0.0
        self._cached = None
        self.status = "waiting for face"
        self._errors = 0
        self._last_latency_ms = 0.0
        self._ctx = get_context("spawn")
        self._in_q = None
        self._out_q = None
        self._proc = None
        self.available = True

    def _ensure_worker(self) -> None:
        if self._proc is not None:
            return
        try:
            old_backend = os.environ.get("KERAS_BACKEND")
            old_worker = os.environ.get("DEEPFACE_WORKER")
            os.environ["KERAS_BACKEND"] = "tensorflow"
            os.environ["DEEPFACE_WORKER"] = "1"
            self._in_q = self._ctx.Queue(maxsize=1)
            self._out_q = self._ctx.Queue()
            self._proc = self._ctx.Process(target=_worker, args=(self._in_q, self._out_q, self.actions),
                                           daemon=True)
            self._proc.start()
            if old_backend is None:
                os.environ.pop("KERAS_BACKEND", None)
            else:
                os.environ["KERAS_BACKEND"] = old_backend
            if old_worker is None:
                os.environ.pop("DEEPFACE_WORKER", None)
            else:
                os.environ["DEEPFACE_WORKER"] = old_worker
            self.status = "starting"
            print(f"[emotion/deepface] starting worker (actions={self.actions})")
        except Exception as e:  # noqa: BLE001
            if old_backend is None:
                os.environ.pop("KERAS_BACKEND", None)
            else:
                os.environ["KERAS_BACKEND"] = old_backend
            if old_worker is None:
                os.environ.pop("DEEPFACE_WORKER", None)
            else:
                os.environ["DEEPFACE_WORKER"] = old_worker
            self.status = "unavailable"
            print(f"[emotion/deepface] unavailable ({type(e).__name__}: {e})")

    def update(self, ctx: FrameContext) -> None:
        """Feed one frame's data into the backend's rolling state."""
        if self.available and ctx.face is not None:
            crop = ctx.face.crop
            if crop is not None and crop.size:
                self._crop = crop            # BGR, DeepFace's expected order
                self._ensure_worker()
        debug_log("deepface", f"status={self.status} proc={getattr(self, '_proc', None) is not None} "
                              f"cached={self._cached} latency_ms={self._last_latency_ms:.0f}")

    def compute(self) -> dict | None:
        """Return the backend's current reading dict, or None if not ready."""
        if not self.available or self._proc is None:
            return self._cached
        self._drain()
        if self._crop is None:
            return self._cached
        now = time.time()
        if now - self._last < self.infer_every:
            return self._cached
        self._last = now
        try:
            self._in_q.put_nowait(self._crop.copy())
        except queue.Full:
            pass
        return self._cached

    def _drain(self) -> None:
        while True:
            try:
                msg = self._out_q.get_nowait()
            except queue.Empty:
                return
            event = msg.get("event")
            if event == "ready":
                backend = msg.get("backend", "unknown")
                self.status = f"ready:{backend}"
                print(f"[emotion/deepface] worker ready (keras backend={backend})")
            elif event == "result":
                self.status = "ready"
                self._last_latency_ms = float(msg.get("latency_ms") or 0.0)
                result = msg.get("result") or {}
                self._cached = result or self._cached
            elif event == "error":
                self.status = "error"
                self._errors += 1
                if self._errors <= 3:
                    print(f"[emotion/deepface] {msg.get('error')}")
                if self._errors == 3:
                    print("[emotion/deepface] disabling after repeated worker errors")
                    self.available = False

    def close(self) -> None:
        """Release any resources (models, threads, sockets) held here."""
        if not getattr(self, "_proc", None):
            return
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=0.5)
        for q in (self._in_q, self._out_q):
            try:
                q.close()
                q.cancel_join_thread()
            except Exception:  # noqa: BLE001
                pass
