"""Process-local native runtime budgets for mixed ML workloads."""
from __future__ import annotations

import os
import sys
import threading

_LOCK = threading.Lock()
_STATE = {"configured": False, "profile": "maximum", "logical_cpus": 1,
          "opencv_threads": None, "torch_threads": None,
          "tensorflow_intra_threads": None, "tensorflow_inter_threads": None,
          "errors": []}


def configure_environment(profile: str = "maximum") -> dict:
    """Install conservative defaults before OpenCV/TF/Torch/JAX imports."""
    logical = max(1, int(os.cpu_count() or 1))
    profile = str(profile or "maximum").strip().lower()
    if profile not in {"maximum", "balanced", "realtime"}:
        profile = "maximum"
    native = max(1, min(2 if profile == "maximum" else 3, logical // 4 or 1))
    defaults = {
        "OMP_NUM_THREADS": str(native), "MKL_NUM_THREADS": str(native),
        "OPENBLAS_NUM_THREADS": str(native), "NUMEXPR_NUM_THREADS": str(native),
        "TF_NUM_INTRAOP_THREADS": str(native), "TF_NUM_INTEROP_THREADS": "1",
        "TORCH_NUM_THREADS": str(native), "TOKENIZERS_PARALLELISM": "false",
    }
    respect_existing = os.environ.get("APP_RESPECT_NATIVE_THREAD_ENV") == "1"
    for name, value in defaults.items():
        if not respect_existing or name not in os.environ:
            os.environ[name] = value
    with _LOCK:
        _STATE.update({"configured": True, "profile": profile,
                       "logical_cpus": logical})
    return diagnostics()


def apply_loaded_limits(profile: str = "maximum") -> dict:
    """Apply equivalent APIs to libraries already imported in this process."""
    configure_environment(profile)
    errors: list[str] = []
    native = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            cv2.setNumThreads(native)
            if hasattr(cv2, "ocl"):
                cv2.ocl.setUseOpenCL(False)
            with _LOCK:
                _STATE["opencv_threads"] = native
        except Exception as exc:  # noqa: BLE001
            errors.append(f"opencv:{type(exc).__name__}")
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(native)
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
            with _LOCK:
                _STATE["torch_threads"] = native
        except Exception as exc:  # noqa: BLE001
            errors.append(f"torch:{type(exc).__name__}")
    tensorflow = sys.modules.get("tensorflow")
    if tensorflow is not None:
        try:
            tensorflow.config.threading.set_intra_op_parallelism_threads(native)
            tensorflow.config.threading.set_inter_op_parallelism_threads(1)
            with _LOCK:
                _STATE["tensorflow_intra_threads"] = native
                _STATE["tensorflow_inter_threads"] = 1
        except RuntimeError:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tensorflow:{type(exc).__name__}")
    with _LOCK:
        _STATE["errors"] = errors[-8:]
    return diagnostics()


def diagnostics() -> dict:
    with _LOCK:
        state = dict(_STATE)
    state["environment"] = {name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "TORCH_NUM_THREADS")}
    state["status"] = "degraded" if state.get("errors") else "healthy"
    return state
