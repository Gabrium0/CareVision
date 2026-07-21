"""Close-up-only local skin classification with conservative abstention.

The local model is an experimental source of private corroborating evidence.
It never emits user-facing disease names and never treats uncertainty as a
normal result.  PyTorch is the reference backend; TensorRT is an optional
Jetson deployment backend built from the same pinned model and processor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import threading
import time
from typing import Any

import cv2
import numpy as np

from core.runtime_resources import apply_loaded_limits


DEFAULT_MODEL = "LaurianeMD/vit-skin-disease"
DEFAULT_REVISION = "1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c"


def _normal_label(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("classifier logits must be finite and non-empty")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration temperature must be positive")
    values = values / temperature
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def _normalized_entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return entropy / math.log(probs.size)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for deployment metadata."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SkinCalibration:
    """Validated temperature/abstention settings for one target label."""

    model_revision: str
    target: str
    temperature: float = 1.0
    accept_threshold: float = 1.0
    max_normalized_entropy: float = 0.0
    max_unknown_probability: float = 0.0
    validated: bool = False

    @classmethod
    def load(cls, path: str | Path | None, revision: str, target: str) -> "SkinCalibration":
        if not path or not Path(path).is_file():
            return cls(revision, target)
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
            raise ValueError("skin calibration must use schema version 1")
        if str(raw.get("model_revision", "")) != revision:
            raise ValueError("skin calibration model revision does not match")
        targets = raw.get("targets")
        values = targets.get(target) if isinstance(targets, dict) else None
        if not isinstance(values, dict):
            raise ValueError(f"skin calibration has no target {target!r}")
        calibration = cls(
            revision,
            target,
            float(values.get("temperature", 1.0)),
            float(values.get("accept_threshold", 1.0)),
            float(values.get("max_normalized_entropy", 0.0)),
            float(values.get("max_unknown_probability", 0.0)),
            bool(raw.get("validated", False)),
        )
        for name, value in (
            ("accept_threshold", calibration.accept_threshold),
            ("max_normalized_entropy", calibration.max_normalized_entropy),
            ("max_unknown_probability", calibration.max_unknown_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"skin calibration {name} must be between 0 and 1")
        if not math.isfinite(calibration.temperature) or calibration.temperature <= 0:
            raise ValueError("skin calibration temperature must be positive")
        return calibration


@dataclass(frozen=True)
class LocalSkinLabelScore:
    """One experimental, uncalibrated score from the full model output."""

    label: str
    probability: float

    def private_value(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "probability": round(max(0.0, min(1.0, float(self.probability))), 6),
            "calibrated": False,
        }


@dataclass(frozen=True)
class LocalSkinPrediction:
    """Bounded private result shared by PyTorch and TensorRT backends."""

    status: str
    backend: str
    model: str
    model_revision: str
    target: str
    probability: float
    calibrated: bool
    abstain_reason: str | None
    inference_ms: float
    label_scores: tuple[LocalSkinLabelScore, ...] = ()
    score_type: str = "full_22_class_softmax"

    @classmethod
    def unavailable(cls, backend: str, model: str, revision: str, target: str,
                    reason: str) -> "LocalSkinPrediction":
        return cls("unavailable", backend, model, revision, target, 0.0,
                   False, str(reason)[:80], 0.0)

    def private_value(self) -> dict[str, Any]:
        """Return the only classifier fields allowed into private Results."""
        return {
            "status": self.status,
            "backend": self.backend,
            "model": self.model,
            "model_revision": self.model_revision,
            "target": self.target,
            "probability": round(max(0.0, min(1.0, float(self.probability))), 6),
            "calibrated": bool(self.calibrated),
            "label_scores": [score.private_value() for score in self.label_scores],
            "score_type": self.score_type,
            "abstain_reason": self.abstain_reason,
            "inference_ms": round(max(0.0, float(self.inference_ms)), 3),
        }


class LocalSkinClassifier(ABC):
    """Common lifecycle and abstention policy for local inference engines."""

    backend = "unknown"

    def __init__(self, *, model: str = DEFAULT_MODEL,
                 revision: str = DEFAULT_REVISION,
                 target_labels: list[str] | tuple[str, ...] = ("vitiligo",),
                 calibration_file: str | None = None,
                 min_sharpness: float = 25.0,
                 min_brightness: float = 25.0,
                 max_brightness: float = 230.0,
                 max_clipped_fraction: float = 0.55,
                 **_: Any):
        targets = [_normal_label(item) for item in target_labels if _normal_label(item)]
        if targets != ["vitiligo"]:
            raise ValueError("the first local skin milestone supports only target_labels: [vitiligo]")
        self.model = str(model)
        self.revision = str(revision)
        self.target = targets[0]
        self.calibration_file = calibration_file
        self.min_sharpness = float(min_sharpness)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)
        self.max_clipped_fraction = float(max_clipped_fraction)
        self._calibration = SkinCalibration.load(calibration_file, self.revision, self.target)
        self._processor = None
        self._labels: dict[int, str] = {}
        self._target_index: int | None = None
        self._unknown_indices: tuple[int, ...] = ()
        self._ready = False
        self._load_ms = 0.0
        self._last_ms = 0.0
        self._last_status = "unconfigured"
        self._last_error: str | None = None
        self._latencies: deque[float] = deque(maxlen=100)
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    def _set_labels(self, labels: dict[Any, Any]) -> None:
        parsed = {int(index): str(label) for index, label in labels.items()}
        if set(parsed) != set(range(len(parsed))):
            raise ValueError("model id2label indexes must be contiguous from zero")
        matches = [index for index, label in parsed.items()
                   if _normal_label(label) == self.target]
        if len(matches) != 1:
            raise ValueError(f"model must contain exactly one {self.target!r} label")
        unknown = [index for index, label in parsed.items()
                   if "unknown" in _normal_label(label) or _normal_label(label) == "normal"]
        self._labels = parsed
        self._target_index = matches[0]
        self._unknown_indices = tuple(unknown)

    def _quality_reason(self, image: np.ndarray) -> str | None:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
            return "invalid_image"
        if image.shape[2] != 3 or min(image.shape[:2]) < 64:
            return "insufficient_closeup"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < self.min_brightness:
            return "underexposed"
        if brightness > self.max_brightness:
            return "overexposed"
        clipped = float(np.mean((gray <= 3) | (gray >= 252)))
        if clipped > self.max_clipped_fraction:
            return "exposure_clipping"
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.min_sharpness:
            return "poor_focus"
        return None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        if self._processor is None:
            raise RuntimeError("skin classifier processor is not loaded")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = self._processor(images=Image.fromarray(rgb), return_tensors="np")
        pixels = np.asarray(batch["pixel_values"], dtype=np.float32)
        if pixels.ndim != 4 or pixels.shape[0] != 1:
            raise ValueError("skin processor returned an invalid tensor")
        return np.ascontiguousarray(pixels)

    def _prediction(self, logits: np.ndarray, started: float) -> LocalSkinPrediction:
        if self._target_index is None:
            raise RuntimeError("skin classifier labels are not loaded")
        probabilities = _softmax(logits, self._calibration.temperature)
        if probabilities.size != len(self._labels):
            raise ValueError("skin classifier output count does not match labels")
        target_probability = float(probabilities[self._target_index])
        label_scores = tuple(
            LocalSkinLabelScore(self._labels[index], float(probabilities[index]))
            for index in sorted(
                (index for index in self._labels if index != self._target_index),
                key=lambda index: (-float(probabilities[index]), index),
            )
        )
        top_index = int(np.argmax(probabilities))
        unknown_probability = max(
            (float(probabilities[index]) for index in self._unknown_indices), default=0.0)
        entropy = _normalized_entropy(probabilities)
        reason = None
        if not self._calibration.validated:
            reason = "uncalibrated"
        elif top_index in self._unknown_indices:
            reason = "unknown_or_normal"
        elif top_index != self._target_index:
            reason = "unsupported_top_label"
        elif unknown_probability > self._calibration.max_unknown_probability:
            reason = "unknown_probability"
        elif entropy > self._calibration.max_normalized_entropy:
            reason = "out_of_distribution"
        elif target_probability < self._calibration.accept_threshold:
            reason = "low_confidence"
        elapsed = (time.perf_counter() - started) * 1000.0
        status = "accepted" if reason is None else "abstained"
        with self._lock:
            self._last_ms = elapsed
            self._latencies.append(elapsed)
            self._last_status = status
            self._last_error = None
        return LocalSkinPrediction(
            status, self.backend, self.model, self.revision, self.target,
            target_probability, self._calibration.validated, reason, elapsed,
            label_scores)

    def predict(self, image: np.ndarray) -> LocalSkinPrediction:
        """Run one close-up inference or return a conservative abstention."""
        started = time.perf_counter()
        reason = self._quality_reason(image)
        if reason is not None:
            result = LocalSkinPrediction(
                "abstained", self.backend, self.model, self.revision, self.target,
                0.0, self._calibration.validated, reason,
                (time.perf_counter() - started) * 1000.0)
            with self._lock:
                self._last_ms = result.inference_ms
                self._latencies.append(result.inference_ms)
                self._last_status = result.status
                self._last_error = None
            return result
        if not self.ready:
            return LocalSkinPrediction.unavailable(
                self.backend, self.model, self.revision, self.target, "model_not_ready")
        try:
            return self._prediction(self._infer_logits(self._preprocess(image)), started)
        except BaseException as exc:  # keep an optional model non-fatal
            detail = f"{type(exc).__name__}: {str(exc).strip()}"[:160]
            with self._lock:
                self._last_status = "unavailable"
                self._last_error = detail
            return LocalSkinPrediction.unavailable(
                self.backend, self.model, self.revision, self.target, type(exc).__name__)

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            p95 = (float(np.percentile(list(self._latencies), 95))
                   if self._latencies else 0.0)
            return {
                "backend": self.backend,
                "model": self.model,
                "revision": self.revision,
                "target": self.target,
                "ready": self._ready,
                "calibrated": self._calibration.validated,
                "load_ms": round(self._load_ms, 3),
                "last_inference_ms": round(self._last_ms, 3),
                "p95_inference_ms": round(p95, 3),
                "latency_target_ms": 500.0,
                "latency_target_met": bool(not self._latencies or p95 < 500.0),
                "last_status": self._last_status,
                "last_error": self._last_error,
            }

    @abstractmethod
    def preload(self) -> None:
        """Load processor, labels, and runtime resources."""

    @abstractmethod
    def _infer_logits(self, pixels: np.ndarray) -> np.ndarray:
        """Return one unnormalized logit vector."""

    def close(self) -> None:
        """Release backend resources when supported."""


class TorchSkinClassifier(LocalSkinClassifier):
    """Full-precision reference implementation used during development."""

    backend = "pytorch"

    def __init__(self, *, device: str = "auto", download_if_missing: bool = False,
                 cache_dir: str | None = None,
                 **params: Any):
        super().__init__(**params)
        self.device = str(device)
        self.download_if_missing = bool(download_if_missing)
        self.cache_dir = cache_dir
        self._torch = None
        self._model_instance = None
        self._resolved_device = "uninitialized"

    def preload(self) -> None:
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            apply_loaded_limits("maximum")
            local_only_attempts = (True, False) if self.download_if_missing else (True,)
            last_error: BaseException | None = None
            for local_only in local_only_attempts:
                try:
                    processor = AutoImageProcessor.from_pretrained(
                        self.model, revision=self.revision, local_files_only=local_only,
                        cache_dir=self.cache_dir)
                    model = AutoModelForImageClassification.from_pretrained(
                        self.model, revision=self.revision, local_files_only=local_only,
                        cache_dir=self.cache_dir)
                    break
                except BaseException as exc:
                    last_error = exc
            else:
                raise RuntimeError("pinned skin model is not cached") from last_error
            resolved = self.device
            if resolved == "auto":
                resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
            model.to(resolved)
            model.eval()
            self._set_labels(model.config.id2label)
            with self._lock:
                self._torch = torch
                self._processor = processor
                self._model_instance = model
                self._resolved_device = resolved
                self._ready = True
                self._last_status = "ready"
        except BaseException as exc:
            detail = f"{type(exc).__name__}: {str(exc).strip()}"[:160]
            with self._lock:
                self._ready = False
                self._last_status = "unavailable"
                self._last_error = detail
            raise
        finally:
            with self._lock:
                self._load_ms = (time.perf_counter() - started) * 1000.0

    def _infer_logits(self, pixels: np.ndarray) -> np.ndarray:
        torch = self._torch
        model = self._model_instance
        if torch is None or model is None:
            raise RuntimeError("PyTorch skin model is not loaded")
        try:
            tensor = torch.from_numpy(pixels).to(self._resolved_device)
            with torch.inference_mode():
                return model(pixel_values=tensor).logits.detach().float().cpu().numpy()[0]
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or self._resolved_device == "cpu":
                raise
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._resolved_device = "cpu"
            tensor = torch.from_numpy(pixels)
            with torch.inference_mode():
                return model(pixel_values=tensor).logits.detach().float().cpu().numpy()[0]

    def diagnostics(self) -> dict[str, Any]:
        values = super().diagnostics()
        values["device"] = self._resolved_device
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            values["cuda_allocated_mib"] = round(torch.cuda.memory_allocated() / 1048576, 3)
            values["cuda_reserved_mib"] = round(torch.cuda.memory_reserved() / 1048576, 3)
        return values

    def close(self) -> None:
        with self._lock:
            self._model_instance = None
            self._processor = None
            self._ready = False


class TensorRTSkinClassifier(LocalSkinClassifier):
    """FP16 TensorRT runtime for an engine built on the target Jetson."""

    backend = "tensorrt"

    def __init__(self, *, tensorrt_engine: str, metadata_file: str | None = None,
                 **params: Any):
        super().__init__(**params)
        self.engine_path = Path(tensorrt_engine)
        self.metadata_path = Path(metadata_file) if metadata_file else Path(
            str(self.engine_path) + ".json")
        self._torch = None
        self._trt = None
        self._engine = None
        self._context = None
        self._input_name = "pixel_values"
        self._output_name = "logits"
        self._input_dtype = None
        self._output_dtype = None

    def preload(self) -> None:
        started = time.perf_counter()
        try:
            if not self.engine_path.is_file() or not self.metadata_path.is_file():
                raise FileNotFoundError("TensorRT skin engine or metadata is missing")
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if metadata.get("model") != self.model:
                raise ValueError("TensorRT engine model does not match configuration")
            if metadata.get("model_revision") != self.revision:
                raise ValueError("TensorRT engine revision does not match configuration")
            expected_engine = metadata.get("engine_sha256")
            if expected_engine and file_sha256(self.engine_path) != expected_engine:
                raise ValueError("TensorRT engine hash does not match metadata")
            labels = metadata.get("id2label")
            if not isinstance(labels, dict):
                raise ValueError("TensorRT metadata must contain id2label")
            import tensorrt as trt
            import torch
            from transformers import AutoImageProcessor, ViTImageProcessor

            if not torch.cuda.is_available():
                raise RuntimeError("TensorRT skin backend requires CUDA")
            processor_config = metadata.get("processor_config")
            if isinstance(processor_config, dict):
                processor = ViTImageProcessor.from_dict(processor_config)
            else:  # backwards-compatible with early metadata files
                processor = AutoImageProcessor.from_pretrained(
                    self.model, revision=self.revision, local_files_only=True)
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
            if engine is None:
                raise RuntimeError("could not deserialize TensorRT skin engine")
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError("could not create TensorRT execution context")
            self._input_name = str(metadata.get("input_name", "pixel_values"))
            self._output_name = str(metadata.get("output_name", "logits"))
            if hasattr(engine, "get_tensor_dtype"):
                input_trt_dtype = engine.get_tensor_dtype(self._input_name)
                output_trt_dtype = engine.get_tensor_dtype(self._output_name)
            else:
                input_trt_dtype = engine.get_binding_dtype(
                    engine.get_binding_index(self._input_name))
                output_trt_dtype = engine.get_binding_dtype(
                    engine.get_binding_index(self._output_name))
            dtype_map = {
                np.dtype(np.float16): torch.float16,
                np.dtype(np.float32): torch.float32,
            }
            input_dtype = dtype_map.get(np.dtype(trt.nptype(input_trt_dtype)))
            output_dtype = dtype_map.get(np.dtype(trt.nptype(output_trt_dtype)))
            if input_dtype is None or output_dtype is None:
                raise ValueError("TensorRT skin engine must use float16 or float32 tensors")
            self._set_labels(labels)
            with self._lock:
                self._torch = torch
                self._trt = trt
                self._processor = processor
                self._engine = engine
                self._context = context
                self._input_dtype = input_dtype
                self._output_dtype = output_dtype
                self._ready = True
                self._last_status = "ready"
        except BaseException as exc:
            with self._lock:
                self._ready = False
                self._last_status = "unavailable"
                self._last_error = f"{type(exc).__name__}: {str(exc).strip()}"[:160]
            raise
        finally:
            with self._lock:
                self._load_ms = (time.perf_counter() - started) * 1000.0

    def _infer_logits(self, pixels: np.ndarray) -> np.ndarray:
        torch, engine, context = self._torch, self._engine, self._context
        if torch is None or engine is None or context is None:
            raise RuntimeError("TensorRT skin engine is not loaded")
        input_tensor = torch.from_numpy(pixels).cuda().to(self._input_dtype).contiguous()
        output_tensor = torch.empty((1, len(self._labels)), device="cuda",
                                    dtype=self._output_dtype)
        stream = torch.cuda.current_stream().cuda_stream
        if hasattr(context, "set_tensor_address"):
            context.set_input_shape(self._input_name, tuple(input_tensor.shape))
            context.set_tensor_address(self._input_name, int(input_tensor.data_ptr()))
            context.set_tensor_address(self._output_name, int(output_tensor.data_ptr()))
            if not context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
        else:  # TensorRT 8.x as shipped by older JetPack releases
            input_index = engine.get_binding_index(self._input_name)
            output_index = engine.get_binding_index(self._output_name)
            context.set_binding_shape(input_index, tuple(input_tensor.shape))
            bindings = [0] * engine.num_bindings
            bindings[input_index] = int(input_tensor.data_ptr())
            bindings[output_index] = int(output_tensor.data_ptr())
            if not context.execute_async_v2(bindings=bindings, stream_handle=stream):
                raise RuntimeError("TensorRT execute_async_v2 failed")
        torch.cuda.current_stream().synchronize()
        return output_tensor.detach().cpu().numpy()[0]

    def diagnostics(self) -> dict[str, Any]:
        values = super().diagnostics()
        values.update({"engine": str(self.engine_path), "metadata": str(self.metadata_path)})
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            values["cuda_allocated_mib"] = round(torch.cuda.memory_allocated() / 1048576, 3)
            values["cuda_reserved_mib"] = round(torch.cuda.memory_reserved() / 1048576, 3)
        return values

    def close(self) -> None:
        with self._lock:
            self._context = None
            self._engine = None
            self._processor = None
            self._ready = False


class AutoSkinClassifier(LocalSkinClassifier):
    """Prefer TensorRT on Jetson and fall back to the reference backend."""

    backend = "auto"

    def __init__(self, **params: Any):
        super().__init__(**params)
        self._params = dict(params)
        self._delegate: LocalSkinClassifier | None = None
        self._fallback_reason: str | None = None

    def preload(self) -> None:
        candidates: list[LocalSkinClassifier] = []
        engine = self._params.get("tensorrt_engine")
        is_jetson = platform.machine().lower() in {"aarch64", "arm64"}
        if is_jetson and engine:
            candidates.append(TensorRTSkinClassifier(**self._params))
        candidates.append(TorchSkinClassifier(**self._params))
        errors = []
        for candidate in candidates:
            try:
                candidate.preload()
                self._fallback_reason = ",".join(errors)[:160] or None
                self._delegate = candidate
                self.backend = candidate.backend
                self._processor = candidate._processor
                self._labels = candidate._labels
                self._target_index = candidate._target_index
                self._unknown_indices = candidate._unknown_indices
                with self._lock:
                    self._ready = True
                    self._last_status = "ready"
                    self._load_ms = candidate.diagnostics().get("load_ms", 0.0)
                return
            except BaseException as exc:
                errors.append(f"{candidate.backend}:{type(exc).__name__}")
                candidate.close()
        self._fallback_reason = ",".join(errors)[:160]
        raise RuntimeError(f"no local skin backend available ({self._fallback_reason})")

    def _infer_logits(self, pixels: np.ndarray) -> np.ndarray:
        if self._delegate is None:
            raise RuntimeError("automatic skin backend is not loaded")
        return self._delegate._infer_logits(pixels)

    def predict(self, image: np.ndarray) -> LocalSkinPrediction:
        if self._delegate is None:
            return LocalSkinPrediction.unavailable(
                "auto", self.model, self.revision, self.target, "model_not_ready")
        return self._delegate.predict(image)

    def diagnostics(self) -> dict[str, Any]:
        values = self._delegate.diagnostics() if self._delegate else super().diagnostics()
        values["requested_backend"] = "auto"
        values["fallback_reason"] = self._fallback_reason
        return values

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()
        with self._lock:
            self._ready = False


def build_local_skin_classifier(config: dict[str, Any] | None) -> LocalSkinClassifier | None:
    """Create the configured backend without importing optional ML runtimes."""
    values = dict(config or {})
    if not values.get("enabled", False):
        return None
    backend = str(values.pop("backend", "auto")).strip().lower()
    values.pop("enabled", None)
    values.pop("mode", None)
    if backend == "auto":
        return AutoSkinClassifier(**values)
    if backend == "pytorch":
        return TorchSkinClassifier(**values)
    if backend == "tensorrt":
        return TensorRTSkinClassifier(**values)
    raise ValueError(f"unknown local skin backend {backend!r}")
