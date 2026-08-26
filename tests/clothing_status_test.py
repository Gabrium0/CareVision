import builtins
import time
from types import SimpleNamespace
from concurrent.futures import Future

import numpy as np

from core.context import FrameContext
from core.shared_signals import SharedSignals
from modules.clothing import Clothing
from modules.clothing_advice import ClothingAdvice


def _ctx(timestamp=100.0):
    return FrameContext(
        frame=np.zeros((240, 320, 3), dtype=np.uint8),
        timestamp=timestamp,
        frame_index=0,
        fps=30.0,
    )


def test_pending_model_load_gets_an_explicit_timeout_status(monkeypatch):
    module = Clothing(load_timeout_seconds=1.0)
    module._load_attempted = True
    module._load_future = Future()
    module._load_started = time.monotonic() - 2.0
    monkeypatch.setattr(module, "_torso_crop", lambda ctx: np.zeros((160, 160, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_sleeve_state", lambda ctx: None)

    results = module.process(_ctx())

    status = next(result for result in results if result.key == "status")
    assert status.value == "clothing model load timed out"
    module._load_future.cancel()
    module.close()


def test_dependency_import_failure_reports_deepest_actionable_cause(monkeypatch):
    module = Clothing()
    real_import = builtins.__import__

    def fail_transformers(name, *args, **kwargs):
        if name == "transformers":
            root = RuntimeError("operator torchvision::nms does not exist")
            raise ModuleNotFoundError("Could not import module 'pipeline'") from root
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_transformers)
    module._load()

    assert module._pipe is None
    assert module._cached["status"] == (
        "clothing dependencies unavailable: torchvision::nms missing")
    module.close()


def test_model_load_does_not_publish_ready_before_first_inference(monkeypatch):
    module = Clothing()
    real_import = builtins.__import__
    fake_pipeline = object()

    def import_transformers(name, *args, **kwargs):
        if name == "transformers":
            return SimpleNamespace(pipeline=lambda *args, **kwargs: fake_pipeline)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_transformers)
    module._load()

    assert module._pipe is fake_pipeline
    assert module._cached["status"] == "clothing model ready; waiting for first frame"
    assert module._cached["clothing"] == "..."
    module.close()


def test_start_submits_preload_before_any_frame(monkeypatch):
    module = Clothing()
    submitted = Future()
    monkeypatch.setattr(module._executor, "submit", lambda fn: submitted)

    module.start()

    assert module._load_future is submitted
    assert module._cached["status"] == "loading clothing model"
    submitted.cancel()
    module.close()


def test_cache_first_loader_downloads_only_primary(monkeypatch):
    module = Clothing(download_if_missing=True)
    calls = []
    fake_pipeline = object()
    real_import = builtins.__import__

    def pipeline(task, model, **kwargs):
        calls.append((task, model, kwargs.get("local_files_only", False)))
        if kwargs.get("local_files_only"):
            raise OSError("not cached")
        return fake_pipeline

    def import_transformers(name, *args, **kwargs):
        if name == "transformers":
            return SimpleNamespace(pipeline=pipeline)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_transformers)
    module._load()

    candidates = module._candidates()
    assert calls[:-1] == [(task, name, True) for task, name in candidates]
    assert calls[-1] == (*candidates[0], False)
    assert module._pipe is fake_pipeline
    module.close()


def test_stop_prevents_fallback_and_download_attempts(monkeypatch):
    module = Clothing(download_if_missing=True)
    calls = []
    real_import = builtins.__import__

    def pipeline(task, model, **kwargs):
        calls.append((task, model, kwargs.get("local_files_only", False)))
        module._stopping.set()
        raise OSError("not cached")

    def import_transformers(name, *args, **kwargs):
        if name == "transformers":
            return SimpleNamespace(pipeline=pipeline)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_transformers)
    module._load()

    assert calls == [(*module._candidates()[0], True)]
    module.close()


def test_late_model_load_recovers_from_timeout(monkeypatch):
    module = Clothing(load_timeout_seconds=1.0)
    module._load_attempted = True
    module._load_future = Future()
    module._load_started = time.monotonic() - 2.0
    monkeypatch.setattr(module, "_torso_crop", lambda ctx: np.zeros((160, 160, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_sleeve_state", lambda ctx: None)

    first = module.process(_ctx())
    assert next(result for result in first if result.key == "status").value == (
        "clothing model load timed out")

    module._set_loaded(object(), "zero-shot-image-classification", "cached")
    module._load_future.set_result(None)
    module._last_infer = time.time()
    second = module.process(_ctx(101.0))

    assert next(result for result in second if result.key == "status").value == (
        "clothing model ready; waiting for first frame")
    module.close()


def test_advice_reads_recent_clothing_from_cross_frame_signal():
    shared = SharedSignals.instance()
    shared.set("clothing", {"clothing": "t-shirt", "confidence": 0.9,
                            "status": "ready"}, now=100.0)
    ctx = _ctx(timestamp=105.0)
    ctx.extras["weather"] = {"feels_like_c": 26.0}

    results = ClothingAdvice().process(ctx)

    detected = next(result for result in results if result.key == "detected_clothing")
    recommendation = next(result for result in results if result.key == "recommendation")
    assert detected.value == "t-shirt"
    assert "still unavailable" not in recommendation.value
