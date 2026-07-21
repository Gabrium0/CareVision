"""Offline contracts for local close-up skin classification and fusion."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from agent.skin_dialogue import SkinDialogue, speech_mentions_hypothesis
from modules.local_skin_classifier import (
    DEFAULT_REVISION,
    LocalSkinClassifier,
    LocalSkinPrediction,
    SkinCalibration,
    TensorRTSkinClassifier,
    build_local_skin_classifier,
)
from modules.skin_vision import SkinAnalysis, SkinVision


def _calibration(tmp_path, *, validated=True):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "version": 1,
        "validated": validated,
        "model_revision": DEFAULT_REVISION,
        "targets": {"vitiligo": {
            "temperature": 1.0,
            "accept_threshold": 0.70,
            "max_normalized_entropy": 0.90,
            "max_unknown_probability": 0.20,
        }},
    }), encoding="utf-8")
    return str(path)


class _FakeClassifier(LocalSkinClassifier):
    backend = "fake"

    def __init__(self, logits, **params):
        super().__init__(**params)
        self.logits = np.asarray(logits, dtype=np.float32)
        self.calls = 0

    def preload(self):
        self._set_labels({0: "Vitiligo", 1: "Unknown Normal", 2: "Eczema"})
        self._processor = object()
        self._ready = True

    def _preprocess(self, image):
        return np.zeros((1, 3, 224, 224), np.float32)

    def _infer_logits(self, pixels):
        self.calls += 1
        return self.logits


def _image():
    return np.random.default_rng(4).integers(30, 225, (128, 128, 3), dtype=np.uint8)


def _analysis(features=("discoloration",)):
    return SkinAnalysis("good", True, True, tuple(features), "left forearm",
                        0.82, ("uncertain pigment condition",),
                        ("duration", "spreading"))


def test_calibration_requires_matching_revision_and_bounds(tmp_path):
    path = _calibration(tmp_path)
    calibration = SkinCalibration.load(path, DEFAULT_REVISION, "vitiligo")
    assert calibration.validated is True
    raw = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    raw["model_revision"] = "wrong"
    (tmp_path / "calibration.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        SkinCalibration.load(path, DEFAULT_REVISION, "vitiligo")


def test_shared_policy_accepts_only_calibrated_target_and_never_normalizes_unknown(tmp_path):
    accepted = _FakeClassifier(
        [6.0, 0.0, 0.0], calibration_file=_calibration(tmp_path),
        target_labels=["vitiligo"], min_sharpness=0.0)
    accepted.preload()
    result = accepted.predict(_image())
    assert result.status == "accepted"
    assert result.target == "vitiligo"
    assert result.calibrated is True

    unknown = _FakeClassifier(
        [0.0, 6.0, 0.0], calibration_file=_calibration(tmp_path),
        target_labels=["vitiligo"], min_sharpness=0.0)
    unknown.preload()
    result = unknown.predict(_image())
    assert result.status == "abstained"
    assert result.abstain_reason == "unknown_or_normal"


def test_private_scores_include_every_non_target_label_without_renormalizing(tmp_path):
    model = _FakeClassifier(
        [2.0, 1.0, 3.0], calibration_file=_calibration(tmp_path),
        target_labels=["vitiligo"], min_sharpness=0.0)
    model.preload()

    result = model.predict(_image())
    private = result.private_value()
    assert [item["label"] for item in private["label_scores"]] == [
        "Eczema", "Unknown Normal"]
    assert all(item["calibrated"] is False for item in private["label_scores"])
    assert private["score_type"] == "full_22_class_softmax"
    assert "Vitiligo" not in [item["label"] for item in private["label_scores"]]
    total = result.probability + sum(score.probability for score in result.label_scores)
    assert total == pytest.approx(1.0)


def test_label_scores_follow_id2label_indexes_not_mapping_order(tmp_path):
    model = _FakeClassifier(
        [0.0, 2.0, 1.0], calibration_file=_calibration(tmp_path),
        target_labels=["vitiligo"], min_sharpness=0.0)
    model._set_labels({"2": "Eczema", "0": "Vitiligo", "1": "Unknown Normal"})
    model._processor = object()
    model._ready = True

    result = model.predict(_image())
    assert [score.label for score in result.label_scores] == [
        "Unknown Normal", "Eczema"]


def test_all_model_one_non_vitiligo_labels_are_retained(tmp_path):
    labels = [
        "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous",
        "Candidiasis", "Drug Eruption", "Eczema", "Infestations/Bites",
        "Lichen", "Lupus", "Moles", "Psoriasis", "Rosacea",
        "Seborrheic Keratoses", "Skin Cancer", "Sun/Sunlight Damage",
        "Tinea", "Unknown Normal", "Vascular Tumors", "Vasculitis",
        "Vitiligo", "Warts",
    ]
    model = _FakeClassifier(
        np.arange(len(labels), dtype=np.float32),
        calibration_file=_calibration(tmp_path), target_labels=["vitiligo"],
        min_sharpness=0.0)
    model._set_labels({str(index): label for index, label in enumerate(labels)})
    model._processor = object()
    model._ready = True

    result = model.predict(_image())
    assert len(result.label_scores) == 21
    assert {score.label for score in result.label_scores} == set(labels) - {"Vitiligo"}
    assert result.label_scores[0].label == "Warts"


def test_label_scores_are_empty_when_inference_did_not_run():
    model = _FakeClassifier([1.0, 0.0, 0.0], target_labels=["vitiligo"])
    result = model.predict(np.zeros((128, 128, 3), np.uint8))
    assert result.abstain_reason == "underexposed"
    assert result.private_value()["label_scores"] == []


def test_uncalibrated_and_poor_closeups_abstain(tmp_path):
    model = _FakeClassifier([8.0, 0.0, 0.0], target_labels=["vitiligo"])
    model.preload()
    assert model.predict(_image()).abstain_reason == "uncalibrated"
    dark = np.zeros((128, 128, 3), np.uint8)
    assert model.predict(dark).abstain_reason == "underexposed"
    assert model.calls == 1


def test_factory_rejects_broad_targets_and_tensorrt_revision_mismatch(tmp_path):
    with pytest.raises(ValueError, match="vitiligo"):
        build_local_skin_classifier({
            "enabled": True, "backend": "pytorch",
            "target_labels": ["vitiligo", "eczema"],
        })
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    metadata = tmp_path / "model.engine.json"
    metadata.write_text(json.dumps({
        "model": "LaurianeMD/vit-skin-disease",
        "model_revision": "wrong", "id2label": {"0": "Vitiligo"},
    }), encoding="utf-8")
    backend = TensorRTSkinClassifier(
        tensorrt_engine=str(engine), metadata_file=str(metadata),
        target_labels=["vitiligo"])
    with pytest.raises(ValueError, match="revision"):
        backend.preload()


def test_fusion_keeps_vitiligo_private_and_public_wording_neutral(tmp_path, monkeypatch):
    local = _FakeClassifier(
        [6.0, 0.0, 0.0], calibration_file=_calibration(tmp_path),
        target_labels=["vitiligo"], min_sharpness=0.0)
    local.preload()
    monkeypatch.setattr("modules.skin_vision.build_local_skin_classifier",
                        lambda config: local)
    module = SkinVision(
        consent=False,
        local_classifier={"enabled": True, "mode": "screening"})
    try:
        module._local_ready = True
        prediction = local.predict(_image())
        private = module._private_analysis(_analysis(), prediction)
        assert private["local_fusion"] == "corroborated"
        assert "vitiligo" in private["possible_conditions"]
        results = module._local_only_results("guided_closeup", "cid", prediction)
        public = next(result for result in results if result.visibility.value == "public")
        hidden = next(result for result in results if result.visibility.value == "agent_only")
        assert "vitiligo" not in public.message.lower()
        assert "vitiligo" not in json.dumps(public.value).lower()
        assert hidden.value["possible_conditions"] == ["vitiligo"]
        dialogue = SkinDialogue()
        assert speech_mentions_hypothesis("This could be vitiligo.", ["vitiligo"])
    finally:
        module.close()


def test_debug_mode_local_only_never_publishes(tmp_path, monkeypatch):
    local = _FakeClassifier(
        [6.0, 0.0, 0.0], calibration_file=_calibration(tmp_path),
        target_labels=["vitiligo"], min_sharpness=0.0)
    local.preload()
    monkeypatch.setattr("modules.skin_vision.build_local_skin_classifier",
                        lambda config: local)
    module = SkinVision(consent=False,
                        local_classifier={"enabled": True, "mode": "debug"})
    try:
        prediction = local.predict(_image())
        results = module._local_only_results("manual_arm_check", "cid", prediction)
        assert len(results) == 1
        assert results[0].visibility.value == "agent_only"
        assert results[0].key == "local_analysis"
    finally:
        module.close()


def test_hybrid_closeup_fuses_independent_local_and_cloud_results(tmp_path, monkeypatch):
    local = _FakeClassifier(
        [6.0, 0.0, 0.0], calibration_file=_calibration(tmp_path),
        target_labels=["vitiligo"], min_sharpness=0.0)
    local.preload()
    monkeypatch.setattr("modules.skin_vision.build_local_skin_classifier",
                        lambda config: local)
    monkeypatch.setattr("modules.skin_vision.nvidia_api_key", lambda: "secret")
    module = SkinVision(consent=True,
                        local_classifier={"enabled": True, "mode": "screening"})
    try:
        module._local_ready = True
        module._preliminary = _analysis()
        module._correlation_id = "cid"
        monkeypatch.setattr(module, "_call_api", lambda *args, **kwargs: _analysis())
        module._submit(_image(), "closeup", 100.0)
        module._pending.result(timeout=2)
        results = module._consume_pending(101.0)
        public = next(result for result in results if result.key == "visible_skin_change")
        private = next(result for result in results if result.key == "analysis")
        assert public.message == "Possible pigment change on left forearm"
        assert "vitiligo" not in json.dumps(public.value).lower()
        assert private.value["local_fusion"] == "corroborated"
        assert private.value["local_classifier"]["backend"] == "fake"
        assert private.value["local_classifier"]["label_scores"]
        public_json = json.dumps(public.value).lower()
        assert "eczema" not in public_json
        assert "unknown normal" not in public_json
    finally:
        module.close()


def test_cloud_failure_does_not_discard_private_debug_result(tmp_path, monkeypatch):
    local = _FakeClassifier(
        [6.0, 0.0, 0.0], calibration_file=_calibration(tmp_path),
        target_labels=["vitiligo"], min_sharpness=0.0)
    local.preload()
    monkeypatch.setattr("modules.skin_vision.build_local_skin_classifier",
                        lambda config: local)
    monkeypatch.setattr("modules.skin_vision.nvidia_api_key", lambda: "secret")
    module = SkinVision(consent=True,
                        local_classifier={"enabled": True, "mode": "debug"})
    try:
        module._local_ready = True
        monkeypatch.setattr(
            module, "_call_api",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cloud down")))
        module._submit(_image(), "closeup", 100.0, purpose="manual_arm_check")
        module._pending.result(timeout=2)
        results = module._consume_pending(101.0)
        assert any(result.key == "local_analysis" and
                   result.visibility.value == "agent_only" for result in results)
        assert any(result.key == "arm_check" and
                   result.value["status"] == "unavailable" for result in results)
    finally:
        module.close()


def test_async_preload_reports_ready_without_blocking_constructor(monkeypatch):
    local = _FakeClassifier([1.0, 0.0, 0.0], target_labels=["vitiligo"])
    monkeypatch.setattr("modules.skin_vision.build_local_skin_classifier",
                        lambda config: local)
    module = SkinVision(consent=False, local_classifier={"enabled": True})
    try:
        assert local.ready is False
        module.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not module._local_ready:
            module._refresh_local_load()
            time.sleep(0.005)
        assert module._local_ready is True
        assert module.diagnostics()["local_classifier"]["ready"] is True
    finally:
        module.close()


def test_local_classifier_never_runs_on_passive_frames(monkeypatch):
    local = _FakeClassifier([6.0, 0.0, 0.0], target_labels=["vitiligo"],
                            min_sharpness=0.0)
    local.preload()
    monkeypatch.setattr("modules.skin_vision.build_local_skin_classifier",
                        lambda config: local)
    module = SkinVision(consent=False, local_classifier={"enabled": True})
    try:
        module._local_ready = True
        module._local_load_attempted = True
        from core.context import FrameContext
        module.process(FrameContext(_image(), 100.0, 0, 30.0, person_present=True))
        assert local.calls == 0
    finally:
        module.close()
