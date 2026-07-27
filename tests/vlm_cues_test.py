"""VLM appearance cues routed onto the detector cards that own their subject.

Two properties matter more than the plumbing. First, provenance: a model's
opinion published under `dry_lips` must stay unmistakably distinct from the
heuristic's own reading, or a client is shown a guess dressed as a
measurement. Second, containment: agent/corroboration.py matches follow-up
rules by `key.startswith(rule.key)`, so a badly named routed key would let a
model guess trigger a *spoken* check-in. Both are asserted here. All offline.
"""
from __future__ import annotations

import math

import core.registry as registry
from agent.corroboration import DEFAULT_RULES
from modules.skin_vision import (_CUE_ROUTES, _FACIAL_KEYS, FacialCues,
                                 SkinVision, validate_analysis)
from output.dashboard import _MODULE_TRIGGERS


def _raw(confidence: float = 0.8, **cues) -> dict:
    raw = {
        "image_quality": "good", "sufficient_skin_visible": True,
        "finding_present": False, "visible_features": [],
        "body_region": "visible skin", "confidence": 0.1,
        "possible_conditions": [], "follow_up_topics": [],
        "facial_cue_confidence": confidence,
    }
    raw.update({key: "none" for key in _FACIAL_KEYS[:-1]})
    raw["nasal_discharge_visible"] = "no"
    raw.update(cues)
    return raw


def _routed(cues: FacialCues, quality: str = "good"):
    # _routed_cue_results reads only module-level tables, so a bare instance
    # avoids constructing SkinVision (which would spin up workers).
    return SkinVision._routed_cue_results(
        SkinVision.__new__(SkinVision), cues, quality)


# ------------------------------------------------------------ observed()

def test_observed_keeps_negatives_and_drops_unclear():
    cues = FacialCues(lip_dryness="none", under_eye_puffiness="mild",
                      confidence=0.8)          # everything else defaults unclear
    observed = cues.observed()
    assert observed["lip_dryness"] == "none"   # a real answer, not silence
    assert observed["under_eye_puffiness"] == "mild"
    assert "nose_redness" not in observed      # unclear is withheld


def test_positive_still_drops_negatives():
    # The agent path must be unchanged: it may only raise cues actually seen.
    cues = FacialCues(lip_dryness="none", under_eye_puffiness="marked",
                      confidence=0.8)
    assert cues.positive() == {"under_eye_puffiness": "marked"}


# -------------------------------------------------------------- routing

def test_every_route_targets_a_registered_module():
    registry.discover()
    known = set(registry.all_registered())
    for cue, (module, _key) in _CUE_ROUTES.items():
        assert cue in _FACIAL_KEYS, f"{cue} is not a real cue"
        assert module in known, f"{cue} routes to unknown module {module}"


def test_routed_results_carry_module_key_and_provenance():
    cues = FacialCues(lip_dryness="mild", confidence=0.8)
    rows = {r.key: r for r in _routed(cues)}
    row = rows["vlm_lip_dryness"]
    assert row.module == "dry_lips"            # lands on the Lip dryness card
    assert row.source == "nvidia_vlm"          # never mistakable for the heuristic
    assert row.value == {"cue": "lip_dryness", "value": "mild",
                         "image_quality": "good"}
    assert "VLM" in row.message


def test_unclear_cues_are_not_routed():
    assert _routed(FacialCues(confidence=0.8)) == []


def test_unrouted_cue_is_skipped_without_error():
    # nasal_discharge_visible has no owning detector by design.
    cues = FacialCues(nasal_discharge_visible="yes", confidence=0.8)
    assert [r.key for r in _routed(cues)] == []


def test_confidence_is_clamped_like_self_result_would():
    # Bypassing self.result() means re-doing its sanitising; a NaN reaching
    # the dashboard's confidence sort would be a real defect.
    for bad in (float("nan"), -3.0, 12.0):
        cues = FacialCues(lip_dryness="mild", confidence=bad)
        conf = _routed(cues)[0].confidence
        assert math.isfinite(conf) and 0.0 <= conf <= 1.0


# ------------------------------------------------- corroboration containment

def test_routed_keys_cannot_trigger_a_spoken_check_in():
    # corroboration matches `r.module == rule.module and r.key.startswith(
    # rule.key)`. A routed key must never satisfy that, or a model guess
    # would put words in the agent's mouth.
    for _cue, (module, key) in _CUE_ROUTES.items():
        for rule in DEFAULT_RULES:
            if getattr(rule, "module", None) != module:
                continue
            rule_key = getattr(rule, "key", "")
            assert not key.startswith(rule_key), (
                f"{module}/{key} would satisfy corroboration rule {rule_key!r}")


# ------------------------------------------------------------- schema/gating

def test_new_cues_validate_through_the_existing_contract():
    analysis = validate_analysis(
        _raw(forehead_shine="marked", eye_redness="mild",
             visible_skin_marking="none"),
        allow_facial_cues=True, face_crop_available=True)
    observed = analysis.facial_cues.observed()
    assert observed["forehead_shine"] == "marked"
    assert observed["eye_redness"] == "mild"
    assert observed["visible_skin_marking"] == "none"


def test_missing_new_cue_is_rejected():
    raw = _raw()
    del raw["forehead_shine"]
    try:
        validate_analysis(raw, allow_facial_cues=True, face_crop_available=True)
    except ValueError:
        return
    raise AssertionError("a missing required cue must not validate")


def test_low_confidence_suppresses_every_cue():
    analysis = validate_analysis(
        _raw(confidence=0.44, forehead_shine="marked"),
        allow_facial_cues=True, face_crop_available=True)
    assert analysis.facial_cues.observed() == {}
    assert _routed(analysis.facial_cues) == []


def test_poor_image_quality_suppresses_every_cue():
    raw = _raw(forehead_shine="marked")
    raw["image_quality"] = "poor"
    analysis = validate_analysis(raw, allow_facial_cues=True,
                                 face_crop_available=True)
    assert analysis.facial_cues.observed() == {}


# --------------------------------------------------------------- triggers

def test_scan_trigger_offered_exactly_where_cues_are_routed():
    routed = {module for module, _key in _CUE_ROUTES.values()}
    offered = {name for name, trigger in _MODULE_TRIGGERS.items()
               if trigger["action"] == "vlm_scan"}
    assert offered == routed, "scan buttons drifted from the cue routes"


def test_request_scan_declines_when_screening_unavailable():
    screening = SkinVision.__new__(SkinVision)
    screening._key = None
    screening._enabled = False
    assert SkinVision.request_scan(screening) is False
