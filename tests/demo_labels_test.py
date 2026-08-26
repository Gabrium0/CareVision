"""Anti-drift gate for the guest-facing /demo page: every registered module
must resolve to a plain-English (label, blurb) via module_label(), never the
raw underscored slug. A newly added detector that forgets a curated entry
must still render as readable text, not `some_new_detector`, and the
copy must stay observational (non-diagnostic) — this is a non-clinical
screening system, not a medical device.
"""
from __future__ import annotations

import core.registry as registry
from output.dashboard import (_MODULE_LABELS, _launchable_assessments,
                              module_label)

_CLINICAL_WORDS = ("diagnos", "disease", "detects stroke")


def test_every_registered_module_resolves_to_a_human_label():
    registry.discover()
    all_slugs = registry.all_registered()
    assert all_slugs, "expected modules/ to have registered at least one module"
    for slug in all_slugs:
        label, blurb = module_label(slug)
        assert label, f"{slug} resolved to an empty label"
        assert label != slug, f"{slug} rendered as the raw slug, not a human label"
        assert "_" not in label, f"{slug} label still has underscores: {label!r}"


def test_unknown_slug_is_humanized_without_raising():
    label, blurb = module_label("some_new_thing")
    assert label == "Some new thing"
    assert blurb == ""


def test_curated_entries_win_over_docstring_fallback():
    for slug, (label, blurb) in _MODULE_LABELS.items():
        assert module_label(slug) == (label, blurb)


def test_no_clinical_claim_words_in_labels_or_blurbs():
    registry.discover()
    for slug in registry.all_registered():
        label, blurb = module_label(slug)
        text = f"{label} {blurb}".lower()
        for word in _CLINICAL_WORDS:
            assert word not in text, f"{slug} label/blurb contains clinical claim {word!r}: {text!r}"
    for label, blurb in _MODULE_LABELS.values():
        text = f"{label} {blurb}".lower()
        for word in _CLINICAL_WORDS:
            assert word not in text, f"curated entry contains clinical claim {word!r}: {text!r}"


def test_assessment_picker_labels_are_guest_safe_and_non_clinical():
    """The /demo 'try this' picker shows these to a client on a TV: they must
    name an activity, never a clinical test or its result, and never leak a
    raw slug."""
    roster = _launchable_assessments()
    assert roster, "expected at least one launchable assessment"
    for entry in roster:
        label, blurb = entry["label"], entry["blurb"]
        assert label and "_" not in label, f"bad picker label: {label!r}"
        assert label != entry["protocol"], f"{entry['protocol']} shows its raw slug"
        text = f"{label} {blurb}".lower()
        for word in _CLINICAL_WORDS:
            assert word not in text, f"picker entry has clinical claim {word!r}: {text!r}"
