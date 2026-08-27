"""Exactly one microphone may feed the shared audio bus.

`audio.bus.AudioBus` fans every published block to one queue per consumer, and
`audio.stt.Listener` concatenates whatever arrives into the segment it is
building. Two open inputs therefore do not "mix" — they interleave blocks of
different lengths, recorded from different positions in the room, into a single
buffer, which is why a paired-iPad run produced transcripts like "and collective
hope to meet that" from clearly-spoken sentences. A laptop microphone in the
same room also hears the paired device's speaker, an echo path that device's own
cancellation cannot see.

`resolve_microphone_mode` is the single decision point, kept a pure function so
this contract is testable without a camera, a relay, or an audio device.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import resolve_microphone_mode


def test_auto_hands_the_ears_to_the_device_that_carries_the_audio():
    for ipad_audio in ("device", "both"):
        assert resolve_microphone_mode("auto", True, ipad_audio) == "device"
    # Speech kept on laptop speakers: the local input is the sane ears, and the
    # device's own microphone is not part of that arrangement.
    assert resolve_microphone_mode("auto", True, "laptop") == "laptop"


def test_auto_without_an_ipad_is_the_laptop_microphone():
    for ipad_audio in ("device", "both", "laptop"):
        assert resolve_microphone_mode("auto", False, ipad_audio) == "laptop"


def test_explicit_choices_are_honoured():
    assert resolve_microphone_mode("laptop", True, "device") == "laptop"
    assert resolve_microphone_mode("device", True, "laptop") == "device"
    assert resolve_microphone_mode("off", True, "device") == "off"
    assert resolve_microphone_mode("off", False, "laptop") == "off"


def test_device_degrades_to_laptop_without_an_ipad_rather_than_going_deaf():
    """'--mic device' on a webcam run must not silently leave the run deaf."""
    assert resolve_microphone_mode("device", False, "device") == "laptop"


def test_every_mode_resolves_to_exactly_one_source():
    """No combination may ever ask for two live microphones."""
    for mic in ("auto", "laptop", "device", "off"):
        for ipad_source in (True, False):
            for ipad_audio in ("device", "both", "laptop"):
                assert resolve_microphone_mode(mic, ipad_source, ipad_audio) \
                    in ("laptop", "device", "off")
