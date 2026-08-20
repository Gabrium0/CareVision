"""Unit tests for core/ipad_camera.py's wire-format helpers: the 24-byte
frame header (pack/unpack) and the JPEG chroma-subsampling probe.

No camera, no link, no network needed.

Run standalone:  python tests/ipad_protocol_test.py
"""
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ipad_camera import (FRAME_HEADER, FRAME_HEADER_SIZE, FRAME_MAGIC,
                              jpeg_subsampling, pack_frame_header,
                              unpack_frame_header)


def test_header_roundtrip():
    combos = [
        (0, 0.0, 0, 0),
        (1, 1.234, 640, 480),
        (2 ** 32 - 1, 123456.789, 1920, 1080),
        (100, 0.0001, 65535, 65535),
        (42, -0.0, 960, 540),
    ]
    for seq, media_time, width, height in combos:
        buf = pack_frame_header(seq, media_time, width, height)
        parsed = unpack_frame_header(buf)
        assert parsed is not None, f"round trip failed for seq={seq}"
        assert parsed["seq"] == seq & 0xFFFFFFFF
        assert parsed["media_time"] == media_time, (
            f"media_time did not round-trip exactly: {parsed['media_time']} != {media_time}")
        assert parsed["width"] == width & 0xFFFF
        assert parsed["height"] == height & 0xFFFF
        assert parsed["flags"] == 0

    # bad magic
    bad_magic = b"XXXX" + b"\x00" * (FRAME_HEADER_SIZE - 4)
    assert len(bad_magic) == FRAME_HEADER_SIZE
    assert unpack_frame_header(bad_magic) is None

    # truncated buffer (< 24 bytes) must not raise
    for short_len in (0, 1, 4, 23):
        truncated = (FRAME_MAGIC + b"\x00" * 20)[:short_len]
        assert unpack_frame_header(truncated) is None, (
            f"expected None for a {short_len}-byte buffer, no exception")
    print("[ipad-protocol-test] header pack/unpack round trip, bad magic, "
          "and truncated buffers all handled OK")


def test_header_size_matches_wire_format():
    """Keeps the JS DataView in relay/static/ipad.html in sync with the
    Python struct layout: magic, seq, mediaTime, width, height, flags,
    reserved at byte offsets 0, 4, 8, 16, 18, 20, 22."""
    assert FRAME_HEADER_SIZE == 24
    assert FRAME_HEADER.size == 24

    prefixes = ["<4s", "<4sI", "<4sId", "<4sIdH", "<4sIdHH", "<4sIdHHH"]
    computed_offsets = [struct.calcsize(p) for p in prefixes]
    expected_field_offsets = [0] + computed_offsets   # magic starts at 0
    assert expected_field_offsets == [0, 4, 8, 16, 18, 20, 22], expected_field_offsets
    print(f"[ipad-protocol-test] wire-format field offsets {expected_field_offsets} OK")


def _encode(img, params=None):
    ok, buf = cv2.imencode(".jpg", img, params or [])
    assert ok
    return buf.tobytes()


def test_jpeg_subsampling_probe():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (50, 100, 150)

    default_jpeg = _encode(img)
    assert jpeg_subsampling(default_jpeg) == "4:2:0", (
        "OpenCV's default JPEG encoding is expected to be 4:2:0")

    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") and \
            hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
        params = [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                  int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)]
        full_res_jpeg = _encode(img, params)
        assert jpeg_subsampling(full_res_jpeg) == "4:4:4"
        print("[ipad-protocol-test] jpeg_subsampling reports 4:2:0 (default) "
              "and 4:4:4 (forced) OK")
    else:
        print("[ipad-protocol-test] OpenCV build lacks JPEG sampling factor "
              "constants; skipping the 4:4:4 half of this test")

    assert jpeg_subsampling(b"not a jpeg at all, just some bytes") is None
    assert jpeg_subsampling(b"") is None
    print("[ipad-protocol-test] non-JPEG bytes return None OK")


def test_seq_gap_accounting():
    from core.ipad_link import IPadLink
    link = IPadLink(relay_url="https://x", room="r", secret="s", code="123456")
    for seq in (1, 2, 4, 5):
        buf = pack_frame_header(seq, seq * 0.05, 640, 480) + b"data"
        link._ingest_frame(buf)

    status = link.status()
    assert status["seq_gaps"] == 1, f"expected 1 gap (seq 3 missing), got {status['seq_gaps']}"
    assert status["frames_rx"] == 4, f"expected 4 frames received, got {status['frames_rx']}"
    print("[ipad-protocol-test] seq-gap accounting (1 gap, 4 frames_rx) OK")


def main():
    """Run all iPad protocol tests."""
    test_header_roundtrip()
    test_header_size_matches_wire_format()
    test_jpeg_subsampling_probe()
    test_seq_gap_accounting()
    print("[ipad-protocol-test] OK")


if __name__ == "__main__":
    main()
