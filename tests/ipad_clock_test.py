"""Unit tests for core/ipad_camera.py's CaptureClock: mapping the iPad's
capture-relative mediaTime onto local wall-clock timestamps.

No camera, no link, no network -- CaptureClock.map() is pure arithmetic over
(media_time, recv_wall) pairs, so these tests feed it synthetic sequences
directly.

Run standalone:  python tests/ipad_clock_test.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ipad_camera import CaptureClock
from modules._util import dominant_frequency


def test_offset_uses_minimum_not_mean():
    """A mean- or median-based offset estimator fails this test -- averaging
    in the 120ms queueing spikes would smear that delay into the *output*
    cadence, since the mean itself drifts sample to sample as spikes enter
    and leave the rolling window. The minimum, by contrast, is pinned to the
    least-queued sample and simply ignores the spikes once a smaller delay
    has been seen. That's the entire point of choosing min over mean/median:
    it makes the mapped timestamps track the sender's true capture cadence
    (20Hz here) instead of the arrival jitter."""
    clock = CaptureClock()
    delays = itertools.cycle([0.010, 0.010, 0.120, 0.010])
    base = 1_000_000.0
    mapped = []
    for k in range(40):
        media_time = k * 0.05
        delay = next(delays)
        recv_wall = base + media_time + delay
        mapped.append(clock.map(media_time, recv_wall))

    diffs = np.diff(np.asarray(mapped))
    spread = float(np.std(diffs))
    assert spread <= 0.002, (
        f"the 120ms queueing spike leaked into output cadence: "
        f"std(diff)={spread:.5f}s (want <= 0.002s)")
    print(f"[ipad-clock-test] minimum-based offset absorbs queueing spikes "
          f"(cadence std={spread:.5f}s) OK")


def test_monotonic_output():
    """Duplicate and regressing media_time values (out-of-order delivery,
    minor jitter) must never produce a non-increasing output timestamp --
    downstream buffers (TimedBuffer, Pipeline's staleness check) assume
    strictly increasing time."""
    clock = CaptureClock()
    media_times = [0.00, 0.05, 0.05, 0.04, 0.10, 0.09, 0.20, 0.19, 0.30, 0.29, 0.40]
    base = 2_000_000.0
    outputs = []
    for i, mt in enumerate(media_times):
        recv_wall = base + i * 0.05 + 0.01   # arrival clock itself is monotonic
        outputs.append(clock.map(mt, recv_wall))

    diffs = np.diff(np.asarray(outputs))
    assert np.all(diffs > 0), f"output timestamps not strictly increasing: {outputs}"
    print("[ipad-clock-test] duplicate/regressing media_time still yields "
          "strictly increasing output OK")


def test_resync_on_media_time_reset():
    """A page reload restarts the iPad's mediaTime at 0 -- CaptureClock must
    detect that as 'media_time_backwards', re-anchor, and keep emitting
    strictly increasing timestamps straight through the reset."""
    clock = CaptureClock()
    base = 3_000_000.0
    outputs = []
    t = 0.0
    for k in range(20):
        mt = k * 0.05
        recv_wall = base + mt + 0.01
        outputs.append(clock.map(mt, recv_wall))
        t = recv_wall

    # Reload: mediaTime restarts at 0; arrival wall-clock keeps advancing
    # normally (small gap, well under the 2s resync_gap) so only the
    # backwards-time condition should fire, not the arrival-gap one.
    restart_base = t + 0.05
    for k in range(20):
        mt = k * 0.05
        recv_wall = restart_base + mt + 0.01
        outputs.append(clock.map(mt, recv_wall))

    diag = clock.diagnostics()
    assert diag["resyncs"] >= 1, f"expected at least one resync, got {diag}"
    assert diag["last_resync_reason"] == "media_time_backwards", diag

    diffs = np.diff(np.asarray(outputs))
    assert np.all(diffs > 0), "output must stay strictly increasing across the reset"
    print(f"[ipad-clock-test] media_time reset triggers resync "
          f"(resyncs={diag['resyncs']}) and output stays monotonic OK")


def test_resync_on_gap():
    """An arrival gap longer than resync_gap (2s) -- e.g. a backgrounded
    Safari tab resuming -- must trigger a 'gap' resync even though
    media_time itself never went backwards."""
    clock = CaptureClock()
    base = 4_000_000.0
    clock.map(0.00, base + 0.01)
    clock.map(0.05, base + 0.06)
    # media_time advances only 50ms, but 2.5s elapses on the arrival clock
    clock.map(0.10, base + 0.06 + 2.5)

    diag = clock.diagnostics()
    assert diag["last_resync_reason"] == "gap", diag
    assert diag["resyncs"] >= 1
    print(f"[ipad-clock-test] a >2s arrival gap triggers a 'gap' resync OK")


def test_end_to_end_resampling_fidelity():
    """Synthesizes a 1.2Hz signal (in rPPG's 0.7-3.0Hz band) sampled at a
    steady 20Hz capture rate but delivered with jittery, occasionally spiky
    arrival times -- the situation CaptureClock exists for. Path A uses
    CaptureClock-mapped timestamps; Path B uses the raw (jittery) arrival
    wall time as if it were the sample time, which is what a naive
    implementation would do. Both are resampled onto a uniform 30Hz grid and
    FFT'd; Path A must recover 1.2Hz accurately and at least as well as
    Path B, turning the design rationale in CaptureClock's docstring into a
    checked claim rather than an assertion in prose."""
    rng = np.random.default_rng(42)
    fs_capture = 20.0
    duration_s = 25.0
    n = int(duration_s * fs_capture)
    true_freq = 1.2

    base = 5_000_000.0
    media_times = np.arange(n) / fs_capture
    values = np.sin(2 * np.pi * true_freq * media_times)

    # Jittered, occasionally spiky one-way delay (simulates WebRTC/data
    # channel queueing under load).
    jitter = 0.010 + rng.uniform(0.0, 0.020, size=n)
    spikes = (np.arange(n) % 9 == 0)
    jitter[spikes] += 0.400
    recv_walls = base + media_times + jitter

    clock = CaptureClock(window_seconds=10.0)
    mapped_ts = np.array([clock.map(float(mt), float(rw))
                          for mt, rw in zip(media_times, recv_walls)])

    fs_out = 30.0

    def _recover_freq(timestamps, samples):
        t0, t1 = timestamps[0], timestamps[-1]
        grid = np.arange(t0, t1, 1.0 / fs_out)
        resampled = np.interp(grid, timestamps, samples)
        return dominant_frequency(resampled, fs_out, 0.7, 3.0)

    result_a = _recover_freq(mapped_ts, values)
    result_b = _recover_freq(recv_walls, values)
    assert result_a is not None, "Path A (CaptureClock) failed to recover any peak"
    assert result_b is not None, "Path B (raw arrival) failed to recover any peak"

    freq_a, _ = result_a
    freq_b, _ = result_b
    err_a = abs(freq_a - true_freq)
    err_b = abs(freq_b - true_freq)

    assert err_a <= 0.05, (
        f"CaptureClock-mapped timestamps failed to recover {true_freq}Hz: "
        f"got {freq_a:.3f}Hz (err={err_a:.3f}Hz)")
    assert err_a <= err_b, (
        f"CaptureClock mapping should be at least as accurate as raw arrival "
        f"time: err_a={err_a:.3f}Hz vs err_b={err_b:.3f}Hz "
        f"(freq_a={freq_a:.3f}Hz, freq_b={freq_b:.3f}Hz)")
    print(f"[ipad-clock-test] end-to-end resampling recovers {true_freq}Hz "
          f"(Path A err={err_a:.3f}Hz, Path B err={err_b:.3f}Hz) OK")


def main():
    """Run all CaptureClock tests."""
    test_offset_uses_minimum_not_mean()
    test_monotonic_output()
    test_resync_on_media_time_reset()
    test_resync_on_gap()
    test_end_to_end_resampling_fidelity()
    print("[ipad-clock-test] OK")


if __name__ == "__main__":
    main()
