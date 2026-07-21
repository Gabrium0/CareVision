"""Shared helpers for detection modules: timed buffers, spectral analysis,
face-ROI sampling. Underscore prefix keeps the registry from importing it
as a module package.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def native_detail_context(ctx):
    """Pair native pixels/depth with geometry inferred on the bounded context."""
    native = ctx.extras.get("_native_detail_context")
    if native is None or native is ctx:
        return ctx
    cached = ctx.extras.get("_native_pose_context_cache")
    if cached is not None:
        return cached
    import copy
    from core.context import PoseData
    detail = copy.copy(native)
    # Keep caches on the native frame while preserving non-private quality
    # policy fields used by detail modules.
    detail.extras = native.extras
    if ctx.pose is not None:
        sx, sy = native.w / max(ctx.w, 1), native.h / max(ctx.h, 1)
        x1, y1, x2, y2 = ctx.pose.bbox
        detail.pose = PoseData(np.array(ctx.pose.landmarks, copy=True),
                               (int(round(x1 * sx)), int(round(y1 * sy)),
                                int(round(x2 * sx)), int(round(y2 * sy))))
    ctx.extras["_native_pose_context_cache"] = detail
    return detail


class TimedBuffer:
    """Rolling (timestamp, value) buffer with a fixed time horizon."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.t: deque[float] = deque()
        self.v: deque = deque()

    def push(self, timestamp: float, value) -> None:
        """Append one sample and discard anything older than the time horizon."""
        self.t.append(timestamp)
        self.v.append(value)
        while self.t and timestamp - self.t[0] > self.seconds:
            self.t.popleft()
            self.v.popleft()

    def __len__(self) -> int:
        return len(self.t)

    def span(self) -> float:
        """Time covered by the current buffer, in seconds."""
        return (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return timestamps and values as float arrays for signal processing."""
        return np.asarray(self.t, dtype=np.float64), np.asarray(self.v, dtype=np.float64)

    def resampled(self, fs: float) -> tuple[np.ndarray, float] | None:
        """Uniformly resample values at fs Hz; returns (signal, fs) or None."""
        if len(self.t) < 8 or self.span() < 2.0:
            return None
        t, v = self.arrays()
        t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
        if len(t_uniform) < 8:
            return None
        return np.interp(t_uniform, t, v), fs


def pooled_skin_sample(buffer: "TimedBuffer", sample: np.ndarray,
                       timestamp: float) -> np.ndarray:
    """Push `sample` into `buffer` and return its temporal mean over the
    buffer's window, instead of the raw single-frame value.

    A near-stationary subject's per-frame color sample is noisy after MJPEG
    4:2:0 chroma subsampling (half the color resolution discarded before the
    frame reaches this code); averaging over a short rolling window recovers
    chroma signal-to-noise at the cost of a few seconds of lag. `sample` may
    be scalar or a fixed-length vector (e.g. an (r,g,b) chromaticity triple).
    """
    buffer.push(timestamp, sample)
    _, values = buffer.arrays()
    if len(values) == 0:
        return np.asarray(sample, dtype=np.float64)
    return values.mean(axis=0)


def dominant_frequency(signal: np.ndarray, fs: float,
                       fmin: float, fmax: float) -> tuple[float, float] | None:
    """Strongest frequency in [fmin, fmax] Hz.

    Returns (frequency_hz, prominence 0..1) where prominence is peak power
    over total band power — a crude quality/confidence measure.
    """
    sig = signal - np.mean(signal)
    n = len(sig)
    if n < 16:
        return None
    # Windowing reduces FFT edge artifacts on short rolling buffers.
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(sig * window)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if band.sum() < 3 or spectrum[band].sum() <= 0:
        return None
    band_power = spectrum[band]
    peak_idx = int(np.argmax(band_power))
    # Treat one sharp spectral peak as higher quality than diffuse band energy.
    prominence = float(band_power[peak_idx] / band_power.sum())
    return float(freqs[band][peak_idx]), prominence


def bandpass(signal: np.ndarray, fs: float, fmin: float, fmax: float,
             order: int = 3) -> np.ndarray | None:
    """Butterworth band-pass filter a 1-D signal (or None if too short)."""
    nyq = fs / 2.0
    # filtfilt needs enough samples for padding; returning None avoids noisy startup values.
    if fmax >= nyq or len(signal) < 3 * (order + 1):
        return None
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, signal)


def patch_brightness(patch: np.ndarray) -> float:
    """Mean perceived luminance (0-255) of a BGR patch (Rec.601 weights)."""
    p = patch.reshape(-1, 3).astype(np.float32)          # BGR
    return float((0.114 * p[:, 0] + 0.587 * p[:, 1] + 0.299 * p[:, 2]).mean())


def low_light_factor(brightness: float, dark: float = 25.0,
                     good: float = 90.0) -> float:
    """Lighting-quality multiplier in [0, 1] for confidence scaling.

    rPPG SNR is photon-limited: below ~`dark` mean luminance the pulse is
    buried in shot noise/quantization and readings are untrustworthy, so we
    ramp confidence down to 0; at/above `good` lighting is adequate (1.0).
    """
    if good <= dark:
        return 1.0
    return float(np.clip((brightness - dark) / (good - dark), 0.0, 1.0))


def rppg_input_quality(brightness: float, effective_hz: float,
                       acceptance_ratio: float = 1.0,
                       regularity: float = 1.0,
                       target_hz: float = 20.0) -> float:
    """Capture-quality score independent of the BPM algorithm's confidence."""
    cadence = min(1.0, max(0.0, float(effective_hz)) / max(target_hz, 1e-6))
    return float(np.clip(low_light_factor(brightness) * cadence
                         * np.clip(acceptance_ratio, 0.0, 1.0)
                         * np.clip(regularity, 0.0, 1.0), 0.0, 1.0))


def _chrominance(rgb: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Combine two orthogonal chrominance projections with alpha tuning
    (shared core of CHROM/POS): S = Xs - (std(Xs)/std(Ys)) * Ys."""
    sy = float(np.std(ys))
    alpha = (float(np.std(xs)) / sy) if sy > 1e-9 else 0.0
    sig = xs - alpha * ys
    return sig - np.mean(sig)


def chrom(rgb: np.ndarray) -> np.ndarray | None:
    """CHROM (de Haan & Jeanne 2013) pulse signal from an (N,3) R,G,B window.

    Channels are divided by their temporal mean so overall brightness/lighting
    cancels, then combined on the chrominance plane. Illumination-robust, so it
    holds up far better than a raw green-channel mean in poor/uneven light.
    """
    c = np.asarray(rgb, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 16 or c.shape[1] != 3:
        return None
    mean = c.mean(axis=0)
    if np.any(mean <= 1e-6):
        return None
    cn = c / mean                                    # temporal DC normalization
    r, g, b = cn[:, 0], cn[:, 1], cn[:, 2]
    xs = 3.0 * r - 2.0 * g
    ys = 1.5 * r + g - 1.5 * b
    return _chrominance(cn, xs, ys)


def pos(rgb: np.ndarray, fs: float = 30.0) -> np.ndarray | None:
    """POS (Wang et al. 2017) pulse signal from an (N,3) R,G,B window.

    Plane-orthogonal-to-skin projection with the canonical overlap-add over
    short (~1.6 s) windows: each window is normalized by its own temporal mean,
    projected, alpha-tuned, and summed back. Normalizing per short window (not
    over the whole buffer) is what makes POS robust to slow illumination drift.
    """
    c = np.asarray(rgb, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 16 or c.shape[1] != 3:
        return None
    n = c.shape[0]
    win = max(16, int(1.6 * fs))
    if win > n:                                      # buffer shorter than a window
        win = n
    out = np.zeros(n, dtype=np.float64)
    for m in range(0, n - win + 1):
        seg = c[m:m + win]
        mean = seg.mean(axis=0)
        if np.any(mean <= 1e-6):
            continue
        cn = seg / mean
        r, g, b = cn[:, 0], cn[:, 1], cn[:, 2]
        s1 = g - b                                   # POS projection matrix row 1
        s2 = -2.0 * r + g + b                        # POS projection matrix row 2
        sd2 = float(np.std(s2))
        alpha = (float(np.std(s1)) / sd2) if sd2 > 1e-9 else 0.0
        h = s1 + alpha * s2
        out[m:m + win] += h - np.mean(h)             # overlap-add
    return out - np.mean(out)


def peak_intervals(signal: np.ndarray, fs: float,
                   min_distance_s: float) -> np.ndarray:
    """Inter-peak intervals in seconds (for HRV from a pulse waveform)."""
    # min_distance_s encodes the fastest plausible beat rate and suppresses double counts.
    peaks, _ = find_peaks(signal, distance=max(1, int(min_distance_s * fs)))
    if len(peaks) < 3:
        return np.array([])
    return np.diff(peaks) / fs


def roi_patch(ctx, center_idx: int, radius_frac: float = 0.04) -> np.ndarray | None:
    """Square skin patch around a face landmark; radius relative to face width."""
    ctx = ctx.extras.get("_native_detail_context", ctx)
    px = ctx.face_px()
    if px is None:
        return None
    x1, y1, x2, y2 = ctx.face.bbox
    r = max(2, int(radius_frac * (x2 - x1)))
    cx, cy = px[center_idx].astype(int)
    ax1, ay1 = max(0, cx - r), max(0, cy - r)
    ax2, ay2 = min(ctx.w, cx + r), min(ctx.h, cy + r)
    patch = ctx.frame[ay1:ay2, ax1:ax2]
    return patch if patch.size else None


def mouth_aspect_ratio(ctx) -> float | None:
    """Mouth aspect ratio (inner-lip height / mouth width) from face landmarks;
    same calculation modules/yawn.py uses to detect yawns. A rising or falling
    delta between consecutive calls also flags talking (yawn.py distinguishes
    the two by sustained-open duration; talking is comparatively fast/small)."""
    from extractors import face_landmarks as FL
    px = ctx.face_px()
    if px is None:
        return None
    w = np.linalg.norm(px[FL.MOUTH_LEFT] - px[FL.MOUTH_RIGHT]) + 1e-6
    h = np.linalg.norm(px[FL.MOUTH_TOP_INNER] - px[FL.MOUTH_BOTTOM_INNER])
    return float(h / w)


def polygon_mask(shape_hw: tuple, points_px: np.ndarray) -> np.ndarray:
    """Binary mask (uint8) of the polygon given by points_px."""
    mask = np.zeros(shape_hw[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points_px.astype(np.int32)], 255)
    return mask


def face_skin_mask(ctx) -> np.ndarray | None:
    """Face-oval mask minus eyes and mouth — 'skin only' pixels."""
    region = face_skin_region(ctx)
    if region is None:
        return None
    _, local, (x1, y1, x2, y2) = region
    detail = ctx.extras.get("_native_detail_context", ctx)
    mask = np.zeros(detail.frame.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = local
    return mask


def face_skin_region(ctx, padding: float = 0.05):
    """Return (face patch, local skin mask, bounds) without full-frame arrays."""
    from extractors import face_landmarks as FL
    ctx = ctx.extras.get("_native_detail_context", ctx)
    cache = ctx.extras.setdefault("_face_skin_region_cache", {})
    cache_key = round(float(padding), 4)
    if cache_key in cache:
        return cache[cache_key]
    px = ctx.face_px()
    if px is None:
        return None
    bx1, by1, bx2, by2 = ctx.face.bbox
    pad = int(max(bx2 - bx1, by2 - by1) * float(padding))
    x1, y1 = max(0, bx1 - pad), max(0, by1 - pad)
    x2, y2 = min(ctx.w, bx2 + pad), min(ctx.h, by2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    local = px - np.array([x1, y1], dtype=np.float32)
    patch = ctx.frame[y1:y2, x1:x2]
    mask = polygon_mask(patch.shape, local[FL.FACE_OVAL])
    for hole in (FL.LEFT_EYE_RING, FL.RIGHT_EYE_RING, FL.OUTER_LIPS):
        cv2.fillPoly(mask, [local[hole].astype(np.int32)], 0)
    result = (patch, mask, (x1, y1, x2, y2))
    cache[cache_key] = result
    return result


def face_color_plane(ctx, name: str):
    """Return one cached face-ROI color conversion for the current frame."""
    ctx = ctx.extras.get("_native_detail_context", ctx)
    cache = ctx.extras.setdefault("_face_color_planes", {})
    key = str(name).lower()
    if key in cache:
        return cache[key]
    region = face_skin_region(ctx)
    if region is None:
        return None
    frame = region[0]
    conversions = {"gray": cv2.COLOR_BGR2GRAY, "hsv": cv2.COLOR_BGR2HSV,
                   "lab": cv2.COLOR_BGR2LAB, "ycrcb": cv2.COLOR_BGR2YCrCb,
                   "rgb": cv2.COLOR_BGR2RGB}
    if key not in conversions:
        raise ValueError(f"unknown face color plane: {name!r}")
    cache[key] = cv2.cvtColor(frame, conversions[key])
    return cache[key]


def arm_rois(ctx) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Oriented-box ROIs around visible arm segments from pose landmarks.

    Returns (label, polygon_px, anchors_px) per segment — upper arm
    (shoulder->elbow) and forearm (elbow->wrist), both sides — for segments
    whose endpoint landmarks are confidently visible. The box half-width is
    proportional to segment length so the ROI scales with distance; anchors
    are the two joint pixels, used by arm_skin_mask as the depth reference.
    """
    from extractors import pose as P
    ctx = native_detail_context(ctx)
    px = ctx.pose_px()
    if px is None:
        return []
    lm = ctx.pose.landmarks
    segments = (("left upper arm", P.L_SHOULDER, P.L_ELBOW),
                ("left forearm", P.L_ELBOW, P.L_WRIST),
                ("right upper arm", P.R_SHOULDER, P.R_ELBOW),
                ("right forearm", P.R_ELBOW, P.R_WRIST))
    rois = []
    for label, a, b in segments:
        if lm[a, 3] < 0.5 or lm[b, 3] < 0.5:
            continue
        pa, pb = px[a], px[b]
        seg = pb - pa
        length = float(np.linalg.norm(seg))
        if length < 12.0:                    # degenerate at extreme distance
            continue
        normal = np.array([-seg[1], seg[0]]) / length
        half_w = 0.30 * length
        poly = np.array([pa + normal * half_w, pb + normal * half_w,
                         pb - normal * half_w, pa - normal * half_w])
        rois.append((label, poly, np.array([pa, pb])))
    return rois


def arm_skin_mask(ctx, polygon: np.ndarray, anchors: np.ndarray | None = None,
                  min_skin_fraction: float = 0.35,
                  depth_window_m: float = 0.12) -> np.ndarray | None:
    """Skin-only mask (uint8 0/255) inside an arm ROI, or None when sleeved.

    Skin chroma is judged against the person's own face skin (median Cr/Cb
    of face_skin_mask pixels) when a face is visible — robust across skin
    tones — falling back to the classic YCrCb skin range otherwise. With
    depth available, pixels farther than `depth_window_m` from the median
    anchor-joint depth are dropped (background seen past the arm, torso
    behind a hanging arm). If skin covers less than `min_skin_fraction` of
    the ROI the segment is treated as sleeved and None is returned.
    """
    region = arm_skin_region(ctx, polygon, anchors, min_skin_fraction,
                             depth_window_m)
    if region is None:
        return None
    _, local, (x1, y1, x2, y2) = region
    detail = native_detail_context(ctx)
    mask = np.zeros(detail.frame.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = local
    return mask


def arm_skin_region(ctx, polygon: np.ndarray, anchors: np.ndarray | None = None,
                    min_skin_fraction: float = 0.35,
                    depth_window_m: float = 0.12):
    """Return (arm patch, local mask, bounds), doing all work inside the ROI."""
    ctx = native_detail_context(ctx)
    # Keep a guaranteed background border so flood-fill can identify enclosed
    # discolored skin holes even when the oriented polygon touches its bounds.
    x1 = max(0, int(np.floor(polygon[:, 0].min())) - 2)
    y1 = max(0, int(np.floor(polygon[:, 1].min())) - 2)
    x2 = min(ctx.w, int(np.ceil(polygon[:, 0].max())) + 3)
    y2 = min(ctx.h, int(np.ceil(polygon[:, 1].max())) + 3)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = ctx.frame[y1:y2, x1:x2]
    local_poly = polygon - np.array([x1, y1], dtype=np.float32)
    roi = polygon_mask(patch.shape, local_poly) > 0
    roi_area = int(roi.sum())
    if roi_area < 200:                        # too small for stable statistics
        return None
    # Per-frame cache in ctx.extras: the full-frame conversions and the face
    # chroma reference are identical for all four arm segments of one frame.
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    reference = ctx.extras.get("_arm_face_chroma_reference")
    face = face_skin_region(ctx)
    if reference is None and face is not None and int((face[1] > 0).sum()) >= 400:
        face_ycc = cv2.cvtColor(face[0], cv2.COLOR_BGR2YCrCb)
        reference = (float(np.median(face_ycc[:, :, 1][face[1] > 0])),
                     float(np.median(face_ycc[:, :, 2][face[1] > 0])))
        ctx.extras["_arm_face_chroma_reference"] = reference
    if reference is not None:
        ref_cr, ref_cb = reference
        skin_chroma = ((np.abs(cr - ref_cr) < 12.0) & (np.abs(cb - ref_cb) < 12.0))
    else:
        skin_chroma = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127))
    mask = roi & skin_chroma
    # Fill enclosed holes: a bruise/rash/mole center is often too discolored
    # to pass the chroma test, but it sits INSIDE arm skin — dropping it would
    # hide exactly what the arm modules screen for. Flood-fill the background
    # from a corner; anything neither background nor skin is an interior hole.
    fill = mask.astype(np.uint8)
    if fill[0, 0] == 0:                       # seed must start in background
        ff_mask = np.zeros((fill.shape[0] + 2, fill.shape[1] + 2), np.uint8)
        cv2.floodFill(fill, ff_mask, (0, 0), 1)
        mask |= (fill == 0) & roi
    if ctx.depth is not None:
        local_depth = ctx.depth[y1:y2, x1:x2]
        ref = None
        if anchors is not None:
            depths = [ctx.depth_m(p[0], p[1]) for p in anchors]
            depths = [d for d in depths if d is not None]
            if depths:
                ref = float(np.median(depths))
        if ref is None:
            in_roi = local_depth[roi]
            valid = in_roi[in_roi > 0]
            if valid.size:
                ref = float(np.median(valid)) * ctx.depth_scale
        if ref is not None:
            depth_m = local_depth.astype(np.float32) * ctx.depth_scale
            mask &= (local_depth > 0) & (np.abs(depth_m - ref) < depth_window_m)
    if mask.sum() / roi_area < min_skin_fraction:
        return None                           # sleeved (or mostly occluded)
    return patch, mask.astype(np.uint8) * 255, (x1, y1, x2, y2)
