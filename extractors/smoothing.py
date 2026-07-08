"""Vectorized One-Euro filter for landmark de-jittering.

One-Euro (Casiez et al. 2012) adapts its cutoff to motion speed: it smooths
hard when the point is still (killing sensor jitter) but backs off during fast
movement (preserving intentional motion). This is why it's preferred over a
plain EMA here — an EMA with enough smoothing to steady a resting face would
also blur real motion.

IMPORTANT: apply this to FACE landmarks (emotion, asymmetry, EAR, pain — all
low-frequency) but NOT to pose landmarks that feed tremor (3-12 Hz) or gait,
where smoothing would attenuate the very signal being measured.
"""
from __future__ import annotations

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroArray:
    """One-Euro filter over a fixed-shape landmark array, indexed per element."""

    def __init__(self, mincutoff: float = 1.5, beta: float = 0.05,
                 dcutoff: float = 1.0, reset_gap: float = 0.5):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.reset_gap = reset_gap          # reset if frames were dropped
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        """Reset internal state so the next call starts fresh."""
        self._x_prev = self._dx_prev = self._t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if (self._x_prev is None or self._t_prev is None
                or x.shape != self._x_prev.shape
                or t - self._t_prev > self.reset_gap):
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x
        dt = max(t - self._t_prev, 1e-3)
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.dcutoff, dt)
        edx = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        a = _alpha_vec(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev, self._dx_prev, self._t_prev = x_hat, edx, t
        return x_hat


def _alpha_vec(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)
