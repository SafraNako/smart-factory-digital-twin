"""Sensor models: noise, bias, quantisation, lost samples and the three classic sensor faults."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorSpec:
    sigma: float                 # measurement noise standard deviation (same unit as the signal)
    resolution: float = 0.0      # quantisation step; 0 = none
    dropout: float = 0.0         # probability that a sample is lost (reported as NaN)
    bias: float = 0.0

    def effective_sigma(self):
        """Noise std including the variance ``resolution^2 / 12`` of uniform quantisation."""
        return float(np.sqrt(self.sigma ** 2 + self.resolution ** 2 / 12.0))


@dataclass(frozen=True)
class SensorFault:
    kind: str                    # "stuck", "drift" or "burst"
    start: float                 # hours
    end: float = float("inf")
    rate: float = 0.0            # drift: bias growth per hour

    def __post_init__(self):
        if self.kind not in ("stuck", "drift", "burst"):
            raise ValueError("sensor fault kind must be 'stuck', 'drift' or 'burst'")


def read(spec: SensorSpec, signal, t, rng, faults=()):
    """Readings of ``signal`` sampled at times ``t``; lost samples are NaN."""
    signal = np.asarray(signal, dtype=float)
    t = np.asarray(t, dtype=float)
    y = signal + spec.bias + spec.sigma * rng.standard_normal(len(signal))
    for f in faults:
        active = (t >= f.start) & (t < f.end)
        if f.kind == "drift":
            y = y + np.where(active, f.rate * (t - f.start), 0.0)
    if spec.resolution > 0:
        y = np.round(y / spec.resolution) * spec.resolution
    for f in faults:
        active = (t >= f.start) & (t < f.end)
        if f.kind == "stuck" and active.any():
            first = int(np.argmax(active))
            y[active] = y[first]
    if spec.dropout > 0:
        y = np.where(rng.random(len(y)) < spec.dropout, np.nan, y)
    for f in faults:
        if f.kind == "burst":
            y = np.where((t >= f.start) & (t < f.end), np.nan, y)
    return y
