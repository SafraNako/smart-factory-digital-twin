"""Control charts on standardised innovations, with their run-length theory.

For a stream ``z`` that is N(0, 1) when the machine is healthy:

* Shewhart: alarm when ``|z| > c``; the average run length (ARL) is ``1 / P(alarm)`` exactly.
* CUSUM (one-sided upper): ``S = max(0, S + z - k)``, alarm when ``S > h``. The ARL is computed by the
  Brook-Evans Markov-chain discretisation and, independently, by Siegmund's closed-form approximation.
"""
from __future__ import annotations

import math

import numpy as np


def phi_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def shewhart_arl(c, shift=0.0):
    """Exact average run length of ``|z| > c`` when ``z ~ N(shift, 1)``."""
    p = phi_cdf(-c - shift) + phi_cdf(-c + shift)
    return 1.0 / p


def cusum_arl_brook_evans(k, h, shift=0.0, m=400):
    """ARL of the upper CUSUM from a zero start, for ``z ~ N(shift, 1)`` (Brook and Evans, 1972).

    ``[0, h)`` is cut into ``m`` states of width ``2h / (2m - 1)`` (state 0 centred at 0); the ARL vector
    solves ``(I - R) L = 1`` with ``R`` the transient transition matrix.
    """
    w = 2.0 * h / (2 * m - 1)
    drift = k - shift                     # S' = S + z - k, with z - shift ~ N(0,1)
    # entries depend only on j - i: tabulate Phi((d + 1/2) w + drift) once for d = -m .. m-1
    d = np.arange(-m, m)
    table = np.array([phi_cdf((x + 0.5) * w + drift) for x in d])
    i = np.arange(m)[:, None]
    j = np.arange(m)[None, :]
    R = table[(j - i) + m] - table[(j - i - 1) + m]
    R[:, 0] = table[(0 - np.arange(m)) + m]              # state 0 collects everything at or below w/2
    return float(np.linalg.solve(np.eye(m) - R, np.ones(m))[0])


def cusum_arl_siegmund(k, h, shift=0.0):
    """Siegmund's approximation of the upper-CUSUM ARL for a normal shift (``shift - k`` is the drift)."""
    delta = shift - k
    b = h + 1.166
    if abs(delta) < 1e-12:
        return b * b
    return (math.exp(-2.0 * delta * b) + 2.0 * delta * b - 1.0) / (2.0 * delta * delta)


def cusum_two_sided_arl(k, h, shift=0.0, m=400):
    """ARL of the pair of upper and lower CUSUMs: ``1 / (1/L+ + 1/L-)`` (Khan, 1981), accurate to a few percent."""
    up = cusum_arl_brook_evans(k, h, shift, m)
    down = cusum_arl_brook_evans(k, h, -shift, m)
    return 1.0 / (1.0 / up + 1.0 / down)


def cusum_threshold(arl0, k=0.5, m=400):
    """Smallest ``h`` (to 1e-3) for which the two-sided in-control ARL reaches ``arl0``."""
    lo, hi = 0.5, 30.0
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        if cusum_two_sided_arl(k, mid, 0.0, m) < arl0:
            lo = mid
        else:
            hi = mid
    return hi


class Cusum:
    """Streaming two-sided CUSUM on a standardised channel; NaN samples leave the statistics unchanged.

    With ``latch=False`` an alarm resets both statistics (the textbook scheme whose run lengths the ARL
    functions describe). With ``latch=True`` (the twin's default) the statistics keep running, capped at
    ``2 h``, and a direction alarms once on its rising edge and again only after its statistic has fallen
    below ``h / 2``; a persistent fault therefore gives one alarm, not one per ``h / drift`` samples.
    The first alarm time is the same in both modes.
    """

    def __init__(self, k=0.5, h=7.0, latch=True):
        self.k, self.h, self.latch = k, h, latch
        self.up = 0.0
        self.down = 0.0
        self._on_up = False
        self._on_down = False

    @property
    def active(self):
        """True while a statistic is above ``h / 2``: the channel disagrees with the model right now."""
        return max(self.up, self.down) > self.h / 2.0

    def update(self, z):
        """Returns ``+1`` or ``-1`` when an alarm is raised, otherwise ``0``."""
        if z is None or (isinstance(z, float) and math.isnan(z)):
            return 0
        self.up = max(0.0, self.up + z - self.k)
        self.down = max(0.0, self.down - z - self.k)
        if not self.latch:
            if self.up > self.h:
                self.up = self.down = 0.0
                return 1
            if self.down > self.h:
                self.up = self.down = 0.0
                return -1
            return 0
        cap = 2.0 * self.h
        self.up, self.down = min(self.up, cap), min(self.down, cap)
        fired = 0
        if self.up > self.h and not self._on_up:
            self._on_up, fired = True, 1
        elif self.up < self.h / 2.0:
            self._on_up = False
        if self.down > self.h and not self._on_down:
            self._on_down, fired = True, -1
        elif self.down < self.h / 2.0:
            self._on_down = False
        return fired


def run_length(k, h, shift=0.0, rng=None, max_len=10 ** 7):
    """Number of samples until the upper CUSUM alarms on ``z ~ N(shift, 1)`` (simulation)."""
    s, n = 0.0, 0
    while n < max_len:
        n += 1
        s = max(0.0, s + (rng.standard_normal() + shift) - k)
        if s > h:
            return n
    return n
