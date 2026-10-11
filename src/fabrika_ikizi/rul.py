"""Remaining useful life: the first passage of a Wiener process with drift to a wear limit.

For ``dw = mu dt + sigma dW`` started at ``w`` and a limit ``L`` the time to reach it is inverse
Gaussian with mean ``(L - w) / mu`` and shape ``(L - w)^2 / sigma^2``. The cdf is closed form; sampling
uses the Michael-Schucany-Haas transformation; both are checked against simulated paths.
"""
from __future__ import annotations

import math

import numpy as np

from .monitor import phi_cdf


def _log_phi_neg(x):
    """``log Phi(-x)`` for large positive ``x`` without underflow."""
    if x < 25.0:
        return math.log(0.5 * math.erfc(x / math.sqrt(2.0)))
    return -0.5 * x * x - math.log(x * math.sqrt(2.0 * math.pi)) + math.log1p(-1.0 / (x * x))


def ig_cdf(t, mean, shape):
    """Cdf of the inverse Gaussian distribution with the given mean and shape."""
    if t <= 0:
        return 0.0
    s = math.sqrt(shape / t)
    first = phi_cdf(s * (t / mean - 1.0))
    second = math.exp(2.0 * shape / mean + _log_phi_neg(s * (t / mean + 1.0)))
    return min(1.0, first + second)


def ig_quantile(p, mean, shape):
    """Quantile by bisection on the cdf."""
    lo, hi = 0.0, mean
    while ig_cdf(hi, mean, shape) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ig_cdf(mid, mean, shape) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ig_sample(mean, shape, size, rng):
    """Draw from the inverse Gaussian (Michael, Schucany and Haas, 1976); ``mean`` may be an array."""
    mean = np.asarray(mean, dtype=float)
    nu = rng.standard_normal(size)
    y = nu * nu
    x = mean + mean * mean * y / (2.0 * shape) - mean / (2.0 * shape) * np.sqrt(4.0 * mean * shape * y + mean * mean * y * y)
    z = rng.random(size)
    x = np.maximum(x, 1e-300)
    return np.where(z <= mean / (mean + x), x, mean * mean / x)


def first_passage_paths(w0, mu, sigma, limit, dt, n_paths, rng, max_steps=200000):
    """Crossing times of simulated Wiener paths, with the exact Brownian-bridge crossing probability inside each step.

    The time of a crossing is taken at the end of its step (error below ``dt``). Returns an array of times
    (NaN for paths that did not cross within ``max_steps``).
    """
    x = np.full(n_paths, float(w0))
    out = np.full(n_paths, np.nan)
    live = np.arange(n_paths)
    sd = sigma * math.sqrt(dt)
    for step in range(1, max_steps + 1):
        if len(live) == 0:
            break
        nxt = x[live] + mu * dt + sd * rng.standard_normal(len(live))
        hit = nxt >= limit
        with np.errstate(over="ignore"):
            bridge = np.exp(-2.0 * (limit - x[live]) * (limit - nxt) / (sigma * sigma * dt))
        hit |= (~hit) & (rng.random(len(live)) < bridge)
        out[live[hit]] = step * dt
        x[live] = nxt
        live = live[~hit]
    return out


def rul_samples(w_hat, kappa_hat, cov, g_future, sigma, size=4000, rng=None, limit=1.0, propagate=True):
    """Samples of the remaining life given the twin's wear and severity estimates.

    ``cov`` is the 2x2 covariance of ``(w, kappa)``. With ``propagate`` the estimation uncertainty is
    sampled too; without it the plug-in distribution (a single inverse Gaussian) is used.
    """
    rng = rng or np.random.default_rng(0)
    if propagate:
        draws = rng.multivariate_normal([w_hat, kappa_hat], cov, size=size)
        w = draws[:, 0]
        kappa = np.maximum(draws[:, 1], 0.05)
    else:
        w = np.full(size, w_hat)
        kappa = np.full(size, max(kappa_hat, 0.05))
    dist = np.maximum(limit - w, 1e-9)
    mu = kappa * g_future
    return ig_sample(dist / mu, dist * dist / (sigma * sigma), size, rng)


def interval(samples, level=0.8):
    lo = (1.0 - level) / 2.0
    return float(np.quantile(samples, lo)), float(np.quantile(samples, 1.0 - lo))
