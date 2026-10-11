"""Calibrating the line twin from the plant's event log, with exact confidence intervals.

With exponential times the rate of an event type is ``k / T`` for ``k`` events in an exposure ``T``
(right-censored last period included). The exact (Garwood) interval comes from the Gamma distribution
with integer shape, i.e. from Poisson sums, so no special-function library is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .line import Line


def gamma_ppf_int(p, shape):
    """Quantile of the Gamma(shape, 1) distribution for integer ``shape``, by bisection on the Poisson identity
    ``P(Gamma(n) <= x) = 1 - P(Poisson(x) <= n - 1)`` (summed in log space)."""
    if shape < 1:
        return 0.0
    j = np.arange(shape)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, shape)))])    # log j!, j = 0 .. shape - 1

    def cdf(x):
        terms = -x + j * math.log(x) - log_fact
        top = terms.max()
        return 1.0 - math.exp(top + math.log(np.exp(terms - top).sum()))

    lo, hi = 1e-12, float(shape) + 10.0 * math.sqrt(shape) + 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rate_interval(events, exposure, confidence=0.95):
    """Exact interval for an event rate from ``events`` in ``exposure``: ``(rate, low, high)``."""
    a = (1.0 - confidence) / 2.0
    low = gamma_ppf_int(a, events) / exposure if events > 0 else 0.0
    high = gamma_ppf_int(1.0 - a, events + 1) / exposure
    return events / exposure, low, high


@dataclass
class StationEstimate:
    mean_service: tuple             # (estimate, low, high)
    mtbf: tuple                     # (estimate, low, high) or None
    mttr: tuple
    n_services: int
    n_failures: int


def estimate(result, line: Line, confidence=0.95):
    """Estimates and intervals of every station's mean service time, MTBF and MTTR from one run's log."""
    out = []
    log = result.log
    for i, st in enumerate(line.stations):
        services = np.asarray(log["service"][i], dtype=float)
        n, total = len(services), float(services.sum())
        r, lo, hi = rate_interval(n, total, confidence)
        mean_service = (1.0 / r, 1.0 / hi, 1.0 / lo if lo > 0 else float("inf"))
        mtbf = mttr = None
        if math.isfinite(st.mtbf) and log["n_fail"][i] > 0 and log["n_repair"][i] > 0:
            r1, l1, h1 = rate_interval(int(log["n_fail"][i]), float(log["up_time"][i]), confidence)
            r2, l2, h2 = rate_interval(int(log["n_repair"][i]), float(log["down_time"][i]), confidence)
            mtbf = (1.0 / r1, 1.0 / h1, 1.0 / l1)
            mttr = (1.0 / r2, 1.0 / h2, 1.0 / l2)
        out.append(StationEstimate(mean_service, mtbf, mttr, n, int(log["n_fail"][i])))
    return out


def fit_distribution(services):
    """Pick a service-time family that matches the log's mean and variability.

    Squared coefficient of variation near 1 (within three sampling errors, at least 0.1) -> exponential;
    below -> Erlang with ``k = round(1 / cv^2)`` (deterministic once ``k`` exceeds 50); above -> lognormal.
    """
    x = np.asarray(services, dtype=float)
    n = len(x)
    if n < 200:
        return "exp"
    cv2 = float(x.var(ddof=1) / x.mean() ** 2)
    tol = max(0.1, 3.0 * math.sqrt(8.0 / n))
    if abs(cv2 - 1.0) <= tol:
        return "exp"
    if cv2 < 1.0:
        if cv2 < 1.0 / 50.0:
            return "det"
        return f"erlang:{max(int(round(1.0 / cv2)), 2)}"
    return f"lognorm:{math.sqrt(cv2):.3f}"


def calibrated_line(result, line: Line, fit_dist=True):
    """The twin's line: structure and buffers as installed, rates (and, with ``fit_dist``, the service-time family
    and scrap fraction) replaced by what the log shows."""
    est = estimate(result, line)
    log = result.log
    stations = []
    for i, (st, e) in enumerate(zip(line.stations, est)):
        changes = {"mean_service": e.mean_service[0], "dist": "exp"}
        if fit_dist:
            changes["dist"] = fit_distribution(log["service"][i])
            changes["scrap"] = float(log["scrap"][i] / e.n_services) if e.n_services else 0.0
        if e.mtbf is not None:
            changes.update(mtbf=e.mtbf[0], mttr=e.mttr[0])
        stations.append(replace(st, **changes))
    return replace(line, stations=tuple(stations))


def sample_lines(result, line: Line, draws, rng):
    """Lines whose rates are drawn from their Jeffreys posteriors ``Gamma(k + 1/2) / exposure`` (a parametric bootstrap)."""
    log = result.log
    base = calibrated_line(result, line)
    lines = []
    for _ in range(draws):
        stations = []
        for i, st in enumerate(base.stations):
            n = len(log["service"][i])
            total = float(np.sum(log["service"][i]))
            changes = {"mean_service": total / rng.gamma(n + 0.5)}
            if math.isfinite(st.mtbf):
                changes["mtbf"] = float(log["up_time"][i]) / rng.gamma(int(log["n_fail"][i]) + 0.5)
                changes["mttr"] = float(log["down_time"][i]) / rng.gamma(int(log["n_repair"][i]) + 0.5)
            stations.append(replace(st, **changes))
        lines.append(replace(base, stations=tuple(stations)))
    return lines
