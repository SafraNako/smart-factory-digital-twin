"""The line twin: predictions and what-if analysis from a calibrated line model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import calibrate, ctmc, line as L


def exact_applicable(line: L.Line):
    try:
        ctmc.check_supported(line)
        return True
    except ValueError:
        return False


def predict_throughput(line: L.Line, horizon=20000.0, replications=8, seed=0):
    """Throughput in parts per minute: exact when the model allows it, otherwise a replicated simulation.

    Returns ``(value, standard_error, method)``; the exact solution has standard error 0.
    """
    if exact_applicable(line):
        return ctmc.solve(line).throughput, 0.0, "exact"
    res = L.replicate(line, horizon, replications, warmup=horizon * 0.05, seed=seed)
    m, se = L.mean_and_se([r.throughput for r in res])
    return m, se, "simulation"


@dataclass
class Prediction:
    estimate: float
    low: float
    high: float
    method: str


def predictive_interval(result: L.LineResult, line: L.Line, draws=200, level=0.9, seed=0):
    """Throughput of the calibrated twin with an interval from the uncertainty of the calibrated rates.

    Rates are drawn from their posteriors; each draw is solved exactly when the model allows it, and
    otherwise simulated briefly (so ``draws`` is capped to keep a simulation-based interval affordable).
    """
    rng = np.random.default_rng(seed)
    calibrated = calibrate.calibrated_line(result, line)
    base, _, method = predict_throughput(calibrated)
    if method == "simulation":
        draws = min(draws, 20)
        values = [predict_throughput(s, replications=2, horizon=4000.0, seed=int(rng.integers(1 << 30)))[0]
                  for s in calibrate.sample_lines(result, line, draws, rng)]
    else:
        values = [predict_throughput(s)[0] for s in calibrate.sample_lines(result, line, draws, rng)]
    a = (1.0 - level) / 2.0
    return Prediction(base, float(np.quantile(values, a)), float(np.quantile(values, 1.0 - a)), method)


def speedup_ranking(line: L.Line, factor=1.1, **kwargs):
    """Throughput gain from making each station ``factor`` times faster (service time divided by ``factor``)."""
    base = predict_throughput(line, **kwargs)[0]
    gains = []
    for i, st in enumerate(line.stations):
        faster = line.with_station(i, mean_service=st.mean_service / factor)
        gains.append(predict_throughput(faster, **kwargs)[0] - base)
    return base, np.array(gains)


def buffer_study(line: L.Line, gap, sizes, **kwargs):
    """Throughput for several capacities of one buffer."""
    return [predict_throughput(line.with_buffer(gap, n), **kwargs)[0] for n in sizes]


def capacity_bound(line: L.Line):
    """Upper bound on throughput (ignoring scrap): the smallest standalone capacity ``availability / mean_service``
    and, for a Poisson source, the arrival rate."""
    bound = min(s.capacity for s in line.stations)
    if line.arrival_rate is not None:
        bound = min(bound, line.arrival_rate)
    return bound
