"""Model checks: each part of the twin against an analytic or independent result.

``run_all()`` returns a list of :class:`Check`; the CLI prints them (``dogrulama``) and the tests assert
them. Everything is seeded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import calibrate, ctmc, kalman, linetwin, machine, monitor, rul, scenarios, twin
from . import line as L


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


# ----------------------------------------------------------------------------------------------
# Kalman filter
def _model_twin(dt=0.5):
    return twin.MachineTwin(machine.MachineParams(), dt, scenarios.TEMP_SENSOR, scenarios.VIB_SENSOR)


def _synthetic(mt, rng, steps, missing=0.1):
    """Data drawn from the twin's own linear-Gaussian model (severity as a random walk)."""
    x0, P0 = mt.initial(np.array([np.nan, np.nan]), 0.5)
    x = rng.multivariate_normal(x0, P0)
    u = np.asarray(machine.daily_load(np.arange(steps) * mt.dt))
    H, c, R, Q = mt.H(), mt.c(), mt.R, mt.Q()
    xs, ys = [x], []
    for k in range(steps):
        y = H @ x + c + rng.multivariate_normal(np.zeros(2), R)
        if missing:
            y = np.where(rng.random(2) < missing, np.nan, y)
        ys.append(y)
        if k < steps - 1:
            x = mt.F(u[k]) @ x + mt.b(u[k]) + rng.multivariate_normal(np.zeros(3), Q)
            xs.append(x)
    return u, np.array(xs), np.array(ys), (x0, P0)


def _run_filter(mt, u, ys, prior):
    kf = kalman.KalmanFilter(*prior)
    H, c, R, Q = mt.H(), mt.c(), mt.R, mt.Q()
    for k in range(len(ys)):
        kf.update(ys[k], H, c, R)
        if k < len(ys) - 1:
            kf.predict(mt.F(u[k]), mt.b(u[k]), Q)
    return kf


def kalman_vs_batch(seed=0, steps=26):
    """The filter's final estimate and covariance equal the weighted least-squares solution over the window."""
    mt = _model_twin()
    rng = np.random.default_rng(seed)
    u, xs, ys, prior = _synthetic(mt, rng, steps, missing=0.2)
    kf = _run_filter(mt, u, ys, prior)
    n = steps
    est, cov = kalman.batch_estimate(prior[0], prior[1], [mt.F(u[k]) for k in range(n - 1)],
                                     [mt.b(u[k]) for k in range(n - 1)], [mt.Q()] * (n - 1), ys,
                                     [mt.H()] * n, [mt.c()] * n, [mt.R] * n)
    dx = float(np.abs(kf.x - est).max())
    dP = float(np.abs(kf.P - cov).max() / np.abs(cov).max())
    return Check("Kalman filter vs batch least squares", dx < 1e-8 and dP < 1e-8,
                 f"{steps} steps with 20% lost samples: largest estimate difference {dx:.1e}, relative covariance difference {dP:.1e}")


def kalman_consistency(seed=0, runs=150, steps=100):
    """On data from the model itself, the normalised estimation error squared averages the state dimension (3)."""
    mt = _model_twin()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(runs):
        u, xs, ys, prior = _synthetic(mt, rng, steps)
        kf = _run_filter(mt, u, ys, prior)
        values.append(kalman.nees(xs[-1] - kf.x, kf.P))
    values = np.array(values)
    mean, se = values.mean(), values.std(ddof=1) / math.sqrt(runs)
    inside = float(np.mean(values < 7.815))         # 95% quantile of chi-square with 3 dof
    # scalar random walk: the filter's steady-state prior variance against the closed form
    q, r = 0.02, 0.5
    kf = kalman.KalmanFilter([0.0], [[1.0]])
    F, Hm, Q, R = np.eye(1), np.eye(1), np.array([[q]]), np.array([[r]])
    for _ in range(400):
        kf.update(np.array([0.0]), Hm, np.zeros(1), R)
        kf.predict(F, np.zeros(1), Q)
    closed = kalman.scalar_steady_state(q, r)
    ok = abs(mean - 3.0) < 3.0 * se and abs(inside - 0.95) < 3.0 * math.sqrt(0.95 * 0.05 / runs) and abs(kf.P[0, 0] - closed) < 1e-9
    return Check("Kalman filter consistency", ok,
                 f"{runs} runs: mean NEES {mean:.2f} (expected 3, standard error {se:.2f}), {100 * inside:.0f}% inside the 95% chi-square bound; "
                 f"scalar steady state {kf.P[0, 0]:.6f} vs closed form {closed:.6f}")


def innovations_white(seed=0):
    """On the simulated plant the standardised innovations have mean 0, standard deviation 1 and no lag-1 correlation."""
    params = machine.MachineParams()
    truth = machine.simulate_plant(params, 800.0, 0.25, seed)
    rng = np.random.default_rng(seed + 1)
    y_t, y_v = scenarios.observe(truth, rng)
    run = twin.MachineTwin(params, 0.25, scenarios.TEMP_SENSOR, scenarios.VIB_SENSOR).run(truth.t, truth.u, y_t, y_v)
    rows, ok = [], True
    for ch, name in enumerate(("temperature", "vibration")):
        z = run.z[50:, ch]
        z = z[~np.isnan(z)]
        n = len(z)
        mean, sd = float(z.mean()), float(z.std())
        rho = float(np.corrcoef(z[:-1], z[1:])[0, 1])
        ok &= abs(mean) < 3.0 / math.sqrt(n) and abs(sd - 1.0) < 0.06 and abs(rho) < 3.0 / math.sqrt(n)
        rows.append(f"{name} mean {mean:+.3f} sd {sd:.3f} lag-1 {rho:+.3f}")
    nis = float(np.nanmean(run.nis[50:]))
    return Check("innovations are white N(0,1)", ok and abs(nis - 2.0) < 0.15, "; ".join(rows) + f"; mean NIS {nis:.2f} (expected 2)")


# ----------------------------------------------------------------------------------------------
# Control charts and remaining life
def cusum_arl_theory(seed=0):
    """CUSUM run lengths: simulation against the Brook-Evans chain and Siegmund's approximation; Shewhart exactly."""
    rng = np.random.default_rng(seed)
    rows, ok = [], True
    for k, h, shift in ((0.5, 4.0, 0.0), (0.5, 5.0, 0.0), (0.5, 4.0, 1.0), (0.5, 4.0, 2.0)):
        runs = np.array([monitor.run_length(k, h, shift, rng) for _ in range(3000)])
        sim, se = runs.mean(), runs.std() / math.sqrt(len(runs))
        be = monitor.cusum_arl_brook_evans(k, h, shift)
        sg = monitor.cusum_arl_siegmund(k, h, shift)
        ok &= abs(sim - be) < 3.5 * se and abs(sg - be) / be < 0.05
        rows.append(f"h={h:g} shift={shift:g}: sim {sim:.1f}±{se:.1f}, Brook-Evans {be:.1f}, Siegmund {sg:.1f}")
    z = rng.standard_normal(2_000_000)
    p_hat = float(np.mean(np.abs(z) > 3.0))
    exact = 1.0 / monitor.shewhart_arl(3.0)
    ok &= abs(p_hat - exact) < 4.0 * math.sqrt(exact / len(z))
    # the streaming class (reset mode) against the two-sided theory
    two = monitor.cusum_two_sided_arl(0.5, 4.0)
    lengths = []
    for _ in range(600):
        ch, n = monitor.Cusum(0.5, 4.0, latch=False), 0
        while True:
            n += 1
            if ch.update(float(rng.standard_normal())):
                break
        lengths.append(n)
    sim2, se2 = np.mean(lengths), np.std(lengths) / math.sqrt(len(lengths))
    ok &= abs(sim2 - two) / two < 0.08
    return Check("CUSUM and Shewhart run lengths", ok,
                 "; ".join(rows) + f"; Shewhart |z|>3 false-alarm rate {p_hat:.5f} vs {exact:.5f}; two-sided class "
                 f"{sim2:.0f}±{se2:.0f} vs {two:.0f}")


def _ks(sorted_x, cdf_values):
    n = len(sorted_x)
    return float(max(np.max(np.arange(1, n + 1) / n - cdf_values), np.max(cdf_values - np.arange(n) / n)))


def inverse_gaussian_first_passage(seed=0):
    """Simulated wear paths against the inverse-Gaussian first-passage law (cdf, moments and sampler).

    A single Kolmogorov-Smirnov test fails by chance one time in twenty, so six independent batches of paths
    are tested together: under the null the mean distance is ``0.87 / sqrt(n)`` with spread ``0.11 / sqrt(n)``,
    and the bound used is ``1.2 / sqrt(n)``.
    """
    rng = np.random.default_rng(seed)
    mu, sigma, w0 = 1.2e-3, 0.004, 0.05
    b = 1.0 - w0
    mean, shape = b / mu, b * b / sigma ** 2
    n, batches = 1500, 6
    distances, all_times = [], []
    for _ in range(batches):
        x = np.sort(rul.first_passage_paths(w0, mu, sigma, 1.0, 0.5, n, rng))
        distances.append(_ks(x, np.array([rul.ig_cdf(t, mean, shape) for t in x])))
        all_times.append(x)
    times = np.concatenate(all_times)
    samples = np.sort(rul.ig_sample(mean, shape, 4000, rng))
    d_samp = _ks(samples, np.array([rul.ig_cdf(t, mean, shape) for t in samples]))
    d_mean, bound = float(np.mean(distances)), 1.2 / math.sqrt(n)
    mean_err = abs(times.mean() - mean) / mean
    var_ratio = times.var() / (mean ** 3 / shape)
    q = rul.ig_quantile(0.9, mean, shape)
    ok = d_mean < bound and d_samp < 1.36 / math.sqrt(len(samples)) and mean_err < 0.01 and abs(var_ratio - 1) < 0.1 \
        and abs(rul.ig_cdf(q, mean, shape) - 0.9) < 1e-9
    return Check("remaining life: inverse Gaussian first passage", ok,
                 f"{batches * n} simulated paths in {batches} batches: mean KS distance {d_mean:.4f} (bound {bound:.4f}); "
                 f"sampler KS {d_samp:.4f} (5% critical {1.36 / math.sqrt(len(samples)):.4f}); mean {times.mean():.0f} vs {mean:.0f}, "
                 f"variance ratio {var_ratio:.3f}")


def rul_calibration(seed=0, machines=60, check_at=150.0, dt=0.5):
    """Coverage of the 80% remaining-life interval over machines with different wear severity."""
    params = machine.MachineParams()
    tw = twin.MachineTwin(params, dt, scenarios.TEMP_SENSOR, scenarios.VIB_SENSOR)
    cover = {True: 0, False: 0}
    used = 0
    for s in range(machines):
        rng = np.random.default_rng(seed + s)
        kappa = rng.uniform(0.7, 1.4)
        truth = machine.simulate_plant(params, 3000.0, dt, seed + s, kappa=kappa, stop_at_failure=True)
        if truth.fail_index < 0 or truth.fail_time <= check_at:
            continue
        ic = int(check_at / dt)
        y_t, y_v = scenarios.observe(truth, np.random.default_rng(seed + s + 50_000))
        run = tw.run(truth.t[:ic + 1], truth.u[:ic + 1], y_t[:ic + 1], y_v[:ic + 1])
        true = truth.fail_time - check_at
        used += 1
        for propagate in (True, False):
            _, lo, hi = run.rul(ic, propagate=propagate, seed=s)
            cover[propagate] += lo <= true <= hi
    prop, plug = cover[True] / used, cover[False] / used
    return Check("remaining-life interval calibration", prop >= 0.75 and plug < 0.65,
                 f"{used} machines judged at {check_at:g} h: 80% interval covers the true remaining life "
                 f"{100 * prop:.0f}% with the estimation uncertainty propagated, {100 * plug:.0f}% for the plug-in distribution")


def rul_after_fault(q_kappa=0.03, after=100.0, machines=40, seed=0, dt=0.5, fault_start=200.0):
    """Coverage of the 80% remaining-life interval ``after`` hours after an abrupt x3 bearing fault.

    Not a pass/fail check: it measures how the twin's responsiveness setting ``q_kappa`` trades off against
    the lag of its wear-rate estimate (returns ``(machines used, coverage, median estimate / true life)``).
    """
    params = machine.MachineParams()
    tw = twin.MachineTwin(params, dt, scenarios.TEMP_SENSOR, scenarios.VIB_SENSOR, q_kappa=q_kappa)
    cover = used = 0
    ratios = []
    t_check = fault_start + after
    for s in range(machines):
        rng = np.random.default_rng(seed + s)
        truth = machine.simulate_plant(params, 3000.0, dt, seed + s, (machine.Fault("bearing", fault_start, 3.0),),
                                       kappa=rng.uniform(0.7, 1.4), stop_at_failure=True)
        if truth.fail_index < 0 or truth.fail_time <= t_check + 5.0:
            continue
        ic = int(t_check / dt)
        y_t, y_v = scenarios.observe(truth, np.random.default_rng(seed + s + 50_000))
        run = tw.run(truth.t[:ic + 1], truth.u[:ic + 1], y_t[:ic + 1], y_v[:ic + 1])
        med, lo, hi = run.rul(ic, seed=s)
        true = truth.fail_time - t_check
        used += 1
        cover += lo <= true <= hi
        ratios.append(med / true)
    return used, cover / used, float(np.median(ratios))


def healthy_kappa_noise(q_kappa=0.03, machines=10, hours=700.0, dt=0.5, seed=0):
    """How noisy is the wear-rate estimate on healthy machines? Returns ``(rms of kappa_hat - 1, mean of the
    per-machine maximum, largest maximum)`` after a 100 h burn-in; the wear-rate alarm level is 1.8."""
    params = machine.MachineParams()
    tw = twin.MachineTwin(params, dt, scenarios.TEMP_SENSOR, scenarios.VIB_SENSOR, q_kappa=q_kappa)
    rms, peaks = [], []
    for s in range(machines):
        truth = machine.simulate_plant(params, hours, dt, seed + s, stop_at_failure=True)
        y_t, y_v = scenarios.observe(truth, np.random.default_rng(seed + s + 50_000))
        run = tw.run(truth.t, truth.u, y_t, y_v)
        k = run.kappa[int(100.0 / dt):]
        rms.append(np.mean((k - 1.0) ** 2))
        peaks.append(k.max())
    return float(np.sqrt(np.mean(rms))), float(np.mean(peaks)), float(np.max(peaks))


def fault_detection(seed=0):
    """Detection delays for injected faults, and the false-alarm rate on healthy machines against the CUSUM design."""
    rows, ok = [], True
    # cooling degradation: immediate temperature alarm
    delays = [scenarios.run_makine("sogutma", hours=420.0, seed=seed + s).summary["tespit_gecikmesi_saat"] for s in range(4)]
    ok &= all(d < 2.0 for d in delays)
    rows.append(f"cooling x1.3: median delay {np.median(delays):.2f} h")
    # bearing wear acceleration: wear-rate alarm before failure
    res = [scenarios.run_makine("yatak", hours=700.0, seed=seed + s).summary for s in range(6)]
    found = [r["tespit_gecikmesi_saat"] for r in res if not math.isnan(r["tespit_gecikmesi_saat"])]
    before = [r["ariza_saati"] - 300.0 - r["tespit_gecikmesi_saat"] for r in res if not math.isnan(r["tespit_gecikmesi_saat"])]
    ok &= len(found) >= 5 and min(before) > 0
    rows.append(f"bearing x3: {len(found)}/6 detected, median delay {np.median(found):.0f} h, "
                f"{np.median(before):.0f} h of life left at the alarm")
    # false alarms: exposure until the first alarm against the design run length, exact Poisson interval
    arl0, events, exposure = 500.0, 0, 0
    for s in range(24):
        params = machine.MachineParams()
        truth = machine.simulate_plant(params, 400.0, 0.5, seed + 100 + s)
        y_t, y_v = scenarios.observe(truth, np.random.default_rng(seed + 900 + s))
        run = twin.MachineTwin(params, 0.5, scenarios.TEMP_SENSOR, scenarios.VIB_SENSOR, arl0=arl0).run(
            truth.t, truth.u, y_t, y_v)
        first = [a for a in run.alarms if a[1] in ("T+", "T-", "v+", "v-")]
        if first:
            events += 1
            exposure += int(round(first[0][0] / 0.5)) + 1
        else:
            exposure += len(truth.t)
    rate, lo, hi = calibrate.rate_interval(events, exposure, confidence=0.99)
    theory = 2.0 / arl0                                     # two channels, each with run length arl0
    ok &= lo <= theory <= hi
    rows.append(f"healthy: {events} first alarms in {exposure} samples = {rate:.4f}/sample "
                f"(99% interval [{lo:.4f}, {hi:.4f}]) vs design {theory:.4f}")
    return Check("fault detection and false alarms", ok, "; ".join(rows))


# ----------------------------------------------------------------------------------------------
# Production line
def line_exact_vs_closed_forms():
    """The Markov-chain solver against textbook closed forms."""
    worst = 0.0
    for mu1, mu2, n in ((1.0, 1.0, 0), (1.0, 1.0, 3), (1.0, 0.8, 2), (0.7, 1.0, 4), (2.0, 1.0, 5)):
        ln = L.Line((L.Station("a", 1 / mu1), L.Station("b", 1 / mu2)), (n,))
        worst = max(worst, abs(ctmc.solve(ln).throughput - ctmc.two_station_closed_form(mu1, mu2, n)))
    identical = abs(ctmc.solve(L.Line((L.Station("a", 1.0), L.Station("b", 1.0)), (0,))).throughput - 2.0 / 3.0)
    mm = L.Line((L.Station("s", 1.0),), (), arrival_rate=0.8, queue_capacity=4)
    r = ctmc.solve(mm)
    loss, mean_n = ctmc.mm1k(0.8, 1.0, 5)
    single = L.Line((L.Station("s", 0.5, mtbf=40.0, mttr=10.0),), ())
    avail = abs(ctmc.solve(single).throughput - single.stations[0].capacity)
    ok = worst < 1e-10 and identical < 1e-10 and abs(r.loss_probability - loss) < 1e-10 and abs(r.wip - mean_n) < 1e-10 and avail < 1e-10
    return Check("exact line solver vs closed forms", ok,
                 f"two reliable stations (birth-death chain) worst difference {worst:.1e}; no buffer, identical rates 2/3: {identical:.1e}; "
                 f"M/M/1/5 loss {r.loss_probability:.5f} vs {loss:.5f}, mean number {r.wip:.4f} vs {mean_n:.4f}; "
                 f"single unreliable machine throughput = rate x availability: {avail:.1e}")


def line_des_vs_exact(seed=0):
    """Replicated event simulation against the exact stationary solution."""
    lines = {
        "two stations, reliable": L.Line((L.Station("a", 1.0), L.Station("b", 1.2)), (2,)),
        "two stations, failures": L.Line((L.Station("a", 1.0, mtbf=40.0, mttr=5.0), L.Station("b", 1.2, mtbf=60.0, mttr=8.0)), (3,)),
        "three stations": L.Line((L.Station("a", 1.0, mtbf=50.0, mttr=5.0), L.Station("b", 1.1),
                                  L.Station("c", 0.9, mtbf=70.0, mttr=7.0)), (2, 2)),
        "Poisson source": L.Line((L.Station("a", 1.0, mtbf=80.0, mttr=6.0), L.Station("b", 0.9)), (2,), arrival_rate=0.7, queue_capacity=4),
    }
    rows, ok = [], True
    for k, (name, ln) in enumerate(lines.items()):
        exact = ctmc.solve(ln)
        res = L.replicate(ln, 20000.0, 10, warmup=500.0, seed=seed + 100 * k)
        th, se = L.mean_and_se([r.throughput for r in res])
        z = (th - exact.throughput) / se
        wip = np.mean([r.wip for r in res])
        busy = np.mean([r.busy for r in res], axis=0)
        ok &= abs(z) < 4.5 and abs(wip / exact.wip - 1) < 0.03 and np.abs(busy - exact.busy).max() < 0.01
        rows.append(f"{name}: throughput {th:.4f}±{se:.4f} vs {exact.throughput:.4f} (z {z:+.1f}), WIP {wip:.3f} vs {exact.wip:.3f}")
    return Check("event simulation vs exact solution", ok, "; ".join(rows))


def line_mg1_and_little(seed=0):
    """Pollaczek-Khinchine mean wait for M/G/1 with four service laws, and Little's law."""
    rows, ok = [], True
    for dist, cv2 in (("exp", 1.0), ("det", 0.0), ("erlang:2", 0.5), ("lognorm:1.0", 1.0)):
        rho, mean = 0.8, 1.0
        ln = L.Line((L.Station("s", mean, dist=dist),), (), arrival_rate=rho / mean)
        theory = (rho / mean) * mean ** 2 * (1.0 + cv2) / (2.0 * (1.0 - rho))
        res = L.replicate(ln, 40000.0, 8, warmup=1000.0, seed=seed)
        w, se = L.mean_and_se([r.mean_wait for r in res])
        little = max(abs(r.little_residual) for r in res)
        ok &= abs(w - theory) < 5.0 * se and little < 0.01
        rows.append(f"{dist}: wait {w:.3f}±{se:.3f} vs {theory:.3f}")
    return Check("M/G/1 mean wait and Little's law", ok, "; ".join(rows) + "; Little residual below 1% in every run")


def calibration_coverage(seed=0, runs=40):
    """Coverage of the exact rate intervals, on pure exponential samples and on event logs of a simulated line."""
    rng = np.random.default_rng(seed)
    hit = 0
    n_pure = 400
    for _ in range(n_pure):
        k = int(rng.integers(3, 40))
        rate = rng.uniform(0.1, 3.0)
        total = rng.gamma(k, 1.0 / rate)                      # time of the k-th event
        _, lo, hi = calibrate.rate_interval(k, total)
        hit += lo <= rate <= hi
    pure = hit / n_pure
    truth_line = L.Line((L.Station("a", 1.0, mtbf=40.0, mttr=5.0), L.Station("b", 1.2, mtbf=60.0, mttr=8.0)), (3,))
    cover, total_params = 0, 0
    for r in range(runs):
        res = L.simulate(truth_line, 6000.0, 0.0, seed + r)
        for st, e in zip(truth_line.stations, calibrate.estimate(res, truth_line)):
            for value, (_, lo, hi) in ((st.mean_service, e.mean_service), (st.mtbf, e.mtbf), (st.mttr, e.mttr)):
                cover += lo <= value <= hi
                total_params += 1
    logs = cover / total_params
    return Check("calibration interval coverage", pure >= 0.94 and logs >= 0.90,
                 f"exact 95% rate intervals contain the true rate in {100 * pure:.1f}% of {n_pure} exponential samples and "
                 f"{100 * logs:.1f}% of {total_params} service/MTBF/MTTR intervals from {runs} simulated event logs")


def _binom_cdf(k, n, p):
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1)))


def twin_prediction_coverage(seed=0, runs=30):
    """The calibrated line twin's 90% throughput interval against the exact throughput of the true line."""
    truth_line = L.Line((L.Station("a", 1.0, mtbf=60.0, mttr=8.0), L.Station("b", 1.15, mtbf=90.0, mttr=10.0)), (2,))
    exact = ctmc.solve(truth_line).throughput
    cover, errs, widths = 0, [], []
    for r in range(runs):
        plant = L.simulate(truth_line, 6000.0, 0.0, seed + r)
        p = linetwin.predictive_interval(plant, truth_line, draws=60, level=0.9, seed=seed + r)
        cover += p.low <= exact <= p.high
        errs.append(p.estimate - exact)
        widths.append(p.high - p.low)
    rate = cover / runs
    p_value = _binom_cdf(cover, runs, 0.9)          # chance of so few hits if the interval were exactly calibrated
    return Check("line twin prediction vs exact truth", p_value > 0.001,
                 f"{runs} plant logs of 6000 min: the twin's 90% interval contains the exact throughput {exact:.4f} "
                 f"in {cover}/{runs} runs ({100 * rate:.0f}%; binomial lower-tail p = {p_value:.2f} for a calibrated 90%); "
                 f"mean error {np.mean(errs):+.4f}, mean width {np.mean(widths):.4f}")


def whatif_and_bottleneck():
    """Speeding the bottleneck gains most; throughput rises with buffer size but stays below the bottleneck's capacity."""
    ln = scenarios.example_line("montaj")
    base, gains = linetwin.speedup_ranking(ln, 1.1)
    bottleneck = ln.bottleneck
    sizes = [0, 1, 2, 4, 8, 16]
    th = linetwin.buffer_study(ln, 1, sizes)
    bound = linetwin.capacity_bound(ln)
    ok = int(np.argmax(gains)) == bottleneck and np.all(np.diff(th) > 0) and th[-1] < bound and \
        all(g > 0 for g in gains)
    return Check("what-if analysis", ok,
                 f"10% speed-up gains {[round(float(g), 4) for g in gains]} (largest at {ln.stations[int(np.argmax(gains))].name}, "
                 f"the capacity bottleneck is {ln.stations[bottleneck].name}); throughput for buffer sizes {sizes}: "
                 f"{[round(v, 3) for v in th]}, below the capacity bound {bound:.3f}")


ALL = (kalman_vs_batch, kalman_consistency, innovations_white, cusum_arl_theory, inverse_gaussian_first_passage,
       rul_calibration, fault_detection, line_exact_vs_closed_forms, line_des_vs_exact, line_mg1_and_little,
       calibration_coverage, twin_prediction_coverage, whatif_and_bottleneck)


def run_all():
    return [check() for check in ALL]
