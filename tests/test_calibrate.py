import math

import numpy as np
import pytest

from fabrika_ikizi import calibrate
from fabrika_ikizi import line as L
from fabrika_ikizi.line import Line, Station


def test_garwood_intervals_match_published_values():
    # exact 95% Poisson intervals for the count (Garwood): k = 1 -> [0.0253, 5.572], k = 10 -> [4.795, 18.39], k = 0 -> [0, 3.689]
    r, lo, hi = calibrate.rate_interval(1, 1.0)
    assert (lo, hi) == pytest.approx((0.0253, 5.572), abs=2e-3)
    r, lo, hi = calibrate.rate_interval(10, 1.0)
    assert (r, lo, hi) == pytest.approx((10.0, 4.795, 18.39), abs=2e-3)
    r, lo, hi = calibrate.rate_interval(0, 1.0)
    assert (r, lo) == (0.0, 0.0) and hi == pytest.approx(-math.log(0.025), abs=1e-6)


def test_interval_scales_with_exposure_and_confidence():
    a = calibrate.rate_interval(10, 1.0)
    b = calibrate.rate_interval(10, 4.0)
    assert np.allclose(np.array(a) / 4.0, b)
    narrow, wide = calibrate.rate_interval(30, 10.0, 0.8), calibrate.rate_interval(30, 10.0, 0.99)
    assert wide[1] < narrow[1] < narrow[0] < narrow[2] < wide[2]


def test_gamma_quantile_is_the_inverse_of_the_poisson_identity():
    for shape in (1, 5, 40, 500):
        for p in (0.025, 0.5, 0.975):
            x = calibrate.gamma_ppf_int(p, shape)
            j = np.arange(shape)
            log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, shape)))])
            cdf = 1.0 - np.exp(-x + j * math.log(x) - log_fact).sum()
            assert cdf == pytest.approx(p, abs=1e-8)
    assert calibrate.gamma_ppf_int(0.5, 1) == pytest.approx(math.log(2.0))           # exponential median


def test_large_counts_are_handled_and_close_to_the_normal_approximation():
    r, lo, hi = calibrate.rate_interval(6000, 6000.0)
    assert (hi - lo) / 2.0 == pytest.approx(1.96 / math.sqrt(6000), rel=0.02)


def test_exact_intervals_cover_the_true_rate_at_the_nominal_level():
    rng = np.random.default_rng(0)
    hit = 0
    for _ in range(600):
        k = int(rng.integers(2, 30))
        total = rng.gamma(k, 1.0 / 1.7)
        _, lo, hi = calibrate.rate_interval(k, total)
        hit += lo <= 1.7 <= hi
    assert 0.93 < hit / 600 < 0.995


@pytest.fixture(scope="module")
def plant():
    line = Line((Station("a", 1.0, mtbf=40.0, mttr=5.0), Station("b", 1.2, mtbf=60.0, mttr=8.0)), (3,))
    return line, L.simulate(line, 30000.0, 0.0, seed=1)


def test_estimates_recover_the_true_parameters_inside_their_intervals(plant):
    line, res = plant
    for st, e in zip(line.stations, calibrate.estimate(res, line)):
        for truth, (est, lo, hi) in ((st.mean_service, e.mean_service), (st.mtbf, e.mtbf), (st.mttr, e.mttr)):
            assert lo <= truth <= hi and abs(est / truth - 1) < 0.1
        assert e.n_services > 1000 and e.n_failures > 100


def test_calibrated_line_keeps_the_structure_and_replaces_the_rates(plant):
    line, res = plant
    cal = calibrate.calibrated_line(res, line)
    assert cal.buffers == line.buffers and len(cal.stations) == 2
    assert [s.name for s in cal.stations] == ["a", "b"]
    assert all(s.dist == "exp" for s in cal.stations)
    assert cal.stations[0].mean_service != line.stations[0].mean_service
    assert cal.stations[0].mtbf == pytest.approx(40.0, rel=0.15)


def test_reliable_stations_get_no_failure_parameters():
    line = Line((Station("a", 1.0), Station("b", 1.2, mtbf=60.0, mttr=8.0)), (2,))
    res = L.simulate(line, 20000.0, 0.0, seed=2)
    est = calibrate.estimate(res, line)
    assert est[0].mtbf is None and est[1].mtbf is not None
    assert math.isinf(calibrate.calibrated_line(res, line).stations[0].mtbf)


@pytest.mark.parametrize("dist,expected_prefix", [("exp", "exp"), ("det", "det"), ("erlang:4", "erlang:"), ("lognorm:1.5", "lognorm:")])
def test_distribution_fitting_picks_the_matching_family(dist, expected_prefix):
    rng = np.random.default_rng(1)
    draw = L.sampler(dist, 1.0)
    x = [draw(rng) for _ in range(5000)]
    fitted = calibrate.fit_distribution(x)
    assert fitted.startswith(expected_prefix)
    if fitted.startswith("erlang:"):
        assert int(fitted.split(":")[1]) in (3, 4, 5)
    assert calibrate.fit_distribution(x[:50]) == "exp"                  # too little data to say otherwise


def test_fitted_line_recovers_scrap_and_variability():
    line = Line((Station("a", 1.0, dist="erlang:4", scrap=0.05), Station("b", 1.1, dist="det")), (3,))
    res = L.simulate(line, 20000.0, 0.0, seed=3)
    cal = calibrate.calibrated_line(res, line)
    assert cal.stations[0].dist.startswith("erlang:") and cal.stations[1].dist == "det"
    assert cal.stations[0].scrap == pytest.approx(0.05, abs=0.01)
    assert calibrate.calibrated_line(res, line, fit_dist=False).stations[0].scrap == 0.05      # untouched without fitting


def test_sampled_lines_scatter_around_the_calibrated_one(plant):
    line, res = plant
    rng = np.random.default_rng(0)
    draws = calibrate.sample_lines(res, line, 200, rng)
    means = np.array([s.stations[0].mean_service for s in draws])
    cal = calibrate.calibrated_line(res, line)
    assert means.mean() == pytest.approx(cal.stations[0].mean_service, rel=0.02) and means.std() > 0
    assert all(s.buffers == line.buffers for s in draws)
