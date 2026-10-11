import math

import numpy as np
import pytest

from fabrika_ikizi import rul

MEAN, SHAPE = 790.0, 0.95 ** 2 / 0.004 ** 2


def test_ig_cdf_is_a_distribution_function():
    xs = np.linspace(100.0, 3000.0, 60)
    cdf = np.array([rul.ig_cdf(x, MEAN, SHAPE) for x in xs])
    assert np.all(np.diff(cdf) >= 0) and cdf[0] < 1e-6 and cdf[-1] > 1 - 1e-6
    assert rul.ig_cdf(0.0, MEAN, SHAPE) == 0.0 and rul.ig_cdf(-5.0, MEAN, SHAPE) == 0.0


def test_ig_quantile_inverts_the_cdf():
    for p in (0.05, 0.5, 0.95):
        q = rul.ig_quantile(p, MEAN, SHAPE)
        assert rul.ig_cdf(q, MEAN, SHAPE) == pytest.approx(p, abs=1e-9)
    assert rul.ig_quantile(0.1, MEAN, SHAPE) < rul.ig_quantile(0.9, MEAN, SHAPE)


def test_ig_cdf_survives_a_very_large_shape_without_overflow():
    # shape / mean very large: exp(2 shape / mean) would overflow if multiplied naively
    value = rul.ig_cdf(100.0, 100.0, 1e6)
    assert 0.0 <= value <= 1.0 and math.isfinite(value)
    assert rul.ig_cdf(100.0, 100.0, 1e6) == pytest.approx(0.5, abs=0.01)     # nearly deterministic: the median is the mean


def test_known_special_case_standard_ig():
    # IG(mean 1, shape 1): cdf(1) = Phi(0) + e^2 Phi(-2) = 0.5 + e^2 * 0.0227501 = 0.66810...
    assert rul.ig_cdf(1.0, 1.0, 1.0) == pytest.approx(0.5 + math.exp(2.0) * 0.0227501319, abs=1e-6)


def test_sampler_has_the_right_moments_and_cdf():
    rng = np.random.default_rng(0)
    x = rul.ig_sample(MEAN, SHAPE, 200000, rng)
    assert x.mean() == pytest.approx(MEAN, rel=0.002)
    assert x.var() == pytest.approx(MEAN ** 3 / SHAPE, rel=0.02)
    for p in (0.1, 0.5, 0.9):
        assert np.mean(x <= rul.ig_quantile(p, MEAN, SHAPE)) == pytest.approx(p, abs=0.004)
    assert np.all(x > 0)


def test_sampler_accepts_array_means():
    rng = np.random.default_rng(1)
    means = np.array([10.0, 100.0, 1000.0])
    shapes = 50.0 * means                          # the same coefficient of variation (0.14) for each mean
    x = rul.ig_sample(np.tile(means, (20000, 1)), np.tile(shapes, (20000, 1)), (20000, 3), rng)
    assert np.allclose(x.mean(axis=0) / means, 1.0, atol=0.01)


def test_first_passage_paths_follow_the_inverse_gaussian_law():
    rng = np.random.default_rng(2)
    mu, sigma, w0 = 1.2e-3, 0.004, 0.05
    b = 1 - w0
    paths = rul.first_passage_paths(w0, mu, sigma, 1.0, 0.5, 4000, rng)
    assert not np.isnan(paths).any()
    assert paths.mean() == pytest.approx(b / mu, rel=0.01)
    assert paths.var() == pytest.approx(b * sigma ** 2 / mu ** 3, rel=0.1)
    assert np.mean(paths <= rul.ig_quantile(0.5, b / mu, b * b / sigma ** 2)) == pytest.approx(0.5, abs=0.03)


def test_bridge_correction_removes_the_discretisation_bias():
    """Without it a coarse step misses crossings between samples and lengthens the life; with it the step size barely matters."""
    mu, sigma, w0 = 1.2e-3, 0.004, 0.05
    coarse = rul.first_passage_paths(w0, mu, sigma, 1.0, 4.0, 4000, np.random.default_rng(3))
    fine = rul.first_passage_paths(w0, mu, sigma, 1.0, 0.25, 4000, np.random.default_rng(4))
    assert coarse.mean() == pytest.approx(fine.mean(), rel=0.015)


def test_rul_samples_propagate_uncertainty_and_clip_at_the_limit():
    rng = np.random.default_rng(0)
    cov = np.array([[0.0004, 0.0], [0.0, 0.04]])
    plug = rul.rul_samples(0.5, 1.0, cov, 1.2e-3, 0.004, size=4000, rng=rng, propagate=False)
    prop = rul.rul_samples(0.5, 1.0, cov, 1.2e-3, 0.004, size=4000, rng=rng, propagate=True)
    lo_p, hi_p = rul.interval(prop)
    lo_u, hi_u = rul.interval(plug)
    assert hi_p - lo_p > 1.5 * (hi_u - lo_u)
    assert np.median(plug) == pytest.approx((0.5 / 1.2e-3), rel=0.05)
    worn = rul.rul_samples(1.2, 1.0, cov, 1.2e-3, 0.004, size=500, rng=rng, propagate=False)
    assert worn.max() < 1e-3                        # already past the limit: no life left
    assert np.all(np.isfinite(rul.rul_samples(0.9, 0.0, cov, 1.2e-3, 0.004, size=500, rng=rng)))   # a zero severity estimate is floored


def test_interval_levels():
    x = np.arange(1000.0)
    assert rul.interval(x, 0.8) == pytest.approx((99.9, 899.1))
    assert rul.interval(x, 0.5)[0] == pytest.approx(249.75)
