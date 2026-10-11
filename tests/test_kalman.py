import numpy as np
import pytest

from fabrika_ikizi import checks, kalman
from fabrika_ikizi.kalman import KalmanFilter


def test_matches_the_batch_solution_with_lost_samples():
    assert checks.kalman_vs_batch(seed=1).ok
    assert checks.kalman_vs_batch(seed=7, steps=12).ok


def test_scalar_random_walk_converges_to_the_closed_form_prior_variance():
    q, r = 0.05, 0.4
    kf = KalmanFilter([0.0], [[2.0]])
    for _ in range(300):
        kf.update(np.array([0.0]), np.eye(1), np.zeros(1), np.array([[r]]))
        kf.predict(np.eye(1), np.zeros(1), np.array([[q]]))
    p = kalman.scalar_steady_state(q, r)
    assert kf.P[0, 0] == pytest.approx(p, rel=1e-10)
    assert p == pytest.approx(0.5 * (q + np.sqrt(q * q + 4 * q * r)))
    assert p * p - q * p - q * r == pytest.approx(0.0, abs=1e-15)


def test_kalman_gain_weights_the_measurement_by_relative_precision():
    kf = KalmanFilter([0.0], [[4.0]])
    nu, S = kf.update(np.array([2.0]), np.eye(1), np.zeros(1), np.array([[1.0]]))
    assert nu[0] == 2.0 and S[0, 0] == 5.0
    assert kf.x[0] == pytest.approx(2.0 * 4.0 / 5.0)         # gain P / (P + R)
    assert kf.P[0, 0] == pytest.approx(4.0 * 1.0 / 5.0)


def test_all_missing_leaves_the_state_unchanged():
    kf = KalmanFilter([1.0, 2.0], np.diag([1.0, 2.0]))
    nu, S = kf.update(np.array([np.nan, np.nan]), np.eye(2), np.zeros(2), np.eye(2))
    assert np.isnan(nu).all() and np.isnan(S).all()
    assert np.array_equal(kf.x, [1.0, 2.0]) and np.array_equal(kf.P, np.diag([1.0, 2.0]))


def test_partial_missing_equals_a_filter_with_only_the_available_row():
    H = np.array([[1.0, 0.0], [0.0, 2.0]])
    R = np.diag([0.5, 0.3])
    a = KalmanFilter([0.0, 0.0], np.diag([1.0, 1.0]))
    b = KalmanFilter([0.0, 0.0], np.diag([1.0, 1.0]))
    nu, S = a.update(np.array([np.nan, 1.5]), H, np.zeros(2), R)
    b.update(np.array([1.5]), H[1:], np.zeros(1), R[1:, 1:])
    assert np.allclose(a.x, b.x) and np.allclose(a.P, b.P)
    assert np.isnan(nu[0]) and not np.isnan(nu[1]) and np.isnan(S[0, 0])


def test_covariance_stays_symmetric_positive_definite_over_a_long_run():
    mt = checks._model_twin()
    rng = np.random.default_rng(0)
    u, xs, ys, prior = checks._synthetic(mt, rng, 400)
    kf = checks._run_filter(mt, u, ys, prior)
    assert np.allclose(kf.P, kf.P.T) and np.linalg.eigvalsh(kf.P).min() > 0


def test_nis_and_nees_definitions():
    nu = np.array([1.0, np.nan, 2.0])
    S = np.full((3, 3), np.nan)
    S[np.ix_([0, 2], [0, 2])] = [[2.0, 0.0], [0.0, 4.0]]
    value, dof = kalman.nis(nu, S)
    assert dof == 2 and value == pytest.approx(0.5 + 1.0)
    assert kalman.nis(np.array([np.nan]), np.array([[np.nan]]))[1] == 0
    assert kalman.nees(np.array([1.0, 2.0]), np.diag([1.0, 4.0])) == pytest.approx(2.0)


def test_filter_beats_the_raw_measurement_on_data_from_its_model():
    mt = checks._model_twin()
    rng = np.random.default_rng(2)
    u, xs, ys, prior = checks._synthetic(mt, rng, 300, missing=0.0)
    kf = checks.kalman.KalmanFilter(*prior)
    errs = []
    for k in range(len(ys)):
        kf.update(ys[k], mt.H(), mt.c(), mt.R)
        errs.append(kf.x[1] - xs[k][1])
        if k < len(ys) - 1:
            kf.predict(mt.F(u[k]), mt.b(u[k]), mt.Q())
    raw = (ys[:, 1] - mt.params.v0) / mt.params.kv - xs[:, 1]
    assert np.std(errs[50:]) < 0.5 * np.std(raw[50:])


def test_consistency_check_passes_and_batch_estimate_has_a_posterior_covariance():
    assert checks.kalman_consistency(runs=60, steps=60, seed=2).ok
