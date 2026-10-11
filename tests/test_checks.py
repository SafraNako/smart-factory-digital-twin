import pytest

from fabrika_ikizi import checks


@pytest.mark.parametrize("check", checks.ALL, ids=lambda c: c.__name__)
def test_every_model_check_passes(check):
    result = check()
    assert result.ok, result.detail
    assert result.name and result.detail


def test_the_checks_can_fail(monkeypatch):
    """A check that cannot fail proves nothing: break the filter's covariance update and the batch comparison must notice."""
    from fabrika_ikizi import kalman
    real = kalman.KalmanFilter.update

    def overconfident(self, y, H, c, R):
        out = real(self, y, H, c, R)
        self.P = self.P * 0.5
        return out

    monkeypatch.setattr(kalman.KalmanFilter, "update", overconfident)
    assert not checks.kalman_vs_batch().ok
    assert not checks.kalman_consistency(runs=60, steps=60).ok


def test_the_line_check_notices_a_wrong_closed_form(monkeypatch):
    monkeypatch.setattr(checks.ctmc, "two_station_closed_form", lambda mu1, mu2, n: 0.9 * mu2)
    assert not checks.line_exact_vs_closed_forms().ok


def test_a_more_responsive_twin_tracks_an_abrupt_fault_better():
    """The documented trade-off: a larger process noise on the wear-rate multiplier follows a sudden x3 fault sooner."""
    slow_used, slow_cover, slow_ratio = checks.rul_after_fault(q_kappa=0.03, after=50.0, machines=20)
    fast_used, fast_cover, fast_ratio = checks.rul_after_fault(q_kappa=0.08, after=50.0, machines=20)
    assert slow_used >= 15 and fast_used >= 15
    assert fast_cover > slow_cover + 0.3
    assert slow_ratio > 1.3 > fast_ratio > 0.8                 # the slow twin overestimates the remaining life
