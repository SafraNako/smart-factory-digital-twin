import math

import numpy as np
import pytest

from fabrika_ikizi import monitor


def test_shewhart_arl_is_the_reciprocal_alarm_probability():
    assert monitor.shewhart_arl(3.0) == pytest.approx(370.398, rel=1e-5)
    assert monitor.shewhart_arl(3.0, shift=0.0) == pytest.approx(1.0 / (2.0 * monitor.phi_cdf(-3.0)))
    assert monitor.shewhart_arl(3.0, shift=2.0) < 20.0          # a 2-sigma shift is caught within a few samples
    assert monitor.phi_cdf(0.0) == 0.5 and monitor.phi_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)


@pytest.mark.parametrize("k,h,shift", [(0.5, 4.0, 0.0), (0.5, 4.0, 1.0), (0.25, 5.0, 0.5)])
def test_brook_evans_matches_simulation_and_siegmund(k, h, shift):
    rng = np.random.default_rng(1)
    runs = np.array([monitor.run_length(k, h, shift, rng) for _ in range(2000)])
    be = monitor.cusum_arl_brook_evans(k, h, shift)
    assert abs(runs.mean() - be) < 3.5 * runs.std() / math.sqrt(len(runs))
    assert monitor.cusum_arl_siegmund(k, h, shift) == pytest.approx(be, rel=0.05)


def test_brook_evans_converges_with_the_number_of_states():
    values = [monitor.cusum_arl_brook_evans(0.5, 4.0, 0.0, m) for m in (50, 100, 200, 400)]
    assert abs(values[-1] - values[-2]) < abs(values[1] - values[0])
    assert values[-1] == pytest.approx(335.36, abs=0.05)


def test_known_value_of_the_textbook_cusum():
    # the widely tabulated two-sided in-control ARL for k = 1/2, h = 5 is about 465 (and 168 for h = 4);
    # the one-sided chart has twice that
    assert monitor.cusum_arl_brook_evans(0.5, 4.0) == pytest.approx(336.0, abs=1.5)
    assert monitor.cusum_arl_brook_evans(0.5, 5.0) == pytest.approx(930.0, abs=3.0)
    assert monitor.cusum_two_sided_arl(0.5, 5.0) == pytest.approx(465.0, abs=2.0)
    assert monitor.cusum_two_sided_arl(0.5, 4.0) == pytest.approx(168.0, abs=1.0)


def test_arl_falls_with_the_shift_and_two_sided_is_shorter():
    arls = [monitor.cusum_arl_brook_evans(0.5, 5.0, s) for s in (0.0, 0.5, 1.0, 2.0)]
    assert all(a > b for a, b in zip(arls, arls[1:]))
    assert monitor.cusum_two_sided_arl(0.5, 5.0) < monitor.cusum_arl_brook_evans(0.5, 5.0)
    assert monitor.cusum_two_sided_arl(0.5, 5.0) == pytest.approx(0.5 * monitor.cusum_arl_brook_evans(0.5, 5.0), rel=0.01)


def test_threshold_search_hits_the_requested_in_control_arl():
    for target in (200.0, 2000.0):
        h = monitor.cusum_threshold(target)
        assert monitor.cusum_two_sided_arl(0.5, h) == pytest.approx(target, rel=0.01)
    assert monitor.cusum_threshold(2000.0) > monitor.cusum_threshold(200.0)


def test_streaming_cusum_detects_both_directions_and_ignores_nan():
    up = monitor.Cusum(0.5, 5.0, latch=False)
    hits = [up.update(2.0) for _ in range(6)]
    assert +1 in hits and -1 not in hits
    down = monitor.Cusum(0.5, 5.0, latch=False)
    assert -1 in [down.update(-2.0) for _ in range(6)]
    c = monitor.Cusum(0.5, 5.0)
    c.update(1.5)
    before = (c.up, c.down)
    assert c.update(float("nan")) == 0 and (c.up, c.down) == before


def test_latching_gives_one_alarm_for_a_persistent_shift_and_rearms_after_recovery():
    latched = monitor.Cusum(0.5, 5.0, latch=True)
    reset = monitor.Cusum(0.5, 5.0, latch=False)
    shifted = [latched.update(2.0) for _ in range(200)]
    shifted_reset = [reset.update(2.0) for _ in range(200)]
    assert sum(1 for a in shifted if a) == 1
    assert sum(1 for a in shifted_reset if a) > 10
    assert shifted.index(1) == shifted_reset.index(1)               # the first alarm is at the same sample
    for _ in range(50):
        latched.update(0.0)
    assert [latched.update(2.0) for _ in range(20)].count(1) == 1   # re-armed
    assert max(latched.up, latched.down) <= 10.0 + 2.0              # statistics are capped at 2 h


def test_cusum_on_white_noise_matches_the_two_sided_theory():
    rng = np.random.default_rng(5)
    lengths = []
    for _ in range(300):
        c, n = monitor.Cusum(0.5, 3.5, latch=False), 0
        while not c.update(float(rng.standard_normal())):
            n += 1
        lengths.append(n + 1)
    theory = monitor.cusum_two_sided_arl(0.5, 3.5)
    assert np.mean(lengths) == pytest.approx(theory, rel=0.15)
