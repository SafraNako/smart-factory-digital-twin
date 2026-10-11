import math

import numpy as np
import pytest

from fabrika_ikizi import ctmc
from fabrika_ikizi import line as L
from fabrika_ikizi.line import Line, Station


@pytest.mark.parametrize("mu1,mu2,buffer", [(1.0, 1.0, 0), (1.0, 1.0, 3), (1.0, 0.8, 2), (0.7, 1.0, 4), (2.0, 1.0, 5)])
def test_two_reliable_stations_match_the_birth_death_closed_form(mu1, mu2, buffer):
    line = Line((Station("a", 1 / mu1), Station("b", 1 / mu2)), (buffer,))
    result = ctmc.solve(line)
    assert result.throughput == pytest.approx(ctmc.two_station_closed_form(mu1, mu2, buffer), abs=1e-12)
    assert result.n_states == buffer + 3


def test_identical_stations_without_buffer_give_two_thirds():
    assert ctmc.solve(Line((Station("a", 1.0), Station("b", 1.0)), (0,))).throughput == pytest.approx(2.0 / 3.0)
    n = 5
    assert ctmc.solve(Line((Station("a", 1.0), Station("b", 1.0)), (n,))).throughput == pytest.approx((n + 2) / (n + 3))


def test_single_unreliable_machine_runs_at_rate_times_availability():
    st = Station("a", 0.5, mtbf=40.0, mttr=10.0)
    r = ctmc.solve(Line((st,), ()))
    assert r.throughput == pytest.approx(2.0 * 0.8) and r.down[0] == pytest.approx(0.2) and r.busy[0] == pytest.approx(0.8)


def test_mm1k_loss_and_mean_number():
    loss, mean_n = ctmc.mm1k(0.8, 1.0, 5)
    r = ctmc.solve(Line((Station("s", 1.0),), (), arrival_rate=0.8, queue_capacity=4))
    assert r.loss_probability == pytest.approx(loss) and r.wip == pytest.approx(mean_n)
    assert r.throughput == pytest.approx(0.8 * (1 - loss))                    # accepted arrivals are served: Little's law
    # a known value: rho = 1 splits the probability evenly over k + 1 states
    assert ctmc.mm1k(1.0, 1.0, 3)[0] == pytest.approx(0.25)


def test_probabilities_of_each_station_add_up_and_flow_is_conserved():
    line = Line((Station("a", 1.0, mtbf=50.0, mttr=5.0), Station("b", 1.1), Station("c", 0.9, mtbf=70.0, mttr=7.0)), (2, 2))
    r = ctmc.solve(line)
    assert np.allclose(r.busy + r.blocked + r.starved + r.down, 1.0, atol=1e-10)
    # every part that leaves a station enters the next: completion rates agree along the line
    rates = r.busy / np.array([s.mean_service for s in line.stations])
    assert np.allclose(rates, rates[-1], atol=1e-10)
    assert r.down[0] == pytest.approx(1 - line.stations[0].availability) and r.down[1] == 0.0
    assert r.starved[0] == 0.0                                                # a saturated source never starves the first station


def test_exact_solution_agrees_with_replicated_simulation():
    line = Line((Station("a", 1.0, mtbf=40.0, mttr=5.0), Station("b", 1.2, mtbf=60.0, mttr=8.0)), (3,))
    exact = ctmc.solve(line)
    res = L.replicate(line, 20000.0, 8, warmup=500.0, seed=3)
    th, se = L.mean_and_se([r.throughput for r in res])
    assert abs(th - exact.throughput) < 4.5 * se
    assert np.mean([r.wip for r in res]) == pytest.approx(exact.wip, rel=0.03)
    assert np.mean([r.mean_buffers[0] for r in res]) == pytest.approx(exact.mean_buffers[0], rel=0.03)


def test_throughput_rises_with_buffer_size_and_is_below_the_capacity_bound():
    base = Line((Station("a", 1.0, mtbf=50.0, mttr=5.0), Station("b", 1.0, mtbf=50.0, mttr=5.0)), (0,))
    values = [ctmc.solve(base.with_buffer(0, n)).throughput for n in (0, 1, 2, 4, 8, 16, 32)]
    assert np.all(np.diff(values) > 0) and values[-1] < base.stations[0].capacity


def test_unsupported_models_are_refused():
    with pytest.raises(ValueError, match="exponential"):
        ctmc.solve(Line((Station("a", 1.0, dist="det"),), ()))
    with pytest.raises(ValueError, match="scrap"):
        ctmc.solve(Line((Station("a", 1.0, scrap=0.1),), ()))
    with pytest.raises(ValueError, match="finite buffers"):
        ctmc.solve(Line((Station("a", 1.0), Station("b", 1.0)), (math.inf,)))
    with pytest.raises(ValueError, match="waiting room"):
        ctmc.solve(Line((Station("a", 1.0),), (), arrival_rate=0.5))
