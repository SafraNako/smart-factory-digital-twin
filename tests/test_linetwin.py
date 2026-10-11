import numpy as np
import pytest

from fabrika_ikizi import ctmc, linetwin, scenarios as sc
from fabrika_ikizi import line as L
from fabrika_ikizi.line import Line, Station


def small_line():
    return Line((Station("a", 1.0, mtbf=60.0, mttr=8.0), Station("b", 1.15, mtbf=90.0, mttr=10.0)), (2,))


def test_exact_when_possible_and_simulation_otherwise():
    exact = linetwin.predict_throughput(small_line())
    assert exact[2] == "exact" and exact[1] == 0.0 and exact[0] == pytest.approx(ctmc.solve(small_line()).throughput)
    erlang = Line((Station("a", 1.0, dist="erlang:3"), Station("b", 1.2)), (3,))
    value, se, method = linetwin.predict_throughput(erlang, horizon=8000.0, replications=4)
    assert method == "simulation" and se > 0 and 0.5 < value < 0.85
    assert linetwin.exact_applicable(small_line()) and not linetwin.exact_applicable(erlang)


def test_predictive_interval_brackets_the_calibrated_estimate_and_the_exact_truth():
    line = small_line()
    exact = ctmc.solve(line).throughput
    plant = L.simulate(line, 30000.0, 0.0, seed=4)
    p = linetwin.predictive_interval(plant, line, draws=80, level=0.9, seed=1)
    assert p.low < p.estimate < p.high and p.method == "exact"
    assert p.low <= exact <= p.high
    short = L.simulate(line, 2000.0, 0.0, seed=4)
    q = linetwin.predictive_interval(short, line, draws=80, level=0.9, seed=1)
    assert (q.high - q.low) > 2 * (p.high - p.low)                  # less data, wider interval


def test_speedup_gains_largest_at_the_capacity_bottleneck():
    line = Line((Station("a", 1.0, mtbf=120.0, mttr=8.0), Station("b", 1.2, mtbf=90.0, mttr=10.0),
                 Station("c", 0.9, mtbf=200.0, mttr=15.0)), (4, 4))
    base, gains = linetwin.speedup_ranking(line, 1.1)
    assert base == pytest.approx(ctmc.solve(line).throughput)
    assert int(np.argmax(gains)) == line.bottleneck == 1
    assert np.all(gains > 0) and gains[1] > 2 * gains[0]


def test_a_non_bottleneck_speedup_gains_little_with_a_large_buffer():
    line = Line((Station("a", 1.0), Station("b", 2.0)), (6,))
    _, gains = linetwin.speedup_ranking(line, 1.2)
    assert gains[0] < 0.1 * gains[1]


def test_buffer_study_is_monotone_and_bounded():
    line = Line((Station("a", 1.0, mtbf=50.0, mttr=5.0), Station("b", 1.0, mtbf=50.0, mttr=5.0)), (0,))
    th = linetwin.buffer_study(line, 0, [0, 1, 2, 4, 8])
    assert np.all(np.diff(th) > 0) and th[-1] < linetwin.capacity_bound(line)


def test_capacity_bound():
    line = Line((Station("a", 1.0, mtbf=90.0, mttr=10.0), Station("b", 2.0)), (3,))
    assert linetwin.capacity_bound(line) == pytest.approx(0.5)               # the second station: 1 / 2.0
    fed = Line(line.stations, line.buffers, arrival_rate=0.3, queue_capacity=5)
    assert linetwin.capacity_bound(fed) == pytest.approx(0.3)
    assert linetwin.exact_applicable(small_line())


def test_the_twin_of_the_example_line_agrees_with_the_plant():
    line = sc.example_line("montaj")
    r = sc.run_hat(line, horizon=15000.0, seed=3, draws=60)
    s = r.summary
    assert s["tahmin_alt"] <= s["gozlenen_uretim"] <= s["tahmin_ust"]
    assert s["darbogaz"] == s["darbogaz_gozlenen"] == "Kaynak"
    assert abs(s["little_artigi"]) < 0.01 and s["tahmin_yontemi"] == "exact"
