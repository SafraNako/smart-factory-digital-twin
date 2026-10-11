import math

import numpy as np
import pytest

from fabrika_ikizi import line as L
from fabrika_ikizi.line import Line, Station


def det_line(times, buffers, **kw):
    return Line(tuple(Station(f"s{i}", m, dist="det") for i, m in enumerate(times)), tuple(buffers), **kw)


# --- model objects -------------------------------------------------------------------------
@pytest.mark.parametrize("dist,cv2", [("exp", 1.0), ("det", 0.0), ("erlang:3", 1 / 3), ("lognorm:0.5", 0.25)])
def test_samplers_have_the_requested_mean_and_variability(dist, cv2):
    rng = np.random.default_rng(0)
    draw = L.sampler(dist, 2.0)
    x = np.array([draw(rng) for _ in range(200000)])
    assert x.mean() == pytest.approx(2.0, rel=0.01)
    assert x.var() / x.mean() ** 2 == pytest.approx(cv2, abs=0.02)


def test_validation_errors():
    with pytest.raises(ValueError):
        Station("a", 0.0)
    with pytest.raises(ValueError):
        Station("a", 1.0, scrap=1.0)
    with pytest.raises(ValueError):
        Station("a", 1.0, mtbf=10.0, mttr=0.0)
    with pytest.raises(ValueError):
        Station("a", 1.0, dist="weibull")
    with pytest.raises(ValueError):
        Station("a", 1.0, dist="erlang:0")
    with pytest.raises(ValueError):
        Line((Station("a", 1.0), Station("b", 1.0)), ())
    with pytest.raises(ValueError):
        Line((Station("a", 1.0), Station("b", 1.0)), (1.5,))
    with pytest.raises(ValueError):
        Line((Station("a", 1.0),), (), arrival_rate=-1.0)
    with pytest.raises(ValueError):
        L.simulate(det_line([1.0], []), 10.0, warmup=20.0)


def test_capacity_availability_and_bottleneck():
    st = Station("a", 2.0, mtbf=90.0, mttr=10.0)
    assert st.availability == pytest.approx(0.9) and st.capacity == pytest.approx(0.45)
    assert Station("b", 2.0).availability == 1.0
    line = Line((Station("a", 1.0), Station("b", 2.0), Station("c", 1.5, mtbf=90.0, mttr=10.0)), (3, 3))
    assert line.bottleneck == 1
    faster = line.with_station(1, mean_service=1.0)
    assert faster.stations[1].mean_service == 1.0 and line.stations[1].mean_service == 2.0     # frozen originals
    assert faster.bottleneck == 2
    assert line.with_buffer(0, 9).buffers == (9, 3)


# --- deterministic lines have exact answers ------------------------------------------------
def test_single_deterministic_station_throughput():
    r = L.simulate(det_line([2.0], []), 1000.0)
    assert r.throughput == pytest.approx(0.5, abs=1e-3) and r.busy[0] == pytest.approx(1.0, abs=1e-3)
    assert r.starved[0] == 0 and r.blocked[0] == 0 and r.down[0] == 0


def test_the_slower_station_sets_the_rate_and_the_faster_one_waits():
    r = L.simulate(det_line([1.0, 2.0], [3]), 3000.0, warmup=100.0)
    assert r.throughput == pytest.approx(0.5, abs=1e-3)
    assert r.busy[0] == pytest.approx(0.5, abs=0.01) and r.blocked[0] == pytest.approx(0.5, abs=0.01)   # first is blocked half the time
    assert r.busy[1] == pytest.approx(1.0, abs=0.01) and r.starved[1] == pytest.approx(0.0, abs=0.01)
    assert r.mean_buffers[0] == pytest.approx(3.0, abs=0.05)                                              # the buffer stays full


def test_slow_first_station_starves_the_second():
    r = L.simulate(det_line([2.0, 1.0], [3]), 3000.0, warmup=100.0)
    assert r.throughput == pytest.approx(0.5, abs=1e-3)
    assert r.starved[1] == pytest.approx(0.5, abs=0.01) and r.blocked[0] == pytest.approx(0.0, abs=0.01)
    assert r.mean_buffers[0] < 0.1


def test_zero_buffer_hands_parts_over_directly():
    r = L.simulate(det_line([1.0, 1.0], [0]), 2000.0, warmup=100.0)
    assert r.throughput == pytest.approx(1.0, abs=1e-3)
    r = L.simulate(det_line([1.0, 1.0, 1.0], [0, 0]), 2000.0, warmup=100.0)
    assert r.throughput == pytest.approx(1.0, abs=1e-3)


def test_infinite_buffer_never_blocks():
    r = L.simulate(det_line([1.0, 2.0], [math.inf]), 1000.0, warmup=100.0)
    assert r.blocked[0] == 0 and r.mean_buffers[0] > 100          # the fast station piles up work in front of the slow one


def test_throughput_is_nondecreasing_in_buffer_size():
    base = L.Line((Station("a", 1.0), Station("b", 1.0)), (0,))
    values = []
    for n in (0, 1, 3, 8):
        res = L.replicate(base.with_buffer(0, n), 6000.0, 6, warmup=200.0, seed=1)
        values.append(L.mean_and_se([r.throughput for r in res])[0])
    assert values == sorted(values) and values[-1] > values[0] + 0.1


# --- failures, scrap, accounting -----------------------------------------------------------
def test_failures_cost_the_availability_and_log_the_up_and_down_times():
    st = Station("a", 1.0, dist="det", mtbf=100.0, mttr=25.0)
    res = L.replicate(Line((st,), ()), 20000.0, 8, warmup=500.0, seed=2)
    down = np.mean([r.down[0] for r in res])
    th, se = L.mean_and_se([r.throughput for r in res])
    assert down == pytest.approx(0.2, abs=0.01) and th == pytest.approx(0.8, abs=4 * se + 0.01)
    log = res[0].log
    assert log["n_fail"][0] > 100
    assert log["up_time"][0] / log["n_fail"][0] == pytest.approx(100.0, rel=0.2)
    assert log["down_time"][0] / log["n_repair"][0] == pytest.approx(25.0, rel=0.2)
    assert log["up_time"][0] + log["down_time"][0] == pytest.approx(20000.0)       # the whole run is accounted for


def test_a_part_in_process_resumes_after_a_repair():
    """A deterministic 10-minute job interrupted by failures still takes exactly 10 minutes of up time."""
    st = Station("a", 10.0, dist="det", mtbf=3.0, mttr=2.0)
    res = L.simulate(Line((st,), ()), 20000.0, 0.0, seed=3)
    # busy time = completions x 10, regardless of the interruptions
    assert res.busy[0] * res.horizon == pytest.approx(len(res.log["service"][0]) * 10.0, rel=0.01)
    assert res.throughput == pytest.approx(0.1 * st.availability, rel=0.05)


def test_scrap_fraction_and_good_throughput():
    line = Line((Station("a", 1.0, dist="det"), Station("b", 1.0, dist="det", scrap=0.1)), (2,))
    r = L.simulate(line, 20000.0, 500.0, seed=4)
    assert r.scrapped / (r.scrapped + r.exits) == pytest.approx(0.1, abs=0.01)
    assert r.throughput == pytest.approx(0.9, abs=0.02)
    assert r.log["scrap"][1] > 1000 and r.log["scrap"][0] == 0


def test_state_fractions_add_up_to_one():
    line = L.Line((Station("a", 1.0, mtbf=40.0, mttr=5.0), Station("b", 1.2)), (2,))
    r = L.simulate(line, 8000.0, 200.0, seed=5)
    total = r.busy + r.blocked + r.starved + r.down
    assert np.allclose(total, 1.0, atol=1e-9)


def test_little_law_and_oee():
    line = L.Line((Station("a", 1.0, mtbf=60.0, mttr=6.0, scrap=0.02), Station("b", 1.1)), (3,))
    r = L.simulate(line, 30000.0, 1000.0, seed=6)
    assert abs(r.little_residual) < 0.01
    for (avail, perf, quality, oee), st in zip(r.oee(line.stations), line.stations):
        assert oee == pytest.approx(avail * perf * quality)
        assert 0 < oee <= 1 and quality == pytest.approx(1.0 - st.scrap)
    assert r.oee(line.stations)[1][0] == pytest.approx(1.0)       # the reliable station is always available


def test_arrivals_waiting_room_and_loss():
    line = Line((Station("s", 1.0, dist="exp"),), (), arrival_rate=0.5, queue_capacity=2)
    r = L.simulate(line, 40000.0, 500.0, seed=7)
    arrivals = r.exits + r.lost
    assert 0.5 < r.exits / arrivals < 1.0 and r.lost > 0
    assert r.mean_queue <= 2.0
    # an overloaded station with a small waiting room loses most arrivals but never queues beyond its capacity
    jam = L.simulate(Line((Station("s", 4.0, dist="det"),), (), arrival_rate=1.0, queue_capacity=1), 4000.0, 100.0, seed=8)
    assert jam.lost / (jam.lost + jam.exits) == pytest.approx(0.75, abs=0.03) and jam.mean_queue <= 1.0


def test_results_repeat_with_the_seed_and_differ_between_seeds():
    line = L.Line((Station("a", 1.0, mtbf=40.0, mttr=5.0), Station("b", 1.2)), (2,))
    a, b, c = L.simulate(line, 3000.0, 0.0, 1), L.simulate(line, 3000.0, 0.0, 1), L.simulate(line, 3000.0, 0.0, 2)
    assert a.throughput == b.throughput and a.exits == b.exits and a.throughput != c.throughput
