import numpy as np
import pytest

from fabrika_ikizi import machine
from fabrika_ikizi.machine import Fault, MachineParams


def quiet(**kw):
    """Parameters without randomness or wear, for closed-form checks."""
    base = dict(g0=0.0, g1=0.0, sigma_w=0.0, q_t=0.0, w0=0.0)
    base.update(kw)
    return MachineParams(**base)


def test_time_constant_is_resistance_times_capacity():
    p = MachineParams()
    assert p.tau == pytest.approx(8.0 * 0.0417)


def test_temperature_relaxes_to_the_steady_state_with_the_thermal_time_constant():
    p = quiet()
    dt = 0.05
    truth = machine.simulate_plant(p, 6.0, dt, load=lambda t: np.full_like(np.asarray(t, dtype=float), 0.5))
    t_inf = p.t_amb + p.r_th * (p.p0 + p.p1 * 0.25)
    assert truth.temp[0] == pytest.approx(t_inf)               # starts in equilibrium for the first load
    # a step in load: start from the 0.5-load equilibrium, switch to 1.0, check exp(-t/tau)
    def step_load(t):
        t = np.asarray(t, dtype=float)
        return np.where(t < 0.5, 0.5, 1.0)
    tr = machine.simulate_plant(p, 4.0, dt, load=step_load)
    t1 = p.t_amb + p.r_th * (p.p0 + p.p1)
    k = int(round((0.5 + p.tau) / dt))
    elapsed = k * dt - 0.5                                     # about one time constant after the step
    frac = (t1 - tr.temp[k]) / (t1 - t_inf)
    assert frac == pytest.approx(np.exp(-elapsed / p.tau), abs=1e-9)    # the gap left is exp(-elapsed / tau), exactly
    assert abs(elapsed - p.tau) < dt
    assert tr.temp[-1] == pytest.approx(t1, abs=2e-3)


def test_exact_thermal_step_is_independent_of_the_step_size():
    p = quiet()
    def step_load(t):
        return np.where(np.asarray(t, dtype=float) < 1.0, 0.2, 0.9)
    coarse = machine.simulate_plant(p, 3.0, 0.5, load=step_load)
    fine = machine.simulate_plant(p, 3.0, 0.05, load=step_load)
    assert coarse.temp[-1] == pytest.approx(fine.temp[-1], abs=1e-9)


def test_wear_drift_and_diffusion_match_the_model():
    p = MachineParams(q_t=0.0)
    dt, n = 0.25, 20000
    truth = machine.simulate_plant(p, n * dt, dt, seed=3, load=lambda t: np.full_like(np.asarray(t, dtype=float), 0.6))
    inc = np.diff(truth.wear)
    drift = float(p.wear_rate(0.6)) * dt
    assert inc.mean() == pytest.approx(drift, abs=4 * p.sigma_w * np.sqrt(dt) / np.sqrt(len(inc)))
    assert inc.std() == pytest.approx(p.sigma_w * np.sqrt(dt), rel=0.03)


def test_failure_index_is_the_first_crossing_and_truncation_works():
    p = MachineParams()
    full = machine.simulate_plant(p, 2500.0, 0.5, seed=1)
    assert full.fail_index > 0 and full.wear[full.fail_index] >= 1.0 and np.all(full.wear[:full.fail_index] < 1.0)
    cut = machine.simulate_plant(p, 2500.0, 0.5, seed=1, stop_at_failure=True)
    assert len(cut.t) == full.fail_index + 1 and cut.fail_index == full.fail_index
    assert cut.fail_time == pytest.approx(full.fail_time)
    short = machine.simulate_plant(p, 10.0, 0.5, seed=1)
    assert short.fail_index == -1 and np.isnan(short.fail_time)


def test_bearing_fault_multiplies_the_wear_rate():
    p = MachineParams(sigma_w=0.0, q_t=0.0)
    flat = lambda t: np.full_like(np.asarray(t, dtype=float), 0.5)
    tr = machine.simulate_plant(p, 100.0, 0.25, load=flat, faults=(Fault("bearing", 50.0, 3.0),))
    rate = np.diff(tr.wear) / 0.25
    assert rate[10] == pytest.approx(float(p.wear_rate(0.5)))
    assert rate[-1] == pytest.approx(3.0 * float(p.wear_rate(0.5)))
    assert tr.kappa[0] == 1.0 and tr.kappa[-1] == 3.0


def test_cooling_fault_raises_the_steady_temperature_by_the_extra_resistance():
    p = quiet()
    flat = lambda t: np.full_like(np.asarray(t, dtype=float), 0.7)
    tr = machine.simulate_plant(p, 40.0, 0.1, load=flat, faults=(Fault("cooling", 10.0, 1.5),))
    power = p.p0 + p.p1 * 0.49
    assert tr.temp[0] == pytest.approx(p.t_amb + p.r_th * power)
    assert tr.temp[-1] == pytest.approx(p.t_amb + 1.5 * p.r_th * power, abs=1e-6)


def test_friction_heat_raises_temperature_with_wear():
    p = quiet(w0=0.5)
    tr = machine.simulate_plant(p, 1.0, 0.1, load=lambda t: np.full_like(np.asarray(t, dtype=float), 0.3))
    assert tr.temp[0] == pytest.approx(p.t_amb + p.r_th * (p.p0 + p.p1 * 0.09 + p.p2 * 0.5))


def test_daily_load_profile_and_mean_wear_rate():
    assert machine.daily_load(np.array([0.0, 7.9, 8.0, 15.9, 16.0, 23.9, 24.0])).tolist() == [0.85, 0.85, 0.55, 0.55, 0.1, 0.1, 0.85]
    p = MachineParams()
    assert machine.mean_wear_rate(p) == pytest.approx(p.g0 + p.g1 * (0.85 ** 2 + 0.55 ** 2 + 0.1 ** 2) / 3.0, rel=1e-3)


def test_vibration_is_linear_in_wear_and_seed_repeats():
    p = MachineParams()
    a = machine.simulate_plant(p, 50.0, 0.5, seed=4)
    b = machine.simulate_plant(p, 50.0, 0.5, seed=4)
    assert np.array_equal(a.wear, b.wear) and np.array_equal(a.temp, b.temp)
    assert np.allclose(a.vibration, p.v0 + p.kv * a.wear)
    assert not np.array_equal(a.wear, machine.simulate_plant(p, 50.0, 0.5, seed=5).wear)


def test_fault_validation():
    with pytest.raises(ValueError):
        Fault("motor", 10.0)
    with pytest.raises(ValueError):
        Fault("bearing", 10.0, -1.0)
