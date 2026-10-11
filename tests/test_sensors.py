import numpy as np
import pytest

from fabrika_ikizi import sensors
from fabrika_ikizi.sensors import SensorFault, SensorSpec

T = np.arange(0.0, 1000.0, 0.25)
SIGNAL = np.full_like(T, 50.0)


def read(spec, faults=(), seed=0):
    return sensors.read(spec, SIGNAL, T, np.random.default_rng(seed), faults)


def test_noise_bias_and_effective_sigma():
    y = read(SensorSpec(0.3, bias=0.5))
    assert y.mean() == pytest.approx(50.5, abs=4 * 0.3 / np.sqrt(len(y)))
    assert y.std() == pytest.approx(0.3, rel=0.05)
    assert SensorSpec(0.3, resolution=0.0).effective_sigma() == 0.3
    assert SensorSpec(0.3, resolution=0.1).effective_sigma() == pytest.approx(np.sqrt(0.09 + 0.01 / 12))


def test_quantisation_puts_readings_on_the_grid_and_adds_the_expected_variance():
    spec = SensorSpec(0.3, resolution=0.1)
    y = read(spec)
    assert np.allclose(np.round(y / 0.1) * 0.1, y)
    assert y.std() == pytest.approx(spec.effective_sigma(), rel=0.05)


def test_dropout_fraction():
    y = read(SensorSpec(0.1, dropout=0.2))
    assert np.isnan(y).mean() == pytest.approx(0.2, abs=0.03)


def test_stuck_sensor_repeats_its_last_value():
    y = read(SensorSpec(0.1), (SensorFault("stuck", 400.0, 600.0),))
    stuck = (T >= 400.0) & (T < 600.0)
    assert np.all(y[stuck] == y[stuck][0])
    assert y[~stuck].std() > 0.05                      # outside the window it still varies


def test_drift_grows_linearly_and_stops_at_the_end():
    spec = SensorSpec(0.01)
    y = read(spec, (SensorFault("drift", 100.0, 500.0, rate=0.02),))
    assert y[T == 300.0][0] - 50.0 == pytest.approx(0.02 * 200.0, abs=0.05)
    assert y[T == 600.0][0] - 50.0 == pytest.approx(0.0, abs=0.05)


def test_burst_removes_samples_in_the_window_only():
    y = read(SensorSpec(0.1), (SensorFault("burst", 300.0, 340.0),))
    window = (T >= 300.0) & (T < 340.0)
    assert np.isnan(y[window]).all() and not np.isnan(y[~window]).any()


def test_unknown_fault_kind_and_reproducibility():
    with pytest.raises(ValueError):
        SensorFault("melted", 1.0)
    assert np.array_equal(read(SensorSpec(0.2), seed=3), read(SensorSpec(0.2), seed=3))
