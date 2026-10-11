import dataclasses

import numpy as np
import pytest

from fabrika_ikizi import machine, scenarios as sc, twin
from fabrika_ikizi.machine import MachineParams


def healthy(seed=0, hours=500.0, **kw):
    return sc.run_makine("saglikli", hours=hours, seed=seed, **kw)


def test_twin_tracks_temperature_and_wear_of_a_healthy_machine():
    r = healthy()
    s = r.summary
    assert s["sicaklik_rmse"] < 0.2 and s["asinma_rmse"] < 0.01
    assert 0.9 < s["asinma_2sigma_icinde"] <= 1.0           # the claimed uncertainty is honest
    assert s["nis_ort"] == pytest.approx(2.0, abs=0.2)
    assert abs(r.run.wear[-1] - r.truth.wear[-1]) < 3 * r.run.wear_std[-1] + 0.01


def test_estimates_beat_the_raw_sensors():
    r = healthy(seed=1)
    run, truth = r.run, r.truth
    raw_temp = np.nanstd(r.y_temp[50:] - truth.temp[50:])
    raw_wear = np.nanstd((r.y_vib[50:] - truth.params.v0) / truth.params.kv - truth.wear[50:])
    assert np.std(run.temp[50:] - truth.temp[50:]) < 0.5 * raw_temp
    assert np.std(run.wear[50:] - truth.wear[50:]) < 0.5 * raw_wear


def test_run_outputs_have_consistent_shapes_and_accessors():
    r = healthy(hours=100.0)
    run = r.run
    n = len(r.truth.t)
    assert run.x.shape == (n, 3) and run.P.shape == (n, 3, 3) and run.z.shape == (n, 2) and run.nis.shape == (n,)
    assert np.array_equal(run.temp, run.x[:, 0]) and np.array_equal(run.wear, run.x[:, 1]) and np.array_equal(run.kappa, run.x[:, 2])
    assert np.allclose(run.wear_std, np.sqrt(run.P[:, 1, 1]))
    assert np.all(np.linalg.eigvalsh(run.P[-1]) > 0)


def test_uncertainty_shrinks_as_data_arrive():
    run = healthy().run
    assert run.wear_std[200] < run.wear_std[0] / 5
    assert np.sqrt(run.P[-1, 2, 2]) < np.sqrt(run.P[10, 2, 2])


def test_cooling_fault_is_flagged_within_hours_on_the_temperature_chart():
    s = sc.run_makine("sogutma", hours=400.0, seed=1, arl0=1e6).summary
    assert s["tespit_gecikmesi_saat"] < 2.0 and s["ilk_alarm_turu"] == "T+" and s["yanlis_alarm"] == 0
    assert not s["rul_guvenilir"]                            # the twin says its own remaining-life number is suspect


def test_bearing_fault_raises_the_wear_rate_alarm_before_failure():
    for seed in range(3):
        s = sc.run_makine("yatak", hours=700.0, seed=seed, arl0=1e6).summary      # a long design run length: no chance alarms
        assert s["ilk_alarm_turu"] is not None and s["yanlis_alarm"] == 0
        assert 300.0 + s["tespit_gecikmesi_saat"] < s["ariza_saati"] - 20.0
        assert any(kind == "wear-rate" for _, kind in s["alarmlar"])
        assert s["son_siddet_tahmin"] > 2.0


def test_sensor_faults_are_detected():
    for name, limit in (("sensor-kayma", 80.0), ("sensor-takili", 200.0)):
        s = sc.run_makine(name, hours=700.0, seed=2).summary
        assert s["tespit_gecikmesi_saat"] < limit, name


def test_lost_samples_do_not_cause_alarms_or_break_the_estimate():
    s = sc.run_makine("veri-kaybi", hours=700.0, seed=0, arl0=1e6).summary
    assert s["yanlis_alarm"] == 0 and s["asinma_rmse"] < 0.01
    assert s["nis_ort"] < 2.0                                # fewer channels per sample on average


def test_all_samples_missing_is_survivable():
    p = MachineParams()
    t = np.arange(0.0, 20.0, 0.25)
    u = machine.daily_load(t)
    nan = np.full_like(t, np.nan)
    run = twin.MachineTwin(p, 0.25, sc.TEMP_SENSOR, sc.VIB_SENSOR).run(t, u, nan, nan)
    assert np.isfinite(run.x).all() and np.isnan(run.z).all() and run.alarms == []
    assert run.wear_std[-1] > run.wear_std[0]                # no data: the uncertainty only grows


def test_a_wrong_model_is_visible_in_the_innovations():
    """A twin that believes the thermal resistance is 25% lower than the plant's is inconsistent with its sensors."""
    p = MachineParams()
    bad = dataclasses.replace(p, r_th=p.r_th * 0.75)
    good = sc.run_makine("saglikli", hours=300.0, seed=0, params=p).summary
    wrong = sc.run_makine("saglikli", hours=300.0, seed=0, params=p, model_params=bad).summary
    assert good["nis_ort"] < 2.3 and wrong["nis_ort"] > 10.0 and wrong["asinma_rmse"] > 5 * good["asinma_rmse"]


def test_remaining_life_of_a_healthy_machine_is_in_a_sensible_range():
    r = healthy(seed=2, hours=1000.0)
    med, lo, hi = r.run.rul(len(r.truth.t) // 2)
    assert lo < med < hi and 100 < med < 2000
    assert r.run.rul(len(r.truth.t) // 2, propagate=False)[2] - r.run.rul(len(r.truth.t) // 2, propagate=False)[1] < hi - lo


def test_chart_alarm_threshold_follows_the_requested_run_length():
    loose = twin.MachineTwin(MachineParams(), 0.25, sc.TEMP_SENSOR, sc.VIB_SENSOR, arl0=500.0)
    tight = twin.MachineTwin(MachineParams(), 0.25, sc.TEMP_SENSOR, sc.VIB_SENSOR, arl0=50000.0)
    assert tight.h > loose.h > 3.0
