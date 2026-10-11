"""Ready-made machine and line scenarios, and the line file format."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import calibrate, ctmc, linetwin, machine, sensors, twin
from . import line as L

MACHINE_SCENARIOS = {
    "saglikli": "arızasız makine; yanlış alarm sayısı ve kestirim kalitesi",
    "yatak": "300. saatte yatak arızası: aşınma hızı 3 katına çıkar",
    "sogutma": "300. saatte soğutma bozulması: ısıl direnç 1,3 katına çıkar",
    "sensor-takili": "300. saatte titreşim sensörü takılı kalır",
    "sensor-kayma": "300. saatte sıcaklık sensörü 0,02 K/saat kaymaya başlar",
    "veri-kaybi": "yüzde 10 rastgele örnek kaybı ve 300-340. saatler arasında titreşim verisi kesintisi",
}
FAULT_START = 300.0

TEMP_SENSOR = sensors.SensorSpec(0.3, resolution=0.1)
VIB_SENSOR = sensors.SensorSpec(0.15, resolution=0.01)


@dataclass
class MachineResult:
    scenario: str
    truth: machine.Truth
    y_temp: np.ndarray
    y_vib: np.ndarray
    run: twin.TwinRun
    summary: dict
    info: dict = field(default_factory=dict)


def scenario_setup(name):
    """Plant faults, sensor faults and extra dropout for a scenario."""
    pf, tf, vf, drop = (), (), (), 0.0
    if name == "yatak":
        pf = (machine.Fault("bearing", FAULT_START, 3.0),)
    elif name == "sogutma":
        pf = (machine.Fault("cooling", FAULT_START, 1.3),)
    elif name == "sensor-takili":
        vf = (sensors.SensorFault("stuck", FAULT_START),)
    elif name == "sensor-kayma":
        tf = (sensors.SensorFault("drift", FAULT_START, rate=0.02),)
    elif name == "veri-kaybi":
        vf = (sensors.SensorFault("burst", FAULT_START, FAULT_START + 40.0),)
        drop = 0.10
    elif name != "saglikli":
        raise ValueError(f"unknown scenario {name!r}; choose from {tuple(MACHINE_SCENARIOS)}")
    return pf, tf, vf, drop


def observe(truth, rng, temp_faults=(), vib_faults=(), dropout=0.0):
    ts = sensors.SensorSpec(TEMP_SENSOR.sigma, TEMP_SENSOR.resolution, dropout)
    vs = sensors.SensorSpec(VIB_SENSOR.sigma, VIB_SENSOR.resolution, dropout)
    return (sensors.read(ts, truth.temp, truth.t, rng, temp_faults),
            sensors.read(vs, truth.vibration, truth.t, rng, vib_faults))


def run_makine(scenario="saglikli", hours=1000.0, dt=0.25, seed=0, kappa=1.0, params=None, model_params=None,
               arl0=10000.0, q_kappa=0.03):
    params = params or machine.MachineParams()
    pf, tf, vf, drop = scenario_setup(scenario)
    truth = machine.simulate_plant(params, hours, dt, seed, pf, kappa=kappa, stop_at_failure=True)
    rng = np.random.default_rng(seed + 10_000)
    y_t, y_v = observe(truth, rng, tf, vf, drop)
    mt = twin.MachineTwin(params, dt, TEMP_SENSOR, VIB_SENSOR, arl0=arl0, model_params=model_params, q_kappa=q_kappa)
    run = mt.run(truth.t, truth.u, y_t, y_v)
    summary = machine_summary(scenario, truth, run, drop > 0)
    return MachineResult(scenario, truth, y_t, y_v, run, summary, {"params": params})


def machine_summary(scenario, truth, run, with_dropout=False):
    err = run.x - np.column_stack([truth.temp, truth.wear, truth.kappa])
    skip = min(40, len(truth.t) // 4)
    sd_w = run.wear_std
    s = {
        "senaryo": scenario, "ornek": len(truth.t), "sure_saat": float(truth.t[-1]),
        "ariza_saati": truth.fail_time,
        "sicaklik_rmse": float(np.sqrt(np.mean(err[skip:, 0] ** 2))),
        "asinma_rmse": float(np.sqrt(np.mean(err[skip:, 1] ** 2))),
        "asinma_2sigma_icinde": float(np.mean(np.abs(err[skip:, 1]) < 2.0 * sd_w[skip:])),
        "nis_ort": float(np.nanmean(run.nis[skip:])),
        "son_asinma_gercek": float(truth.wear[-1]), "son_asinma_tahmin": float(run.wear[-1]),
        "son_siddet_tahmin": float(run.kappa[-1]),
    }
    start = math.inf if scenario in ("saglikli", "veri-kaybi") else FAULT_START     # no real fault: every alarm is false
    before = [a for a in run.alarms if a[0] < start]
    after = [a for a in run.alarms if a[0] >= start]
    s["yanlis_alarm"] = len(before)
    s["alarmlar"] = [(round(t, 2), k) for t, k in run.alarms]
    s["tespit_gecikmesi_saat"] = float(after[0][0] - start) if after else float("nan")
    s["ilk_alarm_turu"] = after[0][1] if after else None
    # remaining life is judged 100 h after a fault (time for the twin to react), else three quarters into the record
    t_check = min(FAULT_START + 100.0, 0.9 * truth.t[-1]) if scenario not in ("saglikli", "veri-kaybi") else 0.75 * truth.t[-1]
    checkpoint = int(np.argmin(np.abs(truth.t - t_check)))
    med, lo, hi = run.rul(checkpoint)
    s["rul_ornek_saati"] = float(truth.t[checkpoint])
    s["rul_medyan"], s["rul_alt"], s["rul_ust"] = med, lo, hi
    s["rul_gercek"] = float(truth.fail_time - truth.t[checkpoint]) if truth.fail_index >= 0 else float("nan")
    # a chart that is high at the checkpoint means the sensors disagree with the model: the wear estimate, and so the RUL, is suspect
    s["rul_guvenilir"] = not bool(run.active[checkpoint])
    return s


# ----------------------------------------------------------------------------------------------
def example_line(kind="montaj"):
    """Two ready-made lines: ``montaj`` (exponential, exactly solvable) and ``gercekci`` (Erlang/lognormal times, scrap)."""
    if kind == "montaj":
        return L.Line((L.Station("Pres", 1.0, mtbf=120.0, mttr=8.0),
                       L.Station("Kaynak", 1.2, mtbf=90.0, mttr=10.0),
                       L.Station("Boya", 0.9, mtbf=200.0, mttr=15.0)), (4, 4))
    if kind == "gercekci":
        return L.Line((L.Station("Pres", 1.0, dist="erlang:4", mtbf=120.0, mttr=8.0),
                       L.Station("Kaynak", 1.2, dist="lognorm:0.5", mtbf=90.0, mttr=10.0, scrap=0.02),
                       L.Station("Boya", 0.9, dist="erlang:3", mtbf=200.0, mttr=15.0),
                       L.Station("Muayene", 1.0, dist="det", scrap=0.03)), (5, 5, 5))
    raise ValueError(f"unknown line {kind!r}; choose from ('montaj', 'gercekci')")


@dataclass
class LineScenarioResult:
    line: L.Line
    plant: L.LineResult
    summary: dict
    info: dict = field(default_factory=dict)


def run_hat(line, horizon=20000.0, warmup=500.0, seed=0, draws=100):
    """Run the plant, calibrate the twin from its log, and compare the twin's prediction with the plant."""
    plant = L.simulate(line, horizon, warmup, seed)
    pred = linetwin.predictive_interval(plant, line, draws=draws, seed=seed)
    cal = calibrate.calibrated_line(plant, line)
    est = calibrate.estimate(plant, line)
    oee = plant.oee(line.stations)
    cap = linetwin.capacity_bound(line)
    summary = {
        "istasyon_sayisi": len(line.stations), "sure_dk": horizon,
        "gozlenen_uretim": plant.throughput, "tahmin": pred.estimate, "tahmin_alt": pred.low,
        "tahmin_ust": pred.high, "tahmin_yontemi": pred.method, "kapasite_siniri": cap,
        "darbogaz": line.stations[line.bottleneck].name, "darbogaz_gozlenen": line.stations[int(np.argmax(
            plant.busy + plant.down))].name,
        "ort_wip": plant.wip, "dongu_suresi": plant.cycle_time, "little_artigi": plant.little_residual,
        "fire": plant.scrapped, "oee_hat": float(plant.throughput * line.stations[-1].mean_service
                                                 / max(line.stations[-1].availability, 1e-9)),
        "istasyonlar": [
            {"ad": st.name, "mesgul": float(plant.busy[i]), "bloke": float(plant.blocked[i]),
             "aclik": float(plant.starved[i]), "ariza": float(plant.down[i]),
             "kullanilabilirlik": o[0], "performans": o[1], "kalite": o[2], "oee": o[3],
             "sure_tahmin": e.mean_service[0], "sure_alt": e.mean_service[1], "sure_ust": e.mean_service[2],
             "sure_gercek": st.mean_service}
            for i, (st, o, e) in enumerate(zip(line.stations, oee, est))],
        "tampon_ort": [float(x) for x in plant.mean_buffers],
    }
    return LineScenarioResult(line, plant, summary, {"calibrated": cal, "estimates": est})


# ----------------------------------------------------------------------------------------------
def line_to_dict(line):
    return {
        "istasyonlar": [{"ad": s.name, "sure": s.mean_service, "dagilim": s.dist,
                         "mtbf": None if math.isinf(s.mtbf) else s.mtbf,
                         "mttr": s.mttr, "fire": s.scrap} for s in line.stations],
        "tamponlar": [None if math.isinf(b) else int(b) for b in line.buffers],
        "varis_hizi": line.arrival_rate,
        "bekleme_kapasitesi": None if math.isinf(line.queue_capacity) else int(line.queue_capacity),
    }


def line_from_dict(data):
    try:
        stations = tuple(L.Station(s["ad"], float(s["sure"]), s.get("dagilim", "exp"),
                                   math.inf if s.get("mtbf") is None else float(s["mtbf"]),
                                   float(s.get("mttr", 0.0)), float(s.get("fire", 0.0)))
                         for s in data["istasyonlar"])
        buffers = tuple(math.inf if b is None else int(b) for b in data.get("tamponlar", []))
        q = data.get("bekleme_kapasitesi")
        return L.Line(stations, buffers, data.get("varis_hizi"), math.inf if q is None else int(q))
    except KeyError as exc:
        raise ValueError(f"hat dosyasında eksik anahtar: {exc}")


def load_line(path):
    with open(path, encoding="utf-8") as fh:
        return line_from_dict(json.load(fh))


def save_line(line, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(line_to_dict(line), fh, ensure_ascii=False, indent=2)


def exact_summary(line):
    """Exact solver output as a dict, or None when the model is outside its scope."""
    if not linetwin.exact_applicable(line):
        return None
    r = ctmc.solve(line)
    return {"durum_sayisi": r.n_states, "uretim": r.throughput, "ort_wip": r.wip,
            "mesgul": r.busy.tolist(), "bloke": r.blocked.tolist(), "aclik": r.starved.tolist(),
            "ariza": r.down.tolist()}
