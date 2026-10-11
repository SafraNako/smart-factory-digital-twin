import json
import math

import pytest

from fabrika_ikizi import scenarios as sc


def test_scenario_setup_names_and_errors():
    assert set(sc.MACHINE_SCENARIOS) == {"saglikli", "yatak", "sogutma", "sensor-takili", "sensor-kayma", "veri-kaybi"}
    for name in sc.MACHINE_SCENARIOS:
        sc.scenario_setup(name)
    with pytest.raises(ValueError, match="unknown scenario"):
        sc.scenario_setup("yangin")


def test_machine_summary_has_everything_the_cli_prints():
    s = sc.run_makine("yatak", hours=500.0, seed=0).summary
    for key in ("sicaklik_rmse", "asinma_rmse", "asinma_2sigma_icinde", "nis_ort", "alarmlar", "yanlis_alarm",
                "tespit_gecikmesi_saat", "rul_medyan", "rul_alt", "rul_ust", "rul_gercek", "rul_guvenilir", "ariza_saati"):
        assert key in s
    assert s["rul_alt"] <= s["rul_medyan"] <= s["rul_ust"]
    json.dumps(s, default=float)


def test_the_machine_run_stops_at_failure_and_reports_it():
    r = sc.run_makine("yatak", hours=2000.0, seed=1)
    assert r.truth.fail_index == len(r.truth.t) - 1 and r.summary["ariza_saati"] == pytest.approx(r.truth.t[-1])
    assert r.summary["rul_gercek"] > 0


def test_wear_severity_option_shortens_the_life():
    slow = sc.run_makine("saglikli", hours=3000.0, seed=2, kappa=0.8).summary["ariza_saati"]
    fast = sc.run_makine("saglikli", hours=3000.0, seed=2, kappa=1.5).summary["ariza_saati"]
    assert fast < 0.7 * slow


def test_example_lines_and_unknown_name():
    montaj, gercekci = sc.example_line("montaj"), sc.example_line("gercekci")
    assert len(montaj.stations) == 3 and len(gercekci.stations) == 4
    assert all(s.dist == "exp" and s.scrap == 0 for s in montaj.stations)
    assert any(s.scrap > 0 for s in gercekci.stations) and any(s.dist != "exp" for s in gercekci.stations)
    with pytest.raises(ValueError, match="unknown line"):
        sc.example_line("kayik")


@pytest.mark.parametrize("kind", ["montaj", "gercekci"])
def test_line_file_round_trip(tmp_path, kind):
    line = sc.example_line(kind)
    path = tmp_path / "hat.json"
    sc.save_line(line, path)
    assert sc.load_line(path) == line
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["istasyonlar"][0]["ad"] == "Pres" and data["tamponlar"][0] == line.buffers[0]


def test_line_file_with_infinite_values_and_poisson_source(tmp_path):
    path = tmp_path / "hat.json"
    path.write_text(json.dumps({"istasyonlar": [{"ad": "A", "sure": 1.0}, {"ad": "B", "sure": 1.5, "dagilim": "det"}],
                                "tamponlar": [None], "varis_hizi": 0.4, "bekleme_kapasitesi": None}), encoding="utf-8")
    line = sc.load_line(path)
    assert math.isinf(line.buffers[0]) and math.isinf(line.queue_capacity) and line.arrival_rate == 0.4
    assert math.isinf(line.stations[0].mtbf) and line.stations[1].dist == "det"


def test_line_file_errors(tmp_path):
    with pytest.raises(ValueError, match="eksik anahtar"):
        sc.line_from_dict({"istasyonlar": [{"ad": "A"}]})
    with pytest.raises(ValueError):
        sc.line_from_dict({"istasyonlar": [{"ad": "A", "sure": 1.0}, {"ad": "B", "sure": 1.0}], "tamponlar": []})


def test_exact_summary_is_none_outside_its_scope():
    assert sc.exact_summary(sc.example_line("gercekci")) is None
    ex = sc.exact_summary(sc.example_line("montaj"))
    assert ex["durum_sayisi"] == 384 and ex["uretim"] == pytest.approx(0.61837, abs=1e-4)


def test_hat_run_reports_the_stations_and_the_oee():
    r = sc.run_hat(sc.example_line("gercekci"), horizon=3000.0, warmup=300.0, seed=1, draws=6)
    s = r.summary
    assert len(s["istasyonlar"]) == 4 and s["fire"] > 0 and s["tahmin_yontemi"] == "simulation"
    for x in s["istasyonlar"]:
        assert x["oee"] == pytest.approx(x["kullanilabilirlik"] * x["performans"] * x["kalite"])
        assert abs(x["mesgul"] + x["bloke"] + x["aclik"] + x["ariza"] - 1.0) < 1e-9
