import json

import pytest

from fabrika_ikizi import cli


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_senaryolar_lists_machine_scenarios_and_lines(capsys):
    code, out, _ = run(capsys, "senaryolar")
    assert code == 0 and all(k in out for k in ("saglikli", "yatak", "sogutma", "sensor-takili", "montaj", "gercekci"))


def test_makine_text_json_and_plot(capsys, tmp_path):
    png = tmp_path / "m.png"
    code, out, _ = run(capsys, "makine", "--senaryo", "yatak", "--saat", 600, "--tohum", 1, "--grafik", png)
    assert code == 0 and "Makine ikizi: senaryo yatak" in out and "kalan ömür" in out and "wear-rate" in out
    assert png.stat().st_size > 20000
    code, out, _ = run(capsys, "makine", "--senaryo", "sogutma", "--saat", 400, "--json")
    data = json.loads(out)
    assert data["senaryo"] == "sogutma" and data["tespit_gecikmesi_saat"] < 2.0 and data["rul_guvenilir"] is False


def test_makine_cooling_text_warns_that_the_remaining_life_is_unreliable(capsys):
    code, out, _ = run(capsys, "makine", "--senaryo", "sogutma", "--saat", 400)
    assert "güvenilir değil" in out


def test_simule_then_izle_round_trip(capsys, tmp_path):
    csv = tmp_path / "akis.csv"
    code, out, _ = run(capsys, "simule", "--senaryo", "yatak", "--saat", 600, "--tohum", 2, "--cikti", csv)
    assert code == 0 and csv.exists() and "örnek yazıldı" in out
    png = tmp_path / "i.png"
    code, out, _ = run(capsys, "izle", "--veri", csv, "--grafik", png)
    assert code == 0 and "İzleme:" in out and "kalan ömür" in out and png.stat().st_size > 20000
    code, out, _ = run(capsys, "izle", "--veri", csv, "--json", "--arl0", 1e6)
    data = json.loads(out)
    assert data["ornek"] > 1000 and 0.0 < data["son_asinma_tahmin"] < 1.2 and data["rul_alt"] <= data["rul_medyan"] <= data["rul_ust"]
    assert any(kind == "wear-rate" for _, kind in data["alarmlar"])


def test_izle_accepts_parameter_overrides_and_rejects_unknown_ones(capsys, tmp_path):
    csv = tmp_path / "akis.csv"
    run(capsys, "simule", "--senaryo", "saglikli", "--saat", 200, "--cikti", csv)
    good = tmp_path / "p.json"
    good.write_text(json.dumps({"r_th": 8.0, "kv": 4.5}))
    assert run(capsys, "izle", "--veri", csv, "--parametre", good, "--json")[0] == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"yanlis": 1}))
    code, _, err = run(capsys, "izle", "--veri", csv, "--parametre", bad)
    assert code == 2 and "bilinmeyen parametre" in err


def test_hat_exact_line(capsys, tmp_path):
    png = tmp_path / "h.png"
    code, out, _ = run(capsys, "hat", "--hat", "montaj", "--sure", 8000, "--cekilis", 30, "--tohum", 2, "--grafik", png)
    assert code == 0 and "Hat ikizi:" in out and "kesin çözüm" in out and "darboğaz" in out and png.stat().st_size > 10000
    assert "aralığın içinde" in out
    code, out, _ = run(capsys, "hat", "--hat", "montaj", "--sure", 5000, "--cekilis", 20, "--json")
    data = json.loads(out)
    assert data["kesin_cozum"]["durum_sayisi"] == 384 and data["tahmin_yontemi"] == "exact"
    assert len(data["istasyonlar"]) == 3


def test_hat_from_a_file_and_hat_yaz(capsys, tmp_path):
    path = tmp_path / "hat.json"
    assert run(capsys, "hat-yaz", "--hat", "montaj", "--cikti", path)[0] == 0
    code, out, _ = run(capsys, "hat", "--hat-dosyasi", path, "--sure", 4000, "--cekilis", 10)
    assert code == 0 and "Pres" in out and "Boya" in out


def test_ne_olur_ranking_and_buffer_table(capsys):
    code, out, _ = run(capsys, "ne-olur", "--hat", "montaj", "--tampon-arasi", 1, "--tampon-boyutlari", "0,2,8")
    assert code == 0 and "kazancı" in out and "tamponu büyüklüğüne göre" in out
    ranking = [line.split()[0] for line in out.splitlines() if line.startswith("    ") and "parça/dk (" in line]
    assert ranking[0] == "Kaynak"
    code, out, _ = run(capsys, "ne-olur", "--hat", "montaj", "--json")
    data = json.loads(out)
    assert max(data["kazanc"], key=data["kazanc"].get) == "Kaynak"


def test_usage_errors(capsys, tmp_path):
    code, _, err = run(capsys, "ne-olur", "--hat", "montaj", "--tampon-arasi", 7)
    assert code == 2 and "arasında" in err
    code, _, err = run(capsys, "izle", "--veri", tmp_path / "yok.csv")
    assert code == 2 and "hata" in err
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    code, _, err = run(capsys, "izle", "--veri", bad)
    assert code == 2 and "eksik sütun" in err


def test_dogrulama_exit_code(capsys, monkeypatch):
    from fabrika_ikizi import checks
    ok, bad = checks.Check("a", True, "fine"), checks.Check("b", False, "broken")
    monkeypatch.setattr(checks, "run_all", lambda: [ok])
    code, out, _ = run(capsys, "dogrulama")
    assert code == 0 and "tüm doğrulamalar geçti" in out
    monkeypatch.setattr(checks, "run_all", lambda: [ok, bad])
    code, out, _ = run(capsys, "dogrulama")
    assert code == 1 and "1 başarısız" in out


def test_unknown_command_and_number_formatting():
    with pytest.raises(SystemExit) as exc:
        cli.main(["uc"])
    assert exc.value.code == 2
    assert cli.f(3.14159, 2) == "3,14" and cli.f(float("nan")) == "-" and cli.pct(0.256) == "%25,6"


def test_rul_sinama_command(capsys):
    code, out, _ = run(capsys, "rul-sinama", "--q-kappa", 0.08, "--sonra", 100, "--makine", 12, "--saglam-makine", 0, "--json")
    data = json.loads(out)
    assert code == 0 and data["makine"] >= 8 and 0.0 <= data["kapsama"] <= 1.0 and 0.5 < data["medyan_oran"] < 2.0
    code, out, _ = run(capsys, "rul-sinama", "--makine", 6, "--saglam-makine", 2)
    assert "kapsama oranı" in out and "bedeli" in out
