import numpy as np
import pytest

from fabrika_ikizi import machine, scenarios as sc, telemetry


def stream(n=40):
    t = np.arange(n) * 0.25
    return t, machine.daily_load(t), 40.0 + np.arange(n) * 0.01, 1.5 + np.arange(n) * 0.001


def test_round_trip_with_lost_samples(tmp_path):
    t, u, yt, yv = stream()
    yt = yt.copy()
    yv = yv.copy()
    yt[5] = np.nan
    yv[7:9] = np.nan
    path = tmp_path / "a.csv"
    telemetry.write_stream(path, t, u, yt, yv)
    t2, u2, yt2, yv2, dt = telemetry.read_stream(path)
    assert dt == 0.25 and np.array_equal(t, t2) and np.array_equal(u, u2)
    assert np.allclose(yt, yt2, equal_nan=True) and np.allclose(yv, yv2, equal_nan=True)
    lines = path.read_text().splitlines()
    assert lines[6].startswith("1.25,0.85,,")                         # sample 5: the temperature cell is empty
    assert lines[8].endswith(",") and lines[9].endswith(",")          # samples 7 and 8: the vibration cell is empty


def test_header_and_missing_columns(tmp_path):
    t, u, yt, yv = stream()
    path = tmp_path / "a.csv"
    telemetry.write_stream(path, t, u, yt, yv)
    assert path.read_text().splitlines()[0] == "zaman_saat,yuk,sicaklik_C,titresim_mm_s"
    bad = tmp_path / "b.csv"
    bad.write_text("zaman_saat,yuk,sicaklik_C\n" + "\n".join(f"{i},0.5,40" for i in range(20)))
    with pytest.raises(ValueError, match="titresim_mm_s"):
        telemetry.read_stream(bad)


def test_semicolon_files_with_decimal_commas_are_read(tmp_path):
    t, u, yt, yv = stream()
    lines = ["zaman_saat;yuk;sicaklik_C;titresim_mm_s"] + [f"{a};{b};{c};{d}".replace(".", ",") for a, b, c, d in zip(t, u, yt, yv)]
    path = tmp_path / "tr.csv"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    t2, u2, yt2, yv2, dt = telemetry.read_stream(path)
    assert dt == pytest.approx(0.25) and np.allclose(yt2, yt) and np.allclose(yv2, yv)


def test_irregular_sampling_and_bad_numbers_are_rejected(tmp_path):
    t, u, yt, yv = stream()
    t = t.copy()
    t[10] += 0.1
    path = tmp_path / "irr.csv"
    telemetry.write_stream(path, t, u, yt, yv)
    with pytest.raises(ValueError, match="düzenli"):
        telemetry.read_stream(path)
    path = tmp_path / "bad.csv"
    path.write_text("zaman_saat,yuk,sicaklik_C,titresim_mm_s\n" + "\n".join(f"{i * 0.25},0.5,abc,1.0" for i in range(20)))
    with pytest.raises(ValueError, match="okunamadı"):
        telemetry.read_stream(path)
    path = tmp_path / "short.csv"
    telemetry.write_stream(path, *[a[:5] for a in stream()])
    with pytest.raises(ValueError, match="en az 10"):
        telemetry.read_stream(path)
    path = tmp_path / "gap.csv"
    path.write_text("zaman_saat,yuk,sicaklik_C,titresim_mm_s\n" + "\n".join(f"{i * 0.25},,40,1.0" for i in range(20)))
    with pytest.raises(ValueError, match="yük"):
        telemetry.read_stream(path)


def test_a_simulated_stream_written_and_read_back_gives_the_same_twin_result(tmp_path):
    from fabrika_ikizi import twin
    from fabrika_ikizi.machine import MachineParams
    r = sc.run_makine("saglikli", hours=200.0, seed=1)
    path = tmp_path / "s.csv"
    telemetry.write_stream(path, r.truth.t, r.truth.u, r.y_temp, r.y_vib)
    t, u, yt, yv, dt = telemetry.read_stream(path)
    run = twin.MachineTwin(MachineParams(), dt, sc.TEMP_SENSOR, sc.VIB_SENSOR).run(t, u, yt, yv)
    assert np.allclose(run.x, r.run.x)
