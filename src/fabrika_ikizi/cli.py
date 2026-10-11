"""Command line interface: ``fabrika`` (subcommands are ASCII-transliterated Turkish)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys

import numpy as np

from . import __version__


class UsageError(Exception):
    pass


def f(x, nd=2):
    """Number with a Turkish decimal comma."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=1):
    return f"%{f(100.0 * x, nd)}"


def dump(data):
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return clean(v.item())
        return v
    print(json.dumps(clean(data), ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------------------------
def machine_text(s, header):
    lines = [header,
             f"  sıcaklık tahmin hatası (rms): {f(s['sicaklik_rmse'], 3)} °C; aşınma tahmin hatası (rms): {f(s['asinma_rmse'], 4)}; "
             f"gerçek aşınma ±2σ içinde: {pct(s['asinma_2sigma_icinde'])}",
             f"  yenilik tutarlılığı (NIS ort., beklenen ≈ ölçülen kanal sayısı): {f(s['nis_ort'], 2)}"]
    if s.get("ariza_saati") is not None and not math.isnan(s.get("ariza_saati", float("nan"))):
        lines.append(f"  makine {f(s['ariza_saati'], 1)}. saatte arızalandı (aşınma 1'e ulaştı)")
    lines.append(f"  alarmlar ({len(s['alarmlar'])}): " + (", ".join(f"{k}@{f(t, 1)}h" for t, k in s["alarmlar"][:8])
                                                        + (" ..." if len(s["alarmlar"]) > 8 else "") if s["alarmlar"] else "yok"))
    if "yanlis_alarm" in s:
        lines.append(f"  yanlış alarm: {s['yanlis_alarm']}; arıza tespit gecikmesi: {f(s['tespit_gecikmesi_saat'], 1)} saat "
                     f"({s['ilk_alarm_turu'] or '-'})")
    lines.append(f"  kalan ömür ({f(s['rul_ornek_saati'], 0)}. saatte): medyan {f(s['rul_medyan'], 0)} saat, %80 aralık "
                 f"[{f(s['rul_alt'], 0)}; {f(s['rul_ust'], 0)}]"
                 + (f"; gerçek {f(s['rul_gercek'], 0)}" if s.get("rul_gercek") is not None and not math.isnan(s["rul_gercek"]) else "")
                 + ("" if s.get("rul_guvenilir", True) else "  [DİKKAT: sensör-model tutarsızlığı alarmı var, tahmin güvenilir değil]"))
    return "\n".join(lines)


def cmd_senaryolar(args):
    from .scenarios import MACHINE_SCENARIOS
    print("makine senaryoları (fabrika makine --senaryo ...):")
    for k, v in MACHINE_SCENARIOS.items():
        print(f"  {k:14s} {v}")
    print("hatlar (fabrika hat --hat ...): montaj (üstel süreler, tam çözülebilir), gercekci (Erlang/lognormal süreler, fire)")
    return 0


def cmd_makine(args):
    from . import render, scenarios
    r = scenarios.run_makine(args.senaryo, hours=args.saat, seed=args.tohum, kappa=args.siddet, q_kappa=args.q_kappa)
    if args.json:
        dump(r.summary)
    else:
        print(machine_text(r.summary, f"Makine ikizi: senaryo {args.senaryo}, {r.summary['ornek']} örnek, "
                                       f"{f(r.summary['sure_saat'], 0)} saat, tohum {args.tohum}"))
    if args.grafik:
        render.plot_machine(r.run, r.y_temp, r.y_vib, r.info["params"], args.grafik, truth=r.truth,
                            title=f"Makine ikizi — senaryo: {args.senaryo}")
        print(f"grafik: {args.grafik}")
    return 0


def cmd_simule(args):
    import numpy as np
    from . import machine, scenarios, telemetry
    params = machine.MachineParams()
    pf, tf, vf, drop = scenarios.scenario_setup(args.senaryo)
    truth = machine.simulate_plant(params, args.saat, args.adim, args.tohum, pf, kappa=args.siddet,
                                   stop_at_failure=True)
    y_t, y_v = scenarios.observe(truth, np.random.default_rng(args.tohum + 10_000), tf, vf, drop)
    telemetry.write_stream(args.cikti, truth.t, truth.u, y_t, y_v)
    print(f"{len(truth.t)} örnek yazıldı: {args.cikti} (senaryo {args.senaryo}, adım {f(args.adim, 3)} saat"
          + (f", makine {f(truth.fail_time, 1)}. saatte arızalandı" if truth.fail_index >= 0 else "") + ")")
    return 0


def parse_params(path):
    from .machine import MachineParams
    params = MachineParams()
    if path:
        with open(path, encoding="utf-8") as fh:
            changes = json.load(fh)
        names = {x.name for x in dataclasses.fields(MachineParams)}
        bad = sorted(set(changes) - names)
        if bad:
            raise UsageError(f"bilinmeyen parametre: {', '.join(bad)} (geçerli: {', '.join(sorted(names))})")
        params = dataclasses.replace(params, **changes)
    return params


def cmd_izle(args):
    from . import render, telemetry, twin
    from .sensors import SensorSpec
    t, u, y_t, y_v, dt = telemetry.read_stream(args.veri)
    params = parse_params(args.parametre)
    tw = twin.MachineTwin(params, dt, SensorSpec(args.sigma_sicaklik), SensorSpec(args.sigma_titresim),
                          arl0=args.arl0, q_kappa=args.q_kappa)
    run = tw.run(t, u, y_t, y_v)
    k = min(int(len(t) * 0.75), len(t) - 1) if args.kalan_omur_saati is None else int(np.argmin(np.abs(t - args.kalan_omur_saati)))
    med, lo, hi = run.rul(k)
    s = {"ornek": len(t), "sure_saat": float(t[-1] - t[0]), "adim_saat": dt,
         "son_sicaklik_tahmin": float(run.temp[-1]), "son_asinma_tahmin": float(run.wear[-1]),
         "son_asinma_std": float(run.wear_std[-1]), "son_siddet_tahmin": float(run.kappa[-1]),
         "nis_ort": float(np.nanmean(run.nis)), "kayip_ornek": int(np.isnan(y_t).sum() + np.isnan(y_v).sum()),
         "alarmlar": [(round(a, 2), kind) for a, kind in run.alarms],
         "rul_ornek_saati": float(t[k]), "rul_medyan": med, "rul_alt": lo, "rul_ust": hi}
    if args.json:
        dump(s)
    else:
        print(f"İzleme: {s['ornek']} örnek, {f(s['sure_saat'], 1)} saat, adım {f(dt, 3)} saat, kayıp örnek {s['kayip_ornek']}")
        print(f"  son durum: sıcaklık {f(s['son_sicaklik_tahmin'], 1)} °C, aşınma {f(s['son_asinma_tahmin'], 3)} ± {f(s['son_asinma_std'], 3)}, "
              f"aşınma hızı çarpanı κ {f(s['son_siddet_tahmin'], 2)}")
        print(f"  NIS ort. {f(s['nis_ort'], 2)}; alarmlar ({len(s['alarmlar'])}): "
              + (", ".join(f"{kk}@{f(a, 1)}h" for a, kk in s["alarmlar"][:8]) if s["alarmlar"] else "yok"))
        print(f"  kalan ömür ({f(s['rul_ornek_saati'], 0)}. saatte): medyan {f(med, 0)} saat, %80 aralık [{f(lo, 0)}; {f(hi, 0)}]")
    if args.grafik:
        render.plot_machine(run, y_t, y_v, params, args.grafik, title=f"Makine ikizi — {args.veri}")
        print(f"grafik: {args.grafik}")
    return 0


# ----------------------------------------------------------------------------------------------
def get_line(args):
    from . import scenarios
    if getattr(args, "hat_dosyasi", None):
        return scenarios.load_line(args.hat_dosyasi)
    return scenarios.example_line(args.hat)


def line_header(line):
    parts = []
    for s in line.stations:
        text = f"{s.name} ({f(s.mean_service, 2)} dk {s.dist}"
        if math.isfinite(s.mtbf):
            text += f", MTBF {f(s.mtbf, 0)}/MTTR {f(s.mttr, 0)}"
        if s.scrap:
            text += f", fire {pct(s.scrap, 0)}"
        parts.append(text + ")")
    bufs = " → ".join(str("∞" if math.isinf(b) else int(b)) for b in line.buffers)
    return " → ".join(parts) + (f"; tamponlar: {bufs}" if bufs else "")


def cmd_hat(args):
    from . import render, scenarios
    line = get_line(args)
    r = scenarios.run_hat(line, horizon=args.sure, warmup=min(500.0, args.sure * 0.05), seed=args.tohum, draws=args.cekilis)
    s = r.summary
    exact = scenarios.exact_summary(line)
    if args.json:
        dump({**s, "kesin_cozum": exact})
    else:
        print("Hat ikizi: " + line_header(line))
        print(f"  tesis ({f(s['sure_dk'], 0)} dk simülasyon): üretim {f(s['gozlenen_uretim'], 4)} parça/dk, ort. WIP {f(s['ort_wip'], 2)}, "
              f"çevrim süresi {f(s['dongu_suresi'], 2)} dk (Little artığı {f(s['little_artigi'] * 100, 2)}%), fire {s['fire']}")
        print(f"  ikiz (kayıttan kalibre, {s['tahmin_yontemi']}): üretim {f(s['tahmin'], 4)} parça/dk, %90 aralık "
              f"[{f(s['tahmin_alt'], 4)}; {f(s['tahmin_ust'], 4)}]"
              + ("  → tesisteki değer aralığın içinde" if s["tahmin_alt"] <= s["gozlenen_uretim"] <= s["tahmin_ust"]
                 else "  → tesisteki değer aralığın DIŞINDA"))
        if exact:
            print(f"  gerçek parametrelerle kesin çözüm (Markov zinciri, {exact['durum_sayisi']} durum): {f(exact['uretim'], 4)} parça/dk")
        print(f"  darboğaz (kapasite): {s['darbogaz']}; tesiste en yüklü istasyon: {s['darbogaz_gozlenen']}; "
              f"kapasite üst sınırı {f(s['kapasite_siniri'], 4)} parça/dk")
        print("  istasyon       çalışıyor  bloke  bekliyor  arızalı   OEE   süre tahmini (gerçek)")
        for x in s["istasyonlar"]:
            print(f"  {x['ad']:12s}   {pct(x['mesgul'], 1):>7s} {pct(x['bloke'], 1):>7s} {pct(x['aclik'], 1):>8s} "
                  f"{pct(x['ariza'], 1):>7s} {pct(x['oee'], 1):>7s}   {f(x['sure_tahmin'], 3)} [{f(x['sure_alt'], 3)}; {f(x['sure_ust'], 3)}] ({f(x['sure_gercek'], 3)})")
    if args.grafik:
        render.plot_line(r, args.grafik)
        print(f"grafik: {args.grafik}")
    return 0


def cmd_ne_olur(args):
    from . import linetwin
    line = get_line(args)
    kw = {} if linetwin.exact_applicable(line) else {"horizon": 6000.0, "replications": 4}
    base, gains = linetwin.speedup_ranking(line, factor=args.hizlandir, **kw)
    data = {"temel_uretim": base, "hizlandirma": args.hizlandir,
            "kazanc": {st.name: float(g) for st, g in zip(line.stations, gains)}}
    if args.tampon_arasi is not None:
        if not 0 <= args.tampon_arasi < len(line.buffers):
            raise UsageError(f"--tampon-arasi 0 ile {len(line.buffers) - 1} arasında olmalı")
        sizes = [int(x) for x in args.tampon_boyutlari.split(",")]
        data["tampon"] = dict(zip(sizes, [float(v) for v in linetwin.buffer_study(line, args.tampon_arasi, sizes, **kw)]))
    if args.json:
        dump(data)
        return 0
    print("Ne olur? " + line_header(line))
    print(f"  temel üretim: {f(base, 4)} parça/dk")
    print(f"  her istasyon {f(args.hizlandir, 2)} kat hızlanırsa üretim kazancı:")
    for st, g in sorted(zip(line.stations, gains), key=lambda p: -p[1]):
        print(f"    {st.name:12s} {'+' if g >= 0 else ''}{f(g, 4)} parça/dk ({'+' if g >= 0 else ''}{pct(g / base, 2)})")
    if "tampon" in data:
        a = args.tampon_arasi
        print(f"  {line.stations[a].name} → {line.stations[a + 1].name} tamponu büyüklüğüne göre üretim:")
        for n, v in data["tampon"].items():
            print(f"    {n:>4d} yer: {f(v, 4)} parça/dk")
    return 0


def cmd_rul_sinama(args):
    from . import checks
    used, coverage, ratio = checks.rul_after_fault(q_kappa=args.q_kappa, after=args.sonra, machines=args.makine)
    data = {"makine": used, "q_kappa": args.q_kappa, "sonra_saat": args.sonra, "kapsama": coverage, "medyan_oran": ratio}
    if args.saglam_makine:
        rms, mean_peak, max_peak = checks.healthy_kappa_noise(args.q_kappa, machines=args.saglam_makine)
        data.update(saglam_kappa_rms=rms, saglam_kappa_tepe_ort=mean_peak, saglam_kappa_tepe_maks=max_peak)
    if args.json:
        dump(data)
    else:
        print(f"Aşınma hızı 3 katına çıktıktan {f(args.sonra, 0)} saat sonra kalan ömür aralığı (q_kappa = {f(args.q_kappa, 3)}, {used} makine):")
        print(f"  %80'lik aralığın gerçek kalan ömrü kapsama oranı: {pct(coverage, 0)}; ikiz medyanı / gerçek kalan ömür = {f(ratio, 2)}")
        if args.saglam_makine:
            print(f"  bedeli: sağlam {args.saglam_makine} makinede κ tahmini 1'den rms {f(data['saglam_kappa_rms'], 2)} sapıyor, "
                  f"makine başına en yüksek değer ort. {f(data['saglam_kappa_tepe_ort'], 2)}, en büyüğü {f(data['saglam_kappa_tepe_maks'], 2)} (aşınma-hızı alarmı 1,8'de)")
    return 0


def cmd_hat_yaz(args):
    from . import scenarios
    scenarios.save_line(scenarios.example_line(args.hat), args.cikti)
    print(f"hat dosyası yazıldı: {args.cikti}")
    return 0


def cmd_dogrulama(args):
    from . import checks
    failed = 0
    for check in checks.run_all():
        print(f"[{'TAMAM' if check.ok else 'HATA '}] {check.name}\n        {check.detail}")
        failed += not check.ok
    print(f"\n{failed} başarısız doğrulama" if failed else "\ntüm doğrulamalar geçti")
    return 1 if failed else 0


# ----------------------------------------------------------------------------------------------
def build_parser():
    from .scenarios import MACHINE_SCENARIOS
    p = argparse.ArgumentParser(prog="fabrika", description="Akıllı fabrika dijital ikizi: makine ve üretim hattı")
    p.add_argument("--version", action="version", version=f"fabrika {__version__}")
    sub = p.add_subparsers(dest="komut", required=True)
    sub.add_parser("senaryolar", help="senaryoları listele").set_defaults(func=cmd_senaryolar)

    sp = sub.add_parser("makine", help="simüle makine + sensörler + ikiz (Kalman, CUSUM, kalan ömür)")
    sp.add_argument("--senaryo", default="saglikli", choices=list(MACHINE_SCENARIOS))
    sp.add_argument("--saat", type=float, default=1000.0)
    sp.add_argument("--tohum", type=int, default=0)
    sp.add_argument("--siddet", type=float, default=1.0, help="makinenin gerçek aşınma şiddeti (1 = nominal)")
    sp.add_argument("--q-kappa", type=float, default=0.03, help="ikizin aşınma-hızı çarpanındaki süreç gürültüsü (yüksek = hızlı uyum, gürültülü)")
    sp.add_argument("--grafik", metavar="PNG")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_makine)

    sp = sub.add_parser("simule", help="sensör akışını CSV olarak üret")
    sp.add_argument("--senaryo", default="saglikli", choices=list(MACHINE_SCENARIOS))
    sp.add_argument("--saat", type=float, default=800.0)
    sp.add_argument("--adim", type=float, default=0.25, help="örnekleme aralığı (saat)")
    sp.add_argument("--tohum", type=int, default=0)
    sp.add_argument("--siddet", type=float, default=1.0)
    sp.add_argument("--cikti", required=True, metavar="CSV")
    sp.set_defaults(func=cmd_simule)

    sp = sub.add_parser("izle", help="bir sensör CSV'sini ikize ver (kendi veriniz ya da simule çıktısı)")
    sp.add_argument("--veri", required=True, metavar="CSV")
    sp.add_argument("--parametre", metavar="JSON", help="MachineParams alanlarını değiştiren JSON")
    sp.add_argument("--sigma-sicaklik", type=float, default=0.3, help="sıcaklık sensörü gürültüsü (°C)")
    sp.add_argument("--sigma-titresim", type=float, default=0.15, help="titreşim sensörü gürültüsü (mm/s)")
    sp.add_argument("--arl0", type=float, default=10000.0, help="CUSUM yanlış alarm ortalama koşu uzunluğu (örnek)")
    sp.add_argument("--q-kappa", type=float, default=0.03, help="aşınma-hızı çarpanındaki süreç gürültüsü")
    sp.add_argument("--kalan-omur-saati", type=float, default=None, help="kalan ömrün hesaplanacağı zaman")
    sp.add_argument("--grafik", metavar="PNG")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_izle)

    def line_args(sp):
        sp.add_argument("--hat", default="montaj", choices=["montaj", "gercekci"])
        sp.add_argument("--hat-dosyasi", metavar="JSON", help="kendi hattınız (hat-yaz ile örnek üretilir)")

    sp = sub.add_parser("hat", help="üretim hattı: tesis simülasyonu, kayıttan kalibrasyon, ikiz tahmini")
    line_args(sp)
    sp.add_argument("--sure", type=float, default=20000.0, help="simülasyon süresi (dk)")
    sp.add_argument("--cekilis", type=int, default=100, help="tahmin aralığı için parametre çekilişi")
    sp.add_argument("--tohum", type=int, default=0)
    sp.add_argument("--grafik", metavar="PNG")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_hat)

    sp = sub.add_parser("ne-olur", help="ne olur analizi: istasyon hızlandırma ve tampon büyütme")
    line_args(sp)
    sp.add_argument("--hizlandir", type=float, default=1.1, help="her istasyonun denenen hız çarpanı")
    sp.add_argument("--tampon-arasi", type=int, default=None, help="incelenecek tampon (0 = ilk iki istasyon arası)")
    sp.add_argument("--tampon-boyutlari", default="0,1,2,4,8", help="virgülle ayrılmış kapasiteler")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_ne_olur)

    sp = sub.add_parser("rul-sinama", help="ani yatak arızasından sonra kalan ömür aralığının kapsama oranı (q_kappa ödünleşimi)")
    sp.add_argument("--q-kappa", type=float, default=0.03)
    sp.add_argument("--sonra", type=float, default=100.0, help="arızadan kaç saat sonra değerlendirilecek")
    sp.add_argument("--makine", type=int, default=40)
    sp.add_argument("--saglam-makine", type=int, default=10, help="κ gürültüsünün ölçüleceği sağlam makine sayısı (0 = atla)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_rul_sinama)

    sp = sub.add_parser("hat-yaz", help="örnek hattı JSON dosyası olarak yaz")
    sp.add_argument("--hat", default="montaj", choices=["montaj", "gercekci"])
    sp.add_argument("--cikti", required=True, metavar="JSON")
    sp.set_defaults(func=cmd_hat_yaz)

    sub.add_parser("dogrulama", help="model doğrulamalarını çalıştır").set_defaults(func=cmd_dogrulama)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (UsageError, ValueError, OSError) as exc:
        print(f"hata: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
