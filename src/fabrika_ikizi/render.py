"""Plots (matplotlib, Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_machine(run, y_temp, y_vib, params, path, truth=None, title="Makine ikizi", dpi=110):
    """Dashboard of a twin run: sensors against estimates, wear with its band, innovations, remaining life.

    ``truth`` (a simulated plant) adds the true curves; without it only what the twin knows is drawn.
    """
    t = run.t
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11), sharex=True)

    ax = axes[0]
    ax.plot(t, y_temp, ".", ms=1.5, color="0.6", label="sensör")
    if truth is not None:
        ax.plot(t, truth.temp, color="tab:blue", lw=0.8, label="gerçek")
    ax.plot(t, run.temp, color="crimson", lw=0.8, label="ikiz tahmini")
    ax.set_ylabel("sıcaklık (°C)")
    ax.legend(ncol=3, loc="upper left")

    ax = axes[1]
    sd = run.wear_std
    ax.fill_between(t, run.wear - 2 * sd, run.wear + 2 * sd, color="crimson", alpha=0.25, label="tahmin ±2σ")
    ax.plot(t, run.wear, color="crimson", lw=0.9, label="ikiz tahmini")
    if truth is not None:
        ax.plot(t, truth.wear, color="tab:blue", lw=0.9, label="gerçek aşınma")
    ax.plot(t, (np.asarray(y_vib) - params.v0) / params.kv, ".", ms=1.5, color="0.6", label="titreşimden okunan")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_ylabel("aşınma w (1 = arıza)")
    ax.legend(ncol=4, loc="upper left")

    ax = axes[2]
    ax.plot(t, run.z[:, 0], ".", ms=2, color="tab:blue", label="sıcaklık")
    ax.plot(t, run.z[:, 1], ".", ms=2, color="tab:orange", label="titreşim")
    ax.axhline(0, color="k", lw=0.5)
    top = np.nanmax(np.abs(run.z)) if np.isfinite(run.z).any() else 4.0
    ax.set_ylim(-top * 1.1, top * 1.1)
    for ta, kind in run.alarms:
        ax.axvline(ta, color="crimson", lw=0.9, alpha=0.8)
        ax.text(ta, top * 1.05, kind, color="crimson", fontsize=7, rotation=90, va="top")
    ax.set_ylabel("standart yenilik z")
    ax.legend(ncol=2, loc="upper left")

    ax = axes[3]
    step = max(1, len(t) // 60)
    idx = np.arange(min(max(step, 20), len(t) - 1), len(t), step)
    meds, los, his = zip(*[run.rul(i, size=1500) for i in idx])
    ax.fill_between(t[idx], los, his, color="crimson", alpha=0.25, label="kalan ömür, %80 aralık")
    ax.plot(t[idx], meds, color="crimson", label="ikiz medyanı")
    if truth is not None and truth.fail_index >= 0:
        ax.plot(t[idx], truth.fail_time - t[idx], color="tab:blue", label="gerçek kalan ömür")
        ax.set_title("gerçek kalan ömür, arızadan önce arıza anını da içerir: ikiz bunu önceden bilemez", fontsize=8)
    ax.set_ylabel("kalan ömür (saat)")
    ax.set_xlabel("zaman (saat)")
    ax.set_ylim(bottom=0)
    if run.active is not None and run.active.any():
        ax.fill_between(t, 0, 1, where=run.active, transform=ax.get_xaxis_transform(), color="0.5", alpha=0.25,
                        label="sensör-model tutarsız: kalan ömür güvenilir değil")
    ax.legend(ncol=2, loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_line(result, path, dpi=110):
    """State shares per station, mean buffer levels and the twin's throughput against the plant's."""
    s = result.summary
    stations = s["istasyonlar"]
    names = [x["ad"] for x in stations]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axes[0]
    bottom = np.zeros(len(names))
    for key, label, colour in (("mesgul", "çalışıyor", "tab:green"), ("bloke", "bloke", "tab:orange"),
                               ("aclik", "malzeme bekliyor", "tab:blue"), ("ariza", "arızalı", "tab:red")):
        vals = np.array([x[key] for x in stations])
        ax.bar(names, vals, bottom=bottom, color=colour, label=label)
        bottom += vals
    ax.set_ylabel("sürenin payı")
    ax.set_title("istasyon durumları")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)

    ax = axes[1]
    ax.bar(range(len(s["tampon_ort"])), s["tampon_ort"], color="tab:purple")
    ax.set_xticks(range(len(s["tampon_ort"])))
    ax.set_xticklabels([f"{names[i]}→{names[i + 1]}" for i in range(len(s["tampon_ort"]))], fontsize=8)
    ax.set_ylabel("ortalama tampon doluluğu")
    ax.set_title("tamponlar")

    ax = axes[2]
    ax.errorbar([0], [s["tahmin"]], yerr=[[s["tahmin"] - s["tahmin_alt"]], [s["tahmin_ust"] - s["tahmin"]]],
                fmt="o", color="crimson", capsize=6, label="ikiz tahmini (%90 aralık)")
    ax.plot([0], [s["gozlenen_uretim"]], "s", color="tab:blue", label="tesiste gözlenen")
    ax.axhline(s["kapasite_siniri"], color="k", ls=":", label="darboğaz kapasite sınırı")
    ax.set_xticks([])
    ax.set_ylabel("üretim (parça/dk)")
    ax.set_title("ikiz ↔ tesis")
    ax.set_ylim(min(s["tahmin_alt"], s["gozlenen_uretim"]) - 0.03, s["kapasite_siniri"] + 0.02)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
