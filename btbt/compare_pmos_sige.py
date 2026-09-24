"""PMOS tunneling leakage, silicon vs strained Si0.6Ge0.4 source/drains (and
the NMOS of configs/input_mosfet_2d_btbt.yaml mirrored onto the PMOS bias
axes, as a check that the models treat the reversed polarity the same way).

Reads the CSVs written by btbt/main_btbt2d_sweep.py for
  configs/input_pmos_2d_btbt.yaml        (Si S/D)
  configs/input_pmos_2d_btbt_sige.yaml   (SiGe S/D)
  configs/input_mosfet_2d_btbt.yaml      (NMOS, optional)
and writes out/btbt/pmos_si_vs_sige.png (all components, with the NMOS) and
out/btbt/pmos_leakage_si_vs_sige.png (focused Si vs SiGe comparison).

Usage: python3 -m btbt.compare_pmos_sige
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "btbt")
CASES = [("input_pmos_2d_btbt", "PMOS, Si S/D", "#1f6feb", 1.0),
         ("input_pmos_2d_btbt_sige", "PMOS, strained Si0.6Ge0.4 S/D", "#d6336c", 1.0),
         ("input_mosfet_2d_btbt", "NMOS, Si S/D (mirrored: -Vg, -Vsub)", "#6c757d", -1.0)]
COMPS = [("kane_surface", "-", "Kane, surface (GIDL)"), ("tat_surface", "--", "trap-assisted, surface (GIDL)"),
         ("kane_bulk", "-.", "Kane, bulk junction"), ("tat_bulk", ":", "trap-assisted, bulk junction")]


def load(name, sweep):
    path = os.path.join(OUT, name, f"{sweep}_btbt.csv")
    if not os.path.exists(path):
        return None
    d = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row.pop("model")
            for k, v in row.items():
                d[m][k].append(float(v))
    return {m: {k: np.array(v) for k, v in c.items()} for m, c in d.items()}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for row, (sweep, xl) in enumerate((("idvg", "Vg (V)"), ("idvsub", "Vsub (V)"))):
        ax = axes[row, 0]
        for name, lbl, col, sgn in CASES:
            d = load(name, sweep)
            if d is None:
                continue
            ls = "-" if sgn > 0 else "--"
            if "nonlocal" in d:
                c = d["nonlocal"]
                ax.semilogy(sgn * c["bias_V"], np.abs(c["drain_A_per_um"]), ls, marker="o", ms=3, color=col,
                            label=f"{lbl}: nonlocal Kane+Hurkx")
            if "none" in d and sgn > 0:
                c = d["none"]
                ax.semilogy(sgn * c["bias_V"], np.abs(c["drain_A_per_um"]), ":", color=col, lw=1.2,
                            label=f"{lbl}: SRH only")
        ax.set_xlabel(xl); ax.set_ylabel("|Id| (A/um)")
        ax.set_title(("Id-Vg, |Vds| = 1 V, Vsub = 0" if sweep == "idvg" else "Id-Vsub, |Vds| = 1 V, Vg = 0"))
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
        for ax, (name, lbl, col, sgn) in zip(axes[row, 1:], CASES[:2]):
            d = load(name, sweep)
            if d is None or "nonlocal" not in d:
                ax.set_visible(False)
                continue
            c = d["nonlocal"]
            v = c["bias_V"]
            base = np.interp(v, d["none"]["bias_V"], d["none"]["drain_A_per_um"]) if "none" in d else 0.0
            ax.semilogy(v, np.abs(c["drain_A_per_um"] - base), "k-", lw=2.2, alpha=0.3, label="|Id - Id(SRH only)|")
            for k, ls, cl in COMPS:
                y = c[k + "_A_per_um"]
                ax.semilogy(v, np.where(y > 0, y, np.nan), ls, color=col, label=cl)
                if np.any(y < 0):
                    ax.semilogy(v, np.where(y < 0, -y, np.nan), ls, color=col, marker="x", ms=4, lw=0.6,
                                label=cl + " (negative)")
            ax.set_xlabel(xl); ax.set_ylabel("q x integrated generation, drain side (A/um)")
            ax.set_title(f"{lbl}: nonlocal components")
            lo = np.nanmin(np.abs(c["drain_A_per_um"])) * 1e-3
            ax.set_ylim(max(lo, 1e-22), None)
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
    fig.suptitle("PMOS band-to-band / trap-assisted tunneling leakage: Si vs strained SiGe source/drains", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT, "pmos_si_vs_sige.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    # summary numbers
    for sweep, pts in (("idvg", (0.0, 1.0, 2.0)), ("idvsub", (0.0, 1.0, 2.0, 3.0))):
        for name, lbl, _, sgn in CASES:
            d = load(name, sweep)
            if d is None or "nonlocal" not in d:
                continue
            c = d["nonlocal"]
            vals = [np.abs(c["drain_A_per_um"])[np.argmin(np.abs(c["bias_V"] - sgn * p))] for p in pts]
            print(f"  {sweep:6s} {lbl:40s} " + "  ".join(f"|{'Vg' if sweep == 'idvg' else 'Vsub'}|={p:g}: {v:.2e}"
                                                        for p, v in zip(pts, vals)))


def plot_leakage_comparison(path=os.path.join(OUT, "pmos_leakage_si_vs_sige.png")):
    """Focused Si vs SiGe S/D figure: |Id| for both sweeps, the SiGe/Si
    ratio, and the leakage split by mechanism at a few biases."""
    si, ge = CASES[0], CASES[1]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    for row, (sweep, xl, marks) in enumerate((("idvg", "Vg (V)", (0.0, 1.0, 1.6, 2.0)),
                                              ("idvsub", "Vsub (V)", (0.0, 1.0, 2.0, 3.0)))):
        d_si, d_ge = load(si[0], sweep), load(ge[0], sweep)
        ax = axes[row, 0]
        for d, (_, lbl, col, _) in ((d_si, si), (d_ge, ge)):
            ax.semilogy(d["nonlocal"]["bias_V"], np.abs(d["nonlocal"]["drain_A_per_um"]), "o-", ms=3.5, color=col,
                        label=f"{lbl}: with tunneling")
            ax.semilogy(d["none"]["bias_V"], np.abs(d["none"]["drain_A_per_um"]), ":", color=col, lw=1.3,
                        label=f"{lbl}: SRH only")
        ax.set_xlabel(xl); ax.set_ylabel("|Id| (A/um)")
        ax.set_title("Id-Vg, Vds = -1 V, Vsub = 0" if sweep == "idvg" else "Id-Vsub, Vds = -1 V, Vg = 0")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        if sweep == "idvg":
            ax.axvspan(1.2, 2.05, color="#d6336c", alpha=0.06)
            ax.text(1.62, ax.get_ylim()[1] / 30, "GIDL", ha="center", color="#d6336c")

        ax = axes[row, 1]
        v = d_si["nonlocal"]["bias_V"]
        ratio = np.abs(d_ge["nonlocal"]["drain_A_per_um"]) / np.abs(d_si["nonlocal"]["drain_A_per_um"])
        ax.semilogy(v, ratio, "o-", ms=3.5, color="#6f42c1")
        for m in marks:
            k = np.argmin(np.abs(v - m))
            ax.annotate(f"{ratio[k]:.0f}x", (v[k], ratio[k]), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xlabel(xl); ax.set_ylabel("|Id| SiGe S/D  /  |Id| Si S/D")
        ax.set_title("leakage increase from SiGe source/drains")
        ax.grid(alpha=0.3, which="both")

        ax = axes[row, 2]
        comps = [("kane_surface", "direct BTBT, gate edge (GIDL)", "#1f6feb"),
                 ("tat_surface", "trap-assisted, gate edge (GIDL)", "#74a9f7"),
                 ("kane_bulk", "direct BTBT, drain-body junction", "#2f9e44"),
                 ("tat_bulk", "trap-assisted, drain-body junction", "#8fd19e")]
        w = 0.1
        xs = np.arange(len(marks))
        for j, (k, lbl, col) in enumerate(comps):
            for off, d, hatch in ((-0.22, d_si, ""), (0.22, d_ge, "//")):
                c = d["nonlocal"]
                vals = [c[k + "_A_per_um"][np.argmin(np.abs(c["bias_V"] - m))] for m in marks]
                ax.bar(xs + off + (j - 1.5) * w, np.maximum(vals, 1e-30), w, color=col, hatch=hatch,
                       edgecolor="k", lw=0.4, label=lbl if off < 0 else None)
        ax.set_yscale("log")
        lo = 1e-16
        ax.set_ylim(lo, None)
        ax.set_xticks(xs, [f"{xl.split()[0]} = {m:g} V\nSi   |   SiGe" for m in marks], fontsize=8.5)
        ax.set_ylabel("q x integrated generation, drain side (A/um)")
        ax.set_title("which mechanism: plain = Si S/D, hatched = SiGe S/D")
        ax.grid(alpha=0.3, which="both", axis="y"); ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("PMOS leakage: silicon vs strained Si0.6Ge0.4 source/drains (nonlocal Kane + Hurkx tunneling)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
    plot_leakage_comparison()
