"""P1 validation of the Schrodinger-Poisson channel charge in 1D
(plans/quantum_plan.md 5): bulk MOS capacitors and the double gate.

  python3 -m quantum.validate_1d [out.png]

Prints the summary table (threshold shift, Cinv, centroid, double-gate
volume inversion) and writes a 2x2 figure (default docs/img/quantum_sp_1d.png):
  (a) electron density vs depth, classical vs quantum, bulk 1e18
  (b) low-frequency C-V, classical vs quantum, bulk 1e17 and 1e18
  (c) inversion-charge centroid vs sheet density
  (d) double gate: density across the body at high gate bias, 3-20 nm
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.mesh import build_mos_grid
from core.params import Material
from mos.mos_params import MOSDevice
from quantum.mos1d import QuantumConfig, double_gate_grid, solve_dg, sweep_mos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots 1-4
INK, MUTED, GRID = "#222222", "#6b6b6b", "#dddddd"
TOX = 1.5e-7


def vg_at(res, N):
    Ni = np.array([r["Ninv"] for r in res])
    return float(np.interp(np.log(N), np.log(Ni), [r["VG"] for r in res]))


def bulk(mat, Na, VG):
    dev = MOSDevice(Na=Na, t_ox=TOX)
    g = build_mos_grid(mat, dev, -Na)
    return g, dev, sweep_mos(g, mat, dev, -Na, VG), sweep_mos(g, mat, dev, -Na, VG, QuantumConfig())


def dg_sweep(mat, g, VG, q):
    out, psi, lam = [], None, None
    for v in VG:
        r = solve_dg(g, mat, v, q, psi_init=psi, lam_init=lam)
        psi, lam = r["psi"], r["lam"]
        out.append(r)
    return out


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main(path=None):
    path = path or os.path.join(ROOT, "docs", "img", "quantum_sp_1d.png")
    mat = Material()
    VG = np.round(np.arange(0.2, 2.401, 0.05), 3)
    runs = {Na: bulk(mat, Na, VG) for Na in (1e17, 1e18)}

    print(f"Bulk MOS-cap, tox = {TOX * 1e7:.1f} nm, metal gate (V_FB = 0), (100) Si, Boltzmann subbands")
    print(f"{'Na':>8} {'dVt@1e11':>9} {'dVg@1e12':>9} {'Cinv cl':>8} {'Cinv qm':>8} {'dEOT':>6} {'zc cl':>6} {'zc qm':>6}   (at Ninv = 1e13)")
    for Na, (g, dev, cl, qm) in runs.items():
        Cox = dev.eps_ox / dev.t_ox
        v_cl, v_qm = vg_at(cl, 1e13), vg_at(qm, 1e13)
        C = lambda res, v: float(np.interp(v, VG, [r["C_lf"] for r in res]))
        zc = lambda res, v: float(np.interp(v, VG, [r["centroid"] for r in res])) * 1e7
        c_cl, c_qm = C(cl, v_cl), C(qm, v_qm)
        deot = (dev.eps_ox / c_qm - dev.eps_ox / c_cl) * 1e7
        print(f"{Na:8.0e} {1e3 * (vg_at(qm, 1e11) - vg_at(cl, 1e11)):7.1f}mV {1e3 * (vg_at(qm, 1e12) - vg_at(cl, 1e12)):7.1f}mV "
              f"{c_cl / Cox:8.3f} {c_qm / Cox:8.3f} {deot:5.2f}nm {zc(cl, v_cl):5.2f}  {zc(qm, v_qm):5.2f}  (C/Cox; nm)")

    VGd = np.round(np.arange(-0.2, 1.201, 0.05), 3)
    thick = (3e-7, 5e-7, 10e-7, 20e-7)
    dg = {}
    print("\nSymmetric double gate, tox = 1.5 nm, midgap gates, N_body = 1e15")
    print(f"{'t_body':>7} {'dVt@1e11':>9} {'peak (cl)':>10} {'peak (qm)':>10} {'n_mid/n_max':>12}   (at VG = 1.2 V; peak: distance from the nearer wall)")
    for t in thick:
        g = double_gate_grid(mat, t, TOX)
        cl, qm = dg_sweep(mat, g, VGd, None), dg_sweep(mat, g, VGd, QuantumConfig())
        dg[t] = (g, cl, qm)
        x, s = g["x"], g["is_semi"]
        n = qm[-1]["n"]
        pk = lambda r: min(x[s][np.argmax(r["n"][s])], t - x[s][np.argmax(r["n"][s])]) * 1e7   # from the nearer wall
        print(f"{t * 1e7:5.0f}nm {1e3 * (vg_at(qm, 1e11) - vg_at(cl, 1e11)):7.1f}mV {pk(cl[-1]):8.2f}nm {pk(qm[-1]):8.2f}nm "
              f"{np.interp(t / 2, x, n) / n[s].max():12.2f}")

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
    # (a) density profiles, bulk 1e18
    ax = axs[0, 0]
    g, dev, cl, qm = runs[1e18]
    x = g["x"]
    s = x >= 0
    for N, a in ((1e12, 0.55), (1e13, 1.0)):
        for res, c, ls, lab in ((cl, ORANGE, "--", "classical"), (qm, BLUE, "-", "Schrodinger-Poisson")):
            k = int(np.argmin(np.abs(np.log([r["Ninv"] for r in res]) - np.log(N))))
            ax.plot(x[s] * 1e7, res[k]["n"][s], color=c, ls=ls, lw=2, alpha=a,
                    label=lab if N == 1e13 else None)
    ax.set_xlim(0, 6)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "depth below the Si/SiO2 interface (nm)", "electron density (cm$^{-3}$)",
           "(a) Bulk 1e18: the charge peak moves off the wall\n(light: N_inv ~ 1e12 cm$^{-2}$, dark: ~ 1e13)")

    # (b) C-V
    ax = axs[0, 1]
    for Na, a in ((1e17, 0.55), (1e18, 1.0)):
        g, dev, cl, qm = runs[Na]
        Cox = dev.eps_ox / dev.t_ox
        ax.plot(VG, [r["C_lf"] / Cox for r in cl], color=ORANGE, ls="--", lw=2, alpha=a,
                label="classical" if Na == 1e18 else None)
        ax.plot(VG, [r["C_lf"] / Cox for r in qm], color=BLUE, lw=2, alpha=a,
                label="Schrodinger-Poisson" if Na == 1e18 else None)
    ax.set_xlim(VG[0], VG[-1])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, "gate voltage (V)", "C$_{LF}$ / C$_{ox}$",
           "(b) Low-frequency C-V: higher Vt, lower Cinv\n(light: Na = 1e17, dark: 1e18 cm$^{-3}$)")

    # (c) centroid vs Ninv
    ax = axs[1, 0]
    for Na, a in ((1e17, 0.55), (1e18, 1.0)):
        g, dev, cl, qm = runs[Na]
        for res, c, ls, lab in ((cl, ORANGE, "--", "classical"), (qm, BLUE, "-", "Schrodinger-Poisson")):
            N = np.array([r["Ninv"] for r in res])
            m = (N > 1e11) & (N < 3e13)
            ax.semilogx(N[m], [r["centroid"] * 1e7 for r, k in zip(res, m) if k], color=c, ls=ls, lw=2, alpha=a,
                        label=lab if Na == 1e18 else None)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(bottom=0)
    _style(ax, "inversion sheet density (cm$^{-2}$)", "charge centroid (nm)",
           "(c) Centroid of the inversion charge\n(light: Na = 1e17, dark: 1e18 cm$^{-3}$)")

    # (d) double gate
    ax = axs[1, 1]
    for t, c in zip(thick, (BLUE, ORANGE, AQUA, YELLOW)):
        g, cl, qm = dg[t]
        x, s = g["x"], g["is_semi"]
        n = qm[-1]["n"][s]
        ax.plot(x[s] / t, n / n.max(), color=c, lw=2, label=f"{t * 1e7:.0f} nm")
    g, cl, qm = dg[20e-7]
    x, s = g["x"], g["is_semi"]
    ax.plot(x[s] / 20e-7, cl[-1]["n"][s] / cl[-1]["n"][s].max(), color=MUTED, ls=":", lw=1.5,
            label="classical, 20 nm")
    ax.legend(frameon=False, fontsize=9, title="body thickness", title_fontsize=9, loc="upper center",
              ncol=3)
    ax.set_ylim(0, 1.35)
    _style(ax, "position across the body / thickness", "n / max(n)",
           "(d) Double gate at VG = 1.2 V: volume inversion\nin thin bodies, two interface peaks in thick ones")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
