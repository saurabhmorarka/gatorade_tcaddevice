"""Universal-mobility check of the inversion-layer mobility (plan 1.6, 5.7;
docs/mobility.md): a 1D MOS capacitor (core/mesh.build_mos_grid, classical
or Schrodinger-Poisson via quantum/mos1d.py) solved over a gate sweep, the
SAME solver2d/surface_mobility.SurfaceMobility evaluated on it (a 1D mesh
stand-in), and

    mu_eff = sum(mu n V) / sum(n V)          (split-CV mobility at Vds -> 0)
    E_eff  = (Q_dep + eta Q_inv) / eps_si    (eta 1/2 electrons, 1/3 holes)

against the universal curves (Takagi TED 41, 2357; fits of Chen/Hu, C. Hu
"Modern Semiconductor Devices for ICs" eqs. 6.3.7-8):

    mu_n = 540 / (1 + (E_eff / 0.90 MV/cm)^1.85)
    mu_p = 185 / (1 + E_eff / 0.563 MV/cm)

  python3 scripts/mobility/universal.py [--quantum] [--calibrate] [--out docs/img]
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from core.config import _resolve_material_block           # noqa: E402
from core.mesh import build_mos_grid                        # noqa: E402
from core.params import Q                                   # noqa: E402
from mos.mos_params import MOSDevice                        # noqa: E402
from quantum.mos1d import QuantumConfig, node_volumes, solve_mos   # noqa: E402
from solver2d.surface_mobility import MobilityModel, SurfaceMobility   # noqa: E402

# device frame rows (x = the 1D normal) in crystal axes
FRAME_100 = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
FRAME_110 = [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, 0.0]]
ETA = {"n": 0.5, "p": 1.0 / 3.0}


def universal(E_MVcm, carrier):
    E = np.asarray(E_MVcm, float)
    if carrier == "n":
        return 540.0 / (1.0 + (E / 0.90) ** 1.85)
    return 185.0 / (1.0 + E / 0.563)


def mesh_1d(g, mat):
    x = g["x"]
    N = len(x)
    ox = g["is_oxide"]
    h = np.diff(x)
    edges = np.column_stack([np.arange(N - 1), np.arange(1, N)])
    ones = np.ones(N - 1)
    return SimpleNamespace(
        points=x[:, None], edges=edges, facet_length=ones, facet_length_semi=(~ox[:-1] & ~ox[1:]).astype(float),
        edge_g=np.asarray(g["eps_edge"]) / h, edge_weight=1.0 / h, cv_area=node_volumes(x),
        Cdop=np.asarray(g["Cdop"], float), is_insulator=ox, boundary_point_index=np.array([0, N - 1]),
        boundary_bc_type=["contact:gate", "contact:body"], node_materials=None, node_material_index=None)


def curve(N_sub, carrier, mode="fast", quantum=False, tox=2e-7, frame=FRAME_100, VG=None, model_kw=None,
          interface_factor=0.001):
    """[(E_eff MV/cm, mu_eff, Ninv cm^-2)] over a gate sweep."""
    mat = _resolve_material_block({})
    Cs = -N_sub if carrier == "n" else N_sub
    dev = MOSDevice(Na=N_sub, t_ox=tox)
    g = build_mos_grid(mat, dev, Cs, interface_spacing_debye_factor=interface_factor)
    m = mesh_1d(g, mat)
    sm = SurfaceMobility(m, mat, MobilityModel(mode=mode, frame=frame, **(model_kw or {})))
    s = 1.0 if carrier == "n" else -1.0
    VG = np.linspace(0.3, 3.0, 28) if VG is None else VG
    V = node_volumes(g["x"])
    semi = ~g["is_oxide"]
    oi = g["oxide_index"]
    qc = QuantumConfig() if quantum else None
    out = []
    for vg in VG:
        st = solve_mos(g, mat, dev, Cs, s * vg, qc)
        n, p, psi = st["n"], st["p"], st["psi"]
        mu_n, mu_p, F = sm.nodal(mat, psi, n, p)
        c, mu = (n, mu_n) if carrier == "n" else (p, mu_p)
        w = (c * V)[semi]
        Ninv = w.sum()
        if Ninv < 1e11:
            continue
        mu_eff = float((mu[semi] * w).sum() / Ninv)
        Qs = mat.eps * F[oi] / Q                     # |surface charge| / q from the surface field
        Qdep = max(Qs - Ninv, 0.0)
        E = Q * (Qdep + ETA[carrier] * Ninv) / mat.eps * 1e-6
        out.append((E, mu_eff, Ninv))
    return np.array(out)


def agreement(c, carrier, lo=0.3, hi=1.2):
    """max |mu_eff / universal - 1| over lo <= E_eff <= hi."""
    sel = (c[:, 0] >= lo) & (c[:, 0] <= hi)
    if not sel.any():
        return np.nan
    return float(np.max(np.abs(c[sel, 1] / universal(c[sel, 0], carrier) - 1.0)))


def ratio_at(carrier, k, E0=0.5, mode="fast", quantum=False):
    kw = {f"scale_110_{carrier}": k}
    a = curve(1e16, carrier, mode, quantum, frame=FRAME_110, model_kw=kw)
    b = curve(1e16, carrier, mode, quantum, frame=FRAME_100)
    return float(np.interp(E0, a[:, 0], a[:, 1]) / np.interp(E0, b[:, 0], b[:, 1]))


def calibrate(carrier="p", target=2.0, mode="fast"):
    lo, hi = 1.0, 6.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ratio_at(carrier, mid, mode=mode) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantum", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.calibrate:
        for mode in ("fast", "exact"):
            k = calibrate("p", 2.0, mode)
            print(f"(110) hole surface scale for mu_eff ratio 2.0 at 0.5 MV/cm ({mode}): {k:.3f}")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dopings = (1e16, 1e17, 5e17, 1e18)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for col, mode in enumerate(("fast", "exact")):
        for row, carrier in enumerate(("n", "p")):
            ax = axs[row, col]
            Eu = np.linspace(0.05, 2.0, 100)
            ax.loglog(Eu, universal(Eu, carrier), "k-", lw=2, label="universal (Takagi; Chen/Hu fit)")
            for N in dopings:
                for q in ((False, True) if a.quantum else (False,)):
                    c = curve(N, carrier, mode, q)
                    err = agreement(c, carrier)
                    lab = f"N={N:.0e}{' QM' if q else ''}: max dev {100 * err:.0f}% (0.3-1.2)"
                    ax.loglog(c[:, 0], c[:, 1], "--" if q else "o-", ms=3, label=lab)
                    print(f"{mode} {carrier} N={N:.0e} {'quantum' if q else 'classical'}: "
                          + " ".join(f"{e:.2f}:{m:.0f}" for e, m, _ in c[::3]) + f"  max dev {100 * err:.1f}%")
            ax.set_xlim(0.05, 2.0)
            ax.set_ylim(20, 1000)
            ax.set_xlabel("E_eff (MV/cm)")
            ax.set_ylabel("mu_eff (cm^2/Vs)")
            ax.set_title(f"{'electrons' if carrier == 'n' else 'holes'}, (100), {mode} mode")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
    out = a.out or "out/mobility"
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "universal_mobility.png")
    fig.savefig(path, dpi=110)
    print("->", path)


if __name__ == "__main__":
    main()
