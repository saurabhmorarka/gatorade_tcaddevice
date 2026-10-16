"""Classical vs Schrodinger-Poisson on a 3D device: one figure from two
`python -m fab simulate` output folders (the same device, without and with
--quantum).

    python3 quantum/compare_3d.py out/qnew_classical out/qnew_all docs/img/quantum_3d_planar.png \
        --title "Planar NMOS, Lg 30 nm, tox 1.5 nm"
    python3 quantum/compare_3d.py out/finfet_classical out/finfet_quantum docs/img/quantum_3d_finfet.png \
        --section yz --profile z --at 25 -25 --title "FinFET"

Panels: Id-Vg (log, linear); the electron density along a line through
the channel at the solver's own nodes (--profile: y = depth below the top
interface, the planar default; z = across a fin; with --at the line's
position); the density on a section (xy: along the channel at mid-width,
the planar default; yz: the gate cross-section at the channel centre, where
a fin or the sheets show volume inversion), classical and quantum; and the
quantum potential Lambda_n on the same section.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLUE, ORANGE = "#2a78d6", "#eb6834"         # classical, quantum (the 1D figure's palette)


def _curves(folder):
    out = {}
    with open(os.path.join(folder, "ids_vgs.csv")) as f:
        for row in csv.DictReader(f):
            out.setdefault(float(row["Vds_V"]), []).append((float(row["Vgs_V"]), float(row["Id_A_per_um"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def _state(folder, vds, vgs):
    import pyvista as pv
    return pv.read(os.path.join(folder, "vtk", f"vds{vds:g}_vgs{vgs:+.3f}.vtu"))


def _sample(m, pts_nm, name):
    """Point field at points (nm); NaN outside the mesh / semiconductor."""
    import pyvista as pv
    s = pv.PolyData(np.asarray(pts_nm, float) * 1e-3).sample(m)
    v = np.asarray(s.point_data[name], float)
    v[~np.asarray(s.point_data["vtkValidPointMask"], bool)] = np.nan
    return v


def _line(m, axis, fixed, lo, hi):
    """(coordinate, n) at the solver's own nodes on the line along `axis`
    (0 x, 1 y, 2 z) nearest to the fixed coordinates of the other two axes
    (no interpolation: the octree's cells are 1-2 nm in places)."""
    P = np.asarray(m.points) * 1e3
    n_ = np.asarray(m.point_data["eDensity"], float)
    others = [a for a in range(3) if a != axis]
    ok = np.isfinite(n_) & (P[:, axis] >= lo - 1e-6) & (P[:, axis] <= hi + 1e-6)
    d = np.hypot(P[:, others[0]] - fixed[0], P[:, others[1]] - fixed[1])
    sel = ok & (d < d[ok].min() + 1e-6)
    o = np.argsort(P[sel, axis])
    return P[sel, axis][o], n_[sel][o]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("classical")
    ap.add_argument("quantum")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--section", choices=("xy", "yz"), default="xy")
    ap.add_argument("--profile", choices=("y", "z"), default="y")
    ap.add_argument("--at", type=float, nargs=2, default=None,
                    help="the profile line's other two coordinates (nm): (x, z) for --profile y, "
                         "(x, height) for z; default the domain middle and the top interface")
    ap.add_argument("--y0", type=float, default=0.0, help="the top gate interface depth (nm, y down)")
    ap.add_argument("--depth", type=float, default=6.0, help="the xy section's depth below y0 (nm)")
    ap.add_argument("--window", type=float, nargs=4, default=None,
                    help="yz section window: zmin zmax height_min height_max (nm)")
    ap.add_argument("--vgs", type=float, default=None, help="on-state Vgs (default: the curve's last)")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cc, cq = _curves(a.classical), _curves(a.quantum)
    vds_lo, vds_hi = min(cc), max(cc)
    vgs_on = a.vgs if a.vgs is not None else cc[vds_lo][-1, 0]
    mc, mq = _state(a.classical, vds_lo, vgs_on), _state(a.quantum, vds_lo, vgs_on)
    b = np.array(mc.bounds) * 1e3
    xm, zm = 0.5 * (b[0] + b[1]), 0.5 * (b[4] + b[5])

    plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})
    fig, ax = plt.subplots(2, 3, figsize=(16, 11), gridspec_kw=dict(hspace=0.35, wspace=0.3))
    for v, ls in ((vds_lo, "-"), (vds_hi, "--")):
        for c, col, lab in ((cc, BLUE, "classical"), (cq, ORANGE, "Schrodinger-Poisson")):
            ax[0, 0].semilogy(c[v][:, 0], c[v][:, 1] * 1e6, ls, color=col, lw=2, label=f"{lab}, Vds {v:g} V")
            ax[0, 1].plot(c[v][:, 0], c[v][:, 1] * 1e6, ls, color=col, lw=2, label=f"{lab}, Vds {v:g} V")
    ax[0, 0].set(xlabel="Vgs (V)", ylabel="Id (uA/um)", title="Id-Vg")
    ax[0, 1].set(xlabel="Vgs (V)", ylabel="Id (uA/um)", title="Id-Vg (linear)")
    ax[0, 0].legend(fontsize=8)
    for q in (ax[0, 0], ax[0, 1]):
        q.grid(alpha=0.3)

    # the profile at the solver's nodes, two gate biases
    if a.profile == "y":
        x0, z0 = a.at if a.at else (xm, zm)
        axis, fixed, lo, hi = 1, (x0, z0), a.y0, a.y0 + a.depth
        xlabel = "depth below the gate dielectric (nm)"
    else:
        x0, h0 = a.at if a.at else (xm, -a.y0)
        axis, fixed, lo, hi = 2, (x0, -h0), b[4], b[5]
        xlabel = "z across the body (nm)"
    for vg, alpha in ((vgs_on, 1.0), (round(0.5 * vgs_on, 3), 0.55)):
        try:
            sc, sq = _state(a.classical, vds_lo, vg), _state(a.quantum, vds_lo, vg)
        except Exception:
            continue
        for m, col, lab in ((sc, BLUE, "classical"), (sq, ORANGE, "Schrodinger-Poisson")):
            cc_, n_ = _line(m, axis, fixed, lo, hi)
            ax[0, 2].plot(cc_ - (a.y0 if axis == 1 else 0.0), n_, "o-", ms=3, color=col, lw=2, alpha=alpha,
                          label=f"{lab}, Vgs {vg:g} V")
    ax[0, 2].set(xlabel=xlabel, ylabel="n (cm$^{-3}$)",
                 title=f"n along the profile (solver nodes), Vds {vds_lo:g} V")
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(alpha=0.3)

    # sections
    if a.section == "xy":
        half = 0.5 * (b[1] - b[0])
        h_ = np.linspace(xm - 0.8 * half, xm + 0.8 * half, 321)
        v_ = a.y0 + np.linspace(0.0, a.depth, 121)
        H, V = np.meshgrid(h_, v_, indexing="ij")
        pts = np.column_stack([H.ravel(), V.ravel(), np.full(H.size, a.at[1] if a.at and a.profile == "y" else zm)])
        hl, vl, vplot = "x along the channel (nm)", "depth (nm)", v_ - a.y0
        sec = f"x-y at mid-width"
    else:
        z0_, z1_, hmin, hmax = a.window if a.window else (b[4], b[5], -b[3], -b[2])
        h_ = np.linspace(z0_, z1_, 241)
        v_ = np.linspace(hmin, hmax, 321)                 # heights
        H, V = np.meshgrid(h_, v_, indexing="ij")
        xs = a.at[0] if a.at else xm
        pts = np.column_stack([np.full(H.size, xs), -V.ravel(), H.ravel()])
        hl, vl, vplot = "z (nm)", "height (nm)", v_
        sec = f"gate cross-section y-z at x = {xs:g} nm"
    lo_, hi_ = 16, 20
    im = None
    for k, (m, lab) in enumerate(((mc, "classical"), (mq, "Schrodinger-Poisson"))):
        n_ = _sample(m, pts, "eDensity").reshape(H.shape)
        im = ax[1, k].pcolormesh(h_, vplot, np.log10(np.maximum(n_, 1.0)).T, cmap="viridis", vmin=lo_, vmax=hi_,
                                 shading="auto")
        ax[1, k].set(xlabel=hl, ylabel=vl, title=f"log10 n, {lab}\n{sec}\nVgs {vgs_on:g} V, Vds {vds_lo:g} V")
        ax[1, k].set_aspect("equal" if a.section == "yz" else "auto")
        if a.section == "xy":
            ax[1, k].invert_yaxis()
    fig.colorbar(im, ax=ax[1, :2], label="log10 n (cm$^{-3}$)", shrink=0.9)
    lam = _sample(mq, pts, "eQuantumPotential").reshape(H.shape)
    im = ax[1, 2].pcolormesh(h_, vplot, (lam * 1e3).T, cmap="magma", shading="auto")
    ax[1, 2].set(xlabel=hl, ylabel=vl, title="quantum potential $\\Lambda_n$ (meV): the confinement's\n"
                                             "rise of the electron band edge")
    ax[1, 2].set_aspect("equal" if a.section == "yz" else "auto")
    if a.section == "xy":
        ax[1, 2].invert_yaxis()
    fig.colorbar(im, ax=ax[1, 2], label="meV", shrink=0.9)
    fig.suptitle(a.title)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
