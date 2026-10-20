"""2D long-channel check of the inversion-layer mobility (plan 5.7 step 2;
docs/mobility.md): the 2D NMOS of configs/input_mosfet_2d.yaml (Lg 0.5 um,
tox 5 nm, Na 1e16; 2.5 nm interface spacing) at Vds = 50 mV,

    mu_eff = sum(mu n V) / sum(n V)     over the mid-channel column (the middle half of the gate, top 30 nm)

against the 1D MOS harness (scripts/mobility/universal.py, the same model on
a 0.04 nm grid) at the same N_inv. Validates the FV normal field (the
D-continuity interface value on a coarse triangle mesh) end to end.

  python3 scripts/mobility/long_channel_2d.py [--out out/mobility]
"""
import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from core.params import Q                      # noqa: E402


def run_2d(mode, vgs):
    import main2d_mosfet_sweep as m
    from solver2d.current import contact_current
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh, mat, dev, Cs, eq, vsat = m._context(m.DEFAULT_PATH, None, None, mode)
    gate = next(c for c in mesh.domain.contacts if c.name == "gate")
    x0, x1 = gate.x_range_cm
    L = x1 - x0
    P = mesh.points
    # the middle half of the channel, top 30 nm: inversion electrons only
    xm0, xm1 = x0 + 0.25 * L, x1 - 0.25 * L
    col = (P[:, 0] > xm0) & (P[:, 0] < xm1) & (P[:, 1] >= 0.0) & (P[:, 1] < 30e-7) & ~mesh.is_insulator
    Vds = 0.05

    def at(p_, g):
        return m.solve_mosfet(mesh, mat, dev, Cs, p_, Vds, psi_init=g["psi"], phin_init=g["phin"],
                              phip_init=g["phip"], cold_retry=False, velocity_saturation=vsat)
    ramp, _, _ = m.continuation(lambda p_, g: m.solve_mosfet(mesh, mat, dev, Cs, 0.0, p_, psi_init=g["psi"],
                                                             phin_init=g["phin"], phip_init=g["phip"],
                                                             cold_retry=False, velocity_saturation=vsat),
                                0.0, eq, [Vds])
    states, its, _ = m.continuation(at, 0.0, ramp[Vds], list(vgs))
    out = []
    for v in vgs:
        r = states.get(v)
        if r is None:
            continue
        Id = contact_current(mesh, mat, r, "drain")                 # A/cm
        w = (r["n"] * mesh.cv_area_semi)[col]
        Ninv = float(w.sum() / (xm1 - xm0))
        sm_ = getattr(mesh, "surface_mobility", None)
        if sm_ is not None:
            mu_n = sm_.nodal(mat, r["psi"], r["n"], r["p"])[0]
        else:
            from solver2d.newton_solver_qf_2d import _mesh_mobility_nodal
            mu_n = _mesh_mobility_nodal(mesh, mat)[0]
        out.append((v, Ninv, float((mu_n[col] * w).sum() / w.sum()), Id))
    return np.array(out), its


def main():
    from universal import curve
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/mobility")
    a = ap.parse_args()
    vgs = [round(v, 3) for v in np.arange(0.6, 1.51, 0.1)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    ref = None
    for mode, col in (("off", "k"), ("fast", "C0"), ("exact", "C3")):
        r, its = run_2d(mode, vgs)
        if ref is None:
            ref = r
        ax.semilogx(r[:, 1], r[:, 2], "o-", color=col, label=f"2D mid-channel column, {mode} ({its} Newton its)")
        line = f"{mode} ({its} its): Vgs:Ninv:mu_col " + " ".join(f"{v:.1f}:{n:.1e}:{mu:.0f}" for v, n, mu, _ in r)
        if mode != "off":
            c = curve(1e16, "n", mode, False, tox=5e-7, VG=np.linspace(0.3, 2.0, 35))
            ax.semilogx(c[:, 2], c[:, 1], "--", color=col, label=f"1D MOS (0.04 nm grid), {mode}")
            mu1 = np.interp(np.log(r[:, 1]), np.log(c[:, 2]), c[:, 1])
            dev_ = r[:, 2] / mu1 - 1
            sel = r[:, 1] > 1e12
            line += f"\n   column vs 1D at the same Ninv: {' '.join(f'{100 * d:+.1f}%' for d in dev_[sel])}; " \
                    f"max |dev| (Ninv > 1e12) {100 * np.max(np.abs(dev_[sel])):.1f}%"
            line += f"\n   Id / Id(off): {' '.join(f'{a / b:.3f}' for a, b in zip(r[:, 3], ref[:, 3]))}"
        print(line)
    ax.set_xlabel("N_inv, mid-channel (cm^-2)")
    ax.set_ylabel("mu_eff (cm^2/Vs)")
    ax.set_title("2D NMOS Lg 0.5 um, Vds 50 mV: split-CV mu_eff, 2D column vs 1D MOS", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "long_channel_2d.png"), dpi=110)


if __name__ == "__main__":
    main()
