"""1D validation of the nonlocal BTBT model (btbt/kernel.py, btbt/paths1d.py)
on the drain-substrate diode (configs/input_diode_drain_substrate.yaml,
p 1e17 / n+ 1e20), before the same kernel is used on the 2D MOSFET.

Four solves of the same device, same sweep, all with plain SRH (Et=Ei):
  srh           - no band-to-band tunneling (baseline)
  local_capped  - the 1D local Kane model exactly as shipped (tat/, F_sat=9e5 V/cm)
  local         - the same local Kane, uncapped
  nonlocal      - nonlocal path Kane (same A, B, P), lagged outer iteration

Checks printed:
  * uniform-field limit: for a linear psi, every path has F_eff = F, so the
    nonlocal rate equals the local one to round-off;
  * the nonlocal model's outer iteration converges (|dJ/J| per pass);
  * the nonlocal model carries no current at Va=0 (the local one does);
  * terminal current is taken at BOTH contacts - with nonlocal generation
    the electron/hole current is not constant in between (the tunneling
    electron carries the difference), but pairs are conserved, so the two
    ends must agree.

Usage: python3 -m btbt.main_btbt_1d [config]
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import config as cfg
from core import physics as ph
from core.mesh import build_diode_grid
from core.solver import solve_equilibrium
from tat.newton_solver_tat import newton_gummel_solve, _node_field
from tat.tat import KaneBTBTModel, btbt_generation
from btbt.kernel import path_rate
from btbt.paths1d import nonlocal_generation_1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "btbt", "diode_1d")
OUTER_TOL = 1e-4
OUTER_MAX = 20


def srh_trap_fn(n, p, F_abs, mat, model):
    """Plain SRH in the (n, p, F, mat, model) -> (G, dG_dn, dG_dp, dG_dF)
    trap-generation signature tat/newton_solver_tat.py takes - lets that
    solver run Kane BTBT (or nothing) on top of ordinary SRH, no Hurkx."""
    ni, tn, tp = mat.ni, mat.tau_n, mat.tau_p
    den = tp * (n + ni) + tn * (p + ni)
    num = n * p - ni ** 2
    R = num / den
    return -R, -(p * den - num * tp) / den ** 2, -(n * den - num * tn) / den ** 2, np.zeros_like(R)


NO_KANE = KaneBTBTModel(A=0.0, B=21.6e6, P=2.0)


def uniform_field_check(model, Eg):
    x = np.linspace(0, 1e-5, 2001)
    worst = 0.0
    for F in (3e5, 6e5, 1e6, 2e6):
        psi = F * x
        from btbt.paths1d import find_paths_1d
        l, _, _, _ = find_paths_1d(x, psi, Eg)
        ok = np.isfinite(l)
        G_nl, F_eff = path_rate(l[ok], Eg, model)
        G_loc, _ = btbt_generation(np.full(ok.sum(), F), model)
        worst = max(worst, np.max(np.abs(G_nl / G_loc - 1)))
    return worst


def terminal_J(r):
    return r["Jtot"][0], r["Jtot"][-1]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(ROOT, "configs", "input_diode_drain_substrate.yaml"))
    ap.add_argument("--Na", type=float, default=None, help="override the p-side (substrate) doping, cm^-3")
    args = ap.parse_args()
    c = cfg.load_config(args.config)
    if args.Na:
        c["doping"]["p_side"]["concentration_cm3"] = args.Na
    out = os.path.join(OUT, f"Na{c['doping']['p_side']['concentration_cm3']:.0e}".replace("+", ""))
    os.makedirs(out, exist_ok=True)
    mat, dev, _, _, _, mesh_opts, _ = cfg.build_from_config(c)
    g = build_diode_grid(mat, dev, **mesh_opts)
    x, Cdop, mf = g["x"], g["Cdop"], g["mat_field"]
    cvol = ph._control_volumes(x)
    Eg, Vt = mat.Eg_eV, mat.Vt
    kane = KaneBTBTModel.si_kane_quadratic()
    kane_uncapped = KaneBTBTModel(A=kane.A, B=kane.B, P=kane.P, F_sat_V_cm=np.inf)
    print(f"grid {len(x)} pts, Na={dev.Na:.1e}, Nd={dev.Nd:.1e}")
    print(f"uniform-field limit: max |G_nonlocal/G_local - 1| = {uniform_field_check(kane_uncapped, Eg):.2e}")

    psi_eq, n_eq, p_eq, _ = solve_equilibrium(x, Cdop, mf)
    Va_list = np.round(np.concatenate([np.arange(0.0, -5.01, -0.25), np.arange(0.1, 0.61, 0.1)]), 4)
    models = {"srh": NO_KANE, "local_capped": kane, "local": kane_uncapped, "nonlocal": NO_KANE}
    res = {m: {} for m in models}
    t0 = time.perf_counter()
    for m, km in models.items():
        for branch in (Va_list[Va_list <= 0], Va_list[Va_list > 0]):
            prev = None
            for Va in branch:
                init = {} if prev is None else dict(psi_init=prev["psi"], phin_init=prev["phin"],
                                                    phip_init=prev["phip"])
                solve = lambda G_ext, init: newton_gummel_solve(
                    x, Cdop, mf, Va, psi_eq, n_eq, p_eq, kane_model=km, trap_model="srh",
                    trap_generation_fn=srh_trap_fn, G_ext=G_ext, **init)
                if m != "nonlocal":
                    r = solve(None, init)
                    r["outer"] = 0
                else:
                    r = solve(prev["G_ext"] if prev is not None else None, init)
                    J_old = terminal_J(r)[1]
                    for k in range(1, OUTER_MAX + 1):
                        Gn, Gp, l, F_eff = nonlocal_generation_1d(x, r["psi"], r["phin"], r["phip"], cvol, Eg, Vt,
                                                                  kane_uncapped)
                        r = solve((Gn, Gp), dict(psi_init=r["psi"], phin_init=r["phin"], phip_init=r["phip"]))
                        J_new = terminal_J(r)[1]
                        if abs(J_new - J_old) <= OUTER_TOL * max(abs(J_new), 1e-30):
                            break
                        J_old = J_new
                    r["outer"] = k
                    r["G_ext"] = (Gn, Gp)
                    r["F_eff"], r["Gp"] = F_eff, Gp
                res[m][float(Va)] = r
                prev = r
    print(f"all sweeps: {time.perf_counter() - t0:.1f}s")

    Vs = np.array(sorted(res["srh"]))
    J = {m: np.array([terminal_J(res[m][v])[1] for v in Vs]) for m in models}
    Jl = {m: np.array([terminal_J(res[m][v])[0] for v in Vs]) for m in models}
    print(f"\n{'Va':>6} {'J_srh':>12} | BTBT part J-J_srh: " + " ".join(f"{m:>13}" for m in models if m != 'srh')
          + "  outer  |J0-JL|/J (nonlocal)")
    for i, v in enumerate(Vs):
        mis = abs(Jl["nonlocal"][i] - J["nonlocal"][i]) / max(abs(J["nonlocal"][i]), 1e-30)
        print(f"{v:+6.2f} {J['srh'][i]:12.4e} | {'':18}" + " ".join(f"{J[m][i] - J['srh'][i]:13.4e}" for m in models if m != "srh") +
              f"  {res['nonlocal'][v]['outer']:4d}   {mis:.1e}")

    # I-V
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    sty = {"srh": ("#6c757d", "SRH only"), "local_capped": ("#e8590c", "local Kane, F_sat=9e5 (1D as shipped)"),
           "local": ("#d6336c", "local Kane, uncapped"), "nonlocal": ("#1f6feb", "nonlocal path Kane")}
    rev = Vs <= 0
    for m, (col, lbl) in sty.items():
        a1.semilogy(-Vs[rev], np.abs(J[m][rev]), "o-", ms=3, color=col, label=lbl + " (total)")
        if m != "srh":
            a1.semilogy(-Vs[rev], np.abs(J[m][rev] - J["srh"][rev]), ":", color=col, label=lbl + ": BTBT part")
    a1.set_xlabel("reverse bias -Va (V)"); a1.set_ylabel("|J| (A/cm^2)"); a1.grid(alpha=0.3, which="both")
    a1.set_title(f"1D p {dev.Na:.0e} / n+ {dev.Nd:.0e} diode, reverse bias"); a1.legend(fontsize=8)
    # generation profiles at -3V
    v = -3.0
    rl, rn = res["local"][v], res["nonlocal"][v]
    F_node, *_ = _node_field(rl["psi"], x)
    G_loc, _ = btbt_generation(F_node, kane_uncapped)
    xm = (x - x[np.argmax(np.sign(Cdop) != np.sign(Cdop[0]))]) * 1e7
    a2.semilogy(xm[1:-1], np.maximum(G_loc, 1e-30), color=sty["local"][0], label="local G(F(x))")
    a2.semilogy(xm, np.maximum(rn["Gp"], 1e-30), color=sty["nonlocal"][0], label="nonlocal: hole generation (path start)")
    a2.semilogy(xm, np.maximum(rn["G_ext"][0], 1e-30), "--", color=sty["nonlocal"][0], label="nonlocal: electron generation (path end)")
    a2.set_xlim(-150, 50); a2.set_ylim(1e10, None)
    a2.set_xlabel("x - x_junction (nm)"); a2.set_ylabel("G (cm^-3 s^-1)")
    a2.set_title(f"generation profile at Va = {v:g} V"); a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "btbt_1d_local_vs_nonlocal.png"), dpi=150)
    print(f"wrote {out}/btbt_1d_local_vs_nonlocal.png")


if __name__ == "__main__":
    main()
