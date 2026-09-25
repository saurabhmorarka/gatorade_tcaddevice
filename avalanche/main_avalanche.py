"""Driver script: 1D avalanche-breakdown simulation.

    python3 -m avalanche.main_avalanche [configs/input_diode_breakdown.yaml]

Special, opt-in driver (see newton_solver_avalanche.py). Traces the reverse
I-V curve from near equilibrium straight through the avalanche knee with
arc-length continuation (newton_solver_avalanche.trace_breakdown) - no bias
schedule to tune - and compares it against:
  - the same device with impact ionization switched off (newton_qf), giving
    the numeric multiplication factor M(Va) = J_avalanche / J_no_avalanche;
  - Sze's closed-form breakdown voltage and Miller's M(Va) (analytic.py);
  - the ionization integrals evaluated on the solver's OWN potential at the
    numeric breakdown voltage (avalanche.ionization_integrals_from_field),
    which should be ~1 there, plus analytic.ionization_integral (depletion
    approximation) along the sweep.

Device, mesh and trace settings come from input_diode_breakdown.yaml
(config.py / avalanche_config.py). For the device x mesh robustness matrix,
see avalanche/robustness_matrix.py.
"""
import csv
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.mesh import build_diode_grid
from core import analytic as an
from core import config as cfg
from core import structure_io as sio
from core.newton_solver_qf import newton_gummel_solve as qf_solve
from avalanche import avalanche_config as acfg
from avalanche.avalanche import ionization_integrals_from_field
from avalanche.newton_solver_avalanche import trace_breakdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "avalanche")

# dataviz reference palette; marker/line style is a secondary encoding
C_AVAL, C_NOAVAL, C_MILLER, C_II = "#2a78d6", "#6b6a64", "#eb6834", "#1baf7a"
INK, GRID = "#1f1f1d", "#e4e3dc"
J_BV = 1.0   # A/cm^2 - the current density that defines the numeric BV


def numeric_bv(Va, J, J_ref=J_BV):
    L = np.log(np.abs(J))
    i = int(np.argmax(L > np.log(J_ref)))
    return float(np.interp(np.log(J_ref), L[i - 1:i + 1], Va[i - 1:i + 1])) if i > 0 else np.nan


def no_avalanche_currents(x, Cdop, mat, psi_eq, n_eq, p_eq, Va_list):
    """Voltage-controlled newton_qf continuation along Va_list (ordered from
    equilibrium outward) - the multiplication factor's denominator."""
    J, prev = [], None
    for Va in Va_list:
        kw = {} if prev is None else dict(psi_init=prev["psi"], phin_init=prev["phin"], phip_init=prev["phip"])
        prev = qf_solve(x, Cdop, mat, Va, psi_eq, n_eq, p_eq, **kw)
        J.append(prev["J_mean"])
    return np.array(J)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "configs", "input_diode_breakdown.yaml")
    input_cfg = cfg.load_config(input_path)
    mat, dev, _, _, _, mesh_opts, structure_file = cfg.build_from_config(input_cfg)
    av = acfg.parse_avalanche_config(input_cfg)
    cont = av["continuation"]

    refine = ({"E_crit_V_cm": av["E_crit_V_cm"], "ii_model": av["ii_model"], "cells_per_mfp": av["cells_per_mfp"]}
              if av["enabled"] else None)
    g = build_diode_grid(mat, dev, avalanche_ii_refine=refine, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]
    os.makedirs(OUT, exist_ok=True)
    print(f"Grid: {len(x)} points, Wp={g['Wp']*1e4:.2f} um, Wn={g['Wn']*1e4:.2f} um, h_min={g['h_min']*1e7:.3f} nm")
    print(f"Doping: Na={dev.Na:.2e} cm^-3 (p-side), Nd={dev.Nd:.2e} cm^-3 (n-side)")
    BV_sze = an.breakdown_voltage_sze(mat, dev)
    print(f"Sze closed-form breakdown voltage: {BV_sze:.2f} V")

    print(f"\nTracing I-V through breakdown (driving force: {av['driving_force']}, "
          f"arc-length continuation to |J| = {cont['J_stop_A_cm2']:.0e} A/cm^2)...")
    tr = trace_breakdown(x, Cdop, mat, ii_model=av["ii_model"], driving_force=av["driving_force"],
                         ref_density_cm3=av["ref_density_cm3"], seed_V=cont["seed_V"],
                         J_stop_A_cm2=cont["J_stop_A_cm2"], V_limit=cont["V_limit"], ds_max=cont["ds_max"])
    prob = tr["problem"]
    Va, J = tr["Va"], tr["J"]
    cons = np.array([prob.terminal_current(U)[1] / abs(prob.terminal_current(U)[0]) for U in tr["U"]])
    print(f"  {tr['status']}: {len(Va)} points ({tr['rejections']} arc-length step rejections), "
          f"worst current non-conservation {cons[1:].max():.1e}")
    BV = numeric_bv(Va, J)
    k_bv = int(np.argmin(np.abs(np.log(np.abs(J) / J_BV))))
    In, Ip = ionization_integrals_from_field(x, prob.densities(tr["U"][k_bv])[0], av["ii_model"])
    print(f"  numeric BV (|J| = {J_BV:g} A/cm^2): {-BV:.3f} V   (Sze: {BV_sze:.2f} V)")
    print(f"  ionization integrals on the solved field at BV: electron {In:.3f}, hole {Ip:.3f} (expect ~1)")

    print("Solving the same device without impact ionization for M(Va)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        J_no = no_avalanche_currents(x, Cdop, mat, tr["psi_eq"], tr["n_eq"], tr["p_eq"], Va)
    M = J / J_no
    # Miller's form is only defined below its BV (it diverges there) - mask the rest.
    M_miller = np.where(-Va < BV_sze, an.multiplication_factor_miller(Va, BV_sze), np.nan)
    ii_depl = np.array([an.ionization_integral(mat, dev, v, av["ii_model"]) for v in Va])
    I = J * dev.area

    csv_path = os.path.join(OUT, "breakdown_iv.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "J_A_cm2", "I_A", "J_no_avalanche_A_cm2", "M_numeric", "M_miller",
                    "ionization_integral_depletion_approx", "J_std_over_J"])
        for row in zip(Va, J, I, J_no, M, M_miller, ii_depl, cons):
            w.writerow(row)
    print(f"\nWrote {csv_path}")

    # ---- plots: log I-V, linear knee, M(Va), ionization integral ----
    V = -Va
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    (a_log, a_lin), (a_M, a_ii) = axes
    for ax in axes.ravel():
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvline(BV_sze, color=INK, ls="--", lw=1.2, label=f"Sze BV = {BV_sze:.1f} V")
        ax.set_xlabel("Reverse bias |Va| (V)")

    a_log.semilogy(V, np.abs(J), "-o", color=C_AVAL, lw=2, ms=3, label="numeric, avalanche (arc-length trace)")
    a_log.semilogy(V, np.abs(J_no), "--", color=C_NOAVAL, lw=2, label="numeric, no impact ionization")
    a_log.semilogy(V, np.abs(J_no) * M_miller, ":", color=C_MILLER, lw=2, label="Miller M(Va) x no-avalanche J")
    a_log.axvline(-BV, color=C_AVAL, ls="-.", lw=1, label=f"numeric BV = {-BV:.2f} V (|J| = 1 A/cm²)")
    a_log.set_ylabel("|J| (A/cm²)")
    a_log.set_title("Reverse I–V, log scale", loc="left", color=INK)

    knee = V > 0.85 * (-BV)
    a_lin.plot(V[knee], np.abs(J[knee]), "-o", color=C_AVAL, lw=2, ms=3, label="numeric, avalanche")
    a_lin.set_ylabel("|J| (A/cm²)")
    a_lin.set_title("Breakdown knee, linear scale", loc="left", color=INK)

    rev = V > 0.05
    a_M.semilogy(V[rev], M[rev], "-o", color=C_AVAL, lw=2, ms=3, label="numeric M = J / J_no-avalanche")
    a_M.semilogy(V[rev], M_miller[rev], ":", color=C_MILLER, lw=2, label="Miller M(Va), n=3")
    a_M.set_ylabel("multiplication factor M")
    a_M.set_title("Multiplication factor", loc="left", color=INK)

    a_ii.plot(V, ii_depl, "-", color=C_II, lw=2, label="ionization integral, depletion-approx. field")
    a_ii.plot([-BV], [max(In, Ip)], "D", color=C_AVAL, ms=7,
              label=f"ionization integral on the solved field at numeric BV = {max(In, Ip):.2f}")
    a_ii.axhline(1.0, color=INK, ls=":", lw=1)
    a_ii.set_ylabel("ionization integral")
    a_ii.set_title("Breakdown criterion", loc="left", color=INK)
    for ax in axes.ravel():
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    plot_path = os.path.join(OUT, "breakdown_iv.png")
    fig.savefig(plot_path, dpi=130)
    print(f"Wrote {plot_path}")

    # ---- structure + fields at a few current levels ----
    bias_points = []
    for J_target in (1e-4, 1e-2, 1.0, 1e2, cont["J_stop_A_cm2"]):
        i = int(np.argmin(np.abs(np.log(np.abs(J) / J_target))))
        psi, phin, phip, n, p = prob.densities(tr["U"][i])
        bias_points.append({"label": f"Va={Va[i]:+.3f}V (J={abs(J[i]):.1e} A/cm2)", "bias": float(Va[i]),
                            "fields": {"psi": psi, "n": n, "p": p, "phin": phin, "phip": phip}})
    sio.save_structure(
        os.path.join(OUT, structure_file), device="diode", dim=1,
        material={"eps_r": mat.eps_r, "ni_cm3": mat.ni, "chi_eV": mat.chi_eV, "Eg_eV": mat.Eg_eV},
        regions=[
            {"name": "p-side", "x_range_um": [x[0] * 1e4, 0.0], "kind": "semiconductor", "doping_type": "p"},
            {"name": "n-side", "x_range_um": [0.0, x[-1] * 1e4], "kind": "semiconductor", "doping_type": "n"},
        ],
        x_um=x * 1e4, doping_cm3=Cdop, bias_points=bias_points)
    print(f"Wrote {os.path.join(OUT, structure_file)}")


if __name__ == "__main__":
    main()
