"""Robustness matrix for the avalanche breakdown trace: every device below x
every mesh below is traced from 0 V through breakdown to J_STOP, and each
run is checked, not just plotted.

    python3 -m avalanche.robustness_matrix            # full matrix (~45 s)
    python3 -m avalanche.robustness_matrix gauss n+p  # only devices matching a substring

Devices cover flat / gaussian / log-graded doping on either side, both
polarities (heavy p-side and heavy n-side), 1e17-1e21 doping and BV from
~8 V to ~55 V. Meshes run from the example's very fine default (sub-0.1 nm
junction cells) to a coarse mesh with a few-nm junction cells and ~50-100
nodes in total. The mesh is sized from doping only (no avalanche-specific
refinement), so the matrix also shows how far a generic mesh can be pushed.

Per run: PASS requires the trace to reach J_STOP, and every traced point to
conserve current to CONS_TOL on every resolvable edge (see
core/newton_numerics.resolved_current). Also reported: BV (the Va where
|J| = 1 A/cm^2), its shift relative to the finest mesh, any voltage
snapback, and Sze's closed-form estimate for comparison.

Outputs in out/avalanche/: robustness_matrix.csv, robustness_matrix_iv.png,
doping_sweep_iv.png.
"""
import copy
import csv
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import config as cfg
from core.mesh import build_diode_grid
from avalanche.newton_solver_avalanche import trace_breakdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "avalanche")
BASE_CFG = os.path.join(ROOT, "configs", "input_diode_breakdown.yaml")
J_STOP = 1e3        # A/cm^2
CONS_TOL = 1e-3     # max relative current non-conservation allowed on any traced point


def _flat(c):
    return {"type": "flat", "concentration_cm3": c}


# name: (p_side, n_side, Wp_um, Wn_um)
DEVICES = {
    "p+n flat 1e18/1e17": (_flat(1e18), _flat(1e17), 1.0, 6.0),
    "p+n flat 1e21/1e17": (_flat(1e21), _flat(1e17), 1.0, 6.0),
    "n+p flat 1e17/1e20": (_flat(1e17), _flat(1e20), 6.0, 1.0),
    "p+n flat 1e19/1e16 (high BV)": (_flat(1e19), _flat(1e16), 1.0, 12.0),
    "p+n gaussian p+ / flat n": (
        {"type": "gaussian", "peak_cm3": 1e20, "peak_depth_um": 0.3, "straggle_um": 0.08, "background_cm3": 1e17},
        _flat(1e17), 1.0, 6.0),
    "gaussian p+ / gaussian n-well": (
        {"type": "gaussian", "peak_cm3": 1e20, "peak_depth_um": 0.3, "straggle_um": 0.08, "background_cm3": 1e17},
        {"type": "gaussian", "peak_cm3": 5e17, "peak_depth_um": 0.8, "straggle_um": 0.3, "background_cm3": 5e16},
        1.0, 6.0),
    "log-graded both sides": (
        {"type": "linear", "start_cm3": 1e17, "end_cm3": 1e20, "transition_um": 0.2, "log_ramp": True},
        {"type": "linear", "start_cm3": 3e16, "end_cm3": 1e18, "transition_um": 1.0, "log_ramp": True},
        1.0, 6.0),
    "n+p gaussian n+ / flat p 5e17": (
        _flat(5e17),
        {"type": "gaussian", "peak_cm3": 2e20, "peak_depth_um": 0.15, "straggle_um": 0.04, "background_cm3": 1e17},
        3.0, 1.0),
}

# name: (junction_spacing_debye_factor, growth)
MESHES = {
    "fine (0.05, 1.06)": (0.05, 1.06),
    "medium (0.3, 1.10)": (0.3, 1.10),
    "coarse (1.0, 1.15)": (1.0, 1.15),
    "very coarse (3.0, 1.25)": (3.0, 1.25),
}


def build_device(p_side, n_side, Wp_um, Wn_um, jfac, growth):
    c = copy.deepcopy(cfg.load_config(BASE_CFG))
    c["doping"] = {"p_side": p_side, "n_side": n_side}
    c["thickness"] = {"Wp_um": Wp_um, "Wn_um": Wn_um}
    c["mesh"]["junction_spacing_debye_factor"] = jfac
    c["mesh"]["growth"] = growth
    mat, dev, _, _, _, mesh_opts, _ = cfg.build_from_config(c)
    return mat, dev, build_diode_grid(mat, dev, **mesh_opts)


def sze_bv(mat, g):
    """Sze's one-sided abrupt-junction BV from the LIGHTER side's doping right
    at the junction - exact use of the formula for a flat step junction, an
    abrupt-junction approximation for graded/gaussian profiles."""
    N_B = min(float(g["p_profile"].sample(0.0, g["Wp"])), float(g["n_profile"].sample(0.0, g["Wn"])))
    return 60.0 * (mat.Eg_eV / 1.1) ** 1.5 * (N_B / 1.0e16) ** -0.75, N_B


def bv_at(Va, J, J_ref=1.0):
    """Va where |J| first crosses J_ref (log-interpolated)."""
    L = np.log(np.abs(J))
    i = int(np.argmax(L > np.log(J_ref)))
    if i == 0:
        return np.nan
    return float(np.interp(np.log(J_ref), L[i - 1:i + 1], Va[i - 1:i + 1]))


def run_one(mat, g):
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = trace_breakdown(g["x"], g["Cdop"], mat, J_stop_A_cm2=J_STOP)
    prob = tr["problem"]
    cons = max(prob.terminal_current(U)[1] / abs(prob.terminal_current(U)[0]) for U in tr["U"][1:])
    Va, J = tr["Va"], tr["J"]
    i_min = int(np.argmin(Va))
    snapback = float(np.max(Va[i_min:]) - Va[i_min])
    ok = tr["status"] == "reached_J_stop" and cons < CONS_TOL
    return dict(status=tr["status"], passed=ok, N=len(g["x"]), h_min_nm=float(np.min(np.diff(g["x"])) * 1e7),
                BV=bv_at(Va, J), points=len(Va), rejections=tr["rejections"], max_nonconservation=cons,
                snapback_mV=snapback * 1e3, J_end=float(J[-1]), seconds=time.time() - t0, Va=Va, J=J)


# ---- plotting (colors: dataviz reference palette; marker shape is a secondary encoding) ----
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e4e3dc"
MESH_COLORS = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"]   # one blue ramp, fine -> coarse
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def _style(ax):
    ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("Reverse bias |Va| (V)")
    ax.set_ylabel("|J| (A/cm²)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_matrix(results, path):
    names = list(dict.fromkeys(d for d, _ in results))
    ncol = 2
    nrow = (len(names) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.6 * nrow), squeeze=False)
    for ax, dn in zip(axes.ravel(), names):
        bv = None
        for (mn, col, mk) in zip(MESHES, MESH_COLORS, MARKERS):
            r = results.get((dn, mn))
            if r is None:
                continue
            ax.plot(-r["Va"], np.abs(r["J"]), "-", color=col, lw=2, marker=mk, ms=4, markevery=3,
                    label=f"{mn}: N={r['N']}, BV={-r['BV']:.2f} V" + ("" if r["passed"] else "  FAIL"))
            bv, NB, exact = r["sze"], r["N_B"], r["flat"]
        ax.axvline(bv, color=INK, ls="--", lw=1.2,
                   label=f"Sze BV = {bv:.1f} V" + ("" if exact else " (abrupt approx.)") + f", N_B={NB:.1e}")
        ax.set_title(dn, loc="left", color=INK, fontsize=10)
        _style(ax)
        ax.set_xlim(0, 1.15 * max(bv, max(-r["Va"].min() for (d, _), r in results.items() if d == dn)))
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    n_pass = sum(r["passed"] for r in results.values())
    fig.suptitle(f"Avalanche breakdown I–V: {len(names)} devices × {len(MESHES)} meshes "
                 f"({n_pass}/{len(results)} pass)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_doping_sweep(path):
    """The heavy-side doping sweep (1e18-1e21 on the p-side, 1e17 n-side) that
    used to fail at 1e20/1e21 - full range plus a zoom on the knee."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for Na, col, mk in zip((1e18, 1e19, 1e20, 1e21), SERIES_COLORS, MARKERS):
        mat, dev, g = build_device(_flat(Na), _flat(1e17), 1.0, 6.0, *MESHES["fine (0.05, 1.06)"])
        r = run_one(mat, g)
        for ax in (a1, a2):
            ax.plot(-r["Va"], np.abs(r["J"]), "-", color=col, lw=2, marker=mk, ms=4, markevery=2,
                    label=f"Na={Na:.0e}: BV={-r['BV']:.2f} V, {r['points']} pts, "
                          f"{'PASS' if r['passed'] else 'FAIL'}")
    bv, _ = sze_bv(mat, g)
    for ax in (a1, a2):
        ax.axvline(bv, color=INK, ls="--", lw=1.2, label=f"Sze BV = {bv:.1f} V (Nd=1e17)")
        _style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    a1.set_xlim(0, 16)
    a1.set_title("Heavy-side doping sweep (n-side 1e17) - full range", loc="left", color=INK)
    a2.set_xlim(13.0, 15.5)
    a2.set_title("Zoom on the breakdown knee", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(filters):
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for dn, (ps, ns, Wp, Wn) in DEVICES.items():
        if filters and not any(f in dn for f in filters):
            continue
        for mn, (jf, gr) in MESHES.items():
            mat, dev, g = build_device(ps, ns, Wp, Wn, jf, gr)
            r = run_one(mat, g)
            r["sze"], r["N_B"] = sze_bv(mat, g)
            r["flat"] = ps["type"] == "flat" and ns["type"] == "flat"
            results[(dn, mn)] = r
            print(f"{dn:32s} {mn:24s} N={r['N']:4d} {'PASS' if r['passed'] else 'FAIL'} "
                  f"BV={-r['BV']:8.3f} V (Sze {r['sze']:5.1f})  pts={r['points']:3d} rej={r['rejections']:2d} "
                  f"cons={r['max_nonconservation']:.1e} snapback={r['snapback_mV']:4.0f} mV  {r['seconds']:.1f}s",
                  flush=True)
    # BV shift relative to the finest mesh of the same device
    for (dn, mn), r in results.items():
        ref = results.get((dn, next(iter(MESHES))))
        r["BV_shift_vs_finest_pct"] = 100.0 * (r["BV"] - ref["BV"]) / ref["BV"] if ref else np.nan

    with open(os.path.join(OUT, "robustness_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = ["passed", "status", "N", "h_min_nm", "BV", "BV_shift_vs_finest_pct", "sze", "points",
                "rejections", "max_nonconservation", "snapback_mV", "J_end", "seconds"]
        w.writerow(["device", "mesh"] + keys)
        for (dn, mn), r in results.items():
            w.writerow([dn, mn] + [r[k] for k in keys])
    plot_matrix(results, os.path.join(OUT, "robustness_matrix_iv.png"))
    if not filters:
        plot_doping_sweep(os.path.join(OUT, "doping_sweep_iv.png"))
    n_pass = sum(r["passed"] for r in results.values())
    print(f"\n{n_pass}/{len(results)} runs pass; outputs in {OUT}")
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
