"""Shared harness: run each of the four example configurations and return a
small dict of plain-float summary metrics (no plotting, no file I/O beyond
reading the input YAML) - used by both capture_golden.py (to snapshot
today's numbers as the reference) and test_examples.py (to compare a future
run against that snapshot).

Deliberately summary-only, not full field dumps: a handful of scalar
metrics per example (a physical constant, a boundary-condition-derived
voltage, and C/I at a few representative sweep points) is enough to catch a
real regression - the kind of order-of-magnitude or sign errors this
project's numerics have actually broken on - without the golden files being
so large/precise-per-node that routine, harmless changes (mesh tweaks,
plotting) spuriously fail the suite.
"""
import os
import sys

TCAD1D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TCAD1D_ROOT not in sys.path:
    sys.path.insert(0, TCAD1D_ROOT)

import numpy as np

from core import config as diode_cfg
from core.mesh import build_diode_grid, build_mos_grid
from core.solver import voltage_sweep
from core import analytic as dan
from avalanche import avalanche_config as acfg
from tat import tat_config as tcfg

from mos import mos_config
from mos.mos_solver import cv_sweep
from mos import mos_analytic as man
from core import physics as ph
from core.doping_profiles import DopingProfile

POLY_SWEEP_DOPINGS = [1e17, 1e18, 1e19, 1e20, 1e21, 1e22]


def _pick(arr, fracs=(0.0, 0.5, 1.0)):
    """Values at the given fractional positions along arr (0=first, 1=last)."""
    n = len(arr)
    return [float(arr[min(int(round(f * (n - 1))), n - 1)]) for f in fracs]


def run_diode(path=None):
    path = path or diode_cfg.DEFAULT_PATH
    input_cfg = diode_cfg.load_config(path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = \
        diode_cfg.build_from_config(input_cfg)

    g = build_diode_grid(mat, dev, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]

    psi_eq, n_eq, p_eq, results = voltage_sweep(x, Cdop, mat, dev, Va_list, method=math_model)
    Vbi_num = float(psi_eq[-1] - psi_eq[0])
    Vbi_an = float(dan.built_in_potential(mat, dev))
    I0 = float(dan.shockley_I0(mat, dev))

    Va_arr = np.array([r["Va"] for r in results])
    I_num = np.array([r["I"] for r in results])
    idx = [0, len(Va_arr) // 2, -1]

    return {
        "n_mesh_points": len(x),
        "Vbi_numeric_V": Vbi_num,
        "Vbi_analytic_V": Vbi_an,
        "shockley_I0_A": I0,
        "Va_sample_V": [float(Va_arr[i]) for i in idx],
        "I_numeric_sample_A": [float(I_num[i]) for i in idx],
    }


def run_diode_breakdown(path=None):
    """Avalanche-breakdown example, traced through breakdown with arc-length
    continuation exactly as avalanche/main_avalanche.py does. Regression
    scalars: the numeric breakdown voltage (Va at |J| = 1 A/cm^2 - a sharp,
    well-defined point of a curve that is nearly vertical there), the current
    at two leakage-regime biases (log-interpolated along the trace), and two
    closed-form numbers (Sze's BV and the depletion-approximation
    ionization integral at -10 V) that do not depend on the PDE solve."""
    from avalanche.newton_solver_avalanche import trace_breakdown
    path = path or os.path.join(TCAD1D_ROOT, "configs", "input_diode_breakdown.yaml")
    input_cfg = diode_cfg.load_config(path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = \
        diode_cfg.build_from_config(input_cfg)
    av = acfg.parse_avalanche_config(input_cfg)
    cont = av["continuation"]

    g = build_diode_grid(mat, dev,
                          avalanche_ii_refine={"E_crit_V_cm": av["E_crit_V_cm"],
                                                "ii_model": av["ii_model"],
                                                "cells_per_mfp": av["cells_per_mfp"]},
                          **mesh_opts)
    tr = trace_breakdown(g["x"], g["Cdop"], mat, ii_model=av["ii_model"], driving_force=av["driving_force"],
                         ref_density_cm3=av["ref_density_cm3"], seed_V=cont["seed_V"],
                         J_stop_A_cm2=cont["J_stop_A_cm2"], V_limit=cont["V_limit"], ds_max=cont["ds_max"])
    Va, logJ = tr["Va"], np.log(np.abs(tr["J"]))
    i = int(np.argmax(logJ > 0.0))
    BV = float(np.interp(0.0, logJ[i - 1:i + 1], Va[i - 1:i + 1]))
    # Va decreases along the (pre-breakdown) trace; np.interp needs increasing x
    # Current DENSITY (A/cm^2, ~1e-5 here), not current: the golden check's
    # abs_tol=1e-8 would swallow any change to a ~1e-9 A terminal current.
    J_at = [float(np.exp(np.interp(v, Va[:i][::-1], logJ[:i][::-1]))) for v in (-1.0, -10.0)]

    return {
        "n_mesh_points": len(g["x"]),
        "trace_status": tr["status"],
        "BV_sze_estimate_V": float(dan.breakdown_voltage_sze(mat, dev)),
        "ionization_integral_at_neg10V": float(dan.ionization_integral(mat, dev, -10.0, av["ii_model"])),
        "BV_numeric_at_1A_cm2_V": BV,
        "Va_sample_V": [-1.0, -10.0],
        "J_numeric_sample_A_cm2": J_at,
    }


def run_diode_tat(path=None):
    """Drain-to-substrate TAT+BTBT leakage example (newton_tat). Regression
    scalars are sampled at moderate reverse bias (-1V, -5V) - well inside
    this mechanism's smooth, monotonic leakage-enhancement curve (see
    plans/tat_btbt_plan.md and tat/main_tat.py's own I(Va) plot), not near
    any sharp transition, so this stays stable under the normal
    golden-comparison tolerance the way run_diode_breakdown's own sample
    points deliberately avoid avalanche's sharp runaway."""
    path = path or os.path.join(TCAD1D_ROOT, "configs", "input_diode_drain_substrate.yaml")
    input_cfg = diode_cfg.load_config(path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = \
        diode_cfg.build_from_config(input_cfg)
    tcfg.parse_tat_config(input_cfg)  # exercised for parse-error coverage; models default inside the solver

    g = build_diode_grid(mat, dev, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]

    _, _, _, results = voltage_sweep(x, Cdop, mat, dev, Va_list, method="newton_tat")
    Va_arr = np.array([r["Va"] for r in results])
    I_num = np.array([r["I"] for r in results])

    idx_neg1 = int(np.argmin(np.abs(Va_arr - (-1.0))))
    idx_neg5 = int(np.argmin(np.abs(Va_arr - (-5.0))))

    return {
        "n_mesh_points": len(x),
        "Va_sample_V": [float(Va_arr[idx_neg1]), float(Va_arr[idx_neg5])],
        "I_numeric_sample_A": [float(I_num[idx_neg1]), float(I_num[idx_neg5])],
    }


def run_mos_metal(path=None):
    path = path or mos_config.DEFAULT_PATH
    input_cfg = mos_config.load_config(path)
    mat, dev, Cdop_substrate, VG_list, save_bias_points, mesh_opts, structure_file, Cdop_gate = \
        mos_config.build_from_config(input_cfg)
    assert Cdop_gate is None, f"{path}: expected an ideal-metal-gate example (gate.type: metal)"

    g = build_mos_grid(mat, dev, Cdop_substrate, Cdop_gate=Cdop_gate, **mesh_opts)
    x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]

    Cox = man.C_ox(dev)
    V_FB = float(man.flatband_voltage(dev, mat, Cdop_substrate))
    V_T = float(man.threshold_voltage(mat, dev, Cdop_substrate))

    results = cv_sweep(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate, VG_list, g["oxide_index"])
    VG_arr = np.array([r["VG"] for r in results])
    C_lf = np.array([r["C_lf"] for r in results]) / Cox
    C_hf = np.array([r["C_hf"] for r in results]) / Cox
    idx = [0, len(VG_arr) // 2, -1]

    return {
        "n_mesh_points": len(x),
        "C_ox_F_cm2": float(Cox),
        "V_FB_V": V_FB,
        "V_T_V": V_T,
        "VG_sample_V": [float(VG_arr[i]) for i in idx],
        "C_lf_over_Cox_sample": [float(C_lf[i]) for i in idx],
        "C_hf_over_Cox_sample": [float(C_hf[i]) for i in idx],
    }


def run_mos_poly_single(path=None):
    path = path or os.path.join(TCAD1D_ROOT, "configs", "input_mos_poly.yaml")
    input_cfg = mos_config.load_config(path)
    mat, dev, Cdop_substrate, VG_list, save_bias_points, mesh_opts, structure_file, Cdop_gate = \
        mos_config.build_from_config(input_cfg)
    assert Cdop_gate is not None, f"{path}: expected a poly-gate example (gate.type: poly)"

    g = build_mos_grid(mat, dev, Cdop_substrate, Cdop_gate=Cdop_gate, **mesh_opts)
    x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]
    gate_oxide_index = g["gate_oxide_index"]

    Cox = man.C_ox(dev)
    import copy
    dev_ideal = copy.copy(dev)
    dev_ideal.gate_workfunction_eV = man.semiconductor_workfunction_eV(mat, Cdop_gate)
    V_FB = float(man.flatband_voltage(dev_ideal, mat, Cdop_substrate))
    V_T = float(man.threshold_voltage(mat, dev_ideal, Cdop_substrate))

    results = cv_sweep(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate, VG_list,
                        g["oxide_index"], Cdop_gate=Cdop_gate, gate_oxide_index=gate_oxide_index)
    VG_arr = np.array([r["VG"] for r in results])
    C_lf = np.array([r["C_lf"] for r in results]) / Cox
    C_hf = np.array([r["C_hf"] for r in results]) / Cox
    idx = [0, len(VG_arr) // 2, -1]

    psi_bulk_gate = float(ph.equilibrium_bulk_potential(mat, Cdop_gate))
    poly_bending_last = float(results[-1]["psi"][gate_oxide_index] - psi_bulk_gate)

    return {
        "n_mesh_points": len(x),
        "Cdop_gate_cm3": float(Cdop_gate),
        "C_ox_F_cm2": float(Cox),
        "V_FB_ideal_poly_V": V_FB,
        "V_T_ideal_poly_V": V_T,
        "VG_sample_V": [float(VG_arr[i]) for i in idx],
        "C_lf_over_Cox_sample": [float(C_lf[i]) for i in idx],
        "C_hf_over_Cox_sample": [float(C_hf[i]) for i in idx],
        "poly_bending_at_VGmax_V": poly_bending_last,
    }


def run_mos_poly_sweep(path=None, dopings=POLY_SWEEP_DOPINGS):
    path = path or os.path.join(TCAD1D_ROOT, "configs", "input_mos_poly.yaml")
    input_cfg = mos_config.load_config(path)
    mat, dev, Cdop_substrate, VG_list, save_bias_points, mesh_opts, structure_file, Cdop_gate_cfg = \
        mos_config.build_from_config(input_cfg)
    assert Cdop_gate_cfg is not None, f"{path}: expected a poly-gate example (gate.type: poly)"
    gate_polarity_sign = 1.0 if Cdop_gate_cfg >= 0 else -1.0

    Cox = man.C_ox(dev)
    per_doping = []
    for Ngate in dopings:
        dev.gate_profile = DopingProfile.flat(Ngate)
        Cdop_gate = gate_polarity_sign * Ngate

        g = build_mos_grid(mat, dev, Cdop_substrate, Cdop_gate=Cdop_gate, **mesh_opts)
        x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]

        results = cv_sweep(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate, VG_list,
                            g["oxide_index"], Cdop_gate=Cdop_gate, gate_oxide_index=g["gate_oxide_index"])
        C_lf = np.array([r["C_lf"] for r in results]) / Cox

        import copy
        dev_ideal = copy.copy(dev)
        dev_ideal.gate_workfunction_eV = man.semiconductor_workfunction_eV(mat, Cdop_gate)
        V_FB = float(man.flatband_voltage(dev_ideal, mat, Cdop_substrate))
        V_T = float(man.threshold_voltage(mat, dev_ideal, Cdop_substrate))

        per_doping.append({
            "Ngate_cm3": float(Ngate),
            "V_FB_V": V_FB,
            "V_T_V": V_T,
            "C_lf_over_Cox_at_VGmin": float(C_lf[0]),
            "C_lf_over_Cox_at_VGmax": float(C_lf[-1]),
        })

    return {"C_ox_F_cm2": float(Cox), "per_doping": per_doping}


EXAMPLES = {
    "diode": run_diode,
    "diode_breakdown": run_diode_breakdown,
    "diode_tat": run_diode_tat,
    "mos_metal": run_mos_metal,
    "mos_poly_single": run_mos_poly_single,
    "mos_poly_sweep": run_mos_poly_sweep,
}
