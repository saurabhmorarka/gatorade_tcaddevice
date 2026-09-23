"""Phase-2 2D bias-sweep driver: build the mesh, solve the 2D QF Newton
system across a voltage sweep (warm-starting each point from its neighbor,
the same continuation trick core/solver.py::voltage_sweep uses for 1D),
extract the terminal current density at the anode, and compare directly
against a matching 1D diode solve (same doping, same material) - both
linear and log-scale I-V.

Usage: python3 main2d_sweep.py [configs/input_diode_2d.yaml]
"""
import argparse
import csv
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from core import structure_io as sio
from core.field_save import resolve_save_points
from core.mesh import build_diode_grid
from core.params import Device, Material
from core.solver import voltage_sweep as voltage_sweep_1d
from mesh2d import config2d
from mesh2d.mesh2d import build_mesh2d
from solver2d.current import contact_current_density
from solver2d.efield2d import electric_field_2d
from solver2d.newton_solver_qf_2d import newton_solve_2d
from viz2d.plot2d import plot_field2d

_CM_TO_UM = 1.0e4


def sweep_2d(mesh, mat, Va_list, contact_bias_role="anode", verbose=True):
    """Continuation sweep: solve Va=0 first (cold start), then walk outward
    in both directions warm-starting each point from its already-solved
    neighbor - mirrors core/solver.py::voltage_sweep's own approach.
    Returns {Va: result_dict}."""
    Va_arr = np.asarray(sorted(Va_list), dtype=float)
    zero_idx = int(np.argmin(np.abs(Va_arr)))
    results = {}

    def bias_dict(Va):
        return {c.name: (Va if c.bias_role == contact_bias_role else 0.0) for c in mesh.domain.contacts}

    t0 = time.perf_counter()
    r0 = newton_solve_2d(mesh, mat, bias_dict(float(Va_arr[zero_idx])))
    r0["solve_time_s"] = time.perf_counter() - t0
    results[Va_arr[zero_idx]] = r0
    if verbose:
        print(f"Va={Va_arr[zero_idx]:+.3f} V: iters={r0['iters']}, res_norm={r0['res_norm']:.3e}, "
              f"t={r0['solve_time_s']:.4f}s")

    for direction, idx_range in ((+1, range(zero_idx + 1, len(Va_arr))),
                                  (-1, range(zero_idx - 1, -1, -1))):
        prev = r0
        for idx in idx_range:
            Va = Va_arr[idx]
            t0 = time.perf_counter()
            r = newton_solve_2d(mesh, mat, bias_dict(float(Va)),
                                 psi_init=prev["psi"], phin_init=prev["phin"], phip_init=prev["phip"])
            r["solve_time_s"] = time.perf_counter() - t0
            results[Va] = r
            status = "OK" if r["res_norm"] < 1e-4 else "NOT CONVERGED"
            if verbose:
                print(f"Va={Va:+.3f} V: iters={r['iters']:3d}, res_norm={r['res_norm']:.3e}, "
                      f"t={r['solve_time_s']:.4f}s  [{status}]")
            prev = r
    return results


def main():
    parser = argparse.ArgumentParser(description="gatorade_tcaddevice 2D bias-sweep driver")
    parser.add_argument("config", nargs="?", default=config2d.DEFAULT_PATH)
    parser.add_argument("--plot-bias", default="-1.0,-0.5,0.0,0.5,1.0",
                         help="Comma-separated bias values (nearest swept match) to save static field-map PNGs for")
    args = parser.parse_args()

    cfg = config2d.load_config(args.config)
    domain, mat, mesh_opts, output_cfg, Va_list = config2d.build_from_config(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mesh-quality residual already reported once via main2d.py
        mesh = build_mesh2d(domain, **mesh_opts)
    print(f"mesh: {len(mesh.points)} points, {len(mesh.triangles)} triangles")

    out_dir = os.path.join("out", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)

    print("--- 2D sweep (continuation, warm-started from Va=0 outward) ---")
    t0 = time.perf_counter()
    results_2d = sweep_2d(mesh, mat, Va_list)
    total_time_2d = time.perf_counter() - t0

    n_failed = sum(1 for r in results_2d.values() if r["res_norm"] >= 1e-4)
    print(f"2D sweep: {len(results_2d) - n_failed}/{len(results_2d)} points converged, "
          f"total {total_time_2d:.3f}s ({total_time_2d / len(results_2d):.4f}s/point avg)")

    anode_name = next(c.name for c in domain.contacts if c.bias_role == "anode")
    J_2d = {Va: contact_current_density(mesh, mat, r, anode_name) for Va, r in results_2d.items()}

    # --- Terminal I-V: ALWAYS written for every swept point, regardless of
    # output.save_bias_points (which only controls the much heavier
    # full-field-at-every-mesh-point saves below) - mirrors main.py's own
    # iv_sweep.csv (every point)/fields_by_bias.csv (user-selected subset)
    # split for the 1D diode. ---
    iv_csv_path = os.path.join(out_dir, "iv_sweep.csv")
    with open(iv_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "J_anode_A_cm2", "res_norm", "iters"])
        for Va in sorted(J_2d):
            r = results_2d[Va]
            w.writerow([f"{Va:.4f}", f"{J_2d[Va]:.6e}", f"{r['res_norm']:.3e}", r["iters"]])
    print(f"wrote {iv_csv_path} ({len(J_2d)} bias point(s), every swept point)")

    # --- Matching 1D diode: same doping (Na=p_well conc, Nd=substrate conc), same material ---
    print("--- 1D reference sweep (same doping/material) ---")
    p_region = domain.regions[1]
    n_region = domain.regions[0]
    dev = Device(Na=p_region.concentration_cm3, Nd=n_region.concentration_cm3)
    g = build_diode_grid(mat, dev)
    t0 = time.perf_counter()
    _, _, _, results_1d_list = voltage_sweep_1d(g["x"], g["Cdop"], mat, dev, Va_list, method="newton_qf")
    total_time_1d = time.perf_counter() - t0
    print(f"1D sweep: {len(results_1d_list)} points, total {total_time_1d:.3f}s "
          f"({total_time_1d / len(results_1d_list):.4f}s/point avg)")
    print(f"2D/1D total-time ratio: {total_time_2d / total_time_1d:.1f}x "
          f"(2D mesh: {len(mesh.points)} points vs 1D: {len(g['x'])} nodes)")
    J_1d = {r["Va"]: r["J_mean"] for r in results_1d_list}

    # --- I-V comparison plots ---
    Va_sorted = np.array(sorted(J_2d))
    J2 = np.array([J_2d[v] for v in Va_sorted])
    J1 = np.array([J_1d[v] for v in Va_sorted])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(Va_sorted, J1, "o-", label="1D diode", color="#1f77b4")
    axes[0].plot(Va_sorted, J2, "s--", label="2D (anode-averaged)", color="#d62728")
    axes[0].set_xlabel("Va (V)"); axes[0].set_ylabel("J (A/cm^2)")
    axes[0].set_title("I-V: linear"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].semilogy(Va_sorted, np.abs(J1) + 1e-30, "o-", label="1D diode", color="#1f77b4")
    axes[1].semilogy(Va_sorted, np.abs(J2) + 1e-30, "s--", label="2D (anode-averaged)", color="#d62728")
    axes[1].set_xlabel("Va (V)"); axes[1].set_ylabel("|J| (A/cm^2)")
    axes[1].set_title("I-V: semilog"); axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "iv_comparison_1d_vs_2d.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # --- Field maps at requested bias points ---
    points_um = mesh.points * _CM_TO_UM
    plot_bias = [float(v) for v in args.plot_bias.split(",")]
    for target in plot_bias:
        Va = min(Va_sorted, key=lambda v: abs(v - target))
        r = results_2d[Va]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plot_field2d(points_um, mesh.triangles, r["psi"], ax=axes[0],
                     title=f"psi (V), Va={Va:.3f}V", cmap="RdBu_r", label="psi (V)")
        plot_field2d(points_um, mesh.triangles, r["n"], ax=axes[1],
                     title=f"n (cm^-3), Va={Va:.3f}V", log_scale=True, label="n (cm^-3)")
        plot_field2d(points_um, mesh.triangles, r["p"], ax=axes[2],
                     title=f"p (cm^-3), Va={Va:.3f}V", cmap="magma", log_scale=True, label="p (cm^-3)")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"fields_Va{Va:+.2f}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")

    # --- Vertical cut (band-bending style overlay) through the p-well center ---
    x_center_cm = 0.5 * (p_region.x_range_cm[0] + p_region.x_range_cm[1])
    y_cut_cm = np.linspace(0.0, domain.height_cm, 120)
    cut_pts = np.stack([np.full_like(y_cut_cm, x_center_cm), y_cut_cm], axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("coolwarm")
    plot_vas = [min(Va_sorted, key=lambda v: abs(v - t)) for t in plot_bias]
    for i, Va in enumerate(plot_vas):
        r = results_2d[Va]
        psi_cut = griddata(mesh.points, r["psi"], cut_pts, method="linear")
        color = cmap(i / max(1, len(plot_vas) - 1))
        ax.plot(y_cut_cm * _CM_TO_UM, psi_cut, label=f"Va={Va:+.2f}V", color=color)
    ax.set_xlabel("y (um), depth from top surface")
    ax.set_ylabel("psi (V)")
    ax.set_title(f"Vertical cut through p_well center (x={x_center_cm * _CM_TO_UM:.1f}um)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "vertical_cut_psi.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # --- Structure+fields JSON: every field, at every requested bias point
    # (output.save_bias_points, same "all"/"last"/[list] convention as 1D's
    # own output.save_bias_points - core/field_save.py::resolve_save_points
    # reused verbatim) - this is what viz2d's interactive viewer reads. ---
    save_spec = output_cfg.get("save_bias_points", "all")
    save_idx = resolve_save_points(save_spec, Va_sorted)
    regions = [
        {
            "name": r.name,
            "x_range_um": [r.x_range_cm[0] * _CM_TO_UM, r.x_range_cm[1] * _CM_TO_UM],
            "y_range_um": [r.y_range_cm[0] * _CM_TO_UM, r.y_range_cm[1] * _CM_TO_UM],
            "kind": r.kind,
            "doping_type": r.doping_type,
        }
        for r in domain.regions
    ]
    boundary = [
        {"point_index": int(i), "bc_type": bc_type}
        for i, bc_type in zip(mesh.boundary_point_index, mesh.boundary_bc_type)
    ]
    material_dict = dict(eps_r=mat.eps_r, ni=mat.ni, mu_n=mat.mu_n, mu_p=mat.mu_p,
                          tau_n=mat.tau_n, tau_p=mat.tau_p, chi_eV=mat.chi_eV, Eg_eV=mat.Eg_eV)
    bias_points = []
    for idx in save_idx:
        Va = Va_sorted[idx]
        r = results_2d[Va]
        Ex, Ey = electric_field_2d(mesh.points, mesh.triangles, r["psi"])
        fields = {k: r[k] for k in ("psi", "n", "p", "phin", "phip")}
        fields["Ex"] = Ex
        fields["Ey"] = Ey
        bias_points.append({"label": f"Va={Va:+.3f}V", "bias": float(Va), "fields": fields})
    doc = sio.build_structure(
        device="diode2d", material=material_dict, regions=regions,
        x_um=points_um[:, 0], y_um=points_um[:, 1], doping_cm3=mesh.Cdop,
        bias_points=bias_points, dim=2,
        mesh2d={"triangles": mesh.triangles.tolist(), "boundary": boundary},
    )
    structure_file = output_cfg.get("structure_file")
    if structure_file:
        struct_path = os.path.join(out_dir, structure_file)
        sio.write_structure(struct_path, doc)
        print(f"wrote {struct_path} ({len(bias_points)} bias point(s) saved)")


if __name__ == "__main__":
    main()
