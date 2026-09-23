"""2D MOS capacitor C-V driver: build a mesh with a mesa oxide/gate stack
(mesh2d/geometry2d.py::TopMesa) sitting on top of a light p substrate,
solve the low-frequency (quasi-static) C-V sweep (solver2d/poisson2d_mos.py
+ solver2d/mos_charge2d.py), and compare directly against a matching 1D MOS
capacitor (mos/mos_solver.py) built with the same substrate doping, oxide
thickness/permittivity, and gate work function.

Usage: python3 main2d_mos_sweep.py [configs/input_mos_2d.yaml]
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

import yaml

from core import structure_io as sio
from core.config import _resolve_material_block
from core.field_save import resolve_save_points
from core.mesh import build_mos_grid
from mesh2d.config2d import build_domain_from_config
from mesh2d.mesh2d import build_mesh2d
from mos.mos_params import MOSDevice
from mos.mos_solver import cv_sweep as cv_sweep_1d
from solver2d.efield2d import electric_field_2d
from solver2d.mos_charge2d import cv_sweep_2d
from viz2d.plot2d import plot_field2d

_CM_TO_UM = 1.0e4
_UM_TO_CM = 1.0e-4

DEFAULT_PATH = os.path.join("configs", "input_mos_2d.yaml")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_from_config(cfg):
    domain = build_domain_from_config(cfg)
    material = _resolve_material_block(cfg.get("material") or {})

    substrate = cfg["geometry"]["substrate"]
    Na = float(substrate["concentration_cm3"])
    Cdop_substrate = -Na if substrate["doping_type"] == "p" else Na

    oxide_region = next(r for r in domain.regions if r.kind == "insulator")
    t_ox_cm = oxide_region.y_range_cm[1] - oxide_region.y_range_cm[0]
    dev = MOSDevice(Na=Na, t_ox=t_ox_cm, eps_ox_r=oxide_region.eps_r,
                     gate_workfunction_eV=(cfg.get("gate") or {}).get("workfunction_eV"))

    mesh_cfg = cfg.get("mesh") or {}
    mesh_opts = dict(
        h_min_cm=float(mesh_cfg.get("h_min_um", 0.2)) * _UM_TO_CM,
        h_max_cm=float(mesh_cfg.get("h_max_um", 2.0)) * _UM_TO_CM,
        growth=float(mesh_cfg.get("growth", 1.3)),
        min_angle_deg=int(mesh_cfg.get("min_angle_deg", 32)),
    )

    vs = cfg.get("voltage_sweep") or {}
    VG_list = np.linspace(float(vs.get("vg_start_V", -1.0)), float(vs.get("vg_stop_V", 2.0)),
                           int(vs.get("vg_points", 31)))

    output_cfg = cfg.get("output") or {}
    return domain, material, dev, Cdop_substrate, mesh_opts, output_cfg, VG_list


def main():
    parser = argparse.ArgumentParser(description="gatorade_tcaddevice 2D MOS capacitor C-V driver")
    parser.add_argument("config", nargs="?", default=DEFAULT_PATH)
    parser.add_argument("--plot-bias", default="-1.0,0.0,0.6,1.2,2.0",
                         help="Comma-separated VG values (nearest swept match) to save field-map PNGs for")
    args = parser.parse_args()

    cfg = load_config(args.config)
    domain, mat, dev, Cdop_substrate, mesh_opts, output_cfg, VG_list = build_from_config(cfg)

    # Grade tightly from BOTH the oxide/substrate interface (y=0, the
    # default domain.junction_segments() target) AND the mesa's own top
    # surface (the gate contact) - grading from the interface alone left
    # only a single degenerate layer of triangles spanning the oxide's
    # entire 10nm thickness (confirmed: only 2 distinct y-values inside the
    # oxide), since distance-based grading measured from one edge only
    # relaxes almost immediately across a gap this thin, and `triangle`'s
    # area constraint alone doesn't force extra layers in a particular
    # direction - it happily satisfies area with one long, thin, near-
    # degenerate triangle instead. Pulling tight from both surfaces forces
    # several genuine vertical layers across the whole oxide, which is what
    # actually resolves the oxide's own vertical capacitance correctly.
    mesa = domain.top_mesas[0]
    mesa_top_segment = ((mesa.x_range_cm[0], -mesa.height_cm), (mesa.x_range_cm[1], -mesa.height_cm))
    interface_segments = domain.junction_segments() + [mesa_top_segment]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = build_mesh2d(domain, mat=mat, interface_segments=interface_segments, **mesh_opts)
    print(f"mesh: {len(mesh.points)} points, {len(mesh.triangles)} triangles "
          f"({int(np.sum(mesh.is_insulator))} insulator points)")

    out_dir = os.path.join("out", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)

    print("--- 2D low-frequency C-V sweep (continuation, warm-started outward from VG list order) ---")
    t0 = time.perf_counter()
    results_2d_list = cv_sweep_2d(mesh, mat, dev, Cdop_substrate, VG_list, gate_contact_name="gate")
    total_time_2d = time.perf_counter() - t0
    n_failed = sum(1 for r in results_2d_list if r["res_norm"] >= 1e-4)
    print(f"2D sweep: {len(results_2d_list) - n_failed}/{len(results_2d_list)} points converged, "
          f"total {total_time_2d:.3f}s ({total_time_2d / len(results_2d_list):.4f}s/point avg)")

    cv_csv_path = os.path.join(out_dir, "cv_sweep.csv")
    with open(cv_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["VG_V", "Qs_C_cm2", "C_lf_F_cm2", "res_norm", "iters"])
        for r in results_2d_list:
            w.writerow([f"{r['VG']:.4f}", f"{r['Qs']:.6e}", f"{r['C_lf']:.6e}",
                        f"{r['res_norm']:.3e}", r["iters"]])
    print(f"wrote {cv_csv_path} ({len(results_2d_list)} bias point(s))")

    # --- Matching 1D MOS capacitor: same substrate doping, oxide thickness/
    # permittivity, gate work function, material. ---
    print("--- 1D reference C-V sweep (same doping/oxide/material) ---")
    grid = build_mos_grid(mat, dev, Cdop_substrate)
    t0 = time.perf_counter()
    results_1d = cv_sweep_1d(grid["x"], grid["Cdop"], grid["eps_edge"], grid["ni_arr"], mat, dev,
                              Cdop_substrate, VG_list, grid["oxide_index"])
    total_time_1d = time.perf_counter() - t0
    print(f"1D sweep: {len(results_1d)} points, total {total_time_1d:.3f}s "
          f"({total_time_1d / len(results_1d):.4f}s/point avg)")
    print(f"2D/1D total-time ratio: {total_time_2d / total_time_1d:.1f}x "
          f"(2D mesh: {len(mesh.points)} points vs 1D: {len(grid['x'])} nodes)")

    VG_arr = np.array([r["VG"] for r in results_2d_list])
    C2 = np.array([r["C_lf"] for r in results_2d_list])
    C1 = np.array([r["C_lf"] for r in results_1d])
    Cox = dev.eps_ox / dev.t_ox

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(VG_arr, C1 / Cox, "o-", label="1D MOS cap", color="#1f77b4")
    axes[0].plot(VG_arr, C2 / Cox, "s--", label="2D MOS cap (gate-averaged)", color="#d62728")
    axes[0].set_xlabel("VG (V)"); axes[0].set_ylabel("C / Cox")
    axes[0].set_title("Low-frequency C-V"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].semilogy(VG_arr, np.abs(C1) + 1e-30, "o-", label="1D MOS cap", color="#1f77b4")
    axes[1].semilogy(VG_arr, np.abs(C2) + 1e-30, "s--", label="2D MOS cap (gate-averaged)", color="#d62728")
    axes[1].set_xlabel("VG (V)"); axes[1].set_ylabel("|C| (F/cm^2)")
    axes[1].set_title("Low-frequency C-V: semilog"); axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "cv_comparison_1d_vs_2d.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # --- Field maps at requested bias points ---
    points_um = mesh.points * _CM_TO_UM
    plot_bias = [float(v) for v in args.plot_bias.split(",")]
    for target in plot_bias:
        VG = min(VG_arr, key=lambda v: abs(v - target))
        r = next(r for r in results_2d_list if r["VG"] == VG)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_field2d(points_um, mesh.triangles, r["psi"], ax=axes[0],
                     title=f"psi (V), VG={VG:.3f}V", cmap="RdBu_r", label="psi (V)")
        plot_field2d(points_um, mesh.triangles, r["n"], ax=axes[1],
                     title=f"n (cm^-3), VG={VG:.3f}V", log_scale=True, label="n (cm^-3)")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"fields_VG{VG:+.2f}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")

    # --- Vertical cut through the gate center ---
    gate_region = next(r for r in domain.regions if r.kind == "insulator")
    x_center_cm = 0.5 * (gate_region.x_range_cm[0] + gate_region.x_range_cm[1])
    y_cut_cm = np.linspace(gate_region.y_range_cm[0], domain.height_cm, 150)  # oxide top -> substrate bottom
    cut_pts = np.stack([np.full_like(y_cut_cm, x_center_cm), y_cut_cm], axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("coolwarm")
    plot_vgs = [min(VG_arr, key=lambda v: abs(v - t)) for t in plot_bias]
    for i, VG in enumerate(plot_vgs):
        r = next(r for r in results_2d_list if r["VG"] == VG)
        psi_cut = griddata(mesh.points, r["psi"], cut_pts, method="linear")
        color = cmap(i / max(1, len(plot_vgs) - 1))
        ax.plot(y_cut_cm * _CM_TO_UM, psi_cut, label=f"VG={VG:+.2f}V", color=color)
    ax.axvline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("y (um), depth from oxide top (gate at left, substrate to the right)")
    ax.set_ylabel("psi (V)")
    ax.set_title(f"Vertical cut through gate center (x={x_center_cm * _CM_TO_UM:.1f}um)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "vertical_cut_psi.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # --- Structure+fields JSON ---
    save_spec = output_cfg.get("save_bias_points", "all")
    save_idx = resolve_save_points(save_spec, VG_arr)
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
        r = results_2d_list[idx]
        Ex, Ey = electric_field_2d(mesh.points, mesh.triangles, r["psi"])
        fields = {k: r[k] for k in ("psi", "n", "p")}
        fields["Ex"] = Ex
        fields["Ey"] = Ey
        bias_points.append({"label": f"VG={VG_arr[idx]:+.3f}V", "bias": float(VG_arr[idx]), "fields": fields})
    doc = sio.build_structure(
        device="mos2d", material=material_dict, regions=regions,
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
