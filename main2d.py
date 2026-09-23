"""Phase-1 2D driver: load a 2D input YAML, build the point-cloud mesh +
structure, save it via core/structure_io.py (dim=2), and render it with
viz2d - structure/mesh only, no solve. See main2d_sweep.py for the actual
2D bias-sweep driver (solver2d/newton_solver_qf_2d.py).

Usage: python3 main2d.py [configs/input_diode_2d.yaml] [--interactive]
"""
import argparse
import os
import sys

import matplotlib
if "--interactive" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import structure_io as sio
from mesh2d import config2d
from mesh2d.mesh2d import build_diode2d_mesh
from viz2d.plot2d import plot_structure2d

_CM_TO_UM = 1.0e4


def build_structure_doc(domain, mesh, material):
    x_um = mesh.points[:, 0] * _CM_TO_UM
    y_um = mesh.points[:, 1] * _CM_TO_UM

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

    material_dict = dict(eps_r=material.eps_r, ni=material.ni, mu_n=material.mu_n,
                          mu_p=material.mu_p, tau_n=material.tau_n, tau_p=material.tau_p,
                          chi_eV=material.chi_eV, Eg_eV=material.Eg_eV)

    return sio.build_structure(
        device="diode2d",
        material=material_dict,
        regions=regions,
        x_um=x_um,
        y_um=y_um,
        doping_cm3=mesh.Cdop,
        bias_points=[],
        dim=2,
        mesh2d={"triangles": mesh.triangles.tolist(), "boundary": boundary},
    )


def main():
    parser = argparse.ArgumentParser(description="gatorade_tcaddevice 2D structure/mesh driver (phase 1: no solver yet)")
    parser.add_argument("config", nargs="?", default=config2d.DEFAULT_PATH)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = config2d.load_config(args.config)
    domain, material, mesh_opts, output_cfg, _Va_list = config2d.build_from_config(cfg)
    mesh = build_diode2d_mesh(domain, **mesh_opts)

    print(f"points: {len(mesh.points)}, triangles: {len(mesh.triangles)}, "
          f"internal edges: {len(mesh.edges)}, boundary points: {len(mesh.boundary_point_index)}")
    if mesh.n_negative_subareas:
        print(f"WARNING: {mesh.n_negative_subareas} negative box-method sub-areas "
              f"(obtuse-triangle mesh-quality issue) - check mesh2d/pointcloud.py spacing")
    if mesh.n_floored:
        print(f"WARNING: {mesh.n_floored} points had their control-volume area floored "
              f"(see mesh2d/fvgeometry.py's cv_area_floor) - a second, independent "
              f"mesh-quality issue from the box method near quadtree T-junctions")

    doc = build_structure_doc(domain, mesh, material)

    structure_file = output_cfg.get("structure_file")
    out_dir = os.path.join("out", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    if structure_file:
        out_path = os.path.join(out_dir, structure_file)
        sio.write_structure(out_path, doc)
        print(f"wrote {out_path}")

    ax, layers = plot_structure2d(doc)
    if args.interactive:
        from viz2d.plot2d import _interactive_show
        _interactive_show(ax, layers)
    else:
        out_path = os.path.join(out_dir, "structure.png")
        ax.figure.savefig(out_path, dpi=150)
        plt.close(ax.figure)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
