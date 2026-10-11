"""Octree vs rectilinear accuracy on a fab device: build the device mesh
with each named setting, solve equilibrium, then Id at (Vg, Vd) = (0, 0.05)
and (0.6, 0.05), and print nodes, timings and currents.

  TCAD_LINSOLVE=petsc_ilu ~/miniconda3/envs/tcad/bin/python -W ignore \\
      mesh3d/studies/mesher_convergence.py configs/fab_planar.yaml "oct default,cart default"

Settings are `grid` options of fab.simulate.build_device (octree: h_fine,
h_fine_other, h_max, growth; cartesian: h_fine, h_max, growth). Reference
numbers (full planar device, 2026-10-11) are in DEVELOPMENT_LOG 46 and
plans/3d_handoff.md.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fab import flow                                          # noqa: E402
from fab.simulate import build_device, drain_current          # noqa: E402
from main2d_mosfet_sweep import solve_mosfet                  # noqa: E402
from mos.mos_params import MOSDevice                          # noqa: E402
import numpy as np                                            # noqa: E402
import mesh3d.octree as octree                                # noqa: E402

_target_spacing = octree.target_spacing


def capped(kinds_capped, h_cap):
    """A target_spacing that also caps the spacing at h_cap (nm, per axis)
    in voxels of the given kinds - to localize which region's coarse
    leaves drive an error."""
    def f(label, kinds, cell_net, h, *a):
        tgt = _target_spacing(label, kinds, cell_net, h, *a)
        sel = np.isin(np.array(list(kinds) + [octree.GAS])[label], kinds_capped)
        for ax in range(3):
            tgt[ax][sel] = np.minimum(tgt[ax][sel], max(h_cap[ax] / h[ax], 1.0))
        return tgt
    return f


# TCAD_VSAT: 0 = off, 1 = on with the edge-field force, or a force name (efield | hybrid | gradqf)
VSAT = {"0": False, "1": True}.get(os.environ.get("TCAD_VSAT", "1"), os.environ.get("TCAD_VSAT"))

RUNS = {
    "cart default": ("cartesian", {}),
    "cart fine": ("cartesian", dict(h_fine=(1.0, 0.5, 1.0), h_max=(4.0, 4.0, 4.0))),
    "oct default": ("octree", {}),
    "oct mid": ("octree", dict(h_fine=(1.0, 0.5, 1.0), h_fine_other=(2.0, 2.0, 2.0), h_max=(8.0, 8.0, 8.0))),
    "oct mid2": ("octree", dict(h_fine=(1.0, 0.5, 1.0), h_fine_other=(2.0, 2.0, 2.0), h_max=(6.0, 6.0, 6.0),
                                growth=1.25)),
    "oct fine": ("octree", dict(h_fine=(1.0, 0.5, 1.0), h_fine_other=(1.0, 1.0, 1.0), h_max=(4.0, 4.0, 4.0))),
    # one change at a time from "oct default" (h_fine (2, .5, 2), h_fine_other 2, h_max 8, growth 1.35)
    "oct hmax4": ("octree", dict(h_max=(4.0, 4.0, 4.0))),
    "oct hmax4 x": ("octree", dict(h_max=(4.0, 8.0, 8.0))),
    "oct hmax4 y": ("octree", dict(h_max=(8.0, 4.0, 8.0))),
    "oct hmax4 z": ("octree", dict(h_max=(8.0, 8.0, 4.0))),
    "oct other1": ("octree", dict(h_fine_other=(1.0, 1.0, 1.0))),
    "oct other1 y": ("octree", dict(h_fine_other=(2.0, 1.0, 2.0))),
    "oct fine xz": ("octree", dict(h_fine=(1.0, 0.5, 1.0))),
    # h_max 4 (x, y) only in one kind of material (default elsewhere)
    "oct hmax4 semi": ("octree", {}, ([octree.SEMI], (4.0, 4.0, 8.0))),
    "oct hmax4 ins": ("octree", {}, ([octree.INSULATOR], (4.0, 4.0, 8.0))),
    # default sizing, only the root tiling (voxels) changed
    "oct root 8-16-8": ("octree", dict(root=[8, 16, 8])),
    "oct root 16-16-16": ("octree", dict(root=[16, 16, 16])),
    "oct root 12-24-12": ("octree", dict(root=[12, 24, 12])),
}


def main(cfg_path, names):
    cfg = flow.load(cfg_path)
    for name in names:
        mesher, g, *cap = RUNS[name]
        octree.target_spacing = capped(*cap[0]) if cap else _target_spacing
        try:
            t = time.time()
            d = build_device(cfg, mesher=mesher, grid=g, log=lambda *a: None)
            tb = time.time() - t
            dev = MOSDevice(gate_workfunction_eV=d["wf"])
            t = time.time()
            r = solve_mosfet(d["mesh"], d["mat"], dev, d["Cs"], 0.0, 0.0, velocity_saturation=VSAT)
            current = []
            for vg, vd in ((0.0, 0.05), (0.6, 0.05)):
                r = solve_mosfet(d["mesh"], d["mat"], dev, d["Cs"], vg, vd, psi_init=r["psi"], phin_init=r["phin"],
                                 phip_init=r["phip"], velocity_saturation=VSAT)
                current.append(drain_current(d, r))
            print(f"{name}: {len(d['mesh'].points)} nodes, build {tb:.1f}s solve {time.time() - t:.1f}s, "
                  f"Id(Vg=0)={current[0]:.4e} Id(Vg=0.6)={current[1]:.4e} A", flush=True)
        except Exception as ex:                                # report and go on to the next setting
            print(f"{name}: FAILED {str(ex)[:300]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else ["oct default", "cart default"])
