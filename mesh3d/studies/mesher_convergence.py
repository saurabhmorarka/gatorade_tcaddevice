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

RUNS = {
    "cart default": ("cartesian", {}),
    "cart fine": ("cartesian", dict(h_fine=(1.0, 0.5, 1.0), h_max=(4.0, 4.0, 4.0))),
    "oct default": ("octree", {}),
    "oct mid": ("octree", dict(h_fine=(1.0, 0.5, 1.0), h_fine_other=(2.0, 2.0, 2.0), h_max=(8.0, 8.0, 8.0))),
    "oct mid2": ("octree", dict(h_fine=(1.0, 0.5, 1.0), h_fine_other=(2.0, 2.0, 2.0), h_max=(6.0, 6.0, 6.0),
                                growth=1.25)),
    "oct fine": ("octree", dict(h_fine=(1.0, 0.5, 1.0), h_fine_other=(1.0, 1.0, 1.0), h_max=(4.0, 4.0, 4.0))),
}


def main(cfg_path, names):
    cfg = flow.load(cfg_path)
    for name in names:
        mesher, g = RUNS[name]
        try:
            t = time.time()
            d = build_device(cfg, mesher=mesher, grid=g, log=lambda *a: None)
            tb = time.time() - t
            dev = MOSDevice(gate_workfunction_eV=d["wf"])
            t = time.time()
            r = solve_mosfet(d["mesh"], d["mat"], dev, d["Cs"], 0.0, 0.0, velocity_saturation=True)
            current = []
            for vg, vd in ((0.0, 0.05), (0.6, 0.05)):
                r = solve_mosfet(d["mesh"], d["mat"], dev, d["Cs"], vg, vd, psi_init=r["psi"], phin_init=r["phin"],
                                 phip_init=r["phip"], velocity_saturation=True)
                current.append(drain_current(d, r))
            print(f"{name}: {len(d['mesh'].points)} nodes, build {tb:.1f}s solve {time.time() - t:.1f}s, "
                  f"Id(Vg=0)={current[0]:.4e} Id(Vg=0.6)={current[1]:.4e} A", flush=True)
        except Exception as ex:                                # report and go on to the next setting
            print(f"{name}: FAILED {str(ex)[:300]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else ["oct default", "cart default"])
