"""Quick 3D regression test: the tiny planar NMOS from the virtual fab
(configs/fab_planar_tiny.yaml), octree mesh, default physics (hybrid velocity
saturation), solved at a handful of bias points through the same continuation
driver the full Id-Vg sweeps use. It is compared against
testsuite/golden/3d_planar_tiny.json.

What it guards, end to end: the fab build and its structure check, the
octree mesher and its mandatory gate (exact volumes, Voronoi facets, no stray
metal volume), the metal-volume contacts, and the 3D drift-diffusion solve.

Bias points, deliberately few so the test runs as fast as the 2D ones
(the full curves are `python -m fab simulate`):
  off   Vg = 0,   Vd = 0.05 V
  lin   Vg = 0.6, Vd = 0.05 V
  sat   Vg = 0.6, Vd = 0.6 V
Equilibrium is solved first; each point is reached by continuation from the
previous one.

Tolerance: rtol 2e-3, the drift between linear-solver backends (SuperLU in
the venv, PETSc ILU/LU in conda) on converged states; currents below `FLOOR`
are only checked against it.

Run with:      python3 testsuite/test_3d_examples.py [-v]
Capture with:  python3 testsuite/test_3d_examples.py --capture
(capture only after independently verifying the new numbers - see README.md)
"""
import json
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GOLDEN = os.path.join(ROOT, "testsuite", "golden", "3d_planar_tiny.json")
CONFIG = os.path.join(ROOT, "configs", "fab_planar_tiny.yaml")
RTOL = 2e-3
FLOOR = 1e-13          # A


def run():
    """{name: value} for the tiny planar device, and the wall time."""
    from fab import flow
    from fab.simulate import build_device, drain_current
    from main2d_mosfet_sweep import continuation, solve_mosfet
    from mos.mos_params import MOSDevice
    t0 = time.perf_counter()
    d = build_device(flow.load(CONFIG), mesher="octree", log=lambda *a: None)
    mesh, mat, Cs = d["mesh"], d["mat"], d["Cs"]
    dev = MOSDevice(gate_workfunction_eV=d["wf"])

    def at(Vgs, Vds):
        return lambda p, g: solve_mosfet(mesh, mat, dev, Cs, *((p, Vds) if Vgs is None else (Vgs, p)),
                                         psi_init=g["psi"], phin_init=g["phin"], phip_init=g["phip"],
                                         cold_retry=False, velocity_saturation=True)
    out = {"nodes": float(len(mesh.points)), "stray_metal_nodes": float(mesh.quality["stray_metal_nodes"])}
    eq = solve_mosfet(mesh, mat, dev, Cs, 0.0, 0.0, velocity_saturation=True)
    off, _, _ = continuation(at(0.0, None), 0.0, eq, [0.05])
    lin, _, _ = continuation(at(None, 0.05), 0.0, off[0.05], [0.6])
    sat, _, _ = continuation(at(0.6, None), 0.05, lin[0.6], [0.6])
    for name, states, key in (("Id_off_A", off, 0.05), ("Id_lin_A", lin, 0.6), ("Id_sat_A", sat, 0.6)):
        if key in states:
            out[name] = drain_current(d, states[key])
    return out, time.perf_counter() - t0


def compare(golden, new):
    bad = []
    for k, g in golden.items():
        if k not in new:
            bad.append(f"{k}: missing (point did not converge?)")
            continue
        x = new[k]
        if k in ("nodes", "stray_metal_nodes"):
            if x != g:
                bad.append(f"{k}: golden {g:g} new {x:g}")
            continue
        if abs(g) < FLOOR and abs(x) < FLOOR:
            continue
        if abs(x - g) > RTOL * max(abs(g), FLOOR):
            bad.append(f"{k}: golden {g:.6e} new {x:.6e} (rel {abs(x - g) / max(abs(g), FLOOR):.2e} > {RTOL:g})")
    return bad


class Test3DExamples(unittest.TestCase):
    def test_planar_tiny(self):
        if not os.path.exists(GOLDEN):
            self.skipTest(f"no golden file {GOLDEN} (run with --capture)")
        with open(GOLDEN) as f:
            golden = json.load(f)
        new, wall = run()
        print(f"\n  planar_tiny (3D octree): {wall:.1f}s wall-clock", end="")
        bad = compare(golden, new)
        self.assertFalse(bad, f"{len(bad)} mismatch(es):\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    if "--capture" in sys.argv:
        new, wall = run()
        with open(GOLDEN, "w") as f:
            json.dump(new, f, indent=1, sort_keys=True)
        print(f"captured {GOLDEN} ({wall:.1f}s): {new}")
    else:
        unittest.main()
