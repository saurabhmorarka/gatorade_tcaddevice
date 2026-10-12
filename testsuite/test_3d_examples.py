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

It also checks the exported current-density vectors: their integral over
planes across the channel must match the solver's exact plane current
(sum of crossing edge currents) within 2% (measured: within 0.7%).

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
VECTOR_RTOL = 0.02     # exported J vectors vs the exact plane current (measured <= 0.7%)


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
    if 0.6 in lin:
        out["_vector_flux_check"] = vector_flux_check(d, lin[0.6])
    return out, time.perf_counter() - t0


def vector_flux_check(d, r):
    """Worst relative difference, over planes across the channel, between
    the solver's exact current through the plane (sum of the edge currents
    crossing it) and the integral of the exported TotalCurrentDensity
    vectors sampled on a fine grid in the plane - through the octree VTK
    writer (half-width mirroring included). Checks that the arrows the
    viewer draws carry the solver's conserved current."""
    import tempfile
    import numpy as np
    import pyvista as pv
    from fab.simulate import write_state_octree
    from solver2d.newton_solver_qf_2d import edge_currents
    m = d["mesh"]
    In, Ip = edge_currents(m, d["mat"], r["psi"], r["n"], r["p"], r["phin"], r["phip"], r["velocity_saturation"])
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "s.vtu")
        write_state_octree(f, d, r, {}, "")
        g = pv.read(f)
    b = g.bounds
    P = m.points * 1e4                                                  # um
    xi, xj = P[m.edges[:, 0], 0], P[m.edges[:, 1], 0]
    worst = 0.0
    for fx in (0.41, 0.47, 0.53, 0.59):                                  # between node planes
        xp = b[0] + fx * (b[1] - b[0])
        I_edge = 2.0 * ((In + Ip)[(xi < xp) & (xj > xp)].sum() - (In + Ip)[(xj < xp) & (xi > xp)].sum())
        ys, zs = np.linspace(b[2], b[3], 1201), np.linspace(b[4], b[5], 201)
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        s = pv.PolyData(np.column_stack([np.full(Y.size, xp), Y.ravel(), Z.ravel()])).sample(g)
        Jx = np.nan_to_num(s.point_data["TotalCurrentDensity"][:, 0].reshape(Y.shape))
        I_vec = np.trapezoid(np.trapezoid(Jx, zs, axis=1), ys) * 1e-8    # um^2 -> cm^2
        worst = max(worst, abs(I_vec / I_edge - 1.0))
    return worst


def compare(golden, new):
    bad = []
    if new.get("_vector_flux_check", 0.0) > VECTOR_RTOL:
        bad.append(f"current-density vectors integrate to the plane current only within "
                   f"{new['_vector_flux_check']:.2%} (> {VECTOR_RTOL:.0%})")
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
        new = {k: v for k, v in new.items() if not k.startswith("_")}
        with open(GOLDEN, "w") as f:
            json.dump(new, f, indent=1, sort_keys=True)
        print(f"captured {GOLDEN} ({wall:.1f}s): {new}")
    else:
        unittest.main()
