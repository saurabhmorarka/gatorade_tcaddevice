"""Where the time of a 3D I-V goes: the real `fab simulate` path (mesh,
equilibrium, drain ramp, an Id-Vg walk) in ONE process, with the Newton
pieces timed - Jacobian assembly, line-search residuals, linear solves -
and every continuation step's Newton iterations (direct / efield warm-up).

  TCAD_LINSOLVE=petsc_ilu OMP_NUM_THREADS=1 ~/miniconda3/envs/tcad/bin/python -W ignore \\
      -m fab.profile_iv configs/fab_planar.yaml --vds 1.0 --vgs 0.1,0.2,0.3,0.4,0.5,0.6

OMP_NUM_THREADS=1 matches a `fab simulate` worker. --cprofile adds a
cProfile of the walk (top functions by own time).
"""
import argparse
import cProfile
import os
import pstats
import time
from collections import defaultdict

import numpy as np

from core import linsolve
from fab import flow
from fab.simulate import _solver_at, build_device, drain_current
import solver2d.newton_solver_qf_2d as nsq

T = defaultdict(float)
C = defaultdict(int)


def _timed(name, fn):
    def wrap(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] += time.perf_counter() - t
            C[name] += 1
    return wrap


def instrument(dump=None, dump_from=0, dump_max=0):
    if dump:
        os.makedirs(dump, exist_ok=True)
        solve = linsolve.LinearSolver.solve
        seen = [0]

        def dumped(self, J, b, rtol=None):
            # Save the system sequence (values only after the first: one
            # pattern) for offline linear-solver comparisons.
            k = seen[0] - dump_from
            seen[0] += 1
            if 0 <= k < dump_max:
                A = J.tocsr()
                A.sort_indices()
                extra = dict(indptr=A.indptr, indices=A.indices) if k == 0 else {}
                np.savez(os.path.join(dump, f"sys{k:03d}.npz"), data=A.data, b=b,
                         rtol=-1.0 if rtol is None else rtol, **extra)
            return solve(self, J, b, rtol)
        linsolve.LinearSolver.solve = dumped     # wraps the timed solve: dumping is not timed
        return
    nsq._residual_and_jacobian = _timed("assemble F+J", nsq._residual_and_jacobian)
    nsq._residual = _timed("residual (line search)", nsq._residual)
    linsolve.LinearSolver.solve = _timed("linear solve", linsolve.LinearSolver.solve)


def report(title, wall):
    print(f"--- {title}: {wall:.1f} s wall")
    for k in ("assemble F+J", "residual (line search)", "linear solve"):
        print(f"    {k:24s} {T[k]:7.1f} s  ({C[k]} calls, {1e3 * T[k] / max(C[k], 1):.0f} ms each)")
    rest = wall - sum(T[k] for k in ("assemble F+J", "residual (line search)", "linear solve"))
    print(f"    {'other':24s} {rest:7.1f} s")
    lin = nsq.default_linear_solver(MESH[0]) if MESH else None
    if lin is not None:
        print("    " + lin.summary())
    T.clear()
    C.clear()


MESH = []


def stepper(solve_at, log):
    """Wrap a continuation solve_at to log each step."""
    def at(p, guess):
        t = time.perf_counter()
        r = solve_at(p, guess)
        log.append((p, r["iters"], r.get("iters_warmup", 0), bool(r["converged"]), time.perf_counter() - t))
        return r
    return at


def print_steps(log):
    for p, it, wu, ok, dt in log:
        print(f"      bias {p:+.4f}: {it:3d} its (warm-up/direct-fail {wu:3d}) {'ok ' if ok else 'FAIL'} {dt:6.1f} s")
    its = sum(s[1] for s in log)
    print(f"    {len(log)} steps, {its} Newton iterations, {sum(s[2] for s in log)} in failed-direct + warm-up")
    log.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--mesher", default="octree")
    ap.add_argument("--vds", type=float, default=1.0)
    ap.add_argument("--vgs", default="0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--vsat", default="true", help="true | false | efield | hybrid | gradqf")
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--dump", help="directory: save the linear systems of the Id-Vg walk")
    ap.add_argument("--dump-from", type=int, default=0, help="first walk system to save")
    ap.add_argument("--dump-max", type=int, default=40)
    ap.add_argument("--ohmic", action="store_true", help="ohmic S/D contacts instead of the Schottky default")
    a = ap.parse_args()
    from main2d_mosfet_sweep import continuation, solve_mosfet
    from mos.mos_params import MOSDevice
    vsat = {"true": True, "false": False}.get(a.vsat, a.vsat)

    t = time.perf_counter()
    d = build_device(flow.load(a.config), mesher=a.mesher, log=lambda *x: None)
    MESH.append(d["mesh"])
    d["dev"] = MOSDevice(gate_workfunction_eV=d["wf"])
    import fab.simulate as fs
    d["schottky"] = None if a.ohmic else fs.schottky_contacts(d)
    print(f"mesh: {len(d['mesh'].points)} nodes, {len(d['mesh'].edges)} edges, {time.perf_counter() - t:.1f} s")
    instrument()
    t = time.perf_counter()
    eq = solve_mosfet(d["mesh"], d["mat"], d["dev"], d["Cs"], 0.0, 0.0, velocity_saturation=vsat,
                      schottky=d["schottky"])
    report(f"equilibrium ({eq['iters']} its, warm-up {eq.get('iters_warmup', 0)})", time.perf_counter() - t)

    import fab.simulate as fs
    if vsat is not True:                          # _solver_at hard-codes the default force
        def _at(dd, Vgs, Vds):
            return lambda pp, g: solve_mosfet(dd["mesh"], dd["mat"], dd["dev"], dd["Cs"],
                                              *((pp, Vds) if Vgs is None else (Vgs, pp)), psi_init=g["psi"],
                                              phin_init=g["phin"], phip_init=g["phip"], cold_retry=False,
                                              velocity_saturation=vsat, schottky=dd.get("schottky"))
        fs._solver_at = _at
    steps = []
    t = time.perf_counter()
    ramp, _, _ = continuation(stepper(fs._solver_at(d, 0.0, None), steps), 0.0, eq, [a.vds])
    report(f"drain ramp to {a.vds} V", time.perf_counter() - t)
    print_steps(steps)

    vgs = [float(v) for v in a.vgs.split(",")]
    pr = cProfile.Profile() if a.cprofile else None
    t = time.perf_counter()
    if pr:
        pr.enable()
    if a.dump:
        instrument(a.dump, a.dump_from, a.dump_max)
    states, _, _ = continuation(stepper(fs._solver_at(d, None, a.vds), steps), 0.0, ramp[a.vds], vgs, **fs.GATE_WALK)
    if pr:
        pr.disable()
    report(f"Id-Vg walk at Vds={a.vds}", time.perf_counter() - t)
    print_steps(steps)
    for v in sorted(states):
        print(f"    Vgs={v:.2f}: Id = {drain_current(d, states[v]):.6e} A")
    if pr:
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
