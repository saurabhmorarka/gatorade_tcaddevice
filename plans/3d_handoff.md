# 3D program: handoff (written 2026-10-04 evening, resume 2026-10-05)

Read this first, then `plans/3d_plan.md` (the full plan and measured
performance) and `DEVELOPMENT_LOG.md` section 34.

## Where things stand

- Branch **`3d-perf-core`**, commit `4e14e7e`. Committed, **not merged, not
  pushed**. The user keeps feature branches local until they decide to push,
  and merges only when a phase is finished.
- **Phase 1 (performance core) is mostly done.** Phase 0 (VTK export) has not
  started.

| run (M3 Pro, clean machine) | before | now |
|---|---|---|
| 2D MOSFET full I-V | 17.5 s | 7.5 s |
| 2D BTBT sweep | 93 s | ~38 s (Rust tracer) |
| 2D diode | 4.0 s | 2.8 s |
| 2D MOS-cap C-V | 14 s | 3.9 s |
| 3D extruded MOSFET (28k nodes, 84k unknowns, 138 points) | hours | **73 s** |

The 3D run matches 2D on every point within 1e-3. The user's target is a full
3D I-V in about 1 minute; at the 102-point benchmark definition this testbed
is about 54 s. That still needs confirming on a real FinFET mesh.

## Environments

- **pip venv:** `tcad1d/.venv` (Python 3.13, pymetis added). Run with
  `.venv/bin/python`. Uses SuperLU+METIS.
- **conda env `tcad`:** `~/miniconda3/envs/tcad/bin/python` (Python 3.12,
  PETSc 3.26 + MUMPS + UMFPACK + pymetis), built from `environment.yml`.
  Needed for 3D speed (PETSc block-ILU).
- Run everything from the `tcad1d/` root.
- The system `python3`/`python` lack the deps (miniconda base has no `triangle`).
- The scratchpad from the 2026-10-04 session (`/private/tmp/claude-501/...`)
  is ephemeral. Don't rely on its benchmark scripts or the `benv` env.

## How to check things

```bash
.venv/bin/python testsuite/test_examples.py                 # 1D, 6 tests, ~6 s
.venv/bin/python testsuite/test_2d_examples.py -v           # 2D, 4 tests, ~70 s
.venv/bin/python testsuite/test_vtk.py -v                   # VTK export, 10 tests, ~3 s
.venv/bin/python testsuite/test_native.py -v                # Rust vs numpy tracer, ~9 s
.venv/bin/python testsuite/test_tcadview.py -v              # viewer, headless, ~3 s
~/miniconda3/envs/tcad/bin/python testsuite/test_2d_examples.py -v
.venv/bin/python main2d_mosfet_sweep.py --out-dir /tmp/x    # 2D timing (~7.5 s)
~/miniconda3/envs/tcad/bin/python main2d_mosfet_sweep.py --extrude-um 0.1 --nz 6 --out-dir /tmp/y   # 3D (~75 s)
```

- 2D drivers write tracked files under `out/` unless given `--out-dir`. If a
  run dirties `out/`, restore it with `git checkout out/`.
- Force a linear-solver backend with
  `TCAD_LINSOLVE=superlu|umfpack|scipy_ilu|petsc_lu|petsc_ilu`.

## Gotchas learned the hard way

1. **Orphaned workers.** Killing a driver's parent does not kill its
   `ProcessPoolExecutor` workers (spawned processes named `...spawn_main...`).
   Six orphans at ~200% CPU silently inflated every timing for ~20 minutes and
   made the user's fans loud.
   - After stopping a run: `pkill -f spawn_main`, then check with
     `ps -Ao pid,pcpu,command -r | head`.
   - Check the machine is idle before trusting any timing.
2. **MPICH busy-polling.** conda-forge PETSc's MPICH burns ~2 cores per idle
   process. `core/linsolve.py` sets `FI_PROVIDER=tcp` before importing
   petsc4py. Keep that line above the import.
3. **M3 Pro is 6 performance + 6 efficiency cores.** Default workers = P-core
   count (`performance_cores()` in `main2d_mosfet_sweep.py`). Splitting curves
   onto all 12 cores was slower.
4. **3D is memory-bandwidth bound.** Wall time is ~75 s for 3, 4, 6 or 8
   workers. More processes don't help; less data per iteration does.
5. **ILU needs tight tolerances.** ILU-GMRES at a loose tolerance gives bad
   Newton directions (quasi-Fermi levels of near-absent carriers). The
   inexact-Newton cap is backend-dependent (`LinearSolver.max_rtol`): 1e-4
   for LU-type preconditioners, 1e-8 for ILU.
6. **The early-exit convergence test must include quasi-Fermi updates**
   (density-weighted, `QF_DENSITY_REF=1 cm^-3`). A psi-only version made one
   off-state point 1000x wrong (SS showed 16.5 mV/dec). Always check
   SS/Vt/Ion against 2D after any convergence change.
7. **Noise floors.** Off-state currents below ~1e-13 A/um are contact-current
   round-off (~4e-17 A/um absolute), even with the old solver and between
   BLAS builds. Golden floors are set from this measurement.
8. **Drivers must test `r["converged"]`, not `res_norm < 1e-4`.** The
   prediction exit can stop with |F|_inf up to 0.1.
9. **scipy's `gmres` reports convergence unreliably here.** Use
   `core.linsolve.gmres_right`.
10. In zsh, `$var` does not word-split in loops (use `for a b in ...`).

## Progress on 2026-10-05

- Done on `3d-perf-core` (`5464ad6`): MOS-cap 14 s -> 3.9 s (bit-identical);
  BTBT ~54 s -> ~46 s (grouped Hurkx quadrature, fixed-pattern generation
  Jacobians, longest-first scheduling). DEVELOPMENT_LOG section 35.
- Done on `3d-vtk-export` (`5981d0b`, branched off `3d-perf-core`): Phase 0
  VTK export, `--vtk` on the 2D drivers, `testsuite/test_vtk.py`.
  DEVELOPMENT_LOG section 36.
- Done on `3d-rust-tracer` (`4ca69dd`, off `3d-vtk-export`): Rust toolchain
  installed (user approved). `native/` crate `tcad_native` (PyO3, maturin)
  with the field-line tracer, 8x on tracing, BTBT sweep 45.5 -> 37.6 s.
  Optional: `core/native.py` falls back to numpy; `TCAD_NATIVE=0` forces it.
  Built into both envs with `maturin develop --release` (unset CONDA_PREFIX
  for the venv build). DEVELOPMENT_LOG section 37.
- Done on `3d-viewer` (`b495361`, off `3d-rust-tracer`): Phase 2 first
  milestone, `python3 -m tcadview <file.pvd>`. Viewer deps installed in both
  envs. DEVELOPMENT_LOG section 38.
- Branch chain (all local, unmerged, unpushed): `3d-perf-core` ->
  `3d-vtk-export` -> `3d-rust-tracer` -> `3d-viewer` -> `3d-viewer-sections`.

## Progress on 2026-10-10

- `3d-viewer-sections` (off `3d-viewer`): section views (2D view of an
  x/y/z slice, with probes and cuts inside the device), H/V cuts through the
  last probe, hold, log-space density interpolation, and a one-state bias
  note. DEVELOPMENT_LOG section 39.
- To see a sweep in tcadview, run the driver with `--vtk all`. Plain `--vtk`
  saves only the last bias point of each curve.

## Next steps (in order)

1. **Phase 2 continued:**
   - user feedback on the section view (`3d-viewer-sections`)
   - linked viewports and difference plots
   - contact highlighting
   - current-density export plus vectors
   - GitHub packaging (pyproject entry point, 3-OS CI, PyInstaller)
2. **BTBT critical path** is now the local curves too (Hurkx Gamma plus
   Newton, ~27 s under contention). Profile again before more Rust.
3. **3D speed levers** (only after a real 3D device exists to measure on):
   - better preconditioner (AMG on the Poisson block / field-split)
   - FP32 preconditioner on the CPU (half the memory traffic; no GPU, see decisions)
   - tangent predictor
4. Then Phase 3+: virtual fab (planar, then FinFET, then nanosheet), octree
   mesher (a Rust candidate), the 3D validation ladder.

Before merging `3d-perf-core`, the user's merge gate applies: goldens pass,
moved values explained, and before/after runtimes reported.

## User decisions to respect (all also in memory)

- Own process engine, no ViennaPS. Level-set/voxel virtual fab, three-layer
  YAML.
- Qt desktop viewer, distributable via GitHub.
- PETSc fast path plus a pure-scipy fallback.
- Gate oxide always meshed (rejected the thin-dielectric BC).
- Runtime is first-class: full 3D I-V in ~1 min; parallelize; report
  honestly if it can't be met.
- Rust where it gives 5-10x (speed, CPU or memory). Profile first.
- GPU only where it cuts TOTAL runtime >3x, kept simple. No custom GPU
  solver/Jacobian stack (PETSc has no Mac GPU backend), so the device solve
  stays on the CPU. Candidate to evaluate: the Phase 3 virtual fab
  (level-set/voxel), profiled end to end first. (Decided 2026-10-05.)
- Future solvers: thermal, stress (FEM on the same tets), dopant diffusion,
  borrowing COMSOL concepts. Avalanche stays standalone.
- Show the plan before big implementations. Feature branch per phase. Update
  `DEVELOPMENT_LOG.md` with every commit. Confirm before pushing.
