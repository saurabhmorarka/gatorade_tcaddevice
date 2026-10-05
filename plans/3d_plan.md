# 3D Device Simulation for tcad1d: Structure, Mesh/Solve, Visualization

## Context

tcad1d has a working 2D stack: a balanced-quadtree mesher (`mesh2d/quadtree.py`),
a Voronoi-box finite-volume geometry (`mesh2d/fvgeometry.py`), a fully
coupled quasi-Fermi Newton solver (`solver2d/newton_solver_qf_2d.py`), and a
Plotly viewer (`viz2d/plot2d_web.py`). Going to 3D needs three new
capabilities, each a project of its own:

1. **Structure creator**: a "virtual fab" that builds planar → FinFET →
   nanosheet (→ 2D-material) devices from process steps, with full 3D
   visualization, saved as a YAML recipe that recreates or re-parameterizes
   the device.
2. **3D meshing and solving**: an octree mesh with interface-focused
   refinement, box-FV geometry in 3D, and the same drift-diffusion
   equations as 2D, **fast**.
3. **A TCAD-grade visualizer**: a desktop Qt app with the slicing,
   lin/log, multi-dataset linking and probing of Tecplot / Sentaurus Visual /
   TonyPlot.

Decisions made by the user (2026-10-04):
- Build our own process engine; do **not** adopt ViennaPS.
- Viewer is a **Qt desktop app**, and it must be distributable via GitHub.
- **Runtime is the first-class requirement**: the 3D full I-V should finish in
  **~1 minute** on this Mac (M-series, 12 cores, 18 GB). Use any algorithm or
  hardware that helps, and parallelize everything possible. Keep a CPU-only
  path for non-Mac/no-GPU users. If 1 minute is not achievable, say so
  with numbers.
- Analytic doping profiles for now. Process-based diffusion comes later.
- Future solvers to plan for: **thermal, stress, dopant diffusion**, borrowing
  COMSOL-style multiphysics concepts.

## Shared contract

All three tools exchange the same two things:

- **YAML recipe**: structure, doping, contacts, domain cut, mesh spec.
- **VTK files** (`.vtu` per state, `.pvd` collections for bias sweeps and
  process steps), with one fixed naming convention for regions, materials,
  interfaces and contacts.

ParaView (installed: ParaView 6.0) opens every intermediate result from day
one, so it serves as the reference viewer for checking our own.

---

## Part 1: Structure creator (virtual fab)

### Engine: level sets on a voxel grid, one geometric operation per step

Each material is a signed-distance field (SDF) on a uniform voxel grid (per-axis
spacing). Each process step is a single geometric operation, not a
time-stepped PDE. That keeps every step down to seconds.

| Step | Operation | Result |
|---|---|---|
| `pattern` | 2D polygon mask, with coordinates as parameter expressions | Fin, gate, contact masks |
| `deposit: conformal` | Distance transform from the exposed surface, fill where distance ≤ t | Spacer liner wrapping the gate corners at the correct radius |
| `deposit: directional` / `fill` / `planarize` | Column fill, fill to a height, CMP | STI, ILD, replacement metal gate |
| `etch: anisotropic` | Column-wise removal along the etch direction, by depth or stop-on-material | Classic spacer shape (conformal dep + anisotropic etch), S/D recess |
| `etch: isotropic, selective` | Distance-based removal of one material only | SiGe lateral recess (inner spacer), channel release |
| `epitaxy` | Grows only from exposed Si/SiGe; dilation by a Wulff shape whose slow {111} facets give the diamond profile | Faceted S/D epi, merging between fins, growth from both Si and SiGe when there is no inner spacer |

Sub-voxel accuracy comes from the SDFs. A nanosheet half-cell at 0.5 nm
voxels is about 4M voxels.

The engine sits behind a narrow interface, so better physics (e.g.
flux-dependent etch) can be swapped in later.

### Three-layer YAML

```yaml
# Layer 1 - template + parameters (what users normally edit)
template: nanosheet
params:
  Lg: 12nm
  n_sheets: 3
  T_sheet: 5nm
  T_sus: 10nm            # SiGe sacrificial / suspension thickness
  W_sheet: 30nm
  gate_height: 40nm
  spacer_thk: 6nm
  inner_spacer: {enabled: true, recess: 4nm, material: SiN}
  sd_epi: {material: SiGe0.3, faceted: true, overgrowth: 8nm}
  substrate_thk: 60nm

# Layer 2 - explicit flow, generated from layer 1, editable for custom flows
process:
  - {step: substrate, material: Si}
  - {step: epi_stack, layers: [[SiGe, T_sus], [Si, T_sheet]], repeat: n_sheets}
  - {step: pattern, name: fin_mask, polygon: rect(-W_sheet/2, W_sheet/2, ...)}
  - {step: etch, mode: anisotropic, mask: fin_mask, depth: ...}
  - {step: fill, material: SiO2, to: ...}            # STI
  - {step: deposit, mode: conformal, material: SiN, thickness: spacer_thk}
  - {step: etch, mode: anisotropic, material: SiN, stop_on: [Si, poly]}
  - ...
  - {step: epitaxy, material: SiGe0.3, from: [Si], facets: {111: 0.1, 100: 1.0}, thickness: ...}

# Layer 3 - device definition
doping:
  - {region: sd_epi, species: B, profile: constant, conc: 3e20}
  - {from_region: sd_epi, species: B, profile: erfc, into: [Si], length: 3nm}   # follows the curved epi surface
  - {region: substrate, species: P, profile: constant, conc: 1e17}
contacts:
  - {name: gate, on: metal_gate}
  - {name: source, on: sd_contact_left}
domain:
  cut: half_y          # full | half_x | half_y | quarter | single_sheet
mesh:
  preset: fast          # fast | standard | accurate
```

- Doping profiles are functions of the **signed distance** from a region or
  mask edge. Junctions therefore follow curved epi naturally, at no extra cost.
- Domain-cut planes are always tagged `symmetry`, never `free_surface`.
- Template order: planar → FinFET → nanosheet → (2D-material: monolayer films
  need sub-voxel "sheet" handling, so it comes later).

**User interaction:** the user describes the device in plain language →
Claude edits the YAML → the tool rebuilds it → the viewer's
**process-step slider** scrubs through the build, so a wrong step is visible.

---

## Part 2: Meshing and solving

### Mesh

- **Octree** (3D analog of `quadtree.py`), 2:1 balanced. Refinement is driven by
  Si/oxide interfaces, junctions (doping sign change and gradient), the channel
  and refine boxes. The planes of axis-aligned features are snapped exactly
  using the integer-unit trick from 2D.
- Curved, non-critical interfaces (outer spacer, epi facets) are **staircased**
  in v1. The gate-oxide/channel interfaces stay exact. Tapered fins also
  staircase. Body-fitted curved interfaces are a later phase.
- **Box-FV in 3D:** each tetrahedron adds its share of the Voronoi facet area
  and control volume to the edges and nodes it touches, exactly as each
  triangle does in 2D. Control volumes are split per material at interfaces.
  - A tet's circumcenter can fall outside it, which is harmless inside one
    material (Delaunay sums stay positive) but not across an interface.
  - Mesher rule: **no octree level change within one cell of a material
    interface or boundary.**
- Anisotropy matters for speed. A 1 nm oxide must not force 1 nm cubes
  in all directions. **Anisotropic box refinement** is used where Voronoi
  admissibility allows it (to be tested in Phase 5). The gate oxide is
  always meshed: the user rejected a thin-dielectric boundary condition
  (2026-10-04).
- **Mandatory quality gate on every run** (3D version of
  `mesh_quality.py`):
  - summed edge coupling ≥ 0
  - positive control volumes
  - interface conformity
  - no isolated nodes

  A real mesh-generation failure raises a hard error. A known, bounded
  limitation produces a warning on every run.

### Solver

`newton_solver_qf_2d.py` already works on an edge list plus weights plus volumes, so the
physics is independent of dimension. It moves into a shared FV solver
core used by 2D and 3D, so 2D work and 3D work stay a single code path.

---

## Part 2b: Performance, the 1-minute target

### Measured baseline (2026-10-04, this Mac)

**2D planar NMOS full I-V** (2 Id-Vg + 4 Id-Vd curves, 138 points):
- 17.5 s wall, 6 curves running in parallel processes.
- Mesh: 4,680 nodes, i.e. 14,040 unknowns.
- About 7 Newton iterations per bias point.
- Profile of one curve: **73% of the time is SuperLU** (`gssv`, about 60 ms per solve).
  Jacobian assembly is about 10%, and sparse format conversions (sorting, COO→CSC)
  take another about 10%.
- On the real Jacobian, SuperLU's default COLAMD ordering is already its
  best: factor 56 ms, but **solve with an existing factorization takes 1.3 ms
  (about 40× cheaper)**.

**SuperLU on 3D-sized FV matrices** (synthetic 7-point stencil, 3 coupled unknowns per node):

| nodes | unknowns | factor | memory |
|---|---|---|---|
| 4.9k | 15k | 0.6 s | 0.15 GB |
| 25k | 74k | **19.8 s** | 1.9 GB |
| 43k | 130k | **82 s** | 4.8 GB |

**Real 3D drift-diffusion Jacobians (2026-10-04 spike):** the 2D MOSFET's box-FV
mesh was extruded in z, which gives prisms, and an extruded Voronoi mesh is an
exact 3D box mesh. The **unchanged** `newton_solve_2d` was run on it.
- **Correctness:** Id/W matches 2D to **1.8e-12** (Vg=1 V, Vd=0.05 V). The
  solver core is already dimension-agnostic.
- **Solver benchmarks** on the real Jacobians at 28,080 nodes (84k unknowns),
  via PETSc 3.26 from conda-forge:

| Linear solve per Newton iteration | time |
|---|---|
| SuperLU (current code) | ~6.7 s |
| MUMPS LU, 1 / 2 / 4 / 8 threads | 1.09 / 0.88 / 0.76 / 0.74 s |
| **GMRES + ILU(1), RCM ordering** | **0.33 s** (67 its, residual 5e-11) |
| GMRES + ILU(2) / ILU(4) | 0.50 s / 1.50 s |
| GMRES preconditioned with the previous iterate's LU | 0.3–1.0 s (13–51 its) |

  For comparison in **2D** (14k unknowns), ILU-GMRES (55 ms) is about equal to SuperLU
  (56 ms). The direct-vs-iterative crossover happens on the way to 3D.
- Caveat: PETSc's superlu_dist crashed (SEGV) at 4 OpenMP threads, so
  threaded backends need per-backend testing.

**Conclusion:** SuperLU, today's solver, cannot be used in 3D. A ~100-point I-V
with ~5 factorizations per point would take hours. The 3D speed plan therefore
replaces the linear-solver stack and cuts the number of factorizations.

### Hardware facts that shape the plan

- **Apple-silicon GPUs have no FP64** (Metal has no double precision).
  Drift-diffusion needs FP64: densities span about 40 decades, and scaled
  Jacobians are still conditioned around 1e8–1e12, so FP32/mixed precision
  breaks down. **On the Mac, the core solve runs on the CPU** (12 cores
  plus the AMX matrix units through Accelerate BLAS). On the Mac the GPU is
  used for visualization only.
- Optional **NVIDIA backend** (CUDA FP64, e.g. cuDSS / CuPy) for Linux users
  later, behind the same solver-backend interface. CPU-only remains the
  default everywhere.
- `pypardiso` (Intel MKL) is x86-only and does not run here.

### The speed stack (each item measured on its own)

1. **Fewer factorizations.**
   - **Jacobian reuse:** factor once per bias point, then run
     Newton–Krylov (GMRES) preconditioned with that factorization. Each
     extra iteration costs a solve (40× cheaper), not a factorization.
   - **Tangent (sensitivity) predictor:** dx/dV = −J⁻¹·∂F/∂V, computed with the
     factorization already in hand. Each new bias point then starts nearly
     converged. Goal: 2–3 iterations per point instead of 7.
   - Adaptive bias steps, so easy regions of the curve take big steps.
2. **A better linear solver**, behind one `LinearSolver` interface,
   benchmarked on real Jacobians:
   - multithreaded nested-dissection direct solvers with **symbolic-
     factorization reuse** (the sparsity pattern never changes across
     iterations or bias points): MUMPS, UMFPACK, KLU, Panua PARDISO
     (has arm64 builds)
   - preconditioned Krylov: GMRES with ILU(k)/ILUT plus scaling (the approach
     industry 3D TCAD uses), and algebraic multigrid (pyamg/hypre) on the
     Poisson block, block-preconditioned
   - delivery (decided 2026-10-04): **PETSc** (petsc4py from conda-forge,
     osx-arm64) is the fast path, plus a **pure-scipy fallback** backend so a
     plain `pip install -r requirements.txt` user can still run everything,
     only slower
3. **Fixed-pattern assembly:** the CSR sparsity pattern and scatter maps are
   precomputed once, and each iteration only fills values (no COO→CSC, no index
   sorting). Hot kernels go through numba where NumPy falls short.
4. **Fewer unknowns:**
   - symmetry cuts (half or quarter)
   - Poisson-only in insulators (1 unknown instead of 3)
   - mesh presets that refine only where the current flows (`fast` / `standard` /
     `accurate`)
5. **Parallelism everywhere:**
   - independent curves run in separate processes (already done in 2D)
   - long curves are split into segments seeded from cheap ramps
   - each process uses multithreaded BLAS when fewer curves than cores
     remain
   - mesh building and VTK writing run in parallel with solving

### Feasibility estimate (from the real-Jacobian spike)

Per-iteration cost at 28k nodes is about 0.33 s (ILU-GMRES) plus about 0.06 s
(assembly, extrapolated from the 2D profile), so about 0.4 s.

- **2D:** about 3–5 s for the full I-V (from 17.5 s). The gains come from fewer
  iterations (tangent predictor, factorization reuse) and fixed-pattern assembly,
  not from the linear solver. High confidence.
- **3D planar/FinFET half-domain at ~28k nodes, 102 points:**
  - today's ~7 iterations per point: 102 × 7 × 0.4 s ≈ 290 s serial, ≈ 50–70 s across 6
    parallel curves (with memory-bandwidth contention)
  - with the tangent predictor (~3 iterations per point): **≈ 25–35 s**

  **1 minute is likely achievable.** Risk: ILU may stall at hard bias points
  (high Vd, near Vt). Mitigation: automatic fallback to MUMPS (~1 s/iteration)
  for those steps only.
- **3D full 3-sheet nanosheet (~60–80k nodes):** about 2–3 minutes. Reaching 1 minute
  needs a quarter or single-sheet cut. Will be reported with numbers.

### Phase 1 results so far (2026-10-04, branch `3d-perf-core`)

**What was built:**
- `core/linsolve.py`: a stateful `LinearSolver`.
  - Factorizations are reused across Newton iterations and bias points
    (GMRES preconditioned by the most recent factorization, self-tuning
    refactor rule).
  - Ruiz scaling is done on the CSR values.
  - Backends: METIS-ordered SuperLU (2D default), UMFPACK (opt-in), PETSc
    MUMPS, and PETSc point-block ILU(2)-GMRES (3D default).
- **2D Newton:**
  - inexact solves (Eisenstat-Walker, cap 1e-4 for LU-type preconditioners
    and 1e-8 for ILU)
  - a fixed-pattern assembly plan
  - no redundant final assembly
  - a quadratic-convergence prediction exit that also tracks
    density-weighted quasi-Fermi updates
  - an explicit `converged` flag, which drivers now use
- `mesh3d/extrude.py` plus `main2d_mosfet_sweep.py --extrude-um/--nz`: the 3D testbed.
- `testsuite/test_2d_examples.py`: every bias point of the 4 2D drivers vs goldens.

**Timings** (clean machine):

| run | before | now | check |
|---|---|---|---|
| 2D MOSFET full I-V (138 pts) | 17.5 s | 7.3-7.7 s | goldens pass |
| 2D BTBT/TAT sweep | 93 s | ~60 s | goldens pass |
| 2D diode | 4.0 s | 2.8 s | goldens pass |
| 2D MOS-cap C-V | 14 s | 14 s | Poisson-only solver, not ported yet |
| **3D extruded MOSFET** (28k nodes, 84k unknowns, 138 pts) | ~hours (SuperLU) | **73 s** | all 138 points match 2D within 1e-3; SS/Vt/DIBL/Ion identical |

Scaled to the 102-point benchmark definition, that is about 54 s.

**Lessons recorded in the code:**
1. **ILU needs tight linear tolerances.** At rtol=1e-4 an ILU-GMRES step
   carried large errors in the poorly determined quasi-Fermi levels, so the
   cap is 1e-8.
2. **The prediction exit must track quasi-Fermi levels, not only psi.** A
   psi-only version stopped an off-state point with phin still 3e-4 V off,
   and Id came out 1000x too large.
3. **conda-forge PETSc's MPICH busy-polls about 2 cores per idle process.**
   This is silenced with `FI_PROVIDER=tcp`, which `core/linsolve.py` sets.
4. **Off-state currents below ~1e-13 A/um are contact-current round-off**
   (~4e-17 A/um absolute), even with the old solver.
5. **The M3 Pro is 6P+6E**, so the worker default is the P-core count.

**What limits 3D now: memory bandwidth.** Wall time is the same (~75 s) for 3,
4, 6 or 8 workers; one curve takes 17 s alone vs ~50 s with 6 in parallel.
Further gains must cut data moved per iteration:
- fewer GMRES iterations (better preconditioner: AMG on the Poisson block,
  field-split)
- an FP32 preconditioner (half the bytes; a candidate for Rust or the GPU)
- fewer Newton iterations (tangent predictor)

---

## Part 2c: Multiphysics-ready solver core (thermal, stress, diffusion)

Concepts borrowed from COMSOL:

- **Physics interface:** dependent variables plus a PDE in conservation form,
  ∂ₜs(u) + ∇·Γ(u,∇u) = R(u), active on a **selection** (named regions,
  boundaries, interfaces).
  - Drift-diffusion: ψ, φn, φp on semiconductors, with ψ only in insulators.
  - Thermal: T everywhere.
  - Dopant diffusion: C_i in semiconductors, transient.
  - Stress: displacement u in solids.
- **Discretization per physics, on one mesh:**
  - FV-box for scalar conservation laws (drift-diffusion, thermal, diffusion).
  - P1/P2 **FEM** for elasticity. The box method does not suit a vector
    elasticity PDE, which is why the mesher always produces tets.
- **Multiphysics couplings** are separate objects, not edits to either physics:
  - Joule heat J·E → thermal source
  - T → mobility, n_i, bandgap
  - lattice-mismatch / intrinsic stress → strain → band shifts and
    piezoresistive mobility (key for SiGe S/D pFET)
- **Study steps:** stationary, parametric sweep/continuation, transient (BDF),
  small-signal AC (C-V). **Solver configs:** fully coupled or segregated.
- Kernels are vectorized pure functions with hand-derived Jacobians for
  drift-diffusion. New physics can use automatic differentiation (JAX with
  x64 on CPU) or FD-colored Jacobians. This keeps the door open for the
  general-PDE layer (standing decision #4 in `ARCHITECTURE.md`).
- Avalanche stays standalone (standing decision #1).

---

## Part 3: Visualizer ("tcadview"), Qt desktop app

**Stack:** PySide6 (LGPL, safe to redistribute) + pyvistaqt + VTK.

**Features** (modeled on Sentaurus Visual, Tecplot and TonyPlot):
- Multiple datasets and viewports, with linked camera, slices, fields and color
  ranges. Difference plots between two datasets.
- Region/material tree: visibility, opacity, wireframe, mesh edges, contacts and
  interfaces highlighted.
- Fields:
  - lin, log and signed-log (asinh)
  - derived fields: |E|, |J|, Ec/Ev/Efn/Efp
  - range: auto, manual or per region
  - contour levels and contour-line overlays
- Cuts:
  - x/y/z slice planes with sliders, plus arbitrary planes
  - clip, to cut away half the device
  - cut plane → linked 2D view
  - cut line → 1D profile panel (overlay several, export CSV)
- Isosurfaces (junction N_net=0, n=1e18), thresholds, current streamlines and
  vectors, click-to-probe.
- **I-V panel linked to the 3D view** (click a curve point → jump to that bias),
  bias stepper/animation, process-step slider.
- Session files (YAML) for reproducible figures, plus screenshot and
  publication export.

**Distribution via GitHub:**
- pip-installable package with extras: `pip install "tcad[viewer]"`
- GitHub Actions CI on macOS/Linux/Windows with offscreen-render tests
- PyInstaller-built `.dmg` / `.exe` / AppImage attached to GitHub Releases

---

## Build order (one feature branch per phase, merged when finished)

| # | Phase | Exit criterion |
|---|---|---|
| 0 | Data model: VTK export of existing 1D/2D results, naming conventions, YAML schema skeleton | Existing 2D runs open correctly in ParaView |
| 1 | **Performance core + 3D feasibility spike.** Fixed-pattern assembly, `LinearSolver` backends, Jacobian reuse, tangent predictor in the shared solver. Spike: extrude the 2D MOSFET mesh in z (an extruded Voronoi mesh is a valid 3D box mesh) and benchmark the real 3D Jacobian | 2D full I-V ≈ 3–5 s with goldens unchanged; 3D extruded-MOSFET full I-V timed end to end |
| 2 | tcadview core (Qt): regions, slices, lin/log, probe, cut lines, on existing 2D and extruded-3D data | Replaces `plot2d_web.py` for daily use |
| 3 | Process engine + planar template, process-step slider | Planar MOSFET from YAML, viewed step by step |
| 4 | FinFET, nanosheet (inner spacer, release, epi variants), analytic doping, contacts, domain cuts | All templates build from YAML |
| 5 | Octree mesher + 3D FV geometry + quality gate | Gate passes on every template |
| 6 | 3D solve + multiphysics-ready core. Validation ladder: extruded-2D match → 3D diode vs analytic → half = full/2 → mesh convergence → FinFET → nanosheet | 1-minute benchmark reported |
| 7 | Full viewer: isosurfaces, streamlines, I-V linking, multi-dataset, packaging/Releases | Installable from GitHub |

**1-minute benchmark definition:** a half-symmetric FinFET; Id-Vg at Vd = 0.05
and 0.7 V (21 points each) plus Id-Vd at 4 Vg (15 points each), 102 points
total, measured as wall-clock time on this Mac including mesh build.
