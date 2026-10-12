# 3D program: handoff

## COLD START (2026-10-11, after the merge): read this first

Written to resume with ZERO chat context. The later sections of this file
are older passdowns, kept as history. DEVELOPMENT_LOG.md sections 41-54
tell the full story with numbers.

### UPDATE 2026-10-11 (later): Phase A speed pass in progress on `3d-speed`
- The user chose A (speed). B (octree defaults study) is DEFERRED to a
  FinFET, at the user's request; `test_octree` stays at 2.5% until then.
- Done (DEVELOPMENT_LOG 55):
  - quasi-Fermi box (maximum principle) in `newton_solve_2d`;
  - `vsat_fallback=False` in `fab simulate` (halve the bias step instead);
  - Rust `ball_query` in the mesher;
  - Masetti doping mobility, now the default (the user asked for it in
    the S/D); goldens re-captured after verification.
- Full I-V: 243-261 s -> 172 s.
- Done (DEVELOPMENT_LOG 56), the linear solver:
  - numeric-only ILU refactorization (KSP kept per sparsity pattern);
  - an iteration budget for a reused ILU (3x the fresh one's, min 20);
  - ILU inexact-Newton cap 1e-8 -> 1e-6;
  - `fab simulate` no longer forces MUMPS (ILU now converges on the
    rectilinear mesh too, 6.6x faster, same currents).
  - MUMPS, hypre (BoomerAMG, Euclid), other ILU orderings/levels and
    Krylov types were measured and rejected (table in the log).
- Full I-V: octree 172 s -> 106 s; rectilinear (default settings) 70 s.
- Next in A:
  - assembly + line-search residuals (now ~half of each Newton iteration);
  - the repeated Vgs = 0 walks of the segments.
- Profiling tool: `fab/profile_iv.py` (`--dump DIR` saves the walk's
  linear systems for offline solver comparisons).
- The user's standing concern: speed must survive richer mobility and
  stress models, so prefer model-agnostic speedups.

### 0. Where things stand (one paragraph)
`main` now holds Phases 0-3 and the octree mesher, merged and pushed
(DEVELOPMENT_LOG 54).
- *Device:* a 3D planar NMOS is built by the virtual fab, meshed by the
  exact-Voronoi octree, and solved by drift-diffusion with hybrid
  quasi-Fermi velocity saturation.
- *Full I-V:* 2 x 13 points in 243-261 s, all performance cores.
- *Viewer:* exact material outlines, a zoomable cut plot, and E/current
  vectors.
- *Tests:* the suite covers 1D, 2D and 3D (110 tests, about 3.5 min,
  plus the octree tests under conda).
- *Next:* the user said: "before starting next phases, check with me".
  DO NOT start a phase; present the options in section 6 and wait.

### 1. Working rules the user set (also in memory, MEMORY.md)
- *Git.*
  - Work on feature branches. Never add Claude attribution
    (Co-Authored-By / "Generated with") to commits or PRs.
  - Never rewrite pushed history. Confirm before any push or merge,
    unless the user says to do it (as they did for this merge).
- *Log.* A DEVELOPMENT_LOG.md entry with every commit.
- *Plans.* Show the plan before a big implementation; the user approves,
  then build.
- *Long-term fixes.* Everything must hold for FinFET, nanosheet, any
  gate / oxide / contact style. No planar-only shortcuts. Prefer exact
  geometric or numerical invariants backed by a mandatory gate (memory
  tcad1d-geometry-general-fixes).
- *Mesh quality.* A mandatory gate on every build, never opt-in.
  Voronoi-admissible meshes only.
- *Geometry.* The user inspects 3D structures closely (no slivers, ledges
  or voids); the viewer must show exactly what the solver sees.
- *Physics.*
  - Metals carry no unknowns (Dirichlet surfaces).
  - Contacts are process-built metal volumes.
  - Planar devices have finite width with STI.
  - ILD sits between the spacer and the contacts.
- *Speed.* The full 3D I-V target is about 1 minute.
  - Rust (PyO3, `native/`) only for 5-10x or more, after profiling.
  - GPU only for more than 3x TOTAL gain.
  - Use all performance cores.
- *Tests.* The user wants a super-quick suite. Each 1D/2D/3D case stays
  within the other cases' time (the 3D case takes 42 s).
- *Communication.*
  - Short, plain status updates and honest numbers, including negative
    results.
  - The user runs commands in their own terminal, so give the
    `.venv/bin/python` form (system python3 lacks PyVista/Qt).

### 2. Environments
- *venv:* `.venv/bin/python`, no PETSc (SciPy SuperLU). Viewer, tests.
- *conda:* `~/miniconda3/envs/tcad/bin/python`, with PETSc. Use it for 3D
  solves. `fab simulate` uses PETSc block ILU by default ("auto");
  `TCAD_LINSOLVE=petsc_lu` forces MUMPS (much slower in 3D).
- *Rust kernels* (`native/`): rebuild BOTH environments after any change,
  from `native/`:
  - `env -u CONDA_PREFIX VIRTUAL_ENV=$PWD/../.venv ../.venv/bin/maturin develop --release`
  - `env -u VIRTUAL_ENV CONDA_PREFIX=$HOME/miniconda3/envs/tcad ~/miniconda3/envs/tcad/bin/maturin develop --release`
- *Tests:*
  - Everything: `.venv/bin/python -m unittest discover -s testsuite`
    (110 tests, about 3.5 min, 1 skip). Plus
    `~/miniconda3/envs/tcad/bin/python -m unittest testsuite.test_octree`
    (its rectilinear comparison needs PETSc).
  - Goldens, re-capture only after independent verification:
    - `testsuite/test_2d_examples.py --capture <name>`
    - `testsuite/test_3d_examples.py --capture`
    - the 1D ones: `testsuite/capture_golden.py`
- *Run things:*
  - `TCAD_LINSOLVE=petsc_ilu ~/miniconda3/envs/tcad/bin/python -m fab simulate configs/fab_planar.yaml --mesher octree --out out/fab_planar_octree_par`
  - `.venv/bin/python -m tcadview out/fab_planar_octree_par/vtk/idvg_Vds1.pvd`
  - `.venv/bin/python -m fab build configs/fab_planar.yaml --out out/fab_planar_struct`

### 3. Git state
- `main` = everything (merge of `3d-octree-mesh`, which contains
  `3d-fab-planar`), pushed to origin
  (github.com/saurabhmorarka/gatorade_tcaddevice).
- Old local branches (3d-perf-core ... 3d-viewer-sections) were merged
  earlier.
- For new work, branch from `main`.

### 4. Map of the 3D code
| path | what it is |
|---|---|
| `fab/structure.py`, `templates.py`, `flow.py`, `expr.py` | level-set virtual fab; planar template (STI, flat gate stack, TiN gate, spacer, ILD + CMP, W contacts); `check()` structure gate on every build |
| `fab/simulate.py` | `build_device` -> mesh -> solve. `simulate()`: the parent builds mesh + equilibrium + drain ramps ONCE, then workers (pool initializer, no level sets) run curve SEGMENTS in parallel; `write_state` / `write_state_octree` (vectors mirrored with a z flip) |
| `mesh3d/octree.py` | octree mesher: per-axis `target_spacing`; `BoxTree` (label, size, 2:1 balance, face match at semiconductor + conductor faces, `compact`, `split_for_stray`); `_dual`; `_assemble` (contact node = ON its metal); stray-metal refine loop; `check_octree_mesh` gate (volumes, facets, couplings, conformity, isolated nodes, contacts, NO stray metal volume) |
| `mesh3d/cartesian.py` | graded rectilinear reference mesh |
| `native/src/boxtree.rs`, `voronoi.rs` | Rust tree refine and Voronoi clipper |
| `mesh3d/studies/` | `mesher_convergence.py` (layout/sizing runs; TCAD_VSAT=0/1/name), transition study, leaf stats |
| `solver2d/newton_solver_qf_2d.py` | shared DD solver. `velocity_saturation`: False / True (= "hybrid", the default) / "efield" / "gradqf"; `_edge_force` + exact Jacobian; `VSAT_REF_DENSITY = 1e16`; direct attempt capped at `VSAT_DIRECT_MAXITER = 15`, then an efield warm-up fallback |
| `solver2d/vtk_export.py` | VTK fields; `nodal_vectors` (E, Jn, Jp, J by an edge least-squares fit of the solver's edge currents) |
| `tcadview/` | viewer. `model.State.outline` (exact material boundaries), `vector_fields` / `vector_samples`; app: outlines checkbox, Vectors group, cut-plot toolbar and X/Y axis scales |
| `testsuite/test_3d_examples.py` | tiny planar device, 3 bias points, golden + vector-flux check |
| `configs/fab_planar.yaml`, `fab_planar_tiny.yaml` | full and tiny devices |

### 5. Key numbers (full planar device: Lg 30, W 60, tox 1.5, spacer 8 nm)
- *Mesh:* octree 65.7k nodes, build about 15 s (one stray-metal pass).
- *Results:* Vt 0.204 -> 0.052 V, SS 82 mV/dec, DIBL 160 mV/V,
  Ion 1811 uA/um, Ioff 1.16e-6 A/um. A punch-through path under the
  channel: a device-design issue, not numerics.
- *Time:* 243-261 s total.
  - mesh + equilibrium 33-38 s;
  - drain ramp 43-47 s (sequential);
  - Vds = 1 V curve: slowest segment 165-174 s, about 40 Newton
    iterations per point with hybrid (efield about 3x fewer).
- *Accuracy:*
  - Coarse octree layouts agree within 0.9% with hybrid and sit about 3.5%
    above the fine meshes (a resolution question, not studied yet).
  - The rectilinear default is 68k nodes and about 1% off.

### 6. Next phases: OPTIONS TO PRESENT, wait for the user's choice
A. *Speed pass toward the 1-minute 3D I-V* (the user asked for "whatever
   speeds it up"; parked items):
   - per-point Newton cost at high Vds with hybrid (about 40 iterations).
     Candidates: a doping-relative c_ref (option b of log 51), Newton
     globalization (the row-scaled merit stalls on reasonable starts; log
     51), better predictors;
   - profile the 3D solve (assembly vs linear solve); preconditioner
     reuse; threaded direct solver;
   - the drain ramp (sequential 43 s) and mesh + equilibrium (33 s);
   - limit the stray-metal recheck to clipped cells near metal (+5 s
     today);
   - `_rule_size` halves odd boxes, leaving alternating 0.5/1 nm leaves
     (wasted nodes).
B. *Octree defaults study:* the coarse octree is +3.5% vs the fine meshes;
   find the region one change at a time, pick the defaults, tighten
   `test_octree`'s tolerance back from 2% to 1%, and add a README section
   for the octree mesher.
C. *The user's physics roadmap, in order* (memory
   tcad1d-physics-roadmap-after-3d-mesher):
   1. Schottky S/D contacts (research how others model them first);
   2. Schrodinger-Poisson / quantum correction (charge peak away from the
      interface, lower density);
   3. mobility: surface orientation, surface roughness, Coulomb, phonon,
      interface-charge dependence;
   4. stress and its effect on mobility;
   5. FinFET, nanosheet.
   New physics is prototyped in 1D first (memory).
D. *Smaller items raised before:*
   - capacitance extraction (Cgs/Cgd from terminal charge);
   - the device design (Vt, DIBL, punch-through);
   - an STI corner rounding option;
   - the SuperLU-singular rectilinear equilibrium (PETSc is fine).

### 7. Gotchas learned (recent)
- *Octree outlines:* VTK feature/boundary edges report every hanging-node
  seam. Use `State.outline`.
- *Contacts:* a contact node must lie ON its metal. The old rule (its cell
  reaches the metal) pinned oxide nodes to the gate, a local 3x Cox and up
  to 16% Id error.
- *Velocity saturation:*
  - The efield force throttles at the S/D junctions: the built-in field is
    parallel to the current, about a 20% Id loss at 50 mV.
  - Pure gradqf gives a singular Newton system.
  - Hybrid needs `VSAT_DIRECT_MAXITER` plus the efield warm-up fallback.
    Warm-up FIRST broke 2D continuation sweeps; a per-mesh strategy memory
    cost +65% iterations.
  - The contact-region shift start broke forward-biased diodes (log 51).
- *c_ref = 1e17* changes 2D lightly doped results by 5%; it is a model
  parameter, not a speed knob.
- *Segments:* weighting curve segments by cost was slower, because each
  segment repeats a walk from Vgs = 0.
- *Golden tests* can show "missing (point did not converge?)" when a
  change breaks convergence. Investigate; never just re-capture.

---

## History: older passdowns

## START HERE - passdown of 2026-10-10 night (session limit hit)

**Branches.**
- `main`: Phases 0-2, pushed.
- `3d-fab-planar`: Phase 3 virtual fab, planar template, ILD/contacts,
  rectilinear reference solve, exact voxel view. Committed, NOT merged,
  NOT pushed. The user said "looks good for now".
- `3d-octree-mesh`: branched from `3d-fab-planar`. Phase 5 octree
  mesher; NOTHING implemented yet. The user approved starting it here.
- Rules:
  - no Claude attribution in commits
  - confirm before pushing
  - DEVELOPMENT_LOG entry with every commit (sections 41-43 cover
    Phase 3)

**Progress on `3d-octree-mesh` (2026-10-11, DEVELOPMENT_LOG 45-46).**
- `mesh3d/octree.py` is complete and gated.
  - Rust kernels: `native/src/boxtree.rs` (`boxtree_refine`, identical
    leaves to Python) and `native/src/voronoi.rs` (`voronoi_cells`).
    Rebuild BOTH environments after changes (commands at the top of
    this file).
  - Per-axis sizing; face match at semiconductor AND conductor
    interfaces.
  - Full planar device: about 65k nodes, mesh 8 s, ILU solve works.
- `fab simulate --mesher octree` works and writes VTK
  (`write_state_octree`).
- Tests: `testsuite/test_octree.py`. The rectilinear comparison needs
  PETSc (SuperLU calls the rectilinear equilibrium Jacobian singular).
- The user confirmed that metals must carry no unknowns. Verified:
  - 0 nodes inside metal; gate surface at one potential.
  - S/D Fermi level fixed; psi varies 5.5 mV (the ohmic
    local-neutrality model; the Schottky contact is the real fix).

**NEXT STEP (user approved): find which region drives the octree's
on-current error before choosing defaults.**
- Full device, Id at Vds = 50 mV, Vg = 0.6 V; converged value about
  1.128e-5 A (fine octree 1.1297e-5 with 94k nodes, fine rectilinear
  1.1274e-5 with 169k).

| octree setting | nodes | Id(0.6) |
|---|---|---|
| default: h_fine (2, 0.5, 2), h_fine_other (2, 2, 2), h_max 8, growth 1.35 | 65,633 | 1.0737e-5 (-4.8%) |
| h_fine (1, 0.5, 1) | 84,130 | -3.1% |
| h_fine (1, 0.5, 1), h_fine_other (1, 1, 1), h_max 4 | 94,067 | +0.2% |
| growth 1.25, h_max 6 | 295k | +0.06% (too many nodes) |

- So h_max (4 vs 8) and/or h_fine_other drive it. Suspects:
  - channel x-spacing far from the junctions (up to h_max)
  - the gate-oxide/gate interface spacing
- Method: vary one at a time. Candidate fixes: a semiconductor-only
  h_max, or the gate edge as an x seed.
- Off-current has not converged either: the fine meshes differ 5.5%.
- Scripts (scratchpad, may be gone): `conv.py` built
  `build_device(cfg, mesher=..., grid=dict(...))`, solved equilibrium,
  then (Vg, Vd) = (0, 0.05) and (0.6, 0.05) with
  `TCAD_LINSOLVE=petsc_ilu`, and printed `drain_current`.
- Still to do after that: README section for the octree mesher; then
  ask the user about merging `3d-fab-planar` and `3d-octree-mesh`.

**Done on `3d-fab-planar` today** (DEVELOPMENT_LOG 42-43):
- *Flat gate stack.* The gate dielectric is deposited over the whole
  flat surface; the strip etches only the `gate_oxide` region via
  `etch(regions=)`. This fixed the STI recess step, spacer stringers,
  the gate ledge and voids.
- *Mandatory structure check.* `Structure.check()` runs in
  `flow.build`; one-voxel flakes are despeckled by `_despeckle`.
- *Cell labelling.* Gas iff the cell mean of the union field is >= 0
  (fixes cells no region claimed at curved surfaces).
- *ILD and contacts.* ILD fill + CMP to `ild_cap` above the gate;
  contact trenches through the ILD only; W up to the ILD top (contacts
  taller than the gate). No gas between spacer and contacts.
- *tcadview.* Process builds are drawn as exact voxel faces
  (`model.voxel_surface` / `voxel_slice`); smooth level-set contours
  behind a checkbox.
- *Solver.* An ohmic metal-volume contact's insulator nodes take the
  metal potential V + median psi_bulk (`solver2d/newton_solver_qf_2d.py`),
  where before they got a NaN from ni = 0. The 2D golden tests are
  unchanged.
- *Reference 3D solve.* `python -m fab simulate configs/fab_planar.yaml`
  (conda env, PETSc LU; ILU fails on 16:1 cells). Rectilinear grid,
  67,956 nodes, half width, whole stack. Results: Vt 0.218 -> 0.057 V,
  DIBL 170 mV/V, SS 89 mV/dec, Ion 1827 uA/um, Ioff 1.0e-6 A/um, 434 s.
  This is the baseline the octree mesher is compared against.

**The approved octree design (to implement on `3d-octree-mesh`).**
Write it in `mesh3d/octree.py`, producing the same attribute set as
`mesh3d/cartesian.py` (so `fab/simulate.py` can switch meshers).

1. *Adaptive box tree.*
   - Integer units = fab voxel (1, 0.5, 1 nm). Leaves are boxes; a leaf
     may split along any subset of axes, at fab voxel faces (kd-style,
     sizes need not be powers of 2).
   - Bookkeeping: an int32 "leaf id per fab voxel" array, for O(1) point
     location and neighbour lookup.
2. *Refinement criteria.*
   - Every leaf is a single label, gas and conductors included.
   - Per-axis target spacing graded from the Si/insulator interface,
     junctions (sign change of net doping), doping gradients and gate
     edges: h_fine (2, 0.5, 2) nm, growth 1.35, h_max 8, as in
     `fab/simulate.py`.
   - Coarse in the substrate depth, ILD, metal and gas.
   - 2:1 balance per axis between vertex-adjacent leaves.
3. *Interface rule (local, so refinement doesn't propagate along a
   whole interface).* The leaf across an interface face has the same
   size triple: a mirror image across the interface. Then dual facets
   don't cross interfaces obliquely.
4. *Nodes and control volumes.*
   - Nodes are all leaf corners. The control volumes are the exact
     Voronoi diagram of the nodes, which is always admissible: couplings
     are facet areas >= 0.
   - *Fast path.* The box between midpoints to the axis neighbours IS
     the Voronoi cell iff no other node is closer to any of its 8 corners
     (a vectorized cKDTree query on the corners). Per-material volumes
     and facets come from octants; each octant lies in one leaf.
   - *Other nodes (transitions).* Exact half-space clipping, as in
     `mesh3d/studies/octree_transition_study.py::voronoi_cell`. The
     per-material split comes from clipping the cell against each
     overlapping leaf box.
   - Prototype in Python; port the clipping to Rust (`native/`) if it
     dominates.
5. *Device subset.* Device = semiconductor + insulator leaves.
   Conductor-touching device nodes are contact nodes; the body is the
   bottom face. Sort the contact `boundary_point_index` ascending (the
   solver pairs it with `psi[is_contact]` in node order).
6. *Mandatory gate, raising on failure:*
   - CV > 0
   - per-material volume sums equal the voxel volumes
   - every clipped cell closes, and facet areas match from both sides
   - couplings >= 0
   - conformity: an edge with semiconductor facet area connects two
     nodes with semiconductor CV
   - no isolated nodes; every contact has nodes
7. *Validation.*
   - Same device as the rectilinear baseline: node count, Id-Vg
     agreement, runtime.
   - Does petsc_ilu converge now?
   - Runtime goal: full 3D I-V in about 1 minute.

**Study result behind this design** (script saved at
`mesh3d/studies/octree_transition_study.py`).
- On random 2:1 octrees, centre-fan tets (the 2D quadtree's trick)
  give about 15% negative couplings: rejected.
- The Voronoi dual was exact: volumes sum to the box and are all
  positive, cells close to 4e-9, and facets are symmetric.
- Non-box cells occur only within one fine cell of a level change.

**Open items to raise with the user next session.**
- Merge/push `3d-fab-planar` (needs explicit OK).
- Capacitance extraction (Cgd/Cgs from terminal charge vs bias), now
  possible because the contacts are volumes. The user cares about
  off-state capacitance; I offered it before or after the octree.
- `fab/simulate.py::sample_structure` still labels device cells by
  argmin of phi at coarse-cell centres, not by `cell_labels()`. The
  octree replaces this; fix it if the rectilinear path stays.
- Device design: Vt is nearly 0 at 1 V, DIBL 170. A design question
  for the user (doping/tox), not a mesh question.
- Later: Schottky contacts, STI corner rounding option, Phase 4
  (FinFET, nanosheet).


## Phase 3 status (2026-10-10): first milestone done on `3d-fab-planar`

- Virtual fab `fab/` and planar template (finite width + STI, always):
  `python3 -m fab build configs/fab_planar.yaml`, view with
  `python3 -m tcadview out/fab_planar/build.pvd`. DEVELOPMENT_LOG section 41.
- Rust re-distancing kernel in `native/` (rebuild both envs after pulling:
  venv `env -u CONDA_PREFIX VIRTUAL_ENV=$PWD/../.venv ../.venv/bin/maturin develop --release`,
  conda `env -u VIRTUAL_ENV CONDA_PREFIX=~/miniconda3/envs/tcad ~/miniconda3/envs/tcad/bin/maturin develop --release`,
  both from `native/`).
- Not merged, not pushed. Waiting on user feedback on the planar build.
- No Claude attribution in commits (user rule).

### Gotchas specific to the fab
- Inside is phi < 0; a complement is -phi - EPS (EPS = 2^-20), removals are
  inclusive; level sets are float64 (float32 broke the tie arithmetic).
- Set operations leave exact-zero "phantom" sheets; anything that takes
  column minima or projects onto the zero set must guard against them
  (`_straddles`, `phi <= 0` updates in `_remove`).
- Anything built from a per-column top height is only column-accurate
  laterally; use minima of the solid field along columns instead.
- A deposit must not reach the domain ceiling (raises).

### Next in Phase 3 / then Phase 4
- user review of the planar structure (dimensions, doping defaults)
- epitaxy (Wulff facets) and isotropic selective etch refinements are
  Phase 4 needs (FinFET, nanosheet); octree mesher is Phase 5
- profile again on a FinFET-size build before more Rust

### Phase 3 plan (as proposed; amended by the decisions above)
- `fab/` package:
  - per-material SDFs on a uniform voxel grid, materials never overlap
  - operations: substrate, pattern (polygon masks with parameter
    expressions), deposit (conformal = exact SDF offset, directional/fill),
    etch (anisotropic depth/stop-on, isotropic selective), planarize
  - re-distancing after directional steps (fast marching), profiled first
    (Rust/GPU only per the 5-10x / >3x rules)
- Three-layer YAML with a safe expression evaluator.
- Analytic doping as functions of signed distance: well, Vt, LDD following
  the spacer edge, deep S/D erfc. No halo.
- Planar flow: Si -> STI -> well/Vt -> gate oxide -> poly -> gate pattern +
  etch (stop on oxide) -> extension -> SiN conformal + anisotropic etch
  (spacer) -> deep S/D -> contacts.
- Viewer: each step is a VTK snapshot (voxel material labels, smooth SDF
  contour surfaces, doping), with a `.pvd` over the steps for the existing
  stepper relabelled "Process step".
- Checks:
  - per-operation tests against analytic geometry
  - doping vs analytic profiles
  - an infinite-width planar centre section vs the 2D MOSFET config
  - build runtime reported (target: seconds at ~1 nm voxels)
- Out of scope: tet meshing/solve (Phases 5-6), FinFET/nanosheet (Phase 4).

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
   - section view: done, the user is happy (2026-10-10)
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
