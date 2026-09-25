# Architecture notes

Why the code is organized the way it is, and the decisions already locked
in for the eventual move to 2D/3D device geometry (and beyond). Written so
none of this has to be re-derived from git archaeology again.

## `core/` is shared 1D machinery, not diode-specific

`core/` exists because `mos/` needed the same mesh generator, materials
model, Poisson/continuity solver, and doping-profile machinery the diode
driver (`main.py`) already had — `core/mesh.py`'s `build_diode_grid` and
`build_mos_grid` sit side by side sharing the `_region_nodes` helper
(`core/mesh.py:115`) precisely because the mesh-generation logic is
genuinely shared, not device-specific. The diode itself was never split
into its own `diode/` package because `main.py` staying at the top level
as the primary entry point, plus `avalanche/` and `tat/` importing
`core/newton_solver_qf.py`'s QF-solver building blocks in an
inheritance-like way, means a rename wouldn't actually change the
underlying coupling — it would just add indirection.

**New physics gets its own top-level sibling package** (`avalanche/`,
`tat/`, and any future mechanism), never nested inside `mos/`, so the
codebase doesn't quietly become MOS-owned the way it once quietly became
diode-owned before `core/` existed.

## What in `core/` is 1D-only — do not genericize in place

- `core/mesh.py` (`build_diode_grid`, `build_mos_grid`) — builds a single
  1D node array per device.
- `core/newton_solver_qf.py` — edge-based assembly keyed on `np.diff(x)`
  and linear node ordering along one array (see `_edge_quantities`,
  `unpack_qf`).
- `core/physics.py`'s `solve_continuity_n`/`solve_continuity_p` — explicit
  tridiagonal solves.

When 2D/3D work starts, it will not extend these files with `if dim==2`
branches — that would be far messier than writing new, parallel assembly
code that imports the pure-physics kernel below. These files stay exactly
as they are.

## The reusable pure-physics kernel

Dimension-agnostic today (pure functions of local scalar/array
quantities, no mesh-topology assumptions baked in) and worth protecting
as new mechanisms get added:

- `core/physics.py`: `srh_recombination`, `bernoulli`, `bernoulli_deriv`.
- `tat/tat.py`: `btbt_generation`, `hurkx_tat_generation`,
  `schenk_tat_generation`.
- `avalanche/avalanche.py`'s impact-ionization rate function.
- The `Material`/`Device` dataclasses (`core/params.py`,
  `core/materials.py`) and `core/doping_profiles.py`.

Mobility today is a constant field on `Material`/`Device`, not a computed
function. Any future doping- or field-dependent mobility model should be
written the same way — a pure function of local quantities, no mesh
assumptions — both as good practice now and because it keeps the door
open for the general-PDE direction below.

## Avalanche stays standalone — permanently

`avalanche/`'s impact-ionization physics is numerically finicky (its
generation rate depends on `|Jn|`, `|Jp|` — the very quantities the
continuity equations solve for — a direct positive feedback that makes the
I-V near-vertical at breakdown, traced with arc-length continuation
(`core/arclength.py`) rather than a voltage sweep). It is not expected to
ever run combined with TAT/BTBT or other generation mechanisms in the
same solve, in 1D or later in 2D/3D. `avalanche/newton_solver_avalanche.py`
deliberately keeps its own private copies of the QF-solver building
blocks rather than importing `core/newton_solver_qf.py`'s shared ones —
this is intentional, not an oversight, and should stay that way. Do not
propose or build a shared multi-mechanism solver that merges avalanche
with anything else. (Solver-agnostic numerics with no physics in them —
`core/newton_numerics.py`, `core/jacobian_scaling.py`, `core/arclength.py`
— are shared by every 1D solver, avalanche included; the rule is about
physics and QF building blocks, not generic Newton/continuation code.)

(`tat/newton_solver_tat.py` already sums Kane BTBT + one trap-assisted
model — Hurkx or Schenk — within a single solve. That composability is
scoped to TAT/BTBT's own mechanisms and already works; it is unrelated to
the avalanche non-goal above.)

## Shared QF-solver building blocks

`core/newton_solver_qf.py` exposes `poisson_row_scale`,
`continuity_row_scale`, `unpack_qf`, and `MAX_QF_STEP` (via `__all__`) as
the public surface other QF-based solvers may import —
`tat/newton_solver_tat.py` does. `avalanche/` intentionally does not (see
above).

## The `dim` field in `plot.py`/`structure_io.py`

`core/plot.py:50`'s `_require_1d` guard and `core/structure_io.py`'s
stored `doc["dim"]` field are pre-existing 1D scaffolding. They are
**not** the seam 2D/3D visualization will extend — see below. They can be
left exactly as they are.

## Locked-in decisions for 2D/3D (not built yet)

Not implemented — recorded now so future work doesn't get designed
against the wrong assumption.

- **Visualization**: 2D/3D needs a genuinely different tool — interactive
  slicing through the 3D structure, field values rendered directly on the
  mesh (Tecplot-like), not matplotlib line/contour plots. This will be
  new, separate code (its own package, e.g. `viz3d/`, when it starts),
  not an extension of `core/plot.py`.
- **Meshing**: 2D/3D meshing will use a point-cloud-based approach rather
  than a structured grid extending `core/mesh.py`'s geometric-node-array
  style, for the flexibility point clouds give with irregular device
  geometries. The exact discretization method (point-cloud finite-volume,
  meshless, etc.) is to be decided when 2D work actually starts.
- **Nonlocal tunneling-path search**: a standing roadmap item (see
  `DEVELOPMENT_LOG.md` Session 11) — a genuine nonlocal tunneling-path
  search, as opposed to today's local-field closed-form BTBT/TAT models,
  is planned once 2D/3D geometry exists. It's a natural fit for the
  future 2D/3D + point-cloud-meshing package, not `core/`.
- **General PDE framework**: the long-term ambition is letting a user
  specify their own PDE — equations, constants, variables — rather than
  this codebase staying permanently drift-diffusion-specific (an explicit
  example given: thermal simulation). This is a separate, much larger
  design effort to scope later. Today's requirement is only to keep
  physics parameterized as swappable pure functions/coefficients (per the
  pure-physics-kernel section above) rather than hardcoded into assembly
  code, since that's the direction a general-PDE framework would need
  anyway.
