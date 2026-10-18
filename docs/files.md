# File map (1D core and contacts)

[Back to the README](../README.md)

## Files

| File | Purpose |
|---|---|
| `configs/input_diode.yaml` | **Edit this** for the diode: doping (flat/linear/gaussian per side), thickness, mesh knobs, voltage sweep, solver (`math_model: gummel` \| `newton`), which bias points to save fields for |
| `core/config.py` | Loads `configs/input_diode.yaml` into `Material`/`Device`/voltage-sweep/solver-choice/mesh overrides |
| `configs/input_mos.yaml` | **Edit this** for the MOS capacitor: substrate polarity/doping (flat/linear/gaussian), oxide thickness/permittivity, gate work function, mesh knobs, voltage sweep |
| `mos/mos_config.py` | Loads `configs/input_mos.yaml` into `Material`/`MOSDevice`/voltage-sweep/mesh overrides |
| `core/doping_profiles.py` | Shared doping-profile shapes (flat/linear/gaussian) and their sampling/reference-concentration logic, used by both YAML files and both mesh builders |
| `core/params.py` | Physical constants and diode material/device parameter defaults |
| `mos/mos_params.py` | MOS capacitor device parameters (oxide thickness, gate work function, SiO2 permittivity) |
| `core/mesh.py` | Shared mesh engine (`build_diode_grid`, `build_mos_grid`): geometric refinement at hard interfaces plus adaptive refinement wherever a graded doping profile changes quickly |
| `core/physics.py` | Bernoulli function (+ its derivative), nonlinear Poisson (Newton, generalized to array eps/ni and frozen-carrier modes), Scharfetter-Gummel continuity solves |
| `core/analytic.py` | Diode closed-form comparisons (Vbi, depletion width, Shockley law) |
| `mos/mos_analytic.py` | MOS-cap closed-form comparisons (flat-band/threshold voltage, analytic low-/high-freq C-V) |
| `quantum/bands.py` | Subband masses (m_z, m_d, degeneracy) of the Si valleys for any confinement direction; parabolic holes |
| `quantum/schrodinger1d.py` | Effective-mass Schrodinger equation on a line (BenDaniel-Duke, oxide penetration) and the subband density |
| `quantum/lines.py` | A Schrodinger line through the channel body and the charge-conserving quantum potential Lambda on the DD nodes |
| `quantum/coupling.py` | Predictor-corrector Schrodinger-Poisson outer loop |
| `quantum/mos1d.py`, `quantum/validate_1d.py` | 1D MOS-cap and double-gate drivers; the P1 validation table and figure (docs/quantum.md) |
| `quantum/device.py` | The 2D/3D quantum box: Schrodinger lines from every gate-dielectric interface (wall to wall in fins and sheets, mirrored at a symmetry plane), Lambda on every node, the corrector called inside Newton |
| `quantum/compare_3d.py` | Classical vs Schrodinger-Poisson figure from two `fab simulate` runs (Id-Vg, density profile at the solver nodes, sections, Lambda) |
| `quantum/compare_literature.py` | The 1D Vt shift against Jallepalli et al. (docs/quantum.md) |
| `configs/fab_finfet.yaml`, `configs/fab_nanosheet.yaml` | **Edit these** for the FinFET / nanosheet (fab/templates.py `finfet`, `nanosheet`; docs/finfet_nanosheet.md, S/D epi shapes docs/sd_epi.md) |
| `docs/make_device_figures.py` | Structure figures of the FinFET and nanosheet (3D, gate cut, doping cutaway, 2D cuts) |
| `core/solver.py` | Diode equilibrium solve + bias sweep (dispatches to either solver below) |
| `core/newton_solver.py` | Diode's fully coupled Newton solve: analytic sparse Jacobian, direct sparse solve, backtracking line search |
| `mos/mos_solver.py` | MOS-cap equilibrium C-V sweep (low-frequency) and frozen-carrier quasi-small-signal sweep (high-frequency) |
| `core/field_save.py` | Shared helper: selecting which bias points to save full field profiles for, quasi-Fermi-potential plotting, field CSV export |
| `core/structure_io.py` | Schema + save/load for the `*_structure.json` files each driver writes: device geometry, mesh, doping, and per-bias fields, in one human-readable file `plot.py` (or a future 2D/3D version of this project) can read back |
| `core/plot.py` | Structure/band-diagram/charge-density plot library, driven from a loaded structure file. Also runnable standalone against just a `*_structure.json`: `python3 core/plot.py out/diode/diode_structure.json --interactive` opens one window with every plot and field already loaded, click a checkbox to toggle a curve; `--which`/`--band-fields`/`--charge-fields` narrow down what gets drawn (in either interactive or plain PNG mode) |
| `main.py` | Diode driver: runs the sweep with both solvers (for the benchmark) plus the one from `configs/input_diode.yaml`, generates plots and CSVs in `out/diode/` |
| `mos/mos_main.py` | MOS-cap driver: runs the C-V sweep, generates plots and CSVs in `out/mos/` (or `out/mos_poly/` for a poly-gate input) |
| `contacts/schottky.py` | Schottky contact physics kernel: barrier models (direct, Schottky-Mott, pinned), thermionic-emission boundary quantities, image-force lowering, exact piecewise-linear WKB integral with analytic derivatives, Tsu-Esaki supply, depletion-model rho_c reference |
| `contacts/newton_schottky_1d.py` | 1D quasi-Fermi Newton solve with Schottky ends (TE Robin rows + tunneling sink at the turning points), bias sweep, contact-resistivity extraction, energy-resolved current (`spectrum`) |
| `contacts/study_ti_barriers.py` | Study A: Ti/TiN/NiSi barriers (Schottky-Mott vs pinned vs measured) and rho_c on NMOS Si:P and PMOS Si0.5Ge0.5:B |
| `contacts/plot_schottky_current.py` | Barrier, tunneling paths and tunneling vs over-the-barrier current, NMOS and PMOS, vs doping |
| `contacts/subgrid.py` | Sub-grid Schottky contact for 2D/3D meshes: J(N, u) table from the 1D model (disk cache), nodal source term + Jacobian for `newton_solve_2d(schottky=...)`, terminal current, metal interface area per node from voxel faces |

Requires `numpy`, `scipy`, `matplotlib`, `pyyaml`. Output plots and CSVs are
written to `out/<example>/` (a separate subfolder per example; filenames don't collide).
