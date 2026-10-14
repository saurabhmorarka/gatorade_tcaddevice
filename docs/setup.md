# Setup, testsuite, solvers and viewers

[Back to the README](../README.md)

## Setup

Requires Python 3.9+ (any plain CPython install - no conda needed). Your
system `python3` almost certainly does NOT already have numpy/scipy/
matplotlib/pyyaml installed globally, so create a virtual environment first
rather than trying to run the scripts directly against system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 main.py                  # or any other script below
```

That's the only setup step - every script in this repo (`main.py`,
`mos/mos_main.py`, `avalanche/main_avalanche.py`, `mos/mos_poly_sweep.py`,
`avalanche/robustness_matrix.py`, `testsuite/test_examples.py`, ...) is
then runnable from the repo root, either directly with `python3
<path>/<script>.py` or (for anything inside the `core`/`mos`/`avalanche`
packages) as a module with `python3 -m <package>.<script>`, e.g. `python3
-m mos.mos_main`, no additional configuration, environment variables, or
install steps. All output (plots, CSVs, structure JSON) is written to
`out/<example>/` (e.g. `out/diode/`, `out/mos/`), which already exists in
the repo with the committed reference outputs - reruns just overwrite the
files for whichever example you ran.

To confirm everything works end to end after installing, run the whole
suite in parallel (one process per test class, performance cores; ~70 s
wall for the 132 tests in the pip venv):

```bash
python3 testsuite/run.py                     # quick tier (default): reduced 2D sweeps, no full I-V
TCAD_FULL_TESTS=1 python3 testsuite/run.py   # full 2D sweeps - before a merge
python3 testsuite/run.py -k schottky         # only test ids containing "schottky"
python3 testsuite/run.py --record            # refresh testsuite/timings.json (job ordering)
```

Run it with the interpreter that has the packages (the conda env for the
PETSc / octree tests; they skip themselves elsewhere). Single files still
work on their own:

```bash
python3 testsuite/test_examples.py   # should print "OK" - 5 tests, no failures
python3 testsuite/test_2d_examples.py  # the 4 2D drivers (quick tier; TCAD_FULL_TESTS=1: every bias point)
python3 testsuite/test_vtk.py          # VTK export (also checks ParaView reads it, if installed)
python3 testsuite/test_native.py       # compiled kernels vs numpy (skipped if not built)
python3 testsuite/test_tcadview.py     # viewer, headless (skipped without the viewer deps)
python3 testsuite/test_fab.py          # virtual fab: operations vs analytic geometry, planar template
```

### Fast path (optional): PETSc solvers for 3D

The pip environment above runs everything, using METIS-ordered SuperLU
(`core/linsolve.py`). For 3D-sized problems, use the conda environment, which
adds PETSc point-block ILU-GMRES (about 20x faster than SuperLU on a real 3D
Jacobian):

```bash
conda env create -f environment.yml && conda activate tcad
python3 main2d_mosfet_sweep.py --extrude-um 0.1 --nz 6   # 3D extruded MOSFET, full I-V (~75 s)
```

Set `TCAD_LINSOLVE=superlu|umfpack|scipy_ilu|petsc_lu|petsc_ilu` to force a
backend (default `auto`: SuperLU below 60k unknowns, PETSc ILU above). The
3D roadmap and measured performance are in `plans/3d_plan.md`.

### Compiled kernels (optional, Rust)

Hot loops that numpy can't vectorize well have a Rust version in `native/`.
So far that is the nonlocal-tunneling field-line tracer, about 8x faster
than numpy, which cuts each nonlocal BTBT curve from ~43 s to ~26 s.
Everything runs without it: `core/native.py` falls back to numpy when the
extension isn't installed, and `TCAD_NATIVE=0` forces the fallback.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y   # Rust toolchain
pip install maturin
cd native && maturin develop --release && cd ..   # builds into the active env
python3 testsuite/test_native.py                    # compiled vs numpy, to round-off
```

If both a venv and conda are active, maturin refuses to guess. Unset one
first (for example `env -u CONDA_PREFIX maturin develop --release`).

### Viewing results in ParaView (VTK export)

The 2D drivers take `--vtk` and write `<out-dir>/vtk/*.vtu` plus a `.pvd`
collection whose time axis is the swept bias:

```bash
python3 main2d_sweep.py --vtk                 # diode: the saved bias points
python3 main2d_mos_sweep.py --vtk             # MOS-cap: the saved gate voltages
python3 main2d_mosfet_sweep.py --vtk          # MOSFET: last point of each curve
python3 main2d_mosfet_sweep.py --vtk all      # every bias point (~0.5 MB each in 2D, ~3 MB in 3D)
```

The files are self-contained:
- **Point fields** use Sentaurus-style names: `ElectrostaticPotential`,
  `eDensity`/`hDensity`, quasi-Fermi potentials and energies, band edges,
  `DopingConcentration`, `SpaceCharge`, and `contact_id`.
- **Cell fields:** `region_id`, `material_id` and `ElectricField`.
- **Field data:** region, material and contact names, plus contact biases.

Carrier quantities are NaN in insulators. y is depth, positive downward, so
in ParaView look from -z with view-up (0, -1, 0). The writer is pure numpy
(`core/vtk_io.py`), so nothing extra needs installing.

### tcadview: desktop viewer (optional)

`tcadview` opens the same `.vtu`/`.pvd` files in a Qt window:

```bash
pip install -r requirements-viewer.txt          # PySide6, pyvista, pyvistaqt (already in environment.yml)
python3 main2d_mosfet_sweep.py --vtk all
python3 -m tcadview out/input_mosfet_2d/vtk/idvg_Vds0.05.pvd
python3 -m tcadview result.pvd --field eDensity --state -1 --screenshot shot.png   # headless
```

What it does:
- **Regions panel:** region/material tree with show/hide and per-region
  opacity.
- **Field controls:** field picker with linear, log or signed-log scale,
  auto or manual range, colormap choice and mesh edges. Densities and
  doping default to log scales.
- **Slices (3D):** x/y/z slice planes with sliders. Region outlines stay
  visible while slicing.
- **Probe:** right-click a point to list every field there and its region.
- **Cut line:** type two points or pick them with right-clicks. The 1D
  panel plots the field along the line and follows the bias stepper.
- **Bias stepper:** for a sweep, a slider and prev/next buttons. The
  camera, field and cut line are kept as you step.
- **Section view (3D):** Slices > Section view shows an x/y/z slice
  face-on in a second, 2D view. Right-click probes inside the device. H cut
  and V cut run a cut line through the last clicked point, and Overlay
  curves keeps earlier curves to compare bias points.
- **Process steps:** a virtual-fab build (`fab/`) opens on its final
  structure, with a "Process step" slider. The Material field colours
  each region by its material. Region surfaces are smooth contours of the
  level sets, and the region list grows as layers are added.

The device is drawn the right way up (y is depth). `testsuite/test_tcadview.py`
drives the window headless.
