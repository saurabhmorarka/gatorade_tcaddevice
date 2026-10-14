# FinFET and nanosheet flows: details

[Back to the README](../README.md)

Two new virtual-fab templates at 2 nm-node-class dimensions: a bulk FinFET
and a 3-sheet nanosheet (gate-all-around). Plan:
`plans/finfet_nanosheet_plan.md`. Numbers: DEVELOPMENT_LOG 70.

```bash
python3 -m fab build configs/fab_finfet.yaml        # ~1 s + snapshots
python3 -m fab build configs/fab_nanosheet.yaml     # ~3 s + snapshots
python3 -m tcadview out/fab_finfet/build.pvd        # scrub through the build
python3 docs/make_device_figures.py                 # the pictures below
~/miniconda3/envs/tcad/bin/python -m fab simulate configs/fab_finfet.yaml --mesher octree \
    --vds 0.05 0.7 --vgs-max 0.8 --vgs-step 0.05 [--quantum]
```

## Dimensions

The user set the fin (50 nm tall, 5 nm wide), the sheets (3 x 5 nm thick,
25 nm wide), the 48 nm contacted poly pitch (CPP) and the ~15 nm gate.
The rest are typical published 3-2 nm-node values:

| | FinFET | Nanosheet |
|---|---|---|
| CPP / Lg / spacer | 48 / 15 / 5 nm | 48 / 15 / 5 nm (+ 5 nm inner spacers) |
| body | fin 5 x 50 nm, 24 nm fin pitch | 3 sheets 5 x 25 nm, 10 nm gaps, 46 nm stack pitch |
| gate stack | IL + HfO2 lumped, 2 nm, EOT 0.85 nm; TiN, wf 4.40 eV | same, wrapping every sheet |
| S/D | recessed fin, Si:P epi 7e20 (sx 2, sy 3 nm) | epi from the sheet ends, 7e20 |
| isolation | PTS 3e18 at the fin base | PTS 1e18 under the stack (sharp) |
| contacts | TiN trench contacts, Schottky (0.45 eV) | same |

The domain is one CPP along x (the gate in the middle, half of each shared
contact at the ends) and one fin / stack pitch along z. The width walls
are symmetry planes, i.e. neighbouring fins. Currents are per um of
effective width (2 H_fin + W_fin = 105 nm per fin; 2 x 3 x (25 + 5) =
180 nm per stack).

## The flows

**FinFET** (`template: finfet`, bulk, gate-first):
1. substrate; fin etch 70 nm (50 nm fin + 20 nm in the STI)
2. STI fill and CMP; punch-through stopper implanted through the STI,
   peaked at the fin base
3. fin reveal: STI recess by 50 nm
4. gate dielectric (conformal), TiN fill, gate etch, dielectric strip
   (everything exposed, the fin side walls included)
5. SiN spacer; spacer etch that clears the fin side walls outside the
   gate + spacer footprint
6. S/D fin recess, stopping 5 nm above the STI, so the epi stays above the
   PTS (the sub-fin under the gate was the off-state leakage path)
7. selective epi from the exposed Si, capped 5 nm above the fin top, with
   its analytic in-situ doping (below)
8. ILD, CMP, TiN trench contacts recessed 5 nm into the epi

**Nanosheet** (`template: nanosheet`, replacement metal gate):
1. substrate, then the superlattice: 3 x (SiGe 10 nm, Si 5 nm)
2. stack etch, STI, PTS, STI recess to the stack bottom
3. dummy poly gate; outer spacer (cleared from the stack side walls)
4. S/D recess through the stack
5. SiGe lateral recess (5 nm, isotropic, SiGe only), inner-spacer SiN
   fill, and a directional etch-back to the sheet ends. An isotropic
   etch-back deep enough to clear the sheet ends also hollowed the
   pockets, because the deposit's surface dips at every gap mouth.
6. S/D epi from the sheet ends and the mesa
7. ILD + CMP to open the dummy gate; dummy-gate pull; channel release
   (all SiGe removed)
8. gate dielectric wrapping every sheet; TiN fill between the sheets;
   gate CMP; ILD cap; contacts

<p align="center">
  <img src="img/finfet_gatecut.png" alt="FinFET cut through the gate" width="48%">
  <img src="img/nanosheet_gatecut.png" alt="Nanosheet cut through the gate" width="48%">
</p>
<p align="center">
  <img src="img/finfet_cut.png" alt="FinFET mid-width cutaway, doping" width="48%">
  <img src="img/nanosheet_cut.png" alt="Nanosheet mid-width cutaway, doping" width="48%">
</p>
<p align="center">
  <img src="img/finfet_cuts.png" alt="FinFET 2D cuts: materials and net doping" width="49%">
  <img src="img/nanosheet_cuts.png" alt="Nanosheet 2D cuts: materials and net doping" width="49%">
</p>

## New fab operations

- **`epi_doping`**: the S/D epitaxy's in-situ doping, analytic. Inside the
  epi it is the peak. Outside, it falls off as a Gaussian of the distance
  to the epi, scaled per axis:

  ```math
  N = N_{peak}\,\exp\!\Big(-\tfrac{1}{2}\big(\tfrac{d_x}{s_x}\big)^2-\tfrac{1}{2}\big(\tfrac{d_y}{s_y}\big)^2-\tfrac{1}{2}\big(\tfrac{d_z}{s_z}\big)^2\Big)
  ```

  (d_x, d_y, d_z) is the offset to the nearest epi point in the scaled
  metric (an anisotropic distance transform). For a box-shaped epi this is
  exactly the separable exp(-dx²/2sx²) exp(-dy²/2sy²). sx sets the
  junction under the spacer (the overlap), sy the tail below the epi.
  Checked against the closed form in `test_fab`.
- **Capped conformal deposit** (`to_height` with `mode: conformal`): the
  selective epi grown to a height.
- **`HKstack`**: a 0.6 nm SiO2 interlayer plus 1.4 nm HfO2, lumped into one
  2 nm layer with the same EOT (0.85 nm, eps 9.2). A 0.6 nm interlayer of
  its own would need a 0.25 nm voxel.
- **Template mesh spacing** (`mesh_h_fine`): the octree's planar default
  puts 2 nm in z at z-facing interfaces, which leaves 5 nodes across a 5 nm
  fin. FinFET and nanosheet use (1, 0.5, 0.5) nm. It matters: FinFET DIBL
  went from 52 to 66 mV/V and Ioff tripled with the finer mesh.

## Electrical results

RESULTS_PLACEHOLDER

## Known issues / next

- **FinFET walk time.** About half of it goes to steps near threshold
  where the default velocity-saturation force ("hybrid", the quasi-Fermi
  gradient where carriers are dense) stalls GMRES. 12 of the 14 s of a
  failed attempt are Krylov iterations. With the edge-field force the same
  step converges in 4 iterations.
- **Nanosheet runtime.** The stall rows are electron continuity in the
  p-type PTS region under the stack, where electrons are a negligible
  minority and phin is nearly undetermined. ILU-GMRES leaves them at
  |F| ≈ 8, which blocks the |F|_inf acceptance test and shrinks the
  line-search steps. Exact LU converges the same step in 6 iterations. A
  density-weighted acceptance and merit for continuity rows is the likely
  cure (shared Newton logic: robustness matrix first).
- Corners of fins and sheets take the larger Lambda of their two lines (a
  first pass; 2D cross-section Schrodinger slices, quantum plan P4, would
  replace it).
- PMOS variants (SiGe:B 5e20 epi) work through `type: pmos` but are not
  validated yet.
