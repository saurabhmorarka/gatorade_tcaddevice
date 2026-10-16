# FinFET and nanosheet (GAA) flows: plan

Status: first pass IMPLEMENTED 2026-10-14 (DEVELOPMENT_LOG 70,
docs/finfet_nanosheet.md). The user set the dimensions: fin 50 nm tall and
5 nm wide; 3 sheets 5 nm thick, 25 nm wide; CPP ~48 nm, Lg ~15 nm (2 nm
node); the rest from the literature. Q1-Q5 below were answered by
default: bulk FinFET, IL + HfO2 (lumped HKstack, EOT 0.85 nm), rounded
conformal epi, single fin / stack. The draft numbers below are the
original proposal; the implemented ones are in fab/templates.py.
Roadmap item 5, pulled forward as a lead-in while the Schrodinger-Poisson
channel charge (plans/quantum_plan.md) is built: both devices need it for
volume inversion. The user's request: own fab steps per device; analytic
S/D-epitaxy doping that tapers as a Gaussian in x and in y; the planar 3D
contact doping levels at the contacts.

## 1. What exists and what is missing

The fab engine (`fab/structure.py`) already has most of the operations:

| Need | Existing op |
|---|---|
| fin / sheet-stack patterning | pattern (rect, polygon, union masks) + anisotropic etch |
| STI fill, CMP, recess (fin reveal) | deposit fill + planarize + anisotropic etch of SiO2 to a depth |
| Si/SiGe superlattice | deposit fill to a height, repeated (Si, SiGe alternating) |
| gate dielectric wrapping fins and sheets | conformal deposit (exact distance-field offset) |
| replacement-metal-gate fill between released sheets | deposit fill (everything below a height outside the solid, so cavities too) |
| SiGe lateral recess (inner spacers) and channel release | isotropic etch restricted to `materials: [SiGe]` |
| inner-spacer fill | conformal SiN deposit, then an isotropic etch back |
| selective S/D epitaxy | conformal deposit `on: [Si]` (grows only from exposed Si) |
| high-k | HfO2 already in INSULATOR_EPS (22) |

Missing:

1. **`epi_doping` step (new)**: analytic S/D-epitaxy doping, as you asked.
   - Inside the epi region: the contact-level peak, the same levels as
     the planar 3D contacts:
     - NMOS Si:P, 7e20;
     - PMOS Si0.5Ge0.5:B, 5e20 (the PMOS S/D stack in memory).
   - Outside it, a separable Gaussian tail of the distance to the region:

         N(r) = N_peak * exp(-dx^2 / 2 sx^2) * exp(-dy^2 / 2 sy^2)

     where dx, dy are the x (channel-direction) and y (depth) distances
     outside the epi volume (0 inside).
     - sx (~2-4 nm) sets the junction position and abruptness under the
       spacer: the overlap.
     - sy (~5-10 nm) sets the tail below the recess, which feeds
       sub-fin / bottom-sheet leakage.
   - z (across the fin) is not tapered: the epi wraps the fin.
   - It uses the same level-set distances as `implant`. ~40 lines.
2. **`superlattice` helper** in the template: N x (SiGe t_sige, Si
   t_sheet) as repeated fill deposits. Template code, no engine change.
3. **Voxel / mesh**: 5-6 nm sheets and 1-2 nm dielectrics need 0.5 nm in y
   (have it) and 0.5 nm in z for the fin width.
   - The octree already refines at every interface.
   - Check the mesh gate on a fin and a sheet stack (no slivers, exact
     widths), per the structure-exactness rule.
4. **Quantum box on fins and sheets** (quantum/device.py, already general):
   - lines start at every gate-dielectric interface, so fin sidewalls
     (normal +-z, (110) masses by default) and sheet tops/bottoms come out
     automatically, wall to wall: volume inversion;
   - two gaps to close:
     - (a) *symmetry plane*: a half-width solve (half_z) cuts the fin or
       sheet at its center, so a line would end on the mirror plane, not
       on a wall. Fix: mirror the column across the plane (a wall-to-wall
       line on the reflected profile), or solve the full width.
     - (b) *corners*: a node near a fin top corner is on both a sidewall
       line and a top line. First pass: keep the larger Lambda (the
       stronger confinement). P4 of the quantum plan (2D cross-section
       slices) replaces it.

## 2. FinFET template (`template: finfet`)

Bulk FinFET, single fin first (multi-fin = a union mask + width
parameter), gate-first like planar (the structure is the same as a
replacement-gate flow; keeps the flow short).

1. Substrate p (NMOS), well implant.
2. Fin mask (W_fin), anisotropic Si etch to H_fin + sub-fin depth.
3. STI fill, CMP, then recess the STI to expose H_fin (fin reveal).
4. Punch-through stopper: Gaussian implant peaked at the fin base
   (~1e18), so the sub-fin is not a leakage path.
5. Gate stack, conformal:
   - IL SiO2 (0.5 nm);
   - HfO2 (1.5-2 nm; EOT ~0.8-0.9 nm);
   - work-function metal / TiN fill;
   - gate mask + etch, stopping on the dielectric;
   - dielectric strip.
6. Spacer: conformal SiN plus anisotropic etch with over-etch, so the fin
   sidewalls are cleared outside the gate for the epi.
7. S/D recess: anisotropic Si etch of the fin outside the spacers
   (optional, `sd_recess`).
8. S/D epitaxy: conformal deposit `on: [Si]` of t_epi (Si for NMOS,
   SiGe for PMOS), then `epi_doping` (peak, sx, sy).
9. ILD fill + CMP, contact trench, TiN contact fill on the epi (Schottky
   S/D, as planar, default on).

Default parameters (a 7-5 nm-class bulk FinFET):

| Lg | W_fin | H_fin | spacer | IL / HfO2 | channel doping | PTS |
|---|---|---|---|---|---|---|
| 20 nm | 7 nm | 42 nm | 7 nm | 0.5 / 1.7 nm | 1e16 | 1e18 at the fin base |

## 3. Nanosheet template (`template: nanosheet`)

Replacement-metal-gate flow (needed: the sheets are released before the
gate fills between them).

1. Substrate, then a superlattice:
   - 3 x (SiGe0.3 10 nm / Si 5 nm);
   - a bottom isolation option: a thicker bottom SiGe that is later
     replaced by dielectric, or a doped sub-sheet. First pass: a doped
     sub-sheet (PTS-like implant).
2. Stack mask (W_sheet), anisotropic etch through the stack into the
   substrate, STI fill / CMP / recess to the stack bottom.
3. Dummy gate (Poly) + mask + etch; outer spacer (SiN conformal +
   anisotropic etch).
4. S/D recess through the stack outside the spacers.
5. Inner spacers:
   - isotropic SiGe etch (lateral recess, ~ the spacer thickness);
   - conformal SiN fill of the recesses;
   - isotropic SiN etch back to the sheet ends.
6. S/D epitaxy from the exposed sheet ends (conformal `on: [Si]`), then
   `epi_doping`.
7. ILD fill + CMP to the dummy-gate top; dummy-gate removal (isotropic /
   anisotropic etch of Poly).
8. Channel release: isotropic SiGe etch (removes all SiGe between the
   sheets inside the gate cavity).
9. Gate stack: conformal IL + HfO2 (wraps every sheet), then a TiN fill of
   the cavity, including between the sheets.
10. Contacts as in the FinFET.

Default parameters:

| Lg | sheets | t_sheet | W_sheet | gap (SiGe) | inner spacer | IL / HfO2 |
|---|---|---|---|---|---|---|
| 16 nm | 3 | 5 nm | 25 nm | 10 nm | 5 nm | 0.5 / 1.5 nm |

## 4. Validation (per device, before electrical work)

- Structure:
  - fin width / sheet thickness / gaps exact to the voxel;
  - no stray slivers (the existing `check()` + mesh gate);
  - the snapshots in tcadview, step by step.
- Doping: the epi profile plotted along x and y through the junction;
  junction position vs sx.
- Mesh: octree quality gate; node count and build time.
- Electrical (drift-diffusion, then + quantum):
  - Id-Vg at 0.05 / 0.75 V;
  - Vt, SS (expect ~65-70 mV/dec for these geometries), DIBL, Ion;
  - quantum on vs off: Vt shift from fin-width / sheet-thickness
    confinement (thin-body: tens of mV), volume-inversion profiles.
- Runtime: the ~1 min 3D I-V goal. FinFET and nanosheet meshes are bigger
  than planar, so measure first, then optimize.
- Then the deferred octree defaults study on the FinFET (option B), and
  `test_octree` tolerance 3% -> 1%.

## 5. Order of work

1. `epi_doping` step + test; superlattice helper.
2. FinFET template + structure validation + mesh + DD Id-Vg.
3. Quantum on FinFET: the symmetry-plane line fix, the corner rule,
   sidewall (110) masses, volume inversion vs W_fin.
4. Nanosheet template (the inner-spacer and release steps) + the same
   validation.
5. PMOS variants (SiGe S/D epi 5e20 B) for both.

## 6. Questions for the user

- Q1: Bulk FinFET (STI + punch-through stopper) as the first FinFET, not
  SOI?
- Q2: Gate stack IL SiO2 + HfO2 (EOT ~0.85 nm) for FinFET and nanosheet,
  not the planar 1.5 nm SiO2?
- Q3: S/D epi shape: the conformal (rounded) growth from exposed Si is the
  first pass, not faceted diamonds. OK?
- Q4: Default sizes in the tables above (or your own numbers)?
- Q5: Single fin / single sheet stack first, with multi-fin later?
