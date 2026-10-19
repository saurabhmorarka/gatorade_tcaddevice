# Process stress, 2D (phase 1)

[Back to the README](../README.md)

A linear elastic stress solver that follows the process flow step by step,
so you can see how each stressor's input strain or stress becomes channel
stress, and in which direction. Phase 1 is 2D (plane strain, a cut along
the channel of a wide planar device). Numbers: DEVELOPMENT_LOG 76.

```bash
python3 -m mechanics.planar_flow                      # three flows, tables + figures in out/stress/
python3 -m mechanics.planar_flow --flow pmos_gate_last --set sige_x=0.3 liner_pmos_GPa=0
python3 -m unittest testsuite.test_mechanics          # closed-form and history checks
```

## What is solved

- **Elasticity.** Linear, no plasticity or viscous flow.
  - Every material has E and ν in `core/material_db.py`.
  - Si, Ge and SiGe also carry their cubic stiffness C11, C12, C44, which the solver uses instead.
  - The cubic tensor is rotated to the device frame. The default is a (001) wafer with a <110> channel: x = [110] along the channel, y = [00-1] depth (positive down), z = [-110] across the width.
  - SiGe stiffness follows Vegard's law.
- **Grid and boundaries** (`mechanics/fe2d.py`).
  - One bilinear element per 1 nm pixel.
  - The side walls are symmetry planes, so the result is a dense array of identical devices.
  - The bottom is held and the top is free.
- **Stressors.** Each takes a stress-free strain ε\* (an eigenstrain):
  - **SiGe epitaxy:** the lattice mismatch, f = (a_SiGe − a_Si)/a_Si. The SiGe lattice constant follows Dismukes, a(x) = 5.431 + 0.20x + 0.027x² Å, so x = 0.5 gives 1.97%.
  - **Films** (liner, STI oxide, gate metal, contact metal): an intrinsic stress σ0 (GPa, + tensile). It becomes the isotropic ε0 = −σ0(1−ν)/E that a flat film held by its substrate needs to carry σ0.
- **Process history.** Every step changes the structure on a fixed grid:
  - **Deposit or epitaxy:** new material joins stress-free in the current shape, and its own ε\* switches on.
  - **Etch, CMP or pull:** removed material releases the forces it exerted on the rest, and the structure re-equilibrates. This is where the dummy-gate pull and the cap strip act.
  - **Anneal:** chosen regions give up a fraction r of their stress, i.e. they recrystallize stress-free in their current shape. This is the elastic picture of stress memorization.
- **Per-stressor breakdown.** Everything is linear, so each stressor is carried as its own stress field. The total is their exact sum, from one factorization per step with one right-hand side per stressor.

## Default stressor inputs (rough literature values)

| stressor | default | in `--set` |
|---|---|---|
| SiGe S/D | x = 0.5 (f = 1.97%) | `sige_x` |
| STI oxide (HDP) | −0.3 GPa | `sti_GPa` |
| CESL nitride liner | −2.5 GPa PMOS, +1.5 GPa NMOS | `liner_pmos_GPa`, `liner_nmos_GPa` |
| TiN metal gate | −1.0 GPa | `gate_GPa` |
| W contact | +1.0 GPa | `contact_GPa` |
| SMT nitride cap | +1.5 GPa | `cap_GPa` |

Films vary widely with process; treat these as teaching values.

## Results

Lg 30 nm, spacer 8 nm, 44 nm S/D, 20 nm STI half-width, 40 nm SiGe recess.
Channel stress is the mean over the top 4 nm of the channel, in MPa (+ tensile):

| flow | σxx (channel) | σyy (vertical) | σzz (width) |
|---|---|---|---|
| PMOS SiGe, gate-first | −897 | +1180 | +371 |
| PMOS SiGe, gate-last (RMG) | −2674 | −106 | −205 |
| NMOS SMT (poly gate), cap only | +113 | −209 | −69 |
| NMOS SMT + edge dislocations | +706 | −343 | −80 |

**The poly pull.** In the gate-last flow, channel compression reaches −1455 MPa
after ILD CMP. Pulling the dummy poly takes it to −2895 MPa: the poly had
been holding the channel against the SiGe push. In the gate-first flow a
rigid TiN gate (E = 400 GPa) sits there from the start, and the SiGe gives only
−756 MPa. The vertical stress shows the same thing from the other side. Under
a gate, the SiGe squeezes the channel up against the gate (σyy ≈ +0.9 GPa).
With the gate pulled, the channel surface is free and σyy drops to about 0.

<p align="center"><img src="img/stress_pmos_gate_last.png" width="100%"></p>

**Stress memorization (NMOS).** The S/D and poly are amorphized and then
capped with a +1.5 GPa nitride. The anneal lets them recrystallize
stress-free in their capped shape, and the cap is then stripped. The
channel keeps +203 MPa of tension from the cap alone, after the cap is gone.

**SMT dislocations.** During that anneal, an edge dislocation forms in each
S/D at its gate-side edge. Each one is modelled as a Volterra cut
(`mechanics/dislocation.py`):
- b = a/2<110> = 0.384 nm, along the channel;
- a half plane is missing between the core (20 nm deep) and the surface.

The two dislocations add +594 MPa of channel tension, the largest single
term, for a +706 MPa total. This is the big NMOS boost of SMT. The same
tension in a PMOS would cancel its compressive gain. Turn them off with
`--set smt_disl=0`, or move the core with `--set disl_depth=...`.

The Volterra field is checked against the analytic edge dislocation (Hirth &
Lothe), with σyy and σxy within 20% at 10-20 nm from the core. σxx is offset
by the held side walls, about E·b/L.

<p align="center"><img src="img/stress_nmos_smt.png" width="100%"></p>

<p align="center"><img src="img/stress_pmos_gate_first.png" width="100%"></p>

## Checks (`testsuite/test_mechanics.py`)

- Rotated Si stiffness: C11' = (C11+C12)/2 + C44 along <110>, C13' and C55'; isotropic materials are unchanged by rotation.
- A uniform isotropic film with eigenstrain e on a substrate gives σxx = σzz = −Ee/(1−ν), with the substrate at 0 (to 1e-6).
- SiGe on Si(001) gives σ = −(C11 + C12 − 2C12²/C11)·f along both <110> and <100> (to 1e-6).
- Mesh convergence of the stress under a film edge as the pixel goes 2 → 1 → 0.5 nm.
- Depositing a stressed film and stripping it returns the substrate to zero.
- Two stressors switched on together equal the sum of each alone.
- The memorization sequence (cap, anneal, strip) leaves a stress of the opposite sign to the one under the cap.

## Known limits and next phases

- 2D plane strain only. FinFET and nanosheet cross-sections are 3D problems, which are phase 3 (octree hexes).
- The geometry here is drawn by `mechanics/planar_flow.py`, not yet taken from `fab/` step snapshots. Hooking `fab/flow.py`'s `run_step` is next.
- Dislocations are pure edge, with the line along the width, and the cut is
  vertical. Inclined {111} cuts work in the code (any `cut_dir`) but are not
  in the flow yet. The core position is a user input; there is no
  nucleation model.
- Piezoresistive mobility (phase 4) and the per-edge Δμ/μ = −Π:σ will use these stress fields.
- The magnitudes are high: a dense array, 50% Ge, plane strain, and liner and film values that are rough. The trends and their sign by direction are the teaching point.
