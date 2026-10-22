# Series-resistance partition (linear regime)

`scripts/resistance/partition.py` splits the on-state resistance of a 3D
transistor, `R_tot = Vds / Id`, into channel, spacer, epi and contact, for the
source and the drain side. DEVELOPMENT_LOG 87.

## Method: quasi-Fermi dissipation

Current is conserved, so the power the drain delivers, `Id * Vds`, is
dissipated somewhere. In the finite-volume discretisation that power splits
exactly into

- every mesh edge: `I_e (phi_i - phi_j)`, the Scharfetter-Gummel edge current
  times the drop of the carrier's quasi-Fermi potential (electrons and holes);
- every sub-grid Schottky contact node: `I_node (V_contact - phi_node)`, the
  drop from the metal to the majority quasi-Fermi level just past the
  depletion region (the `u` of the tabulated `J(N, u)`, contacts/subgrid.py).

The sum equals `Id * Vds` (printed as `closure`, 1.0000 in every case below),
and a region's resistance is `R = P_region / Id^2`: additive, no cut planes, no
flat-equipotential assumption. Regions are x-slabs about the gate, taken from
the template (`xg0`, `xg1`, `spacer`):

| region | where |
|---|---|
| channel | between the gate edges |
| spacer | under the spacer, both sides (the extension / fin under it) |
| epi | beyond the spacer up to the contact (epi, fin and sub-fin silicon) |
| contact | the sub-grid metal / semiconductor barrier |

Resistances are per fin (FinFET) or per sheet stack (nanosheet) of the real
device (the half-width solve is doubled). `Id` in uA/um uses the template's
effective width W (as everywhere else).

```
~/miniconda3/envs/tcad/bin/python scripts/resistance/partition.py --out out/rpart
... --devices finfet_pmos --ballistic off --set Lg=12 cpp=45     # one case
```

## Default output of every run

`fab simulate` now writes the partition of every Vds curve with no extra solve:
`partition_Vds<vds>.json` and `.png` in the output folder (fab/partition.py;
`partition=False` to skip), taken from the solved state nearest |Vgs| 0.7 V on
the on-state side, plus a log line `resistance partition at Vgs ...: R_total ...
channel x%, spacer ...`. The figure shows the region map (every semiconductor
node of the symmetry-plane cut coloured by its region, Schottky contact nodes
marked), the dissipation per nm along x with the regions shaded, and the
resistance bars (source light, drain dark). The partition is exact for the
state it reads, so for the linear-regime table use the Vds = 0.05 V curve
(`--vds 0.05 --vgs-max 0.7 --vgs-step 0.35` is enough). The standalone
`scripts/resistance/partition.py` is the same calculation with its own walk.

Run with the conda `tcad` Python (PETSc): with the project `.venv` (SuperLU only)
the NMOS FinFET gate walk stalled near threshold (287 Newton iterations and no
0.7 V point, vs 36 iterations with PETSc).

## Contact: rho_c ~ 1e-9 Ohm cm2 (log 88, 89)

Target (user, literature): Ti / Ti(Si,Ge)x contacts reach rho_c ~ 1e-9 Ohm cm2
at ~1e21 active doping. What the model does:

- **Surface doping.** A `contact_doping` process step caps the S/D doping at
  1e21 cm^-3 right at the contact metal (P for NMOS, B for PMOS), a Gaussian of
  2 nm below it (`ct_surface_peak`, `ct_surface_depth`; the epi bulk stays 7e20 /
  5e20). It is a max with the epi doping, not a sum, and is cut at 6 sigma.
- **NMOS barrier.** Effective Ti(Si)/Si:P barrier 0.10 eV (`contact_barrier_n_eV`;
  was 0.45 -> 0.20): rho_c 1.04e-9 at 1e21 in the 1D table.
- **TiSiGe interlayer (PMOS).** Ti reacts with SiGe to a Ti germanosilicide that
  lowers the hole barrier with the Ge fraction x of the SiGe it touches:
  Phi_Bp = max(0.40 - 0.8 x, 0.02) eV (`SchottkyContact(interlayer="TiSiGe")`;
  Si 0.40, SiGe0.3 seed 0.16, SiGe0.5 epi 0.02: Ge pins the Fermi level near the
  valence band). The plug touches the SiGe0.5 epi and the SiGe0.3 seed (~55:45
  in area), each with its own table. The seed was SiGe0.2 and is SiGe0.3 now
  (`epi_seed_x`).
- **Hole supply.** The 1D tables showed the barrier is not the limit at 1e21: rho_c
  floors at ~2.5e-9 for holes even with a 0 eV barrier, because the hole
  Richardson constant (32, the lightly doped Crowell value) under-counts the
  supply of a degenerate contact. A degenerate p contact draws on the full
  (heavy + light hole) band: A*_p = 120 m_dos/m0 ~ 90 (`contact_richardson_p`),
  with the hole tunneling mass 0.25 (`contact_m_tunnel_p`). That is a
  calibration. SiGe0.5 at 1e21: 0.80e-9; with the seed 1.0e-9 (3D: 0.94-1.03e-9).

Before (log 87): 3.6e-9 NMOS / 1.5e-8 PMOS. In the 3D runs rho_c = R_contact x A
(A the metal interface area of the full fin / stack).

The leak that taught a lesson: the first version of the surface doping measured
distance from the contact metal's level set, which runs on down the whole column
under a trench, and doped a continuous n+ path to the body contact (7 uA of body
current, 41% of the drain current; the source contact then looked 3x better
than the drain). It now uses the labelled metal nodes
(`test_contact_doping_is_a_cap_at_the_metal_only`).

## Result: |Vds| 0.05 V, |Vgs| 0.7 V, octree mesh, default mobility stack

Resistances in kOhm per fin / sheet stack. Ballistic k at the theory default,
Lg 15 nm.

| device | ballistic | Id (uA/um) | R_tot | channel | spacer | epi | contact | shares ch/sp/epi/ct (%) | rho_c S / D (1e-9 Ohm cm2) |
|---|---|---|---|---|---|---|---|---|---|
| FinFET NMOS | off | 251 | 1.89 | 0.60 | 0.33 | 0.38 | 0.59 | 31/17/20/31 | 1.01 / 1.08 |
| FinFET NMOS | on | 213 | 2.24 | 0.93 | 0.32 | 0.40 | 0.59 | 41/14/18/26 | 1.02 / 1.09 |
| FinFET PMOS | off | 188 | 2.53 | 1.40 | 0.43 | 0.27 | 0.43 | 55/17/11/17 | 0.94 / 0.98 |
| FinFET PMOS | on | 152 | 3.13 | 2.00 | 0.43 | 0.28 | 0.43 | 64/14/9/14 | 0.94 / 0.98 |
| nanosheet NMOS | off | 305 | 0.91 | 0.32 | 0.18 | 0.13 | 0.28 | 35/19/15/31 | 0.99 / 1.06 |
| nanosheet NMOS | on | 254 | 1.09 | 0.50 | 0.17 | 0.14 | 0.28 | 45/16/13/26 | 0.99 / 1.06 |
| nanosheet PMOS | off | 169 | 1.64 | 1.02 | 0.28 | 0.11 | 0.23 | 62/17/7/14 | 0.99 / 1.03 |
| nanosheet PMOS | on | 140 | 1.98 | 1.36 | 0.28 | 0.11 | 0.23 | 69/14/6/12 | 0.99 / 1.03 |

Readings:

- Contact is 12-31% of R_tot (it was 52-79% at log 86, 32-46% at log 87); the
  channel is the largest part (31-69%). rho_c is 0.94-1.09e-9 in all eight cases,
  source and drain within 7%.
- Ballistic (Lg 15) only raises the channel part (+55% NMOS FinFET, +40% PMOS).
- Spacer 14-19%, epi 6-20% (the epi is short and heavily doped; the PMOS SiGe:B
  epi is the smallest).
- The partition is exact for the state it reads and closes to 1.000.

## Crystal orientation in mobility and stress

A (001) wafer is the default; `channel_dir` (110 default, or 100) picks the
channel direction. One device frame feeds the surface-orientation sets, the
elastic stiffness (process stress) and the piezoresistance tensor
(mechanics.elastic.channel_frame): with <110> the FinFET side walls are (110),
the fin top and the nanosheet faces (100); with <100> (45 degrees rotated) the
side walls become (100). The inversion-layer mobility already used the per-node
(100) / (110) / (111) sets (docs/mobility.md); the new part is that the frame is
a template parameter reaching mobility, stress and piezoresistance together.
FinFET, |Vds| 0.05 V, |Vgs| 0.7 V, ballistic on: NMOS <100> channel is identical
to <110> (the (110) electron set is the (100) set), PMOS <100> R_channel 2.63 kOhm
vs 2.00 (+31%: the (110) hole sets are 2x better), Id 125 vs 152 uA/um.
Stress: the full piezoresistance tensor (pi11, pi12, pi44 rotated into the
frame) is applied along every edge, so shear and the <100> / <110> difference
come out; surface-specific (confinement) coefficients are not modelled (no
published (110) sidewall pi set found).

## The FinFET PMOS contact-resistance wobble (octree)

Id,lin of the FinFET PMOS jumped 39/68/38/62 uA/um over Lg 12/15/20/30 nm
(cpp = Lg + 33 nm), and the partition showed why: the drain-contact resistance
alternated 8.2 / 2.6 / 8.1 / 2.6 kOhm while its area (222.5 nm^2) and doping
(5e20) were identical.

Cause. The PMOS plug touches two semiconductors, the SiGe0.5 epi and the
lower-Ge SiGe0.2 seed layer (Eg 0.75 / 0.97 eV), in ~55:45 area. The contact
took its Schottky table from the material most of its NODES carry (a node
count vote), and the octree's node count on each side varies with Lg (drain
238 vs 282 nodes at Lg 12, 333 vs 297 at 15). The vote flipped between the two
tables, whose rho_c differ by 3x. The source never flipped (441 vs 285 nodes).

Fix (contacts/subgrid.py). A contact is tabulated per material it touches
(`contact_groups`), each node using its own material's table. The contact
resistance is now 7.4 / 7.3 / 7.3 / 7.3 kOhm and Id,lin 52.4 / 51.9 / 50.3 /
47.9 uA/um over the same Lg (ballistic off): smooth and monotonic.

## Terminal current of nanosheet contacts

While closing the energy balance the nanosheet came out 6-11% high. Two nodes
per contact sit on the metal boundary with no metal interface area (a voxel
corner), so the sub-grid contact does not release them and they stay Dirichlet
at the terminal voltage, and `contact_current` summed only the Schottky nodes,
under-reporting the terminal current by 5.5% (NMOS) / 11% (PMOS). Fixed in
solver2d/current.py (`_pinned_node_current`): nanosheet Id,lin NMOS
161.3 -> 170.6, PMOS 47.3 -> 52.5 uA/um at Vgs 0.7 V; the fields and the
FinFET numbers are unchanged.

Open: these pinned nodes are ideal-ohmic shorts carrying 3-6% of the current
each, so the nanosheet contact resistance is understated. Releasing them (a
zero-area free node) made the nanosheet solves stall in a trial, so it needs
a proper treatment.
