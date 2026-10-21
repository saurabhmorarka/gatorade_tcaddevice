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

## Contact recalibration (log 88)

Target from the literature: silicided Ti contacts on Si:P / SiGe:B reach
rho_c ~ 1-1.5e-9 Ohm cm2 (imec / patent reports 5.7e-10 to 1.5e-9 on 1e21 B
SiGe). The model's rho_c at the S/D doping, from the 1D tables:

| contact | before | now (barrier) |
|---|---|---|
| Si:P 7e20, NMOS | 3.6e-9 (0.45 eV) | 1.6e-9 (0.20 eV effective) |
| SiGe0.5:B 5e20, PMOS | 1.5e-8 (hole barrier Eg - 0.45 = 0.30 eV; 0.52 eV on the SiGe0.2 seed) | 3.6e-9 (hole barrier 0.10 eV on every SiGe, hole tunneling mass 0.25) |

Template parameters `contact_barrier_n_eV` (0.20), `contact_barrier_p_eV`
(0.10), `contact_m_tunnel_p` (0.25). PMOS does not reach 1.5e-9 at the user's
5e20 B: even at 1.5e21 and 0.10 eV the 1D model gives 2.5e-9 (one tunneling
mass, no light-hole / non-parabolic bands), so the PMOS contact remains ~2.5x
above the best measured values.

## Result: |Vds| 0.05 V, |Vgs| 0.7 V, octree mesh, default mobility stack

Resistances in kOhm per fin / sheet stack. Vgs 0.7 V; ballistic k at the theory
default, Lg 15 nm.

| device | ballistic | Id (uA/um) | R_tot (kOhm) | channel | spacer | epi | contact | shares ch/sp/epi/ct (%) | rho_c S / D (1e-9 Ohm cm2) |
|---|---|---|---|---|---|---|---|---|---|
| FinFET NMOS | off | 211.4 | 2.25 | 0.59 | 0.33 | 0.40 | 0.93 | 26/15/18/41 | 1.6 / 1.7 |
| FinFET NMOS | on | 183.6 | 2.59 | 0.92 | 0.32 | 0.42 | 0.93 | 36/12/16/36 | 1.6 / 1.7 |
| FinFET PMOS | off | 130.2 | 3.66 | 1.28 | 0.41 | 0.29 | 1.67 | 35/11/8/46 | 3.6 / 3.8 |
| FinFET PMOS | on | 112.2 | 4.24 | 1.88 | 0.41 | 0.29 | 1.66 | 44/10/7/39 | 3.6 / 3.8 |
| nanosheet NMOS | off | 260.6 | 1.07 | 0.32 | 0.18 | 0.15 | 0.43 | 30/17/14/40 | 1.5 / 1.6 |
| nanosheet NMOS | on | 222.3 | 1.25 | 0.50 | 0.18 | 0.15 | 0.43 | 40/14/12/34 | 1.5 / 1.6 |
| nanosheet PMOS | off | 128.6 | 2.16 | 0.96 | 0.26 | 0.13 | 0.81 | 44/12/6/37 | 3.6 / 3.5 |
| nanosheet PMOS | on | 110.8 | 2.51 | 1.30 | 0.27 | 0.13 | 0.81 | 52/11/5/32 | 3.6 / 3.5 |

Readings:

- Contact falls from 52-79% to 32-46% of R_tot; the channel (26-52%) is now the
  largest part for PMOS and with ballistic on. Ballistic (Lg 15) only raises the
  channel part (+55% NMOS FinFET ... +40% PMOS).
- Spacer is 10-17% and epi 5-18% (PMOS epi 5-8%: SiGe:B at 5e20 is a good
  conductor and the epi is short; it was 3-6% before).
- Source and drain contact resistivities agree within 5% (the earlier 10-14%
  source/drain contact spread came from the mesh and the near-contact split; the
  spacer / epi / channel terms agree within 3%).
- Spacer doping: along the fin centre the doping falls 7e20 -> 5.6e19 at the gate
  edge -> 3.5e18 1.5 nm inside the gate (about 2.7 nm / decade), which is
  already the erfc a ~2 nm diffusion length gives from the epi edge, so a
  further "diffusion" step would not raise R_spacer in this model. If the
  intended profile is steeper (lighter extension) that is an `epi_sx` /
  `ext_*` choice, to be decided.

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
