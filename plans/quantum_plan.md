# Schrodinger-Poisson channel charge: plan (first pass)

Status: APPROVED 2026-10-13 with the recommended answers to Q1-Q4 (quantum
box with a lattice knob; Boltzmann default, Fermi-Dirac switch; inversion
carrier only; off by default until P3). Branch `quantum-sp`.
P1 (1D) DONE: DEVELOPMENT_LOG 69, docs/quantum.md. P2 (2D) and P3 (3D fab
planar, lattice study) DONE, plus the FinFET/nanosheet box (symmetry-plane
mirror, corner rule) and the in-Newton coupling: DEVELOPMENT_LOG 70. Next:
P4 (2D cross-section slices for fin/sheet corners), Fermi-Dirac in 3D,
holes for PMOS.

## 1. Goal

Today the channel charge comes from Poisson + Boltzmann statistics. In that
model the inversion electrons pile up right at the Si/SiO2 interface. In a
real device the wavefunction must vanish (almost) at the oxide wall. So:

- the charge peak sits about 1-2 nm below the interface (the "dark space");
- the lowest subband sits above the conduction-band edge, which raises Vt
  (by about +50-100 mV at 1e18 cm^-3);
- the charge centroid moves away from the gate, which lowers Cinv. With
  tox = 1.5 nm, that adds about 0.3-0.4 nm to the electrical oxide
  thickness.

In thin bodies (FinFET fins, nanosheets) the two gated interfaces face each
other. The confinement is then set by the body thickness, not by the
surface field. For thin bodies (< ~7 nm) the ground state peaks at the
CENTER of the body. This is volume inversion. A model that treats each
interface on its own (van Dort, the simple MLDA) cannot produce it.

Sanity numbers (triangular well, Airy ground state, (100) Si):

| surface field | valley  | m_z   | E0      | <z>     |
|---------------|---------|-------|---------|---------|
| 0.3 MV/cm     | Delta2  | 0.916 | 78 meV  | 1.7 nm  |
| 0.3 MV/cm     | Delta4  | 0.19  | 131 meV | 2.9 nm  |
| 1.0 MV/cm     | Delta2  | 0.916 | 173 meV | 1.15 nm |
| 1.0 MV/cm     | Delta4  | 0.19  | 293 meV | 1.95 nm |

The fab planar device meshes the channel at 0.5 nm in depth. That is coarse
compared with these lengths, so the coupling must conserve charge per
control volume (section 4.3).

## 2. Candidate models

| model | what it is | volume inversion | cost in 3D | verdict |
|---|---|---|---|---|
| van Dort | bandgap widening near the surface, fitted to the Vt shift | no | ~0 | no: not a charge model |
| MLDA (Paasch-Uebensee) | local density times a factor in distance-to-wall | only with ad hoc multi-wall superposition | ~0 | no: wrong for thin bodies |
| density-gradient (Ancona) | an extra PDE per carrier (quantum potential Lambda) | yes, qualitatively; needs calibration | +1-2 unknowns per node (3 -> 4-5): linear solve ~1.5-2.5x | keep as plan B, calibrated against SP |
| Schrodinger-Poisson on lines/slices | solve the effective-mass Schrodinger equation across the channel body, map the quantum density back | yes, from first principles | small, if the slices are batched | **recommended** |

Your request is real Schrodinger-Poisson, so it is the main path. Density-
gradient stays the fallback if SP-on-slices proves too slow or too fragile
in 3D. The 1D SP solver built here would then be its calibration reference.

## 3. Where to solve the Schrodinger equation (your question)

Three options:

**(a) One line at the device center.** One Schrodinger line through the
middle of the channel. Its correction is applied to every channel column as
a function of depth.
- Cheapest: one solve per update.
- Wrong wherever the channel is not uniform:
  - along x under drain bias: near the drain, the surface field and phi_n
    differ from the center, and the barrier top that sets the current sits
    near the source;
  - along z at the STI edge (corner field crowding);
  - at the source/drain ends of the gate.
- Does not extend to FinFET or nanosheet.

**(b) A "quantum box" under the gate oxide.** Lines start at every
gate-dielectric interface point inside the gate footprint. Each runs along
the interface normal into the body. Each line gets its own 1D solve using
the local psi and phi_n. The correction is mapped back to the drift-
diffusion nodes inside the box.
- Cost: number of lines x ~1 ms. At a 2 nm lattice on the half-width
  planar device, that is ~225 lines, ~0.3-0.5 s per update, single thread.
  The lines are independent, so they batch and parallelize.

**(c) Full 2D or 3D Schrodinger.**
- 2D cross-sections at each x slice: needed for tri-gate corners and
  nanosheet edges.
- Full 3D: too expensive and not needed. In a DD-level device the
  transport direction is not quantized.

**Recommendation: (b), with the lattice spacing as a knob.** Option (a) is
then just the coarsest setting (one line per x position, or a single
line). We measure how coarse we can go instead of guessing. 2D cross-section
slices (c) come in the FinFET phase.

How each geometry maps onto lines:

| device | where lines start | line direction | where a line ends |
|---|---|---|---|
| planar | top gate-oxide interface | -y | depth D_q (~20 nm, tapered) |
| FinFET (double gate) | left sidewall oxide | +z | the right sidewall oxide (two hard walls: volume inversion comes out by itself) |
| nanosheet | top surface of each sheet | -y | the bottom surface of the same sheet |
| tri-gate top and fin corners, nanosheet edges | - | - | needs 2D cross-section slices (phase 4) |

The line's end condition does all the work. "Runs until it hits oxide
again" gives volume inversion. "Runs to depth D_q in bulk" gives the planar
triangular well. The code takes a general start point and normal; nothing
is planar-specific.

## 4. Physics and numerics

### 4.1 Schrodinger along a line (effective mass, BenDaniel-Duke)

    -(hbar^2/2) d/dy [ 1/m_z(y) dpsi_k/dy ] + U(y) psi_k = E_k psi_k
    U(y) = -q psi(y) + chi offset (Si: 0; oxide: +3.1 eV for electrons,
           4.5-4.7 eV for holes)

- Grid: a fine uniform auxiliary 1D grid (0.1 nm). It extends ~1 nm into
  the oxide, so the wavefunction can penetrate the barrier (m_ox ~ 0.5).
  Hard walls are a switch, kept for testing.
- Valleys: Delta2 and Delta4 for electrons, from the general ellipsoid
  formulas (Stern-Howard). That gives m_z, the density-of-states mass m_d
  and the degeneracy g for ANY surface normal: (100) planar, (110) FinFET
  sidewalls.
- Holes: parabolic HH/LH/SO masses (first pass; a k.p model is out of
  scope).
- Solver: tridiagonal eigensolver (`eigh_tridiagonal`), lowest N_k states
  per valley (start with N_k = 20, check convergence). States above the
  cutoff are filled with the classical density, so subthreshold tails stay
  right.

### 4.2 Quantum density

    n_QM(y) = sum_valleys sum_k |psi_k(y)|^2 * g * (m_d kT / pi hbar^2)
              * F0((E_F - E_k)/kT),   E_F = -q phi_n (local, interpolated)

Here F0(eta) = ln(1 + e^eta) (2D Fermi-Dirac), or e^eta (Boltzmann).

**Open choice, Q2.** Fermi-Dirac is physically right in strong inversion.
But the rest of the solver is Boltzmann. The correction would then also
fold in degeneracy, but only inside the quantum box. Default to Boltzmann
for a clean comparison, with Fermi-Dirac as a switch?

### 4.3 Coupling to drift-diffusion: a quantum potential Lambda

The Newton system keeps its 3 unknowns (psi, phi_n, phi_p). Each node gets
a band shift, frozen during a Newton solve:

    n = ni * exp((psi - Lambda_n - phi_n)/Vt)
    p = ni * exp((phi_p - psi - Lambda_p)/Vt)

The solver already has this hook for the SiGe band offset (`dEi`). It needs
separate n/p shifts. Current is J_n = -q mu n grad(phi_n), as today.
Lambda enters through n, so transport sees the quantum repulsion
consistently.

Lambda is set per drift-diffusion node so that the charge in its control
volume equals the quantum charge (this conserves charge on the 0.5 nm mesh):

    Lambda_n,i = Vt * ln( n_cl,i * V_i / integral over V_i of n_QM )

- Clip Lambda at about 1 V. At a node right on the interface, n_QM -> 0.
- Taper Lambda to 0 over the last ~25% of a planar line (where the
  artificial bottom wall is) and across the box edges at the gate ends,
  over ~2 nm. Without the taper, a step in Lambda along x becomes a
  spurious barrier in the current path.

**Outer loop (predictor-corrector, Trellakis 1997).**

    Newton(DD with Lambda fixed) -> line solves -> new Lambda -> repeat
    until max|dLambda| < 1 mV

- The density stays exponential in psi inside Newton. This is the
  predictor that makes the loop stable, unlike plain Gummel mixing of
  densities.
- Expect 3-6 outer iterations at a new bias. Each Newton solve is warm and
  short.
- Anderson mixing on Lambda if it stalls.
- Along a sweep, the previous bias point's Lambda is the predictor.

Fully coupled alternative (Lambda's Jacobian in Newton): only if the outer
loop proves slow.

### 4.4 What changes where

- New package `quantum/`:
  - `bands.py`: valleys and masses for a surface normal;
  - `schrodinger1d.py`: line Hamiltonian and eigen-solve, batched over
    lines;
  - `lines.py`: build lines from the mesh (start, normal, end), interpolate
    psi and phi_n onto them, map Lambda back by control volume;
  - `coupling.py`: the outer loop.
- `mos/mos_solver.py`: Lambda in the 1D MOS-cap Poisson (prototype).
- `solver2d/newton_solver_qf_2d.py`:
  - `quantum_shift=(Lambda_n, Lambda_p)` through `_densities`,
    `poisson_solve`, the cold start and the quasi-Fermi box.
- `fab/simulate.py`: a `quantum:` block in the device config (`model: sp`,
  carriers, lattice spacing, D_q, N_k, statistics). Off by default until
  validated (Q4).

## 5. Phases and validation

**P1, 1D MOS-cap** (`mos/`, user rule: prototype in 1D first):
- Bulk (100), Na = 1e17 and 1e18, tox = 1.5 nm, metal gate:
  - n(y) profiles: peak ~1 nm, vanishing at the wall;
  - charge centroid vs Ns;
  - LF C-V vs classical: Cinv drop, Vt shift.
- Checks:
  - the Airy triangular-well limit at low charge;
  - published SP results (Stern 1972; Schred-type MOS-cap curves);
  - van Dort's Vt-shift formula as an order-of-magnitude check;
  - N_k, grid and oxide-penetration convergence.
- Mesh robustness: the drift-diffusion mesh at 0.5 nm and at 0.1 nm must
  give the same Qinv(Vg) within ~1%. This is the test of the control-volume
  mapping.
- Double gate: body thickness 3 / 5 / 10 / 20 nm.
  - Expect a single center peak (volume inversion) when thin, two peaks
    when thick.
  - Vt rise as the body thins (~1/t^2).
  - Check against the infinite-well limit E1 = hbar^2 pi^2 / (2 m_z t^2).
  - This is the FinFET/nanosheet physics, proven in 1D first.

**P2, 2D MOSFET** (solver2d): lines down from the gate oxide.
- Long channel: must match P1 column for column.
- Id-Vg: Vt shift, SS unchanged, Ion drop.
- Lambda continuity at the gate edges (no current kink).
- Outer-loop iteration counts.

**P3, 3D fab planar** (octree):
- Lines from the gate-oxide interface; sweep the lattice spacing (center
  only, then 4, 2 and 1 nm) against the reference run.
- Report Vt, SS, DIBL, Ion, Ioff and runtime. The reference is 72 s; the
  speed rule applies:
  - batch and parallelize the line solves;
  - consider Rust only if they dominate (5-10x rule).
- Testsuite: fast 1D SP tests (Airy limit, double-gate volume inversion,
  charge conservation), one quick 2D test, and no full sweeps (fast-
  testsuite rule).

**P4 (with FinFET/nanosheet, roadmap item 5):**
- 2D cross-section Schrodinger slices (sparse `eigsh` on the (y, z)
  section, one per x plane) for tri-gate corners and nanosheet edges.
- 1D lines remain the fast path where a body is wide in one direction.

## 6. Open questions for the user

- Q1: Where to solve: (b) quantum box with a lattice knob (recommended), or
  start with (a) a single center line?
- Q2: Subband statistics: Boltzmann by default, Fermi-Dirac as a switch?
- Q3: Carriers: the inversion carrier only (electrons for NMOS, holes for
  PMOS) in the first pass? Majority-carrier quantization (accumulation)
  hardly affects Id-Vg at 0-1.2 V.
- Q4: Default: off in `fab simulate` until P3 is validated, then on by
  default (as was done for Schottky)?

## 7. Sources (to check and cite during P1)

- F. Stern, W. E. Howard, Phys. Rev. 163, 816 (1967): subband masses for
  any surface orientation.
- F. Stern, Phys. Rev. B 5, 4891 (1972): self-consistent Si inversion
  layers.
- T. Ando, A. B. Fowler, F. Stern, Rev. Mod. Phys. 54, 437 (1982).
- A. Trellakis et al., J. Appl. Phys. 81, 7880 (1997): predictor-corrector
  SP iteration.
- M. G. Ancona, H. F. Tiersten, Phys. Rev. B 35, 7959 (1987):
  density-gradient.
- G. Paasch, H. Uebensee, phys. stat. sol. (b) 113, 165 (1982): MLDA.
- M. J. van Dort et al., IEEE TED 41, 1994: the Vt-shift model.
- Schred (nanoHUB, D. Vasileska): 1D MOS-cap SP reference.
- B. Majkusiak et al., IEEE TED 45, 1998: double-gate thin-body
  quantization and volume inversion.
- Sentaurus Device user guide, quantization chapter: the van Dort, 1D
  Schrodinger, MLDA and density-gradient options.
