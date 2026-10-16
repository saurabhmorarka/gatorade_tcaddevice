# Schrodinger-Poisson channel charge: details

[Back to the README](../README.md)

Status: phase P1 (1D) done. P2 (2D MOSFET) and P3 (3D fab planar) are next.
Plan: `plans/quantum_plan.md`. Numbers: DEVELOPMENT_LOG 69.

## What it does

Classical drift-diffusion puts the inversion electrons right at the
Si/SiO2 interface. In reality the wavefunction vanishes at the oxide wall:
the charge peaks about 1 nm below the interface and the lowest subband sits
above the band edge. Here the channel charge comes from the effective-mass
Schrodinger equation, solved on lines across the channel body. It is fed
back into the drift-diffusion/Poisson Newton solve as a quantum potential
Lambda:

```math
-\frac{\hbar^2}{2}\frac{d}{ds}\Big(\frac{1}{m_z}\frac{d\psi_k}{ds}\Big)+U(s)\,\psi_k=E_k\,\psi_k,\qquad
n_{QM}(s)=\sum_{v}\frac{g_v m_{d,v}k_BT}{\pi\hbar^2}\sum_k|\psi_k(s)|^2 e^{(E_F-E_k)/k_BT},\qquad
n=n_i e^{(\psi-\phi_n-\Lambda_n)/V_t}
```

## The pieces (`quantum/`)

- `bands.py`: the subband masses for any confinement direction
  (Stern-Howard). (100) gives Delta2 (m_z 0.916, m_d 0.19, g = 2) and Delta4
  (0.19, 0.417, 4). (110), the FinFET sidewall, gives (0.315, 0.324, 4) and
  (0.19, 0.417, 2). Holes are parabolic HH/LH/SO for now.
- `schrodinger1d.py`: BenDaniel-Duke finite differences on a uniform line
  (0.1 nm), with the mass on the half-points. The line runs 1 nm into the
  oxide, so the wavefunction penetrates the barrier (3.15 eV, m_ox = 0.5).
  Solver: tridiagonal LAPACK, lowest 20 levels per valley, about 1 ms per
  valley.
- `lines.py`: a line through the body, and Lambda on the drift-diffusion
  nodes it crosses. A line ends either:
  - at a second oxide wall (double gate, FinFET fin, nanosheet), which
    gives volume inversion with no special case; or
  - at depth D_q = 20 nm in a bulk body. Over the last quarter of the line
    the density blends back to classical, which hides the artificial
    bottom wall.
- `coupling.py`: the predictor-corrector loop (Trellakis 1997). It runs a
  Newton solve with Lambda fixed (the density keeps its exp(psi/Vt)
  response), then the line solves, then a new Lambda, until Lambda moves
  by less than 1 mV. It takes 2-3 outer iterations per bias point.
- `mos1d.py`: the 1D MOS-cap and double-gate drivers.
- `validate_1d.py`: the table and the figure below.

**Charge conservation on any mesh.** Lambda at node i is set so that the
node's drift-diffusion charge n_i V_i equals the quantum charge of its span
on the line:

  Lambda_i = -Vt ln( int n_QM / (n_cl,i V_i) )

The line integrates the piecewise-linear density exactly. Oxide penetration
charge goes to the interface node. So the coarse 0.5 nm fab mesh holds the
right inversion charge. With the 1e18 MOS-cap at VG = 2 V, the quantum
Ninv on a 0.5 nm mesh is within 0.3% of the fine mesh; classical is within
2.5%.

**Statistics.** Subband occupation is Boltzmann by default, matching the
rest of the solver. Then Lambda depends on psi only, because the
quasi-Fermi level cancels. `statistics="fermi"` uses 2D Fermi-Dirac
ln(1 + e^eta), with one Fermi level per line. It lowers Cinv by another
3-4% at 1e13 cm^-2 (inversion-layer degeneracy).

## Validation (1D, `python3 -m quantum.validate_1d`)

- *Line solver vs closed forms:*
  - Airy triangular-well levels within 0.2%;
  - infinite well within 0.4%;
  - the flat-band subband sum gives the band's own Nc within 0.07%.
- *Convergence:* 10 vs 20 vs 40 levels, D_q 12 vs 20 vs 30 nm, and a
  penetration depth of 0.5 vs 1 nm all change Vt by under 0.1 mV.
  - Line spacing 0.1 vs 0.05 nm: about 2 mV and 0.6% in Cinv.
  - Hard wall vs penetration: +3-6 mV and -0.7% Cinv.

Bulk MOS-cap, tox = 1.5 nm, metal gate with V_FB = 0, (100):

| Na | dVt at Ninv = 1e11 | dVg at 1e12 | Cinv/Cox classical | Cinv/Cox SP | centroid classical | centroid SP |
|---|---|---|---|---|---|---|
| 1e17 | +29 mV | +46 mV | 0.944 | 0.836 | 0.54 nm | 1.44 nm |
| 1e18 | +83 mV | +97 mV | 0.951 | 0.851 | 0.25 nm | 1.07 nm |

Both Cinv and the centroid are taken at Ninv = 1e13 cm^-2. In Cinv,
quantization adds about 0.2 nm of electrical oxide thickness. The Vt shift
of +83 mV at 1e18 is inside the +50-100 mV the plan expected.

*Against published Schrodinger-Poisson* (`python3 -m quantum.compare_literature`):
the Vt shift vs doping of Jallepalli et al. (IEEE TED 44, 297, 1997), as
replotted in Asenov et al. (Nanotech 2002, Fig. 1; read off the plot,
+-5 mV). That figure does not state tox or the threshold criterion. With
tox = 5 nm, a hard wall and a constant 3e11 cm^-2:

| Na | 1e17 | 2.5e17 | 5e17 | 1e18 | 2e18 | 3e18 | 5e18 |
|---|---|---|---|---|---|---|---|
| Jallepalli | 33 | 57 | 82 | 125 | 190 | 245 | 330 mV |
| this code | 42 | 63 | 88 | 125 | 181 | 226 | 304 mV |

That is within 10% from 2.5e17 to 5e18. At 1e17 the difference is 9 mV,
about the read-off error. With oxide penetration the shift is 5-10% lower.
At tox = 1.5 nm all shifts are 30-45% smaller: (1 + Cdep/Cox) carries the
surface-potential shift to the gate.

Symmetric double gate, tox = 1.5 nm, midgap gates, undoped body:

| t_body | dVt at 1e11 | charge peak at 1.2 V | n(center) / n(max) |
|---|---|---|---|
| 3 nm | +55 mV | center (volume inversion) | 1.00 |
| 5 nm | +29 mV | 1.1 nm off the walls, merging | 0.75 |
| 10 nm | +11 mV | 1.0 nm off each wall | 0.16 |
| 20 nm | +6 mV | 1.0 nm off each wall | 0.03 |

Classically the charge always peaks at the walls. The 2 nm ground state is
0.080 eV vs 0.103 eV for an infinite well; the difference is penetration
(the well is effectively 2 x 0.16 nm wider).

<p align="center">
  <img src="img/quantum_sp_1d.png" alt="Schrodinger-Poisson in 1D: density vs depth, C-V, centroid, double-gate volume inversion" width="98%">
</p>

## Found on the way

`core/mesh.py` `build_mos_grid` dropped the -t_ox oxide node. The
expression `-np.linspace(t_ox, 0)[::-1]` runs 0 -> -t_ox, so `[:-1]`
removed -t_ox instead of the duplicate 0. Every 1D MOS-cap solve therefore
had a 1.39 nm oxide when 1.5 nm was asked for, while `semiconductor_charge`
divided by 1.5 nm: Gauss's law was off by 7%. Fixed. The `mos_metal` and
`mos_poly_*` goldens were recaptured (metal: C/Cox in depletion +7.7% =
1.5 / 1.393; poly: < 1%).

## Next

- P2: 2D MOSFET. Lines down from the gate oxide; the same Lambda goes into
  `newton_solve_2d` as a separate shift for n and for p.
- P3: 3D fab planar. Lines from every gate-oxide interface point under the
  gate; a lattice-spacing study (center only, then 4, 2 and 1 nm) against
  the 72 s reference run.
- P4 (with FinFET/nanosheet): 2D cross-section slices for tri-gate corners
  and nanosheet edges.
