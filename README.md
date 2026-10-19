# gatorade_tcaddevice

TCAD device simulators built from scratch in Python/NumPy/SciPy (hot loops
in Rust): 1D, then 2D, now 3D. Newest work first; each section links to
its details.

## Process stress through the flow, 2D ([details](docs/stress.md))

A linear elastic stress solver (`mechanics/`) that follows the process flow
step by step and splits the channel stress by stressor: SiGe S/D mismatch,
liners, STI oxide, metal gate, contacts, and stress memorization (anneal).
Si and SiGe are anisotropic (C11, C12, C44, <110> channel); every material
has E and nu in the material database. In a PMOS with SiGe S/D, pulling the
dummy poly gate doubles the channel compression (-1.46 -> -2.90 GPa). An NMOS
capped and annealed keeps +0.2 GPa of tension after the cap is gone.

```bash
python3 -m mechanics.planar_flow
```

<p align="center"><img src="docs/img/stress_pmos_gate_last.png" width="100%"></p>

## FinFET and nanosheet flows, 2 nm-node class ([details](docs/finfet_nanosheet.md))

Two new virtual-fab templates at a 48 nm contacted poly pitch and a 15 nm
gate. The bulk FinFET has a 5 x 50 nm fin, an IL + HfO2 gate stack and a
recessed, in-situ doped S/D epi. The nanosheet is a replacement-metal-gate
flow with three released 5 x 25 nm sheets, inner spacers and the gate
wrapping every sheet. The S/D epi doping is analytic, with a Gaussian
taper of the distance to the epi.

```math
N = N_{peak}\,\exp\!\Big(-\tfrac{1}{2}\big(\tfrac{d_x}{s_x}\big)^2-\tfrac{1}{2}\big(\tfrac{d_y}{s_y}\big)^2\Big)
```

<p align="center">
  <img src="docs/img/finfet_gatecut.png" alt="FinFET cut through the gate" width="48%">
  <img src="docs/img/nanosheet_gatecut.png" alt="Nanosheet cut through the gate" width="48%">
</p>

### S/D epitaxy shapes ([details](docs/sd_epi.md))

The S/D epi grows as an elliptical Si:P blob for NMOS and a
{111}-faceted SiGe diamond for PMOS. It merges with the neighbours only
where the pitch forces it, which happens for the PMOS FinFET but never on
the nanosheets. The contact is etched into the epi as a round-bottomed
plug, so a narrow epi keeps its contact area. Compared with the old
flat-topped block, Ion rises about 3% and Vt, SS and DIBL are unchanged.

## Schrodinger-Poisson channel charge in 3D ([details](docs/quantum.md))

The Schrodinger lines now run in 3D devices: down from the planar gate
oxide, and wall to wall across a fin or a sheet. The quantum potential is
updated inside the Newton loop, so each bias point is a single solve. In
a 5 nm fin, the electrons move from the walls to the middle of the fin
(volume inversion). Vt rises 30 mV and Ion falls 28%. The planar quantum
I-V takes about 100 s (classical: 74 s).

<p align="center">
  <img src="docs/img/quantum_3d_finfet.png" alt="FinFET classical vs Schrodinger-Poisson: Id-Vg, density across the fin, gate cross-sections, Lambda" width="98%">
</p>

## Schrodinger-Poisson channel charge, 1D ([details](docs/quantum.md))

The inversion charge comes from the Schrodinger equation, solved on lines
across the channel body. It enters drift-diffusion as a quantum potential
Lambda that conserves charge on any mesh. In bulk, the charge peak moves
about 1 nm off the oxide and Vt rises by 83 mV at 1e18. In a thin double
gate the charge peaks at the center of the body (volume inversion).

```math
-\frac{\hbar^2}{2}\frac{d}{ds}\Big(\frac{1}{m_z}\frac{d\psi_k}{ds}\Big)+U\psi_k=E_k\psi_k,\qquad
n=n_i\,e^{(\psi-\phi_n-\Lambda_n)/V_t},\qquad
\Lambda_i=-V_t\ln\frac{\int_{V_i} n_{QM}}{n_{cl,i}V_i}
```

<p align="center">
  <img src="docs/img/quantum_sp_1d.png" alt="Schrodinger-Poisson in 1D: density vs depth, C-V, centroid, double-gate volume inversion" width="98%">
</p>

## 3D planar NMOS: Schottky S/D contacts ([details](docs/schottky.md))

The S/D metal/semiconductor contact is modeled with thermionic emission
plus WKB tunneling through the solved barrier. In 2D/3D the barrier is a
sub-grid boundary condition, and it is on by default.

```math
J=\frac{A^{*}T}{k_B}\int \mathcal{T}(E)\,\ln\frac{1+e^{(E_{F,s}-E)/k_BT}}{1+e^{(E_{F,m}-E)/k_BT}}\,dE,
\qquad
\mathcal{T}(E)=\exp\!\Big(-\frac{2}{\hbar}\int_{0}^{x_t}\sqrt{2m^{*}\,(E_c(x)-E)}\,dx\Big)
```

<p align="center">
  <img src="out/schottky/schottky_barrier_current.png" alt="Schottky contact: band diagram with tunneling paths, energy-resolved current, tunneling share vs doping" width="98%">
</p>

## 3D planar NMOS: device solve ([details](docs/fab.md))

Coupled Newton drift-diffusion on an octree mesh with Schottky contacts.
A full Id-Vg takes 74 s: Ion 1729 uA/um, SS 82 mV/dec, DIBL 161 mV/V.
Run it with `python -m fab simulate configs/fab_planar.yaml --mesher octree`.

```math
\nabla\!\cdot(\varepsilon\nabla\psi)=-q\,(p-n+N_D^{+}-N_A^{-}),\qquad
\nabla\!\cdot\mathbf J_n=qR,\qquad
\mathbf J_n=-q\,\mu_n(F)\,n\,\nabla\phi_n,\qquad
\mu_n(F)=\frac{\mu_0}{\big[1+(\mu_0F/v_{sat})^{\beta}\big]^{1/\beta}}
```

<p align="center">
  <img src="docs/img/fab_ids_vgs.png" alt="3D planar NMOS Id-Vg, log and linear" width="80%">
</p>

## 3D virtual fab ([details](docs/fab.md))

The fab builds the device one process step at a time on voxel level sets:
STI, gate, spacer, implants, ILD and contacts. The images show the exact
voxel faces the solver meshes. Build it with
`python -m fab build configs/fab_planar.yaml`.

```math
N(\mathbf r)=\frac{\Phi}{\sqrt{2\pi}\,\Delta R_p}\exp\!\Big(-\frac{(d_{\rm surf}-R_p)^2}{2\Delta R_p^2}\Big)\cdot
\tfrac12\,\mathrm{erfc}\!\Big(\frac{d_{\rm mask}}{\sqrt2\,\sigma_{\rm lat}}\Big)
```

<p align="center">
  <img src="docs/img/fab_structure.png" alt="Planar NMOS from the virtual fab, ILD hidden" width="49%">
  <img src="docs/img/fab_doping_cutaway.png" alt="Mid-width cutaway coloured by net doping" width="49%">
</p>

## 2D GIDL and junction BTBT ([details](docs/2d.md))

Kane band-to-band and Hurkx trap-assisted tunneling, evaluated along
tunneling paths traced through the 2D field: nonlocal, and correct across
Si/SiGe.

```math
G_{\rm BTBT}=A\,F_{\rm eff}^{P}\exp\!\Big(-\frac{B}{F_{\rm eff}}\Big),\qquad F_{\rm eff}=\frac{E_g}{q\,\ell_{\rm path}}
```

<p align="center">
  <img src="out/btbt/input_mosfet_2d_btbt/btbt_generation_maps.png" alt="BTBT generation maps, local vs nonlocal, GIDL and junction" width="80%">
</p>

## 2D NMOS ([details](docs/2d.md))

Box-method drift-diffusion with Scharfetter-Gummel fluxes on a
non-obtuse quadtree mesh. Run it with `python3 main2d_mosfet_sweep.py`
(about 10 s).

```math
J_{n,ij}=\frac{q\,\mu_n V_T}{h_{ij}}\Big[n_j\,B\!\Big(\frac{\psi_j-\psi_i}{V_T}\Big)-n_i\,B\!\Big(\frac{\psi_i-\psi_j}{V_T}\Big)\Big],\qquad B(x)=\frac{x}{e^{x}-1}
```

<p align="center">
  <img src="docs/img/mosfet2d_fields.png" alt="2D NMOS potential, electron density and electron current, off and on" width="98%">
</p>

## 1D avalanche breakdown ([details](docs/1d.md))

The reverse I-V is traced through the breakdown knee by arc-length
continuation. Run it with `python3 -m avalanche.main_avalanche`.

```math
G_{II}=\frac{\alpha_n|\mathbf J_n|+\alpha_p|\mathbf J_p|}{q},\qquad \alpha(F)=a\,e^{-b/F},\qquad \int_0^W\!\alpha\,dx=1\ \text{at } BV
```

<p align="center">
  <img src="out/avalanche/breakdown_iv.png" alt="Avalanche breakdown I-V" width="80%">
</p>

## 1D diode and MOS capacitor ([details](docs/1d.md))

These were the first two devices: a p-n diode I-V (Gummel or coupled
Newton) and MOS C-V (quasi-static and high frequency). Run them with
`python3 main.py` and `python3 -m mos.mos_main`.

```math
\frac{d}{dx}\Big(\varepsilon\frac{d\psi}{dx}\Big)=-q\big(p_0e^{-\psi/V_T}-n_0e^{\psi/V_T}+N_D-N_A\big),\qquad C(V_G)=\frac{dQ_G}{dV_G}
```

<p align="center">
  <img src="out/diode/03_iv_curve.png" alt="Diode I-V curve" width="49%">
  <img src="out/mos/01_cv_curve.png" alt="MOS capacitor C-V curve" width="49%">
</p>

## Setup ([details](docs/setup.md))

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 testsuite/run.py           # parallel quick tier, ~70 s
```

3D solves run faster in the conda env with PETSc (`environment.yml`). The
viewer is `python3 -m tcadview <file.pvd>`. `python3
docs/make_readme_figures.py` regenerates the images in `docs/img/`. For
the code layout see [ARCHITECTURE.md](ARCHITECTURE.md),
[docs/files.md](docs/files.md) and [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md).
