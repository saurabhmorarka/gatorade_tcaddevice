# Stress -> mobility: piezoresistance in the device solver

The process stress computed by `mechanics/` (2D `planar_flow`, 3D
`fab_stress`, [stress.md](stress.md)) now changes the drain current. Every
silicon edge of the device mesh gets a low-field mobility factor from the
stress at its midpoint and its direction. The factor is computed once per
mesh, so the I-V costs the same as without stress. The default stays OFF:
nothing changes unless a stress is given (DEVELOPMENT_LOG 82).

## The model

For an edge with unit direction **t** (device frame: x along the channel,
y depth, z width) and the stress **sigma** at its midpoint:

```
x_c   = - P_c(N) * t.(Pi_c : sigma).t           c = n (electrons), p (holes)
mu_c0 = mu_c0(doping) * f_c(x_c)                 per edge, before velocity saturation
```

- **Pi_c** is the full 4th-order cubic piezoresistance tensor:
  `Pi_iiii = pi11`, `Pi_iijj = pi12`, `Pi_ijij = Pi_ijji = pi44/2`, in crystal
  axes. It is rotated to the device frame with the same rotation as the
  stiffness (`mechanics/elastic.py`; default (001) wafer, <110> channel). So
  shear stress and every edge direction are handled, not only the channel
  direction.
- **f_c** is the mobility factor, in one of two modes:
  - `linear`: `f = 1 + x`, floored at 0.05. This is Smith's small-stress law,
    the Sentaurus "Piezoresistance" factor.
  - `saturating` (the default): the same slope at zero stress, then a smooth
    tanh approach to a gain limit `f_max` (x > 0) and a loss limit `f_min`
    (x < 0):

    ```
    f = 1 + (f_max - 1) tanh( x / (f_max - 1))      x >= 0
    f = 1 - (1 - f_min) tanh(-x / (1 - f_min))      x <  0
    ```

- **P_c(N)** is the optional doping factor (on by default). It uses the edge's
  mean |net doping|; see "Known simplifications".
- The stress factor multiplies the total low-field mobility (Sentaurus'
  default). That is step 4 of `low_field_mobility` in
  `solver2d/newton_solver_qf_2d.py`, after the doping mobility and before
  velocity saturation. Edges in SiGe or touching an insulator keep a factor
  of 1.

### Coefficients and defaults

| quantity | electrons | holes | source |
|---|---|---|---|
| pi11, pi12, pi44 (1e-11/Pa) | -102.2, 53.4, -13.6 | 6.6, -1.1, 138.1 | Smith, Phys. Rev. 94, 42 (1954), lightly doped Si, 300 K |
| (001)/<110>: longitudinal, vertical, transverse | -31.2, 53.4, -17.6 | 71.8, -1.1, -66.3 | the same, rotated; Thompson et al., IEEE TED 53, 1010 (2006), Table I |
| gain limit f_max | 1.8 | 4.0 | holes: Thompson 2006, the maximum Si inversion-layer hole enhancement for uniaxial stress on (001), "~4x" (6-band k.p); electrons: the full valley-repopulation limit, Takagi et al., JAP 80, 1567 (1996); Ungersboeck (TU Wien thesis 2007, ch. 6: saturation near 1% strain) |
| loss limit f_min | 0.5 | 0.5 | a stated default, not a fit |
| doping factor P(N) | Kanda: (300/T) F_-1/2(eta)/F_1/2(eta), eta from N = Nc F_1/2(eta) | Richter: 1/(1 + (N/6e19)^0.43 + (N/7e20)^1.6) at 300 K | Kanda, IEEE TED 29, 64 (1982), the energy-independent relaxation-time form Sentaurus uses; Richter et al., JAP 104, 023715 (2008), Eq. 20 / Table IV (Kanda under-reads p-Si at high doping) |

P(N) values: Kanda 0.99 at 1e18, 0.89 at 1e19, 0.46 at 1e20. Richter 0.85
at 1e18, 0.68 at 1e19, 0.44 at 1e20.

Why these limits. At low stress both modes are Smith's law. The linear law
keeps rising past what the band structure allows. The PMOS gate-last
planar flow (-2.5 GPa) gives +167% linear (`planar_flow` report), and
electrons under compression would go negative. The tanh limits follow
the theory the coefficients come from:
- holes rise towards ~4x (Thompson 2006);
- electrons stop near 1.8x, once the repopulation is complete.

## Stress on the device mesh

`mechanics/stress_field.py` (`StressField`) holds the solved stress per
voxel (pixel in 2D), the voxel materials and the grid. `sample()` is a
masked multilinear interpolation between voxel centres. Only silicon voxels
take part, so an edge on a Si/oxide or Si/SiGe interface takes the silicon
side's stress, never the stress across the interface.

**Frame check** (`solver2d/stress_mobility.stress_factors`). It runs on every
use and fails loudly:
- Every silicon edge midpoint must lie inside the stress grid.
- Each must fall on a silicon voxel of the stress structure, within one
  voxel. The stress voxels are 1 nm majority labels of the 0.5 nm fab voxels,
  so a curved boundary moves by up to one voxel.
- More than 0.5% of edges farther than that is an error. A real
  misalignment moves whole surfaces by several voxels.

Measured exact hits:

| device | silicon edges | on a silicon voxel | within one voxel |
|---|---|---|---|
| FinFET PMOS | 49388 | 99.73% (the misses sit on the SiGe seed boundary) | 100% |
| FinFET NMOS | 87842 | 97.63% (the blob epi's curved surface) | 100% |

The 2D test moves the field 25 nm deep and checks that the error fires.

- **3D.** The fab grid, the stress grid (`fab_stress`, lower corner = the fab
  grid's) and the device mesh (fab node coordinates) share one frame. A saved
  field also carries the fab parameters it was built from. Loading it for a
  device with other parameters is an error.
- **2D.** The `planar_flow` cell is aligned on the device's gate centre
  (`align: gate`). The gate lengths must agree to one pixel.

## How to run

3D (`fab simulate`):

```bash
# solve the stress first (mechanics.fab_stress --fast), save <out>/stress_field.npz, then the I-V
python3 -m fab simulate configs/fab_finfet_pmos.yaml --mesher octree --vds -0.05 -0.7 \
    --vgs-min -0.8 --vgs-max 0 --vgs-step 0.05 --stress
# reuse a saved field; the linear mode; no doping factor; stressor overrides
python3 -m fab simulate configs/fab_finfet.yaml --mesher octree --stress out/x/stress_field.npz --piezo linear
python3 -m fab simulate configs/fab_finfet.yaml --stress --piezo-doping off --stress-set gate_GPa=0
```

A PMOS I-V needs negative `--vds` and the gate walked down with `--vgs-min`
and `--vgs-max 0`. The metrics are mirrored (`mosfet_metrics_signed`): Vt is
negative, and the currents, SS and DIBL are magnitudes.

2D (`main2d_mosfet_sweep.py`): a `stress:` block in the config, see
`configs/input_mosfet_2d_stress_nmos.yaml` / `_pmos.yaml`. These are 30 nm
planar devices on the `planar_flow` cell geometry. `--no-stress` gives the
unstressed reference run.

```yaml
stress:
  source: planar_flow       # or file: <stress_field.npz>
  flow: nmos_smt            # pmos_gate_last | pmos_gate_first
  set: {}                   # planar_flow stressor overrides
  align: gate
  piezo: saturating         # | linear
  doping_factor: true
```

## Results

Saturating mode with the doping factor and the fast (hybrid) velocity
saturation. "Channel sigma" is the `fab_stress` / `planar_flow` channel
mean (xx along the channel, yy vertical, zz width). Ion is at the highest
|Vgs| and |Vds|; Idlin is at |Vds| 0.05 V and the same |Vgs|. Every curve
has every point. These runs were taken while another agent's jobs loaded
the machine (load average 3-11), so compare Newton iterations, not seconds.

**2D, 30 nm planar devices on the planar_flow cell** (`configs/input_mosfet_2d_stress_*.yaml`, |Vds| 1.0 V, |Vgs| 1.0 V):

| device / flow | channel sigma xx/yy/zz (MPa) | mobility factor (channel) | Idlin | Ion (uA/um) | run (s) off / on |
|---|---|---|---|---|---|
| NMOS, nmos_smt (SMT + dislocations + liner) | +702 / -328 / -75 | mu_n x 1.34 (linear 1.37) | +14% (+25% at Vgs 0.4) | 1727 -> 1807 (+4.6%) | 3.6 / 3.5 |
| PMOS, pmos_gate_last (SiGe S/D + liner + RMG) | -2719 / -137 / -219 | mu_p x 2.32 (linear 2.42) | +57% (linear +60%) | 974 -> 1210 (+24%; linear +25%) | 4.4 / 5.0 |

The planar_flow stress solve takes 9-10 s, once per run (`main2d_mosfet_sweep` saves it and the workers load it).

**3D, octree, fast vsat, |Vds| 0.05 / 0.7 V, |Vgs| 0.8 V** (`fab simulate --stress`):

| device | channel sigma xx/yy/zz (MPa) | Idlin | Ion (uA/um) off -> on | Newton its off / on | total (s) off / on |
|---|---|---|---|---|---|
| FinFET NMOS | +189 / +88 / -235 | -0.7% | 1005 -> 1002 (-0.3%) | 333 / 353 | 176 / 175 (+31 s stress solve) |
| FinFET PMOS | -1500 / +611 / -254 | +5% (+18% at -0.4 V) | 537 -> 580 (+8.2%) | 213 / 224 | 132 / 129 (+48 s stress solve) |
| FinFET PMOS, ohmic S/D what-if | same | +29% | 782 -> 856 (+9.5%) | 193 / 197 | 103 / 98 (+46 s) |
| nanosheet NMOS | +442 / +17 / +101 | +1.0% | 1109 -> 1120 (+1.0%) | 362 / 397 | 671 / 457 (+138 s) |
| nanosheet PMOS | -1040 / -80 / -28 | +2.5% (+9.5% at -0.4 V) | 393 -> 425 (+8.1%) | 204 / 261 | 404 / 666 (+166 s) |

**Cost.**
- The factors take 0.2 s per mesh.
- An iteration costs the same with and without them: FinFET NMOS 0.90 s per iteration in both runs.
- What changes is the iteration count. The stressed curves take 0-28% more Newton iterations, because the stress changes the solution, not the cost of a step.
- The 3D stress solve (`mechanics.fab_stress --fast`) is the real extra cost: 31-48 s for a FinFET, 140-170 s for a nanosheet. It runs once, and `--stress <npz>` reuses it.

**Reading the numbers.**
- **The channel factors are as large as expected.** Under the gate the mobility rises:
  - FinFET PMOS, holes x1.68;
  - 2D gate-last PMOS, x2.32;
  - 2D SMT NMOS, electrons x1.34.
- **The 3D PMOS Ion gains (+8%) are far below the literature's +50-100%.** Three things dilute them:
  1. **Series resistance.** Both 3D PMOS are limited by series resistance. Idlin is nearly flat in Vgs (FinFET PMOS 74 uA/um at -0.8 V, against 147 for the NMOS). With ohmic contacts the FinFET PMOS Idlin gain rises to +29%.
  2. **Velocity saturation.** The factor acts on the low-field mobility. At Lg 15 nm and 0.7 V the channel is near velocity saturation, which does not change with stress. Even the 2D PMOS, x2.3 in the channel, gains only +24% in Ion.
  3. **No ballistic link.** There is no quasi-ballistic (backscattering) model to tie Ion to the mobility. That is the planned ballistic work.
- **The FinFET and nanosheet NMOS on main carry little stress.** There is no SMT or dislocation stressor in their 3D flows (that is the `nmos-stress` branch). Their factors are close to 1 and Ion moves by about 1%.

## Known simplifications

- **Bulk coefficients for inversion layers.** Thompson 2006 reports
  inversion-layer coefficients close to bulk for holes at low stress. For
  electrons, confinement already splits the valleys, so bulk pi over-states
  their stress response in thin sheets (Ungersboeck 2007). A
  surface-orientation and confinement-aware set (the FinFET (110)
  sidewalls) is for the mobility-stack work. `PiezoModel.coeffs` takes any
  (pi11, pi12, pi44) per carrier.
- **No band-edge shift.** Stress also moves Ec and Ev (deformation
  potentials), which moves Vt by tens of mV. That is not modelled: only the
  mobility changes. The Vt_cc shifts in the tables come from the current
  level.
- **Low field only.** The factor acts on mu0. The saturation velocity is
  unchanged, so in short devices the Idsat gain is a fraction of the mobility
  gain. A ballistic / backscattering model (planned) links them more
  strongly.
- **No SiGe coefficients.** The SiGe S/D keeps a factor of 1.
- **Net doping, not total.** The doping factor uses the mesh's NET doping
  |N_D - N_A|. Strictly, Kanda/Richter (like Masetti) need the total doping
  N_A + N_D. The two agree wherever one species dominates: everywhere in
  these devices except across a junction.
