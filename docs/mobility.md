# Inversion-layer mobility: phonon, surface roughness, Coulomb, orientation

[Back to the README](../README.md)

`fab simulate --mobility fast` (the default), `--mobility exact` or
`--mobility off`; 2D: `physics: mobility:` in the config or
`main2d_mosfet_sweep.py --mobility ...`. Code: `solver2d/surface_mobility.py`,
wired into `low_field_mobility` (`solver2d/newton_solver_qf_2d.py`).
DEVELOPMENT_LOG 83.

Before this, the low-field mobility was the bulk doping-dependent value
(Masetti) from the NET doping: a carrier at the gate interface moved as fast
as one in the bulk. Real inversion layers lose mobility to the gate field
(phonons, interface roughness) and to charge near the interface. That is the
"universal mobility" every TCAD tool models.

## The pipeline

Per node, Matthiessen's rule, then per edge (the mean of its two nodes), in
the Sentaurus order:

```
1/mu_low = 1/mu_bulk(N_A + N_D [, n, p])                 Masetti (fast) | Klaassen (exact)
         + D * s_orient * (1/mu_ac(F) + 1/mu_sr(F))      Lombardi (fast) | Darwish (exact)
         + D / mu_ic(F, N_it, n or p)                    interface charge (opt-in)
         + D * D_hk * (1/mu_rcs + 1/mu_rps)              remote high-k Coulomb + phonon (opt-in)
mu_edge  = mean of the two nodes  x  stress factor (piezo, docs/piezo_mobility.md)
           [ballistic term: branch ballistic-mobility]   ->  velocity saturation (docs/vsat_modes.md)
D = exp(-x / 10 nm),  x = distance to the nearest semiconductor/insulator interface
```

- **mu_ac = B/F + C·((N + N2)/N0)^λ / (F^(1/3) (T/300)^k)**: 2D acoustic
  phonons (the B term) and the doping-dependent phonon term.
- **mu_sr = (F^A*/δ + F³/η)^-1**: surface roughness, dominant at high field.
  Exact mode: A* = A + α(n + p)/(N + N1)^ν (Darwish), so roughness scattering
  grows with the inversion density.
- **Coulomb, channel doping**: through the bulk model, now from the TOTAL
  doping N_A + N_D. The fab knows both species; `mesh.Ntot` is attached by
  `fab simulate`. A 2D config mesh only has the net doping and falls back to
  it. In exact mode the bulk model is Klaassen's (Philips unified): the
  impurity scattering is screened by the free carriers, so a doped channel
  in strong inversion is not over-degraded the way Masetti does it.
- **Coulomb, interface charge**: `--mobility-set interface_charge=<cm^-2>`
  puts a fixed sheet charge on every semiconductor/insulator interface. It is
  in Poisson (it shifts Vt) and scatters carriers (Sentaurus' interface-charge
  form: mu1 = 40, screened by the inversion density, switched off at flat band).
  Off by default (0). The interface-trap roadmap will feed it the trapped
  charge later.
- **Remote Coulomb and phonons from the high-k** (`highk=1`, off by default):
  Tanimoto's remote Coulomb and Zhu and Ma's remote phonon term, weighted by
  exp(-d_hk / 1 nm), d_hk the distance to the nearest insulator node with
  eps_r > 8. Both are empirical. The (N/3e16)^γ doping factor of the remote
  Coulomb term goes to 0 for an undoped channel, so N is floored at 1e15.
- **Not implemented**: Coulomb2D (local channel impurities as a separate
  component; it double-counts Masetti/Klaassen), the to-current normal field,
  thickness-fluctuation scattering (ThinLayer).

## The normal field F

F at a node is |∇ψ · n̂|, n̂ the normal of the nearest interface node, from one
sparse operator per mesh (one SpMV per evaluation):

- **Interface nodes**: the displacement flux through the node's insulator
  edges divided by ε_Si (D-continuity, ε_ox E_ox = ε_Si F_s). The oxide holds
  no charge, so its field is resolved on any mesh. A test checks it against
  the 1D MOS solution's charge (Gauss, exact) and the analytic surface field.
  Sentaurus uses a ξ-correction of the one-sided value instead; the plain
  one-sided value is the `interface_field=lsq` option.
- **Other nodes**: a least-squares gradient over the node's semiconductor
  neighbours (exact for a linear ψ on triangles, rectilinear and octree
  meshes; tested).
- The interface set is every semiconductor/insulator interface (the Sentaurus
  default): gate dielectric, STI, spacers. The metal-surface nodes of the
  S/D contacts are not a dielectric; the gate contact's are (they sit on the
  gate dielectric).
- F is floored at 1e3 V/cm (Sentaurus `EnormMinimum`).

## Surface orientation

Each interface node's normal is rotated into the crystal frame ((001) wafer,
<110> channel: device x = [110], y = [00-1], z = [-110]) and matched to the
<100>, <110> and <111> families. The (100) planar surface and fin and sheet
tops are (100); the FinFET sidewalls and the sheet edges are (110). A
corner node's bisector normal is nearest <111>.

- **fast**: the nearest family. **exact**: a blend w_k ~ exp(-(1 - |cos θ_k|)/0.02),
  which only acts at corners.
- **Electrons**: the (110) set is the (100) set. FinFET sidewall data
  (Kavalieros 2008, Young SSE 2012, VLSI 2011) show (110) sidewalls like (100);
  a planar (110) wafer is lower (0.6-0.8x): `scale_110_n=0.7`.
- **Holes**: (110) mu_ac and mu_sr x 3.13, calibrated in the 1D harness so
  that mu_eff(110)/mu_eff(100) = 2.0 at E_eff = 0.5 MV/cm (Irie IEDM 2004,
  Takagi TED 41, 2363).
- `--orientation 100` uses the (100) set everywhere.

## How it is solved

The mobility depends on ψ (F), and in exact mode on n and p. It is lagged:
evaluated where Newton assembles its Jacobian, held through that
iteration's line search, and a solve is converged only once the edge
mobility stopped changing (max |dμ/μ| < 1e-4 fast, 1e-6 exact, on the edges
within reach of an interface). The Jacobian keeps its pattern and the cached
assembly plan. The converged state is exactly self-consistent; a current
evaluated after the solve re-evaluates the mobility at the converged state.
With the model off the mesh carries nothing and the solver is unchanged,
bit for bit (tested).

## Fast and exact modes

| | fast (default) | exact |
|---|---|---|
| surface terms | Lombardi (Sentaurus Table 58) | Darwish/Lucent (Table 59) with A* |
| bulk | Masetti, total doping | Klaassen (Philips), carrier screening |
| orientation | nearest family | smooth blend at corners |
| self-consistency | 1e-4 | 1e-6 |

MODE_TABLE

## Universal mobility check

`scripts/mobility/universal.py`: a 1D MOS capacitor (classical, or
Schrodinger-Poisson with `--quantum`) over a gate sweep, the SAME
`SurfaceMobility` evaluated on it, μ_eff = Σμn/Σn (split-CV) against
E_eff = (Q_dep + ηQ_inv)/ε_Si (η = 1/2 electrons, 1/3 holes), and the
universal curves (Takagi; Chen/Hu fits: μ_n = 540/(1 + (E/0.9)^1.85),
μ_p = 185/(1 + E/0.563), E in MV/cm).

UNIVERSAL_TABLE

![universal mobility](img/universal_mobility.png)

## Device results

DEVICE_TABLE

## Parameters and sources

| model | parameter | electrons | holes | source |
|---|---|---|---|---|
| Lombardi | B, C | 4.75e7, 580 | 9.925e6, 2947 | Sentaurus UG N-2017.09 Table 58; ATLAS Table 3-28 |
| | λ, k, δ, A, η | 0.125, 1, 5.82e14, 2, 5.82e30 | 0.0317, 1, 2.0546e14, 2, 2.0546e30 | same |
| | l_crit | 10 nm | 10 nm | same |
| Darwish | B, C, λ, k | 3.61e7, 1.70e4, 0.0233, 1.7 | 1.51e7, 4.18e3, 0.0119, 0.9 | Darwish TED 44, 1529 (1997); Table 59 |
| | δ, A, α⊥, ν | 3.58e18, 2.58, 6.85e-21, 0.0767 | 4.10e15, 2.18, 7.82e-21, 0.123 | same |
| Klaassen | μmax, μmin, θ, Nref, α | 1417, 52.2, 2.285, 9.68e16, 0.68 | 470.5, 44.9, 2.247, 2.23e17, 0.719 | Klaassen SSE 35, 953 (1992); Tables 56/57 |
| interface charge | μ1, c_trans, γ, ν1, ν2, E0, ε, N0 | 40, 1e18, 1.5, 1, 0.5, 2e5, 2, 1e11 | same | Sentaurus Table 63 |
| remote Coulomb | μ0, γ1..γ5, s, c0 | 149, 0.232, 2.1, 0.40, 0.05, 1, 0.1, 3e16 | same | Tanimoto SISPAD 2006; Table 64 |
| remote phonon | μ0, γ1, γ2 | 496.7, 0.68, 0.34 | same | Zhu and Ma EDL 25, 89 (2004); Table 65 |
| orientation | (110) surface scale | 1.0 | 3.13 (μ_eff ratio 2.0 at 0.5 MV/cm) | calibrated here; Irie 2004; Kavalieros 2008 |

The Sentaurus numbers were read from the user guide for the plan
(`out/passdown_logs/mobility_plan.md`, sections 1-2, 6). The interface-charge
and remote-Coulomb formulas are the plan's reading of the guide; verify them
against the manual before quoting them.

Options (`--mobility-set k=v`, or keys of `physics: mobility:` in 2D):
`interface_charge`, `highk`, `orientation`, `interface_field` (dcont | lsq),
`doping` (total | net), `l_crit`, `F_min`, `scale_110_n`, `scale_110_p`,
`surface_scale_n/p` (μ_ac and μ_sr), `delta_scale_n/p` (μ_sr only),
`smoothing`, `tol`.

## Known simplifications

- SiGe nodes (PMOS S/D, nanosheet sacrificial layers) take the Si surface
  parameters. Only Si channels invert, so this is harmless.
- The Lombardi/Darwish sets were fitted with a classical inversion charge.
  With `--quantum` the charge sits ~1 nm deeper and samples a lower field.
- The interface charge is uniform on every interface. A per-interface value
  waits for the interface-trap work.

PASSDOWN
