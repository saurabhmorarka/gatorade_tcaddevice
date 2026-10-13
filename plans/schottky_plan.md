# Schottky S/D contacts: plan

Branch: `schottky-contacts` (from `main` at cdbcd1e). Roadmap item 1.
Status: APPROVED 2026-10-12 (full plan, stop for review after 1D).
Barrier spec (user decision): default Phi_B = Phi_m - chi from the
material DB (Schottky-Mott); `barrier_eV` on the contact overrides it
(implemented as `barrier_n_eV` / `barrier_p_eV`, plus a `pinned` model).

**User requirements added 2026-10-12 (during step 3):**
- Schottky contacts ON BY DEFAULT for every transistor simulation (2D and
  3D S/D contacts), because real devices have them. Ohmic stays as an
  explicit option. Measure the runtime impact on the 3D planar I-V (57 s
  reference) when ported.
- A dedicated Schottky test case in the testsuite: `testsuite/test_schottky.py`
  (1D now; add 2D and 3D cases when the model is ported).

**STATUS 2026-10-12: 1D done (LOG 60-61); 2D/3D sub-grid port done, Schottky
the fab default (LOG 62). Next: section 9 (contact implant, recessed contact).**

## 1. Why, and what goes wrong today

Every semiconductor contact is ideal ohmic: a Dirichlet condition
psi = psi_eq + V and n, p at their neutral values (`core/solver.py`
`contact_values`, `solver2d/newton_solver_qf_2d.py` ~L1100). The 3D
source/drain contacts are already metal volumes from the process, but their
interface with the n+ epi has zero resistance. A real silicide/n+ Si
contact has a barrier of about 0.4-0.7 eV and conducts mostly by tunneling
through a 2-3 nm depletion layer. Its specific contact resistivity
rho_c ~ 1e-9 to 1e-8 Ohm cm^2 depends exponentially on the doping at the
interface and on the barrier height. At today's contact areas that is a
large part of R_sd, so Ion is overestimated.

## 2. How other simulators do it (the standard model)

Sentaurus ("Schottky" contact + "eBarrierTunneling" nonlocal tunneling),
Atlas (Schottky + UST/"universal Schottky tunneling") and the textbook
treatment (Sze; Padovani-Stratton 1966; Crowell-Rideout 1969; Yu 1970) all
split the contact current into the same three pieces:

1. *Thermionic emission (TE) boundary condition*
   (Crowell-Sze recombination-velocity form):
   - psi_s = V - Phi_Bn + (Ec - Ei)/q, i.e. the conduction band at the
     interface sits Phi_Bn above the metal Fermi level whatever the bias.
   - Electrons: J_n.n = q v_n (n_s - n_B), n_B = Nc exp(-Phi_Bn/Vt),
     v_n = A*_n T^2 / (q Nc), about 2.6e6 cm/s in Si.
   - Holes: J_p.n = -q v_p (p_s - p_B), p_B = Nv exp(-(Eg - Phi_Bn)/Vt).
   - n_s and p_s are no longer fixed, so phin and phip become FREE unknowns
     at the interface nodes, with a Robin (flux) row in place of the
     Dirichlet row. psi stays Dirichlet.
2. *Image-force barrier lowering*: dPhi = sqrt(q E_s / (4 pi eps_s)),
   with E_s the normal field at the interface (optional term,
   Sentaurus also has an empirical alpha * E^gamma part).
3. *Tunneling (field emission and thermionic-field emission)*: the
   nonlocal Tsu-Esaki current through the actual barrier
   J_t = (A* T / k) Int Gamma(E) ln[(1 + e^{(E_Fs - E)/kT}) / (1 + e^{(E_Fm - E)/kT})] dE,
   with Gamma(E) the WKB transmission along a path normal to the interface,
   using the SOLVED Ec(x) (not a constant-field or depletion-model
   guess). Sentaurus deposits it as electron generation at the classical
   turning point x_t(E), where Ec(x_t) = E. With that, TE, TFE and FE
   regimes, and the doping and bias dependence, come out of one model
   with no regime switch.
   Regime scale: E00 = (q hbar / 2) sqrt(N / (m* eps)): 0.011 eV at 1e18,
   0.11 eV at 1e20 (Si, m* = 0.26). TE when E00 << kT, FE when E00 >> kT.

## 3. Design for tcad1d

- Contact spec, all levels (1D config, 2D boundary, 3D layer-3 contacts):
  `type: schottky`, `barrier_eV` (optional override; default
  Phi_m - chi from the contact metal's work function), `barrier_lowering: true|false`,
  `tunneling: none|nonlocal`, `richardson` (A*_n, A*_p; Si defaults
  112 / 32 A cm^-2 K^-2), `m_tunnel` (default 0.26 m0 for electrons).
  `type: ohmic` stays the default and stays bit-identical.
- TE part: new rows in the QF Newton residual/Jacobian for interface
  nodes of a Schottky contact. In 1D it is the end node. In 2D/3D it is
  each semiconductor node on the metal interface, flux weighted by its
  Voronoi interface area (the 3D mesh already has these areas).
- Tunneling part: a `generation(psi, phin, phip, jacobian)` callable, the
  same interface the solver already uses for nonlocal BTBT
  (`newton_solve_2d(..., generation=...)`, Gn and Gp may sit at different
  nodes), so it plugs into 1D-QF, 2D and 3D with no solver surgery.
  Electrons tunnel from the metal Fermi level V to the turning point;
  E_Fs is phin at the turning point. The terminal current adds the
  integrated tunneling current to the TE flux.
- Path in 1D: the mesh line itself (reuse the walk pattern of
  `btbt/paths1d.py`). Ec is sampled finer than the mesh along the path
  (linear-in-psi interpolation), so the integral does not depend on the
  mesh spacing (user rule: robustness to coarse meshes).
  Paths in 2D/3D: straight lines along the interface normal from each
  interface node, Ec interpolated from the mesh (the same machinery the
  2D field-line BTBT paths use). This comes AFTER the 1D validation.
- Energy grid: about 40-60 points from max(E_Fm, E_Fs) - few kT up to the
  barrier top; vectorized over all paths.
- New module `contacts/schottky.py` (pure physics kernel: TE rates, image
  force, WKB, Tsu-Esaki integrand), separate from the solvers, following
  the project's "add modules, don't rewrite working ones" pattern.

## 4. Validation (1D first, user rule)

1D metal / n-Si / ohmic back contact, plus a metal / p-Si case for the
hole branch:

1. *TE limit* (N = 1e15-1e16, tunneling off): J-V matches
   J = A* T^2 e^{-Phi_B/Vt} (e^{V/Vt} - 1) within a few %, ideality
   ~1.00. With image force on, the ideality rises to ~1.02-1.05 and the
   reverse current grows with sqrt-of-field (textbook).
2. *Doping sweep* N = 1e17 ... 1e21, Phi_B = 0.4, 0.5, 0.6, 0.7 eV:
   rho_c(N) from dV/dJ at V -> 0 against the Padovani-Stratton/Yu
   analytic TFE/FE curves and against measured NiSi/PtSi TLM data
   (Stavitski et al., EDL 2008). Expect rho_c ~ 1e-8 Ohm cm^2 near
   2-3e20 for Phi_B ~ 0.6 eV, and a slope change at E00 ~ kT.
3. *Ohmic limit*: Phi_B -> 0 or very high doping gives the same I-V as
   today's ohmic contact.
4. *Robustness matrix* (user rule): every doping x barrier x bias point
   converges, forward and reverse to +/-1 V, and the results stay within
   ~1-2% under 2x and 4x mesh coarsening near the contact.
5. *Cost*: Newton iterations and wall time vs the ohmic case.

Then 2D (n+/metal contact on the 2D MOSFET: Id-Vg/Id-Vd, R_contact from
the drop at the contact), then 3D (`fab simulate` planar: Ion drop vs
the ohmic reference 1792 uA/um; re-measure the 57 s I-V time).

## 5. Known limits and risks

- *Boltzmann statistics* (known simplification). At 1e20+ the Fermi level
  sits in the band. The tunneling integral uses phin for E_Fs, so it is
  mostly fine, but the band bending and n_s carry the Boltzmann error.
  Fermi-Dirac in the transport solve is its own project; noted here and
  not done in this pass.
- *3D mesh near the contact*: the depletion layer is ~2-3 nm at 1e20.
  The octree's spacing at the metal/epi interface may not resolve it, and
  sub-sampling cannot recover band bending that the mesh never solved.
  This may need a contact-interface refinement rule in the mesher (to be
  measured in the 3D phase).
- *Image force + tunneling*: double-counting risk. The image force lowers
  the peak of the barrier the WKB path sees. Do it once (in the barrier
  profile), as Sentaurus does.
- *Speed*: the tunneling kernel runs once per Newton iteration over a few
  hundred interface nodes x ~50 energies. It should be cheap next to the
  linear solve, but it will be measured.

## 6. Order of work (each a commit + DEVELOPMENT_LOG entry)

1. `contacts/schottky.py` kernel + unit tests (TE rates, WKB through a
   triangular barrier vs the analytic answer, Tsu-Esaki integral).
2. 1D QF solver: Schottky TE boundary condition; validation 4.1 and 4.3.
3. 1D: image force + nonlocal tunneling; validation 4.2, 4.4, 4.5.
   Plots: rho_c(N) family, J-V, band diagram with turning points.
4. 2D port + validation.
5. 3D port (`fab` contact `type: schottky`) + planar device comparison.
Stop for the user's review after step 3 (1D validated) before porting.
3b. Study A (section 7), 1D, before the review stop.

## 7. Study A (user request 2026-10-12): Ti / TiN / TiSix barrier on n+ epi

RESULTS, the table and ALL literature sources: README section
"Schottky S/D contacts" (and DEVELOPMENT_LOG 60).

Today's S/D contact is Ti or TiN, about 5-7 nm, on highly doped Si epi
(in-situ Si:P up to ~2e21), often with a TiSix layer formed at the
interface. Question: how far off is the plain work-function
(Schottky-Mott) barrier, and how far does Fermi-level pinning move it,
compared with the literature?

- *Schottky-Mott*: Phi_Bn = Phi_m - chi. With the DB values, Ti
  4.33 - 4.05 = 0.28 eV, TiN 4.6 - 4.05 = 0.55 eV (TiSi2: needs a work
  function entry, ~4.5-4.6, to be sourced).
- *Pinned (Cowley-Sze / MIGS form)*:
  Phi_Bn = S (Phi_m - chi) + (1 - S) Phi_CNL,n, with S the pinning factor
  (small for Si, ~0.1-0.3 in the literature) and Phi_CNL,n the charge
  neutrality level measured from Ec. Both become contact parameters
  (`barrier_model: schottky_mott | pinned`, `pinning_S`, `cnl_eV`);
  `barrier_n_eV` still overrides everything.
- *Literature to compare against* (exact values from the full texts):
  Aboelfotoh et al., PRB 1986 (Ti and TiSi2 on n- and p-Si(100):
  silicide formation changes the barrier < 0.1 eV, pinned near mid-gap);
  Sze's barrier table (Ti, TiSi2 on n-Si ~0.5-0.6 eV); TiN/n-Si data;
  rho_c on Si:P epi (1.5e-9 Ohm cm^2 at 2e21 with Ti silicidation, imec;
  Stavitski EDL 2008 for NiSi/PtSi vs doping).
- *Outputs*: a table of barrier per metal (Schottky-Mott vs pinned vs
  measured), and rho_c(N) curves from the 1D model for each, against
  the measured rho_c points. A fitted (S, CNL) that reproduces the
  measured barriers becomes the default for Si.
- *Both device types share ONE contact metal (user, 2026-10-12)*:
  NMOS S/D = Si:P 7e20, PMOS S/D = Si0.5Ge0.5:B 5e20, same Ti/TiN(/TiSix)
  on both. So the study has two interfaces:
  - NMOS: electron barrier Phi_Bn on n+ Si;
  - PMOS: hole barrier Phi_Bp on p+ strained Si0.5Ge0.5 (smaller gap,
    valence band raised vs Si: use the existing
    `strained_sige_on_si_offsets` / `derive_alloy` material).
  One metal choice trades the two barriers against each other (pinned
  near mid-gap, Phi_Bn + Phi_Bp ~ Eg of each side); the study reports
  rho_c for both at their real dopings. The hole branch (TE + tunneling)
  and the heterojunction band reference (delta_Ei) are therefore
  required in step 2-3, not optional.
- Runs after step 3 (needs the tunneling model), still in 1D.

## 8. 2D/3D port design (2026-10-12, after the user's go-ahead)

**Why not resolve the barrier on the device mesh.** Measured in 1D (TiN-like
contacts): rho_c from the explicit model vs the 0.05 nm reference -
7e20: 0.5 nm 1.17x, 1 nm 1.80x, 2 nm 15.2x; 5e20 p: 1 nm 1.84x, 2 nm 6.5x.
The barrier is 0.5-1 nm thick at S/D doping; a 3D mesh cannot carry
0.05-0.2 nm spacing at every contact (user rule: robust to coarse meshes).

**Sub-grid Schottky contact (`contacts/subgrid.py`).**
- For each (semiconductor material, contact parameters, carrier), the
  validated 1D model is solved once on a fine 1D mesh and tabulated as
  J(N, u): the majority current density into the metal vs the doping N at
  the interface and the drop u = V_metal - phi_majority just past the
  barrier (the depletion region and its tunneling are inside the table;
  the neutral bulk is left to the device mesh). Cached on disk by a hash
  of all parameters.
- On the device mesh, the semiconductor nodes touching the metal become
  ordinary free nodes: Poisson keeps them neutral (the barrier dipole is
  sub-grid, no Dirichlet psi), the majority carrier gets the sink
  J(N_i, V - phi_i) A_i / (q cvol_i), the minority carrier a TE Robin flux
  v (c - c(phi = V)) A_i / cvol_i. A_i = the node's metal interface area.
  It enters `newton_solve_2d` through the existing `generation` hook
  (nodal, analytic Jacobian), so assembly and the linear solver are
  untouched; the maximum-principle box stays on (the sink relaxes the
  quasi-Fermi levels toward the metal's, inside the box).
- Generality: valid wherever the interface curvature radius >> the
  barrier width (1-4 nm): planar, FinFET and nanosheet contacts alike. No
  orientation rule: A_i comes from the exact dual-cell facets into the
  metal (octree / rectilinear) or the boundary edges (2D).
- Terminal current of a Schottky contact = sum_i A_i (J_maj + J_min).

**Validation.** 1D: sub-grid model on 0.5-2 nm meshes vs the resolved
model on 0.05 nm (same structure, I-V within ~2%). 2D: wide-contact
slab vs 1D. 3D: planar NMOS with Schottky S/D vs ohmic (Ion drop, R_sd
from the contact drop), runtime vs the 57 s reference.

**Defaults.** `fab` S/D contacts `type: schottky` by default (user rule).
Barrier default: the value Study A infers from the measured NMOS rho_c,
Phi_Bn ~0.45 eV for Ti/TiN on n+ Si (open question: no single pinned
model fits the NMOS and PMOS data, see the README).

## 9. Process features for realistic contacts (user, 2026-10-12; after Schottky is solid)

Real S/D contacts are engineered for low rho_c; add both to the fab flow
(template options, on by default for the planar device once validated):
1. *Contact implant*: an extra shallow, high-dose implant through the
   contact opening so the doping where the metal touches is very high
   (the template's S/D implant leaves only ~4e19 at the surface: Ion 1792
   -> 683 uA/um with Schottky contacts; with ~6e20 at the surface the
   contact costs ~4.5%). The sub-grid table reads the local doping per
   contact node, so this works without model changes.
2. *Recessed contact*: etch the contact trench into the S/D silicon/epi
   before the TiN fill, so the metal contacts the trench floor AND its
   sidewalls (more area). `voxel_interface_areas` counts every
   semiconductor/metal face in any orientation, so the sidewalls are
   picked up automatically.
