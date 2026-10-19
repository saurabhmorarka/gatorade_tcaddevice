# Velocity saturation: fast and exact modes, and the near-threshold stall

[Back to the README](../README.md)

`fab simulate --vsat fast` (the default) or `--vsat exact`.
Numbers: DEVELOPMENT_LOG 75.

## The two driving forces

Velocity saturation needs the field that drives each carrier along its
current. On every mesh edge the solver uses:

- **fast** (`hybrid`): w·|∇φ| + (1−w)·|E| with w = c²/(c² + c_ref²) and
  c_ref = 1e16 cm⁻³. That is the quasi-Fermi gradient where the carrier is
  dense and the edge field where it is sparse.
- **exact** (`mqf`): the gradient of Sentaurus' modified quasi-Fermi potential,
  φ̃ = ψ − (kT/q)·ln((n + n0)/Nc), with n0 = 1e10 cm⁻³. That is ∇φ where n ≫ n0 and the
  field where n ≪ n0, smooth in ψ, φ and n on every edge. Sentaurus Device's default
  is n0 = 0 (pure ∇φ). Its user guide warns that n0 above ni shifts the results, and its
  examples use 1e8 to 1e12.

Why the fast mode is off near threshold: at 1e16 the hybrid already treats the
subthreshold channel and the junction tails as "sparse". There it uses the
edge field, which includes the junction's built-in field. That field is
balanced by diffusion and heats no one.

## Measured error of the fast mode (vs exact)

| device | Ion | Vt lin / sat, DIBL | near threshold | deep off-state (Vgs ≤ 0.1 V, low Vds) |
|---|---|---|---|---|
| 2D NMOS, Lg 0.5 µm | −0.04% | Vt +3 mV | −13 to −14% at Vds 0.05 V (exact is −2..−5% vs no saturation at all) | same |
| FinFET, Lg 15 nm | identical | identical | < 0.1% from Vgs 0.15 V up | up to −40% (SS 62 vs 73) |
| nanosheet PTS | identical (1128) | identical | — | Ioff 2.2e-8 vs 3.2e-8 (SS 61 vs 70) |

In the 2D check at Vds = 0.05 V the lateral field is tiny, so the reference is
a run with no saturation at all. Exact mode lands within 2-5% of it. The fast
mode is 13-14% below it near threshold. In the 3D short-channel devices the
two modes agree everywhere except the deep off-state, where the fast mode
gives less current.

Which deep-off value is right on the FinFET and nanosheet has not been checked
yet. The check is a run without saturation, as in 2D.

## Cost

| device | fast | exact |
|---|---|---|
| 2D NMOS, 138 points | 853 Newton iterations | 1240 (n0 1e10), 1310 (n0 1e8) |
| FinFET full I-V | 176 s | 667 s |
| nanosheet PTS full I-V | 541 s | 1952 s |

## The near-threshold stall, and the fix (both modes)

Traced on the FinFET walk at Vds = 0.7 V:
- After three good Newton iterations, the correction moves ψ by about 1e-9 V and densities by about 5e-4.
- The monotone line search then cuts every step to 1/16 or 1/32. A full step raises the row-scaled merit slightly, through the O(δ²) error of a few degenerate rows.
- Newton crawls about 1% per iteration to the 15-iteration cap, the continuation halves its step, and the same thing happens again.

Sentaurus's default is undamped Newton, declaring divergence only when the
residual rises a lot. The fix follows it: once the correction is small
(ψ < 1e-5 V, densities < 1%), the full step is taken unless the merit grows
more than 1000x. The convergence tests are unchanged, so the converged
answer is the same: the currents match to 6e-7.

## Edges into a contact use the edge field (both modes)

On an edge with a contact node at either end (ohmic or Schottky), the
driving force is the edge field |Δψ|/L, whatever force is chosen
(`VSAT_CONTACT_FIELD`, DEVELOPMENT_LOG 80). This is the Sentaurus default
("at contact elements the field replaces grad φ").

Why: the realistic FinFET at Vds 0.7 V stopped dead at Vgs 0.7654 V; halving
the gate step to 1e-5 V did not help. The residual sat on one Schottky node
at the drain contact's corner nearest the gate. That node lies inside the
contact's depletion (n 4e15 in a 7e20 epi), so the edges feeding it carry a
large φn drop (0.15 V over 0.25 nm) and saturate on it. With a quasi-Fermi
force, such an edge's current is about q·n·vsat·E/|∇φ|: it *falls* as the
drop grows. Once the contact sink asks for more than that maximum, there is
no solution: a fold, not a step-size problem. The edge field does not depend
on φ, so the supply is monotone and the contact always balances.

Effect: the failing step converges in 5 iterations. Full I-V, Id change
against the old runs: FinFET ideal ≤ 0.03%, nanosheet ideal ≤ 0.008%,
nanosheet realistic ≤ 0.04%, FinFET realistic ≤ 0.1%, and the realistic FinFET
now has its Vgs 0.8 V point (466 s instead of 1857 s). The tiny test devices,
whose contacts sit a few nm from the channel, move more: 3D planar tiny
Id_lin -3.5% / Id_off -7.5%, tiny Schottky device Id lin -10% / sat -0.3%.

## What else was tried (rejected)

- **Newton updates applied to the densities** (Genius/DEVSIM style), instead of to the quasi-Fermi levels.
  - It took more iterations in 2D: 893 → 1181.
  - It also accepted a wrong point, 2.4x off, through the update-only convergence test.
- **Freezing the hybrid weight** for one Newton solve, then polishing with the real force. FinFET 230 → 244 s.
- **The exact force as the speed fix.** It does not cure the stall, and it costs 2.5-3x in 3D.

## Sources

- Sentaurus Device User Guide (N-2017.09): "Interpolation of the GradQuasiFermi Driving Force" (eqs. 357-363), "Damped Newton Iterations", "Fully Coupled Solution".
- Genius-TCAD-Open, `src/solver/ddm1/ddm1_semiconductor.cc`: the density floor 1e-2·ni in the quasi-Fermi force, the ESurface interface field, and the default ESimple (edge-field) force.
- Charon User Manual (SAND2022-7653): driving-force reference densities.
- DEVSIM `python_packages/mos_physics.py`: the field along the low-field current direction.
