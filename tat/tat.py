"""Trap-assisted and pure band-to-band tunneling generation, for modeling
reverse-bias drain-to-substrate junction leakage (the dominant leakage
mechanism at moderate reverse bias, well before avalanche onset).

Pure physics, no solver/mesh dependency (mirrors avalanche/avalanche.py's
own stateless-function style) - sanity-checked standalone (see
sanity_probe() below) before being wired into any Newton solve.

Two additive mechanisms, both "local field" models (rate depends only on
the field AT that point, not on the band profile along an actual tunneling
path - see plans/tat_btbt_plan.md's roadmap note on the future nonlocal
path-search extension):

1. Kane band-to-band (Zener) tunneling - a direct electron-hole-pair
   generation, G_btbt = A * F^P * exp(-B/F). Coefficients transcribed
   directly from FLOOXS's working B2BTunnel/simple.tcl (open-source TCAD,
   ~/Desktop/github_flooxs/flooxs), P=2 (quadratic) as the default: the
   indirect-gap-appropriate variant for silicon.

2. Hurkx trap-assisted tunneling - field-enhanced Shockley-Read-Hall
   generation/recombination via a mid-gap trap. This is the mechanism the
   task actually asked for ("trap states inside the bandgap that assist
   this tunneling"): the standard SRH capture/emission rate this project
   already implements at Et=Ei (physics.py:srh_recombination) gets an
   extra field-dependent enhancement factor Gamma(F) on the trap's
   effective lifetime, tau -> tau/(1+Gamma(F)). At F=0, Gamma=0 and this
   exactly reduces to ordinary SRH (a strict generalization, not a
   reverse-bias-only hack). The formula's structure (field-enhanced SRH
   lifetime) was recovered from a primary source that reproduces Hurkx,
   Klaassen & Knuvers (IEEE Trans. Electron Devices 39(2), 1992) directly:
   M.S. Carroll et al., Sandia National Laboratories, SAND2007-1497C.
   Gamma(F) is Hurkx's tunneling integral itself (see hurkx_gamma): an
   earlier closed form used here, Delta*exp(Delta)*E1(Delta), tends to 1 at
   high field, capping the enhancement at 2x SRH - found while porting this
   model to 2D (btbt/, DEVELOPMENT_LOG.md), where silicon's trap-assisted
   leakage should dominate.

Schenk's more microscopic phonon-assisted trap-assisted-tunneling model
(also present in FLOOXS, TclLib/Device/floods/Generic/B2BTunnel/schenk.tcl)
is planned as a second, swappable trap-assisted model once Hurkx is
validated (see plans/tat_btbt_plan.md) - not implemented in this module yet.
"""
from dataclasses import dataclass

import numpy as np


# ---- 1. Kane band-to-band (Zener) tunneling ----

@dataclass
class KaneBTBTModel:
    """G_btbt(F) = A * F^P * exp(-B/F), F in V/cm, G in cm^-3 s^-1.
    Default (A, B, P) is the P=2 (quadratic) fit from FLOOXS's
    B2BTunnel/simple.tcl - the indirect-gap-appropriate Kane variant for
    silicon. The P=1 and P=1.5 variants from the same source are also
    exposed as alternate constructors for cross-checking model
    sensitivity, not because either is expected to be more correct.

    F_sat_V_cm is an ENGINEERING upper-field cap, not a literature-sourced
    validity bound like avalanche's E_floor_V_cm/E_split_V_cm - FLOOXS's
    source (Section 0 of plans/tat_btbt_plan.md) documents no upper field
    limit at all, because exp(-B/F) alone would asymptote to 1 and stop
    growing, but the unbounded F^P prefactor keeps growing forever past
    that. tat/main_tat_doping_sweep.py's own drain-doping sweep (1e18 to
    1e21 cm^-3, substrate fixed at 1e17) found this in practice: peak
    field stays near 2e5-6e5 V/cm across 1e18-1e20 (the doping is already
    one-sided enough there that further doping barely narrows the
    depletion width further), giving physically reasonable leakage
    currents that track each other closely, but at 1e21 the peak field
    jumps to ~1.7e6 V/cm and G_btbt explodes from ~4e16 to ~3e28
    cm^-3 s^-1 (12 orders of magnitude for a 3x field change) - a
    physically implausible current spike confirmed NOT to be a solver/
    Jacobian bug (the plain no-tunneling baseline stays perfectly
    well-behaved at the identical doping/mesh).

    F_sat_V_cm=9e5 is set just ABOVE the peak field this project's own
    shipped 2e20-drain example reaches (~7.84e5 V/cm at Va=-1V) - i.e. the
    highest field this project has actually cross-checked the resulting
    leakage CURRENT at (via main_tat.py's I-V comparison against the
    no-tunneling baseline, not just the bare formula in isolation), not an
    arbitrary round number and not "just above where 1e20 first diverges"
    (that would still be deep in unvalidated-extrapolation territory - at
    F=1.5e6, G_btbt is already ~4e27, thirty orders of magnitude beyond
    the ~2e21 the validated example device ever reaches). Capping F at
    F_sat_V_cm before evaluating the Kane formula (same hard-floor-style
    treatment avalanche.py already uses at its own, opposite, LOW-field
    limit) keeps every doping level's prediction anchored to the same
    order of magnitude as the one case this project has actually
    validated, instead of extrapolating an exponential arbitrarily far
    past it."""
    A: float = 3.4e21    # cm^-1 s^-1 V^-2 (P=2 units)
    B: float = 21.6e6    # V/cm
    P: float = 2.0
    F_sat_V_cm: float = 9.0e5

    @staticmethod
    def si_kane_quadratic() -> "KaneBTBTModel":
        return KaneBTBTModel(A=3.4e21, B=21.6e6, P=2.0)

    @staticmethod
    def si_kane_linear() -> "KaneBTBTModel":
        return KaneBTBTModel(A=1.1e27, B=21.3e6, P=1.0)

    @staticmethod
    def si_kane_three_half() -> "KaneBTBTModel":
        return KaneBTBTModel(A=1.9e24, B=21.9e6, P=1.5)


def btbt_generation(F_abs, model: KaneBTBTModel):
    """Vectorized G_btbt(F) and dG_btbt/dF. F_abs: field magnitude (V/cm),
    any shape, >= 0. No LOW-field floor is applied (unlike avalanche's
    alpha(E) hard floor) - F^P -> 0 as F -> 0 and exp(-B/F) vanishes even
    faster, so the functional form itself vanishes smoothly at low field.
    An upper-field cap IS applied (F_sat_V_cm, see KaneBTBTModel's
    docstring for why) - G_btbt(F) is exactly flat for F >= F_sat_V_cm
    (evaluated at F_sat_V_cm instead of F, so dG/dF is exactly 0 there,
    the same hard-floor style avalanche.py already uses at its own
    low-field limit, not a smoothed transition).

    dG/dF = A*exp(-B/F) * (P*F^(P-1) + F^P * B/F^2)
          = G/F * (P + B/F)      (valid wherever G != 0; handled via the
            safe zero-field branch below rather than a literal division).
    """
    F = np.asarray(F_abs, dtype=float)
    F_safe = np.where(F > 0, F, 1.0)  # avoid 0-division; overwritten below
    F_capped = np.minimum(F_safe, model.F_sat_V_cm)
    G = model.A * F_capped ** model.P * np.exp(-model.B / F_capped)
    dG_dF = G * (model.P / F_capped + model.B / F_capped ** 2)
    zero = F <= 0
    saturated = F >= model.F_sat_V_cm
    G = np.where(zero, 0.0, G)
    dG_dF = np.where(zero | saturated, 0.0, dG_dF)
    return G, dG_dF


# ---- 2. Hurkx trap-assisted tunneling ----

_M0 = 9.1093837015e-31   # electron rest mass, kg
_HBAR = 1.054571817e-34  # J s
_KB_SI = 1.380649e-23    # J/K
_Q_SI = 1.602176634e-19  # C


@dataclass
class HurkxTATModel:
    """Field-enhanced SRH generation/recombination via a trap at
    Et_minus_Ei_eV relative to the intrinsic level (0.0 = midgap, this
    project's existing srh_recombination convention AND the literature's
    standard worst-case/typical assumption for maximal SRH-generation
    traps). m_t_over_m0 is the tunneling effective mass, default 0.25
    (the commonly cited Si literature value), applied identically to both
    carriers absent per-carrier tunneling-mass data."""
    m_t_over_m0: float = 0.25
    Et_minus_Ei_eV: float = 0.0
    Eg_eV: float = 1.12     # sets the tunneling-energy range, see hurkx_gamma


_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


def hurkx_gamma(F_abs, T, model: HurkxTATModel, dE_eV=None):
    """Vectorized Hurkx field-enhancement factor Gamma(F) and dGamma/dF.

    Hurkx, Klaassen & Knuvers (IEEE TED 39(2), 1992), eq. 11: a carrier
    reaches the trap by tunneling through the triangular barrier at an
    energy u*kT below the band edge instead of being thermally excited over it,

        Gamma = integral_0^{dE/kT} exp(u - K u^(3/2)) du,
        K = (4/3) sqrt(2 m_t (kT)^3) / (q hbar F) = (4/(3 sqrt 12)) F_Gamma/F,
        F_Gamma = sqrt(24 m_t (kT)^3) / (q hbar),

    where dE is the trap depth measured from the band edge (default
    Eg/2 - |Et - Ei|; may be an array broadcasting against F, e.g. a
    per-node Ec - Et at a heterojunction). At moderate field the integrand peaks inside the
    range and Gamma ~ 2 sqrt(3 pi) (F/F_Gamma) exp((F/F_Gamma)^2), which is
    the usual Hurkx closed form. At very high field the peak runs into the
    upper limit and Gamma saturates near exp(dE/kT) (~1e9 for a midgap Si
    trap): the carrier then tunnels straight from the trap to the band edge.
    Gamma -> 0 as F -> 0.

    It is evaluated by 96-point Gauss-Legendre quadrature on [0, dE/kT]. The
    integrand is smooth, and its width is always >= ~K^(-2/3). dGamma/dF
    comes from the same quadrature, since dK/dF = -K/F:
        dGamma/dF = integral exp(u - K u^1.5) * K u^1.5 / F du.
    """
    F = np.asarray(F_abs, dtype=float)
    m_t = model.m_t_over_m0 * _M0
    kT = _KB_SI * T
    F_Gamma = np.sqrt(24.0 * m_t * kT ** 3) / (_Q_SI * _HBAR) / 100.0   # V/cm
    if dE_eV is None:
        dE_eV = 0.5 * model.Eg_eV - abs(model.Et_minus_Ei_eV)
    umax = np.asarray(dE_eV, dtype=float)[..., None] * _Q_SI / kT     # per point allowed
    u = 0.5 * umax * (_GL_X + 1.0)
    w = 0.5 * umax * _GL_W
    F_safe = np.where(F > 0, F, 1.0)
    K = (4.0 / (3.0 * np.sqrt(12.0))) * F_Gamma / F_safe
    u15 = u ** 1.5
    E = np.exp(u - K[..., None] * u15)
    Gamma = np.sum(E * w, axis=-1)
    dGamma_dF = np.sum(E * u15 * w, axis=-1) * K / F_safe
    zero = F <= 0
    return np.where(zero, 0.0, Gamma), np.where(zero, 0.0, dGamma_dF)


def hurkx_tat_generation(n, p, F_abs, mat, model: HurkxTATModel):
    """Net generation rate (>0 = generation) from the Hurkx-enhanced SRH
    trap, and its partial derivatives w.r.t. n, p, F_abs (each broadcast
    to the common shape of n, p, F_abs).

    R_trap = (n*p - ni^2) / ( tau_p/(1+Gamma)*(n+n1) + tau_n/(1+Gamma)*(p+p1) )
    G_tat = -R_trap

    n1 = ni*exp(Et_minus_Ei/Vt), p1 = ni*exp(-Et_minus_Ei/Vt) (=ni at Et=Ei,
    this project's existing srh_recombination special case). A single
    Gamma(F) multiplies BOTH tau_n and tau_p (Gamma_n = Gamma_p here, per
    HurkxTATModel's single tunneling mass, see its docstring) - Hurkx's
    own formula keeps Gamma_n, Gamma_p distinct in general (separate
    electron/hole tunneling masses), collapsed here to one value since
    this project has no separate per-carrier tunneling-mass data.

    In reverse bias (n, p << ni), R_trap is strongly negative -> G_tat > 0
    (generation, the same sign convention as avalanche's G_ii). In forward
    bias / near-equilibrium, this reduces toward ordinary SRH
    recombination as Gamma -> 0 at low field - a strict generalization of
    physics.py:srh_recombination, not a reverse-bias-only special case.
    """
    n = np.asarray(n, dtype=float)
    p = np.asarray(p, dtype=float)
    ni = mat.ni
    Vt = mat.Vt
    T = mat.T

    Gamma, dGamma_dF = hurkx_gamma(F_abs, T, model)
    n1 = ni * np.exp(model.Et_minus_Ei_eV / Vt)
    p1 = ni * np.exp(-model.Et_minus_Ei_eV / Vt)

    tau_p_eff = mat.tau_p / (1.0 + Gamma)
    tau_n_eff = mat.tau_n / (1.0 + Gamma)

    num = n * p - ni ** 2
    denom = tau_p_eff * (n + n1) + tau_n_eff * (p + p1)

    R = num / denom
    G_tat = -R

    dR_dn = (p * denom - num * tau_p_eff) / denom ** 2
    dR_dp = (n * denom - num * tau_n_eff) / denom ** 2

    # d(denom)/dGamma = -tau_p*(n+n1)/(1+Gamma)^2 - tau_n*(p+p1)/(1+Gamma)^2
    #                 = -(denom_at_this_Gamma_but_with_(1+Gamma)_extra_factor)
    # more directly: d(tau_x_eff)/dGamma = -tau_x/(1+Gamma)^2 = -tau_x_eff/(1+Gamma)
    ddenom_dGamma = (-tau_p_eff * (n + n1) - tau_n_eff * (p + p1)) / (1.0 + Gamma)
    dR_dGamma = -num * ddenom_dGamma / denom ** 2
    dR_dF = dR_dGamma * dGamma_dF

    return G_tat, -dR_dn, -dR_dp, -dR_dF


# ---- 3. Schenk trap-assisted tunneling (phonon-assisted, second model) ----

@dataclass
class SchenkTATModel:
    """Schenk's more microscopic, phonon-assisted trap-assisted-tunneling
    model - a second, swappable trap-assisted model alongside Hurkx (see
    plans/tat_btbt_plan.md Section 0 for why Hurkx was implemented first
    and this is the planned cross-check). Coefficients transcribed
    directly from FLOOXS's B2BTunnel/schenk.tcl (Section 0 of the plan):
    `A`/`B` are fixed materials constants (not per-trap fit knobs the way
    Hurkx's tau_n/tau_p are), `hbar_omega_eV` is the phonon energy
    mediating the transition (a TO-phonon-like value for Si).

    This implementation is the F=E (plain local field, no quasi-Fermi-
    gradient density correction) special case of FLOOXS's own formula -
    same "local field" tier as Kane and Hurkx here, not the QF-gradient
    refinement FLOOXS's usage docstring lists as optional. At F=E, FLOOXS's
    own n_eff/p_eff density correction (`Elec*((ni/Nc)^(...))`) collapses
    exactly to n, p themselves (the correction exponent is proportional to
    Emfn/F, and Emfn - the quasi-Fermi GRADIENT magnitude - is a separate
    quantity this simplification drops entirely, not merely sets equal to
    F)."""
    A: float = 8.977e20     # V/(cm*eV^1.5), FLOOXS schenk.tcl
    B: float = 2.14667e7    # 1/(cm*s*V^2)
    hbar_omega_eV: float = 0.0186   # phonon energy, Si TO-phonon-like value


def schenk_tat_generation(n, p, F_abs, mat, model: SchenkTATModel):
    """Net generation rate (>0 = generation) from the Schenk phonon-
    assisted trap-assisted-tunneling model, and its partial derivatives
    w.r.t. n, p, F_abs - same call signature as hurkx_tat_generation so
    newton_solver_tat.py can treat "which trap model" as a single
    swappable argument.

    SchenkSRH = (n*p - ni^2) / ((n+ni)*(p+ni))   [F=E special case, see
        SchenkTATModel's docstring - this is dimensionless, NOT itself a
        rate: FLOOXS uses it as a normalized sign-and-magnitude selector
        multiplying the tunneling prefactor below, the same role
        `newton_solver_qf.py`'s ordinary `R` plays before being multiplied
        by 1/tau there.]
    dSchenkSRH/dn = ni/(n+ni)^2, dSchenkSRH/dp = ni/(p+ni)^2 (a clean
        closed form after simplifying the quotient rule - verified
        against the general product/quotient rule by hand).

    s = sign(SchenkSRH) (hard branch at exactly 0, same non-differentiable-
        at-a-single-point acceptance this project already uses for
        avalanche's sign(E)/sign(J)):
    Fc_pm = B*(Eg + s*hw)^1.5      Fc_mp = B*(Eg - s*hw)^1.5
    K(F)  = A*F^3.5 * ( Fc_mp^-1.5*exp(-Fc_mp/F)/(exp(hw/Vt)-1)
                       + Fc_pm^-1.5*exp(-Fc_pm/F)/(1-exp(-hw/Vt)) )
    G_schenk = -SchenkSRH * K(F)

    dG/dn = -(dSchenkSRH/dn)*K(F)   [Fc_pm/Fc_mp/K's dependence on n, p is
        only through the discrete sign branch s, treated as locally
        constant away from SchenkSRH=0, the same way Kane/Hurkx treat
        their own hard branches - NOT smoothed]
    dG/dp = -(dSchenkSRH/dp)*K(F)
    dG/dF = -SchenkSRH * dK/dF,
    dK/dF = A*F^1.5*(3.5*F*(T1+T2) + T1*Fc_mp + T2*Fc_pm)
        where T1, T2 are the two exp(...)-bracketed terms in K(F) above
        (derived via d/dF[Fc^-1.5*exp(-Fc/F)] = Fc^-1.5*exp(-Fc/F)*(Fc/F^2)
        = T*(Fc/F^2), then product rule on F^3.5*(T1+T2)).

    Not validated at cryogenic temperatures (hbar*omega/Vt can overflow
    exp() well below ~50K) - same documented-but-unhandled-outside-
    intended-range spirit as avalanche.py's own 300K-fit limitation.
    """
    n = np.asarray(n, dtype=float)
    p = np.asarray(p, dtype=float)
    F = np.asarray(F_abs, dtype=float)
    ni = mat.ni
    Vt = mat.Vt
    Eg = mat.Eg_eV
    hw = model.hbar_omega_eV

    num = n * p - ni ** 2
    denom = (n + ni) * (p + ni)
    SchenkSRH = num / denom
    dSSRH_dn = ni / (n + ni) ** 2
    dSSRH_dp = ni / (p + ni) ** 2

    s = np.where(SchenkSRH >= 0, 1.0, -1.0)
    Fc_pm = model.B * (Eg + s * hw) ** 1.5
    Fc_mp = model.B * (Eg - s * hw) ** 1.5

    F_safe = np.where(F > 0, F, 1.0)
    exp_hw_Vt = np.exp(hw / Vt)
    T1 = Fc_mp ** -1.5 * np.exp(-Fc_mp / F_safe) / (exp_hw_Vt - 1.0)
    T2 = Fc_pm ** -1.5 * np.exp(-Fc_pm / F_safe) / (1.0 - 1.0 / exp_hw_Vt)
    K = model.A * F_safe ** 3.5 * (T1 + T2)
    dK_dF = model.A * F_safe ** 1.5 * (3.5 * F_safe * (T1 + T2) + T1 * Fc_mp + T2 * Fc_pm)

    zero = F <= 0
    K = np.where(zero, 0.0, K)
    dK_dF = np.where(zero, 0.0, dK_dF)

    G_schenk = -SchenkSRH * K
    dG_dn = -dSSRH_dn * K
    dG_dp = -dSSRH_dp * K
    dG_dF = -SchenkSRH * dK_dF

    return G_schenk, dG_dn, dG_dp, dG_dF


def sanity_probe():
    """Standalone order-of-magnitude / ordering check, run directly
    (`python3 -m tat.tat`), no solver/mesh dependency. Verifies:
    (a) both generation terms are non-negative and finite at a
        representative peak junction field range,
    (b) hurkx_gamma is ~0 at low field and grows with field,
    (c) the "TAT turns on before BTBT dominates" ordering the task
        expects: at moderate field, G_tat should be non-negligible while
        G_btbt is still comparatively small; only at much higher field
        should G_btbt catch up/dominate.
    """
    from core.params import Material

    mat = Material()  # 300K Si defaults, ni=1e10, tau_n=tau_p=1e-9
    kane = KaneBTBTModel.si_kane_quadratic()
    hurkx = HurkxTATModel()

    F_range = np.array([1e3, 1e4, 3e4, 1e5, 2e5, 3e5, 5e5, 7e5, 1e6, 1.5e6, 2e6])
    n = np.full_like(F_range, 1.0)   # deep depletion, reverse bias: n,p << ni
    p = np.full_like(F_range, 1.0)

    G_btbt, _ = btbt_generation(F_range, kane)
    Gamma, _ = hurkx_gamma(F_range, mat.T, hurkx)
    G_tat, _, _, _ = hurkx_tat_generation(n, p, F_range, mat, hurkx)

    print(f"{'F (V/cm)':>12} {'Gamma':>12} {'G_tat':>14} {'G_btbt':>14}")
    for F, g, gt, gb in zip(F_range, Gamma, G_tat, G_btbt):
        print(f"{F:12.2e} {g:12.4e} {gt:14.4e} {gb:14.4e}")

    assert np.all(np.isfinite(G_btbt)) and np.all(G_btbt >= 0)
    assert np.all(np.isfinite(G_tat))
    assert np.all(np.isfinite(Gamma)) and np.all(Gamma >= 0)
    assert Gamma[0] < 1e-3, "Gamma should be ~0 at low field"
    assert np.all(np.diff(Gamma) >= 0), "Gamma should be monotonically increasing with field"

    # Ordering check: find the field where G_tat first exceeds G_btbt
    # substantially (TAT dominates at lower field) vs. where G_btbt
    # catches up (higher field). log-ratio avoids overflow when G_btbt
    # underflows to exactly 0.0 at low field (expected - see printed table).
    with np.errstate(divide="ignore"):
        log_ratio = np.log(G_tat) - np.where(G_btbt > 0, np.log(G_btbt), -700.0)
    print(f"\nlog(G_tat/G_btbt): {log_ratio}")
    assert log_ratio[0] > log_ratio[-1], (
        "expected G_tat to dominate G_btbt at low field and BTBT to catch "
        "up at high field (TAT turns on earlier/softer, per the task's "
        "own description) - got the opposite trend"
    )


if __name__ == "__main__":
    sanity_probe()
