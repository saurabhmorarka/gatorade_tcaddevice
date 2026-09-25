"""Closed-form (textbook) results for a step p-n junction, for comparison
against the numerical drift-diffusion simulation.

References: Sze, "Physics of Semiconductor Devices"; Pierret, "Semiconductor
Device Fundamentals".
"""
import numpy as np

from core.params import Q, Material, Device


def built_in_potential(mat_p: Material, dev: Device, mat_n: Material = None) -> float:
    """Vbi = Vt * ln(Na*Nd/ni^2) for a homojunction (mat_n=None, default -
    every existing call site passes just (mat, dev) and gets this exact
    formula, unchanged - mirrors core.mesh.build_diode_grid's own
    mat_n=None backward-compat pattern).

    mat_n given (a real heterojunction): returns the genuinely independent
    (no PDE/mesh dependency) closed form Vbi_hetero, built from each side's
    own EXACT bulk equilibrium potential (core.physics.equilibrium_
    bulk_potential - material-only, no solve) minus that side's
    delta_Ei (core.materials.delta_Ei_of - also material-only): the SAME
    quantity core.materials.MaterialField.from_regions computes for the
    PDE solve, reused here (not re-derived) so the closed form and the
    solve's own boundary conditions agree by construction.
        psi_p_bulk = equilibrium_bulk_potential(mat_p, -Na) - delta_Ei_of(mat_p, mat_n)
        psi_n_bulk = equilibrium_bulk_potential(mat_n,  Nd) - delta_Ei_of(mat_n, mat_n) = equilibrium_bulk_potential(mat_n, Nd)
        Vbi_hetero = psi_n_bulk - psi_p_bulk
    Reduces to the plain formula above when mat_p=mat_n (delta_Ei=0, and
    equilibrium_bulk_potential's EXACT n0/p0 solve collapses to the
    Na,Nd>>ni approximation the plain formula uses, verified numerically in
    testsuite/test_analytic_tat.py for every doping level this project's
    examples actually use)."""
    if mat_n is None:
        return mat_p.Vt * np.log(dev.Na * dev.Nd / mat_p.ni ** 2)

    from core.materials import delta_Ei_of
    from core.physics import equilibrium_bulk_potential
    psi_p_bulk = equilibrium_bulk_potential(mat_p, -dev.Na) - delta_Ei_of(mat_p, mat_n)
    psi_n_bulk = equilibrium_bulk_potential(mat_n, dev.Nd)  # delta_Ei_of(mat_n, mat_n) == 0
    return psi_n_bulk - psi_p_bulk


def depletion_widths(mat_p: Material, dev: Device, Va: float = 0.0, mat_n: Material = None):
    """Depletion-approximation widths on each side, step junction, under
    applied bias Va (forward positive on the p-side). Returns (xp, xn, W).

    mat_n=None (default): homojunction, today's exact formula (every
    existing call site keeps passing 2-3 positional args and is unaffected).

    mat_n given: two-material (heterojunction) step-junction depletion
    approximation (standard Sze-style derivation - see built_in_potential's
    own docstring for the Vbi this uses). D-field continuity at x=0 still
    reduces to the SAME charge-balance relation Na*xp=Nd*xn regardless of
    eps_p/eps_n (D_max_p=q*Na*xp=D_max_n=q*Nd*xn by construction below, not
    assumed), so only the VOLTAGE-drop relation needs the two eps's:
        V = Vbi-Va = (q/2)*(Na*xp^2/eps_p + Nd*xn^2/eps_n)
    Substituting xp=xn*Nd/Na and solving for xn:
        xn = sqrt(2*V / (q*Nd*(Nd/(Na*eps_p) + 1/eps_n)))
        xp = xn*Nd/Na
    Verified BY HAND (and in testsuite/test_analytic_tat.py) that this
    reduces algebraically EXACTLY to the homojunction W/xp/xn formula below
    when eps_p=eps_n=eps (not merely approximately): with eps_p=eps_n=eps,
    xn^2 = 2*V*eps*Na/(q*Nd*(Na+Nd)), which is exactly W^2*Na^2/(Na+Nd)^2
    with W as defined below - same algebra, not a coincidence.
    W is returned as xp+xn in the heterojunction case (still the total
    depletion width; "W=sqrt(...)" doesn't individually apply once eps_p
    != eps_n, since xp/xn no longer share one common prefactor)."""
    Vbi = built_in_potential(mat_p, dev, mat_n=mat_n)
    V = Vbi - Va  # total potential dropped across the junction under bias
    V = max(V, 1e-6)  # guard against forward bias collapsing the depletion width in this simple formula

    if mat_n is None:
        W = np.sqrt(2 * mat_p.eps * V / Q * (1.0 / dev.Na + 1.0 / dev.Nd))
        xp = W * dev.Nd / (dev.Na + dev.Nd)   # depletion extent into p-side
        xn = W * dev.Na / (dev.Na + dev.Nd)   # depletion extent into n-side
        return xp, xn, W

    eps_p, eps_n = mat_p.eps, mat_n.eps
    xn = np.sqrt(2.0 * V / (Q * dev.Nd * (dev.Nd / (dev.Na * eps_p) + 1.0 / eps_n)))
    xp = xn * dev.Nd / dev.Na
    return xp, xn, xp + xn


def depletion_potential_profile(mat: Material, dev: Device, x: np.ndarray, Va: float = 0.0):
    """Analytic (depletion-approximation) electrostatic potential profile,
    referenced the same way as the simulation (psi=0 where n=p=ni), for a
    step junction at x=0 under bias Va."""
    from core.physics import equilibrium_bulk_potential
    xp, xn, W = depletion_widths(mat, dev, Va)
    psi_p_bulk = equilibrium_bulk_potential(mat, -dev.Na)
    psi_n_bulk = equilibrium_bulk_potential(mat, dev.Nd)

    psi = np.empty_like(x)
    left_dep = (x >= -xp) & (x < 0)
    right_dep = (x >= 0) & (x <= xn)
    bulk_p = x < -xp
    bulk_n = x > xn

    psi[bulk_p] = psi_p_bulk
    psi[bulk_n] = psi_n_bulk + Va
    # quadratic potential in depletion region (integrate the triangular field twice)
    psi[left_dep] = psi_p_bulk + (Q * dev.Na / (2 * mat.eps)) * (x[left_dep] + xp) ** 2
    psi[right_dep] = psi_n_bulk + Va - (Q * dev.Nd / (2 * mat.eps)) * (x[right_dep] - xn) ** 2
    return psi


def shockley_I0(mat: Material, dev: Device) -> float:
    """Long-base ideal diode saturation current: I0 = q*A*ni^2*(Dp/(Lp*Nd) + Dn/(Ln*Na))."""
    J0 = Q * mat.ni ** 2 * (mat.Dp / (mat.Lp * dev.Nd) + mat.Dn / (mat.Ln * dev.Na))
    return J0 * dev.area


def shockley_current(mat: Material, dev: Device, Va: np.ndarray) -> np.ndarray:
    """Ideal Shockley diode law I(Va) = I0*(exp(Va/Vt)-1)."""
    I0 = shockley_I0(mat, dev)
    return I0 * (np.exp(Va / mat.Vt) - 1.0)


def built_in_potential_fd(mat: Material, dev: Device) -> float:
    """Fermi-Dirac analog of built_in_potential(): Vbi from the two sides'
    Fermi-Dirac (not Boltzmann) equilibrium bulk potentials - see
    fermi_dirac.py. Reduces to built_in_potential() wherever neither side
    is degenerate enough for the difference to matter."""
    from core import fermi_dirac as fd
    psi_n = fd.equilibrium_bulk_potential_fd(mat, dev.Nd)
    psi_p = fd.equilibrium_bulk_potential_fd(mat, -dev.Na)
    return psi_n - psi_p


def shockley_I0_fd(mat: Material, dev: Device) -> float:
    """Fermi-Dirac-corrected Shockley I0: the two minority-carrier
    equilibrium reference densities (p0 in the n-side bulk, n0 in the
    p-side bulk) are computed directly from Fermi-Dirac statistics at each
    side's own doping (fermi_dirac.p_fd/n_fd) instead of the Boltzmann
    mass-action shortcut ni^2/N - everything else (the "law of the
    junction" boundary condition, the long-base diffusion solution this
    formula's Dp/Lp, Dn/Ln prefactors come from) is unchanged, so this
    stays a Boltzmann-consistent MINORITY-carrier-injection picture; only
    the equilibrium reference each side's exponential injection is
    measured FROM is corrected for the majority side's own degeneracy."""
    from core import fermi_dirac as fd
    psi_n = fd.equilibrium_bulk_potential_fd(mat, dev.Nd)
    psi_p = fd.equilibrium_bulk_potential_fd(mat, -dev.Na)
    p0_n_side = fd.p_fd(mat, psi_n)   # minority holes in the n-side bulk
    n0_p_side = fd.n_fd(mat, psi_p)   # minority electrons in the p-side bulk
    J0 = Q * (mat.Dp / mat.Lp * p0_n_side + mat.Dn / mat.Ln * n0_p_side)
    return J0 * dev.area


def depletion_capacitance(mat: Material, dev: Device, Va: np.ndarray) -> np.ndarray:
    """Depletion (junction) capacitance per unit area, F/cm^2: C_dep =
    eps/W(Va), from the same depletion-approximation width already used
    for the equilibrium band-diagram comparison. Dominates in reverse
    bias and near zero bias; becomes an increasingly poor approximation
    approaching/exceeding Vbi (W formally -> 0), same caveat depletion_widths()
    already carries."""
    Va = np.atleast_1d(np.asarray(Va, dtype=float))
    W = np.array([depletion_widths(mat, dev, v)[2] for v in Va])
    return mat.eps / W


def diffusion_capacitance(mat: Material, dev: Device, Va: np.ndarray,
                           p0_n_side: float = None, n0_p_side: float = None) -> np.ndarray:
    """Diffusion capacitance per unit area, F/cm^2, from the standard
    long-base minority-charge-storage result: each side's excess stored
    minority charge Q_stored = q*L*p0*(exp(Va/Vt)-1) (the same profile
    integrated to give the Shockley current's diffusion term), differentiated
    wrt Va. Dominates in forward bias once diffusion current exceeds the
    (here-ignored) generation/recombination-only depletion-region current.

    p0_n_side/n0_p_side: equilibrium minority densities to use (defaults to
    the Boltzmann ni^2/N values, matching shockley_I0(); pass the
    Fermi-Dirac equivalents from shockley_I0_fd()'s own calculation for the
    FD-consistent comparison curve)."""
    if p0_n_side is None:
        p0_n_side = mat.ni ** 2 / dev.Nd
    if n0_p_side is None:
        n0_p_side = mat.ni ** 2 / dev.Na
    Va = np.asarray(Va, dtype=float)
    return (Q / mat.Vt) * (mat.Lp * p0_n_side + mat.Ln * n0_p_side) * np.exp(Va / mat.Vt)


def breakdown_voltage_sze(mat: Material, dev: Device) -> float:
    """Sze's empirical avalanche breakdown voltage for a Si one-sided
    abrupt junction (Sze, "Physics of Semiconductor Devices"):
    BV ~= 60*(Eg/1.1)^1.5 * (N_B/1e16)^-0.75 volts, N_B = the LIGHTER
    side's doping (the side that holds essentially all the depletion
    width/field in a one-sided junction, and so sets the breakdown field).
    A pure closed-form number (no simulation dependency) - the primary
    sanity-check target for the avalanche solver's I(Va) runaway."""
    N_B = min(dev.Na, dev.Nd)
    return 60.0 * (mat.Eg_eV / 1.1) ** 1.5 * (N_B / 1.0e16) ** -0.75


def ionization_integral(mat: Material, dev: Device, Va: float, ii_model) -> float:
    """The two-carrier ionization-integral breakdown criterion, evaluated on
    the closed-form depletion-approximation field profile (NOT the numeric
    PDE solve): the larger of the electron- and hole-initiated integrals

        I_n = int alpha_n * exp(-int (alpha_n - alpha_p) dx') dx   (and I_p),

    which reaches 1 exactly where the multiplication factor diverges. An
    independent closed-form cross-check against the numeric avalanche
    solver's breakdown, using the SAME avalanche.ionization_coeffs model but
    the textbook triangular field instead of the self-consistent one.

    An earlier version used the uncoupled single-carrier shortcut
    int max(alpha_n, alpha_p) dx, which overcounts ionization by ~50% here:
    it crossed 1 at ~11 V for the 1e19/1e17 breakdown example - agreeing with
    Sze's 11 V by coincidence - while this coupled form crosses 1 at ~14.0 V,
    matching the numeric solver's 14.08 V (DEVELOPMENT_LOG.md session 23)."""
    from avalanche.avalanche import ionization_integrals_from_field
    xp, xn, W = depletion_widths(mat, dev, Va)
    V = max(built_in_potential(mat, dev) - Va, 1e-6)
    # Triangular field of the depletion approximation: peak E_max at x=0,
    # decaying linearly to 0 at each depletion edge; psi is its integral
    # (only |dpsi/dx| enters the criterion, so the sign convention is moot).
    E_max = 2.0 * V / W
    x = np.linspace(-xp, xn, 4001)
    E_abs = np.clip(E_max * (1.0 - np.abs(x) / np.where(x < 0, xp, xn)), 0.0, None)
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (E_abs[1:] + E_abs[:-1]) * np.diff(x))])
    return max(ionization_integrals_from_field(x, psi, ii_model))


def multiplication_factor_miller(Va: np.ndarray, BV: float, n: float = 3.0) -> np.ndarray:
    """Miller's empirical avalanche multiplication factor M(Va) =
    1/(1-(|Va|/BV)^n), n~3 for a one-sided p+/n- junction (n~4-6 for
    n+/p-). A second, independent closed-form curve (distinct from the
    ionization-integral criterion above) to overlay against the numeric
    M(Va) = I_avalanche(Va)/I_no_avalanche(Va)."""
    Va = np.asarray(Va, dtype=float)
    ratio = np.clip(np.abs(Va) / BV, 0.0, 1.0 - 1e-6)
    return 1.0 / (1.0 - ratio ** n)


def cv_curve_analytic(mat: Material, dev: Device, Va: np.ndarray, use_fd: bool = False):
    """Combined depletion + diffusion capacitance per unit area, F/cm^2 -
    an approximate closed-form reference (the two mechanisms are simply
    added, which is standard pedagogically but only accurate away from the
    transition region where both are comparable). use_fd=True substitutes
    the Fermi-Dirac equilibrium minority references (shockley_I0_fd's
    p0_n_side/n0_p_side) into the diffusion term, and the Fermi-Dirac Vbi
    into the depletion term (via a per-call Device-like override), leaving
    everything else identical - the gap between the two curves is exactly
    the closed-form counterpart of the Boltzmann-vs-Fermi-Dirac comparison
    the numeric solve can't show directly (see fermi_dirac.py's module
    docstring for why the numeric PDE solve itself stays Boltzmann-only)."""
    from core import fermi_dirac as fd
    if not use_fd:
        return depletion_capacitance(mat, dev, Va) + diffusion_capacitance(mat, dev, Va)

    psi_n = fd.equilibrium_bulk_potential_fd(mat, dev.Nd)
    psi_p = fd.equilibrium_bulk_potential_fd(mat, -dev.Na)
    p0_n_side = fd.p_fd(mat, psi_n)
    n0_p_side = fd.n_fd(mat, psi_p)
    Vbi_fd = psi_n - psi_p

    Va = np.atleast_1d(np.asarray(Va, dtype=float))
    V = np.maximum(Vbi_fd - Va, 1e-6)
    W = np.sqrt(2 * mat.eps * V / Q * (1.0 / dev.Na + 1.0 / dev.Nd))
    C_dep_fd = mat.eps / W
    C_diff_fd = diffusion_capacitance(mat, dev, Va, p0_n_side=p0_n_side, n0_p_side=n0_p_side)
    return C_dep_fd + C_diff_fd


# ---- TAT/BTBT reverse-leakage closed forms (no PDE/mesh dependency) ----
# Added to independently validate tat/newton_solver_tat.py's I(Va) - this
# project had closed forms for every OTHER solver (built_in_potential/
# depletion_widths/shockley_current for the plain diode,
# breakdown_voltage_sze/ionization_integral for avalanche) but none for
# TAT/BTBT leakage until now. Both functions below take mat_p/mat_n
# (mat_n=None defaults to mat_p, the SAME backward-compat pattern used
# throughout this project - mesh.build_diode_grid's mat_n=None,
# built_in_potential/depletion_widths above), so ONE implementation covers
# both the homogeneous-Si drain-substrate example and the new SiGe/Si one.

def generation_current_srh(mat_p: Material, dev: Device, Va: float, mat_n: Material = None) -> float:
    """Plain (F=0, no field-enhancement/tunneling) SRH depletion-generation
    leakage current - the F->0 limit tat.tat.hurkx_tat_generation itself
    reduces to, used here as the baseline "before tunneling" closed form
    (the closed-form counterpart of the numeric no-tunneling newton_qf
    comparison sweep tat/main_tat.py already runs).

    At full depletion (n,p -> 0), SRH's generation rate is the textbook
    G = ni/(tau_n+tau_p) (verified against tat.tat.hurkx_tat_generation
    directly: at Gamma=0 (F=0) and n=p=0, tau_n_eff=tau_n, tau_p_eff=tau_p,
    n1=p1=ni (Et=Ei default), R=(0-ni^2)/(tau_p*ni+tau_n*ni)=-ni/(tau_n+tau_p),
    G_tat=-R=ni/(tau_n+tau_p) - matches exactly). Constant across each
    side's depletion width (no field dependence), so the generation
    CURRENT is just G*width per side, piecewise since ni/tau can differ
    across a heterojunction:
        J = q*[ni(mat_p)/(tau_n(mat_p)+tau_p(mat_p))*xp
              + ni(mat_n)/(tau_n(mat_n)+tau_p(mat_n))*xn]
    Sign convention matches this project's Va<0=reverse/leakage-negative
    convention (see tat/main_tat.py's own I(Va) plot) - this returns a
    NEGATIVE current for Va<0 (a generation, not recombination, current
    flowing from n to p), matching the numeric solver's own sign."""
    mat_n = mat_p if mat_n is None else mat_n
    xp, xn, _ = depletion_widths(mat_p, dev, Va, mat_n=mat_n)
    G_p = mat_p.ni / (mat_p.tau_n + mat_p.tau_p)
    G_n = mat_n.ni / (mat_n.tau_n + mat_n.tau_p)
    J = Q * (G_p * xp + G_n * xn)
    return -J * dev.area


def generation_current_tat(mat_p: Material, dev: Device, Va: float,
                            hurkx_model, kane_model, mat_n: Material = None,
                            n_probe: int = 2000) -> float:
    """Field-enhanced closed-form leakage current: Hurkx trap-assisted
    tunneling (field-enhanced SRH) plus Kane band-to-band tunneling,
    integrated over the depletion-approximation's TRIANGULAR field profile
    (same style as ionization_integral's own closed-form field model) -
    the closed-form counterpart of tat/newton_solver_tat.py's full
    newton_tat numeric solve.

    Deliberately reuses tat.tat.hurkx_gamma/btbt_generation directly (the
    SAME fitted models the numeric solver uses) rather than re-deriving
    them - this validates the SOLVER's own discretization/boundary-
    condition/mesh machinery against an independent field profile and
    quadrature, not tat.py's formulas themselves (which already have their
    own standalone tat.tat.sanity_probe() check).

    E_p(x) = E_max_p*(1-|x|/xp) for -xp<=x<=0, E_n(x) = E_max_n*(1-x/xn)
    for 0<=x<=xn (E_max_p=q*Na*xp/eps_p, E_max_n=q*Nd*xn/eps_n - D=eps*E
    continuous at x=0 by construction, D_max_p=D_max_n=q*Na*xp=q*Nd*xn from
    depletion_widths' own charge-balance relation).

    Integrand per side: [ni/(tau_n+tau_p)]*(1+Gamma(F(x))) (Hurkx-enhanced
    SRH generation - reduces to generation_current_srh's plain G at F=0
    since Gamma(0)=0) PLUS the separate, purely additive Kane term
    G_btbt(F(x)) (btbt_generation - no SRH trap involved at all, same
    additive-not-multiplicative treatment newton_solver_tat.py's own Reff
    uses). np.trapz quadrature over n_probe points per side (2000 default,
    matching ionization_integral's own resolution)."""
    from tat.tat import hurkx_gamma, btbt_generation
    mat_n = mat_p if mat_n is None else mat_n
    xp, xn, _ = depletion_widths(mat_p, dev, Va, mat_n=mat_n)

    E_max_p = Q * dev.Na * xp / mat_p.eps
    E_max_n = Q * dev.Nd * xn / mat_n.eps

    x_p = np.linspace(-xp, 0.0, n_probe)
    x_n = np.linspace(0.0, xn, n_probe)
    F_p = E_max_p * (1.0 - np.abs(x_p) / max(xp, 1e-300))
    F_n = E_max_n * (1.0 - x_n / max(xn, 1e-300))
    F_p = np.clip(F_p, 0.0, None)
    F_n = np.clip(F_n, 0.0, None)

    G0_p = mat_p.ni / (mat_p.tau_n + mat_p.tau_p)
    G0_n = mat_n.ni / (mat_n.tau_n + mat_n.tau_p)

    Gamma_p, _ = hurkx_gamma(F_p, mat_p.T, hurkx_model)
    Gamma_n, _ = hurkx_gamma(F_n, mat_n.T, hurkx_model)
    Gbtbt_p, _ = btbt_generation(F_p, kane_model)
    Gbtbt_n, _ = btbt_generation(F_n, kane_model)

    integrand_p = G0_p * (1.0 + Gamma_p) + Gbtbt_p
    integrand_n = G0_n * (1.0 + Gamma_n) + Gbtbt_n

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2.0 renamed trapz
    total_gen_p = float(_trapz(integrand_p, x_p))  # cm^-2 s^-1 (integrated over x, cm)
    total_gen_n = float(_trapz(integrand_n, x_n))

    J = Q * (total_gen_p + total_gen_n)
    return -J * dev.area
