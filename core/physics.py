"""Core drift-diffusion physics: Bernoulli function, nonlinear Poisson (Newton),
and Scharfetter-Gummel continuity solves.

Equations solved (steady state, 1D), all in physical units (cm, s, V, C):

Poisson:      eps * d2(psi)/dx2 = q*(n - p - Cdop)
              n = ni*exp((psi-phin)/Vt),  p = ni*exp((phip-psi)/Vt)

Continuity:   dJn/dx =  q*R          dJp/dx = -q*R
              R = SRH recombination rate (net recombination - generation)

Currents (Scharfetter-Gummel, exponentially fitted, exact for piecewise-linear psi):
  Jn_{i+1/2} = (q*Dn/h_i) * [ n_{i+1}*B(u_{i+1}-u_i) - n_i*B(u_i-u_{i+1}) ]
  Jp_{i+1/2} = (q*Dp/h_i) * [ p_i*B(u_{i+1}-u_i) - p_{i+1}*B(u_i-u_{i+1}) ]
  where u = psi/Vt, B(x) = x/(exp(x)-1) is the Bernoulli function.
  (This reduces to zero net current for the equilibrium Boltzmann profile,
  the defining sanity check for the SG discretization.)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core.params import Q, Material
from core.materials import MaterialField


def _as_field(mat, x) -> MaterialField:
    """Normalize a plain scalar Material into a MaterialField (constant
    arrays, delta_Ei=0) - a MaterialField passed in is returned unchanged.
    Every generalized function below does this as its first step, so a
    homojunction caller passing a plain Material gets bit-for-bit identical
    arithmetic to before (elementwise ops on a repeated-constant array
    equal the scalar op exactly), and a heterojunction caller can pass a
    MaterialField directly."""
    return mat if isinstance(mat, MaterialField) else MaterialField.uniform(mat, x)


def bernoulli(x: np.ndarray) -> np.ndarray:
    """Numerically safe Bernoulli function B(x) = x/(exp(x)-1)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = np.abs(x) < 1e-8
    out[small] = 1.0 - x[small] / 2.0  # Taylor series near 0

    big_pos = x > 40.0
    out[big_pos] = x[big_pos] * np.exp(-x[big_pos])  # avoid overflow, ->0

    big_neg = x < -40.0
    out[big_neg] = -x[big_neg]  # exp(x)->0, B(x) -> -x

    mid = ~(small | big_pos | big_neg)
    out[mid] = x[mid] / np.expm1(x[mid])
    return out


def bernoulli_deriv(x: np.ndarray) -> np.ndarray:
    """dB/dx for the Bernoulli function B(x) = x/(exp(x)-1), needed for the
    analytic Newton Jacobian of the Scharfetter-Gummel flux."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = np.abs(x) < 1e-8
    out[small] = -0.5 + x[small] / 6.0  # Taylor series of B'(x) near 0

    big_pos = x > 40.0
    out[big_pos] = (1.0 - x[big_pos]) * np.exp(-x[big_pos])  # B(x)~x*e^-x here

    big_neg = x < -40.0
    out[big_neg] = -1.0  # B(x) ~ -x here

    mid = ~(small | big_pos | big_neg)
    xm = x[mid]
    denom = np.expm1(xm)
    out[mid] = (denom - xm * np.exp(xm)) / denom ** 2
    return out


def equilibrium_bulk_potential(mat: Material, Cdop: float, ni: float = None,
                                delta_Ei: float = 0.0) -> float:
    """Exact charge-neutral bulk potential (psi with n=p=ni at psi=0) for a
    given net doping Cdop = Nd-Na, solving n0-p0=Cdop, n0*p0=ni^2 exactly
    (valid even when |Cdop| is not >> ni).

    n0 = (Cdop + sqrt(Cdop^2+4ni^2))/2 is the textbook solution, but for
    Cdop<0 (p-type) and |Cdop|>>ni it subtracts two nearly-equal large
    numbers: sqrt(Cdop^2+4ni^2) rounds to exactly |Cdop| in float64 once the
    4ni^2 correction drops below the ~1e-16 relative precision floor (around
    |Cdop|/ni ~ 1e8, i.e. doping several times 1e18 for silicon's ni~1e10) -
    n0 then evaluates to exactly 0.0, and log(0) diverges (caught this via a
    p-sub=5e18 MOS-cap sweep producing NaN/singular-matrix downstream). Fix:
    for Cdop<0, compute the MAJORITY carrier p0 first (well-conditioned,
    adding two positive numbers) and get the minority n0=ni^2/p0 from it -
    algebraically identical, just avoids the cancellation.

    ni: optional override for mat.ni (e.g. the LOCAL ni at a heterojunction
        contact, MaterialField.ni_arr[node] - mat is still needed for Vt).
    delta_Ei: heterojunction Boltzmann-relation offset at this node (see
        MaterialField's docstring) - 0.0 (default) reproduces today's exact
        homojunction formula bit-for-bit. From n = ni*exp((psi-phin+delta_Ei)/Vt)
        at equilibrium (phin=0): psi = Vt*ln(n0/ni) - delta_Ei.
    """
    ni = mat.ni if ni is None else ni
    if Cdop >= 0:
        n0 = (Cdop + np.sqrt(Cdop ** 2 + 4 * ni ** 2)) / 2.0
    else:
        p0 = (-Cdop + np.sqrt(Cdop ** 2 + 4 * ni ** 2)) / 2.0
        n0 = ni ** 2 / p0
    return mat.Vt * np.log(n0 / ni) - delta_Ei


def equilibrium_bulk_potential_arr(Vt: float, ni_arr: np.ndarray, Cdop: np.ndarray,
                                    delta_Ei_arr: np.ndarray = None) -> np.ndarray:
    """Vectorized sibling of equilibrium_bulk_potential, for a per-node
    ni_arr (MaterialField.ni_arr) - same n0/p0 cancellation-avoidance logic,
    applied elementwise via np.where instead of a scalar if/else."""
    ni_arr = np.asarray(ni_arr, dtype=float)
    Cdop = np.asarray(Cdop, dtype=float)
    p0 = (-Cdop + np.sqrt(Cdop ** 2 + 4 * ni_arr ** 2)) / 2.0
    n0_direct = (Cdop + np.sqrt(Cdop ** 2 + 4 * ni_arr ** 2)) / 2.0
    # np.where evaluates BOTH branches everywhere (unlike the scalar
    # function's if/else, which only ever evaluates the selected branch) -
    # p0 can itself round to exactly 0.0 at large positive Cdop (the same
    # cancellation this avoids for Cdop<0), so ni_arr**2/p0 would raise a
    # spurious divide-by-zero warning at nodes where that branch is
    # discarded anyway. Guard the denominator; the result at those nodes is
    # never selected by the final np.where below.
    p0_safe = np.where(p0 > 0, p0, 1.0)
    n0 = np.where(Cdop >= 0, n0_direct, ni_arr ** 2 / p0_safe)
    psi = Vt * np.log(n0 / ni_arr)
    if delta_Ei_arr is not None:
        psi = psi - delta_Ei_arr
    return psi


def _control_volumes(x: np.ndarray) -> np.ndarray:
    """Finite-volume cell width around each node (half the sum of neighboring
    spacings; half-width at the boundary nodes)."""
    h = np.diff(x)
    W = np.zeros_like(x)
    W[1:-1] = (h[:-1] + h[1:]) / 2.0
    W[0] = h[0] / 2.0
    W[-1] = h[-1] / 2.0
    return W


def solve_poisson(x, Cdop, mat: Material, phin, phip, psi_guess,
                   tol=1e-10, max_iter=100, damping_cap=None,
                   eps=None, ni=None, n_frozen=None, p_frozen=None,
                   interfaces=None, delta_Ei=None, quantum_shift=None):
    """Newton solve of the nonlinear Poisson equation for psi(x), given fixed
    quasi-Fermi levels phin(x), phip(x) (both zero at equilibrium).

    Dirichlet BC: psi[0] and psi[-1] are held fixed at psi_guess[0], psi_guess[-1].

    eps: permittivity, either the (scalar) mat.eps default, or an array of
        length len(x)-1 giving a distinct value per mesh EDGE - needed for a
        layered structure (e.g. oxide/semiconductor in a MOS capacitor)
        where permittivity is discontinuous at an interface. Using a
        per-edge (not per-node) value is what makes the finite-volume flux
        automatically enforce D-field continuity across that interface.
    ni: intrinsic concentration, either the (scalar) mat.ni default, or an
        array of length len(x) giving a distinct value per node - e.g. 0 in
        an oxide region (no mobile carriers there at all: n=p=0 identically,
        independent of psi, which is exactly what ni=0 in n=ni*exp(...)
        gives, including a correctly-zeroed Jacobian contribution).
    n_frozen, p_frozen: if given (array of length len(x), NaN where not
        frozen), overrides that carrier's density at those nodes to a fixed
        value instead of the Boltzmann relation - used for the
        quasi-small-signal "high-frequency C-V" trick, where the inversion
        (minority-carrier) charge is held fixed while the majority carrier
        and potential respond to a small gate-voltage perturbation. Use
        n_frozen for a p-type substrate (electrons are the minority/
        inversion carrier - pMOS-cap) or p_frozen for an n-type substrate
        (holes are the minority carrier - nMOS-cap); at most one is normally
        given at a time, but both are accepted for generality.
    interfaces: optional list of core.interfaces.Interface objects, each
        carrying a fixed areal charge density Qit_cm2 (C/cm^2) localized at
        one mesh node (e.g. the Si/SiO2 boundary) - added to that node's
        control-volume-equivalent charge density in the residual (same sign
        convention as Cdop: positive Qit_cm2 = positive/donor-like charge).
        None (default) or every interface's Qit_cm2=0.0 leaves the residual
        bit-identical to not passing this argument at all.
    delta_Ei: heterojunction Boltzmann-relation offset, either None (default,
        0 everywhere - today's exact behavior) or an array of length len(x)
        - see MaterialField's docstring. Added/subtracted only in the two
        n=/p= lines below; every Jacobian entry (dn/dpsi, dp/dpsi) is
        unaffected since delta_Ei doesn't depend on psi.
    quantum_shift: optional (Lambda_n, Lambda_p), each None or an array of
        length len(x), V: the quantum potential of quantum/ (Schrodinger
        on lines). Unlike delta_Ei it lowers BOTH densities,
        n *= exp(-Lambda_n/Vt), p *= exp(-Lambda_p/Vt), and is held fixed
        during this solve (the predictor of the Schrodinger-Poisson loop,
        quantum/coupling.py). None: exactly today's behavior.
    """
    N = len(x)
    h = np.diff(x)
    psi = psi_guess.copy()
    Vt = mat.Vt
    eps_edge = np.broadcast_to(mat.eps if eps is None else eps, N - 1)
    ni_arr = np.broadcast_to(mat.ni if ni is None else ni, N)
    delta_Ei_arr = np.zeros(N) if delta_Ei is None else np.broadcast_to(delta_Ei, N)
    lam_n = lam_p = 0.0
    if quantum_shift is not None:
        lam_n = 0.0 if quantum_shift[0] is None else np.broadcast_to(quantum_shift[0], N)
        lam_p = 0.0 if quantum_shift[1] is None else np.broadcast_to(quantum_shift[1], N)
    n_frozen_mask = np.zeros(N, dtype=bool) if n_frozen is None else ~np.isnan(n_frozen)
    p_frozen_mask = np.zeros(N, dtype=bool) if p_frozen is None else ~np.isnan(p_frozen)
    Qit_node = np.zeros(N)
    if interfaces:
        for iface in interfaces:
            Qit_node[iface.node_index] += iface.Qit_cm2

    hm = h[:-1]   # h_{i-1}, for interior i=1..N-2
    hp = h[1:]    # h_i
    cvol = (hm + hp) / 2.0
    lap_coeff_m = eps_edge[:-1] / hm / cvol
    lap_coeff_p = eps_edge[1:] / hp / cvol

    for it in range(max_iter):
        n = ni_arr * np.exp((psi - phin + delta_Ei_arr - lam_n) / Vt)
        p = ni_arr * np.exp((phip - psi - delta_Ei_arr - lam_p) / Vt)
        if n_frozen is not None:
            n = np.where(n_frozen_mask, n_frozen, n)
        if p_frozen is not None:
            p = np.where(p_frozen_mask, p_frozen, p)

        F = np.zeros(N)
        lower = np.zeros(N)
        diag = np.zeros(N)
        upper = np.zeros(N)

        dn_dpsi = np.where(n_frozen_mask, 0.0, n / Vt)    # frozen -> no psi-dependence
        dp_dpsi = np.where(p_frozen_mask, 0.0, -p / Vt)   # frozen -> no psi-dependence
        # Qit_node is already a CHARGE density (C/cm^2, q baked in - unlike
        # n/p/Cdop, which are number densities that still need the *Q below)
        # once divided by cvol (cm) it's already C/cm^3, so it's added
        # directly, not run through another *Q.
        F[1:-1] = (lap_coeff_p * (psi[2:] - psi[1:-1]) - lap_coeff_m * (psi[1:-1] - psi[:-2])) \
            - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1]) + Qit_node[1:-1] / cvol
        lower[1:-1] = lap_coeff_m
        upper[1:-1] = lap_coeff_p
        diag[1:-1] = -(lap_coeff_m + lap_coeff_p) - Q * (dn_dpsi[1:-1] - dp_dpsi[1:-1])

        # Dirichlet rows: F=0 here always forces the boundary's Newton update
        # to exactly 0 (psi[0]/psi[-1] stay pinned at their initial-guess
        # value) *only* if the row's own diagonal actually stays the pivot
        # spsolve's partial pivoting selects for that column. diag=1 is fine
        # numerically when nearby coefficients are O(1)-ish, but a short
        # Debye length (fine mesh -> huge lap_coeff) combined with a bad
        # early Newton iterate (huge exp-driven charge term) can make the
        # neighboring interior row's coupling entry into this column
        # (lower[1] resp. upper[-2], both pure lap_coeff, unaffected by the
        # charge blowup) outweigh a bare 1.0, so the solver pivots onto that
        # row instead and silently leaks a nonzero, wrong value into what
        # must be an exact zero. Scaling the Dirichlet diagonal to the local
        # lap_coeff magnitude guarantees it stays the largest entry in its
        # column, so pivoting can never dilute the boundary condition -
        # caught via the poly-gate MOS-cap case (mesh.build_mos_grid with
        # Cdop_gate), where the poly's short Debye length makes this a real
        # (not just theoretical) failure mode.
        diag[0] = max(1.0, lap_coeff_m[0] if len(lap_coeff_m) else 1.0)
        diag[-1] = max(1.0, lap_coeff_p[-1] if len(lap_coeff_p) else 1.0)

        J = sp.diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csc")
        delta = spla.spsolve(J, F)

        if damping_cap is not None:
            step = np.clip(delta, -damping_cap, damping_cap)
        else:
            step = delta

        psi = psi - step

        if np.max(np.abs(delta)) < tol:
            break

    n = ni_arr * np.exp((psi - phin + delta_Ei_arr - lam_n) / Vt)
    p = ni_arr * np.exp((phip - psi - delta_Ei_arr - lam_p) / Vt)
    if n_frozen is not None:
        n = np.where(n_frozen_mask, n_frozen, n)
    if p_frozen is not None:
        p = np.where(p_frozen_mask, p_frozen, p)
    return psi, n, p, it + 1


def srh_recombination(mat: Material, n, p):
    """mat may be a plain scalar Material (today's exact behavior) or a
    MaterialField (per-node ni_arr/tau_n_arr/tau_p_arr) - the mass-action
    term n*p-ni(x)^2 needs the LOCAL ni at each node; delta_Ei never enters
    here (it cancels identically in the n*p product at equilibrium, since
    n*p = ni(x)^2 regardless of how psi/phin/phip individually split - see
    the harmonic-snuggling-puddle plan's physics section)."""
    if isinstance(mat, MaterialField):
        ni, tau_n, tau_p = mat.ni_arr, mat.tau_n_arr, mat.tau_p_arr
    else:
        ni, tau_n, tau_p = mat.ni, mat.tau_n, mat.tau_p
    return (n * p - ni ** 2) / (tau_p * (n + ni) + tau_n * (p + ni))


def _sg_potential_n(psi, mf: MaterialField) -> np.ndarray:
    """The Scharfetter-Gummel Bernoulli-argument variable for ELECTRONS,
    generalized for a heterojunction. Plain SG (u=psi/Vt) is exact/zero-
    current-preserving at equilibrium because n_i=A*exp(u_i) with the SAME
    constant A at every node - true for a homojunction (A=ni, node-
    independent) but NOT at a heterojunction, where the equilibrium
    (phin=0) electron density n0_i = ni(x_i)*exp((psi_i+delta_Ei_i)/Vt) has
    a node-DEPENDENT prefactor ni(x_i) (it can jump ~100x right at a
    material step, e.g. SiGe's much larger ni vs Si's - caught via a real
    SiGe/Si sweep showing a ~4 orders of magnitude spurious current spike
    at exactly the interface edge with the plain u=psi/Vt formula). Folding
    ln(ni(x)) into u makes n0_i = 1*exp(u_n,i) with a node-INDEPENDENT
    prefactor again, restoring SG's exact zero-current identity - verified
    by hand: u_n,i = (psi_i+delta_Ei_i)/Vt + ln(ni_i) satisfies
    n0_i = exp(u_n,i) exactly. Reduces to psi/Vt plus a GLOBAL constant
    (ln(ni) is the same everywhere) for a homojunction - Bernoulli only
    depends on DIFFERENCES u_{i+1}-u_i, so a global constant offset is
    exactly cancelled, giving bit-identical behavior to before."""
    return (psi + mf.delta_Ei_arr) / mf.Vt + np.log(mf.ni_arr)


def _sg_potential_p(psi, mf: MaterialField) -> np.ndarray:
    """Same generalization as _sg_potential_n, for HOLES: equilibrium
    (phip=0) p0_i = ni(x_i)*exp(-(psi_i+delta_Ei_i)/Vt) = exp(-u_p,i) with
    u_p,i = (psi_i+delta_Ei_i)/Vt - ln(ni_i) (note the SIGN of the ln(ni)
    term flips relative to electrons - p0's prefactor divides by ni(x_i)
    instead of multiplying by it). Also reduces to psi/Vt (mod an
    inconsequential global constant) for a homojunction."""
    return (psi + mf.delta_Ei_arr) / mf.Vt - np.log(mf.ni_arr)


def solve_continuity_n(x, psi, mat: Material, R, n_bc0, n_bcL):
    """Linear tridiagonal solve for electron density n(x) given psi(x) and a
    (lagged) recombination source R(x), with Dirichlet BC at both contacts.
    mat may be a plain scalar Material or a MaterialField (per-edge Dn) -
    see _sg_potential_n for why a heterojunction needs u generalized beyond
    plain psi/Vt."""
    N = len(x)
    h = np.diff(x)
    mf = _as_field(mat, x)
    Vt = mf.Vt
    u = _sg_potential_n(psi, mf)
    W = _control_volumes(x)

    Bp = bernoulli(u[1:] - u[:-1])   # B(u_{i+1} - u_i), for edge i (i -> i+1)
    Bm = bernoulli(u[:-1] - u[1:])   # B(u_i - u_{i+1})

    coef = Q * mf.Dn_edge / h  # per-edge conductance-like coefficient, edge i between node i,i+1

    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)

    diag[0] = 1.0
    rhs[0] = n_bc0
    diag[-1] = 1.0
    rhs[-1] = n_bcL

    # Jn_{i+1/2} = coef_i * (Bp_i * n_{i+1} - Bm_i * n_i); interior i=1..N-2
    c_m = coef[:-1]   # edge (i-1,i), for i=1..N-2
    c_p = coef[1:]    # edge (i,i+1)
    lower[1:-1] = c_m * Bm[:-1]
    diag[1:-1] = -c_p * Bm[1:] - c_m * Bp[:-1]
    upper[1:-1] = c_p * Bp[1:]
    rhs[1:-1] = Q * R[1:-1] * W[1:-1]

    A = sp.diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csc")
    n = spla.spsolve(A, rhs)
    return n


def solve_continuity_p(x, psi, mat: Material, R, p_bc0, p_bcL):
    """Linear tridiagonal solve for hole density p(x). mat may be a plain
    scalar Material or a MaterialField (per-edge Dp) - see _sg_potential_p
    for why a heterojunction needs u generalized beyond plain psi/Vt."""
    N = len(x)
    h = np.diff(x)
    mf = _as_field(mat, x)
    Vt = mf.Vt
    u = _sg_potential_p(psi, mf)
    W = _control_volumes(x)

    Bp = bernoulli(u[1:] - u[:-1])
    Bm = bernoulli(u[:-1] - u[1:])

    coef = Q * mf.Dp_edge / h

    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)

    diag[0] = 1.0
    rhs[0] = p_bc0
    diag[-1] = 1.0
    rhs[-1] = p_bcL

    # Jp_{i+1/2} = coef_i * (Bp_i * p_i - Bm_i * p_{i+1}); interior i=1..N-2
    c_m = coef[:-1]
    c_p = coef[1:]
    lower[1:-1] = -c_m * Bp[:-1]
    diag[1:-1] = c_p * Bp[1:] + c_m * Bm[:-1]
    upper[1:-1] = -c_p * Bm[1:]
    rhs[1:-1] = -Q * R[1:-1] * W[1:-1]

    A = sp.diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csc")
    p = spla.spsolve(A, rhs)
    return p


def edge_currents(x, psi, n, p, mat: Material):
    """Electron, hole, and total current density (A/cm^2) at each cell edge.
    mat may be a plain scalar Material or a MaterialField (per-edge Dn/Dp) -
    see _sg_potential_n/_sg_potential_p for why electrons and holes need
    their OWN Bernoulli-argument arrays at a heterojunction (they agree,
    reducing to a single shared psi/Vt-based Bp/Bm, for a homojunction)."""
    h = np.diff(x)
    mf = _as_field(mat, x)
    u_n = _sg_potential_n(psi, mf)
    u_p = _sg_potential_p(psi, mf)
    Bp_n = bernoulli(u_n[1:] - u_n[:-1])
    Bm_n = bernoulli(u_n[:-1] - u_n[1:])
    Bp_p = bernoulli(u_p[1:] - u_p[:-1])
    Bm_p = bernoulli(u_p[:-1] - u_p[1:])

    Jn = (Q * mf.Dn_edge / h) * (n[1:] * Bp_n - n[:-1] * Bm_n)
    Jp = (Q * mf.Dp_edge / h) * (p[:-1] * Bp_p - p[1:] * Bm_p)
    return Jn, Jp, Jn + Jp
