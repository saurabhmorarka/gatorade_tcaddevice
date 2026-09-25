"""Solver-agnostic damped Newton iteration with a ROW-NORMALIZED merit
function, plus helpers for reporting a terminal current only from edges
where it is numerically resolvable. Used by every 1D quasi-Fermi Newton
solver (core/newton_solver_qf.py, tat/newton_solver_tat.py,
avalanche/newton_solver_avalanche.py) - same category as
core/jacobian_scaling.py and core/bank_rose_damping.py: generic numerics,
no physics, so avalanche/'s standalone rule (ARCHITECTURE.md) is unaffected.

WHY (DEVELOPMENT_LOG.md session 23): the QF solvers used to test
convergence, and run their backtracking line search, on the max-norm of the
RAW (globally scaled) residual. In a heavily doped region the majority-
carrier current is a huge conductance q*mu*p/h times a quasi-Fermi
difference that is below double-precision resolution of phi itself (for a
reverse-bias leakage current, dphi ~ J*h/(q*mu*p) ~ 1e-17 V at p=1e19,
h=0.1nm), so those rows carry a roundoff FLOOR in the raw residual - ~1e2 at
1e19 doping, growing 10x per decade of doping. Measured directly: at 1e19,
Va=-0.5V Newton stalled with |F|_inf=24 in a majority-hole row whose own
roundoff floor was 94 - already at machine precision, yet far above
f_tol=1e-9. Worse, the max-norm line search could then make no progress on
anything else: rows in the depletion region that were genuinely unconverged
(|F|~3e3, floor 1e-11) sat UNDER the heavy side's noise maximum, so no step
length could reduce the max-norm and Newton froze on a wrong answer. Every
reverse-bias point of the no-avalanche solver at >=1e19 was affected.

THE FIX (standard production-simulator practice - convergence judged on
per-row-scaled residuals / update size, not a raw residual): divide each
residual row by its own diagonal Jacobian entry (see row_scales). F_i/d_i is
then the Newton correction row i alone would ask of its own unknown, in the
unknowns' units (volts for psi/phin/phip), so every row is judged on the same scale and a
heavily doped row's roundoff floor becomes ~eps*|phi| (~1e-15 V) instead of
~1e2. The line search minimizes 0.5*||D^-1 F||_2^2 with D frozen for the
step - the exact Newton direction is a guaranteed descent direction for any
fixed row weighting (grad = J^T D^-2 F, so delta.grad = -||D^-1 F||^2 < 0),
which is NOT true of the max-norm the old loop used.
"""
import numpy as np
import scipy.sparse as sp

from core.jacobian_scaling import equilibrated_spsolve

EPS = np.finfo(float).eps

# Default convergence tolerance on max|F_i/d_i| - volts for the potential
# unknowns. 1e-10 V is ~4e-9 thermal voltages; the achievable floor is
# ~eps*|phi| ~ 1e-15 V, so this is tight but always reachable.
ROW_TOL = 1e-10
# A row whose residual is within this many eps of the magnitude of the
# terms it sums (estimated as (|J| @ |U|)_i) is at its own roundoff floor.
NOISE_FACTOR = 8.0


def residual_merit(F, J, U, d):
    """max_i (|F_i| - noise_i)^+ / d_i: each row's residual in units of its
    own unknown, after discounting that row's floating-point noise floor
    noise_i = NOISE_FACTOR*eps*(|J| @ |U|)_i (roughly the size of the terms
    the row sums, e.g. conductance*|phi| for a flux). A row can never show a
    residual below its own roundoff; without this discount a row with a
    small diagonal but large off-diagonal terms (an electron row on a p+ side
    during avalanche, say) sits permanently just above tolerance and the
    solve is wrongly declared unconverged (DEVELOPMENT_LOG.md session 23)."""
    Ua = np.abs(U)
    noise = NOISE_FACTOR * EPS * (abs(J) @ Ua)
    return float(np.max(np.maximum(np.abs(F) - noise, 0.0) / d))


def row_scales(J, min_diag_fraction=1e-6):
    """Per-row residual scale: the row's DIAGONAL Jacobian entry |J_ii|, so
    F_i/J_ii is the Newton correction row i asks of its own unknown (volts
    for psi/phin/phip) - classic Jacobi scaling. Falls back to the row's
    largest entry only where the diagonal is degenerate (< min_diag_fraction
    of it), e.g. core/arclength.py's constraint row, whose diagonal t_V goes
    to ~0 exactly where the I-V curve turns vertical.

    Why not the row maximum everywhere: in a hole row at an n+ node with
    Kane BTBT, the largest entry is dG_btbt/dpsi, not the row's own phip
    coupling (p ~ 1 cm^-3 there), so F/rowmax reported ~1e-4 V while phip
    actually needed to move by volts - and the line search, minimizing that
    misleading merit, refused the steps that converge (DEVELOPMENT_LOG.md
    session 23). Measured: every physical row has |J_ii| >= ~1e-3 of its row
    maximum, so the diagonal is always a meaningful scale there."""
    Jr = J.tocsr()
    row_max = np.maximum(abs(Jr).max(axis=1).toarray().ravel(), 1e-300)
    diag = np.abs(Jr.diagonal())
    return np.where(diag >= min_diag_fraction * row_max, diag, row_max)


def uniform_step_clip(delta, N, max_psi=1.0, max_qf=5.0):
    """Shorten the WHOLE Newton step by one scalar so that no psi entry
    moves more than max_psi and no phin/phip entry more than max_qf (volts).
    A uniform rescale keeps the exact Newton direction (a descent direction
    for the line-search merit); clipping components independently does not
    (see avalanche/newton_solver_avalanche.py's history of that bug). Assumes
    the [psi(N), phin(N), phip(N), ...extra] layout; extra trailing unknowns
    (e.g. an arc-length Va) are limited like psi."""
    m_psi = np.max(np.abs(delta[:N])) if N else 0.0
    m_qf = np.max(np.abs(delta[N:3 * N])) if N else 0.0
    m_extra = np.max(np.abs(delta[3 * N:])) if len(delta) > 3 * N else 0.0
    scale = max(m_psi / max_psi, m_qf / max_qf, m_extra / max_psi, 1.0)
    return delta / scale if scale > 1.0 else delta


def componentwise_step_clip(delta, N, max_psi=1.0, Vt=0.025852):
    """Per-entry limiter: psi (and any extra trailing unknowns) clipped to
    +-max_psi; phin/phip LOG-DAMPED, d -> sign(d)*Vt*ln(1 + |d|/Vt) - the
    classic device-simulator potential damping: updates well below Vt pass
    almost unchanged, a 20 V request (a barely-constrained minority
    quasi-Fermi level, whose density responds as exp(d/Vt)) becomes ~0.17 V.
    NOT a descent direction in general - only ever used by damped_newton as
    a first trial step, accepted only if it actually lowers the merit."""
    d = delta.copy()
    d[:N] = np.clip(d[:N], -max_psi, max_psi)
    q = d[N:3 * N]
    d[N:3 * N] = np.sign(q) * Vt * np.log1p(np.abs(q) / Vt)
    d[3 * N:] = np.clip(d[3 * N:], -max_psi, max_psi)
    return d


def damped_newton(U0, residual_and_jacobian, residual_only, step_clip=None,
                  tol=ROW_TOL, maxiter=50, max_halvings=30, verbose=False, label="Newton",
                  trial_clip=None):
    """Damped Newton on F(U)=0 with a row-normalized merit (see module
    docstring). Returns (U, merit, iterations, converged), merit =
    residual_merit at the returned U (converged: merit < tol).

    residual_and_jacobian(U) -> (F, J sparse); residual_only(U) -> F.
    step_clip(delta) -> delta: scalar step shortening applied before the
        backtracking line search (e.g. uniform_step_clip) - keeps the Newton
        direction, so backtracking is guaranteed to find a decrease.
    trial_clip(delta) -> delta: optional per-component limiter (e.g.
        componentwise_step_clip). Whenever step_clip actually had to shorten
        the step, the full trial_clip step is tried FIRST and accepted if it
        lowers the merit; otherwise the usual backtracking on the step_clip
        direction follows. Why both: when a single barely-constrained entry
        (a minority quasi-Fermi level where that carrier is ~1 cm^-3) asks for
        a huge change, the uniform rescale shrinks every other entry with it
        and Newton crawls (seen: a Kane-BTBT solve at Va=0 asking for a 20 V
        hole quasi-Fermi move at one n+ node, stuck at 1e-4 steps for 50
        iterations); limiting just that entry fixes it. But a
        per-component-limited step is not a descent direction in general
        (the avalanche solver's old wrong-branch bug), so it is only ever
        accepted when it verifiably helps. When no clipping is needed at all
        this is plain Newton."""
    U = U0.copy()
    F, J = residual_and_jacobian(U)
    d = row_scales(J)
    r = F / d
    for it in range(maxiter + 1):
        merit = residual_merit(F, J, U, d)
        if merit < tol:
            return U, merit, it, True
        if it == maxiter:
            break
        raw = equilibrated_spsolve(J, -F)
        delta = step_clip(raw) if step_clip is not None else raw
        f0 = 0.5 * float(r @ r)
        accepted = False
        if trial_clip is not None and delta is not raw:
            U_try = U + trial_clip(raw)
            r_try = residual_only(U_try) / d
            f_try = 0.5 * float(r_try @ r_try)
            if np.isfinite(f_try) and f_try < f0:
                accepted, step = True, -1.0   # step=-1 marks the per-component trial in verbose output
        if not accepted:
            step = 1.0
            for _ in range(max_halvings):
                U_try = U + step * delta
                r_try = residual_only(U_try) / d
                f_try = 0.5 * float(r_try @ r_try)
                if np.isfinite(f_try) and f_try <= f0 * (1.0 - 1e-4 * step):
                    break
                step *= 0.5
            else:
                if verbose:
                    print(f"  {label} it {it + 1}: line search found no decrease (merit={merit:.3e})")
                return U, merit, it, False
        U = U_try
        F, J = residual_and_jacobian(U)
        d = row_scales(J)
        r = F / d
        if verbose:
            print(f"  {label} it {it + 1}: merit={residual_merit(F, J, U, d):.3e}  step={step:.3g}")
    return U, residual_merit(F, J, U, d), maxiter, False


def edge_current_noise(phin, phip, n, p, h, q_mu_n, q_mu_p):
    """Per-edge roundoff uncertainty (A/cm^2) of a plain-gradient edge
    current q*mu*c_avg*dphi/h: the edge conductance times the resolution of
    the stored potentials (a few eps*|phi|). Where a carrier is heavily
    majority this can exceed the true (leakage) current by orders of
    magnitude - see the module docstring. q_mu_n/q_mu_p: Q*mobility, scalar
    or per-edge arrays."""
    gn = q_mu_n * 0.5 * (n[:-1] + n[1:]) / h
    gp = q_mu_p * 0.5 * (p[:-1] + p[1:]) / h
    phin_mag = np.maximum(np.abs(phin[:-1]), np.abs(phin[1:]))
    phip_mag = np.maximum(np.abs(phip[:-1]), np.abs(phip[1:]))
    return 4.0 * EPS * (gn * phin_mag + gp * phip_mag)


def resolved_current(Jtot, noise, rel_resolution=1e-4):
    """Terminal current and a self-consistency figure computed only over
    edges where the current is numerically resolved.

    In 1D steady state the total current Jn+Jp is the same on every edge,
    so it can be read off anywhere - read it where its roundoff is smallest
    (one of the depletion-region/light-side edges). Self-consistency
    (J_std/|J|) is then taken over the edges whose noise is below
    rel_resolution*|J|; edges in a heavily doped majority region, where
    the current is not representable in double precision at all, are
    excluded rather than being allowed to report a fake 10-100%
    non-conservation. Returns (J_rep, J_std, k_best, n_resolved_edges)."""
    k = int(np.argmin(noise))
    J_rep = float(Jtot[k])
    ok = noise <= rel_resolution * max(abs(J_rep), 1e-300)
    ok[k] = True
    J_std = float(np.std(Jtot[ok]))
    return J_rep, J_std, k, int(ok.sum())
