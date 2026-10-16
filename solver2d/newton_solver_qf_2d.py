"""2D fully-coupled Newton solve using quasi-Fermi potentials (psi, phin,
phip) over a point-cloud/box-FV mesh (mesh2d/mesh2d.py) - the point-cloud
generalization of core/newton_solver_qf.py. Lives in solver2d/ (not
mesh2d/) since this is solver/physics code, not mesh geometry - mesh2d/
builds the Mesh2D object this module consumes, the same way core/mesh.py
builds the grid core/newton_solver_qf.py consumes for 1D. See that
module's docstring for the full physics derivation (plain-gradient
current, no Scharfetter-Gummel) and why this formulation was chosen: it is
the one that resolved the 1D extreme-doping convergence failure, and per
the project's mesh-robustness principle, a 2D solver especially cannot
depend on the user hand-resolving the Debye length everywhere the way
fine 1D meshing could.

The only real generalization is the discretization geometry: 1D's two
fixed neighbors (i-1, i+1) become an arbitrary-degree graph edge list
(mesh2d/fvgeometry.py's Voronoi-box edges). This module loops (vectorized,
no per-node Python loop) over that edge list and lets scipy.sparse's COO
format sum duplicate (row, col) entries automatically on conversion to
CSC - the same accumulation 1D achieves implicitly via its two explicitly
named e_lo/e_hi edges, just generalized to however many edges a point
happens to have.

Boundary conditions need no special-casing beyond that: every point gets
the SAME generic Poisson/continuity row, contact points included, and is
then overwritten with a Dirichlet row afterward (contact points are
excluded from the generic row assembly via a boolean mask - see
`active_i`/`active_j` below - so the Dirichlet overwrite is a clean replacement,
not a sum). A `symmetry` or `free_surface` point never gets a special
row at all: its control volume simply has fewer incident edges than an
interior point's, which is the entire implementation of "default
insulating" (see mesh2d/boundary.py and mesh2d/fvgeometry.py's module
docstrings).

Materials: this module takes a plain scalar Material (today's homogeneous-
silicon first example) rather than a MaterialField - MaterialField's
eps_edge/mu_n_edge/etc. arrays are sized len(x)-1, a 1D-topology-specific
convention that doesn't generalize to an arbitrary edge count without
being rebuilt anyway, so heterojunction-in-2D support is left as future
work rather than force-fitting MaterialField here.
"""
import warnings
import weakref

import numpy as np
import scipy.sparse as sp

from core.params import Q, Material
from core.jacobian_scaling import equilibrated_spsolve
from core.linsolve import LinearSolver
from core.physics import equilibrium_bulk_potential_arr
from core.solver import contact_values
from core.bank_rose_damping import bank_rose_solve

MAX_QF_STEP = 5.0
MAX_PSI_STEP = 1.0
PSI_UPDATE_TOL = 1e-9   # V - converged once the potential update is this small...
RES_ACCEPT = 1e-4       # ...and |F|_inf is below this (see _run_newton)
ETA_MAX = 1e-4          # loosest inexact-Newton linear-solve tolerance (see _run_newton);
                        # measured on the 2D MOSFET: 1e-2 doubles the Newton iterations,
                        # 1e-6 spends it back in Krylov iterations/refactorizations
ETA_MAX_SCHOTTKY = 1e-7  # the cap after a damped step, with sub-grid Schottky contacts (back to ETA_MAX
                         # after a full step): a bias step puts a large residual on the contact nodes'
                         # sink rows, and a solve relative to that leaves the rest too inexact -
                         # quasi-1D TiN slabs (n+ 7e20, p+ 5e20, ~1e7 A/cm^2) stall at 1e-6 and converge
                         # at 1e-7 (with the linear solver's coarse correction, DEVELOPMENT_LOG 65; 1e-9
                         # before it). Planar 3D I-V: 1e-9 / 1e-7 / no cap 44 / 40 / 37 s, same currents
# V - run the in-Newton quantum corrector once the update is below this. Full
# planar device, Vds = 1 V gate walk: 1e-3 -> 122 iterations, 1e-2 -> 110,
# 3e-2 -> 99 (classical 69); Lambda contracts fast (each corrector ~100x
# smaller than the last), so an early first corrector costs nothing.
QUANTUM_TRIGGER = 1e-2
QUAD_PREDICT_MAX = 1e-3  # V - only trust the quadratic-convergence prediction below this update
RES_PREDICT_MAX = 1e-1   # |F|_inf sanity bound for the prediction exit (see _run_newton)
UPD_PSI_TOL = 1e-6       # V - update-based convergence (Sentaurus-style, see _run_newton): the Newton
UPD_DENS_TOL = 1e-4      # correction's largest psi move and largest relative density change
UPD_DENS_MEAN = 1e-5     # (|dn|/(n + UPD_DENS_REF), p likewise), and the node mean of the latter
UPD_DENS_REF = 1e10      # (Sentaurus: mean < 10^-Digits, Digits 5; ErrRef 1e8-1e10 cm^-3)
QF_DENSITY_REF = 1.0     # cm^-3 - quasi-Fermi updates are weighted by n/(n+ref) in the prediction exit

# One core.linsolve.LinearSolver per mesh per process, so a factorization
# (and the sparsity-pattern analysis/ordering) is reused across Newton
# iterations AND across the bias points of a sweep without any driver
# having to manage solver state. Keyed by id(mesh), with a weak reference
# to tell a recycled id from the same mesh.
_LINEAR_SOLVERS = {}
_ASSEMBLY_PLANS = {}


def default_linear_solver(mesh):
    key = id(mesh)
    entry = _LINEAR_SOLVERS.get(key)
    if entry is None or entry[0]() is not mesh:
        # block_size=3: U = [psi; phin; phip], field-major (point-block ILU
        # for the iterative 3D backends; ignored by the direct ones)
        entry = (weakref.ref(mesh), LinearSolver(block_size=3))
        _LINEAR_SOLVERS[key] = entry
    return entry[1]


def _uniform_clip(delta, N):
    """Same uniform-RESCALE convention as
    avalanche/newton_solver_avalanche.py's own `_clip` helper (see that
    module's long docstring for the full rationale): cap the step's
    WORST-OFFENDING component (psi against MAX_PSI_STEP, phin/phip against
    MAX_QF_STEP) by rescaling the ENTIRE delta vector by one global scalar,
    not by clipping each component independently - only a uniform rescale
    preserves J(U)*delta=-F(U)'s guarantee that delta is a descent
    direction for ||F||^2.

    ONLY used by the optional damping="bank_rose" path below (see
    newton_solve_2d's docstring for why plain backtracking stays the
    default and does its own, simpler per-component clip inline in
    _run_newton) - kept here rather than as a lambda so
    _run_newton's bank_rose branch and any future caller share one
    implementation."""
    max_psi = np.max(np.abs(delta[:N])) if N else 0.0
    max_qf = np.max(np.abs(delta[N:3 * N])) if 2 * N else 0.0
    scale = max(max_psi / MAX_PSI_STEP, max_qf / MAX_QF_STEP, 1.0)
    return delta / scale if scale > 1.0 else delta

# Regularization field scale (V/cm) for the Caughey-Thomas |E| -> smooth
# even function of E: E_reg = sqrt(E^2 + E_SMOOTH^2). This is what makes
# mu(E) (and hence d(mu)/dE) a well-defined, smooth function of E right at
# E=0 - the literal |E|/E^beta forms have a kink or (for beta<1) a
# diverging derivative there, which would poison the Jacobian at exactly
# the equilibrium/near-flatband bias points this project starts every
# Newton continuation from. 1 V/cm is ~4-7 orders of magnitude below any
# physically meaningful field in this device (channel fields are
# ~1e3-1e6 V/cm), so it is a pure numerical regularization, not a physics
# approximation.
E_SMOOTH = 1.0


def unpack_qf(U, N):
    return U[:N], U[N:2 * N], U[2 * N:3 * N]


def mobility_doping(N_abs, mu_max, mu_min, N_ref, alpha):
    """Caughey-Thomas DOPING-dependence low-field mobility (see
    core/params.py's Material.mu_n_max/etc. docstring for the formula's
    citation/parameter source):

        mu(N) = mu_min + (mu_max - mu_min) / (1 + (N/N_ref)^alpha)

    N_abs is the LOCAL TOTAL (ionized) doping magnitude, cm^-3 - the usual
    convention for this model (not net doping, since even a compensated
    region's mobility is degraded by both dopant species' scattering).
    Depends ONLY on the fixed mesh doping array, never on the solved
    psi/phin/phip, so (unlike the old field-dependent mobility_field())
    this needs no Jacobian term of its own - it is exactly as simple as a
    scalar constant mobility, just spatially varying."""
    return mu_min + (mu_max - mu_min) / (1.0 + (N_abs / N_ref) ** alpha)


def mobility_masetti(N_abs, coef):
    """Masetti DOPING-dependence low-field mobility (see core/params.py's
    Material.doping_mobility for the formula and source); coef =
    (mu_min1, mu_min2, mu1, Pc, Cr, Cs, alpha, beta, mu_max). Like
    mobility_doping, a fixed function of the mesh doping: no Jacobian
    term. N is floored at 1 cm^-3 (undoped: exp(-Pc/N) -> 0, mu -> mu_max)."""
    mu_min1, mu_min2, mu1, Pc, Cr, Cs, alpha, beta, mu_max = coef
    N = np.maximum(N_abs, 1.0)
    return (mu_min1 * np.exp(-Pc / N) + (mu_max - mu_min2) / (1.0 + (N / Cr) ** alpha)
            - mu1 / (1.0 + (Cs / N) ** beta))


def mobility_field(E, mu0, vsat, beta):
    """Caughey-Thomas velocity-saturation mobility and its derivative
    w.r.t. the (signed) driving field E:

        mu(E) = mu0 / (1 + (mu0*|E|/vsat)^beta)^(1/beta)

    |E| is replaced by the smooth even regularization sqrt(E^2+E_SMOOTH^2)
    (see E_SMOOTH's docstring above) so mu and dmu_dE are both finite,
    smooth functions of E everywhere including E=0 - required since E=0
    is exactly where every Newton continuation in this project starts
    (equilibrium/flatband). Returns (mu, dmu_dE), both arrays shaped like E.

    Derived by direct differentiation of mu(E) treating x = mu0*Ereg/vsat:
    mu = mu0*(1+x^beta)^(-1/beta), so
    dmu/dEreg = -mu0*(mu0/vsat)*x^(beta-1)*(1+x^beta)^(-1/beta-1),
    and dmu/dE = dmu/dEreg * dEreg/dE = dmu/dEreg * (E/Ereg).
    Note beta_n=2 and beta_p=1 (this project's defaults, core/params.py)
    both give x^(beta-1) with a non-negative integer exponent (x^1 or x^0),
    so this derivative is itself finite and smooth at x=0 (E=0) even
    without the E_SMOOTH regularization - E_SMOOTH is still kept for
    robustness against any future non-default beta<1 choice.
    """
    Ereg = np.sqrt(E * E + E_SMOOTH ** 2)
    x = mu0 * Ereg / vsat
    # dmu/dEreg = -mu * (mu0/vsat) * x^(beta-1) / base; beta = 2 and 1 (the
    # electron and hole defaults) spelled out - the same values without
    # the general pow (DEVELOPMENT_LOG 59)
    if beta == 2.0:
        base = 1.0 + x * x
        mu = mu0 / np.sqrt(base)
        dmu_dEreg = -mu * (mu0 / vsat) * x / base
    elif beta == 1.0:
        base = 1.0 + x
        mu = mu0 / base
        dmu_dEreg = -mu * (mu0 / vsat) / base
    else:
        base = 1.0 + x ** beta
        mu = mu0 * base ** (-1.0 / beta)
        dmu_dEreg = -mu * (mu0 / vsat) * x ** (beta - 1.0) / base
    dmu_dE = dmu_dEreg * (E / Ereg)
    return mu, dmu_dE


def _densities(psi, phin, phip, ni_arr, Vt, dEi=0.0):
    """dEi: per-node heterojunction band shift (mesh.dEi_arr, see
    _mesh_dEi) - 0 for a single-material device."""
    n = ni_arr * np.exp((psi + dEi - phin) / Vt)
    p = ni_arr * np.exp((phip - psi - dEi) / Vt)
    return n, p


def _mesh_dEi(mesh):
    return mesh.dEi_arr if getattr(mesh, "dEi_arr", None) is not None else 0.0


class QuantumShift:
    """The quantum potentials Lambda_n, Lambda_p (V, per node) of the
    Schrodinger-Poisson channel charge (quantum/, plans/quantum_plan.md):
    n = ni exp((psi + dEi - phin - Lambda_n)/Vt), p = ni exp((phip - psi -
    dEi - Lambda_p)/Vt), held fixed during a Newton solve.

    They are folded exactly into the two per-node band quantities the
    solver already has: ni' = ni exp(-(Lambda_n + Lambda_p)/2Vt) and
    dEi' = dEi - (Lambda_n - Lambda_p)/2. So the residual, the Jacobian,
    SRH (n p = ni'^2 in equilibrium: the confinement-widened gap) and the
    Scharfetter-Gummel heterojunction shifts (zero equilibrium current
    across a Lambda step) all follow with no other change. Passed
    explicitly (not swapped into the mesh), so the mesh-keyed caches (ILU,
    Jacobian index, assembly plans) survive a Lambda update."""

    def __init__(self, mesh, mat, lam_n=None, lam_p=None):
        N = len(mesh.points)
        self.lam_n = np.zeros(N) if lam_n is None else np.asarray(lam_n, float)
        self.lam_p = np.zeros(N) if lam_p is None else np.asarray(lam_p, float)
        ni, _ = _mesh_ni_edge_g(mesh, mat)
        dEi = mesh.dEi_arr if getattr(mesh, "dEi_arr", None) is not None else np.zeros(N)
        self.ni = ni * np.exp(-(self.lam_n + self.lam_p) / (2.0 * mat.Vt))
        self.dEi = dEi - 0.5 * (self.lam_n - self.lam_p)
        self.sg = _sg_shift_arrays(self.ni, self.dEi, mesh.edges[:, 0], mesh.edges[:, 1], mat.Vt)


def _bands(mesh, mat, qs):
    """(ni, dEi) per node: the mesh's own, or with the quantum shift."""
    if qs is None:
        return _mesh_ni_edge_g(mesh, mat)[0], _mesh_dEi(mesh)
    return qs.ni, qs.dEi


_MESH_CACHE = {}


def _mesh_cached(mesh, key, build):
    """build() once per (mesh, key): mesh-only quantities (edge lengths,
    nodal mobility, heterojunction shifts) that every residual and
    Jacobian evaluation used to recompute - ~15% of a 3D Newton iteration
    (DEVELOPMENT_LOG 59). Mesh arrays are never modified after the mesh is
    built; anything that also depends on the material carries the
    material's parameters in `key`."""
    entry = _MESH_CACHE.get(id(mesh))
    if entry is None or entry[0]() is not mesh:
        try:
            entry = (weakref.ref(mesh), {})
        except TypeError:                 # a stand-in mesh (e.g. a SimpleNamespace in a test): no cache
            return build()
        _MESH_CACHE[id(mesh)] = entry
    store = entry[1]
    if key not in store:
        store[key] = build()
    return store[key]


def _edge_len(mesh):
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    return _mesh_cached(mesh, "edge_len", lambda: np.linalg.norm(mesh.points[jj] - mesh.points[ii], axis=1))


def _edge_divergence(ii, jj, flux, N):
    """sum over edges of +flux at node i and -flux at node j (np.bincount:
    several times faster than np.add.at)."""
    return np.bincount(ii, flux, minlength=N) - np.bincount(jj, flux, minlength=N)


def _sg_shifts(mesh, ii, jj, Vt):
    """Per-edge additions to the Scharfetter-Gummel argument d =
    (psi_j - psi_i)/Vt at a heterojunction (the 2D form of core/physics.py's
    _sg_potential_n/_sg_potential_p): with u_n = (psi+dEi)/Vt + ln(ni) and
    u_p = (psi+dEi)/Vt - ln(ni), equilibrium n = exp(u_n)*const and
    p = exp(-u_p)*const, so SG on d_n = u_n,j - u_n,i (d_p likewise) carries
    exactly zero current in equilibrium across a material step. The plain
    d leaves a spurious current wherever ni or dEi jumps. Both shifts are
    0 on a single-material mesh, and on edges touching an insulator node
    (those edges carry no carrier flux anyway)."""
    dEi = getattr(mesh, "dEi_arr", None)
    if dEi is None or mesh.ni_arr is None:
        return 0.0, 0.0
    return _mesh_cached(mesh, ("sg_shifts", Vt), lambda: _sg_shifts_build(mesh, ii, jj, Vt))


def _sg_shifts_build(mesh, ii, jj, Vt):
    return _sg_shift_arrays(mesh.ni_arr, mesh.dEi_arr, ii, jj, Vt)


def _sg_shift_arrays(ni, dEi, ii, jj, Vt):
    ok = (ni[ii] > 0) & (ni[jj] > 0)
    lr = np.where(ok, np.log(np.where(ok, ni[jj], 1.0) / np.where(ok, ni[ii], 1.0)), 0.0)
    de = np.where(ok, (dEi[jj] - dEi[ii]) / Vt, 0.0)
    return de + lr, de - lr


def _mesh_mobility_nodal(mesh, mat):
    """Static (solve-independent) per-NODE doping-dependent mobility
    arrays, following the same precomputed-array pattern as
    _mesh_ni_edge_g's ni_arr/edge_g below (mesh.Cdop is always populated,
    homogeneous mesh or not, so - unlike ni_arr - this needs no
    homogeneous-mesh fallback branch)."""
    model = getattr(mat, "doping_mobility", "caughey_thomas")
    key = ("mu_nodal", model, tuple(mat.masetti_n), tuple(mat.masetti_p), mat.mu_n_max, mat.mu_n_min,
           mat.N_ref_n, mat.alpha_n, mat.mu_p_max, mat.mu_p_min, mat.N_ref_p, mat.alpha_p)
    return _mesh_cached(mesh, key, lambda: _mesh_mobility_nodal_build(mesh, mat, model))


def _mesh_mobility_nodal_build(mesh, mat, model):
    N_abs = np.abs(mesh.Cdop)
    if model == "masetti":
        return mobility_masetti(N_abs, mat.masetti_n), mobility_masetti(N_abs, mat.masetti_p)
    if model != "caughey_thomas":
        raise ValueError(f"doping_mobility must be 'masetti' or 'caughey_thomas', not {model!r}")
    mu_n_node = mobility_doping(N_abs, mat.mu_n_max, mat.mu_n_min, mat.N_ref_n, mat.alpha_n)
    mu_p_node = mobility_doping(N_abs, mat.mu_p_max, mat.mu_p_min, mat.N_ref_p, mat.alpha_p)
    return mu_n_node, mu_p_node


VSAT_FORCES = ("efield", "hybrid", "gradqf")
# Carrier density (cm^-3) where the hybrid driving force is half quasi-Fermi
# gradient, half field - the avalanche solver's value
# (avalanche/newton_solver_avalanche.py, DEFAULT_REF_DENSITY_CM3).
VSAT_REF_DENSITY = 1.0e16
# Newton iterations allowed to a quasi-Fermi-force solve from the given start
# before falling back to the edge-field warm-up (newton_solve_2d).
VSAT_DIRECT_MAXITER = 15


def vsat_force(velocity_saturation):
    """None (no velocity saturation) or one of VSAT_FORCES; True means the
    default, "hybrid" (DEVELOPMENT_LOG 49-50; "efield" was the default
    before)."""
    if velocity_saturation is False or velocity_saturation is None:
        return None
    if velocity_saturation is True:
        return "hybrid"
    if velocity_saturation not in VSAT_FORCES:
        raise ValueError(f"velocity_saturation must be a bool or one of {VSAT_FORCES}, not {velocity_saturation!r}")
    return velocity_saturation


def _edge_force(force, dpsi, dphi, c_i, c_j, L, Vt, sign):
    """Driving force F (V/cm, >= 0) of one carrier per edge and its partials
    (dF/dpsi_i, dF/dpsi_j, dF/dphi_i, dF/dphi_j). sign = +1 for electrons
    (dc/dpsi = c/Vt, dc/dphi = -c/Vt), -1 for holes.

      efield: |dpsi|/L
      gradqf: |dphi|/L - the force that drives (and heats) the carrier;
              parallel to its current, J = -q*mu*c*grad(phi)
      hybrid: w*|dphi|/L + (1-w)*|dpsi|/L, w = c^2/(c^2 + c_ref^2), c the
              edge-mean density: the quasi-Fermi gradient where the
              carrier is dense, the field where it is sparse (where phi is
              ill-determined and the carrier carries no current anyway).

    |x| is smoothed to sqrt(x^2 + E_SMOOTH^2) so F is differentiable at 0."""
    E = dpsi / L
    G = dphi / L
    Ea = np.sqrt(E ** 2 + E_SMOOTH ** 2)
    Ga = np.sqrt(G ** 2 + E_SMOOTH ** 2)
    z = np.zeros_like(E)
    if force == "efield":
        dE = E / Ea / L
        return Ea, (-dE, dE, z, z)
    if force == "gradqf":
        dG = G / Ga / L
        return Ga, (z, z, -dG, dG)
    c = 0.5 * (c_i + c_j)
    w = 1.0 / (1.0 + (VSAT_REF_DENSITY / np.maximum(c, 1e-300)) ** 2)
    dw_dc = 2.0 * w * (1.0 - w) / np.maximum(c, 1e-300)
    F = w * Ga + (1.0 - w) * Ea
    dE = (1.0 - w) * E / Ea / L
    dG = w * G / Ga / L
    k = dw_dc * (Ga - Ea) * 0.5 / Vt                       # through c = (c_i + c_j)/2
    return F, (-dE + sign * k * c_i, dE + sign * k * c_j, -dG - sign * k * c_i, dG - sign * k * c_j)


def edge_mobility(mesh, mat, psi, velocity_saturation=False, n=None, p=None, phin=None, phip=None):
    """Per-edge (mu_n_e, mu_p_e, dmu_n, dmu_p); dmu_* is None without
    velocity saturation, else the 4 partials (d/dpsi_i, d/dpsi_j,
    d/dphi_i, d/dphi_j) of that carrier's edge mobility.

    mu0 is the doping-dependent (Caughey-Thomas) low-field mobility,
    averaged to the edge. With velocity saturation it is further reduced by
    the Caughey-Thomas high-field model (mobility_field), driven per edge by
    the force selected with velocity_saturation (see _edge_force):

      * "efield": the electrostatic field projected on the edge,
        dpsi/edge_len. Not |E| at a node, which includes the large,
        current-free vertical gate field. But the edge projection still
        sees (a) the built-in field of the source/drain junctions, which is
        balanced by diffusion and heats nobody - about a 20% Id cut at
        Vd = 50 mV in the planar 3D device - and (b) part of the vertical
        gate field on the diagonal edges of non-rectilinear (octree
        Voronoi) meshes, which made Id depend on the tree layout
        (DEVELOPMENT_LOG 47).
      * "gradqf": the carrier's quasi-Fermi gradient, the force parallel to
        its current. Tried first in 2D (session notes in
        DEVELOPMENT_LOG): phin is undetermined where n ~ 0, and Newton
        turned erratic ramping Vds past ~0.5 V at Vgs = 0.
      * True / "hybrid" (the default): the quasi-Fermi gradient where the
        carrier is dense, the field where it is sparse - the avalanche solver's resolution of the
        same trade-off (FLOOXS/Charon damp the gradient force the same way).
    In equilibrium the edge current is 0 whatever mu is."""
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    mu_n_node, mu_p_node = _mesh_mobility_nodal(mesh, mat)
    mu_n0 = 0.5 * (mu_n_node[ii] + mu_n_node[jj])
    mu_p0 = 0.5 * (mu_p_node[ii] + mu_p_node[jj])
    force = vsat_force(velocity_saturation)
    if force is None:
        return mu_n0, mu_p0, None, None
    L = _edge_len(mesh)
    dpsi = psi[jj] - psi[ii]
    if force == "efield":
        z = np.zeros_like(dpsi)
        n = p = np.ones_like(psi)
        phin = phip = np.zeros_like(psi)
    Fn, dFn = _edge_force(force, dpsi, phin[jj] - phin[ii], n[ii], n[jj], L, mat.Vt, +1.0)
    Fp, dFp = _edge_force(force, dpsi, phip[jj] - phip[ii], p[ii], p[jj], L, mat.Vt, -1.0)
    mu_n, dmu_n_dF = mobility_field(Fn, mu_n0, mat.vsat_n, mat.beta_n)
    mu_p, dmu_p_dF = mobility_field(Fp, mu_p0, mat.vsat_p, mat.beta_p)
    return mu_n, mu_p, tuple(dmu_n_dF * d for d in dFn), tuple(dmu_p_dF * d for d in dFp)


def _semiconductor_edge_mask(mesh):
    """Boolean (E,) mask, True on edges where BOTH endpoints are real
    semiconductor (not insulator/oxide) - the physically correct no-flux
    BC at a semiconductor/insulator interface (Jn.n_hat = Jp.n_hat = 0
    there, current is confined to the semiconductor). Found (2026-09-13,
    jointly - see this module's mobility-doping-model commit and the
    matching root-cause writeup) to be a genuine, previously-undiagnosed
    discretization bug when missing: an edge with one endpoint a real
    semiconductor node (nonzero n or p) and the other an is_oxide_free
    insulator node (pinned phin=phip=0 - an ARBITRARY placeholder, not a
    real potential) computes a nonzero Jn_e/Jp_e driven by that arbitrary
    pinned value, using n_avg=0.5*(n_real+0) (still nonzero since only the
    OXIDE side's density is zero). The is_oxide_free row-pinning logic
    below only stops that node's OWN row from being assembled via the
    normal continuity equation - it does nothing to stop this edge's
    current from still being added into the ADJACENT semiconductor (or
    even CONTACT) node's row, so it silently leaks a spurious current with
    no governing conservation law. Confirmed empirically at a real MOSFET
    off-state bias point: 99.995% of the entire (spurious) drain current
    came from a single such edge at the corner where the drain contact
    meets the gate-oxide/mesa wall, tracking the DRAIN's own fixed
    Dirichlet BC (hence flat vs. Vgs) and completely independent of
    substrate doping (hence flat vs. a 1e16->1e18 doping sweep) - exactly
    the two symptoms flagged (but never root-caused) in earlier sessions.
    Returns an all-True mask (no insulator anywhere) for a homogeneous
    mesh (mesh.is_insulator is None, e.g. the plain diode)."""
    if mesh.is_insulator is None:
        return np.ones(len(mesh.edges), dtype=bool)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    return ~(mesh.is_insulator[ii] | mesh.is_insulator[jj])


def _mesh_ni_edge_g(mesh, mat):
    """A homogeneous-silicon mesh (the diode) has mesh.ni_arr/edge_g unset
    (mesh2d.build_mesh2d only populates them when built with mat= given -
    see that module's docstring); a heterogeneous mesh (a MOSFET or MOS
    capacitor, with an oxide region) has them populated. Falling back to
    mat.ni (scalar broadcast) / mat.eps*mesh.edge_weight here means this
    solver works unchanged for both kinds of mesh."""
    ni_arr = mesh.ni_arr if mesh.ni_arr is not None else np.full(len(mesh.points), mat.ni)
    g_e = mesh.edge_g if mesh.edge_g is not None else mat.eps * mesh.edge_weight
    return ni_arr, g_e


def _mesh_semi_geometry(mesh):
    """(facet_semi, cv_semi, cv_semi_safe): the semiconductor-only part of
    each edge's Voronoi facet and each node's control volume (see
    mesh2d/fvgeometry.py::build_fv_geometry's tri_insulator docstring).
    Carrier current flows only through facet_semi - an edge wholly inside
    the oxide has facet_semi=0, which IS the no-flux semiconductor/
    insulator BC, and an edge along the interface keeps only its silicon
    half. Carrier/doping charge and recombination live only in cv_semi -
    an interface node's oxide half-cell holds none. For a homogeneous mesh
    both equal the plain facet_length/cv_area. cv_semi_safe replaces 0
    (a pure-oxide node, whose continuity rows are pinned anyway) by the full
    cv_area so dividing by it is always finite."""
    facet_semi = mesh.facet_length_semi if getattr(mesh, "facet_length_semi", None) is not None \
        else mesh.facet_length
    cv_semi = mesh.cv_area_semi if getattr(mesh, "cv_area_semi", None) is not None else mesh.cv_area
    cv_semi_safe = np.where(cv_semi > 0.0, cv_semi, mesh.cv_area)
    return facet_semi, cv_semi, cv_semi_safe


def bernoulli(x):
    """B(x) = x/(exp(x)-1) and B'(x), overflow-safe, Taylor near 0."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, np.clip(x, -700.0, 700.0))
    B = np.where(small, 1.0 - x / 2.0 + x * x / 12.0, xs / np.expm1(xs))
    dB = np.where(small, -0.5 + x / 6.0, B * (1.0 - B) / xs - B)
    return B, dB


def bernoulli_pair(x):
    """(B(x), B'(x), B(-x), B'(-x)) from ONE evaluation, by B(-x) = B(x) + x
    and B'(-x) = -B'(x) - 1. Evaluated at a = |x| and shifted to the other
    side, so the added term never cancels (B(a) and a are both >= 0);
    bernoulli() at -x as well would cost a second exp - 4 Bernoulli calls
    per edge-current evaluation became 1 (DEVELOPMENT_LOG 59)."""
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    b, db = bernoulli(a)
    b_neg, db_neg = b + a, -db - 1.0          # B(-a), B'(-a)
    pos = x >= 0.0
    return (np.where(pos, b, b_neg), np.where(pos, db, db_neg),
            np.where(pos, b_neg, b), np.where(pos, db_neg, db))


def edge_currents(mesh, mat, psi, n, p, phin, phip, velocity_saturation=False, edge_mask=None,
                  jacobian=False, qs=None):
    """Electron/hole current through every edge's (semiconductor) facet,
    direction i -> j, A per unit depth - the ONE edge-current formula the
    residual, the Jacobian and solver2d/current.py all share.

    Scharfetter-Gummel, written in this solver's quasi-Fermi unknowns
    (nodal n, p = ni*exp(...) of psi/phin/phip, so no change of variables):
        In = Q*mu_n*Vt*facet/L * (n_j*B(d) - n_i*B(-d))
        Ip = Q*mu_p*Vt*facet/L * (p_i*B(d) - p_j*B(-d)),   d = (psi_j-psi_i)/Vt
    For d -> 0 this reduces exactly to the plain-gradient -Q*mu*n*dphi/L
    the solver used before; unlike it, it stays exact when the density
    varies exponentially along the edge. The arithmetic mean n_avg that
    form needed is badly wrong there: across a p/n junction p_avg ~
    p_p+/2, so a junction-crossing edge (the 2D diode's anode-end node to
    its n-type surface neighbor) became an ohmic hole shunt - found as a
    ~1e4x excess diode current near 0 V once domain-boundary edges started
    carrying flux (see mesh2d/fvgeometry.py).

    jacobian=True also returns the 8 partial derivatives
    (dIn/dpsi_i, dIn/dpsi_j, dIn/dphin_i, dIn/dphin_j,
     dIp/dpsi_i, dIp/dpsi_j, dIp/dphip_i, dIp/dphip_j)."""
    Vt = mat.Vt
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    edge_len = _edge_len(mesh)
    facet_semi, _, _ = _mesh_semi_geometry(mesh)
    mu_n, mu_p, dmu_n, dmu_p = edge_mobility(mesh, mat, psi, velocity_saturation, n, p, phin, phip)
    geo = Q * Vt * facet_semi / edge_len
    if edge_mask is not None:
        geo = geo * edge_mask
    d = (psi[jj] - psi[ii]) / Vt
    sn, sp_ = _sg_shifts(mesh, ii, jj, Vt) if qs is None else qs.sg
    Bp, dBp, Bm, dBm = bernoulli_pair(d + sn)
    if sp_ is sn:                  # single material: both shifts are the same 0.0
        Bpp, dBpp, Bmp, dBmp = Bp, dBp, Bm, dBm
    else:
        Bpp, dBpp, Bmp, dBmp = bernoulli_pair(d + sp_)
    Sn = n[jj] * Bp - n[ii] * Bm
    Sp = p[ii] * Bpp - p[jj] * Bmp
    In = geo * mu_n * Sn
    Ip = geo * mu_p * Sp
    if not jacobian:
        return In, Ip
    cn, cp = geo * mu_n, geo * mu_p
    # d/dpsi: through n,p (dn/dpsi = n/Vt, dp/dpsi = -p/Vt), through B(+-d)
    # (dd/dpsi_j = 1/Vt), and through mu (velocity saturation, dmu/d(dpsi)).
    dSn_dpsi_j = (n[jj] * Bp + n[jj] * dBp + n[ii] * dBm) / Vt
    dSn_dpsi_i = -(n[ii] * Bm + n[jj] * dBp + n[ii] * dBm) / Vt
    dSp_dpsi_j = (p[jj] * Bmp + p[ii] * dBpp + p[jj] * dBmp) / Vt
    dSp_dpsi_i = -(p[ii] * Bpp + p[ii] * dBpp + p[jj] * dBmp) / Vt
    dIn_dpsi_i = cn * dSn_dpsi_i
    dIn_dpsi_j = cn * dSn_dpsi_j
    dIp_dpsi_i = cp * dSp_dpsi_i
    dIp_dpsi_j = cp * dSp_dpsi_j
    # d/dphi: through n,p (dn/dphin = -n/Vt, dp/dphip = p/Vt).
    dIn_dphin_i = cn * n[ii] * Bm / Vt
    dIn_dphin_j = -cn * n[jj] * Bp / Vt
    dIp_dphip_i = cp * p[ii] * Bpp / Vt
    dIp_dphip_j = -cp * p[jj] * Bmp / Vt
    if dmu_n is not None:          # velocity saturation: mu depends on the edge's psi and phi
        vn, vp = geo * Sn, geo * Sp
        dIn_dpsi_i = dIn_dpsi_i + vn * dmu_n[0]
        dIn_dpsi_j = dIn_dpsi_j + vn * dmu_n[1]
        dIn_dphin_i = dIn_dphin_i + vn * dmu_n[2]
        dIn_dphin_j = dIn_dphin_j + vn * dmu_n[3]
        dIp_dpsi_i = dIp_dpsi_i + vp * dmu_p[0]
        dIp_dpsi_j = dIp_dpsi_j + vp * dmu_p[1]
        dIp_dphip_i = dIp_dphip_i + vp * dmu_p[2]
        dIp_dphip_j = dIp_dphip_j + vp * dmu_p[3]
    return In, Ip, (dIn_dpsi_i, dIn_dpsi_j, dIn_dphin_i, dIn_dphin_j,
                    dIp_dpsi_i, dIp_dpsi_j, dIp_dphip_i, dIp_dphip_j)


def poisson_row_scale(mat: Material, h_typ: float) -> float:
    return mat.eps * mat.Vt / h_typ ** 2


def continuity_row_scale(mat: Material, h_typ: float) -> float:
    """Q*Dn*ni/h_typ**2 - one power of h_typ MORE than
    core/newton_solver_qf.py's 1D formula (Q*Dn*ni/h_typ). 1D's continuity
    row divides a flux-density difference by cvol_i, a control-volume
    LENGTH (~h_typ); this module's 2D row instead divides by cv_area, a
    control-volume AREA (~h_typ**2) - so matching 1D's row-scale formula
    verbatim under-compensates by one power of h_typ, leaving every row
    roughly 1/h_typ too large (caught via a finite-difference Jacobian
    check whose typical, non-pathological row magnitude came out ~1e9-1e10
    instead of O(1) even at a physically sane near-equilibrium test state -
    not a Jacobian correctness bug, since FD and analytic agreed once eps
    was chosen to resolve derivatives at that same huge scale, but a
    genuine row-conditioning bug all the same, worth fixing before trusting
    Newton's convergence on it). Matches poisson_row_scale's own h_typ**2
    convention, which was already correct for the same reason."""
    return Q * mat.Dn * mat.ni / h_typ ** 2


def _residual(U, mesh, mat, is_contact, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale,
              edge_mask=None, velocity_saturation=False, generation=None, qs=None):
    N = len(mesh.points)
    psi, phin, phip = unpack_qf(U, N)
    _, g_e = _mesh_ni_edge_g(mesh, mat)
    ni_arr, dEi = _bands(mesh, mat, qs)
    n, p = _densities(psi, phin, phip, ni_arr, mat.Vt, dEi)

    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]

    _, cv_semi, cv_semi_safe = _mesh_semi_geometry(mesh)
    # Semiconductor-only facets already make every oxide edge carry zero
    # current; edge_mask (optional) can exclude further edges.
    In_e, Ip_e = edge_currents(mesh, mat, psi, n, p, phin, phip, velocity_saturation, edge_mask, qs=qs)

    div_psi = _edge_divergence(ii, jj, g_e * (psi[jj] - psi[ii]), N)

    div_n = _edge_divergence(ii, jj, In_e, N)

    div_p = _edge_divergence(ii, jj, Ip_e, N)

    # SRH denom is exactly 0 at an insulator node (n=p=ni_arr=0 there,
    # since ni_arr=0 makes _densities return 0 regardless of psi/phin/phip)
    # - guard the divide rather than let it raise/NaN; R is physically
    # moot there anyway (no recombination in an insulator).
    denom = mat.tau_p * (n + ni_arr) + mat.tau_n * (p + ni_arr)
    denom_safe = np.where(denom > 0.0, denom, 1.0)
    R = np.where(denom > 0.0, (n * p - ni_arr ** 2) / denom_safe, 0.0)

    Rpsi = (div_psi - Q * (n - p - mesh.Cdop) * cv_semi) / mesh.cv_area / poisson_scale
    Rn = (div_n / cv_semi_safe - Q * R) / cont_scale
    Rp = (div_p / cv_semi_safe + Q * R) / cont_scale
    if generation is not None:
        Gn, Gp, _, _ = generation(psi, phin, phip, jacobian=False)
        Rn = Rn + Q * Gn / cont_scale
        Rp = Rp - Q * Gp / cont_scale

    # phin/phip have NO governing equation at a non-contact insulator node
    # (n=p=0 there regardless of their value, since ni_arr=0 - there is no
    # current, no recombination, nothing for a continuity equation to
    # balance). Left alone, that makes the corresponding Jacobian row
    # either identically zero (an interior oxide node with only other
    # oxide neighbors - an exactly singular row) or driven by a spurious
    # "current" from a real semiconductor neighbor that has no physical
    # meaning (a node one layer into the oxide from the interface). Pin
    # both to 0 (arbitrary but harmless, exactly mirroring how a Dirichlet
    # contact row is overwritten below) at every such node instead.
    is_oxide_free = mesh.is_insulator & ~is_contact if mesh.is_insulator is not None \
        else np.zeros(N, dtype=bool)
    Rn = np.where(is_oxide_free, phin, Rn)
    Rp = np.where(is_oxide_free, phip, Rp)

    Rpsi[is_contact] = psi[is_contact] - psi_bc
    Rn[is_contact] = phin[is_contact] - phin_bc
    Rp[is_contact] = phip[is_contact] - phip_bc

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, mesh, mat, is_contact, psi_bc, phin_bc, phip_bc,
                            poisson_scale, cont_scale, edge_mask=None, velocity_saturation=False,
                            generation=None, qs=None):
    """generation: optional extra carrier generation (see newton_solve_2d).

    edge_mask: optional boolean (E,) array - see _residual()'s own
    (much longer) docstring for the full rationale; None (default, every
    existing caller) reproduces this function's exact prior behavior
    unchanged. Applied here by masking cn_e/cp_e (the common per-edge
    mobility*geometry prefactor every dIn_d*/dIp_d* Jacobian term is built
    from), plus In_e/Ip_e for F itself - equivalent to zeroing the excluded
    edges' current (and its derivatives) entirely."""
    N = len(mesh.points)
    psi, phin, phip = unpack_qf(U, N)
    _, g_e = _mesh_ni_edge_g(mesh, mat)
    ni_arr, dEi = _bands(mesh, mat, qs)
    Vt = mat.Vt
    n, p = _densities(psi, phin, phip, ni_arr, Vt, dEi)

    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]

    _, cv_semi, cv_semi_safe = _mesh_semi_geometry(mesh)
    In_e, Ip_e, (dIn_dpsi_i, dIn_dpsi_j, dIn_dphin_i, dIn_dphin_j,
                 dIp_dpsi_i, dIp_dpsi_j, dIp_dphip_i, dIp_dphip_j) = edge_currents(
        mesh, mat, psi, n, p, phin, phip, velocity_saturation, edge_mask, jacobian=True, qs=qs)

    div_psi = _edge_divergence(ii, jj, g_e * (psi[jj] - psi[ii]), N)
    div_n = _edge_divergence(ii, jj, In_e, N)
    div_p = _edge_divergence(ii, jj, Ip_e, N)

    # SRH denom is exactly 0 at an insulator node (n=p=ni_arr=0 there) -
    # guard the divide; R and its derivatives are physically moot there
    # (no recombination in an insulator) and must not propagate a NaN into
    # the rest of the (otherwise perfectly well-posed) linear system.
    denom = mat.tau_p * (n + ni_arr) + mat.tau_n * (p + ni_arr)
    denom_safe = np.where(denom > 0.0, denom, 1.0)
    num = n * p - ni_arr ** 2
    R = np.where(denom > 0.0, num / denom_safe, 0.0)

    Rpsi = (div_psi - Q * (n - p - mesh.Cdop) * cv_semi) / mesh.cv_area / poisson_scale
    Rn = (div_n / cv_semi_safe - Q * R) / cont_scale
    Rp = (div_p / cv_semi_safe + Q * R) / cont_scale
    dGn_dU = dGp_dU = None
    if generation is not None:
        Gn, Gp, dGn_dU, dGp_dU = generation(psi, phin, phip, jacobian=True)
        Rn = Rn + Q * Gn / cont_scale
        Rp = Rp - Q * Gp / cont_scale

    # See _residual()'s matching comment: phin/phip are pinned to 0 (not
    # solved via the normal continuity equation) at every non-contact
    # insulator node - otherwise either an exactly-singular all-zero
    # Jacobian row (an interior oxide node with only other oxide
    # neighbors) or a spurious coupling to a real semiconductor neighbor's
    # phin/phip with no physical basis.
    is_oxide_free = mesh.is_insulator & ~is_contact if mesh.is_insulator is not None \
        else np.zeros(N, dtype=bool)
    Rn = np.where(is_oxide_free, phin, Rn)
    Rp = np.where(is_oxide_free, phip, Rp)

    Rpsi[is_contact] = psi[is_contact] - psi_bc
    Rn[is_contact] = phin[is_contact] - phin_bc
    Rp[is_contact] = phip[is_contact] - phip_bc
    F = np.concatenate([Rpsi, Rn, Rp])

    dn_dpsi = n / Vt
    dn_dphin = -n / Vt
    dp_dpsi = -p / Vt
    dp_dphip = p / Vt
    dR_dn = np.where(denom > 0.0, (p * denom - num * mat.tau_p) / denom_safe ** 2, 0.0)
    dR_dp = np.where(denom > 0.0, (n * denom - num * mat.tau_n) / denom_safe ** 2, 0.0)

    # The (rows, cols) of every block below depend only on the mesh and on
    # which nodes are contacts / pinned oxide: built once (_jacobian_index)
    # and reused, so a call computes only the values (DEVELOPMENT_LOG 59).
    P = _jacobian_index(mesh, is_contact, is_oxide_free)
    Ai, Aj, Ci, Cj = P["Ai"], P["Aj"], P["Ci"], P["Cj"]
    iAi, jAi, iAj, jAj = P["iAi"], P["jAi"], P["iAj"], P["jAj"]
    iCi, jCi, iCj, jCj = P["iCi"], P["jCi"], P["iCj"], P["jCj"]
    free, free_np, oxide_free_idx = P["free"], P["free_np"], P["oxide_free"]
    is_np_fixed = P["is_np_fixed"]
    rows_list, cols_list, data_list = [], [], []

    def add(rows, cols, vals):
        rows_list.append(rows)
        cols_list.append(cols)
        data_list.append(vals)

    semi_frac = cv_semi / mesh.cv_area     # charge lives only in the semiconductor part
    cv = mesh.cv_area

    # --- Poisson: per-edge Laplacian coupling, row masked to non-contact ---
    gA_i = g_e[Ai] / cv[iAi] / poisson_scale
    gA_j = g_e[Aj] / cv[jAj] / poisson_scale
    add(iAi, iAi, -gA_i)
    add(iAi, jAi, gA_i)
    add(jAj, jAj, -gA_j)
    add(jAj, iAj, gA_j)

    # --- Poisson: per-node local charge derivative, non-contact nodes only ---
    add(free, free,
        (-Q * (dn_dpsi[free] - dp_dpsi[free]) * semi_frac[free] / poisson_scale))
    add(free, N + free, (-Q * dn_dphin[free] * semi_frac[free] / poisson_scale))
    add(free, 2 * N + free, (Q * dp_dphip[free] * semi_frac[free] / poisson_scale))

    # Continuity rows additionally exclude is_oxide_free nodes (pinned to
    # phin=0/phip=0 above, not assembled normally here).
    s_i = 1.0 / (cv_semi_safe[iCi] * cont_scale)
    s_j = -1.0 / (cv_semi_safe[jCj] * cont_scale)
    for row0, dI in ((N, (dIn_dpsi_i, dIn_dpsi_j, dIn_dphin_i, dIn_dphin_j)),
                     (2 * N, (dIp_dpsi_i, dIp_dpsi_j, dIp_dphip_i, dIp_dphip_j))):
        # --- per-edge current derivative: d/dpsi_i, d/dpsi_j, d/dphi_i, d/dphi_j ---
        add(row0 + iCi, iCi, dI[0][Ci] * s_i)
        add(row0 + iCi, jCi, dI[1][Ci] * s_i)
        add(row0 + iCi, row0 + iCi, dI[2][Ci] * s_i)
        add(row0 + iCi, row0 + jCi, dI[3][Ci] * s_i)
        add(row0 + jCj, iCj, dI[0][Cj] * s_j)
        add(row0 + jCj, jCj, dI[1][Cj] * s_j)
        add(row0 + jCj, row0 + iCj, dI[2][Cj] * s_j)
        add(row0 + jCj, row0 + jCj, dI[3][Cj] * s_j)
        # --- per-node recombination derivative (electrons -R, holes +R) ---
        sg = -Q / cont_scale if row0 == N else Q / cont_scale
        add(row0 + free_np, free_np, sg * (dR_dn[free_np] * dn_dpsi[free_np] + dR_dp[free_np] * dp_dpsi[free_np]))
        add(row0 + free_np, N + free_np, sg * dR_dn[free_np] * dn_dphin[free_np])
        add(row0 + free_np, 2 * N + free_np, sg * dR_dp[free_np] * dp_dphip[free_np])

    # --- Extra generation: d(Gn)/dU, d(Gp)/dU (sparse N x 3N over
    # U = (psi, phin, phip), or None for a source held fixed during this
    # solve) ---
    for dG, row0, sgn in ((dGn_dU, N, 1.0), (dGp_dU, 2 * N, -1.0)):
        if dG is not None:
            c = sp.coo_matrix(dG)
            keep = ~is_np_fixed[c.row]
            add(row0 + c.row[keep], c.col[keep], sgn * Q * c.data[keep] / cont_scale)

    # --- phin/phip identity-pinning rows at non-contact insulator nodes
    # (see the matching Rn/Rp override above) - these rows have NO other
    # contribution (excluded from every block above via is_np_fixed), so a
    # plain diagonal 1 is exact, not just a pivoting safety net. ---
    add(N + oxide_free_idx, N + oxide_free_idx, np.ones(len(oxide_free_idx)))
    add(2 * N + oxide_free_idx, 2 * N + oxide_free_idx, np.ones(len(oxide_free_idx)))

    with_gen = dGn_dU is not None or dGp_dU is not None
    interior_data = np.concatenate(data_list)
    dirichlet_idx = P["dirichlet"]
    # --- Dirichlet rows: scaled-identity, same pivoting-safety trick as
    # core/newton_solver_qf.py (diag = max(1.0, largest existing entry in
    # that column) so equilibration/pivoting never dilutes it). ---
    if with_gen:
        interior_rows = np.concatenate(rows_list)
        interior_cols = np.concatenate(cols_list)
        col_max = np.zeros(3 * N)
        np.maximum.at(col_max, interior_cols, np.abs(interior_data))
        dirichlet_data = np.maximum(1.0, col_max[dirichlet_idx])
        rows = np.concatenate([interior_rows, dirichlet_idx])
        cols = np.concatenate([interior_cols, dirichlet_idx])
    else:
        if "rows" not in P:
            interior_cols = np.concatenate(cols_list)
            P["rows"] = np.concatenate(rows_list + [dirichlet_idx]).astype(np.int32)
            P["cols"] = np.concatenate([interior_cols, dirichlet_idx]).astype(np.int32)
            slot_of = np.full(3 * N, -1)
            slot_of[dirichlet_idx] = np.arange(len(dirichlet_idx))
            P["dir_pos"] = np.flatnonzero(slot_of[interior_cols] >= 0)       # entries in a Dirichlet column
            P["dir_slot"] = slot_of[interior_cols[P["dir_pos"]]]
        col_max = np.zeros(len(dirichlet_idx))
        np.maximum.at(col_max, P["dir_slot"], np.abs(interior_data[P["dir_pos"]]))
        dirichlet_data = np.maximum(1.0, col_max)
        rows, cols = P["rows"], P["cols"]
    data = np.concatenate([interior_data, dirichlet_data])
    # CSR with explicit zeros kept: the sparsity pattern then never changes
    # between iterations, which core/linsolve.py relies on to reuse its
    # pattern analysis (ordering) and factorizations. The (rows, cols)
    # index arrays depend only on the mesh and which nodes are contacts /
    # pinned oxide, so the COO->CSR mapping is computed once (an "assembly
    # plan") and every later call is a single bincount - no per-iteration
    # sort or duplicate merge. A generation term with its own sparse
    # derivative (tunneling) adds entries whose pattern is usually, but not
    # always, the same every iteration, so its plan is reused only after
    # checking that (rows, cols) match the ones it was built from.
    J = _assemble_with_plan(mesh, is_contact, rows, cols, data, 3 * N, verify=with_gen)
    return F, J


_JAC_INDEX = {}


def _jacobian_index(mesh, is_contact, is_oxide_free):
    """The mesh/contact-only index arrays of _residual_and_jacobian's
    blocks (edges whose i / j row is assembled, their endpoints, free
    nodes), once per (mesh, contact set)."""
    key = (id(mesh), is_contact.tobytes())
    P = _JAC_INDEX.get(key)
    if P is not None and P["mesh"] is not None and P["mesh"]() is mesh:
        return P
    N = len(mesh.points)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    is_np_fixed = is_contact | is_oxide_free
    node = np.arange(N)
    Ai, Aj = np.flatnonzero(~is_contact[ii]), np.flatnonzero(~is_contact[jj])
    Ci, Cj = np.flatnonzero(~is_np_fixed[ii]), np.flatnonzero(~is_np_fixed[jj])
    contact_idx = node[is_contact]
    try:
        ref = weakref.ref(mesh)
    except TypeError:                     # a stand-in mesh: not cached
        ref = None
    P = dict(mesh=ref, Ai=Ai, Aj=Aj, Ci=Ci, Cj=Cj,
             iAi=ii[Ai], jAi=jj[Ai], iAj=ii[Aj], jAj=jj[Aj],
             iCi=ii[Ci], jCi=jj[Ci], iCj=ii[Cj], jCj=jj[Cj],
             free=node[~is_contact], free_np=node[~is_np_fixed], oxide_free=node[is_oxide_free],
             is_np_fixed=is_np_fixed,
             dirichlet=np.concatenate([contact_idx, N + contact_idx, 2 * N + contact_idx]))
    if ref is not None:
        _JAC_INDEX[key] = P
    return P


def _assemble_with_plan(mesh, is_contact, rows, cols, data, n, verify=False):
    key = (id(mesh), n, len(rows), is_contact.tobytes(), verify)
    plan = _ASSEMBLY_PLANS.get(key)
    if (plan is None or plan[0]() is not mesh
            or (verify and not (np.array_equal(plan[5], rows) and np.array_equal(plan[6], cols)))):
        lin = rows.astype(np.int64) * n + cols
        uniq, slot = np.unique(lin, return_inverse=True)
        indptr = np.concatenate([[0], np.cumsum(np.bincount(uniq // n, minlength=n))])
        plan = (weakref.ref(mesh), slot, indptr, (uniq % n).astype(np.int32), len(uniq),
                rows.copy() if verify else None, cols.copy() if verify else None)
        _ASSEMBLY_PLANS[key] = plan
    _, slot, indptr, indices, nnz = plan[:5]
    return sp.csr_matrix((np.bincount(slot, weights=data, minlength=nnz), indices, indptr), shape=(n, n))


def _row_absmax(J):
    """Largest |entry| per row of a CSR matrix with no empty rows (every
    row of this solver's Jacobian has at least its diagonal)."""
    return np.maximum.reduceat(np.abs(J.data), J.indptr[:-1])


_POISSON_SOLVERS = {}


def _poisson_linear_solver(mesh):
    """A scalar (psi-only) LinearSolver per mesh, kept like
    default_linear_solver's: the pattern never changes, so its ILU is
    reused across calls."""
    key = id(mesh)
    entry = _POISSON_SOLVERS.get(key)
    if entry is None or entry[0]() is not mesh:
        entry = (weakref.ref(mesh), LinearSolver(rtol=1e-10))
        _POISSON_SOLVERS[key] = entry
    return entry[1]


def poisson_solve(mesh, mat, psi, phin, phip, is_contact, psi_bc, poisson_scale, tol=1e-9, maxiter=60,
                  max_step=1.0, verbose=False, qs=None):
    """Nonlinear Poisson for psi with phin and phip FROZEN (the Gummel
    half-step): the same Poisson rows as the coupled system, with no
    mobility, current or recombination term - so it is cheap and the same
    for any transport model. Newton with each node's update clipped to
    max_step V (the Boltzmann charge makes the problem monotone, so this
    always converges). Returns (psi, iterations)."""
    N = len(mesh.points)
    _, g_e = _mesh_ni_edge_g(mesh, mat)
    ni_arr, dEi = _bands(mesh, mat, qs)
    Vt = mat.Vt
    _, cv_semi, _ = _mesh_semi_geometry(mesh)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    free = ~is_contact
    # the Laplacian rows (fixed): row r = sum_e g_e (psi_j - psi_r) / cv_r / scale, Dirichlet at contacts
    cv = mesh.cv_area
    r = np.concatenate([ii, jj, ii, jj])
    c = np.concatenate([jj, ii, ii, jj])
    v = np.concatenate([g_e / cv[ii], g_e / cv[jj], -g_e / cv[ii], -g_e / cv[jj]]) / poisson_scale
    keep = free[r]
    Lap = sp.csr_matrix((v[keep], (r[keep], c[keep])), shape=(N, N))
    semi_frac = cv_semi / cv
    lin = _poisson_linear_solver(mesh)
    psi = psi.copy()
    psi[is_contact] = psi_bc
    it = 0
    for it in range(1, maxiter + 1):
        n, p = _densities(psi, phin, phip, ni_arr, Vt, dEi)
        F = Lap @ psi - Q * (n - p - mesh.Cdop) * semi_frac / poisson_scale
        F[is_contact] = 0.0
        d = np.where(free, -Q * (n + p) / Vt * semi_frac / poisson_scale, 1.0)
        J = Lap + sp.diags(d)
        dpsi = lin.solve(J, -F)
        dpsi[is_contact] = 0.0
        dpsi = np.clip(dpsi, -max_step, max_step)
        psi += dpsi
        if verbose:
            print(f"  Poisson it {it}: max|dpsi| = {np.abs(dpsi).max():.2e}")
        if np.abs(dpsi).max() < tol:
            break
    return psi, it


def newton_solve_2d(mesh, mat: Material, bias_by_contact, f_tol=1e-9, maxiter=50, verbose=False,
                     psi_init=None, phin_init=None, phip_init=None,
                     psi_bc_override=None, phin_bc_override=None, phip_bc_override=None,
                     damping="line_search", cold_retry=True, velocity_saturation=False,
                     generation=None, linear_solver="auto", vsat_warmup=True, qf_box=True,
                     vsat_fallback=True, poisson_start=True, schottky=None, quantum=None,
                     quantum_update=None, quantum_tol=3e-3):
    """Solve the 2D QF system with each contact held at the voltage given
    in `bias_by_contact` ({contact_name: volts} - any contact not listed
    defaults to 0V). Returns a dict matching core/newton_solver_qf.py's
    shape (psi, n, p, phin, phip, iters).

    This assumes every contact is an ideal OHMIC contact on real
    semiconductor (mass-action/charge-neutrality BC via contact_values) by
    default - correct for a diode's anode/cathode or a MOSFET's source/
    drain/body, but WRONG for an ideal-metal gate sitting on an insulator
    (see solver2d/poisson2d_mos.py's own gate BC: psi_bulk + (VG-V_FB), not
    an ohmic relation, which is meaningless where ni=0 anyway). For any
    such contact, pass its already-computed Dirichlet value(s) via
    `psi_bc_override`/`phin_bc_override`/`phip_bc_override`
    ({contact_name: value}) to bypass the ohmic-BC computation entirely for
    that contact - the driver (e.g. main2d_mosfet_sweep.py) is expected to
    compute the gate's psi_bc itself (mos.mos_analytic.flatband_voltage,
    same as the MOS capacitor) and pass it through here.

    damping: "line_search" (default) - the plain backtracking line search
    below, unchanged since this module's first version. "bank_rose" -
    Bank & Rose (1981) Algorithm Global (core/bank_rose_damping.py),
    ported from avalanche/newton_solver_avalanche.py's own use of it,
    available for a caller that wants to try it on a SPECIFIC hard
    continuation step (e.g. a warm-started point deep in subthreshold at
    high Vds) rather than as a blanket replacement.

    "bank_rose" is NOT a safe drop-in replacement for "line_search" here
    and must not be made the default - confirmed via direct instrumentation
    (2026-09-20 session) on the Vgs=0V, Vds=1.0V cold-start point (which
    "line_search" converges cleanly, 83 iterations, res_norm~1e-5): Algorithm
    Global's eq-3.1 acceptance test compares the norm of the CLIPPED next
    Newton correction at two candidate damping levels, not ||F|| itself. From
    a cold start this far from the target bias, the raw (unclipped)
    correction is large enough that step_clip_fn (_uniform_clip above)
    saturates at its ceiling on EVERY trial, not rarely as the clip's own
    docstring assumes - and a trial whose clipped-correction norm happens to
    land under the ceiling can pass the test and get ACCEPTED even when its
    actual ||F|| is objectively worse than a REJECTED, less-damped trial's
    (directly measured: t=1 (K=0) gave |F|=1.76e10, smaller/better than the
    t=0.056 (K=10) trial that was accepted at |F|=3.57e10 - the accepted
    trial was the worse one). K then ratchets to its ceiling within a
    handful of further iterations (each subsequent trial only marginally
    worse, which still fails the test and keeps growing K), freezing t~0 and
    stalling with a huge, unimproving residual - reproduced with
    MAX_QF_STEP/MAX_PSI_STEP shrunk all the way from 5.0/1.0 down to
    0.1/0.1, so this is NOT a step-cap tuning issue: it is a structural
    mismatch between step_clip_fn's saturation and what Algorithm Global's
    global sufficient-decrease test assumes about g_k's meaning, triggered
    whenever clipping engages on essentially every trial rather than as a
    rare safety net (as it does starting a Vds=1.0V solve cold, but
    should NOT if this is instead invoked on an already-close warm start).
    Use only on warm-started continuation steps close to a known-good
    neighbor, never as the cold-start solver.

    psi_init/phin_init/phip_init, if given, warm-start Newton from a
    previous bias point's converged solution (see main2d_sweep.py) instead
    of the fresh charge-neutral-plus-tiny-perturbation guess - the same
    idea 1D's own voltage_sweep uses, needed here because a handful of
    reverse-bias points failed to converge from a cold start every time
    (e.g. -0.9V, -0.7V while -0.8V/-1.0V converged fine) even though this
    is otherwise the mild, first-example doping case. If a warm start
    still fails to converge, retry once from the fresh cold start (the one
    piece of core/newton_solver_qf.py's fuller Gummel-restart robustness
    layer ported here - a full 2D Gummel solver is still deferred).
    cold_retry=False skips that retry (and the non-convergence warning) -
    for a continuation driver that handles a failed step itself by
    shrinking the bias step, where a cold restart only wastes time.

    generation: optional extra electron/hole generation (e.g. band-to-band
    tunneling, btbt/), a callable generation(psi, phin, phip, jacobian) ->
    (Gn, Gp, dGn_dU, dGp_dU): per-node rates in cm^-3 s^-1 (electrons
    and holes may be generated at DIFFERENT nodes - a nonlocal tunneling
    path), and their sparse (N x 3N) derivatives w.r.t. U = (psi, phin,
    phip), or None for a source held fixed during the solve (lagged by the
    caller). Added as -G to the
    recombination term of the electron/hole continuity rows. Only the
    default line-search damping supports it. None (default) leaves the
    solve unchanged.

    linear_solver: "auto" (default) - the per-mesh cached
    core.linsolve.LinearSolver (factorization reuse + inexact Newton, see
    _run_newton); a LinearSolver instance - use that one; "legacy" - the
    original equilibrated SuperLU spsolve with an exact solve every
    iteration (kept for A/B comparisons).

    velocity_saturation: False, "efield", True / "hybrid" or "gradqf" - the
    driving force of the high-field mobility (edge_mobility). With a
    quasi-Fermi force (hybrid, gradqf) and vsat_warmup=True (default), a
    point that does not converge within VSAT_DIRECT_MAXITER iterations is
    redone: converged with "efield" from the same start, then with the
    requested force from there. "iters" counts everything, "iters_warmup"
    the iterations before the final stage (0 if the direct solve worked).
    vsat_fallback=False skips the fallback and returns the failed direct
    attempt, for a continuation driver that halves its bias step instead
    (fab/simulate.py). The 2D drivers keep it: the BTBT turn-on ramp
    needs it.

    qf_box: project the Newton iterates' quasi-Fermi levels onto the range
    of the semiconductor contacts' Fermi levels (the maximum principle; see
    the comment where it is set up). Off when generation is given (tunneling
    or impact-ionization pairs can carry them outside); kept on with a
    generation hook marked keeps_qf_box (the Schottky contact sink).

    schottky: optional contacts.subgrid.SubgridContacts - sub-grid Schottky
    contacts (plans/schottky_plan.md section 8). Their semiconductor nodes
    are released from the Dirichlet set (free nodes, kept neutral by
    Poisson), the contact law enters as a generation term, and the metal's
    insulator-side nodes sit at the metal's own potential V - Phi_Bn +
    (Ec - Ei). None (default) leaves the solve unchanged.

    quantum_update: state -> (Lambda_n, Lambda_p), the Schrodinger-Poisson
    corrector (quantum.device.DeviceQuantum.update) called INSIDE the Newton
    loop: once the update drops below QUANTUM_TRIGGER the quantum potentials
    are recomputed from the current psi and Newton continues on the new
    band shift. The solve is converged only when Newton is converged and the
    last corrector changed Lambda by less than quantum_tol. One Newton solve
    per bias point instead of an outer loop of solves."""
    if vsat_warmup and vsat_force(velocity_saturation) in ("hybrid", "gradqf"):
        # A quasi-Fermi force can derail Newton from a poor start: a new
        # contact bias enters as a jump of phin/phip across the first edge
        # at the contact (0.05 V over 0.5 nm ~ 1e6 V/cm), the force reads
        # that as velocity saturation, the edge's phi-conductance collapses
        # (a saturated edge carries ~q*c*vsat whatever its phi drop), and
        # Newton diverges (3D planar device, DEVELOPMENT_LOG 49-50).
        # So: try the requested force directly (a good start - the previous
        # point of a continuation - converges as usual), capped so a bad
        # start fails fast; if it fails, converge with the edge-field force
        # (blind to phi, so it settles phi) and continue from there. Not
        # the warm-up first: from an efield state the high-bias pinch-off
        # differs so much that continuation sweeps failed (2D NMOS).
        kw = dict(f_tol=f_tol, verbose=verbose, psi_bc_override=psi_bc_override,
                  phin_bc_override=phin_bc_override, phip_bc_override=phip_bc_override, damping=damping,
                  generation=generation, linear_solver=linear_solver, qf_box=qf_box, poisson_start=poisson_start,
                  schottky=schottky, quantum=quantum, quantum_update=quantum_update, quantum_tol=quantum_tol)
        try:
            r = newton_solve_2d(mesh, mat, bias_by_contact, psi_init=psi_init, phin_init=phin_init,
                                phip_init=phip_init, maxiter=min(maxiter, VSAT_DIRECT_MAXITER), cold_retry=False,
                                velocity_saturation=velocity_saturation, vsat_warmup=False, **kw)
        except RuntimeError:              # e.g. an iterative linear solve that broke down on the bad start
            r = {"converged": False, "iters": min(maxiter, VSAT_DIRECT_MAXITER)}
        if r["converged"]:
            r["iters_warmup"] = 0
            return r
        if not vsat_fallback:
            # The caller (a continuation driver) halves its bias step
            # instead, which can be far cheaper: the walk's first gate step
            # at Vds = 1 V on the full planar device cost 50 iterations
            # through the fallback, 15 + 5 by halving (DEVELOPMENT_LOG 55).
            r["iters_warmup"] = r["iters"]
            return r
        it_direct = r["iters"]
        try:
            w = newton_solve_2d(mesh, mat, bias_by_contact, psi_init=psi_init, phin_init=phin_init,
                                phip_init=phip_init, maxiter=maxiter, cold_retry=cold_retry,
                                velocity_saturation="efield", **kw)
            r = newton_solve_2d(mesh, mat, bias_by_contact, psi_init=w["psi"], phin_init=w["phin"],
                                phip_init=w["phip"], maxiter=maxiter, cold_retry=False,
                                velocity_saturation=velocity_saturation, vsat_warmup=False, **kw)
        except RuntimeError:          # a linear-solver breakdown: a failed point (the caller halves its step)
            return {"converged": False, "iters": it_direct + maxiter, "iters_warmup": it_direct}
        r["iters_warmup"] = it_direct + w["iters"]
        r["iters"] = r["iters_warmup"] + r["iters"]
        return r

    N = len(mesh.points)
    Vt = mat.Vt
    ni_arr, _ = _mesh_ni_edge_g(mesh, mat)
    # quantum: (Lambda_n, Lambda_p) or a QuantumShift. The contacts keep the
    # mesh's own ni / dEi (no quantum box reaches a contact).
    qs = quantum if quantum is None or isinstance(quantum, QuantumShift) else QuantumShift(mesh, mat, *quantum)
    ni_q, dEi_q = _bands(mesh, mat, qs)
    if schottky is not None:
        generation = schottky.generation(bias_by_contact, extra=generation)
    psi_bc_override = psi_bc_override or {}
    phin_bc_override = phin_bc_override or {}
    phip_bc_override = phip_bc_override or {}

    boundary_bc_type = np.array(mesh.boundary_bc_type)
    is_contact_boundary = np.array([bc.startswith("contact:") for bc in boundary_bc_type])
    contact_point_idx = mesh.boundary_point_index[is_contact_boundary]
    contact_names = [bc.split(":", 1)[1] for bc in boundary_bc_type[is_contact_boundary]]
    if schottky is not None:
        keep = ~np.isin(contact_point_idx, schottky.released())
        contact_point_idx = contact_point_idx[keep]
        contact_names = [nm for nm, k in zip(contact_names, keep) if k]

    is_contact_full = np.zeros(N, dtype=bool)
    is_contact_full[contact_point_idx] = True

    # ni_arr is 0 at an insulator node (mesh.is_insulator), which would
    # divide-by-zero into a NaN "equilibrium potential" there (moot anyway,
    # since that node is never a free/non-contact unknown in the physics
    # sense - it's either a Dirichlet override, e.g. a gate on oxide, or an
    # interior insulator node with no equilibrium relation to speak of).
    # Substitute a safe placeholder ni there so this stays finite and
    # doesn't poison the cold-start guess with a NaN.
    is_insulator = mesh.is_insulator if mesh.is_insulator is not None else np.zeros(N, dtype=bool)
    ni_safe_eq = np.where(is_insulator, 1.0, ni_arr)
    dEi_eq = mesh.dEi_arr if getattr(mesh, "dEi_arr", None) is not None else None
    psi_eq = np.where(is_insulator, 0.0,
                      equilibrium_bulk_potential_arr(Vt, ni_safe_eq, mesh.Cdop, delta_Ei_arr=dEi_eq))

    psi_bc = np.empty(len(contact_point_idx))
    phin_bc = np.empty(len(contact_point_idx))
    phip_bc = np.empty(len(contact_point_idx))
    # An ohmic contact that is a metal VOLUME (3D virtual-fab devices) also
    # touches insulator nodes (its sidewalls in the ILD/spacer). The metal is
    # one equipotential: those nodes take the potential of the contact's
    # semiconductor side (ohmic = metal at the contacted region's Fermi
    # level, psi = V + psi_bulk), median over its semiconductor nodes; the
    # local-neutrality formula below would divide by ni = 0 there.
    names_arr = np.array(contact_names, dtype=object)
    metal_psi0 = {}
    for name in set(contact_names) - set(psi_bc_override):
        if schottky is not None and name in schottky.specs:
            metal_psi0[name] = schottky.metal_psi0(name)
            continue
        pts = contact_point_idx[names_arr == name]
        semi_pts = pts[~is_insulator[pts]]
        if len(semi_pts) < len(pts):
            if not len(semi_pts):
                raise ValueError(f"ohmic contact {name!r} touches no semiconductor node")
            metal_psi0[name] = float(np.median(psi_eq[semi_pts]))
    for k, (pt, name) in enumerate(zip(contact_point_idx, contact_names)):
        v_applied = bias_by_contact.get(name, 0.0)
        if name in psi_bc_override:
            psi_bc[k] = psi_bc_override[name]
            phin_bc[k] = phin_bc_override[name]
            phip_bc[k] = phip_bc_override[name]
        elif is_insulator[pt]:
            psi_bc[k] = metal_psi0[name] + v_applied
            phin_bc[k] = phip_bc[k] = v_applied
        else:
            n_bc, p_bc = contact_values(mat, mesh.Cdop[pt], ni=ni_arr[pt])
            psi_bc[k] = psi_eq[pt] + v_applied
            dEi_pt = dEi_eq[pt] if dEi_eq is not None else 0.0
            phin_bc[k] = psi_bc[k] + dEi_pt - Vt * np.log(n_bc / ni_arr[pt])
            phip_bc[k] = psi_bc[k] + dEi_pt + Vt * np.log(p_bc / ni_arr[pt])

    def _cold_start():
        # phin0=phip0=0 EXACTLY everywhere (the natural equilibrium guess)
        # makes every edge's dphin_e/dphip_e term vanish identically, which
        # singles out an actually-singular direction in the Jacobian - the
        # same "flat interior at Va=0" critical point core/newton_solver_qf.py's
        # own comments describe (there recovered via a Gummel-restart;
        # confirmed here directly via spsolve raising MatrixRankWarning:
        # Matrix is exactly singular at the very first Newton step). Break
        # the EXACT flat degeneracy with a tiny, deterministic, physically
        # negligible perturbation (~1e-6*Vt, six orders of magnitude below
        # any physically meaningful potential) so Newton has a well-posed
        # direction to move in from the first step.
        rng = np.random.default_rng(0)
        psi0 = psi_eq.copy()
        psi0[contact_point_idx] = psi_bc
        qf_eq = np.concatenate([phin_bc[semi_bc], phip_bc[semi_bc]])
        if poisson_start and len(qf_eq) and np.ptp(qf_eq) < 1e-12:
            # Equilibrium (every semiconductor contact at one Fermi level):
            # phin = phip = that level everywhere is exact, so solve Poisson
            # for psi first and leave the coupled Newton ~nothing to do. Full
            # planar 3D device: 9 -> 2 coupled iterations, 5.3 -> 1.7 s
            # (DEVELOPMENT_LOG 58). Not under bias: there flat quasi-Fermi
            # levels are a worse start than local neutrality (the 2D diode's
            # cold retry at 0.2 V then failed), and as a Gummel half-step
            # before every warm start it cost more than it saved.
            flat = np.full(N, qf_eq[0])
            psi0, _ = poisson_solve(mesh, mat, psi0, flat, flat, is_contact_full, psi_bc, poisson_scale, qs=qs)
            # psi is then right from the first coupled iteration, which
            # would pass the update test with the perturbation below only
            # cut to ~1e-12 V (the 2D diode's equilibrium current 6e-9
            # instead of 1e-13 A/cm^2): take at least a second iteration.
            state["min_iters"] = 2
        phin0 = 1e-6 * Vt * rng.standard_normal(N)
        phin0[contact_point_idx] = phin_bc
        phip0 = 1e-6 * Vt * rng.standard_normal(N)
        phip0[contact_point_idx] = phip_bc
        return psi0, phin0, phip0

    # Use the mesh generator's own intended nominal minimum spacing, NOT
    # np.min(actual edge lengths) - mesh2d/pointcloud.py's jitter (needed to
    # break Delaunay-of-a-grid degeneracy, see its own docstring) can by
    # design push an isolated edge a little below h_min_cm, and since this
    # single global h_typ sets EVERY row's normalization for the whole
    # system, one such outlier edge anywhere throws off poisson_scale/
    # cont_scale for the entire solve, not just its own two nodes (caught
    # via a finite-difference Jacobian check whose residual blew up to
    # ~1e11-1e12 well away from any Dirichlet or degenerate-mesh row before
    # this fix).
    h_typ = mesh.h_min_cm
    poisson_scale = poisson_row_scale(mat, h_typ)
    cont_scale = continuity_row_scale(mat, h_typ)
    edge_mask = None
    if linear_solver == "legacy":
        lin = None
    elif linear_solver in (None, "auto"):
        lin = default_linear_solver(mesh)
    else:
        lin = linear_solver
    if lin is not None and hasattr(lin, "set_coarse"):
        # Schottky S/D islands are near-null directions for ILU (see
        # contacts/subgrid.py coarse_space); none otherwise.
        lin.set_coarse(schottky.coarse_space_cached() if schottky is not None else None)

    state = {}
    # Quasi-Fermi box (maximum principle). Without generation, the solution's
    # phin and phip lie between the lowest and highest Fermi level of the
    # semiconductor contacts (Markowich, "The Stationary Semiconductor
    # Device Equations", 1986; recombination only pulls them together), at
    # every node, for any mobility model - measured exactly so on the 3D
    # planar device. Newton iterates are projected onto that box at the free
    # semiconductor nodes (contacts and the pinned oxide rows keep their
    # values). Without it Newton can chase a false solution at infinity:
    # where a gate depletes an n+ corner below the hybrid c_ref, raising
    # phin lowers n, the drive turns to the edge field, the edges
    # saturate, and phin ran to +98 V with n = 0 (DEVELOPMENT_LOG 55). Full
    # planar device, drain ramp to 1 V: 173 -> 58 Newton iterations, no
    # failed step; converged results unchanged.
    semi_bc = ~is_insulator[contact_point_idx]
    box_free = ~is_insulator & ~is_contact_full
    # Not in equilibrium (a box of zero width): it would set phin = phip = 0
    # EXACTLY at every node, the flat degenerate state _cold_start breaks on
    # purpose - harmless for that solve, but the next one (the BTBT turn-on
    # ramp) then started from it and failed.
    qf_lo = qf_hi = None
    if qf_box and (generation is None or getattr(generation, "keeps_qf_box", False)) and semi_bc.any():
        lo = min(phin_bc[semi_bc].min(), phip_bc[semi_bc].min())
        hi = max(phin_bc[semi_bc].max(), phip_bc[semi_bc].max())
        if schottky is not None:
            vs = [bias_by_contact.get(nm, 0.0) for nm in schottky.specs]
            lo, hi = min([lo] + vs), max([hi] + vs)
        if hi > lo:
            qf_lo, qf_hi = lo, hi

    def _project(U_):
        if qf_lo is None:
            return U_
        U_ = U_.copy()
        for b in (1, 2):
            v = U_[b * N:(b + 1) * N]
            v[box_free] = np.clip(v[box_free], qf_lo, qf_hi)
        return U_

    def _run_newton(psi0, phin0, phip0):
        nonlocal qs, ni_q, dEi_q
        for k_ in ("q_ok", "q_first_it", "q_mix", "q_dlam", "q_hist"):
            state.pop(k_, None)
        psi0 = psi0.copy(); phin0 = phin0.copy(); phip0 = phip0.copy()
        psi0[contact_point_idx] = psi_bc
        phin0[contact_point_idx] = phin_bc
        phip0[contact_point_idx] = phip_bc
        U = np.concatenate([psi0, phin0, phip0])
        U = _project(U)

        if damping == "bank_rose":
            if generation is not None:
                raise NotImplementedError("generation= is only supported with damping='line_search'")
            # See newton_solve_2d's own docstring for why this path is
            # opt-in only (not the default) and should only be used on an
            # already-close warm start, never a cold start.
            F_fn = lambda U_: _residual(U_, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                         poisson_scale, cont_scale, qs=qs)
            FJ_fn = lambda U_: _residual_and_jacobian(U_, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                                        poisson_scale, cont_scale, qs=qs)
            U, res_norm, it, _K = bank_rose_solve(U, FJ_fn, F_fn, f_tol=f_tol, maxiter=maxiter,
                                                   step_clip_fn=lambda d: _uniform_clip(d, N),
                                                   verbose=verbose)
            return U, res_norm, it

        if damping != "line_search":
            raise ValueError(f"damping must be 'line_search' or 'bank_rose', got {damping!r}")

        F, J = _residual_and_jacobian(U, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                       poisson_scale, cont_scale, edge_mask, velocity_saturation, generation, qs=qs)
        res_norm = np.max(np.abs(F))

        # Step control: each Newton direction is scaled uniformly so no
        # potential moves by more than MAX_PSI_STEP, then backtracked on a
        # ROW-SCALED residual merit (see below) rather than the raw
        # max-norm |F|_inf this solver originally used.
        #
        # Why: continuity rows are normalized by one global Q*Dn*ni/h^2,
        # so at an n+ node (n~1e19-1e20) the same relative current
        # imbalance shows up ~1e9x larger than in the channel. A
        # quadratically convergent Newton step's O(delta^2) error at such a
        # node (measured: a step moving no potential by more than 0.54*Vt
        # raised |F|_inf from 1.3 to 1.4e5, growing exactly as t^2 in the
        # step length t) made the raw max-norm backtracking cut every step
        # to ~1e-3 - ~100-200 crawling iterations and stalls at high
        # Vgs/Vds - while the same full steps converged in 5 iterations.
        #
        # Linear solves are INEXACT (Eisenstat & Walker 1996, choice 2):
        # each Newton system is solved only to a relative residual
        # eta_k = 0.9*(merit_k/merit_{k-1})^2, capped at ETA_MAX - loose
        # while Newton is still far away (where an exact direction buys
        # nothing), tightening automatically as it converges, so the local
        # convergence rate is kept. With a reused factorization as GMRES
        # preconditioner (core/linsolve.py) a loose solve is a few cheap
        # triangular solves instead of a fresh factorization.
        #
        # The Jacobian is only assembled when another iteration will
        # actually use it: the line search already evaluated the residual
        # at the accepted point, so convergence is checked on that first.
        it = 0
        tiny_step_streak = 0
        state["converged"] = False
        best_U, best_res = U, res_norm
        merit_prev = None
        upd_prev = None
        upd_last = None
        for it in range(1, maxiter + 1):
            if res_norm < f_tol:
                break
            row_scale = 1.0 / np.maximum(_row_absmax(J), 1e-300)
            merit = np.linalg.norm(row_scale * F)
            if lin is None:
                delta = equilibrated_spsolve(J, -F)
            else:
                eta_cap = min(state.get("eta_max", ETA_MAX), lin.max_rtol(len(F)))
                eta = eta_cap if merit_prev is None else min(eta_cap, 0.9 * (merit / merit_prev) ** 2)
                delta = lin.solve(J, -F, rtol=eta)
            merit_prev = merit
            delta[N:3 * N] = np.clip(delta[N:3 * N], -MAX_QF_STEP, MAX_QF_STEP)
            max_dpsi = np.max(np.abs(delta[:N]))
            # density-weighted quasi-Fermi update (see the prediction exit below)
            n_cur, p_cur = _densities(U[:N], U[N:2 * N], U[2 * N:], ni_q, Vt, dEi_q)
            max_dqf = max(np.max(np.abs(delta[N:2 * N]) * n_cur / (n_cur + QF_DENSITY_REF)),
                          np.max(np.abs(delta[2 * N:]) * p_cur / (p_cur + QF_DENSITY_REF)))
            # Update-based convergence estimate from the FULL correction (the
            # error of the current iterate), as Sentaurus (Digits/ErrRef) and
            # DEVSIM test it: the potential's move and each carrier's relative
            # density change (largest and node mean), the density floored at UPD_DENS_REF so a carrier
            # that is essentially absent does not count.
            en = (np.abs(delta[:N] - delta[N:2 * N]) / Vt * n_cur / (n_cur + UPD_DENS_REF))[box_free]
            ep = (np.abs(delta[2 * N:] - delta[:N]) / Vt * p_cur / (p_cur + UPD_DENS_REF))[box_free]
            dn_rel, dp_rel = np.max(en, initial=0.0), np.max(ep, initial=0.0)
            upd_small = (max_dpsi < UPD_PSI_TOL and max(dn_rel, dp_rel) < UPD_DENS_TOL
                         and max(np.mean(en) if en.size else 0.0, np.mean(ep) if ep.size else 0.0) < UPD_DENS_MEAN)
            if max_dpsi > MAX_PSI_STEP:
                delta = delta * (MAX_PSI_STEP / max_dpsi)
                max_dpsi = MAX_PSI_STEP

            # Merit = L2 norm of the ROW-SCALED residual, each row divided
            # by its own largest Jacobian entry at the current iterate (held
            # fixed across the trial steps) - roughly "how many volts is
            # this node's equation from satisfied", comparable across an
            # n+ node and a depleted channel node alike.
            step = 1.0
            accepted = False
            for _ in range(20):
                U_try = _project(U + step * delta)
                F_try = _residual(U_try, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                   poisson_scale, cont_scale, edge_mask, velocity_saturation, generation, qs=qs)
                merit_try = np.linalg.norm(row_scale * F_try)
                if np.isfinite(merit_try) and merit_try < merit * (1 - 1e-4 * step):
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                U, F = U_try, F_try
            if schottky is not None:
                state["eta_max"] = ETA_MAX_SCHOTTKY if step < 1.0 else ETA_MAX
            upd = step * max_dpsi if accepted else 0.0
            upd_all = step * max(max_dpsi, max_dqf) if accepted else 0.0
            res_norm = np.max(np.abs(F))
            if res_norm < best_res:
                best_U, best_res = U, res_norm
            if verbose:
                print(f"  Newton(QF,2D) it {it}: |F|_inf={res_norm:.3e}  step={step:.3g}  "
                      f"max|dpsi|={upd:.2e}  merit={merit_try if accepted else merit:.2e}  "
                      f"dn/n={dn_rel:.1e} dp/p={dp_rel:.1e}")

            # Update-based convergence (the usual device-simulator test):
            # once Newton is in its quadratic regime the potential update
            # collapses (5e-9 V -> 1e-15 V in consecutive iterations) while
            # |F|_inf can sit on a ~1e-5 round-off floor set by the n+
            # rows' global ni normalization - f_tol=1e-9 alone then burns
            # ~20 more iterations that change nothing.
            #
            # Plus a quadratic-convergence PREDICTION: two consecutive full
            # steps in the quadratic regime (e_{k+1} ~ K e_k^2, so
            # K ~ upd/upd_prev^2) predict the next update ~ upd^3/upd_prev^2.
            # When that is already below PSI_UPDATE_TOL, the extra iteration
            # that would merely confirm it (an assembly + a solve per bias
            # point) is skipped; the returned state's error is then below
            # PSI_UPDATE_TOL instead of far below it. This exit checks |F|_inf
            # only against the loose RES_PREDICT_MAX, not RES_ACCEPT: |F|_inf is
            # dominated by the n+ rows (normalized by one global Q*Dn*ni/h^2,
            # ~1e9x inflated there), and was measured at 2e-4..6e-3 - i.e. a
            # ~1e-14 relative current imbalance at n~1e20 - on iterations
            # whose update was already 1e-7..1e-8 V with a predicted next
            # update of ~1e-15 V; requiring RES_ACCEPT cost one full extra
            # iteration on most bias points for no change in the answer.
            #
            # The prediction tracks the quasi-Fermi updates as well as psi,
            # weighted by n/(n+QF_DENSITY_REF) (p likewise) - the usual
            # device-simulator trick of discounting a carrier that is
            # essentially absent, whose quasi-Fermi level is undetermined
            # noise. Psi alone is not enough: measured on the extruded 3D
            # MOSFET's Vds=0.05V off-state anchor, a psi-only prediction
            # stopped with psi right to 4e-11 V but phin still 3e-4 V off,
            # and the off-state Id came out 1000x too large (8e-12 vs
            # 7.6e-15 A/um).
            converged = upd < PSI_UPDATE_TOL and accepted and res_norm < RES_ACCEPT
            # The update test: the correction just computed already moves
            # nothing (psi < UPD_PSI_TOL, densities < UPD_DENS_TOL / UPD_DENS_MEAN),
            # whatever |F|_inf says. |F|_inf alone stalled the nanosheet: one
            # electron row in a sheet at ~0.2 (raw, unscaled), the row-scaled
            # residual at round-off, the line search crawling at steps of
            # 1/32 - 53 of a gate step's 75 iterations changed nothing.
            # Only once Newton has stopped converging (a damped step, or an
            # update that shrank less than 10x): while it converges
            # quadratically the next iteration is cheap and the residual test
            # decides - a near-equilibrium diode current (a small difference
            # of large drift and diffusion terms) moved 0.4% when a 1e-5
            # density update was accepted mid-convergence there.
            stagnant = upd_last is not None and upd_all > 0.5 * upd_last
            converged = converged or (upd_small and (step < 1.0 or stagnant))
            upd_last = upd_all
            if (not converged and accepted and step == 1.0 and upd_prev is not None
                    and upd_all < QUAD_PREDICT_MAX and upd_prev > upd_all and res_norm < RES_PREDICT_MAX):
                converged = upd_all ** 3 / upd_prev ** 2 < PSI_UPDATE_TOL
            upd_prev = upd_all if (accepted and step == 1.0) else None
            converged = converged and it >= state.get("min_iters", 1)
            if (quantum_update is not None and accepted and not state.get("q_ok")
                    and (converged or upd_all < QUANTUM_TRIGGER)):
                # Schrodinger-Poisson corrector inside the Newton loop. Once
                # it moves Lambda by less than quantum_tol it is not called
                # again: the remaining Newton updates (< QUANTUM_TRIGGER)
                # move Lambda by less than that.
                lam_new = quantum_update({"psi": U[:N], "phin": U[N:2 * N], "phip": U[2 * N:]})
                ln, lp = (np.zeros(N) if l_ is None else np.asarray(l_, float) for l_ in lam_new)
                dlam = 0.0 if qs is None else max(np.max(np.abs(ln - qs.lam_n)), np.max(np.abs(lp - qs.lam_p)))
                # under-relax a corrector that stopped contracting (the
                # Lambda fixed point can oscillate at high drain bias)
                mix = state.get("q_mix", 1.0)
                if qs is not None and state.get("q_dlam") is not None and dlam > 0.5 * state["q_dlam"]:
                    mix = max(0.25, 0.5 * mix)
                state["q_mix"] = mix
                if qs is not None and mix < 1.0:
                    ln = qs.lam_n + mix * (ln - qs.lam_n)
                    lp = qs.lam_p + mix * (lp - qs.lam_p)
                state["q_updates"] = state.get("q_updates", 0) + 1
                state.setdefault("q_first_it", it)
                state["q_dlam"] = dlam
                state.setdefault("q_hist", []).append((it, round(float(dlam), 5)))
                state["q_ok"] = qs is not None and dlam <= quantum_tol
                if qs is None or dlam > quantum_tol:
                    qs = QuantumShift(mesh, mat, ln, lp)
                    ni_q, dEi_q = qs.ni, qs.dEi
                    converged = False
                    upd_prev = merit_prev = None
                    if it < maxiter:
                        F, J = _residual_and_jacobian(U, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                                       poisson_scale, cont_scale, edge_mask, velocity_saturation,
                                                       generation, qs=qs)
                        res_norm = np.max(np.abs(F))
                        best_U, best_res = U, res_norm
                    continue
            if converged:
                state["converged"] = True
                break
            tiny_step_streak = tiny_step_streak + 1 if step < 1e-4 else 0
            if tiny_step_streak >= 2:
                break
            if it < maxiter:
                F, J = _residual_and_jacobian(U, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                               poisson_scale, cont_scale, edge_mask, velocity_saturation,
                                               generation, qs=qs)
        if best_res < res_norm:
            U, res_norm = best_U, best_res
        return U, res_norm, it

    if psi_init is None:
        U, res_norm, it = _run_newton(*_cold_start())
    else:
        U, res_norm, it = _run_newton(psi_init, phin_init, phip_init)
        if res_norm > 1.0 and cold_retry:
            # A warm start can inherit a bad basin from its source point
            # (mirroring core/newton_solver_qf.py's own Gummel-restart
            # rationale) - retry from the fresh cold start rather than
            # accepting a stuck, wrong answer.
            U_retry, res_retry, it_retry = _run_newton(*_cold_start())
            if res_retry < res_norm:
                U, res_norm, it = U_retry, res_retry, it_retry

    if res_norm > 1.0 and cold_retry:
        warnings.warn(
            f"Newton(QF,2D) solve did not converge at bias_by_contact={bias_by_contact} "
            f"(|F|_inf={res_norm:.3e} at iteration {it}) even after a cold-start retry.")

    psi, phin, phip = unpack_qf(U, N)
    n, p = _densities(psi, phin, phip, ni_q, Vt, dEi_q)
    # converged: the update-based/predicted test passed, or |F|_inf reached
    # RES_ACCEPT (the criterion every driver used on its own before this
    # flag existed). Drivers should test this flag, not res_norm: the
    # quadratic-prediction exit can stop with |F|_inf up to RES_PREDICT_MAX.
    converged = bool(state.get("converged")) or res_norm < RES_ACCEPT
    if quantum_update is not None:          # and the quantum potentials settled
        converged = converged and bool(state.get("q_ok"))
    return {"psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
            "iters": it, "res_norm": res_norm, "converged": converged,
            "velocity_saturation": velocity_saturation, "schottky": schottky, "bias": dict(bias_by_contact),
            "qshift": qs, "q_updates": state.get("q_updates", 0), "q_dlam": state.get("q_dlam"),
            "q_first_it": state.get("q_first_it", it), "q_hist": state.get("q_hist", [])}
