"""Inversion-layer mobility stack (docs/mobility.md, DEVELOPMENT_LOG 83):
acoustic phonon + surface roughness (Lombardi / Darwish), Coulomb (channel
doping through the bulk model, interface charge, remote high-k charge and
phonons) and surface orientation, on any box-FV mesh the drift-diffusion
solver takes (2D triangles, 3D rectilinear and octree, and a 1D stand-in).

Low-field pipeline (newton_solver_qf_2d.low_field_mobility), per NODE:

    1/mu = 1/mu_bulk(N_tot [, n, p])                       bulk: Masetti (fast) | Klaassen (exact)
         + D * s_orient * (1/mu_ac(F) + 1/mu_sr(F))        Lombardi (fast) | Darwish (exact)
         + D / mu_ic(F, N_it, c)                           interface charge (opt-in)
         + D * D_hk * (1/mu_rcs + 1/mu_rps)                remote Coulomb / phonon (opt-in, high-k)
    D = exp(-x / l_crit),  x = distance to the nearest semiconductor/insulator interface

then the edge value = the mean of its two nodes, the static per-edge factors
(stress), the ballistic slot (another branch) and velocity saturation.

F (the normal field) is |grad psi . n_hat|, n_hat the normal of the nearest
interface node: one sparse operator per mesh (one SpMV per evaluation).
Away from an interface it is a least-squares gradient over the node's
semiconductor neighbours; ON an interface node it is the displacement flux
through the node's insulator edges divided by eps_si (D-continuity), which
does not depend on how finely the inversion layer is meshed.

The mobility depends on psi (and in exact mode on n, p). Newton lags it:
the state is recomputed from the iterate at every Jacobian assembly and
held fixed through that iteration's line search, and a solve is converged
only once the edge mobility stopped changing (SurfaceMobility.tol). The
converged state is exactly self-consistent; the Jacobian keeps its sparsity.
"""
from dataclasses import dataclass, fields

import numpy as np
import scipy.sparse as sp

# --- parameters (Si, 300 K; docs/mobility.md has the sources) ---------------------------
# Sentaurus "Enhanced Lombardi" defaults (UG N-2017.09 Table 58; ATLAS Table 3-28)
LOMBARDI = {
    "n": dict(B=4.75e7, C=5.80e2, lam=0.125, k=1.0, N0=1.0, N1=1.0, N2=1.0, delta=5.82e14, A=2.0, alpha=0.0,
              nu=1.0, eta=5.82e30),
    "p": dict(B=9.925e6, C=2.947e3, lam=0.0317, k=1.0, N0=1.0, N1=1.0, N2=1.0, delta=2.0546e14, A=2.0, alpha=0.0,
              nu=1.0, eta=2.0546e30),
}
# Darwish / "Lucent" (TED 44, 1529 (1997); Sentaurus Table 59, ATLAS Table 3-29)
DARWISH = {
    "n": dict(B=3.61e7, C=1.70e4, lam=0.0233, k=1.7, N0=1.0, N1=1.0, N2=1.0, delta=3.58e18, A=2.58,
              alpha=6.85e-21, nu=0.0767, eta=1e50),
    "p": dict(B=1.51e7, C=4.18e3, lam=0.0119, k=0.9, N0=1.0, N1=1.0, N2=1.0, delta=4.10e15, A=2.18,
              alpha=7.82e-21, nu=0.123, eta=1e50),
}
# Klaassen / Philips unified mobility (SSE 35, 953 (1992); Sentaurus Tables 56/57, As and B)
KLAASSEN = {
    "n": dict(mu_max=1417.0, mu_min=52.2, theta=2.285, N_ref=9.68e16, alpha=0.68, m=1.0),
    "p": dict(mu_max=470.5, mu_min=44.9, theta=2.247, N_ref=2.23e17, alpha=0.719, m=1.258),
}
KLAASSEN_COMMON = dict(f_CW=2.459, f_BH=3.828, c_D=0.21, N_D_ref=4e20, c_A=0.5, N_A_ref=7.2e20,
                       s=(0.89233, 0.41372, 0.19778, 0.28227, 0.005978, 1.80618, 0.72169),
                       r=(0.7643, 2.2999, 6.5502, 2.3670, -0.01552, 0.6478))
# Coulomb scattering by interface charge (Sentaurus Neg/PosInterfaceCharge, Table 63 defaults)
COULOMB_IC = dict(mu1=40.0, k=1.0, c_trans=1e18, gamma=1.5, nu1=1.0, nu2=0.5, E0=2e5, eps=2.0, N0=1e11)
# remote Coulomb (Tanimoto, SISPAD 2006) and remote phonon (Zhu & Ma, EDL 25, 89 (2004)), Tables 64/65
RCS = dict(mu0=149.0, g1=0.23187, g2=2.1, g3=0.40, g4=0.05, g5=1.0, s=0.1, c0=3e16, N_floor=1e15)
RPS = dict(mu0=496.7, g1=0.68, g2=0.34)

FAMILIES = ("100", "110", "111")
# (110) hole surface mobility: mu_ac and mu_sr x this, calibrated in the 1D MOS harness
# (scripts/mobility/universal.py) so mu_eff(110)/mu_eff(100) = 2.0 at E_eff = 0.5 MV/cm
# (Irie IEDM 2004; Takagi TED 41, 2363). Electrons: (110) FinFET sidewalls measure like
# (100) (Kavalieros 2008, Young SSE 2012): 1.0; a planar (110) wafer is ~0.6-0.8.
SCALE_110_P = 3.13
SCALE_110_N = 1.0
# surface-roughness delta x this, per mode and carrier: the classical 1D MOS fit to the
# universal curves (scripts/mobility/universal.py --calibrate-delta; docs/mobility.md)
DELTA_SCALE = {"fast": {"n": 1.0, "p": 1.0}, "exact": {"n": 1.0, "p": 1.0}}
MODES = ("off", "fast", "exact")
D_MIN = 1e-4              # nodes with D below this (x > 9 l_crit) skip the surface terms


@dataclass
class MobilityModel:
    """Options of the inversion-layer mobility (docs/mobility.md)."""
    mode: str = "fast"               # off | fast (Lombardi + Masetti) | exact (Darwish + Klaassen, A*)
    orientation: str = "auto"        # auto (per-interface (100)/(110)/(111) sets) | 100 (one set)
    interface_field: str = "dcont"   # dcont (D-continuity on interface nodes) | lsq (one-sided gradient)
    doping: str = "total"            # total (N_A + N_D where the mesh carries it) | net (|N_D - N_A|)
    interface_charge: float = 0.0    # cm^-2 (units of q, signed): fixed charge on every Si/insulator interface
    highk: bool = False              # remote Coulomb + phonon from a high-k dielectric (eps_r > highk_eps_r)
    highk_eps_r: float = 8.0
    l_crit: float = 1e-6             # cm
    l_crit_highk: float = 1e-6       # cm
    F_min: float = 1e3               # V/cm (Sentaurus EnormMinimum)
    scale_110_n: float = SCALE_110_N
    scale_110_p: float = SCALE_110_P
    surface_scale_n: float = 1.0     # calibration knobs: mu_ac and mu_sr x this
    surface_scale_p: float = 1.0
    delta_scale_n: float = None      # surface-roughness delta x this (None: the mode's calibrated default,
    delta_scale_p: float = None      # DELTA_SCALE; 1 = the published parameter set)
    smoothing: float = None          # orientation blend width (None: nearest family in fast, 0.02 in exact)
    tol: float = None                # self-consistency, max |d mu / mu| (None: 1e-4 fast, 1e-6 exact)
    frame: object = None             # device x, y, z in crystal axes (default (001) wafer, <110> channel)

    @classmethod
    def from_options(cls, opts):
        """From a config / CLI dict (or a mode string). None or mode off -> None."""
        if opts is None:
            return None
        if isinstance(opts, str):
            opts = {"mode": opts}
        o = dict(opts)
        names = {f.name for f in fields(cls)}
        alias = {"interface_charge_cm2": "interface_charge"}
        kw = {}
        for k, v in o.items():
            k = alias.get(k, k)
            if k not in names:
                raise ValueError(f"unknown mobility option {k!r} (known: {sorted(names)})")
            kw[k] = v
        m = cls(**kw)
        if m.mode not in MODES:
            raise ValueError(f"mobility mode must be one of {MODES}, not {m.mode!r}")
        if m.orientation not in ("auto", "100"):
            raise ValueError(f"orientation must be auto or 100, not {m.orientation!r}")
        if m.interface_field not in ("dcont", "lsq"):
            raise ValueError(f"interface_field must be dcont or lsq, not {m.interface_field!r}")
        if m.doping not in ("total", "net"):
            raise ValueError(f"doping must be total or net, not {m.doping!r}")
        for k in ("interface_charge", "highk_eps_r", "l_crit", "l_crit_highk", "F_min", "scale_110_n",
                  "scale_110_p", "surface_scale_n", "surface_scale_p"):
            setattr(m, k, float(getattr(m, k)))
        for c in ("n", "p"):
            setattr(m, f"delta_scale_{c}", m.delta_scale(c))
        m.highk = m.highk in (True, 1, "1", "true", "on", "yes")
        return None if m.mode == "off" else m

    @property
    def sigma(self):
        if self.smoothing is not None:
            return float(self.smoothing)
        return 0.02 if self.mode == "exact" else 0.0

    def delta_scale(self, c):
        v = getattr(self, f"delta_scale_{c}")
        if v is None:
            return DELTA_SCALE.get(self.mode, {}).get(c, 1.0)
        return float(v)

    @property
    def tolerance(self):
        if self.tol is not None:
            return float(self.tol)
        return 1e-6 if self.mode == "exact" else 1e-4


# --- the model formulas (pure functions of node arrays) ---------------------------------
def lombardi(F, N, n_plus_p, prm, T=300.0):
    """(1/mu_ac + 1/mu_sr) of the Lombardi / Darwish surface terms, F in V/cm,
    N total doping, n_plus_p for the A* exponent (Darwish; 0 = plain A)."""
    Fr = np.maximum(F, 1.0)
    mu_ac = prm["B"] / Fr + prm["C"] * ((N + prm["N2"]) / prm["N0"]) ** prm["lam"] / (
        np.cbrt(Fr) * (T / 300.0) ** prm["k"])
    A = prm["A"]
    if prm["alpha"]:
        A = A + prm["alpha"] * n_plus_p / (N + prm["N1"]) ** prm["nu"]
    inv_sr = np.exp(A * np.log(Fr)) / prm["delta"] + Fr ** 3 / prm["eta"]
    return 1.0 / mu_ac + inv_sr


def klaassen(ND, NA, n, p, carrier, T=300.0):
    """Philips unified bulk mobility (Klaassen 1992): lattice + donor/acceptor
    + carrier-carrier scattering with Brooks-Herring / Conwell-Weisskopf
    screening by the free carriers, and dopant clustering."""
    q = KLAASSEN[carrier]
    o = KLAASSEN["p" if carrier == "n" else "n"]
    c = KLAASSEN_COMMON
    t = T / 300.0
    mx, mn, al = q["mu_max"], q["mu_min"], q["alpha"]
    mu_L = mx * t ** (-q["theta"])
    mu_N = mx * mx / (mx - mn) * t ** (3 * al - 1.5)
    mu_c = mx * mn / (mx - mn) * t ** (-0.5)
    NDs = ND * (1.0 + 1.0 / (c["c_D"] + (c["N_D_ref"] / np.maximum(ND, 1.0)) ** 2))
    NAs = NA * (1.0 + 1.0 / (c["c_A"] + (c["N_A_ref"] / np.maximum(NA, 1.0)) ** 2))
    if carrier == "n":
        same, other, minority = NDs, NAs, p
    else:
        same, other, minority = NAs, NDs, n
    N_sc = np.maximum(same + other + minority, 1.0)
    cc = np.maximum(n + p, 1.0)
    m_ratio = q["m"] / o["m"]
    P_CW = 3.97e13 * (t ** 3 / N_sc) ** (2.0 / 3.0)
    P_BH = 1.36e20 / cc * q["m"] * t * t
    P = 1.0 / (c["f_CW"] / P_CW + c["f_BH"] / P_BH)
    s1, s2, s3, s4, s5, s6, s7 = c["s"]
    G = 1.0 - s1 / (s2 + (t / q["m"]) ** s4 * P) ** s3 + s5 / ((q["m"] / t) ** s7 * P) ** s6
    r1, r2, r3, r4, r5, r6 = c["r"]
    Pr = P ** r6
    Fp = (r1 * Pr + r2 + r3 * m_ratio) / (Pr + r4 + r5 * m_ratio)
    N_eff = np.maximum(same + G * other + minority / Fp, 1.0)
    mu_D = mu_N * (N_sc / N_eff) * (q["N_ref"] / N_sc) ** al + mu_c * cc / N_eff
    return 1.0 / (1.0 / mu_L + 1.0 / mu_D)


def interface_coulomb(F, Nit, c, T=300.0, prm=COULOMB_IC):
    """1/mu of Coulomb scattering by an interface charge Nit (cm^-2, |.|),
    screened by the carrier density c (Sentaurus' interface-charge component,
    as read for the plan; f(F) switches it off at flat band)."""
    x = np.maximum(np.abs(Nit), 1.0) / prm["N0"]
    f = -np.expm1(-(np.maximum(F, 1.0) / prm["E0"]) ** prm["eps"])
    mu = prm["mu1"] * (T / 300.0) ** prm["k"] / x ** prm["nu1"] * (
        1.0 + (c / (prm["c_trans"] * x ** prm["nu2"])) ** prm["gamma"])
    return f / mu


def remote_highk(F, N, c, T=300.0):
    """1/mu_rcs + 1/mu_rps of the remote Coulomb (Tanimoto) and remote
    phonon (Zhu & Ma) terms; N floored at RCS['N_floor']."""
    r = RCS
    x = np.maximum(N, r["N_floor"]) / 3e16
    g = 1.0 + r["s"] * (c / r["c0"]) / x ** r["g5"]
    mu_rcs = r["mu0"] * x ** r["g1"] * (T / 300.0) ** r["g2"] * g ** (r["g3"] + r["g4"] * np.log(x))
    mu_rps = RPS["mu0"] * (np.maximum(F, 1.0) / 1e6) ** (-RPS["g1"]) * (T / 300.0) ** (-RPS["g2"])
    return 1.0 / mu_rcs + 1.0 / mu_rps


def family_cosines(nrm, R=None):
    """(K, 3) |cos| of each device-frame normal against the <100>, <110>,
    <111> families (FAMILIES order)."""
    from mechanics.elastic import FRAME_001_110, frame
    R = frame(FRAME_001_110 if R is None else R)
    v = np.zeros((len(nrm), 3))
    v[:, :nrm.shape[1]] = nrm
    c = np.sort(np.abs(v @ R), axis=1)[:, ::-1]            # crystal components, largest first
    c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-300)
    return np.column_stack([c[:, 0], (c[:, 0] + c[:, 1]) / np.sqrt(2.0), c.sum(axis=1) / np.sqrt(3.0)])


def orientation_weights(nrm, sigma=0.0, R=None):
    """(K, 3) family weights: one-hot on the best family (sigma 0) or the
    smooth blend w_k ~ exp(-(1 - |cos_k|) / sigma)."""
    cs = family_cosines(nrm, R)
    if not sigma:
        w = np.zeros_like(cs)
        w[np.arange(len(cs)), np.argmax(cs, axis=1)] = 1.0
        return w
    w = np.exp(-(1.0 - cs) / sigma)
    return w / w.sum(axis=1, keepdims=True)


# --- geometry: interfaces, distances, normals, the normal-field operator ------------------
def _node_eps(mesh, mat):
    from solver2d.newton_solver_qf_2d import node_materials
    mats, idx = node_materials(mesh, mat)
    eps = np.array([m.eps for m in mats])
    return np.full(len(mesh.points), mat.eps) if idx is None else eps[idx]


def interface_geometry(mesh, mat, l_crit=1e-6, interface_field="dcont", highk_eps_r=8.0):
    """Static, solve-independent geometry of the surface terms (dict):
    interface nodes and their normals, each semiconductor node's distance
    and normal (nearest interface node), D, the normal-field operator, the
    interface area per node (for a sheet charge) and the distance to the
    nearest high-k insulator node."""
    from scipy.spatial import cKDTree
    P = np.asarray(mesh.points, float)
    if P.ndim == 1:
        P = P[:, None]
    N, dim = P.shape
    ins = np.zeros(N, bool) if getattr(mesh, "is_insulator", None) is None else np.asarray(mesh.is_insulator)
    semi = ~ins
    # metal-surface nodes of the semiconductor contacts (S/D metal volumes in an ILD)
    # are insulator nodes but not a dielectric: a gate-like contact (all its nodes
    # insulator) does count - its nodes sit on the gate dielectric
    btype = np.asarray(mesh.boundary_bc_type, dtype=object)
    is_c = np.array([str(b).startswith("contact:") for b in btype], bool)
    bidx = np.asarray(mesh.boundary_point_index, int)[is_c]
    bnm = [str(b).split(":", 1)[1] for b in btype[is_c]]
    semi_contacts = {nm for nm, i in zip(bnm, bidx) if semi[i]}
    metal = np.zeros(N, bool)
    for nm, i in zip(bnm, bidx):
        if nm in semi_contacts and ins[i]:
            metal[i] = True
    diel = ins & ~metal
    E = np.asarray(mesh.edges)
    ii, jj = E[:, 0], E[:, 1]
    e1, e2 = semi[ii] & diel[jj], semi[jj] & diel[ii]
    a = np.concatenate([ii[e1], jj[e2]])
    b = np.concatenate([jj[e1], ii[e2]])
    eid = np.concatenate([np.flatnonzero(e1), np.flatnonzero(e2)])
    fac = np.asarray(mesh.facet_length, float)[eid]
    d = P[b] - P[a]
    L = np.linalg.norm(d, axis=1)
    t = d / L[:, None]
    v = np.zeros((N, dim))
    for k in range(dim):
        v[:, k] = -np.bincount(a, fac * t[:, k], minlength=N)
    vn = np.linalg.norm(v, axis=1)
    is_if = np.zeros(N, bool)
    is_if[a] = True
    # a degenerate sum (insulator on opposite sides, a one-cell sliver): the largest facet's edge
    weak = is_if & (vn < 1e-9 * np.maximum(np.bincount(a, fac, minlength=N), 1e-300))
    if weak.any():
        o = np.lexsort((-fac, a))
        first = np.unique(a[o], return_index=True)
        for node, k in zip(*first):
            if weak[node]:
                v[node] = -t[o[k]]
        vn = np.linalg.norm(v, axis=1)
    nrm = np.zeros((N, dim))
    nrm[is_if] = v[is_if] / vn[is_if, None]
    s = np.einsum("ek,ek->e", t, nrm[a])                     # < 0: the edge points into the insulator
    area = np.bincount(a, fac * np.maximum(-s, 0.0), minlength=N)

    geo = dict(N=N, dim=dim, semi=semi, is_interface=is_if, area=area, n_interface=int(is_if.sum()))
    dist = np.full(N, np.inf)
    node_nrm = np.zeros((N, dim))
    near = np.full(N, -1)
    if is_if.any():
        ifn = np.flatnonzero(is_if)
        tree = cKDTree(P[ifn])
        sn = np.flatnonzero(semi)
        dd, k = tree.query(P[sn])
        dist[sn], near[sn] = dd, ifn[k]
        node_nrm[sn] = nrm[ifn[k]]
    D = np.where(semi, np.exp(-dist / l_crit), 0.0)
    act = semi & (D > D_MIN)
    geo.update(dist=dist, normal=node_nrm, nearest=near, D=D, active=act)

    # the normal-field operator: rows = active nodes, (Fop @ psi) = d psi / d n (V/cm)
    m = semi[ii] & semi[jj]
    src = np.concatenate([ii[m], jj[m]])
    dst = np.concatenate([jj[m], ii[m]])
    dv = P[dst] - P[src]
    w = 1.0 / np.einsum("ek,ek->e", dv, dv)
    A = np.zeros((N, dim, dim))
    for p_ in range(dim):
        for q_ in range(dim):
            A[:, p_, q_] = np.bincount(src, w * dv[:, p_] * dv[:, q_], minlength=N)
    sel = act[src]
    src, dst, dv, w = src[sel], dst[sel], dv[sel], w[sel]
    Ainv = np.zeros_like(A)
    an = np.flatnonzero(act)
    if len(an):
        Ainv[an] = np.linalg.pinv(A[an], rcond=1e-8)
    coef = w * np.einsum("ek,ekl,el->e", node_nrm[src], Ainv[src], dv)
    rows, cols, vals = [src, src], [dst, src], [coef, -coef]
    use_dc = np.zeros(N, bool)
    if interface_field == "dcont" and len(a):
        g = np.asarray(mesh.edge_g, float)[eid]
        into = (s < -0.2) & act[a] & (fac > 0)
        S = np.bincount(a[into], fac[into] * s[into] ** 2, minlength=N)
        use_dc = (S > 0) & is_if
        eps_n = _node_eps(mesh, mat)
        ai, bi = a[into], b[into]
        k_ = use_dc[ai]
        ai, bi = ai[k_], bi[k_]
        cdc = -g[into][k_] * s[into][k_] / (eps_n[ai] * S[ai])
        keep = ~use_dc[np.concatenate(rows[:1])]          # drop the LSQ rows of D-continuity nodes
        rows = [src[keep], src[keep], ai, ai]
        cols = [dst[keep], src[keep], bi, ai]
        vals = [coef[keep], -coef[keep], cdc, -cdc]
    rr, cc, vv = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    row_of = np.full(N, -1)
    row_of[an] = np.arange(len(an))
    geo["Fop"] = sp.csr_matrix((vv, (row_of[rr], cc)), shape=(len(an), N))
    geo["dcont"] = use_dc
    # distance to the nearest high-k insulator node (an incident edge with eps_r > highk_eps_r)
    from core.params import EPS0
    fl = np.asarray(mesh.facet_length, float)
    eg = getattr(mesh, "edge_g", None)
    dist_hk = np.full(N, np.inf)
    if eg is not None and ins.any():
        Le = np.linalg.norm(P[jj] - P[ii], axis=1)
        eps_e = np.where(fl > 0, np.asarray(eg, float) * Le / np.where(fl > 0, fl, 1.0), 0.0) / EPS0
        hk_e = (eps_e > highk_eps_r) & ins[ii] & ins[jj]
        hk = np.zeros(N, bool)
        hk[ii[hk_e]] = hk[jj[hk_e]] = True
        hk &= diel
        if hk.any():
            sn = np.flatnonzero(semi)
            dist_hk[sn] = cKDTree(P[hk]).query(P[sn])[0]
    geo["dist_highk"] = dist_hk
    return geo


class MobilityState:
    """The lagged low-field mobility of one iterate: per-edge mu_n0, mu_p0
    (before the static per-edge factors) and the nodal normal field."""
    __slots__ = ("mu_n", "mu_p", "F", "weight")

    def __init__(self, mu_n, mu_p, F, weight):
        self.mu_n, self.mu_p, self.F, self.weight = mu_n, mu_p, F, weight

    def rel_change(self, other):
        """max |d mu / mu| between two states over the edges that matter
        (surface-affected carrier edges)."""
        if other is None or not self.weight.any():
            return 0.0
        w = self.weight
        dn = np.abs(other.mu_n[w] - self.mu_n[w]) / self.mu_n[w]
        dp = np.abs(other.mu_p[w] - self.mu_p[w]) / self.mu_p[w]
        return float(max(dn.max(), dp.max()))


class SurfaceMobility:
    """The inversion-layer mobility of one mesh (attach_mobility puts it on
    mesh.surface_mobility). Static parts are built once; state() evaluates
    the solution-dependent mobility (one SpMV + elementwise terms)."""

    def __init__(self, mesh, mat, model):
        from solver2d.newton_solver_qf_2d import _semiconductor_edge_mask, node_materials
        from solver2d.stress_mobility import silicon_nodes
        self.model = model
        self.T = float(getattr(mat, "T", 300.0))
        g = interface_geometry(mesh, mat, model.l_crit, model.interface_field, model.highk_eps_r)
        self.geo = g
        N = g["N"]
        self.N = N
        self.semi = g["semi"]
        self.act = np.flatnonzero(g["active"])
        self.D = g["D"]
        net = np.asarray(mesh.Cdop, float)
        tot = getattr(mesh, "Ntot", None)
        if model.doping == "total" and tot is not None:
            Nt = np.maximum(np.asarray(tot, float), np.abs(net))
        else:
            Nt = np.abs(net)
        self.Ntot = Nt
        self.ND = 0.5 * (Nt + net)
        self.NA = 0.5 * (Nt - net)
        # bulk: static (Masetti / Caughey-Thomas per node material) from the chosen doping
        self.mu_b = bulk_nodal(mesh, mat, Nt)
        self.si = silicon_nodes(mesh, mat) & self.semi
        mats, _ = node_materials(mesh, mat)
        self.klaassen = model.mode == "exact"
        base = DARWISH if model.mode == "exact" else LOMBARDI
        self.sets = {c: dict(base[c], delta=base[c]["delta"] * model.delta_scale(c)) for c in ("n", "p")}
        # orientation: per active node, the surface-term multiplier sum_k w_k / scale_k
        a = self.act
        if model.orientation == "auto" and len(a):
            W = orientation_weights(g["normal"][a], model.sigma, model.frame)
        else:
            W = np.zeros((len(a), 3))
            W[:, 0] = 1.0
        self.W = W
        sc = {"n": np.array([1.0, model.scale_110_n, 1.0]), "p": np.array([1.0, model.scale_110_p, 1.0])}
        self.orient = {c: (W / sc[c]).sum(axis=1) / getattr(model, f"surface_scale_{c}") for c in ("n", "p")}
        self.family = np.argmax(W, axis=1) if len(a) else np.zeros(0, int)
        Da = self.D[a]
        self.Da = Da
        self.Na = Nt[a]
        self.ic = None
        if model.interface_charge:
            self.ic = abs(model.interface_charge)
        self.Dhk = None
        if model.highk:
            dk = np.exp(-g["dist_highk"][a] / model.l_crit_highk)
            self.Dhk = np.where(np.isfinite(g["dist_highk"][a]), dk, 0.0)
        ii, jj = np.asarray(mesh.edges)[:, 0], np.asarray(mesh.edges)[:, 1]
        self.ii, self.jj = ii, jj
        carrier_edge = _semiconductor_edge_mask(mesh) & (np.asarray(
            getattr(mesh, "facet_length_semi", None) if getattr(mesh, "facet_length_semi", None) is not None
            else mesh.facet_length) > 0)
        self.weight = carrier_edge & (0.5 * (self.D[ii] + self.D[jj]) > 0.1)
        self.tol = model.tolerance

    # -- evaluation -------------------------------------------------------------------
    def normal_field(self, psi):
        F = np.full(self.N, self.model.F_min)
        if len(self.act):
            g = self.geo["Fop"] @ np.asarray(psi, float)
            F[self.act] = np.sqrt(g * g + self.model.F_min ** 2)
        return F

    def nodal(self, mat, psi, n=None, p=None):
        """(mu_n, mu_p, F) per node."""
        T = self.T
        F = self.normal_field(psi)
        mu_n_b, mu_p_b = self.mu_b
        if self.klaassen and n is not None:
            s = self.si
            mu_n_b, mu_p_b = mu_n_b.copy(), mu_p_b.copy()
            mu_n_b[s] = klaassen(self.ND[s], self.NA[s], n[s], p[s], "n", T)
            mu_p_b[s] = klaassen(self.ND[s], self.NA[s], n[s], p[s], "p", T)
        inv_n = 1.0 / mu_n_b
        inv_p = 1.0 / mu_p_b
        a = self.act
        if len(a):
            Fa = F[a]
            na = n[a] if n is not None else np.zeros(len(a))
            pa = p[a] if p is not None else np.zeros(len(a))
            npa = na + pa
            add_n = self.orient["n"] * lombardi(Fa, self.Na, npa, self.sets["n"], T)
            add_p = self.orient["p"] * lombardi(Fa, self.Na, npa, self.sets["p"], T)
            if self.ic is not None:
                add_n = add_n + interface_coulomb(Fa, self.ic, na, T)
                add_p = add_p + interface_coulomb(Fa, self.ic, pa, T)
            if self.Dhk is not None:
                add_n = add_n + self.Dhk * remote_highk(Fa, self.Na, na, T)
                add_p = add_p + self.Dhk * remote_highk(Fa, self.Na, pa, T)
            inv_n = inv_n.copy()
            inv_p = inv_p.copy()
            inv_n[a] += self.Da * add_n
            inv_p[a] += self.Da * add_p
        return 1.0 / inv_n, 1.0 / inv_p, F

    def state(self, mat, psi, n=None, p=None):
        mu_n, mu_p, F = self.nodal(mat, psi, n, p)
        ii, jj = self.ii, self.jj
        return MobilityState(0.5 * (mu_n[ii] + mu_n[jj]), 0.5 * (mu_p[ii] + mu_p[jj]), F, self.weight)

    def report(self):
        g = self.geo
        fam = np.bincount(self.family, minlength=3) if len(self.family) else np.zeros(3, int)
        on_if = g["is_interface"][self.act] if len(self.act) else np.zeros(0, bool)
        fam_if = np.bincount(self.family[on_if], minlength=3) if on_if.any() else np.zeros(3, int)
        return dict(mode=self.model.mode, orientation=self.model.orientation, n_interface=g["n_interface"],
                    n_active=int(len(self.act)), dcont_nodes=int(g["dcont"].sum()),
                    families_interface={f: int(c) for f, c in zip(FAMILIES, fam_if)},
                    families_active={f: int(c) for f, c in zip(FAMILIES, fam)},
                    interface_charge=self.model.interface_charge, highk=self.model.highk,
                    highk_nodes_reachable=int(np.isfinite(g["dist_highk"]).any()))


def bulk_nodal(mesh, mat, N_abs):
    """The static doping-dependent bulk mobility per node (each node its own
    material's Masetti / Caughey-Thomas parameters) at the doping N_abs."""
    from solver2d.newton_solver_qf_2d import _mobility_from_doping, node_materials
    mats, idx = node_materials(mesh, mat)
    if idx is None:
        return _mobility_from_doping(mat, N_abs)
    mu_n = np.empty(len(N_abs))
    mu_p = np.empty(len(N_abs))
    for k, m in enumerate(mats):
        on = idx == k
        if on.any():
            a, b = _mobility_from_doping(m, N_abs)
            mu_n[on], mu_p[on] = a[on], b[on]
    return mu_n, mu_p


def attach_mobility(mesh, mat, model, log=None):
    """Put the inversion-layer mobility on the mesh (model a MobilityModel,
    an options dict or a mode string; None / off removes it) and, with an
    interface charge, the fixed sheet charge on the interface nodes
    (mesh.fixed_charge, read by the Poisson rows). Attach before the
    equilibrium solve: the interface charge is electrostatics. Returns the
    report dict or None."""
    if not isinstance(model, MobilityModel):
        model = MobilityModel.from_options(model)
    if model is None:
        mesh.surface_mobility = None
        mesh.fixed_charge = None
        return None
    sm = SurfaceMobility(mesh, mat, model)
    mesh.surface_mobility = sm
    mesh.fixed_charge = model.interface_charge * sm.geo["area"] if model.interface_charge else None
    rep = sm.report()
    if log:
        log(f"inversion-layer mobility ({rep['mode']}, orientation {rep['orientation']}): "
            f"{rep['n_interface']} interface nodes {rep['families_interface']}, {rep['n_active']} nodes in reach"
            + (f", interface charge {model.interface_charge:g} cm^-2" if model.interface_charge else "")
            + (", remote high-k terms" if model.highk else ""))
    return rep
