"""Schottky metal/semiconductor contact: the pure-physics part, independent
of dimension and mesh (plans/schottky_plan.md). A solver turns these
pieces into boundary rows and a tunneling generation term.

The contact current has three parts (the standard Sentaurus/Atlas split):

1. Thermionic emission (TE), as a boundary condition (Crowell-Sze
   recombination-velocity form). The band at the interface is pinned to the
   metal Fermi level by the barrier: Ec(0) - E_Fm = q Phi_Bn whatever the
   bias. The flux of each carrier into the metal is

       electrons: v_n (n_s - n_B),  n_B = Nc exp(-Phi_Bn/Vt),  v_n = A*_n T^2 / (q Nc)
       holes:     v_p (p_s - p_B),  p_B = Nv exp(-Phi_Bp/Vt),  v_p = A*_p T^2 / (q Nv)

   with n_s, p_s the densities at the interface (free unknowns now).
   The barrier of the contacted region's MAJORITY carrier sets psi at
   the interface and its boundary density; the minority one follows from
   mass action (n_B p_B = ni^2), so equilibrium carries exactly zero
   current under this project's Boltzmann statistics even where
   Vt ln(Nc Nv / ni^2) differs slightly from the tabulated Eg.

2. Image-force barrier lowering, dPhi = sqrt(q E / (4 pi eps)) at the
   interface field E. It enters the TE boundary densities only, relative
   to its equilibrium value (so equilibrium stays current-free), as in
   Sentaurus' BarrierLowering.

3. Tunneling through the barrier (field and thermionic-field emission),
   nonlocal: for each carrier energy E below the barrier top, the WKB
   transmission through the SOLVED band profile U(s) along a path normal
   to the interface,

       Gamma(E) = exp(-C sqrt(m_t/m0) Int_0^{s_t} sqrt(U(s) - E) ds),
       C = 2 sqrt(2 m0 q) / hbar,

   with s_t the classical turning point, U(s_t) = E. The Tsu-Esaki current
   into the metal (A/cm^2) is

       J = (A* T^2 / Vt) Int Gamma(E) N(E) dE,
       N(E) = ln[(1 + exp((E_Fs - E)/Vt)) / (1 + exp((E_Fm - E)/Vt))],

   E in eV, E_Fs the semiconductor quasi-Fermi level at the turning point.
   Above the barrier top Gamma = 1 and the integral is the TE current, so
   only E < U(0) is tunneling (TE covers the rest; no double counting).
   The carriers leave the semiconductor at s_t: the solver books the
   current as a recombination (electrons, or holes) there.

Energies are written for ELECTRONS: U = Ec, E_F = -phi. For holes the
same routines are used with hole energies (U = -Ev, E_F = +phip): the
energy axis is flipped and everything else is identical.

U is linear between path nodes (linear psi between mesh nodes), and the
WKB integral is done exactly on each linear piece, so the transmission
needs no sub-sampling and its derivatives with respect to the nodal U are
analytic and finite (also at the turning point).
"""
from dataclasses import dataclass, replace

import numpy as np

from core.params import Q

H_BAR = 1.054571817e-34        # J s
M0 = 9.1093837015e-31          # kg
# 2 sqrt(2 m0 q)/hbar in cm^-1 eV^-1/2: Gamma = exp(-WKB_C sqrt(m_r) Int sqrt(U - E[eV]) ds[cm])
WKB_C = 2.0 * np.sqrt(2.0 * M0 * Q) / H_BAR * 1e-2


@dataclass
class SchottkyContact:
    """Parameters of one Schottky contact.

    Barrier, in order of precedence:
      barrier_n_eV / barrier_p_eV - the electron (or hole) barrier given
          directly; the other is Eg - this one.
      barrier_model "schottky_mott" - Phi_Bn = Phi_m - chi;
      barrier_model "pinned" - Cowley-Sze / MIGS form
          Phi_Bn = S (Phi_m - chi) + (1 - S) (cnl_eV - chi),
          S the pinning factor (1 = no pinning, 0 = full), cnl_eV the
          charge-neutrality level below vacuum (None = mid-gap of the
          contacted semiconductor, chi + Eg/2).
    Phi_m is workfunction_eV, or the work function of `metal` in
    core.material_db.

    richardson_n/_p: effective Richardson constants A** (A cm^-2 K^-2);
    Si defaults 112 / 32 (Crowell; Sze). m_tunnel_n/_p: WKB tunneling
    masses in m0 (Si conductivity masses 0.26 / 0.39 as defaults - fitting
    parameters in practice; set per material).
    """
    metal: str = None
    workfunction_eV: float = None
    barrier_model: str = "schottky_mott"
    pinning_S: float = 0.1
    cnl_eV: float = None
    barrier_n_eV: float = None
    barrier_p_eV: float = None
    richardson_n: float = 112.0
    richardson_p: float = 32.0
    m_tunnel_n: float = 0.26
    m_tunnel_p: float = 0.39
    barrier_lowering: bool = False
    tunneling: bool = True
    # Ti germanosilicide interlayer (Ti + SiGe react to Ti(Si,Ge)x): the hole
    # barrier falls with the contacted SiGe's Ge fraction x,
    # Phi_Bp = max(tisige_phi_p0 - tisige_slope x, tisige_floor) (eV; Si 0.40,
    # 0.24 at x = 0.2, 0.16 at 0.3, the 0.02 floor from x = 0.475: Ge pins the Fermi level near the valence band); `for_material`
    # resolves it per semiconductor into an explicit barrier_p_eV.
    interlayer: str = None
    tisige_phi_p0: float = 0.40
    tisige_slope: float = 0.80
    tisige_floor: float = 0.02

    def for_material(self, mat):
        """This contact as seen by semiconductor `mat`: with the TiSiGe
        interlayer, the hole barrier of its Ge fraction (mat.x_Ge, 0 = Si) as an
        explicit barrier_p_eV; otherwise the contact itself."""
        if self.interlayer != "TiSiGe":
            return self
        x = float(getattr(mat, "x_Ge", 0.0) or 0.0)
        return replace(self, barrier_p_eV=max(self.tisige_phi_p0 - self.tisige_slope * x, self.tisige_floor),
                       barrier_n_eV=None, interlayer=None)

    def metal_workfunction(self):
        if self.workfunction_eV is not None:
            return self.workfunction_eV
        if self.metal is None:
            raise ValueError("Schottky contact needs barrier_n_eV/barrier_p_eV, workfunction_eV or metal")
        from core import material_db
        return material_db.get(self.metal).workfunction_eV

    def barriers(self, chi_eV, Eg_eV):
        """(Phi_Bn, Phi_Bp) in eV for a semiconductor with electron affinity
        chi_eV and gap Eg_eV."""
        if self.barrier_n_eV is not None:
            return self.barrier_n_eV, Eg_eV - self.barrier_n_eV
        if self.barrier_p_eV is not None:
            return Eg_eV - self.barrier_p_eV, self.barrier_p_eV
        phi_m = self.metal_workfunction()
        if self.barrier_model == "schottky_mott":
            phi_bn = phi_m - chi_eV
        elif self.barrier_model == "pinned":
            cnl = chi_eV + Eg_eV / 2.0 if self.cnl_eV is None else self.cnl_eV
            phi_bn = self.pinning_S * (phi_m - chi_eV) + (1.0 - self.pinning_S) * (cnl - chi_eV)
        else:
            raise ValueError(f"barrier_model must be 'schottky_mott' or 'pinned', got {self.barrier_model!r}")
        return phi_bn, Eg_eV - phi_bn


def te_velocity(richardson, T, N_band):
    """Thermionic recombination velocity A* T^2 / (q N_band), cm/s."""
    return richardson * T ** 2 / (Q * N_band)


def boundary_state(contact, chi_eV, Eg_eV, Nc, Nv, ni, Vt, majority, V=0.0, dEi=0.0):
    """Interface state of a Schottky contact at metal bias V.
    majority: "n" or "p", the majority carrier of the contacted region.
    Returns dict(psi_s, n_B, p_B, phi_bn, phi_bp) - psi_s is the Dirichlet
    potential of the interface node, n_B/p_B the TE boundary densities
    (without image-force lowering). Convention n = ni exp((psi - phin + dEi)/Vt)."""
    phi_bn, phi_bp = contact.barriers(chi_eV, Eg_eV)
    if majority == "n":
        n_B = Nc * np.exp(-phi_bn / Vt)
        p_B = ni ** 2 / n_B
        psi_s = V - dEi + Vt * np.log(n_B / ni)
    elif majority == "p":
        p_B = Nv * np.exp(-phi_bp / Vt)
        n_B = ni ** 2 / p_B
        psi_s = V - dEi - Vt * np.log(p_B / ni)
    else:
        raise ValueError(f"majority must be 'n' or 'p', got {majority!r}")
    return dict(psi_s=psi_s, n_B=n_B, p_B=p_B, phi_bn=phi_bn, phi_bp=phi_bp)


def image_force_lowering(E_field, eps):
    """Image-force barrier lowering sqrt(q|E| / (4 pi eps)) in V and its
    derivative d/d|E|. eps in F/cm, E in V/cm."""
    E = np.maximum(np.abs(E_field), 1e-30)
    d = np.sqrt(Q * E / (4.0 * np.pi * eps))
    return d, d / (2.0 * E)


def energy_grid(E_top, E_bottom, dE):
    """Midpoint energies below the barrier top, at fixed offsets from E_top
    (so the grid does not move while E_bottom changes between Newton
    iterations; points without a turning point just contribute zero)."""
    k = max(int(np.ceil((E_top - E_bottom) / dE)), 0)
    return E_top - (np.arange(k) + 0.5) * dE


def wkb_integral(s, U, E):
    """Int_0^{s_t} sqrt(U(s) - E) ds for U linear between the path nodes s
    (s[0] = 0 at the interface), for each energy E (K,).

    Returns (I, dI_dU, jt, t, ok):
      I (K,)        - the integral (eV^1/2 cm); 0 where not ok;
      dI_dU (K, M)  - its derivative w.r.t. each nodal U;
      jt (K,)       - turning segment: s_t lies between nodes jt and jt+1;
      t (K,)        - fraction of that segment, s_t = s[jt] + t (s[jt+1] - s[jt]);
      ok (K,)       - E < U[0] (below the barrier top) and a turning point
                      exists on the path.
    Exact for piecewise-linear U."""
    s = np.asarray(s, dtype=float)
    U = np.asarray(U, dtype=float)
    E = np.atleast_1d(np.asarray(E, dtype=float))
    K, M = len(E), len(U)
    A = U[None, :] - E[:, None]                         # (K, M), > 0 inside the barrier
    below = A <= 0.0
    first = np.argmax(below, axis=1)                    # first node at/below E
    ok = (A[:, 0] > 0.0) & below.any(axis=1)
    jt = np.where(ok, first - 1, 0)
    L = np.diff(s)                                      # (M-1,)

    a = A[:, :-1]
    b = A[:, 1:]
    seg = np.arange(M - 1)[None, :]
    full = seg < jt[:, None]                            # segments fully inside the barrier
    turn = (seg == jt[:, None]) & ok[:, None]

    I_seg = np.zeros((K, M - 1))
    da = np.zeros((K, M - 1))
    db = np.zeros((K, M - 1))

    # Full segments: (2/3) L (a^1.5 - b^1.5)/(a - b), a, b > 0.
    af = np.where(full, a, 1.0)
    bf = np.where(full, b, 1.0)
    diff = af - bf
    close = np.abs(diff) <= 1e-9 * np.maximum(af, bf)
    dsafe = np.where(close, 1.0, diff)
    sa, sb = np.sqrt(af), np.sqrt(bf)
    Lb = np.broadcast_to(L, (K, M - 1))
    f = (2.0 / 3.0) * Lb * (af * sa - bf * sb) / dsafe
    fa = (2.0 / 3.0) * Lb * (1.5 * sa * dsafe - (af * sa - bf * sb)) / dsafe ** 2
    fb = (2.0 / 3.0) * Lb * (-1.5 * sb * dsafe + (af * sa - bf * sb)) / dsafe ** 2
    m = np.sqrt((af + bf) / 2.0)
    f = np.where(close, Lb * m, f)
    fa = np.where(close, Lb / (4.0 * m), fa)
    fb = np.where(close, Lb / (4.0 * m), fb)
    I_seg = np.where(full, f, I_seg)
    da = np.where(full, fa, da)
    db = np.where(full, fb, db)

    # Turning segment: a > 0 >= b, (2/3) L a^1.5 / (a - b).
    at = np.where(turn, a, 1.0)
    bt = np.where(turn, b, 0.0)
    dt = at - bt
    g = (2.0 / 3.0) * Lb * at ** 1.5 / dt
    ga = (2.0 / 3.0) * Lb * (1.5 * np.sqrt(at) * dt - at ** 1.5) / dt ** 2
    gb = (2.0 / 3.0) * Lb * at ** 1.5 / dt ** 2
    I_seg = np.where(turn, g, I_seg)
    da = np.where(turn, ga, da)
    db = np.where(turn, gb, db)

    I = I_seg.sum(axis=1)
    dI_dU = np.zeros((K, M))
    dI_dU[:, :-1] += da
    dI_dU[:, 1:] += db
    a_jt = A[np.arange(K), jt]
    b_jt = A[np.arange(K), np.minimum(jt + 1, M - 1)]
    t = np.where(ok, a_jt / np.where(ok, a_jt - b_jt, 1.0), 0.0)
    I = np.where(ok, I, 0.0)
    dI_dU[~ok] = 0.0
    return I, dI_dU, jt, t, ok


def supply(E, EF_s, EF_m, Vt):
    """Tsu-Esaki supply function N(E) = ln[(1 + e^{(EF_s-E)/Vt}) / (1 + e^{(EF_m-E)/Vt})]
    and dN/dEF_s. All energies in eV."""
    us = (EF_s - E) / Vt
    um = (EF_m - E) / Vt
    N = np.logaddexp(0.0, us) - np.logaddexp(0.0, um)
    dN = 0.5 * (1.0 + np.tanh(us / 2.0)) / Vt          # logistic(us)/Vt
    return N, dN


def path_tunneling(s, U, EF_nodes, EF_m, E, m_rel, richardson, T, Vt):
    """Tunneling current into the metal along one path, per energy bin.

    s (M,), U (M,): path positions (cm, s[0] = 0 at the interface) and the
    carrier's band energy there (eV, Ec for electrons, -Ev for holes);
    EF_nodes (M,): the semiconductor quasi-Fermi level (eV) at the path
    nodes; EF_m: the metal Fermi level; E (K,): energy-bin midpoints, all
    with width dE (multiply the returned densities by dE).

    Returns dict:
      j (K,)        - current density per eV, A cm^-2 eV^-1 (> 0: carriers
                      flow semiconductor -> metal);
      dj_dU (K, M)  - its derivative w.r.t. the nodal band energies;
      dj_dEF (K, 2) - w.r.t. EF at the two nodes bracketing the turning point;
      dt_dU (K, 2)  - d t / d U at those two nodes (the sink moves with t);
      jt, t, ok     - from wkb_integral (turning point location)."""
    I, dI_dU, jt, t, ok = wkb_integral(s, U, E)
    c = WKB_C * np.sqrt(m_rel)
    gamma = np.where(ok, np.exp(-c * I), 0.0)
    M = len(U)
    jt1 = np.minimum(jt + 1, M - 1)
    EF_t = (1.0 - t) * EF_nodes[jt] + t * EF_nodes[jt1]
    N, dN = supply(E, EF_t, EF_m, Vt)
    pref = richardson * T ** 2 / Vt
    j = pref * gamma * N
    K = len(E)
    rows = np.arange(K)
    dj_dU = (-c * pref * gamma * N)[:, None] * dI_dU
    # t = a/(a - b) moves with U at the two bracketing nodes, and EF_t with it.
    a = U[jt] - E
    bb = U[jt1] - E
    den = np.where(ok, (a - bb) ** 2, 1.0)
    dt_da, dt_db = -bb / den, a / den
    dj_dt = pref * gamma * dN * (EF_nodes[jt1] - EF_nodes[jt])
    np.add.at(dj_dU, (rows, jt), dj_dt * dt_da)
    np.add.at(dj_dU, (rows, jt1), dj_dt * dt_db)
    dj_dEF = (pref * gamma * dN)[:, None] * np.stack([1.0 - t, t], axis=1)
    j = np.where(ok, j, 0.0)
    dj_dU[~ok] = 0.0
    dj_dEF[~ok] = 0.0
    dt_dU = np.where(ok[:, None], np.stack([dt_da, dt_db], axis=1), 0.0)
    return dict(j=j, dj_dU=dj_dU, dj_dEF=dj_dEF, dt_dU=dt_dU, jt=jt, t=t, ok=ok, gamma=gamma)


def rho_c_depletion_model(N, phi_b, Vt, Nband, eps, richardson, T, m_rel, npts=4001, dE=1e-4):
    """Analytic reference for the specific contact resistivity (Ohm cm^2) at
    zero bias (the classic Yu / Chang-Fang-Sze calculation): the same
    Tsu-Esaki integral, but through a depletion-approximation (parabolic)
    barrier instead of a self-consistent one, plus TE over the top.
    N: majority doping magnitude, phi_b: barrier to the majority carrier,
    Nband: that carrier's band DOS. Boltzmann bulk Fermi level, as the solver.
    The self-consistent barrier has a free-carrier tail, so the solver's
    rho_c is somewhat higher (1.1x at 2e21 up to ~1.7x at 1e19, Si)."""
    xi = -Vt * np.log(N / Nband)                  # band edge - EF in the bulk
    vbb = phi_b - xi - Vt
    W = np.sqrt(2 * eps * vbb / (Q * N))
    s = np.linspace(0.0, W, npts)
    U = xi + (phi_b - xi) * (1 - s / W) ** 2      # band edge - EF, metal EF = 0
    E = np.arange(min(xi, 0.0) + dE / 2, phi_b, dE)
    I, _, _, _, ok = wkb_integral(s, U, E)
    g = np.where(ok, np.exp(-WKB_C * np.sqrt(m_rel) * I), 0.0)
    f = 0.5 * (1 + np.tanh(-E / Vt / 2)) / Vt     # dN/dEF at EF_s = EF_m = 0
    G = richardson * T ** 2 / Vt * (np.sum(g * f) * dE + np.log1p(np.exp(-phi_b / Vt)))
    return 1.0 / G
