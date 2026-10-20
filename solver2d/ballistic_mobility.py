"""Ballistic (channel-length) mobility: the quasi-ballistic correction of
the drift-diffusion mobility in short channels (docs/ballistic.md,
DEVELOPMENT_LOG 85). Default OFF.

    1/mu0 = 1/mu_low + w / (gamma^s * mu_bal)          per edge, low field
    mu_bal = q v_T L_ch / (2 k T),   v_T = sqrt(2 k T / (pi m*))

- mu_low is the low-field mobility after the doping (and later surface /
  Coulomb) terms and the stress factor gamma (low_field_mobility steps 1-4
  in solver2d/newton_solver_qf_2d.py); this is step 5, before velocity
  saturation (the Sentaurus order bulk -> surface -> stress -> ballistic
  -> high field; Sentaurus "DopingDependence(BalMob)" / Shur, IEEE EDL 23,
  511 (2002)).
- v_T is the unidirectional thermal (injection) velocity of a
  non-degenerate 2D gas, m* the transport (conductivity) mass along the
  channel: 0.19 m0 for electrons (the Delta-2 valleys of (001) Si, m_t),
  0.40 m0 for unstrained holes. k = mu_bal / L_ch = 23.9 / 16.4
  cm^2/(V s nm). Sentaurus' default is k = 20 for both.
- The identity behind it (Lundstrom, EDL 18, 361 (1997)): the
  backscattering coefficient r = L / (L + lambda) with the mean free path
  lambda = 2 (kT/q) mu / v_T gives the linear-regime apparent mobility
  1/mu_app = 1/mu + 1/mu_bal exactly. It only ever LOWERS the mobility.
- gamma is the stress factor of the edge (mesh.edge_mu0_factors['stress']): a mass-
  driven mobility gain gamma raises v_T by gamma^(1/2) only (Khakifirooz /
  Antoniadis: hole ballistic velocity gain ~2x for ~4x mobility), so
  mu_bal is scaled by gamma^s, s = 0.5 by default.
- w in [0, 1] is the channel mask: the gated channel (x within the gate,
  smoothed over `smooth_nm`) of low doping (|N| below n_max_cm3, smooth in
  log N). The S/D and the deep body are not ballistic channels.

mu_bal does not depend on the solution: the factor is computed once per
mesh (set_ballistic_mobility) and the Newton Jacobian, including the
velocity-saturation partials, which take mu0 as a per-edge constant, is
unchanged.
"""
from dataclasses import dataclass

import numpy as np

Q = 1.602176634e-19
KB = 1.380649e-23
M0 = 9.1093837015e-31
NM_PER_CM = 1e7

# transport masses (m0) behind the default k: electrons m_t of the (001)
# Delta-2 valleys; unstrained Si holes (heavy/light hole mix along <110>)
M_STAR = {"n": 0.19, "p": 0.40}
PLACEMENTS = ("lowfield", "total")
STRESS_FACTOR = "stress"           # the set_edge_mobility_factor name of the piezo factor


def thermal_velocity(m_rel, T=300.0):
    """Unidirectional thermal velocity (cm/s) of a non-degenerate 2D gas,
    v_T = sqrt(2 k T / (pi m*))."""
    return np.sqrt(2.0 * KB * T / (np.pi * m_rel * M0)) * 100.0


def ballistic_k(m_rel, T=300.0):
    """k = mu_bal / L_ch in cm^2/(V s nm): q v_T / (2 k T) per nm."""
    return thermal_velocity(m_rel, T) / (2.0 * KB * T / Q) / NM_PER_CM


def ballistic_mobility(L_nm, m_rel, T=300.0):
    """mu_bal (cm^2/Vs) of a channel L_nm long."""
    return ballistic_k(m_rel, T) * L_nm


def matthiessen(mu_low, mu_bal, w=1.0):
    """1 / (1/mu_low + w/mu_bal), elementwise; w = 0 or mu_bal = inf gives mu_low."""
    return 1.0 / (1.0 / mu_low + w / mu_bal)


@dataclass
class BallisticModel:
    L_nm: float = None             # channel length (nm); None: the device gate length
    k_n: float = None              # cm^2/(V s nm); None: from the mass (ballistic_k)
    k_p: float = None
    m_n: float = M_STAR["n"]
    m_p: float = M_STAR["p"]
    T: float = 300.0
    stress_exp: float = 0.5        # s: mu_bal scaled by gamma^s
    n_max_cm3: float = 3e19        # channel mask: |doping| above this is S/D, not channel
    smooth_nm: float = 1.0         # channel mask: tanh width at the gate edges
    margin_nm: float = 0.0         # channel mask: extends the gate window on each side
    placement: str = "lowfield"    # "lowfield": before velocity saturation (Matthiessen into mu0);
                                   # "total": on the field-dependent DD mobility (Sentaurus Eq. 367)

    @classmethod
    def from_options(cls, opts):
        o = dict(opts or {})
        o.pop("enabled", None)
        if "k" in o:
            k = o.pop("k")
            o.setdefault("k_n", k)
            o.setdefault("k_p", k)
        bad = set(o) - set(cls.__dataclass_fields__)
        if bad:
            raise ValueError(f"unknown ballistic option(s) {sorted(bad)}")
        m = cls(**{k: (v if k == "placement" else float(v) if v is not None else None) for k, v in o.items()})
        if m.placement not in PLACEMENTS:
            raise ValueError(f"ballistic placement must be one of {PLACEMENTS}, not {m.placement!r}")
        return m

    def k(self, c):
        k = self.k_n if c == "n" else self.k_p
        return k if k is not None else ballistic_k(self.m_n if c == "n" else self.m_p, self.T)


def channel_mask(mesh, gate_x_nm, model):
    """(E,) weight w in [0, 1]: 1 on edges in the gated, low-doped channel.
    gate_x_nm = (x0, x1) of the gate along the channel (x, nm)."""
    P = np.asarray(mesh.points, float) * NM_PER_CM
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    xm = 0.5 * (P[ii, 0] + P[jj, 0])
    x0, x1 = gate_x_nm[0] - model.margin_nm, gate_x_nm[1] + model.margin_nm
    s = max(model.smooth_nm, 1e-6)
    wx = 0.25 * (1.0 + np.tanh((xm - x0) / s)) * (1.0 - np.tanh((xm - x1) / s))
    N = 0.5 * (np.abs(mesh.Cdop[ii]) + np.abs(mesh.Cdop[jj]))
    wn = 0.5 * (1.0 - np.tanh(np.log10(np.maximum(N, 1.0) / model.n_max_cm3) / 0.1))
    w = wx * wn
    ins = getattr(mesh, "is_insulator", None)
    if ins is not None:
        w = np.where(ins[ii] | ins[jj], 0.0, w)
    return w


def gate_window_nm(mesh, gate="gate", L_nm=None):
    """(x0, x1) nm of the gate: centred on the gate contact's nodes, L_nm
    long (else the contact nodes' x extent)."""
    bc = np.asarray(mesh.boundary_bc_type).astype(str)
    idx = np.asarray(mesh.boundary_point_index)[bc == f"contact:{gate}"]
    if not len(idx):
        raise ValueError(f"no contact {gate!r} on the mesh")
    x = np.asarray(mesh.points, float)[idx, 0] * NM_PER_CM
    if L_nm is None:
        return float(x.min()), float(x.max())
    xc = 0.5 * (x.min() + x.max())
    return xc - 0.5 * L_nm, xc + 0.5 * L_nm


def set_ballistic_mobility(mesh, model=None, gate_x_nm=None, L_nm=None, log=None):
    """Attach the ballistic term to the mesh (model None: remove it).
    L_nm: the channel length (model.L_nm wins); gate_x_nm: the gate window
    (default: from the gate contact, gate_window_nm). Sets
    mesh.edge_mu_ballistic = dict(inv_n, inv_p, s) with inv_c = w/mu_bal,c
    per edge (0 off the channel). Returns a report."""
    if model is None:
        mesh.edge_mu_ballistic = None
        return None
    L = model.L_nm if model.L_nm is not None else L_nm
    if L is None or not L > 0:
        raise ValueError("ballistic mobility needs the channel length (L_nm)")
    if gate_x_nm is None:
        gate_x_nm = gate_window_nm(mesh, L_nm=L)
    w = channel_mask(mesh, gate_x_nm, model)
    mu_n, mu_p = model.k("n") * L, model.k("p") * L
    mesh.edge_mu_ballistic = dict(inv_n=w / mu_n, inv_p=w / mu_p, s=model.stress_exp, placement=model.placement)
    rep = dict(L_nm=float(L), k_n=model.k("n"), k_p=model.k("p"), mu_bal_n=mu_n, mu_bal_p=mu_p,
               gate_x_nm=tuple(float(v) for v in gate_x_nm), n_channel_edges=int((w > 0.5).sum()),
               placement=model.placement)
    if log:
        log(f"ballistic mobility: L_ch {L:g} nm, mu_bal n/p = {mu_n:.0f}/{mu_p:.0f} cm2/Vs "
            f"(k {rep['k_n']:.1f}/{rep['k_p']:.1f} per nm), gate x {gate_x_nm[0]:.1f}..{gate_x_nm[1]:.1f} nm, "
            f"{rep['n_channel_edges']} channel edges")
    return rep


def _inverse(mesh, b):
    """Per-edge (w/(gamma^s mu_bal) n, p)."""
    inv_n, inv_p, s = b["inv_n"], b["inv_p"], b["s"]
    # the stress factor alone (not other static factors, e.g. a later orientation one)
    fac = (getattr(mesh, "edge_mu0_factors", None) or {}).get(STRESS_FACTOR)
    if fac is not None and s:
        inv_n = inv_n * fac[0] ** (-s)
        inv_p = inv_p * fac[1] ** (-s)
    return inv_n, inv_p


def apply_ballistic(mesh, mu_n0, mu_p0):
    """Step 5 of low_field_mobility (placement "lowfield"): 1/mu0 = 1/mu_low
    + w/(gamma^s mu_bal). Without a ballistic term on the mesh, or with the
    "total" placement (apply_ballistic_total does it): untouched, bit for bit."""
    b = getattr(mesh, "edge_mu_ballistic", None)
    if b is None or b.get("placement", "lowfield") != "lowfield":
        return mu_n0, mu_p0
    inv_n, inv_p = _inverse(mesh, b)
    return 1.0 / (1.0 / mu_n0 + inv_n), 1.0 / (1.0 / mu_p0 + inv_p)


def apply_ballistic_total(mesh, mu_n, mu_p, dmu_n=None, dmu_p=None):
    """Placement "total" (Sentaurus Eq. 367, Mobility(BalMob)): the ballistic
    term combined with the field-dependent drift-diffusion mobility,
    mu = 1/(1/mu_DD(F) + w/(gamma^s mu_bal)); the partials of mu_DD scale by
    (mu/mu_DD)^2. Returns (mu_n, mu_p, dmu_n, dmu_p), untouched (the same
    objects) unless the mesh has a "total" ballistic term."""
    b = getattr(mesh, "edge_mu_ballistic", None)
    if b is None or b.get("placement", "lowfield") != "total":
        return mu_n, mu_p, dmu_n, dmu_p
    inv_n, inv_p = _inverse(mesh, b)
    out_n = 1.0 / (1.0 / mu_n + inv_n)
    out_p = 1.0 / (1.0 / mu_p + inv_p)
    if dmu_n is not None:
        sn, sp = (out_n / mu_n) ** 2, (out_p / mu_p) ** 2
        dmu_n = tuple(sn * d for d in dmu_n)
        dmu_p = tuple(sp * d for d in dmu_p)
    return out_n, out_p, dmu_n, dmu_p
