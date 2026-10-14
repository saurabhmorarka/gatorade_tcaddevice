"""1D prototypes of the Schrodinger-Poisson channel charge
(plans/quantum_plan.md, phase P1):

  - bulk MOS capacitor (mos/: metal gate | oxide | substrate), one line
    from the gate oxide to depth D_q;
  - symmetric double gate (gate | oxide | body | oxide | gate), one line
    wall to wall: the 1D version of a FinFET fin or a nanosheet.

Both run the classical (Poisson + Boltzmann) and the quantum solve through
the same code path; quantum=None is classical.
"""
from dataclasses import dataclass

import numpy as np

from core import physics as ph
from core.params import EPS0
from mos.mos_solver import solve_mos_equilibrium, semiconductor_charge
from quantum.coupling import sp_loop
from quantum.lines import line_1d, quantum_potential


@dataclass
class QuantumConfig:
    carrier: str = "n"                 # the inversion carrier
    depth: float = 20e-7               # D_q, cm: line length into a bulk body
    t_pen: float = 1e-7                # line extension into the oxide, cm
    h: float = 1e-8                    # line spacing, cm
    n_states: int = 20                 # levels per valley
    statistics: str = "boltzmann"      # or "fermi" (2D Fermi-Dirac subbands)
    normal: tuple = (0.0, 1.0, 0.0)    # confinement direction, crystal axes
    taper_frac: float = 0.25
    tol: float = 1e-3                  # V, outer loop
    hard_wall: bool = False            # True: no oxide penetration (t_pen = 0)


def node_volumes(x):
    """The 1D charge volume of each node, as physics.solve_poisson uses it."""
    h = np.diff(x)
    V = np.empty(len(x))
    V[1:-1] = 0.5 * (h[:-1] + h[1:])
    V[0], V[-1] = 0.5 * h[0], 0.5 * h[-1]
    return V


def _shift(lam, carrier):
    return (lam, None) if carrier == "n" else (None, lam)


def _line_update(line, x, phi, qc, mat):
    def update(state):
        psi = state["psi"]
        lam, info = quantum_potential(line, np.interp(line.s, x, psi), psi[line.dd_idx], phi[line.dd_idx],
                                      qc.carrier, mat, qc.normal, qc.n_states, qc.statistics)
        full = np.zeros(len(x))
        full[line.dd_idx] = lam
        q = np.zeros(len(x))
        q[line.dd_idx] = info["q"]
        state["line_info"] = info
        return full, q
    return update


def _sheet(x, n, V, mask):
    return float(np.sum((n * V)[mask]))


def _centroid(x, n, V, mask, x0):
    w = (n * V)[mask]
    return float(np.sum(w * np.abs(x[mask] - x0)) / max(np.sum(w), 1e-300))


# --- bulk MOS capacitor -----------------------------------------------------------------
def mos_line(g, qc):
    x = g["x"]
    t_pen = 0.0 if qc.hard_wall else qc.t_pen
    return line_1d(x, ~g["is_oxide"], node_volumes(x), 0.0, x[-1], depth=qc.depth, t_pen=t_pen,
                   h=qc.h, taper_frac=qc.taper_frac)


def solve_mos(g, mat, dev, Cdop_sub, VG, quantum=None, psi_init=None, lam_init=None):
    """One gate voltage. Returns psi, n, p, Lambda, Ninv (cm^-2), the charge
    centroid (cm), Qs and the outer iteration count."""
    x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]
    psi_bulk = ph.equilibrium_bulk_potential(mat, Cdop_sub)
    carrier = "n" if Cdop_sub < 0 else "p"
    semi = ~g["is_oxide"]

    def solve_dd(lam):
        nonlocal psi_init
        shift = None if lam is None else _shift(lam, carrier)
        r = solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_sub, VG, psi_bulk,
                                  psi_init=psi_init, quantum_shift=shift, interfaces=g.get("interfaces"))
        psi_init = r["psi"]
        return r

    if quantum is None:
        st, lam, k = solve_dd(None), np.zeros(len(x)), 0
    else:
        quantum = QuantumConfig(**{**quantum.__dict__, "carrier": carrier})
        line = mos_line(g, quantum)
        upd = _line_update(line, x, np.zeros(len(x)), quantum, mat)
        lam0 = np.zeros(len(x)) if lam_init is None else lam_init
        st, lam, k, _ = sp_loop(solve_dd, upd, lam0, tol=quantum.tol)
        upd(st)
    V = node_volumes(x)
    dens = st["n"] if carrier == "n" else st["p"]
    st.update(lam=lam, outer=k, VG=VG,
              Ninv=_sheet(x, dens, V, semi),
              centroid=_centroid(x, dens, V, semi, 0.0),
              Qs=semiconductor_charge(x, st["psi"], dev, g["oxide_index"], g["gate_oxide_index"]))
    return st


def sweep_mos(g, mat, dev, Cdop_sub, VG_list, quantum=None):
    out, psi, lam = [], None, None
    for VG in VG_list:
        r = solve_mos(g, mat, dev, Cdop_sub, VG, quantum, psi_init=psi, lam_init=lam)
        psi, lam = r["psi"], r["lam"]
        out.append(r)
    VGa = np.array([r["VG"] for r in out])
    C = -np.gradient(np.array([r["Qs"] for r in out]), VGa)
    for r, c in zip(out, C):
        r["C_lf"] = c
    return out


# --- symmetric double gate ----------------------------------------------------------------
def double_gate_grid(mat, t_body, t_ox, N_body=1e15, eps_ox_r=3.9, h_si=0.05e-7, h_ox=0.1e-7):
    """gate | oxide | body (p-type N_body) | oxide | gate, uniform spacing in
    each layer; x = 0 and t_body are the Si/SiO2 walls."""
    n_ox = max(2, int(round(t_ox / h_ox)))
    n_si = max(4, int(round(t_body / h_si)))
    x = np.unique(np.concatenate([np.linspace(-t_ox, 0, n_ox + 1), np.linspace(0, t_body, n_si + 1),
                                  np.linspace(t_body, t_body + t_ox, n_ox + 1)]))
    semi = (x >= 0) & (x <= t_body)
    xm = 0.5 * (x[1:] + x[:-1])
    eps_edge = np.where((xm > 0) & (xm < t_body), mat.eps, eps_ox_r * EPS0)
    return dict(x=x, Cdop=np.where(semi, -N_body, 0.0), eps_edge=eps_edge, ni_arr=np.where(semi, mat.ni, 0.0),
                is_semi=semi, t_body=t_body, t_ox=t_ox, eps_ox=eps_ox_r * EPS0)


def solve_dg(g, mat, VG, quantum=None, psi_init=None, lam_init=None, wf_minus_midgap=0.0):
    """Both gates at VG (gate work function = midgap + wf_minus_midgap); the
    body is in equilibrium with a source at phi = 0 (a MOSFET channel)."""
    x, semi = g["x"], g["is_semi"]
    psi_gate = VG - wf_minus_midgap
    zero = np.zeros(len(x))

    def solve_dd(lam):
        nonlocal psi_init
        guess = np.full(len(x), psi_gate) if psi_init is None else psi_init.copy()
        guess[0] = guess[-1] = psi_gate
        shift = None if lam is None else (lam, None)
        psi, n, p, it = ph.solve_poisson(x, g["Cdop"], mat, zero, zero, guess, eps=g["eps_edge"],
                                         ni=g["ni_arr"], damping_cap=0.5, max_iter=200, quantum_shift=shift)
        psi_init = psi
        return dict(psi=psi, n=n, p=p, iters=it)

    if quantum is None:
        st, lam, k = solve_dd(None), zero.copy(), 0
    else:
        t_pen = 0.0 if quantum.hard_wall else min(quantum.t_pen, g["t_ox"])
        line = line_1d(x, semi, node_volumes(x), 0.0, g["t_body"], depth=max(quantum.depth, g["t_body"]),
                       t_pen=t_pen, h=quantum.h)
        upd = _line_update(line, x, zero, quantum, mat)
        st, lam, k, _ = sp_loop(solve_dd, upd, zero if lam_init is None else lam_init, tol=quantum.tol)
        upd(st)
    V = node_volumes(x)
    st.update(lam=lam, outer=k, VG=VG, Ninv=_sheet(x, st["n"], V, semi))
    return st
