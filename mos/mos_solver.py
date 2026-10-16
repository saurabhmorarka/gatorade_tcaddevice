"""MOS capacitor equilibrium solve and C-V sweep.

Unlike the diode, a MOS capacitor has no current path at all in steady
state (the gate is an ideal insulator) - so at every DC gate voltage the
whole structure sits at a single, uniform Fermi level (phin=phip=0
throughout, same convention as the diode's equilibrium solve), *provided*
enough time has passed for generation-recombination to fully populate any
inversion layer. That assumption is exactly the "quasi-static" or
"low-frequency" C-V measurement: no continuity equations, no G-R kinetics,
and no AC/small-signal analysis are needed for it - just a sequence of
equilibrium nonlinear-Poisson solves (physics.solve_poisson, reused as-is
with an oxide/semiconductor eps(x) and ni(x) profile), one per gate
voltage, differentiated numerically: C(V_G) = dQ_s/dV_G.

The high-frequency C-V curve, where minority (inversion) carriers can't
follow a fast small-signal probe, is a genuine "quasi-small-signal"
calculation: take the low-frequency solution's minority-carrier density at
a given DC gate bias, freeze it (via physics.solve_poisson's n_frozen/
p_frozen), perturb the gate voltage by a small delta, and let only the
majority carrier and potential respond. See the module docstring exchange
in DEVELOPMENT_LOG.md for the reasoning behind not needing a literal
frequency-domain AC solve for either curve.
"""
import numpy as np

from core import physics as ph
from core.params import Material
from mos.mos_params import MOSDevice
from mos import mos_analytic as man


def solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat: Material, dev: MOSDevice,
                           Cdop_substrate, VG, psi_bulk,
                           psi_init=None, n_frozen=None, p_frozen=None,
                           damping_cap=0.5, max_iter=200, Cdop_gate=None,
                           interfaces=None, quantum_shift=None):
    """Single-gate-voltage equilibrium (or quasi-small-signal, if
    n_frozen/p_frozen given) solve. Returns dict with psi, n, p, iters.

    Cdop_gate=None (default): ideal metal gate, x[0] is the gate/oxide
    interface and its Dirichlet BC comes from the work-function-based
    flat-band voltage (mos_analytic.flatband_voltage) - original behavior.

    Cdop_gate given: real poly gate (see mesh.build_mos_grid). x[0] is now
    the OUTER metal contact on top of the poly, an ohmic contact - same
    convention used at the substrate's back contact (x[-1]): the contact
    potential is the poly's own charge-neutral equilibrium potential plus
    the applied bias, with no separate flat-band-voltage bookkeeping. The
    poly's band bending (and any depletion right at the oxide interface)
    then falls out of the Poisson solve itself, self-consistently, exactly
    like the substrate's does - no new physics, just a third semiconductor
    region for the same solver to see.
    """
    N = len(x)
    if Cdop_gate is None:
        V_FB = man.flatband_voltage(dev, mat, Cdop_substrate)
        psi_gate_bc = psi_bulk + (VG - V_FB)
    else:
        psi_bulk_gate = ph.equilibrium_bulk_potential(mat, Cdop_gate)
        psi_gate_bc = psi_bulk_gate + VG

    if psi_init is None:
        frac = np.linspace(1.0, 0.0, N)
        psi_guess = psi_bulk + (psi_gate_bc - psi_bulk) * frac
    else:
        psi_guess = psi_init.copy()
    psi_guess[0] = psi_gate_bc
    psi_guess[-1] = psi_bulk

    # Quasi-Fermi levels: flat 0 everywhere is correct ONLY when the gate has
    # no mobile carriers of its own (the ideal metal case - ni_arr=0 there,
    # so phin/phip are moot regardless of value). A real poly gate has real
    # carriers, and since the MOS cap carries zero current in steady state,
    # each side of the oxide is independently in local equilibrium with ITS
    # OWN contact - there is no reason the two sides should share one Fermi
    # level once a bias VG is applied between them (that's specifically what
    # "applying a voltage" means: splitting the two contacts' electrochemical
    # potentials by qV). So phin/phip must be VG on the gate side and 0 on
    # the substrate side, with the step falling inside the oxide, where
    # ni_arr=0 makes its exact placement physically moot. Using a flat 0 for
    # a poly gate instead (as this used to) forces n=ni*exp((psi-0)/Vt) at
    # the gate contact even though psi there legitimately carries the full
    # VG offset - an artificial, exponentially large charge spike with no
    # physical basis, which screens itself out within nanometers and makes
    # the entire rest of the structure (and the whole substrate response!)
    # spuriously look VG-independent. Using x<0 (poly+oxide) vs x>=0
    # (substrate) as the split is harmless for the ideal metal gate too
    # (ni_arr=0 on the x<0 side there regardless), so this is applied
    # unconditionally rather than only for Cdop_gate is not None.
    phin = np.where(x < 0.0, VG, 0.0)
    phip = np.where(x < 0.0, VG, 0.0)
    psi, n, p, iters = ph.solve_poisson(
        x, Cdop, mat, phin, phip, psi_guess,
        eps=eps_edge, ni=ni_arr, damping_cap=damping_cap, max_iter=max_iter,
        n_frozen=n_frozen, p_frozen=p_frozen, interfaces=interfaces,
        quantum_shift=quantum_shift,
    )
    # Quasi-Fermi potentials are only meaningful where there are carriers to
    # define them for (ni_arr>0, i.e. the semiconductor) - in the oxide,
    # n=p=0 identically (it's an insulator) and phin/phip are undefined, not
    # some huge/meaningless number, so they're masked to NaN there.
    is_semi = ni_arr > 0
    ni_safe = np.where(is_semi, ni_arr, 1.0)
    phin = np.where(is_semi, psi - mat.Vt * np.log(np.clip(n, 1e-300, None) / ni_safe), np.nan)
    phip = np.where(is_semi, psi + mat.Vt * np.log(np.clip(p, 1e-300, None) / ni_safe), np.nan)
    return {"psi": psi, "n": n, "p": p, "iters": iters, "VG": VG, "phin": phin, "phip": phip}


def semiconductor_charge(x, psi, dev: MOSDevice, oxide_index, gate_oxide_index=0):
    """Total semiconductor-side charge per unit area (C/cm^2), from Gauss's
    law applied to the (uniform, since charge-free) field across the oxide:
    Q_s = eps_ox * (psi_interface - psi_gate_side) / t_ox. (Sign checked
    against accumulation, where excess majority-carrier charge at the
    surface must come out positive for a p-substrate.)

    gate_oxide_index is the node right at the GATE's side of the oxide -
    for an ideal metal gate that's node 0 itself (the default); for a poly
    gate (see mesh.build_mos_grid) it's the last poly node, since x[0] is
    now the outer metal contact further away, across the poly's own
    (possibly depleted) band-bending."""
    psi_gate_side = psi[gate_oxide_index]
    psi_interface = psi[oxide_index]
    return dev.eps_ox * (psi_interface - psi_gate_side) / dev.t_ox


def cv_sweep(x, Cdop, eps_edge, ni_arr, mat: Material, dev: MOSDevice,
             Cdop_substrate, VG_list, oxide_index, dV_hf=2e-3,
             Cdop_gate=None, gate_oxide_index=0, interfaces=None):
    """Low- and high-frequency C-V sweep across VG_list (with continuation
    for robustness/speed). Returns a list of per-point result dicts."""
    psi_bulk = ph.equilibrium_bulk_potential(mat, Cdop_substrate)
    is_p_sub = Cdop_substrate < 0

    results = []
    psi_prev = None
    for VG in VG_list:
        lf = solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate,
                                    VG, psi_bulk, psi_init=psi_prev, Cdop_gate=Cdop_gate,
                                    interfaces=interfaces)
        Qs_lf = semiconductor_charge(x, lf["psi"], dev, oxide_index, gate_oxide_index)

        # High-frequency point: freeze the MINORITY carrier (electrons for a
        # p-substrate, holes for an n-substrate) at its low-frequency value,
        # perturb VG by a small dV, and only let the majority carrier and
        # potential respond. This freeze must be restricted to the
        # SUBSTRATE side (x>=0): with a poly gate, that same carrier
        # species (e.g. electrons, for an n+ poly) is the gate's own
        # MAJORITY carrier, and it must stay free to respond to the VG
        # perturbation - freezing it everywhere (as if the gate were an
        # ideal metal with no carriers to freeze in the first place) pins
        # the whole poly and collapses C_hf to ~0 regardless of doping.
        n_frozen = p_frozen = None
        if is_p_sub:
            n_frozen = np.where(x >= 0.0, lf["n"], np.nan)
        else:
            p_frozen = np.where(x >= 0.0, lf["p"], np.nan)

        hf_pert = solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate,
                                         VG + dV_hf, psi_bulk, psi_init=lf["psi"],
                                         n_frozen=n_frozen, p_frozen=p_frozen, Cdop_gate=Cdop_gate,
                                         interfaces=interfaces)
        Qs_hf_pert = semiconductor_charge(x, hf_pert["psi"], dev, oxide_index, gate_oxide_index)
        # C = dQ_gate/dVG = -dQ_semiconductor/dVG (charge neutrality of the
        # two-terminal capacitor: gate charge is the negative of the
        # semiconductor charge, and it's the gate charge that increases
        # with VG, giving the conventional positive capacitance).
        C_hf = -(Qs_hf_pert - Qs_lf) / dV_hf

        results.append({
            "VG": VG, "psi": lf["psi"], "n": lf["n"], "p": lf["p"],
            "phin": lf["phin"], "phip": lf["phip"],
            "iters": lf["iters"], "Qs": Qs_lf, "C_hf": C_hf,
        })
        psi_prev = lf["psi"]

    Q_arr = np.array([r["Qs"] for r in results])
    VG_arr = np.array([r["VG"] for r in results])
    # Low-frequency capacitance: numerical dQ_gate/dV = -dQ_semiconductor/dV
    # of the equilibrium sweep itself (see the C_hf sign note above).
    C_lf = -np.gradient(Q_arr, VG_arr)
    for r, c in zip(results, C_lf):
        r["C_lf"] = c

    return results
