"""2D MOS-capacitor equilibrium solve: the point-cloud/box-FV generalization
of mos/mos_solver.py::solve_mos_equilibrium, over a heterogeneous-
permittivity Mesh2D (mesh2d/mesh2d.py, built with `mat` given so it carries
edge_g/ni_arr/is_insulator - see that module's docstring).

Like the 1D MOS capacitor, there is no current path at all in steady state
(the gate is an ideal insulator), so - exactly as mos/mos_solver.py's own
module docstring explains - the whole structure sits at a single applied
gate voltage's worth of band-bending with NO continuity equations needed:
just a nonlinear Poisson solve for psi alone, with phin/phip PRESCRIBED
(not solved for) the same way solve_mos_equilibrium prescribes them. This
is why this module is much simpler than solver2d/newton_solver_qf_2d.py's
fully-coupled (psi, phin, phip) diode solve - only 1/3 as many unknowns,
and no edge-current assembly at all.

The gate is modeled as an ideal metal contact sitting directly on the
oxide's own top surface (mesh2d/geometry2d.py::TopMesa) - the same "x[0] IS
the gate/oxide interface, no separate metal region meshed" convention as
mos/mesh.py::build_mos_grid's Cdop_gate=None case, generalized to 2D via
mesh2d's Dirichlet contact tagging instead of a fixed node index.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from core.params import Q, Material
from core.physics import equilibrium_bulk_potential, equilibrium_bulk_potential_arr
from core.jacobian_scaling import equilibrated_spsolve
from mos import mos_analytic as man
from solver2d.newton_solver_qf_2d import poisson_row_scale


def _densities(psi, phin, phip, ni_arr, Vt):
    n = ni_arr * np.exp((psi - phin) / Vt)
    p = ni_arr * np.exp((phip - psi) / Vt)
    return n, p


def _semi_frac(mesh):
    cv_semi = getattr(mesh, "cv_area_semi", None)
    return np.ones(len(mesh.cv_area)) if cv_semi is None else cv_semi / mesh.cv_area


def _residual(psi, mesh, Vt, Cdop, ni_arr, phin, phip, is_contact, psi_bc, poisson_scale):
    n, p = _densities(psi, phin, phip, ni_arr, Vt)
    N = len(psi)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]

    div_psi = np.zeros(N)
    np.add.at(div_psi, ii, mesh.edge_g * (psi[jj] - psi[ii]))
    np.add.at(div_psi, jj, mesh.edge_g * (psi[ii] - psi[jj]))

    # Charge lives only in the semiconductor part of each control volume
    # (an interface node's oxide half-cell holds none) - see
    # newton_solver_qf_2d.py::_mesh_semi_geometry.
    semi_frac = _semi_frac(mesh)
    Rpsi = (div_psi / mesh.cv_area - Q * (n - p - Cdop) * semi_frac) / poisson_scale
    Rpsi[is_contact] = psi[is_contact] - psi_bc[is_contact]
    return Rpsi


def _residual_and_jacobian(psi, mesh, Vt, Cdop, ni_arr, phin, phip, is_contact, psi_bc, poisson_scale):
    n, p = _densities(psi, phin, phip, ni_arr, Vt)
    N = len(psi)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    g_e = mesh.edge_g

    div_psi = np.zeros(N)
    np.add.at(div_psi, ii, g_e * (psi[jj] - psi[ii]))
    np.add.at(div_psi, jj, g_e * (psi[ii] - psi[jj]))

    # Charge lives only in the semiconductor part of each control volume
    # (an interface node's oxide half-cell holds none) - see
    # newton_solver_qf_2d.py::_mesh_semi_geometry.
    semi_frac = _semi_frac(mesh)
    Rpsi = (div_psi / mesh.cv_area - Q * (n - p - Cdop) * semi_frac) / poisson_scale
    Rpsi[is_contact] = psi[is_contact] - psi_bc[is_contact]

    dn_dpsi = n / Vt
    dp_dpsi = -p / Vt

    rows, cols, data = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); data.append(v)

    active_i = ~is_contact[ii]
    active_j = ~is_contact[jj]
    cv_i, cv_j = mesh.cv_area[ii], mesh.cv_area[jj]

    add(ii[active_i], ii[active_i], (-g_e / cv_i / poisson_scale)[active_i])
    add(ii[active_i], jj[active_i], (g_e / cv_i / poisson_scale)[active_i])
    add(jj[active_j], jj[active_j], (-g_e / cv_j / poisson_scale)[active_j])
    add(jj[active_j], ii[active_j], (g_e / cv_j / poisson_scale)[active_j])

    node = np.arange(N)
    free = node[~is_contact]
    add(free, free, -Q * (dn_dpsi[free] - dp_dpsi[free]) * semi_frac[free] / poisson_scale)

    interior_rows = np.concatenate(rows)
    interior_cols = np.concatenate(cols)
    interior_data = np.concatenate(data)

    contact_idx = node[is_contact]
    dirichlet_rows, dirichlet_cols, dirichlet_data = [], [], []
    for i in contact_idx:
        col_mask = interior_cols == i
        local_max = np.max(np.abs(interior_data[col_mask])) if np.any(col_mask) else 0.0
        dirichlet_rows.append(i)
        dirichlet_cols.append(i)
        dirichlet_data.append(max(1.0, local_max))

    rows_all = np.concatenate([interior_rows, dirichlet_rows])
    cols_all = np.concatenate([interior_cols, dirichlet_cols])
    data_all = np.concatenate([interior_data, dirichlet_data])
    J = sp.coo_matrix((data_all, (rows_all, cols_all)), shape=(N, N)).tocsc()
    return Rpsi, J


def solve_mos_equilibrium_2d(mesh, mat: Material, dev, Cdop_substrate, VG,
                              gate_contact_name="gate", f_tol=1e-10, maxiter=100,
                              psi_init=None, damping_cap=0.5, verbose=False):
    """Single-gate-voltage equilibrium solve over `mesh` (built by
    mesh2d.build_mesh2d(..., mat=mat) so it carries edge_g/ni_arr/
    is_insulator). Returns dict with psi, n, p, iters, res_norm.

    Every contact other than `gate_contact_name` is held at its own LOCAL
    ohmic equilibrium potential (equilibrium_bulk_potential at that
    contact's own mesh.Cdop, same ideal-ohmic convention as the diode's
    contact BC) - i.e. grounded, no bias swept there. The gate contact's
    Dirichlet value uses mos.mos_analytic.flatband_voltage (an ideal metal
    gate directly on the oxide, ni_arr=0 there so mesh.Cdop is moot for it,
    matching mos/mos_solver.py's Cdop_gate=None convention)."""
    N = len(mesh.points)
    if mesh.edge_g is None or mesh.ni_arr is None:
        raise ValueError(
            "solve_mos_equilibrium_2d requires a mesh built with mesh2d.build_mesh2d(..., mat=mat) "
            "so edge_g/ni_arr/is_insulator are populated")
    Vt = mat.Vt

    boundary_bc_type = np.array(mesh.boundary_bc_type)
    is_contact_boundary = np.array([bc.startswith("contact:") for bc in boundary_bc_type])
    contact_point_idx = mesh.boundary_point_index[is_contact_boundary]
    contact_names = [bc.split(":", 1)[1] for bc in boundary_bc_type[is_contact_boundary]]

    is_contact = np.zeros(N, dtype=bool)
    is_contact[contact_point_idx] = True

    psi_bulk_substrate = equilibrium_bulk_potential(mat, Cdop_substrate)
    V_FB = man.flatband_voltage(dev, mat, Cdop_substrate)

    psi_bc_full = np.zeros(N)
    for pt, name in zip(contact_point_idx, contact_names):
        if name == gate_contact_name:
            psi_bc_full[pt] = psi_bulk_substrate + (VG - V_FB)
        else:
            psi_bc_full[pt] = equilibrium_bulk_potential(mat, mesh.Cdop[pt])
    psi_bc = psi_bc_full[contact_point_idx]

    # phin/phip: prescribed (not solved), VG on the insulator/gate side, 0
    # on the semiconductor side - moot wherever ni_arr=0 (the whole oxide,
    # including the gate contact itself), same convention as
    # mos/mos_solver.py::solve_mos_equilibrium's own docstring explains.
    phin = np.where(mesh.is_insulator, VG, 0.0)
    phip = phin.copy()

    if psi_init is None:
        # Charge-neutral bulk potential in the semiconductor; an arbitrary
        # (but finite) 0 in the insulator, where ni_arr=0 makes psi's exact
        # value electrically moot until Newton moves it toward the
        # gate-to-substrate field the Dirichlet BCs actually impose.
        ni_safe = np.where(mesh.is_insulator, 1.0, mesh.ni_arr)  # avoid a 0/0 warning in the
                                                                     # insulator, discarded below anyway
        psi0 = np.where(mesh.is_insulator, 0.0,
                         equilibrium_bulk_potential_arr(Vt, ni_safe, mesh.Cdop))
    else:
        psi0 = psi_init.copy()
    psi0[contact_point_idx] = psi_bc

    h_typ = mesh.h_min_cm
    poisson_scale = poisson_row_scale(mat, h_typ)

    psi = psi0
    F, J = _residual_and_jacobian(psi, mesh, Vt, mesh.Cdop, mesh.ni_arr, phin, phip,
                                   is_contact, psi_bc_full, poisson_scale)
    res_norm = np.max(np.abs(F))
    it = 0
    for it in range(1, maxiter + 1):
        if res_norm < f_tol:
            break
        delta = equilibrated_spsolve(J, -F)
        if damping_cap is not None:
            delta = np.clip(delta, -damping_cap, damping_cap)

        step = 1.0
        for _ in range(20):
            psi_try = psi + step * delta
            F_try = _residual(psi_try, mesh, Vt, mesh.Cdop, mesh.ni_arr, phin, phip,
                               is_contact, psi_bc_full, poisson_scale)
            res_try = np.max(np.abs(F_try))
            if np.isfinite(res_try) and res_try < res_norm * (1 - 1e-4 * step):
                break
            step *= 0.5
        else:
            psi_try, res_try = psi, res_norm

        psi = psi_try
        F, J = _residual_and_jacobian(psi, mesh, Vt, mesh.Cdop, mesh.ni_arr, phin, phip,
                                       is_contact, psi_bc_full, poisson_scale)
        res_norm = np.max(np.abs(F))
        if verbose:
            print(f"  Newton(MOS,2D) it {it}: |F|_inf={res_norm:.3e}  step={step:.3g}  VG={VG:.3f}")

    if res_norm > 1.0:
        warnings.warn(f"Newton(MOS,2D) solve did not converge at VG={VG} V (|F|_inf={res_norm:.3e})")

    n, p = _densities(psi, phin, phip, mesh.ni_arr, Vt)
    return {"psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
            "iters": it, "res_norm": res_norm, "VG": VG}
