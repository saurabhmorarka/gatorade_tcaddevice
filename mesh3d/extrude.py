"""Extrude a 2D box-FV mesh (mesh2d.mesh2d.Mesh2D) along z into a genuine 3D
box-FV mesh of prisms - the first rung of the 3D validation ladder in
plans/3d_plan.md, and the 3D performance testbed until the octree mesher
exists.

Why this is an exact 3D box-FV mesh, not an approximation: the Voronoi cell
of a node in a tensor product (2D point set) x (1D z-grid) is the tensor
product of its 2D Voronoi cell and its 1D z-cell. So
  * an in-plane edge on layer k gets facet AREA = (2D facet length) x dz_cv[k]
  * a vertical edge between layers k and k+1 gets facet area = the node's 2D
    control-volume area, and length dz[k]
  * a node's control VOLUME = (2D control area) x dz_cv[k]
where dz_cv[k] is the node's z-cell length (half the spacing to each
neighbor layer). Every per-material split (semiconductor-only facet/volume,
eps-weighted coupling) extrudes the same way. The 2D solver
(solver2d/newton_solver_qf_2d.py) only consumes edges/weights/volumes plus
the edge length computed from `points`, so it runs unchanged on the result -
and because the physics is z-invariant, the current per unit width must
equal the 2D result (measured: agreement to 1.8e-12).

The z-boundaries (z=0, z=W) are `symmetry` planes by construction: nothing
crosses them, which is the box method's default zero-flux condition.
"""
import dataclasses

import numpy as np

from core.params import EPS0


def extrude_mesh(mesh, mat, width_cm, nz, eps_r_insulator=3.9):
    """Return a 3D copy of `mesh` extruded over z in [0, width_cm] with `nz`
    uniformly spaced layers (nz >= 2). Contacts extrude to full-width
    surfaces. eps_r_insulator: permittivity of the (single) insulator
    material, needed for the vertical-edge coupling through oxide
    control-volume parts (the 2D mesh stores only eps-weighted in-plane
    couplings)."""
    if nz < 2:
        raise ValueError("nz must be >= 2")
    N = len(mesh.points)
    z = np.linspace(0.0, width_cm, nz)
    dz = np.diff(z)
    dzcv = np.zeros(nz)
    dzcv[:-1] += dz / 2
    dzcv[1:] += dz / 2

    pts = np.concatenate([np.c_[mesh.points, np.full(N, zk)] for zk in z])
    ne = len(mesh.edges)
    layer_scale = np.repeat(dzcv, ne)

    # in-plane edges
    e_in = np.concatenate([mesh.edges + k * N for k in range(nz)])
    g2 = mesh.edge_g if mesh.edge_g is not None else mat.eps * mesh.edge_weight
    fl_semi2 = mesh.facet_length_semi if mesh.facet_length_semi is not None else mesh.facet_length
    ew_in = np.tile(mesh.edge_weight, nz) * layer_scale
    fl_in = np.tile(mesh.facet_length, nz) * layer_scale
    fls_in = np.tile(fl_semi2, nz) * layer_scale
    g_in = np.tile(g2, nz) * layer_scale

    # vertical edges
    idx = np.arange(N)
    e_v = np.concatenate([np.c_[idx + k * N, idx + (k + 1) * N] for k in range(nz - 1)])
    A = mesh.cv_area
    A_semi = mesh.cv_area_semi if mesh.cv_area_semi is not None else A
    eps_area = mat.eps * A_semi + EPS0 * eps_r_insulator * (A - A_semi)
    dz_e = np.repeat(dz, N)
    ew_v = np.tile(A, nz - 1) / dz_e
    fl_v = np.tile(A, nz - 1)
    fls_v = np.tile(A_semi, nz - 1)
    g_v = np.tile(eps_area, nz - 1) / dz_e

    def tile(a):
        return None if a is None else np.tile(a, nz)

    cv_scale = np.repeat(dzcv, N)
    out = dataclasses.replace(
        mesh, points=pts, triangles=np.zeros((0, 3), dtype=int),
        edges=np.vstack([e_in, e_v]), edge_weight=np.r_[ew_in, ew_v],
        facet_length=np.r_[fl_in, fl_v], facet_length_semi=np.r_[fls_in, fls_v],
        edge_g=np.r_[g_in, g_v], cv_area=tile(A) * cv_scale, cv_area_semi=tile(A_semi) * cv_scale,
        Cdop=tile(mesh.Cdop), ni_arr=tile(mesh.ni_arr), is_insulator=tile(mesh.is_insulator),
        dEi_arr=tile(mesh.dEi_arr), Eg_arr=tile(mesh.Eg_arr), ec_off_arr=tile(mesh.ec_off_arr),
        boundary_point_index=np.concatenate([mesh.boundary_point_index + k * N for k in range(nz)]),
        boundary_bc_type=list(mesh.boundary_bc_type) * nz)
    out.width_cm = width_cm          # 3D marker: currents are totals over this width
    out.n_layers = nz
    return out
