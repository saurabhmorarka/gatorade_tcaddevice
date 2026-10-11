"""Box-FV mesh on a rectilinear (tensor-product, graded) grid from per-cell
material labels - the bridge from the virtual fab's voxel structures to the
existing drift-diffusion solver (solver2d/newton_solver_qf_2d.py), which
only consumes edges, facet areas, control volumes and per-node data.

Why this is an exact box-FV mesh: the Voronoi cell of a node of a
rectilinear grid is the box between the midpoints to its neighbours, so
  * a node's control volume is the sum of the 2^d corner octants of its
    adjacent cells (1/2^d of each cell's volume),
  * an edge's facet (the dual face it crosses) is made of 2^(d-1) quarter
    faces, one inside each cell sharing the edge,
and every per-material split is exact: a facet part belongs to exactly one
cell, hence one material. So
  facet_length        sum of the parts in device cells (area in 3D, length in 2D)
  facet_length_semi   the parts in semiconductor cells (carrier flux)
  edge_g              sum over parts of eps_cell * part / edge length
  cv_area(_semi)      the octants in device (semiconductor) cells
Material boundaries are cell faces (staircased at the grid spacing, which
is why the grid is graded fine at interfaces); a node on a Si/oxide face
is a semiconductor node whose oxide octants hold no carriers - the same
convention as the triangle meshes (mesh2d/fvgeometry.py).

Conductors (gate poly, contact metal) are not part of the mesh: the device
nodes touching a conductor cell become that contact's Dirichlet nodes.
Faces of the grid can be contacts too (the body at the bottom). Every other
boundary is the box method's natural zero-flux (symmetry / free surface).
"""
import itertools
from types import SimpleNamespace

import numpy as np

from core.params import EPS0

SEMI, INSULATOR, CONDUCTOR = "semi", "insulator", "conductor"


class CartesianMesh(SimpleNamespace):
    """The mesh's attributes (a class, not a bare SimpleNamespace, so the
    solver can key its cached linear solver on a weak reference)."""


def _padded(a, fill=0):
    """a padded by one cell on every side (the outside of the grid)."""
    return np.pad(a, 1, constant_values=fill)


def _cell_widths(axes):
    """Per-axis cell widths broadcast to the cell array shape."""
    d = len(axes)
    out = []
    for a, x in enumerate(axes):
        shape = [1] * d
        shape[a] = len(x) - 1
        out.append(np.diff(x).reshape(shape))
    return out


def _node_sum(cell_vals_padded, shape):
    """For every node, the sum over its 2^d adjacent cells of a padded
    per-cell value (cells outside the grid count as 0)."""
    d = len(shape)
    out = np.zeros(shape)
    for o in itertools.product((0, 1), repeat=d):
        out += cell_vals_padded[tuple(slice(o[a], o[a] + shape[a]) for a in range(d))]
    return out


def _node_any(cell_mask_padded, shape):
    d = len(shape)
    out = np.zeros(shape, bool)
    for o in itertools.product((0, 1), repeat=d):
        out |= cell_mask_padded[tuple(slice(o[a], o[a] + shape[a]) for a in range(d))]
    return out


def build_cartesian_mesh(axes_cm, cell_material, materials, node_cdop, mat, face_contacts=None):
    """Box-FV mesh (the attribute set solver2d/newton_solver_qf_2d.py reads).

    axes_cm        list of d (2 or 3) increasing node-coordinate arrays, cm
    cell_material  int array, shape (n0-1, ..., nd-1 - 1): index into
                   `materials`, -1 = not part of the device (gas)
    materials      list of dicts: {name, kind: semi|insulator|conductor,
                   eps_r (insulators), contact (conductors: contact name)}
    node_cdop      net doping at every grid node (cm^-3), shape (n0, ..)
    mat            the semiconductor's core Material (eps, ni, Nc, Eg, Vt)
    face_contacts  {contact name: (axis, "min" | "max")}: a whole grid face
    Returns a namespace with points (N, d), edges, edge_weight, edge_g,
    facet_length(_semi), cv_area(_semi), Cdop, ni_arr, is_insulator, band
    arrays, boundary_point_index/bc_type (contacts), domain.contacts, plus
    grid_index (N, d) and node_id (grid -> mesh node, -1) for output."""
    axes = [np.asarray(x, float) for x in axes_cm]
    d = len(axes)
    shape = tuple(len(x) for x in axes)
    cm = np.asarray(cell_material)
    if cm.shape != tuple(n - 1 for n in shape):
        raise ValueError(f"cell_material shape {cm.shape} does not match axes {shape}")
    kinds = np.array([m["kind"] for m in materials] + ["gas"])
    eps_r = np.array([mat.eps / EPS0 if m["kind"] == SEMI else float(m.get("eps_r", 1.0)) for m in materials]
                     + [0.0])
    k = np.where(cm >= 0, cm, len(materials))                      # -1 -> the trailing "gas"
    is_dev = np.isin(kinds[k], (SEMI, INSULATOR))
    is_semi = kinds[k] == SEMI
    eps_cell = EPS0 * eps_r[k] * is_dev

    widths = _cell_widths(axes)
    vol = np.prod(np.broadcast_arrays(*widths), axis=0) if d > 1 else widths[0]
    cv = _node_sum(_padded(vol * is_dev), shape) / 2 ** d
    cv_semi = _node_sum(_padded(vol * is_semi), shape) / 2 ** d

    node_id = np.full(shape, -1, dtype=np.int64)
    dev_nodes = cv > 0
    node_id[dev_nodes] = np.arange(int(dev_nodes.sum()))

    # edges: along axis a between grid nodes i and i + e_a
    e_i, e_j, e_fl, e_fls, e_g, e_w = [], [], [], [], [], []
    for a in range(d):
        others = [b for b in range(d) if b != a]
        part = np.ones(cm.shape)                                      # the cell's quarter face
        for b in others:
            part = part * widths[b] / 2.0
        P_dev, P_semi, P_eg = (_padded(part * is_dev), _padded(part * is_semi), _padded(part * eps_cell))
        eshape = list(shape)
        eshape[a] -= 1
        fl, fls, g = np.zeros(eshape), np.zeros(eshape), np.zeros(eshape)
        for o in itertools.product((0, 1), repeat=d - 1):
            sl = [slice(None)] * d
            sl[a] = slice(1, shape[a])
            for b, ob in zip(others, o):
                sl[b] = slice(ob, ob + shape[b])
            sl = tuple(sl)
            fl += P_dev[sl]
            fls += P_semi[sl]
            g += P_eg[sl]
        hshape = [1] * d
        hshape[a] = shape[a] - 1
        h = np.diff(axes[a]).reshape(hshape)
        keep = fl > 0
        idx = np.argwhere(keep)
        jdx = idx.copy()
        jdx[:, a] += 1
        e_i.append(node_id[tuple(idx.T)])
        e_j.append(node_id[tuple(jdx.T)])
        e_fl.append(fl[keep])
        e_fls.append(fls[keep])
        e_g.append((g / h)[keep])
        e_w.append((fl / h)[keep])
    edges = np.column_stack([np.concatenate(e_i), np.concatenate(e_j)])
    if (edges < 0).any():
        raise RuntimeError("an edge with a facet in a device cell has a non-device end node")

    # contacts
    contact_nodes = {}
    for ci, m in enumerate(materials):
        if m["kind"] != CONDUCTOR:
            continue
        touch = _node_any(_padded(cm == ci, False), shape) & dev_nodes
        contact_nodes.setdefault(m["contact"], np.zeros(shape, bool))
        contact_nodes[m["contact"]] |= touch
    for name, (a, side) in (face_contacts or {}).items():
        face = np.zeros(shape, bool)
        sl = [slice(None)] * d
        sl[a] = 0 if side == "min" else shape[a] - 1
        face[tuple(sl)] = True
        contact_nodes.setdefault(name, np.zeros(shape, bool))
        contact_nodes[name] |= face & dev_nodes
    names = list(contact_nodes)
    owner = np.full(shape, -1)
    for c, name in enumerate(names):
        clash = contact_nodes[name] & (owner >= 0)
        if clash.any():
            other = names[int(owner[clash][0])]
            raise ValueError(f"{int(clash.sum())} node(s) touch both contact {other!r} and {name!r}")
        owner[contact_nodes[name]] = c
    b_idx = np.concatenate([node_id[contact_nodes[nm]] for nm in names]) if names else np.zeros(0, int)
    b_type = np.array(sum([[f"contact:{nm}"] * int(contact_nodes[nm].sum()) for nm in names], []), dtype=object)
    # ascending node order: the solver pairs its per-contact-node boundary
    # values (in boundary_point_index order) with psi[is_contact] (in node
    # order), so the two orders must agree
    order = np.argsort(b_idx, kind="stable")
    b_idx, b_type = b_idx[order], list(b_type[order])

    grid_index = np.argwhere(dev_nodes)                       # C order == node_id order
    pts = np.column_stack([axes[a][grid_index[:, a]] for a in range(d)])
    semi_node = cv_semi[dev_nodes] > 0
    ni = np.where(semi_node, mat.ni, 0.0)
    contacts = []
    for nm in names:
        p = pts[node_id[contact_nodes[nm]]]
        contacts.append(SimpleNamespace(name=nm, x_range_cm=(float(p[:, 0].min()), float(p[:, 0].max()))))
    hmin = min(float(np.diff(x).min()) for x in axes)
    return CartesianMesh(
        points=pts, triangles=np.zeros((0, 3), dtype=int), edges=edges,
        edge_weight=np.concatenate(e_w), edge_g=np.concatenate(e_g),
        facet_length=np.concatenate(e_fl), facet_length_semi=np.concatenate(e_fls),
        cv_area=cv[dev_nodes], cv_area_semi=cv_semi[dev_nodes],
        Cdop=np.where(semi_node, np.asarray(node_cdop, float)[dev_nodes], 0.0),
        ni_arr=ni, is_insulator=~semi_node,
        dEi_arr=np.zeros(len(pts)), Eg_arr=np.full(len(pts), mat.Eg_eV),
        ec_off_arr=np.full(len(pts), mat.Vt * np.log(mat.Nc / mat.ni)),
        boundary_point_index=b_idx, boundary_bc_type=b_type, boundary_edges=set(),
        domain=SimpleNamespace(contacts=contacts), h_min_cm=hmin,
        grid_axes=axes, grid_index=grid_index, node_id=node_id, dim=d)
