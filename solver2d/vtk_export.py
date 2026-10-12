"""Device results -> VTK (.vtu/.pvd), the shared data contract of the 3D plan
(plans/3d_plan.md, "Shared contract"): every structure and solved state opens
in ParaView, and later in tcadview, with one naming convention.

Geometry: coordinates in um. A 2D Mesh2D becomes triangles; a mesh from
mesh3d.extrude.extrude_mesh becomes wedges (prisms), one per base triangle
per z layer.

Naming convention (Sentaurus-style field names, so TCAD users recognise
them; units listed in the file's `field_units`):

  point data   ElectrostaticPotential        V
               eQuasiFermiPotential          V    (phin)
               hQuasiFermiPotential          V    (phip)
               eDensity, hDensity            cm^-3 (NaN in insulators, like the
                                                    quasi-Fermi fields)
               DopingConcentration           cm^-3 (signed net, Nd - Na)
               SpaceCharge                   C/cm^3, q (p - n + Nd - Na)
               ConductionBandEnergy          eV   (NaN in insulators)
               ValenceBandEnergy             eV   (NaN in insulators)
               eQuasiFermiEnergy             eV   (= -phin)
               hQuasiFermiEnergy             eV   (= -phip)
               contact_id                    index into contact_names, -1 = none
               ElectricField                 V/cm, 3-vector per node (nodal_vectors)
               eCurrentDensity               A/cm^2, 3-vector (NaN in insulators)
               hCurrentDensity               A/cm^2, 3-vector (NaN in insulators)
               TotalCurrentDensity           A/cm^2, 3-vector (NaN in insulators)
  cell data    region_id                     index into region_names, -1 = none
               material_id                   index into material_names
               ElectricField                 V/cm, 3 components (constant per cell)
               AbsElectricField              V/cm
  field data   region_names, material_names, contact_names (strings)
               contact_bias                  V, per contact_names entry (if given)
               label                         e.g. "Vgs=+1.000V Vds=0.05V"
               field_units                   "name:unit" strings

Coordinates are the solver's: y is depth, positive DOWN into the substrate
(as in Sentaurus). In ParaView, look from -z with view-up (0, -1, 0) to see
the device the right way up.

Energies use the solver's reference: Ec = -(psi + dEi) + (Ec - Ei), Ev = Ec -
Eg (see mesh2d/mesh2d.py), so the quasi-Fermi energies and band edges share
one axis.
"""
import os

import numpy as np

from core import material_db
from core.params import Q
from core import vtk_io

CM_TO_UM = 1.0e4

FIELD_UNITS = {
    "ElectrostaticPotential": "V", "eQuasiFermiPotential": "V", "hQuasiFermiPotential": "V",
    "eDensity": "cm^-3", "hDensity": "cm^-3", "DopingConcentration": "cm^-3", "SpaceCharge": "C/cm^3",
    "ConductionBandEnergy": "eV", "ValenceBandEnergy": "eV", "eQuasiFermiEnergy": "eV",
    "hQuasiFermiEnergy": "eV", "ElectricField": "V/cm", "AbsElectricField": "V/cm",
    "eCurrentDensity": "A/cm^2", "hCurrentDensity": "A/cm^2", "TotalCurrentDensity": "A/cm^2",
}


def _edge_fit(mesh, values, weights):
    """Per-node vector g (N, 3) that best fits, in weighted least squares,
    the directional values `values` (E,) along each edge's unit direction:
    min sum_e w_e (t_e . g - value_e)^2 over the edges of each node. Exact
    for a uniform vector field on any mesh whose edges at a node span the
    space (a box cell's 6 axis edges give the components directly). Nodes
    with no weighted edge get NaN."""
    P = np.asarray(mesh.points, dtype=float)
    if P.shape[1] == 2:
        P = np.c_[P, np.zeros(len(P))]
    N = len(P)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    d = P[jj] - P[ii]
    t = d / np.linalg.norm(d, axis=1)[:, None]
    w = np.asarray(weights, dtype=float)
    A = np.zeros((N, 3, 3))
    b = np.zeros((N, 3))
    wtt = w[:, None, None] * t[:, :, None] * t[:, None, :]
    wtv = (w * values)[:, None] * t
    for idx in (ii, jj):
        np.add.at(A, idx, wtt)
        np.add.at(b, idx, wtv)
    tr = np.trace(A, axis1=1, axis2=2)
    ok = tr > 0
    # a node whose edges lie in a plane (or a line) is singular there: a
    # tiny relative ridge keeps the out-of-plane component at 0
    A[ok] += (1e-9 * tr[ok])[:, None, None] * np.eye(3)
    g = np.full((N, 3), np.nan)
    g[ok] = np.linalg.solve(A[ok], b[ok][:, :, None])[:, :, 0]
    return g


def nodal_vectors(mesh, mat, state):
    """Vector fields at the nodes of a solved state:
      ElectricField   -grad psi, fitted to dpsi / L along every edge (weights:
                      the edge's facet area, all materials)
      e/h/TotalCurrentDensity  fitted to I_e / facet_semi along every
                      semiconductor edge - the solver's own Scharfetter-
                      Gummel edge currents (solver2d.newton_solver_qf_2d.
                      edge_currents), so the arrows carry the conserved
                      current; NaN where a node has no semiconductor edge.
    Units V/cm and A/cm^2 (the mesh is in cm); 3 components (z = 0 in 2D)."""
    from solver2d.newton_solver_qf_2d import edge_currents, _mesh_semi_geometry
    psi, n, p = state["psi"], state["n"], state["p"]
    zero = np.zeros(len(psi))
    phin, phip = state.get("phin", zero), state.get("phip", zero)
    P = np.asarray(mesh.points, dtype=float)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    L = np.linalg.norm(P[jj] - P[ii], axis=1)
    facet_semi, _, _ = _mesh_semi_geometry(mesh)
    E = -_edge_fit(mesh, (psi[jj] - psi[ii]) / L, mesh.facet_length)
    In, Ip = edge_currents(mesh, mat, psi, n, p, phin, phip, state.get("velocity_saturation", False))
    fs = np.where(facet_semi > 0, facet_semi, 1.0)
    Jn = _edge_fit(mesh, In / fs, facet_semi)
    Jp = _edge_fit(mesh, Ip / fs, facet_semi)
    return {"ElectricField": E, "eCurrentDensity": Jn, "hCurrentDensity": Jp, "TotalCurrentDensity": Jn + Jp}


def _material_name(eps_r, Eg_eV, insulator):
    """Registry name of the material with these properties (the Material
    dataclass carries no name yet - the 3D YAML recipe will name materials
    explicitly)."""
    cat = "insulator" if insulator else "semiconductor"
    for name, props in material_db.MATERIALS.items():
        if props.category != cat or abs(props.eps_r - eps_r) > 1e-6 * eps_r:
            continue
        if insulator or Eg_eV is None or abs(props.Eg_eV_300K - Eg_eV) < 0.02:
            return name
    return "Insulator" if insulator else "Semiconductor"


def _tri_regions(mesh, mat):
    """Per base triangle: region index (last region wins, tested at the
    centroid, which is never on a region edge), plus region and material
    names."""
    dom = mesh.domain
    tris = mesh.base_triangles if hasattr(mesh, "base_triangles") else mesh.triangles
    n2 = len(mesh.points) // getattr(mesh, "n_layers", 1)
    c = mesh.points[:n2, :2][tris].mean(axis=1)
    rid = np.full(len(tris), -1)
    for k, rg in enumerate(dom.regions):
        (x0, x1), (y0, y1) = rg.x_range_cm, rg.y_range_cm
        rid[(c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)] = k
    region_names = [rg.name for rg in dom.regions]
    mat_names, mat_of_region = [], []
    for rg in dom.regions:
        if rg.kind == "insulator":
            nm = _material_name(rg.eps_r, None, True)
        else:
            m = rg.material if rg.material is not None else mat
            nm = _material_name(m.eps_r, m.Eg_eV, False)
        if nm not in mat_names:
            mat_names.append(nm)
        mat_of_region.append(mat_names.index(nm))
    mid = np.where(rid >= 0, np.asarray(mat_of_region)[np.maximum(rid, 0)], -1)
    return tris, rid, mid, region_names, mat_names


def _p1_grad(xy, f):
    """(M,2) gradient of the linear interpolant on triangles; xy (M,3,2), f (M,3)."""
    d1 = xy[:, 1] - xy[:, 0]
    d2 = xy[:, 2] - xy[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    f1 = f[:, 1] - f[:, 0]
    f2 = f[:, 2] - f[:, 0]
    return np.c_[(f1 * d2[:, 1] - f2 * d1[:, 1]) / det, (f2 * d1[:, 0] - f1 * d2[:, 0]) / det]


def device_cells(mesh, mat):
    """Cells and per-cell ids for `mesh`. Returns dict: points_um (N,3),
    cells, cell_type, region_id, material_id, region_names, material_names,
    and the pieces cell_field() needs."""
    tris, rid, mid, region_names, mat_names = _tri_regions(mesh, mat)
    nz = getattr(mesh, "n_layers", None)
    pts = mesh.points
    if nz is None:
        cells, ctype = tris, vtk_io.VTK_TRIANGLE
        reg, matid = rid, mid
        pts3 = np.c_[pts, np.zeros(len(pts))]
    else:
        n2 = len(pts) // nz
        cells = np.concatenate([np.c_[tris + k * n2, tris + (k + 1) * n2] for k in range(nz - 1)])
        ctype = vtk_io.VTK_WEDGE
        reg, matid = np.tile(rid, nz - 1), np.tile(mid, nz - 1)
        pts3 = pts
    return dict(points_um=pts3 * CM_TO_UM, cells=cells, cell_type=ctype, region_id=reg.astype(np.int32),
                material_id=matid.astype(np.int32), region_names=region_names, material_names=mat_names,
                tris=tris, nz=nz)


def cell_field(mesh, geo, psi):
    """(M,3) electric field -grad(psi) in V/cm, exact for the linear
    (triangle) or linear x linear (wedge) interpolant at the cell centroid."""
    tris = geo["tris"]
    if geo["nz"] is None:
        g = _p1_grad(mesh.points[tris], psi[tris])
        return -np.c_[g, np.zeros(len(g))]
    nz = geo["nz"]
    n2 = len(mesh.points) // nz
    xy = mesh.points[:n2, :2][tris]
    z = mesh.z_cm
    out = []
    for k in range(nz - 1):
        fb, ft = psi[tris + k * n2], psi[tris + (k + 1) * n2]
        gxy = _p1_grad(xy, 0.5 * (fb + ft))
        gz = (ft - fb).mean(axis=1) / (z[k + 1] - z[k])
        out.append(np.c_[gxy, gz])
    return -np.concatenate(out)


def contact_ids(mesh):
    names = []
    cid = np.full(len(mesh.points), -1, dtype=np.int32)
    for i, bc in zip(mesh.boundary_point_index, mesh.boundary_bc_type):
        if bc.startswith("contact:"):
            nm = bc.split(":", 1)[1]
            if nm not in names:
                names.append(nm)
            cid[i] = names.index(nm)
    return cid, names


def state_point_data(mesh, mat, state, vectors=True):
    """Named per-node fields of one solved state (dict with psi, phin, phip,
    n, p), or just the structure (doping) when state is None. vectors: also
    the nodal E and current-density vectors (nodal_vectors)."""
    pd = {"DopingConcentration": np.asarray(mesh.Cdop, dtype=float)}
    if state is None:
        return pd
    psi = state["psi"]
    zero = np.zeros(len(psi))
    phin, phip = state.get("phin", zero), state.get("phip", zero)
    n, p = state["n"], state["p"]
    ins = mesh.is_insulator if mesh.is_insulator is not None else np.zeros(len(psi), bool)
    dEi = mesh.dEi_arr if mesh.dEi_arr is not None else 0.0
    Eg = mesh.Eg_arr if mesh.Eg_arr is not None else mat.Eg_eV
    ec_off = mesh.ec_off_arr if mesh.ec_off_arr is not None else mat.Vt * np.log(mat.Nc / mat.ni)
    Ec = np.where(ins, np.nan, -(psi + dEi) + ec_off)
    # carrier quantities are undefined in insulators (no carriers; phin/phip
    # there are solver placeholders): NaN, so log scales and ranges ignore
    # them, the way Sentaurus omits these fields in insulator regions
    def semi(a):
        return np.where(ins, np.nan, a)
    pd.update({
        "ElectrostaticPotential": psi, "eQuasiFermiPotential": semi(phin), "hQuasiFermiPotential": semi(phip),
        "eDensity": semi(n), "hDensity": semi(p), "SpaceCharge": Q * (p - n + mesh.Cdop),
        "ConductionBandEnergy": Ec, "ValenceBandEnergy": Ec - Eg,
        "eQuasiFermiEnergy": semi(-phin), "hQuasiFermiEnergy": semi(-phip),
    })
    if vectors and "n" in state and getattr(mesh, "edges", None) is not None:
        pd.update(nodal_vectors(mesh, mat, state))
    return pd


def write_state(path, mesh, mat, state=None, label="", contact_bias=None, geo=None, compress=True):
    """Write one structure (state=None) or solved state to `path` (.vtu).
    contact_bias: {contact name: V}. `geo` = device_cells(mesh, mat), passed
    in when writing many states of one mesh."""
    geo = geo or device_cells(mesh, mat)
    cid, cnames = contact_ids(mesh)
    pd = state_point_data(mesh, mat, state)
    pd["contact_id"] = cid
    cd = {"region_id": geo["region_id"], "material_id": geo["material_id"]}
    if state is not None:
        E = cell_field(mesh, geo, state["psi"])
        cd["ElectricField"] = E
        cd["AbsElectricField"] = np.sqrt((E ** 2).sum(axis=1))
    fd = {"region_names": geo["region_names"], "material_names": geo["material_names"],
          "field_units": [f"{k}:{v}" for k, v in FIELD_UNITS.items()]}
    if cnames:
        fd["contact_names"] = cnames
        if contact_bias is not None:
            fd["contact_bias"] = np.array([float(contact_bias.get(c, 0.0)) for c in cnames])
    if label:
        fd["label"] = label
    vtk_io.write_vtu(path, geo["points_um"], geo["cells"], geo["cell_type"], point_data=pd, cell_data=cd,
                     field_data=fd, compress=compress)


def write_states(out_dir, stem, mesh, mat, states, compress=True):
    """Write one .vtu per state, named <stem>_<bias>.vtu so that separate
    processes writing parts of one sweep never collide. states: list of
    dicts {state, bias (the swept value), label, contact_bias}. Returns the
    .pvd entries (see write_pvd)."""
    os.makedirs(out_dir, exist_ok=True)
    geo = device_cells(mesh, mat)
    entries = []
    for s in states:
        fname = f"{stem}_{s['bias']:+.4f}.vtu"
        write_state(os.path.join(out_dir, fname), mesh, mat, s["state"], label=s.get("label", ""),
                    contact_bias=s.get("contact_bias"), geo=geo, compress=compress)
        entries.append(dict(file=fname, timestep=s["bias"], name=s.get("label")))
    return entries


def write_pvd(out_dir, stem, entries):
    """<stem>.pvd over `entries`, ordered by bias. Returns its path."""
    path = os.path.join(out_dir, f"{stem}.pvd")
    vtk_io.write_pvd(path, sorted(entries, key=lambda e: e["timestep"]))
    return path


def write_sweep(out_dir, stem, mesh, mat, states, compress=True):
    """write_states + write_pvd in one call. Returns the .pvd path."""
    return write_pvd(out_dir, stem, write_states(out_dir, stem, mesh, mat, states, compress))
