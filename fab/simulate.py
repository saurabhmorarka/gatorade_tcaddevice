"""Electrical simulation of a virtual-fab structure: the fab's voxel
structure -> a graded rectilinear box-FV mesh (mesh3d/cartesian.py) -> the
drift-diffusion solver the 2D MOSFET uses (solver2d/newton_solver_qf_2d.py,
with its continuation driver) -> Id-Vg curves, figures of merit, and the
solved 3D states as VTK for tcadview.

Terminals come from the device layer (layer 3) and are whole process
regions: every device node touching the gate metal (oxide, spacers, the STI
under the endcaps) is the gate; every node touching a contact metal volume
is that source/drain contact (ohmic for now; a Schottky type is planned);
"surface: bottom" is the body.

The grid is a subset of the fab's own nodes, graded from a fine spacing at
every interface and junction the structure has (detected, not hand-placed)
to a coarse one in between, so interfaces sit on grid planes and the doping
at a grid node is the fab's value there. Mirror-symmetric devices are
solved on half the width (z) with a symmetry plane, and mirrored back for
output.

  python3 -m fab simulate configs/fab_planar.yaml [--vds 0.05 1.0] [--vgs-max 1.2]
"""
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy import ndimage

from core import vtk_io
from core.config import _resolve_material_block
from fab import flow
from fab.expr import evaluate, resolve_params
from fab.structure import CONDUCTORS, INSULATOR_EPS, is_semiconductor
from mesh3d.cartesian import CONDUCTOR, INSULATOR, SEMI, build_cartesian_mesh

NM_TO_CM = 1e-7


# --- the device grid -------------------------------------------------------------------
def _features(lab, net, semi, axis, coords, min_frac):
    """Coordinates (fab nodes) along `axis` where material labels or the
    sign of the net doping change on a sizeable part of the cross-section."""
    out = []
    other = int(np.prod([n for a, n in enumerate(lab.shape) if a != axis]))
    ch = np.diff(lab, axis=axis) != 0                        # cells i | i+1 -> node i+1
    cnt = ch.sum(axis=tuple(a for a in range(3) if a != axis))
    out += list(coords[1:-1][cnt > min_frac * other])
    s = np.sign(net) * semi
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis], sl1[axis] = slice(0, -1), slice(1, None)
    flip = (s[tuple(sl0)] * s[tuple(sl1)]) < 0                   # junction between nodes i, i+1
    cnt = flip.sum(axis=tuple(a for a in range(3) if a != axis))
    out += list(coords[:-1][cnt > min_frac * other])
    return sorted(set(np.round(out, 9)))


def graded_axis(fine, lo, hi, features, h_fine, h_max, growth):
    """Nodes from the fab's `fine` coordinates in [lo, hi]: spacing
    h_fine + (growth - 1) * distance to the nearest feature, capped at h_max,
    always landing on every feature."""
    pts = fine[(fine >= lo - 1e-9) & (fine <= hi + 1e-9)]
    feats = np.array(sorted({lo, hi, *[f for f in features if lo < f < hi]}))
    out = [pts[0]]
    x = pts[0]
    while x < pts[-1] - 1e-9:
        dist = np.min(np.abs(feats - x))
        step = min(h_max, h_fine + (growth - 1.0) * dist)
        nxt_feat = feats[feats > x + 1e-9][0]
        target = min(x + step, nxt_feat)
        x = pts[pts <= target + 1e-9][-1]
        if x <= out[-1]:
            x = pts[pts > out[-1] + 1e-9][0]
        out.append(x)
    return np.array(out)


def device_grid(st, top_nm=None, half_z=True, h_fine=(2.0, 0.5, 2.0), h_max=(8.0, 8.0, 8.0), growth=1.35,
                min_frac=0.02):
    """Graded node axes (nm) for the device solve, as subsets of the fab grid.
    The domain: the full x range, heights from the bottom up to top_nm above
    the original surface (default: the top of the solid - the whole gate,
    contacts and ILD, whose coupling is much of the off-state
    capacitance), and half the z range when half_z."""
    g = st.grid
    x, y, z = g.axes
    if top_nm is None:
        solid_rows = np.flatnonzero((st.cell_labels() >= 0).any(axis=(0, 2)))
        y_lo = y[solid_rows[0]]
    else:
        y_lo = -top_nm
    y_hi = y[-1]
    z_hi = (z[0] + z[-1]) / 2 if half_z else z[-1]
    jy = (y >= y_lo - 1e-9)
    kz = (z <= z_hi + 1e-9)
    lab = st.cell_labels()[:, jy[:-1] & jy[1:], :][:, :, kz[:-1] & kz[1:]]
    nlab = st.node_labels()[:, jy, :][:, :, kz]
    semi = st.semiconductor_nodes(nlab)
    net = (st.donors - st.acceptors)[:, jy, :][:, :, kz]
    ys, zs = y[jy], z[kz]
    axes = []
    for a, (c, lo, hi) in enumerate(((x, x[0], x[-1]), (ys, y_lo, y_hi), (zs, z[0], z_hi))):
        feats = _features(lab, net, semi, a, c, min_frac)
        axes.append(graded_axis(c, lo, hi, feats, h_fine[a], h_max[a], growth))
    return axes


def _fab_index(axis, coords):
    """Fractional fab-node index of coordinates along a uniform fab axis."""
    return (np.asarray(coords) - axis[0]) / (axis[1] - axis[0])


def sample_structure(st, axes_nm):
    """Per-cell region index (argmin of the region level sets at the cell
    centre, -1 = gas) and net doping at the nodes, on the device grid."""
    g = st.grid
    centres = [(a[:-1] + a[1:]) / 2 for a in axes_nm]
    C = np.meshgrid(*[_fab_index(g.axes[a], centres[a]) for a in range(3)], indexing="ij")
    coords = np.stack([c.ravel() for c in C])
    best = np.full(coords.shape[1], 0.0)
    lab = np.full(coords.shape[1], -1, dtype=np.int64)
    for k, r in enumerate(st.regions):
        v = ndimage.map_coordinates(r.phi, coords, order=1, mode="nearest")
        take = v < best
        lab[take] = k
        best = np.where(take, v, best)
    idx = [np.rint(_fab_index(g.axes[a], axes_nm[a])).astype(int) for a in range(3)]
    for a in range(3):
        if not np.allclose(g.axes[a][idx[a]], axes_nm[a]):
            raise ValueError("device grid nodes must be fab nodes")
    net = (st.donors - st.acceptors)[np.ix_(*idx)]
    return lab.reshape([len(c) for c in centres]), net


def device_materials(st):
    """One mesh material per fab region: semiconductor, insulator (eps from
    the table) or conductor (the contact whose region it is)."""
    by_region = {c["region"]: c for c in st.contacts if c.get("region")}
    out = []
    for r in st.regions:
        if is_semiconductor(r.material):
            out.append(dict(name=r.name, kind=SEMI))
        elif r.material in CONDUCTORS:
            if r.name not in by_region:
                raise ValueError(f"conductor region {r.name!r} ({r.material}) is not a contact in the device layer")
            out.append(dict(name=r.name, kind=CONDUCTOR, contact=by_region[r.name]["name"]))
        else:
            out.append(dict(name=r.name, kind=INSULATOR, eps_r=INSULATOR_EPS.get(r.material, 3.9)))
    return out


def device_voxels(st, half_z=True):
    """The device domain as voxels of the fab grid: (labels, node net
    doping, origin nm, voxel nm). Heights from the bottom to the top of the
    solid; half the width (to the symmetry plane) when half_z."""
    g = st.grid
    lab = st.cell_labels()
    rows = np.flatnonzero((lab >= 0).any(axis=(0, 2)))
    j0 = int(rows[0])
    nz = lab.shape[2]
    if half_z and nz % 2:
        raise ValueError("half-width solve needs an even number of voxels across z")
    kz = nz // 2 if half_z else nz
    net = (st.donors - st.acceptors)[:, j0:, :kz + 1]
    origin = np.array([g.axes[0][0], g.axes[1][j0], g.axes[2][0]])
    return lab[:, j0:, :kz], net, origin, np.asarray(g.h, float)


def build_device(cfg, top_nm=None, half_z=True, log=print, grid=None, mesher="cartesian"):
    """Fab build -> device mesh. Returns a dict with the mesh, the silicon
    material, the gate workfunction, the structure, axes and bookkeeping.
    mesher: "cartesian" (graded rectilinear, the reference) or "octree"
    (mesh3d/octree.py: adaptive boxes, exact Voronoi dual, quality gate)."""
    t0 = time.perf_counter()
    st, _ = flow.build(cfg, log=lambda *a: None)
    fl = flow.expand(cfg)
    p = resolve_params({k: v for k, v in fl["params"].items() if v is not None})
    if mesher == "octree":
        return _build_device_octree(st, p, half_z, log, grid or {}, t0)
    axes_nm = device_grid(st, top_nm=top_nm, half_z=half_z, **(grid or {}))
    lab, net = sample_structure(st, axes_nm)
    mats = device_materials(st)
    mat = _resolve_material_block({})
    faces = {c["name"]: (1, "max") for c in st.contacts if c.get("surface") == "bottom"}
    mesh = build_cartesian_mesh([a * NM_TO_CM for a in axes_nm], lab, mats, net, mat, face_contacts=faces)
    gate = next(c for c in st.contacts if c.get("type") == "gate")
    wf = evaluate(gate["workfunction_eV"], p)
    body = next(c["name"] for c in st.contacts if c.get("surface") == "bottom")
    body_nodes = mesh.boundary_point_index[np.array(mesh.boundary_bc_type) == f"contact:{body}"]
    Cs = float(np.median(mesh.Cdop[body_nodes]))
    zmid = axes_nm[2][-1] if half_z else None
    W_nm = evaluate("W", p) if "W" in p else None
    log(f"device mesh: {' x '.join(str(len(a)) for a in axes_nm)} grid, {len(mesh.points)} nodes, "
        f"{len(mesh.edges)} edges; terminals "
        + ", ".join(f"{c.name} {int((np.array(mesh.boundary_bc_type) == 'contact:' + c.name).sum())}"
                    for c in mesh.domain.contacts)
        + f"; gate wf {wf:.2f} eV ({time.perf_counter() - t0:.1f} s)")
    return dict(mesh=mesh, mat=mat, wf=wf, Cs=Cs, st=st, axes_nm=axes_nm, cell_region=lab, half_z=half_z,
                zmid=zmid, W_nm=W_nm, L_nm=evaluate("Lg", p) if "Lg" in p else None, params=p)


def _build_device_octree(st, p, half_z, log, opts, t0):
    from mesh3d.octree import build_octree_mesh
    lab, net, origin, h = device_voxels(st, half_z)
    mats = device_materials(st)
    mat = _resolve_material_block({})
    mesh = build_octree_mesh(lab, mats, h, net, mat, origin_nm=origin, face_contacts={
        c["name"]: (1, "max") for c in st.contacts if c.get("surface") == "bottom"}, log=log, **opts)
    gate = next(c for c in st.contacts if c.get("type") == "gate")
    wf = evaluate(gate["workfunction_eV"], p)
    body = next(c["name"] for c in st.contacts if c.get("surface") == "bottom")
    body_nodes = mesh.boundary_point_index[np.array(mesh.boundary_bc_type) == f"contact:{body}"]
    Cs = float(np.median(mesh.Cdop[body_nodes]))
    log(f"device mesh (octree): {len(mesh.points)} nodes, {len(mesh.edges)} edges; terminals "
        + ", ".join(f"{c.name} {int((np.array(mesh.boundary_bc_type) == 'contact:' + c.name).sum())}"
                    for c in mesh.domain.contacts)
        + f"; gate wf {wf:.2f} eV ({time.perf_counter() - t0:.1f} s)")
    return dict(mesh=mesh, mat=mat, wf=wf, Cs=Cs, st=st, axes_nm=None, cell_region=None, half_z=half_z,
                zmid=None, W_nm=evaluate("W", p) if "W" in p else None,
                L_nm=evaluate("Lg", p) if "Lg" in p else None, params=p, mesher="octree",
                voxels=(lab, origin, h))


# --- solving ------------------------------------------------------------------------------
_CTX = {}


def _region_info(d):
    """(region names, sorted material names, material index per region) -
    from the structure, or from d["region_info"] where the structure was
    dropped (worker processes get no level sets)."""
    if "region_info" not in d:
        regions = d["st"].regions
        materials = sorted({r.material for r in regions})
        d["region_info"] = ([r.name for r in regions], materials,
                            np.array([materials.index(r.material) for r in regions]))
    return d["region_info"]


def schottky_contacts(d, log=print):
    """The sub-grid Schottky boundary condition (contacts/subgrid.py) for
    every `type: schottky` contact of the device, or None. Interface areas
    come from the same voxels / cells the mesh was built from."""
    from contacts.schottky import SchottkyContact
    from contacts.subgrid import SubgridContacts, voxel_interface_areas
    st, p, mesh = d["st"], d["params"], d["mesh"]
    by_name = {}
    for c in st.contacts:
        if c.get("type") != "schottky":
            continue
        kw = {k: evaluate(c[k], p) for k in ("barrier_n_eV", "barrier_p_eV") if c.get(k) is not None}
        if not kw:
            region = next(r for r in st.regions if r.name == c["region"])
            kw["metal"] = region.material
        by_name[c["name"]] = SchottkyContact(**kw)
    if not by_name:
        return None
    mats = device_materials(st)
    kinds = [m["kind"] for m in mats]
    contact_of = [m.get("contact") for m in mats]
    if d.get("mesher") == "octree":
        lab, origin, h = d["voxels"]
        axes_cm = [(origin[a] + h[a] * np.arange(lab.shape[a] + 1)) * NM_TO_CM for a in range(3)]
    else:
        lab = d["cell_region"]
        axes_cm = [np.asarray(a) * NM_TO_CM for a in d["axes_nm"]]
    area = voxel_interface_areas(lab, kinds, contact_of, axes_cm, mesh, list(by_name))
    sk = SubgridContacts.setup(mesh, d["mat"], by_name, area, log=log)
    log("Schottky contacts: " + ", ".join(
        f"{nm} ({s['carrier']}, Phi_Bn {by_name[nm].barriers(d['mat'].chi_eV, d['mat'].Eg_eV)[0]:.2f} eV, "
        f"{len(s['nodes'])} nodes, {area[s['nodes']].sum() * 1e14:.0f} nm^2)" for nm, s in sk.specs.items()))
    return sk


def _context(cfg_path, top_nm, half_z, mesher="cartesian"):
    """Device, mesh and equilibrium, built once per process (a worker
    started by simulate() gets it from _init_worker instead)."""
    key = (cfg_path, top_nm, half_z, mesher)
    if key not in _CTX:
        from main2d_mosfet_sweep import solve_mosfet
        from mos.mos_params import MOSDevice
        d = build_device(flow.load(cfg_path), top_nm=top_nm, half_z=half_z, log=lambda *a: None, mesher=mesher)
        d["dev"] = MOSDevice(gate_workfunction_eV=d["wf"])
        d["schottky"] = schottky_contacts(d, log=lambda *a: None)
        d["eq"] = solve_mosfet(d["mesh"], d["mat"], d["dev"], d["Cs"], 0.0, 0.0, velocity_saturation=True,
                               schottky=d["schottky"])
        _CTX[key] = d
    return _CTX[key]


def _init_worker(key, d):
    _CTX[key] = d


def drain_current(d, r):
    """Total drain current of the full device (A), from the solved half."""
    from solver2d.current import contact_current
    I = contact_current(d["mesh"], d["mat"], r, "drain")
    return I * (2.0 if d["half_z"] else 1.0)


def _solver_at(d, Vgs, Vds):
    from main2d_mosfet_sweep import solve_mosfet
    mesh, mat, dev, Cs = d["mesh"], d["mat"], d["dev"], d["Cs"]
    return lambda pp, gss: solve_mosfet(mesh, mat, dev, Cs, *((pp, Vds) if Vgs is None else (Vgs, pp)),
                                        psi_init=gss["psi"], phin_init=gss["phin"], phip_init=gss["phip"],
                                        cold_retry=False, velocity_saturation=True, vsat_fallback=False,
                                        schottky=d.get("schottky"))


def _write_point(task, d, v, state):
    """One solved point as a .vtu (if task['vtk'] asks for it); its .pvd
    entry or None."""
    if not (task["vtk"] == "all" or v in task.get("vtk_keep", ())):
        return None
    os.makedirs(task["vtk_dir"], exist_ok=True)
    writer = write_state_octree if d.get("mesher") == "octree" else write_state
    f = f"vds{task['Vds']:g}_vgs{v:+.3f}.vtu"
    name = f"Vgs={v:+.3f}V Vds={task['Vds']:+.3f}V"
    writer(os.path.join(task["vtk_dir"], f), d, state, {"gate": v, "drain": task["Vds"]}, name)
    return dict(file=f, timestep=v, name=name)


# Gate-walk step control (DEVELOPMENT_LOG 57): extrapolate only psi (the
# gate moves psi; the quasi-Fermi levels stay pinned by the source/drain
# contacts), start at 0.05 V (a first 0.1 V step at Vds = 1 V failed after
# 15 iterations), and double the step after <= 5 iterations (the psi-only
# steps take 5). Full planar device, Vds = 1 V walk: 119 -> 69 iterations.
GATE_WALK = dict(h_init=0.05, fast_iters=5, psi_only=True)


def run_curve(task):
    """One Id-Vg curve (or a segment of one) at fixed task['Vds']: from the curve's
    Vgs = 0 state (task['base'] key into the context, else ramped here from
    equilibrium), walk by continuation through task['vgs'] - the walk from
    0 to the first value is adaptive and not recorded."""
    from main2d_mosfet_sweep import continuation
    t0 = time.perf_counter()
    d = _context(task["config"], task["top_nm"], task["half_z"], task.get("mesher", "cartesian"))
    Vds = task["Vds"]
    base = d.get("base", {}).get(Vds)
    it0 = 0
    if base is None:
        ramp, it0, _ = continuation(_solver_at(d, 0.0, None), 0.0, d["eq"], [Vds])
        if Vds not in ramp:
            return dict(task=task, points={}, entries=[], iters=it0, time_s=time.perf_counter() - t0)
        base = ramp[Vds]
    vgs = sorted(v for v in task["vgs"] if v > 0)
    # Each point's .vtu is written in a background thread while Newton
    # walks on (zlib and most of numpy release the GIL): the writes were
    # ~0.4-0.6 s x 13 on the slowest curve's path (DEVELOPMENT_LOG 59).
    with ThreadPoolExecutor(max_workers=1) as io:
        pending = []

        def on_point(v, r):
            pending.append(io.submit(_write_point, task, d, v, r))
        if task.get("include_zero", True):
            on_point(0.0, base)
        states, it1, _ = continuation(_solver_at(d, None, Vds), 0.0, base, vgs, on_point=on_point,
                                      **GATE_WALK)
        if task.get("include_zero", True):
            states = {0.0: base, **states}
        pts = {v: drain_current(d, r) for v, r in states.items()}
        entries = sorted((e for e in (f.result() for f in pending) if e), key=lambda e: e["timestep"])
    return dict(task=task, points=pts, entries=entries, iters=it0 + it1, time_s=time.perf_counter() - t0)


# --- output ---------------------------------------------------------------------------------
def _mirror(a, axis):
    """Append the mirror image along `axis` (the last plane is the symmetry plane)."""
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(-2, None, -1)
    return np.concatenate([a, a[tuple(sl)]], axis=axis)


def write_state(path, d, state, bias, label):
    """A solved state on the device grid as a hexahedral .vtu (full width,
    mirrored), with the solver2d field names; conductor and insulator cells
    included so the gate and contacts show."""
    from solver2d.vtk_export import FIELD_UNITS, state_point_data
    mesh, axes_nm = d["mesh"], d["axes_nm"]
    shape = tuple(len(a) for a in axes_nm)
    pd_mesh = state_point_data(mesh, d["mat"], state)
    grid = {}
    for k, v in pd_mesh.items():
        v = np.asarray(v)
        a = np.full(shape + v.shape[1:], np.nan)
        a[tuple(mesh.grid_index.T)] = v
        grid[k] = a
    cell = d["cell_region"]
    ax = [a.copy() for a in axes_nm]
    if d["half_z"]:
        nz = shape[2]
        grid = {k: _mirror(v, 2) for k, v in grid.items()}
        for v in grid.values():
            if v.ndim == 4:                       # a vector: its z component flips in the mirror image
                v[:, :, nz:, 2] *= -1.0
        cell = np.concatenate([cell, cell[:, :, ::-1]], axis=2)
        ax[2] = np.concatenate([ax[2], 2 * ax[2][-1] - ax[2][-2::-1]])
    shape = tuple(len(a) for a in ax)
    # cell field E = -grad psi from the 8 corners
    psi = grid["ElectrostaticPotential"]
    E = []
    for a in range(3):
        dpsi = np.diff(psi, axis=a)
        h = np.diff(ax[a]).reshape([-1 if b == a else 1 for b in range(3)]) * NM_TO_CM
        g = dpsi / h
        for b in range(3):
            if b != a:
                g = 0.5 * (np.take(g, range(0, g.shape[b] - 1), axis=b) + np.take(g, range(1, g.shape[b]), axis=b))
        E.append(-g)
    E = np.stack(E, axis=-1)
    keep = cell >= 0
    ci = np.argwhere(keep)
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    conn = np.stack([np.ravel_multi_index(tuple((ci + np.array(o)).T), shape) for o in corners], axis=1)
    used, inv = np.unique(conn, return_inverse=True)
    conn = inv.reshape(conn.shape)
    gi = np.unravel_index(used, shape)
    pts = np.column_stack([ax[a][gi[a]] for a in range(3)]) * 1e-3          # um
    pdat = {k: v[gi] for k, v in grid.items()}
    region_names, materials, region_mat = _region_info(d)
    Ec = E[keep]
    cnames = [c.name for c in mesh.domain.contacts]
    vtk_io.write_vtu(path, pts, conn, vtk_io.VTK_HEXAHEDRON, point_data=pdat,
                     cell_data={"region_id": cell[keep].astype(np.int32),
                                "material_id": region_mat[cell[keep]].astype(np.int32),
                                "ElectricField": Ec, "AbsElectricField": np.sqrt((Ec ** 2).sum(axis=1))},
                     field_data={"region_names": region_names, "material_names": materials,
                                 "contact_names": cnames,
                                 "contact_bias": np.array([float(bias.get(c, 0.0)) for c in cnames]),
                                 "label": label, "field_units": [f"{k}:{v}" for k, v in FIELD_UNITS.items()]},
                     compress=True)


def write_state_octree(path, d, state, bias, label):
    """A solved state on the octree mesh as a hexahedral .vtu: every
    device and conductor leaf a cell, point fields at the leaf corners
    (the device nodes' values; NaN inside conductors), mirrored to the full
    width for a half-width solve."""
    from solver2d.vtk_export import FIELD_UNITS, state_point_data
    mesh = d["mesh"]
    if "vtu_geom" not in d:                       # the cells and points depend on the mesh only: once
        d["vtu_geom"] = _octree_vtu_geometry(d)
    inv, dev, mirrored, pts, cells_lab = d["vtu_geom"]
    pd_mesh = state_point_data(mesh, d["mat"], state)
    pdat = {}
    for k, v in pd_mesh.items():
        v = np.asarray(v)
        ok = (dev >= 0).reshape((-1,) + (1,) * (v.ndim - 1))
        a = np.where(ok, v[np.maximum(dev, 0)], np.nan)
        if v.ndim == 2:                           # a vector: its z component flips in the mirror image
            a[mirrored, 2] *= -1.0
        pdat[k] = a
    region_names, materials, region_mat = _region_info(d)
    cnames = [c.name for c in mesh.domain.contacts]
    vtk_io.write_vtu(path, pts, inv.reshape(-1, 8), vtk_io.VTK_HEXAHEDRON, point_data=pdat,
                     cell_data={"region_id": cells_lab.astype(np.int32),
                                "material_id": region_mat[cells_lab].astype(np.int32)},
                     field_data={"region_names": region_names, "material_names": materials,
                                 "contact_names": cnames,
                                 "contact_bias": np.array([float(bias.get(c, 0.0)) for c in cnames]),
                                 "label": label, "field_units": [f"{k}:{v}" for k, v in FIELD_UNITS.items()]},
                     compress=True)


def _octree_vtu_geometry(d):
    """write_state_octree's mesh-only part: the leaf hexahedra (mirrored for
    a half-width solve), their unique corner points, and each point's
    device node (-1 inside conductors)."""
    mesh = d["mesh"]
    lab, origin, h = d["voxels"]
    LO, HI, leaf_label = mesh.leaves
    keep = leaf_label >= 0
    LO, HI, leaf_label = LO[keep], HI[keep], leaf_label[keep]
    nshape = tuple(np.array(lab.shape) + 1)
    gid = np.full(nshape, -1, dtype=np.int64)
    gid[tuple(mesh.grid_index.T)] = np.arange(len(mesh.grid_index))
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    cidx = np.stack([np.stack([np.where(o[a], HI[:, a], LO[:, a]) for a in range(3)], axis=1) for o in corners],
                    axis=1)                                             # (L, 8, 3) node grid indices
    cells_lab = leaf_label
    if d["half_z"]:
        zmax = lab.shape[2]
        mir = cidx.copy()
        mir[:, :, 2] = 2 * zmax - mir[:, :, 2]
        mir = mir[:, [3, 2, 1, 0, 7, 6, 5, 4]]                          # keep the hexahedra right-handed
        cidx = np.concatenate([cidx, mir])
        cells_lab = np.concatenate([leaf_label, leaf_label])
    flat = cidx.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    src = uniq.copy()
    if d["half_z"]:
        src[:, 2] = np.where(src[:, 2] > lab.shape[2], 2 * lab.shape[2] - src[:, 2], src[:, 2])
    dev = gid[tuple(src.T)]
    mirrored = uniq[:, 2] > lab.shape[2] if d["half_z"] else np.zeros(len(uniq), bool)
    pts = (origin + uniq * h) * 1e-3                                    # um
    return inv, dev, mirrored, pts, cells_lab


def simulate(config, out_dir, vds=(0.05, 1.0), vgs_max=1.2, vgs_step=0.1, top_nm=None, half_z=True,
             vtk="all", workers=None, log=print, mesher="cartesian"):
    from solver2d.mosfet_metrics import mosfet_metrics
    # The linear solver is core/linsolve's "auto" (PETSc block ILU at these
    # sizes) unless TCAD_LINSOLVE says otherwise. This used to force direct
    # LU (MUMPS): ILU-GMRES gave Newton bad directions on the rectilinear
    # mesh and equilibrium never converged. With the quasi-Fermi box
    # (DEVELOPMENT_LOG 55) and the reuse budget (56) ILU converges there to
    # the same currents, 6.6x faster (40 s vs 263 s, one Id-Vg walk).
    t0 = time.perf_counter()
    from core.parallel import performance_cores
    from main2d_mosfet_sweep import continuation
    key = (config, top_nm, half_z, mesher)
    d = _context(config, top_nm, half_z, mesher)                 # mesh + equilibrium, once
    _region_info(d)
    log(f"  mesh + equilibrium: {time.perf_counter() - t0:.0f} s")
    # the drain ramps at Vgs = 0 (off: the gentlest place to raise Vds), once
    ramps, _, _ = continuation(_solver_at(d, 0.0, None), 0.0, d["eq"], sorted(vds))
    d["base"] = ramps
    log(f"  drain ramps: {time.perf_counter() - t0:.0f} s")
    vgs = list(np.round(np.arange(vgs_step, vgs_max + 1e-9, vgs_step), 6))
    # One worker per curve. Splitting a curve into segments (each walking
    # from Vgs = 0 to its first point) cannot shorten it: a segment's walk
    # takes the same continuation steps as the curve, so the last segment
    # costs about the whole curve, and 6 busy processes slow each other
    # (a walk ran ~1.7x slower). Full planar device, octree: 92 s split
    # vs 83 s per curve; rectilinear 65 vs 56 s (DEVELOPMENT_LOG 57).
    tasks = []
    for v in vds:
        if v not in ramps:
            log(f"  Vds={v:g} V: drain ramp failed - curve skipped")
            continue
        tasks.append(dict(config=config, Vds=v, vgs=vgs, top_nm=top_nm, half_z=half_z, vtk=vtk,
                          vtk_keep=(0.0, max(vgs)), mesher=mesher, vtk_dir=os.path.join(out_dir, "vtk")))
    n_work = workers or performance_cores()
    ctx = {k: v for k, v in d.items() if k != "st"}               # no level sets to the workers
    # one BLAS / OpenMP thread per worker process (inherited at spawn)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=min(n_work, len(tasks)), initializer=_init_worker,
                             initargs=(key, ctx)) as ex:
        curve_results = list(ex.map(run_curve, tasks))
    results = []
    for v in vds:
        parts = [r for r in curve_results if r["task"]["Vds"] == v]
        if not parts:
            continue
        pts, entries = {}, {}
        for r in parts:
            pts.update(r["points"])
            for e in r["entries"]:
                entries.setdefault(e["timestep"], e)
        results.append(dict(task=dict(Vds=v), points=pts, entries=[entries[k] for k in sorted(entries)],
                            iters=sum(r["iters"] for r in parts), time_s=max(r["time_s"] for r in parts)))
    W_um = (d["W_nm"] or 1e3) * 1e-3
    curves = []
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        v = sorted(r["points"])
        ids = [r["points"][x] / W_um for x in v]
        curves.append(dict(Vds=r["task"]["Vds"], Vgs=np.array(v), Id=np.array(ids)))
        log(f"  Vds={r['task']['Vds']:g} V: {len(v)} points, {r['iters']} Newton iterations, "
            f"{r['time_s']:.0f} s")
        if r.get("entries"):
            vtk_io.write_pvd(os.path.join(out_dir, "vtk", f"idvg_Vds{r['task']['Vds']:g}.pvd"), r["entries"])
    with open(os.path.join(out_dir, "ids_vgs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Vds_V", "Vgs_V", "Id_A_per_um"])
        for c in curves:
            for a, b in zip(c["Vgs"], c["Id"]):
                w.writerow([c["Vds"], a, b])
    m = None
    if len(curves) >= 2 and all(len(c["Vgs"]) > 3 for c in curves):
        m = mosfet_metrics(curves[0], curves[-1], (d["L_nm"] or 1e3) * 1e-3)
        log(f"  Vt,cc = {m['Vt_cc_lo']:.3f} V -> {m['Vt_cc_hi']:.3f} V, DIBL = {m['DIBL_mV_per_V']:.0f} mV/V, "
            f"SS = {m['SS_lo']:.0f} mV/dec")
        log(f"  Ion = {m['Ion'] * 1e6:.0f} uA/um, Ioff = {m['Ioff']:.2e} A/um, Ion/Ioff = {m['Ion_Ioff']:.2e}")
    _plot(curves, m, os.path.join(out_dir, "ids_vgs.png"), d)
    log(f"total {time.perf_counter() - t0:.0f} s -> {out_dir}")
    return curves, m


def _plot(curves, m, path, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for c in curves:
        a1.semilogy(c["Vgs"], np.abs(c["Id"]), "o-", ms=3, label=f"Vds = {c['Vds']:g} V")
        a2.plot(c["Vgs"], c["Id"] * 1e6, "o-", ms=3, label=f"Vds = {c['Vds']:g} V")
    for a in (a1, a2):
        a.set_xlabel("Vgs (V)")
        a.grid(True, alpha=0.3)
        a.legend()
    a1.set_ylabel("Id (A/um)")
    a2.set_ylabel("Id (uA/um)")
    p = d["params"]
    title = f"3D planar {p.get('type', '')}: Lg {p.get('Lg', 0):g} nm, W {p.get('W', 0):g} nm, tox {p.get('tox', 0):g} nm"
    if m:
        title += f"  |  Vt {m['Vt_cc_lo']:.2f} V, SS {m['SS_lo']:.0f} mV/dec, DIBL {m['DIBL_mV_per_V']:.0f} mV/V"
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)
