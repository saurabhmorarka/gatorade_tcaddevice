"""Process stress -> low-field mobility on the device mesh (piezoresistance).

For every silicon edge of the mesh (both nodes silicon), with unit
direction t (device frame) and the process stress sigma sampled at the
edge midpoint (mechanics/stress_field.py, the silicon side of any
interface):

    x_c = -P_c(N) t.(Pi_c:sigma).t          c = n, p
    mu_c0(edge) *= f_c(x_c)                 (mechanics/piezo.py: linear or saturating)

Pi_c is the full cubic piezoresistance tensor rotated to the device frame
(default (001) wafer, <110> channel), P_c(N) the optional doping factor at
the edge (the mean of its nodes' |net doping|). The factors are computed
once per mesh and attached with set_edge_mobility_factor: the stress does
not depend on the solution, so the Newton Jacobian only carries a constant
factor and the solve costs the same. Edges in SiGe or touching an
insulator keep factor 1 (no SiGe coefficients yet; no carrier flux).

The model is described in docs/piezo_mobility.md.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from mechanics import piezo
from solver2d.newton_solver_qf_2d import node_materials, set_edge_mobility_factor

NM_PER_CM = 1e7


@dataclass
class PiezoModel:
    mode: str = "saturating"         # "saturating" | "linear"
    doping_factor: bool = True
    coeffs: object = "smith"         # a mechanics.piezo.COEFFS name or {carrier: (pi11, pi12, pi44)}
    frame: object = None             # device x, y, z in crystal axes (default (001) / <110>)
    T: float = 300.0
    f_max_n: float = None            # saturation limits (None: mechanics.piezo.SATURATION)
    f_min_n: float = None
    f_max_p: float = None
    f_min_p: float = None

    @classmethod
    def from_options(cls, opts):
        """From a config / CLI dict: piezo (mode), doping_factor, coeffs, the
        limits."""
        o = dict(opts or {})
        kw = {}
        if "piezo" in o:
            kw["mode"] = o.pop("piezo")
        for k in ("mode", "doping_factor", "coeffs", "frame", "T", "f_max_n", "f_min_n", "f_max_p", "f_min_p"):
            if k in o:
                kw[k] = o[k]
        m = cls(**kw)
        if m.mode not in piezo.MODES:
            raise ValueError(f"piezo mode must be one of {piezo.MODES}, not {m.mode!r}")
        return m


def silicon_nodes(mesh, mat):
    """Nodes whose semiconductor is silicon (by band gap and affinity: the
    meshes carry Material objects, not names) and not an insulator."""
    from core.config import _resolve_material_block
    si = _resolve_material_block({})
    mats, idx = node_materials(mesh, mat)
    is_si = np.array([abs(m.Eg_eV - si.Eg_eV) < 1e-9 and abs(m.chi_eV - si.chi_eV) < 1e-9 for m in mats])
    out = np.full(len(mesh.points), bool(is_si[0])) if idx is None else is_si[idx]
    if getattr(mesh, "is_insulator", None) is not None:
        out &= ~mesh.is_insulator
    return out


def edge_geometry(mesh):
    """(midpoints nm (E, 3), unit directions (E, 3)) in the device frame; a
    2D mesh's z is 0."""
    P = np.asarray(mesh.points, float) * NM_PER_CM
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    d = P[jj] - P[ii]
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    return 0.5 * (P[ii] + P[jj]), t


def stress_factors(mesh, mat, field, model=None, materials=("Si",), check_tol=0.005):
    """Per-edge (f_n, f_p) and a report dict. Raises if the field does not
    cover the mesh or more than check_tol of the silicon edges sample a
    voxel that is not silicon in the stress structure (a frame / alignment
    mismatch between the stress grid and the device mesh)."""
    model = model or PiezoModel()
    E = len(mesh.edges)
    si = silicon_nodes(mesh, mat)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    on = si[ii] & si[jj]
    mid, t = edge_geometry(mesh)
    f_n, f_p = np.ones(E), np.ones(E)
    rep = dict(n_edges=E, n_si_edges=int(on.sum()))
    if not on.any():
        return f_n, f_p, rep
    m_on = mid[on]
    inside = field.inside(m_on)
    if not inside.all():
        bad = m_on[~inside]
        raise ValueError(f"{int((~inside).sum())} silicon edge midpoints lie outside the stress grid "
                         f"(e.g. {np.round(bad[0], 2)} nm; grid {field.lo} + {np.array(field.shape) * field.h} nm)")
    # the frame check: the voxel each silicon midpoint falls in must be silicon in
    # the stress structure (a point on a voxel face may take either side: the
    # check accepts any of the voxels touching it). The stress voxels are
    # coarser than the fab's (1 nm vs 0.5 nm: majority labels), so a curved
    # or sub-voxel silicon boundary moves by up to one voxel: "exact" counts
    # every miss, the check itself allows one voxel and fails above
    # check_tol (a real frame shift moves whole surfaces by several voxels)
    vox_si = field.material_mask(materials)
    near_si = ndimage.binary_dilation(vox_si, np.ones((3,) * field.dim, bool))
    hit = np.zeros(len(m_on), bool)
    hit1 = np.zeros(len(m_on), bool)
    for s in np.ndindex(*(2,) * field.dim):
        off = (np.array(s) - 0.5) * 1e-6 * field.h
        v = tuple(field.voxel_of(m_on[:, :field.dim] + off).T)
        hit |= vox_si[v]
        hit1 |= near_si[v]
    rep["frame_mismatch_exact"] = float(1.0 - hit.mean())
    miss = float(1.0 - hit1.mean())
    rep["frame_mismatch"] = miss
    if miss > check_tol:
        raise ValueError(f"stress field / device mesh frame check failed: {100 * miss:.2f}% of the silicon "
                         f"edges are more than one stress voxel from silicon (limit {100 * check_tol:g}%)")
    sig = field.sample(m_on, materials)
    N = 0.5 * (np.abs(mesh.Cdop[ii[on]]) + np.abs(mesh.Cdop[jj[on]]))
    out = {}
    for c in ("n", "p"):
        Pi = piezo.pi_tensor(c, model.frame, model.coeffs)
        x = piezo.linear_change(t[on], sig, Pi)
        if model.doping_factor:
            x = x * piezo.doping_factor(N, c, model.T)
        lim = dict(f_max=getattr(model, f"f_max_{c}"), f_min=getattr(model, f"f_min_{c}"))
        out[c] = piezo.factor(x, c, model.mode, **lim)
        rep[f"x_{c}"] = (float(x.min()), float(x.max()))
    f_n[on], f_p[on] = out["n"], out["p"]
    rep.update(mode=model.mode, doping_factor=model.doping_factor,
               f_n=(float(out["n"].min()), float(out["n"].max())),
               f_p=(float(out["p"].min()), float(out["p"].max())))
    return f_n, f_p, rep


def channel_report(mesh, f_n, f_p, field, mid=None, t=None, box_nm=None):
    """Mean factors and mean stress over the silicon edges along x inside
    box_nm ((x0, x1), (y0, y1)[, (z0, z1)]) - a channel probe for logs."""
    mid, t = edge_geometry(mesh) if mid is None else (mid, t)
    sel = np.abs(t[:, 0]) > 0.99
    for a, (lo, hi) in enumerate(box_nm or ()):
        sel &= (mid[:, a] >= lo) & (mid[:, a] <= hi)
    sel &= (f_n != 1.0) | (f_p != 1.0)
    if not sel.any():
        return None
    sig = field.sample(mid[sel])
    return dict(n_edges=int(sel.sum()), f_n=float(f_n[sel].mean()), f_p=float(f_p[sel].mean()),
                sigma_MPa=(sig.mean(axis=0) * 1e3).round(0).tolist())


def apply_stress_mobility(mesh, mat, field, model=None, log=print, name="stress"):
    """Compute and attach the stress factors to the mesh; returns the report."""
    f_n, f_p, rep = stress_factors(mesh, mat, field, model)
    set_edge_mobility_factor(mesh, name, f_n, f_p)
    if log:
        log(f"stress mobility ({rep.get('mode')}, doping factor {'on' if rep.get('doping_factor') else 'off'}): "
            f"{rep['n_si_edges']} silicon edges, frame mismatch {100 * rep.get('frame_mismatch_exact', 0):.2f}% exact, {100 * rep.get('frame_mismatch', 0):.2f}% beyond one voxel, "
            f"mu_n x {rep.get('f_n', (1, 1))[0]:.2f}..{rep.get('f_n', (1, 1))[1]:.2f}, "
            f"mu_p x {rep.get('f_p', (1, 1))[0]:.2f}..{rep.get('f_p', (1, 1))[1]:.2f}")
    return rep


# --- 2D drivers: the stress: block of a main2d_mosfet_sweep config ------------------
def field_from_config(st, domain=None, log=print):
    """The StressField of a 2D config's stress: block, aligned to the device.
    source: planar_flow (flow, set) or file: <npz>. align: gate (default) moves
    the field so the stress cell's gate centre is on the device gate's, and
    checks the gate lengths agree to one pixel; or offset_nm: [dx, dy]."""
    from mechanics.stress_field import StressField
    if st.get("file"):
        field = StressField.load(st["file"])
    elif st.get("source", "planar_flow") == "planar_flow":
        from mechanics.planar_flow import DEFAULTS, run_flow
        par = dict(DEFAULTS)
        par.update({k: float(v) for k, v in (st.get("set") or {}).items()})
        flow = st.get("flow", "nmos_smt")
        b = run_flow(flow, par=par)
        field = StressField.from_planar_flow(b, flow=flow, par=par)
    else:
        raise ValueError(f"stress source must be planar_flow or a file, not {st.get('source')!r}")
    if "offset_nm" in st:
        return field.shifted(st["offset_nm"])
    if st.get("align", "gate") == "gate":
        gate = next(c for c in domain.contacts if c.name == "gate")
        gx = np.array(gate.x_range_cm) * NM_PER_CM
        cx = np.array(field.meta["gate_x_nm"])
        if abs((gx[1] - gx[0]) - (cx[1] - cx[0])) > field.h + 1e-6:
            raise ValueError(f"stress cell gate length {cx[1] - cx[0]:g} nm != device gate length "
                             f"{gx[1] - gx[0]:g} nm: the stress does not belong to this device")
        dx = gx.mean() - cx.mean()
        if log:
            log(f"stress field aligned on the gate: x shifted by {dx:+.1f} nm")
        return field.shifted([dx, 0.0])
    return field
