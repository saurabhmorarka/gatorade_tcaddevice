"""Top-level 2D mesh builder: combines mesh2d/geometry2d.py (blocky regions/
contacts), mesh2d/pointcloud.py (triangle-library constrained quality
triangulation), mesh2d/fvgeometry.py (Voronoi-box FV geometry),
mesh2d/mesh_quality.py (mandatory quality gate) and mesh2d/boundary.py (BC
tagging) into one mesh object - the 2D analog of
core/mesh.py::build_diode_grid, but device-generic (build_mesh2d takes any
Domain2D, not a diode specifically) since the quality gate and FV assembly
have nothing diode-specific about them and every future 2D device (MOS,
etc.) should get the same mandatory checks for free by going through this
one function."""
import warnings
from dataclasses import dataclass

import numpy as np

from core.params import EPS0
from mesh2d.boundary import tag_boundary_points
from mesh2d.fvgeometry import build_fv_geometry
from mesh2d.mesh_quality import check_mesh_quality
from mesh2d.pointcloud import build_point_cloud
from mesh2d.quadtree import build_quadtree_mesh, interface_and_junction_segments


@dataclass
class Mesh2D:
    domain: object                # geometry2d.Domain2D
    points: np.ndarray             # (N,2) cm
    triangles: np.ndarray          # (M,3) int
    edges: np.ndarray              # (E,2) int, internal edges only
    edge_weight: np.ndarray        # (E,) float
    facet_length: np.ndarray       # (E,) float, cm
    cv_area: np.ndarray            # (N,) cm^2
    boundary_edges: set
    n_negative_subareas: int
    n_floored: int
    quality_report: dict           # see mesh2d/mesh_quality.py::mesh_quality_report
    Cdop: np.ndarray               # (N,) cm^-3, signed net doping per point
    boundary_point_index: np.ndarray   # int, indices into points/Cdop
    boundary_bc_type: list             # str per boundary point
    h_min_cm: float                # the mesh generator's own intended nominal minimum
                                     # spacing - NOT the same as the smallest actual edge
                                     # length; solvers should use this, not a raw
                                     # np.min(edge_lengths), as their normalization
                                     # length scale (see newton_solver_qf_2d.py)
    edge_g: np.ndarray = None      # (E,) float, only set when `mat` is passed to
                                     # build_mesh2d - per-edge, already eps-weighted
                                     # conductance (see fvgeometry.py::build_fv_geometry),
                                     # correct across a heterogeneous-permittivity
                                     # interface (e.g. a MOS capacitor's oxide/
                                     # semiconductor boundary), unlike a homogeneous
                                     # mat.eps * edge_weight.
    ni_arr: np.ndarray = None      # (N,) cm^-3, only set when `mat` is passed - intrinsic
                                     # concentration per point, 0 at an insulator point
                                     # (no mobile carriers there at all).
    is_insulator: np.ndarray = None  # (N,) bool, only set when `mat` is passed.
    facet_length_semi: np.ndarray = None  # (E,) cm - facet length inside semiconductor
                                            # triangles only (carrier current), see
                                            # fvgeometry.py; == facet_length without `mat`
    cv_area_semi: np.ndarray = None       # (N,) cm^2 - control volume inside semiconductor
                                            # (carrier/doping charge, recombination)
    # Heterojunction arrays (only set when `mat` is passed; all relative to
    # that base material): delta_Ei = Xi(node material) - Xi(base) (eV, see
    # core.materials.Xi - enters n = ni exp((psi+dEi-phin)/Vt)), the bandgap,
    # and Ec - Ei = Vt ln(Nc/ni), so Ec = -(psi+dEi) + ec_off and
    # Ev = Ec - Eg in the solver's energy reference. For a single-material
    # device dEi = 0, Eg = mat.Eg_eV, ec_off = Vt ln(Nc/ni) everywhere.
    dEi_arr: np.ndarray = None
    Eg_arr: np.ndarray = None
    ec_off_arr: np.ndarray = None


def _drop_isolated_points(points, triangles, cv_area_floor, eps_tri=None, tri_insulator=None,
                           max_passes=3):
    """A rectangle corner's triangle can occasionally end up with its only
    non-boundary edge not shared by any neighboring triangle - a point with
    ZERO internal edges gets no Poisson/continuity coupling to the rest of
    the system at all, making its own 3x3 local Jacobian block exactly
    singular (confirmed directly: scipy.sparse.linalg.spsolve raised
    MatrixRankWarning: Matrix is exactly singular with such a point
    present). These points carry no physically meaningful area anyway, so
    the fix is to drop them and rebuild the FV geometry from the remaining
    triangles (iterating in case that ever isolates another point).

    eps_tri, if given, is dropped/reindexed in lockstep with `triangles` so
    a later fv rebuild's per-triangle permittivity array stays aligned."""
    for _ in range(max_passes):
        fv = build_fv_geometry(points, triangles=triangles, cv_area_floor=cv_area_floor, eps_tri=eps_tri,
                               tri_insulator=tri_insulator)
        deg = np.zeros(len(points), dtype=int)
        np.add.at(deg, fv.edges[:, 0], 1)
        np.add.at(deg, fv.edges[:, 1], 1)
        isolated = deg == 0
        if not np.any(isolated):
            return fv
        keep = ~isolated
        new_index = np.cumsum(keep) - 1
        points = points[keep]
        tri_keep = ~np.any(isolated[triangles], axis=1)
        triangles = new_index[triangles[tri_keep]]
        if eps_tri is not None:
            eps_tri = eps_tri[tri_keep]
        if tri_insulator is not None:
            tri_insulator = tri_insulator[tri_keep]
    raise RuntimeError(
        f"mesh2d: {np.sum(isolated)} isolated point(s) remained after {max_passes} "
        "drop-and-rebuild passes - a persistent degeneracy, not a one-off corner quirk; "
        "needs investigation rather than dropping further points.")


def build_mesh2d(domain, h_min_cm, h_max_cm, growth=1.3, cv_area_floor_factor=0.1,
                  min_angle_deg=32, interface_segments=None, mat=None,
                  mesh_style="unstructured", refine_boxes=(), interface_h_cm=None,
                  junction_h_cm=None):
    """Build a Mesh2D for any Domain2D (blocky regions/contacts) - the
    single entry point every 2D device driver should go through, so the
    mandatory quality gate (mesh2d/mesh_quality.py) and FV-robustness
    regularizations (cv_area_floor, isolated-point removal) apply to every
    kind of run, not just the diode example that first exercised them.

    `interface_segments`, if given, overrides the default grading target
    (domain.junction_segments(), i.e. every doping-type boundary) - e.g. a
    future oxide/semiconductor interface that should drive its own tight-
    near/relaxed-away mesh grading independent of doping-junction geometry.

    cv_area_floor_factor sets the box-method control-volume area floor
    (mesh2d/fvgeometry.py::build_fv_geometry) as a fraction of h_min_cm^2.

    `mat`, if given (a core.params.Material), turns on the heterogeneous-
    permittivity machinery needed once a domain has an "insulator" region
    (e.g. a MOS capacitor's oxide): per-triangle eps is looked up from
    domain.material_props_at at each triangle's centroid, producing
    Mesh2D.edge_g (eps-weighted conductance, see fvgeometry.py) and
    Mesh2D.ni_arr/is_insulator (0/True at an insulator point). Omitting it
    (the default) reproduces today's homogeneous-silicon diode mesh
    byte-for-bit - solvers for a single-material device keep using
    mat.eps * mesh.edge_weight directly, unaffected by this parameter.

    mesh_style="quadtree" (see mesh2d/quadtree.py) builds a balanced
    square-quadtree mesh instead: a coarse h_max_cm background, refined to
    h inside each (x0, x1, y0, y1, h) entry of `refine_boxes`, and graded
    from interface_h_cm at every semiconductor/insulator interface and
    from junction_h_cm at every metallurgical junction (each defaults to
    h_min_cm). Every triangle is 45-45-90, so here an obtuse triangle is a
    hard error rather than a warning. `interface_segments`/`min_angle_deg`
    only apply to the unstructured style."""
    if mesh_style == "quadtree":
        iface, junc = interface_and_junction_segments(domain)
        graded = []
        if iface:
            graded.append((iface, interface_h_cm or h_min_cm))
        if junc:
            graded.append((junc, junction_h_cm or h_min_cm))
        points, triangles, _info = build_quadtree_mesh(
            domain, h_max_cm, refine_boxes=refine_boxes, graded_segments=graded, growth=growth)
        quality_report = check_mesh_quality(points, triangles, forbid_obtuse=True)
    elif mesh_style == "unstructured":
        points, triangles = build_point_cloud(
            domain, h_min_cm, h_max_cm, growth=growth, min_angle_deg=min_angle_deg,
            interface_segments=interface_segments)
        quality_report = check_mesh_quality(points, triangles)
    else:
        raise ValueError(f"mesh_style must be 'unstructured' or 'quadtree', got {mesh_style!r}")

    eps_tri = tri_insulator = None
    if mat is not None:
        centroids = points[triangles].mean(axis=1)
        tri_insulator, eps_r_tri = domain.material_props_at(centroids[:, 0], centroids[:, 1], mat)
        eps_tri = eps_r_tri * EPS0

    fv = _drop_isolated_points(points, triangles, cv_area_floor=cv_area_floor_factor * h_min_cm ** 2,
                                eps_tri=eps_tri, tri_insulator=tri_insulator)
    Cdop = domain.doping_at(fv.points[:, 0], fv.points[:, 1])
    tags = tag_boundary_points(fv.points, domain)

    ni_arr = is_insulator = None
    dEi_arr = Eg_arr = ec_off_arr = None
    if mat is not None:
        from core.materials import delta_Ei_of
        is_insulator, _ = domain.material_props_at(fv.points[:, 0], fv.points[:, 1], mat)
        mats = [mat] + domain.semiconductor_materials()
        k = domain.semiconductor_material_index(fv.points[:, 0], fv.points[:, 1])
        pick = lambda f: np.array([f(m) for m in mats])[k]
        ni_arr = np.where(is_insulator, 0.0, pick(lambda m: m.ni))
        dEi_arr = np.where(is_insulator, 0.0, pick(lambda m: delta_Ei_of(m, mat)))
        Eg_arr = pick(lambda m: m.Eg_eV)
        ec_off_arr = pick(lambda m: m.Vt * np.log(m.Nc / m.ni))

    return Mesh2D(
        domain=domain,
        points=fv.points,
        triangles=fv.triangles,
        edges=fv.edges,
        edge_weight=fv.edge_weight,
        facet_length=fv.facet_length,
        cv_area=fv.cv_area,
        boundary_edges=fv.boundary_edges,
        n_negative_subareas=fv.n_negative_subareas,
        n_floored=fv.n_floored,
        quality_report=quality_report,
        Cdop=Cdop,
        boundary_point_index=tags.point_index,
        boundary_bc_type=tags.bc_type,
        h_min_cm=h_min_cm,
        edge_g=fv.edge_g,
        ni_arr=ni_arr,
        is_insulator=is_insulator,
        facet_length_semi=fv.facet_length_semi if fv.facet_length_semi is not None else fv.facet_length,
        cv_area_semi=fv.cv_area_semi if fv.cv_area_semi is not None else fv.cv_area,
        dEi_arr=dEi_arr,
        Eg_arr=Eg_arr,
        ec_off_arr=ec_off_arr,
    )


# Backwards-compatible alias - build_mesh2d is device-generic, but the name
# used throughout main2d.py/configs/input_diode_2d.yaml predates that
# generalization.
build_diode2d_mesh = build_mesh2d
