"""Voronoi-box finite-volume geometry built on top of a Delaunay
triangulation of a point cloud - the unstructured-mesh generalization of
core/mesh.py's 1D finite-volume scheme (there, flux between adjacent nodes
scales as (u[i+1]-u[i])/(x[i+1]-x[i]); here, the analogous per-edge quantity
is the shared Voronoi-facet length divided by the edge length, a classic
result often called the "box integration method" - see e.g. Fichtner et al.
1983).

For each internal Delaunay edge shared by exactly two triangles with
circumcenters C1, C2, the segment of the true Voronoi diagram separating
the two points is exactly the segment C1-C2 (perpendicular to the edge when
the mesh is well-shaped). The edge's flux weight is |C1-C2| / |edge length|
- computed as the sum of the two triangles' signed half-facets (edge
midpoint to circumcenter), which is the same thing for a Delaunay edge and
extends to an edge on the domain boundary (one triangle, one half-facet:
flux ALONG the boundary between its two nodes is real). Nothing ever
crosses the domain's edge itself - there is no node out there to flux
against - which is exactly the "default insulating" boundary condition
every non-contact boundary point needs (see mesh2d/boundary.py), so no
separate Neumann assembly code is needed.

Each point's control-volume area is the sum, over every triangle containing
it, of the quadrilateral (point, edge midpoint, triangle circumcenter, other
edge midpoint) - the standard box-method decomposition. This is exact and
positive for acute/right triangles; an obtuse triangle can push its
circumcenter outside itself and make one sub-quad's area negative (the
classic "non-Delaunay-conforming" box-method pitfall). `build_fv_geometry`
reports how many negative sub-areas it found so a bad point cloud can be
caught before it reaches the solver, rather than silently corrupting the
mass matrix.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay


@dataclass
class FVGeometry:
    points: np.ndarray            # (N,2) cm
    triangles: np.ndarray         # (M,3) int, CCW-oriented vertex indices
    edges: np.ndarray             # (E,2) int, every edge (internal and domain-boundary)
    edge_weight: np.ndarray       # (E,) float, Voronoi-facet length / edge length
    facet_length: np.ndarray      # (E,) float, cm - the Voronoi-facet length itself
                                   # (needed separately from edge_weight to turn a current
                                   # DENSITY on an edge into an actual current through it)
    cv_area: np.ndarray           # (N,) float, cm^2, control-volume area per point
    boundary_edges: set           # {(i,j)} perimeter edges (only 1 adjacent triangle)
    n_negative_subareas: int      # mesh-quality diagnostic; should be 0 for a good mesh
    n_floored: int                # count of points whose cv_area hit cv_area_floor - a
                                    # second, independent mesh-quality diagnostic (a point
                                    # can have zero negative sub-areas of its own and still
                                    # end up with a pathologically small total area)
    facet_length_semi: np.ndarray = None  # (E,) cm, only set when tri_insulator is given -
                                            # facet length inside semiconductor triangles
    cv_area_semi: np.ndarray = None       # (N,) cm^2, likewise - control-volume area
                                            # inside semiconductor triangles
    n_negative_facets: int = 0            # edges whose summed signed facet was negative
                                            # (clamped to 0); 0 for a non-obtuse mesh
    edge_g: np.ndarray = None      # (E,) float, only set when eps_tri is given to
                                     # build_fv_geometry - the per-edge conductance
                                     # eps*edge_weight ALREADY split and weighted by each
                                     # adjacent triangle's own permittivity (see
                                     # build_fv_geometry's docstring), for a heterogeneous-
                                     # permittivity mesh (e.g. a MOS capacitor's oxide/
                                     # semiconductor interface) - a homogeneous mesh (the
                                     # diode) can keep using mat.eps * edge_weight instead.


def _signed_area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def _circumcenter(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0.0:
        # Degenerate (collinear) triangle - shouldn't occur for a valid
        # Delaunay triangulation of a non-degenerate point cloud.
        return np.array([np.nan, np.nan])
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / d
    return np.array([ux, uy])


def build_fv_geometry(points_cm, triangles=None, cv_area_floor=None, eps_tri=None,
                      tri_insulator=None):
    """triangles, if given (from mesh2d/pointcloud.py's `triangle`-library
    constrained conforming-Delaunay refinement), are used directly instead
    of recomputing a plain scipy.spatial.Delaunay triangulation - the
    constrained triangulation respects region/domain boundary segments a
    plain unconstrained Delaunay of the same points would not necessarily
    preserve. cv_area_floor (cm^2), if given, is a lower bound applied to every
    point's control-volume area after assembly. Needed in practice: a point
    sitting at a quadtree 2:1-balance T-junction can end up with most of its
    "true" local area attributed to a neighboring point instead (an
    asymmetric corner-quad split, not a negative-subarea case, so
    n_negative_subareas alone doesn't catch it), leaving its OWN cv_area
    orders of magnitude smaller than its neighbors' - which then blows up
    every row that divides by it (caught via a finite-difference Jacobian
    check: a node with cv_area ~40x below the mesh's h_min^2 turned an O(1)
    divergence term into an O(1e13) residual). Per the project's mesh-
    robustness principle, this is a physically-motivated regularization
    (bounding how thin a control volume's flux/charge normalization can get)
    rather than a mesh-refinement patch, and is preferred over one.

    eps_tri: optional (M,) array, one relative-permittivity-times-eps0
    value per triangle. When given, each internal edge's Voronoi facet
    (the segment C1-C2 between its two adjacent triangles' circumcenters)
    is split at its own midpoint M - which lies on the same perpendicular
    bisector of the edge as C1 and C2, since all three are, by definition,
    equidistant from the edge's endpoints, so M sits between them for a
    well-shaped mesh - into a d1=|M-C1| piece belonging to triangle 1 and a
    d2=|M-C2| piece belonging to triangle 2. The resulting conductance
    eps_tri1*d1/edge_len + eps_tri2*d2/edge_len is the exact box-FV
    generalization of a uniform eps*facet_len/edge_len to a piecewise-
    constant permittivity field (this is how a MOS capacitor's oxide/
    semiconductor interface gets correct D-field continuity in this mesh -
    see solver2d/poisson2d_mos.py). Returned as FVGeometry.edge_g, already
    eps-weighted (unlike the plain geometric edge_weight, which a
    homogeneous-material solver like the diode's still uses directly).

    tri_insulator: optional (M,) bool, True for a triangle inside an
    insulator. When given, the box method is split by material, as a
    control volume straddling a semiconductor/insulator interface must
    be: facet_length_semi is the part of each edge's facet lying in
    semiconductor triangles (carrier current flows only there - an edge
    wholly inside the oxide gets 0, which IS the no-flux interface BC,
    and an edge lying along the interface keeps only its silicon half),
    and cv_area_semi is the part of each node's control volume lying in
    semiconductor (where its carrier/doping charge and recombination
    live - an interface node's oxide half-cell holds none)."""
    points = np.asarray(points_cm, dtype=float)
    if triangles is not None:
        simplices = np.asarray(triangles, dtype=int)
    else:
        simplices = Delaunay(points).simplices

    # Orient every triangle CCW so the per-vertex quad decomposition below
    # traverses consistently and a negative sub-area is a real quality flag,
    # not an artifact of inconsistent winding.
    oriented = simplices.copy()
    for t in range(len(oriented)):
        i, j, k = oriented[t]
        if _signed_area2(points[i], points[j], points[k]) < 0.0:
            oriented[t, [1, 2]] = oriented[t, [2, 1]]

    circumcenters = np.array([
        _circumcenter(points[i], points[j], points[k]) for (i, j, k) in oriented
    ])

    edge_tris = {}
    for t, (i, j, k) in enumerate(oriented):
        for (a, b) in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            edge_tris.setdefault(key, []).append(t)

    semi_tri = None if tri_insulator is None else ~np.asarray(tri_insulator, dtype=bool)

    def half_facet(t, i, j):
        """Signed distance from edge (i,j)'s midpoint to triangle t's
        circumcenter, positive toward t's own third vertex - the piece of
        the edge's Voronoi facet that lies inside t (negative only when t
        is obtuse at the vertex opposite this edge)."""
        a, b = points[i], points[j]
        mid = 0.5 * (a + b)
        third = [v for v in oriented[t] if v != i and v != j][0]
        e = b - a
        nrm = np.array([-e[1], e[0]])
        if np.dot(points[third] - mid, nrm) < 0.0:
            nrm = -nrm
        return float(np.dot(circumcenters[t] - mid, nrm / np.linalg.norm(nrm)))

    # Every edge, internal OR on the domain boundary, carries flux through
    # its facet = sum over its adjacent triangles of that triangle's own
    # half-facet. A boundary edge (one triangle) is a real flux path along
    # the boundary between its two nodes - excluding it (as this module
    # used to) leaves a convex-corner node of a right-angle mesh with no
    # flux coupling at all, a singular continuity row.
    edges, edge_weight, facet_length, edge_g, facet_semi = [], [], [], [], []
    boundary_edges = set()
    n_negative_facets = 0
    for (i, j), tris in edge_tris.items():
        if len(tris) == 1:
            boundary_edges.add((i, j))
        edge_len = np.linalg.norm(points[i] - points[j])
        halves = [half_facet(t, i, j) for t in tris]
        facet_len = sum(halves)
        if facet_len < 0.0:
            n_negative_facets += 1
            halves = [max(h, 0.0) for h in halves]
            facet_len = sum(halves)
        edges.append((i, j))
        edge_weight.append(facet_len / edge_len)
        facet_length.append(facet_len)
        if eps_tri is not None:
            edge_g.append(sum(eps_tri[t] * max(h, 0.0) for t, h in zip(tris, halves)) / edge_len)
        if semi_tri is not None:
            facet_semi.append(sum(max(h, 0.0) for t, h in zip(tris, halves) if semi_tri[t]))

    n = len(points)
    cv_area = np.zeros(n)
    cv_semi = np.zeros(n)
    n_negative = 0
    area_scale = np.max(np.abs([_signed_area2(points[i], points[j], points[k])
                                  for (i, j, k) in oriented])) if len(oriented) else 1.0
    for t, (i, j, k) in enumerate(oriented):
        C = circumcenters[t]
        for (a, b, c) in ((i, j, k), (j, k, i), (k, i, j)):
            m_ab = 0.5 * (points[a] + points[b])
            m_ac = 0.5 * (points[a] + points[c])
            quad = [points[a], m_ab, C, m_ac]
            # Shoelace sum, consistent with the CCW orientation enforced
            # above: a negative signed area means the circumcenter fell
            # outside this triangle's corner at vertex a (an obtuse angle at
            # a) - the box-method pitfall this function exists to catch.
            # (A right triangle's degenerate zero-area pieces come out as
            # +-1e-16-relative rounding noise, not counted.)
            signed = 0.5 * sum(
                quad[m][0] * quad[(m + 1) % 4][1] - quad[(m + 1) % 4][0] * quad[m][1]
                for m in range(4)
            )
            if signed < -1e-10 * area_scale:
                n_negative += 1
            cv_area[a] += abs(signed)
            if semi_tri is not None and semi_tri[t]:
                cv_semi[a] += abs(signed)

    n_floored = 0
    if cv_area_floor is not None:
        below = cv_area < cv_area_floor
        n_floored = int(np.sum(below))
        cv_area = np.where(below, cv_area_floor, cv_area)
        if semi_tri is not None:
            cv_semi = np.where((cv_semi > 0.0) & (cv_semi < cv_area_floor), cv_area_floor, cv_semi)

    return FVGeometry(
        points=points,
        triangles=oriented,
        edges=np.array(edges, dtype=int),
        edge_weight=np.array(edge_weight, dtype=float),
        facet_length=np.array(facet_length, dtype=float),
        cv_area=cv_area,
        boundary_edges=boundary_edges,
        n_negative_subareas=n_negative,
        n_floored=n_floored,
        edge_g=np.array(edge_g, dtype=float) if eps_tri is not None else None,
        facet_length_semi=np.array(facet_semi, dtype=float) if semi_tri is not None else None,
        cv_area_semi=cv_semi if semi_tri is not None else None,
        n_negative_facets=n_negative_facets,
    )
