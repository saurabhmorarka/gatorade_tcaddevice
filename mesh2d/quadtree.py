"""Balanced square-quadtree mesher: a coarse, rectangular background grid
that is refined only inside user-specified boxes and graded toward
interfaces/junctions, then cut into triangles - every triangle is a
45-45-90 right triangle, so the mesh has NO obtuse angles by construction
(not by a post-hoc check that happens to pass).

Why squares, and why a 2:1-balanced tree
----------------------------------------
The box (Voronoi finite-volume) method needs a non-obtuse triangulation
(see mesh2d/mesh_quality.py's module docstring). A square cut along one
diagonal gives two right isosceles triangles. A square with one or more
"hanging" nodes (the midpoint of a side, created because the neighbor
across that side was refined one level further) instead gets a center point
and a fan: every fan triangle is again a right isosceles triangle, whatever
combination of its four sides carry a midpoint. That only holds if (a)
every cell is a SQUARE (a non-square cell's fan puts an obtuse angle at the
center opposite each unsplit long side - the reason an earlier, rectangular-
cell quadtree attempt came out ~15% obtuse) and (b) no side ever carries
more than one hanging node, i.e. neighbors differ by at most one level (the
standard 2:1 balance). The fan's center point lives strictly inside its own
cell, so resolving a hanging node never pushes a new node into a
neighbor - refinement stays local to the boxes/interfaces that asked for
it; there is no global X or Y grid line.

Exact geometric alignment
-------------------------
Every region/contact/mesa coordinate must end up exactly on a cell edge
(boundary tagging and the per-triangle material lookup both depend on it).
All work is done in INTEGER units of a base length `u` (the largest length
<= the finest requested spacing that divides every feature-to-feature
distance), relative to an origin chosen among the feature coordinates to
make as many of them as possible land on coarse grid lines. A cell is
split whenever a feature segment passes through its open interior or a
feature point (region/contact corner) sits on it anywhere other than a
corner - so a feature only forces refinement locally, along itself, and
only as far down as its own coordinate's dyadic alignment requires.

Refinement controls (all lengths in cm)
---------------------------------------
  h_max_cm            coarsest allowed cell (the background grid)
  refine_boxes        [(x0, x1, y0, y1, h), ...] - every cell touching the
                      box is refined to <= h
  graded_segments     [(segments, h0), ...] - axis-aligned segments (e.g.
                      the oxide/silicon interface, the metallurgical
                      junctions) with the spacing wanted ON them; spacing
                      grows geometrically (`growth`) with distance, the
                      same law as mesh2d/pointcloud.py::_target_spacing.
"""
import math

import numpy as np

from mesh2d.pointcloud import _target_spacing

_ALIGN_TOL = 1e-6   # relative-to-u tolerance for "is an integer multiple of u"


def _feature_geometry(domain):
    """Axis-aligned feature segments (domain outline + every region side)
    and feature points (every region/mesa/contact corner) in cm."""
    verts, edges, _ = domain.outer_boundary()
    segs = [(tuple(verts[a]), tuple(verts[b])) for a, b in edges]
    for region in domain.regions[1:]:
        x0, x1 = region.x_range_cm
        y0, y1 = region.y_range_cm
        segs += [((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)),
                 ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))]
    points = [tuple(p) for p in domain.region_corners()]
    return segs, points


def interface_and_junction_segments(domain):
    """Split domain.junction_segments() into (semiconductor/insulator
    interfaces, metallurgical junctions between semiconductor regions) -
    they usually want different spacings (the MOS inversion layer is a few
    nm thick; a junction's depletion edge is tens of nm)."""
    interface, junction = [], []
    for region in domain.regions[1:]:
        x0, x1 = region.x_range_cm
        y0, y1 = region.y_range_cm
        for p0, p1 in [((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)),
                       ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))]:
            mid = (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
            if domain.boundary_point_role(*mid) is not None:
                continue
            (interface if region.kind == "insulator" else junction).append((p0, p1))
    return interface, junction


def _is_multiple(value, u):
    q = value / u
    return abs(q - round(q)) <= _ALIGN_TOL


def _choose_unit(coords_x, coords_y, h_finest):
    """Largest u = h_finest/m (m = 1, 2, ...) dividing every feature-to-
    feature distance on both axes."""
    diffs = [c - coords_x[0] for c in coords_x] + [c - coords_y[0] for c in coords_y]
    for m in range(1, 1001):
        u = h_finest / m
        if all(_is_multiple(d, u) for d in diffs):
            return u
    raise ValueError(
        "quadtree mesh: the device's feature coordinates share no common unit <= "
        f"{h_finest * 1e7:.3g} nm (checked h_finest/m for m up to 1000) - snap the geometry "
        "(region/contact/mesa edges) to a common grid, e.g. multiples of 1 nm.")


def _valuation2(k, cap):
    """2-adic valuation of integer k (how many times it halves evenly), capped."""
    if k == 0:
        return cap
    v = 0
    while k % 2 == 0 and v < cap:
        k //= 2
        v += 1
    return v


def _choose_origin(coords, u, cap):
    """Origin (one of the feature coordinates) that puts the feature set on
    the coarsest possible dyadic grid lines overall, so the alignment
    constraint forces as little extra refinement as possible."""
    best, best_score = coords[0], -1
    for o in coords:
        score = sum(_valuation2(int(round((c - o) / u)), cap) for c in coords)
        if score > best_score:
            best, best_score = o, score
    return best


class _Tree:
    """Leaves of an integer-coordinate quadtree: key (ix, iy, s) with s a
    power of two and ix, iy multiples of s."""

    def __init__(self):
        self.leaves = set()
        self.sizes = set()

    def add(self, cell):
        self.leaves.add(cell)
        self.sizes.add(cell[2])

    def leaf_at(self, X, Y):
        """Leaf containing the unit square with lower-left corner (X, Y)."""
        for s in self.sizes:
            key = (X - X % s, Y - Y % s, s)
            if key in self.leaves:
                return key
        return None


def _split(cell):
    x, y, s = cell
    h = s // 2
    return [(x, y, h), (x + h, y, h), (x, y + h, h), (x + h, y + h, h)]


def build_quadtree_mesh(domain, h_max_cm, refine_boxes=(), graded_segments=(), growth=1.3,
                         h_finest_cm=None):
    """Returns (points (N,2) cm, triangles (M,3) int, info dict)."""
    feat_segs, feat_pts = _feature_geometry(domain)
    coords_x = sorted({p[0] for s in feat_segs for p in s} | {p[0] for p in feat_pts})
    coords_y = sorted({p[1] for s in feat_segs for p in s} | {p[1] for p in feat_pts})

    h_candidates = [h_max_cm] + [b[4] for b in refine_boxes] + [h for _, h in graded_segments]
    if h_finest_cm is None:
        h_finest_cm = min(h_candidates)
    u = _choose_unit(coords_x, coords_y, h_finest_cm)

    # Root: a square of side 2^k * u big enough to hold the domain on
    # either side of the chosen origin (cells outside the domain are
    # dropped, so the oversize costs nothing).
    x_min, x_max = coords_x[0], coords_x[-1]
    y_min, y_max = coords_y[0], coords_y[-1]
    span = max(x_max - x_min, y_max - y_min)
    k = max(1, math.ceil(math.log2(span / u + 1e-9)) + 1)
    ox = _choose_origin(coords_x, u, k)
    oy = _choose_origin(coords_y, u, k)
    L = 2 ** k   # root half-width, in units of u
    x0_cm, y0_cm = ox - L * u, oy - L * u
    root_size = 2 * L

    def to_int(v, o):
        return int(round((v - o) / u))

    def cm(X, o):
        return o + X * u

    iseg = []   # (X0, X1, Y0, Y1) integer, axis-aligned
    for (a, b) in feat_segs:
        X0, X1 = sorted((to_int(a[0], x0_cm), to_int(b[0], x0_cm)))
        Y0, Y1 = sorted((to_int(a[1], y0_cm), to_int(b[1], y0_cm)))
        iseg.append((X0, X1, Y0, Y1))
    ipts = {(to_int(p[0], x0_cm), to_int(p[1], y0_cm)) for p in feat_pts}

    graded_int = []
    for segs, h0 in graded_segments:
        rects = []
        for (a, b) in segs:
            rects.append((min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])))
        graded_int.append((rects, h0))

    def cuts_feature(cell):
        X, Y, s = cell
        for (X0, X1, Y0, Y1) in iseg:
            if Y0 == Y1:   # horizontal
                if Y < Y0 < Y + s and X0 < X + s and X1 > X:
                    return True
            else:          # vertical
                if X < X0 < X + s and Y0 < Y + s and Y1 > Y:
                    return True
        for (PX, PY) in ipts:
            if X <= PX <= X + s and Y <= PY <= Y + s:
                if (PX - X) % s or (PY - Y) % s:   # on the cell, not at a corner
                    return True
        return False

    def target_h(cell):
        X, Y, s = cell
        cx0, cx1 = cm(X, x0_cm), cm(X + s, x0_cm)
        cy0, cy1 = cm(Y, y0_cm), cm(Y + s, y0_cm)
        h = h_max_cm
        for (bx0, bx1, by0, by1, bh) in refine_boxes:
            if bx0 < cx1 and bx1 > cx0 and by0 < cy1 and by1 > cy0:
                h = min(h, bh)
        for rects, h0 in graded_int:
            d = math.inf
            for (sx0, sx1, sy0, sy1) in rects:
                dx = max(0.0, sx0 - cx1, cx0 - sx1)
                dy = max(0.0, sy0 - cy1, cy0 - sy1)
                d = min(d, math.hypot(dx, dy))
            h = min(h, _target_spacing(d, h0, h_max_cm, growth))
        return h

    def inside(cell):
        X, Y, s = cell
        return domain.contains(cm(X + 0.5 * s, x0_cm), cm(Y + 0.5 * s, y0_cm), tol=0.0)

    def overlaps_domain(cell):
        X, Y, s = cell
        return (cm(X, x0_cm) < x_max and cm(X + s, x0_cm) > x_min
                and cm(Y, y0_cm) < y_max and cm(Y + s, y0_cm) > y_min)

    tree = _Tree()
    stack = [(0, 0, root_size)]
    while stack:
        cell = stack.pop()
        if not overlaps_domain(cell):
            continue
        s = cell[2]
        if s > 1 and (cuts_feature(cell) or s * u > target_h(cell) * (1 + 1e-9)):
            stack.extend(_split(cell))
            continue
        if inside(cell):
            tree.add(cell)

    # 2:1 balance across shared sides (repeat until stable).
    changed = True
    while changed:
        changed = False
        for cell in list(tree.leaves):
            if cell not in tree.leaves:
                continue
            X, Y, s = cell
            if s < 4:
                continue
            step = s // 4
            need = False
            probes = ([(X + t, Y - 1) for t in range(0, s, step)] +
                      [(X + t, Y + s) for t in range(0, s, step)] +
                      [(X - 1, Y + t) for t in range(0, s, step)] +
                      [(X + s, Y + t) for t in range(0, s, step)])
            for (PX, PY) in probes:
                nb = tree.leaf_at(PX, PY)
                if nb is not None and nb[2] * 2 < s:
                    need = True
                    break
            if need:
                tree.leaves.discard(cell)
                for c in _split(cell):
                    tree.add(c)
                changed = True

    # Triangulate: corners of every leaf are vertices; a leaf with any side
    # midpoint present as a vertex (a hanging node) gets a center fan,
    # otherwise one diagonal.
    vid = {}
    pts = []

    def vertex(X2, Y2):   # doubled-integer coordinates (centers are half-integers)
        key = (X2, Y2)
        if key not in vid:
            vid[key] = len(pts)
            pts.append((cm(X2 / 2.0, x0_cm), cm(Y2 / 2.0, y0_cm)))
        return vid[key]

    leaves = sorted(tree.leaves)
    n_owner = {}
    for (X, Y, s) in leaves:
        for (a, b) in ((X, Y), (X + s, Y), (X + s, Y + s), (X, Y + s)):
            vertex(2 * a, 2 * b)
            n_owner[(2 * a, 2 * b)] = n_owner.get((2 * a, 2 * b), 0) + 1

    tris = []
    n_fan = 0
    for (X, Y, s) in leaves:
        c = [(2 * X, 2 * Y), (2 * (X + s), 2 * Y), (2 * (X + s), 2 * (Y + s)), (2 * X, 2 * (Y + s))]
        mids = [((c[i][0] + c[(i + 1) % 4][0]) // 2, (c[i][1] + c[(i + 1) % 4][1]) // 2) for i in range(4)]
        hanging = [m in vid for m in mids]
        if not any(hanging):
            v = [vid[p] for p in c]
            # A convex domain corner belongs to this leaf only; the diagonal
            # must pass through it, or it would sit on a single triangle
            # with no internal edge (an isolated, singular FV node).
            if n_owner[c[1]] == 1 or n_owner[c[3]] == 1:
                tris.append((v[0], v[1], v[3]))
                tris.append((v[1], v[2], v[3]))
            else:
                tris.append((v[0], v[1], v[2]))
                tris.append((v[0], v[2], v[3]))
            continue
        n_fan += 1
        center = vertex(2 * X + s, 2 * Y + s)
        for i in range(4):
            a, b = vid[c[i]], vid[c[(i + 1) % 4]]
            if hanging[i]:
                m = vid[mids[i]]
                tris.append((center, a, m))
                tris.append((center, m, b))
            else:
                tris.append((center, a, b))

    points = np.array(pts, dtype=float)
    # Snap to exact feature coordinates (cm(...) is exact in integer units,
    # but o + X*u in floating point can differ from the YAML-derived value
    # by an ulp - boundary tagging compares against the latter).
    for arr, feats in ((points[:, 0], coords_x), (points[:, 1], coords_y)):
        for f in feats:
            arr[np.abs(arr - f) <= 1e-6 * u] = f

    info = dict(unit_cm=u, n_leaves=len(leaves), n_fan_leaves=n_fan,
                min_leaf_cm=min(l[2] for l in leaves) * u, max_leaf_cm=max(l[2] for l in leaves) * u)
    return points, np.array(tris, dtype=int), info
