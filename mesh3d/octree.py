"""Adaptive box-tree (octree-like) box-FV mesh from per-voxel material
labels: the 3D mesher for virtual-fab structures (plans/3d_handoff.md,
"the approved octree design").

Why not tetrahedra with templates
---------------------------------
The 2D quadtree resolves a hanging node with a centre point and a fan of
45-45-90 triangles, which is non-obtuse by construction. Carried to 3D
(centre point + tets on each face triangle) it is not: on random 2:1
octrees ~15% of the edges get a negative coupling - the tets'
circumcentres land in the neighbouring cell (mesh3d/studies/
octree_transition_study.py). Instead the control volumes ARE the Voronoi
diagram of the mesh nodes (= the circumcentric dual of their Delaunay
tetrahedralization). A Voronoi diagram is admissible for any point set:
every coupling is a facet area >= 0 and every control volume is > 0. The
3D counterpart of "no obtuse triangles" holds by construction, and the
gate (check_octree_mesh) verifies it on every build.

The tree
--------
Integer units are the fab voxel (so every material interface the fab
built is a unit-cell face). Leaves are boxes; a leaf may split along any
one axis, at any unit position (kd-style: sizes are not powers of two),
which lets a leaf be thin in y under the gate oxide and wide in x and z.
A leaf is split until
  * it holds a single label (material region, gas or conductor),
  * its size along each axis is <= the target spacing anywhere inside it
    (graded from interfaces and junctions, see target_spacing),
  * 2:1 balance: along each axis it is at most twice as long as any leaf
    it shares a vertex with,
  * face match at interfaces: across a face where the label changes, the
    leaves on the two sides have the same tangential extent (no hanging
    node on an interface; refinement can still vary ALONG the interface,
    nothing propagates over a whole interface).

The dual
--------
Nodes are all leaf corners (fab grid nodes, so doping is read exactly).
For most nodes the Voronoi cell is the box between the midpoints to the
nearest node along each axis; that box IS the Voronoi cell iff no other
node is closer to any of its 8 corners (convexity), which one KD-tree
query per corner decides. Such a cell splits into octants, each inside one
leaf, so per-material volumes and facet parts are exact and vectorized.
Every other node (near a level change) gets its cell by exact half-space
clipping; where it overlaps leaves of several labels, the cell and its
facets are clipped against each leaf box for the per-material split.

Conductors and gas are not part of the device: their leaves shape the
tree (exact terminal surfaces) but carry no unknowns. A device node with
volume in a conductor leaf is a Dirichlet node of that contact.
"""
import itertools
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from core.params import EPS0

SEMI, INSULATOR, CONDUCTOR, GAS = "semi", "insulator", "conductor", "gas"
_TOL = 1e-9
# a facet smaller than this times its edge length squared is round-off: a
# degenerate (cospherical) node set, whose true facet is a point or a line
_FACET_REL = 1e-8


class OctreeMesh(SimpleNamespace):
    """The mesh's attributes (a class, so the solver can key its cached
    linear solver on a weak reference)."""


# --- sizing ---------------------------------------------------------------------------
def target_spacing(label, kinds, cell_net, h, h_fine, h_fine_other, h_max, growth):
    """Per-axis target leaf size (units) at every voxel.

    label      int array (nx, ny, nz), -1 = gas, else index into kinds
    kinds      list of SEMI / INSULATOR / CONDUCTOR per label
    cell_net   net doping per voxel (sign changes = junctions)
    h          voxel size per axis (nm)
    h_fine     spacing (nm) wanted at semiconductor interfaces and junctions
    h_fine_other  spacing at the other device interfaces (insulator /
               insulator, insulator / conductor)
    The spacing grows as h_fine + (growth - 1) * distance, capped at h_max;
    gas and conductor voxels only need what their labels force."""
    h = np.asarray(h, float)
    kind = np.array(list(kinds) + [GAS])[label]
    semi = kind == SEMI
    dev = semi | (kind == INSULATOR)
    big = 1e9

    def changes(mask_a, mask_b):
        """Voxels of a with a face neighbour in b (and vice versa)."""
        out = np.zeros(label.shape, bool)
        for a in range(3):
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[a], sl1[a] = slice(0, -1), slice(1, None)
            sl0, sl1 = tuple(sl0), tuple(sl1)
            hit = (mask_a[sl0] & mask_b[sl1]) | (mask_b[sl0] & mask_a[sl1])
            out[sl0] |= hit
            out[sl1] |= hit
        return out

    def dist(seed):
        if not seed.any():
            return np.full(label.shape, big)
        return ndimage.distance_transform_edt(~seed, sampling=h)

    # semiconductor interfaces (to anything) and junctions
    seed = changes(semi, ~semi)
    s = np.sign(cell_net) * semi
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a], sl1[a] = slice(0, -1), slice(1, None)
        flip = (s[tuple(sl0)] * s[tuple(sl1)]) < 0
        seed[tuple(sl0)] |= flip
        seed[tuple(sl1)] |= flip
    d_semi = dist(seed)
    # other device interfaces: a label change with a device voxel on one side
    other = np.zeros(label.shape, bool)
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a], sl1[a] = slice(0, -1), slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        hit = (label[sl0] != label[sl1]) & (dev[sl0] | dev[sl1])
        other[sl0] |= hit
        other[sl1] |= hit
    d_other = dist(other)
    tgt = np.empty((3,) + label.shape)
    for a in range(3):
        t = np.minimum(h_fine[a] + (growth - 1.0) * d_semi, h_fine_other[a] + (growth - 1.0) * d_other)
        t = np.minimum(t, h_max[a])
        tgt[a] = np.where(dev, t, big) / h[a]
    return np.maximum(tgt, 1.0)


# --- the tree -------------------------------------------------------------------------
class BoxTree:
    """Leaves as integer boxes [lo, hi) in voxel units, with an owner array
    (leaf id per voxel) for point location and neighbour lookup."""

    def __init__(self, label, tgt, root):
        self.label = label
        self.tgt = tgt
        self.shape = np.array(label.shape)
        self.lo, self.hi = [], []
        self.alive = []
        self.owner = np.empty(label.shape, dtype=np.int32)
        edges = [list(range(0, n, r)) + [n] for n, r in zip(label.shape, root)]
        for i0, i1 in zip(edges[0][:-1], edges[0][1:]):
            for j0, j1 in zip(edges[1][:-1], edges[1][1:]):
                for k0, k1 in zip(edges[2][:-1], edges[2][1:]):
                    self._add((i0, j0, k0), (i1, j1, k1))

    def _add(self, lo, hi):
        i = len(self.lo)
        self.lo.append(np.array(lo))
        self.hi.append(np.array(hi))
        self.alive.append(True)
        self.owner[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = i
        return i

    def box(self, i):
        lo, hi = self.lo[i], self.hi[i]
        return (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))

    def split(self, i, axis, pos):
        lo, hi = self.lo[i], self.hi[i]
        assert lo[axis] < pos < hi[axis]
        self.alive[i] = False
        m1, m0 = hi.copy(), lo.copy()
        m1[axis] = pos
        m0[axis] = pos
        return self._add(lo, m1), self._add(m0, hi)

    def shell_ids(self, i):
        """Leaves sharing at least a vertex with leaf i."""
        lo = np.maximum(self.lo[i] - 1, 0)
        hi = np.minimum(self.hi[i] + 1, self.shape)
        ids = np.unique(self.owner[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        return ids[ids != i]

    def face_ids(self, i, axis, side):
        """(owner ids, labels) of the voxels just across one face, or None
        on the domain wall."""
        lo, hi = self.lo[i].copy(), self.hi[i].copy()
        if side < 0:
            if lo[axis] == 0:
                return None
            hi[axis] = lo[axis]
            lo[axis] -= 1
        else:
            if hi[axis] == self.shape[axis]:
                return None
            lo[axis] = hi[axis]
            hi[axis] += 1
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        return self.owner[sl], self.label[sl]

    def leaves(self):
        ids = np.flatnonzero(self.alive)
        return ids, np.array([self.lo[i] for i in ids]), np.array([self.hi[i] for i in ids])

    # -- refinement rules: each returns (axis, position) or None ------------------------
    def _rule_label(self, i):
        sub = self.label[self.box(i)]
        if (sub == sub.flat[0]).all():
            return None
        size = self.hi[i] - self.lo[i]
        best = None
        for a in range(3):
            if size[a] < 2:
                continue
            others = tuple(b for b in range(3) if b != a)
            ch = np.flatnonzero(np.any(np.diff(sub, axis=a) != 0, axis=others)) + 1
            if len(ch):
                k = ch[np.argmin(np.abs(ch - size[a] / 2))]
                score = abs(k - size[a] / 2) / size[a]
                if best is None or score < best[0]:
                    best = (score, a, int(self.lo[i][a] + k))
        return best[1:] if best else None

    def _rule_size(self, i):
        size = self.hi[i] - self.lo[i]
        b = self.box(i)
        worst = None
        for a in range(3):
            if size[a] >= 2:
                t = self.tgt[a][b].min()
                if size[a] > t + 1e-9 and (worst is None or size[a] / t > worst[0]):
                    worst = (size[a] / t, a)
        if worst is None:
            return None
        a = worst[1]
        return a, int(self.lo[i][a] + size[a] // 2)

    def _rule_balance(self, i, nbrs):
        size = self.hi[i] - self.lo[i]
        for n in nbrs:
            ns = self.hi[n] - self.lo[n]
            for a in range(3):
                if size[a] > 2 * ns[a] and size[a] >= 2:
                    return a, int(self.lo[i][a] + size[a] // 2)
        return None

    def _rule_face(self, i):
        my = self.label[self.lo[i][0], self.lo[i][1], self.lo[i][2]]
        for axis in range(3):
            for side in (-1, 1):
                f = self.face_ids(i, axis, side)
                if f is None:
                    continue
                own, lab = f
                for n in np.unique(own[lab != my]):
                    for b in range(3):
                        if b == axis:
                            continue
                        if self.lo[n][b] > self.lo[i][b]:
                            return b, int(self.lo[n][b])
                        if self.hi[n][b] < self.hi[i][b]:
                            return b, int(self.hi[n][b])
        return None

    def refine(self, log=None):
        queue = list(np.flatnonzero(self.alive))
        queued = set(queue)
        n_split = 0
        while queue:
            i = queue.pop()
            queued.discard(i)
            if not self.alive[i]:
                continue
            s = self._rule_label(i) or self._rule_size(i)
            nbrs = None
            if s is None:
                nbrs = self.shell_ids(i)
                s = self._rule_balance(i, nbrs) or self._rule_face(i)
            if s is None:
                continue
            if nbrs is None:
                nbrs = self.shell_ids(i)
            kids = self.split(i, *s)
            n_split += 1
            for j in list(kids) + list(nbrs):
                if self.alive[j] and j not in queued:
                    queue.append(j)
                    queued.add(j)
        if log:
            log(f"  tree: {int(np.sum(self.alive))} leaves after {n_split} splits")


# --- exact Voronoi clipping (transition nodes) ---------------------------------------
def _clip(faces, n, c, label):
    """faces {label: [vertex, ...]} of a convex polyhedron; keep n.x <= c;
    the cut face gets `label`. A plane that cuts nothing (every vertex
    inside or on it) leaves the polyhedron as it is: a face lying in the
    plane must not be duplicated by a "cut face" (axis-aligned cells and
    leaf boxes share planes all the time)."""
    if all(n @ v - c <= _TOL for poly in faces.values() for v in poly):
        return faces
    out, cut = {}, []
    for lab, poly in faces.items():
        new = []
        m = len(poly)
        for k in range(m):
            a, b = poly[k], poly[(k + 1) % m]
            da, db = n @ a - c, n @ b - c
            if da <= _TOL:
                new.append(a)
                if da >= -_TOL:
                    cut.append(a)
            if (da < -_TOL and db > _TOL) or (da > _TOL and db < -_TOL):
                q = a + (da / (da - db)) * (b - a)
                new.append(q)
                cut.append(q)
        if len(new) >= 3:
            out[lab] = new
    if len(cut) >= 3:
        P = np.unique(np.round(np.array(cut), 9), axis=0)
        if len(P) >= 3:
            cen = P.mean(0)
            nn = n / np.linalg.norm(n)
            u = P[np.argmax(np.linalg.norm(P - cen, axis=1))] - cen
            u /= np.linalg.norm(u)
            w = np.cross(nn, u)
            ang = np.arctan2((P - cen) @ w, (P - cen) @ u)
            out[label] = list(P[np.argsort(ang)])
    return out


def _area_vec(poly):
    P = np.asarray(poly)
    return 0.5 * np.cross(P - P[0], np.roll(P, -1, 0) - P[0]).sum(0)


def _volume(faces):
    return sum(float(np.dot(np.asarray(f[0]), _area_vec(f))) for f in faces.values()) / 3.0


def _box_faces(lo, hi):
    C = {v: np.array([(hi if v[a] else lo)[a] for a in range(3)], float)
         for v in itertools.product((0, 1), repeat=3)}
    faces = {}
    for a in range(3):
        b, c = [k for k in range(3) if k != a]
        for side in (0, 1):
            pts = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                key = [0, 0, 0]
                key[a], key[b], key[c] = side, u, v
                pts.append(C[tuple(key)])
            out = np.zeros(3)
            out[a] = 1 if side else -1
            if _area_vec(pts) @ out < 0:
                pts = pts[::-1]
            faces[("B", a, side)] = pts
    return faces


def _clip_to_box(faces, lo, hi):
    for a in range(3):
        n = np.zeros(3)
        n[a] = 1.0
        faces = _clip(faces, n, hi[a], ("X", a, 1))
        faces = _clip(faces, -n, -lo[a], ("X", a, 0))
    return faces


def voronoi_cell(p, nbrs, lo, hi):
    """Exact Voronoi cell of p (nm) among nbrs [(id, point)] inside the box
    [lo, hi]: {neighbour id or ("B", axis, side): polygon}. Neighbours are
    clipped nearest first; once the next one is farther than twice the
    cell's farthest vertex its bisector cannot cut the cell (security
    radius), so the rest are skipped. Returns (faces, complete): complete
    is False if the neighbours ran out before the security radius did."""
    faces = _box_faces(lo, hi)
    nbrs = sorted(nbrs, key=lambda t: float(np.dot(t[1] - p, t[1] - p)))
    vmax = np.inf
    for k, (j, q) in enumerate(nbrs):
        d = float(np.linalg.norm(q - p))
        if d > 2.0 * vmax * (1 + 1e-9):
            return faces, True
        n = q - p
        faces = _clip(faces, n, (q @ q - p @ p) / 2.0, j)
        if k % 4 == 3 or k == len(nbrs) - 1:
            vmax = max(float(np.linalg.norm(np.asarray(v) - p)) for f in faces.values() for v in f)
    return faces, False


# --- the mesh -------------------------------------------------------------------------
def build_octree_mesh(label, materials, h_nm, node_net, mat, origin_nm=(0.0, 0.0, 0.0), face_contacts=None,
                      cell_net=None, h_fine=(2.0, 0.5, 2.0), h_fine_other=(2.0, 2.0, 2.0),
                      h_max=(8.0, 8.0, 8.0), growth=1.35, root=None, log=print):
    """Box-FV mesh (the attribute set solver2d/newton_solver_qf_2d.py reads)
    on an adaptive box tree over a voxel label array.

    label      int (nx, ny, nz) per voxel: index into `materials`, -1 gas
    materials  [{name, kind: semi|insulator|conductor, eps_r, contact}]
    h_nm       voxel size per axis (nm)
    node_net   net doping (cm^-3) on the voxel-grid nodes, (nx+1, ny+1, nz+1)
    mat        the semiconductor's core Material
    face_contacts {name: (axis, "min" | "max")}: a whole domain face"""
    t0 = time.perf_counter()
    label = np.asarray(label)
    h = np.asarray(h_nm, float)
    kinds = [m["kind"] for m in materials]
    if cell_net is None:
        n = node_net
        cell_net = sum(n[i:n.shape[0] - 1 + i, j:n.shape[1] - 1 + j, k:n.shape[2] - 1 + k]
                       for i in (0, 1) for j in (0, 1) for k in (0, 1)) / 8.0
    tgt = target_spacing(label, kinds, cell_net, h, h_fine, h_fine_other, h_max, growth)
    if root is None:
        root = [max(1, int(round(2 * h_max[a] / h[a]))) for a in range(3)]
    tree = BoxTree(label, tgt, root)
    tree.refine(log=log)
    t_tree = time.perf_counter() - t0
    ids, LO, HI = tree.leaves()
    leaf_label = label[LO[:, 0], LO[:, 1], LO[:, 2]]
    dual = _dual(tree, ids, LO, HI, leaf_label, label, h, log)
    m = _assemble(dual, materials, h, node_net, mat, np.asarray(origin_nm, float), face_contacts, label.shape)
    m.leaves = (LO, HI, leaf_label)
    m.tree_time_s = t_tree
    m.build_time_s = time.perf_counter() - t0
    check_octree_mesh(m, label, materials, h, log=log)
    log(f"  octree mesh: {len(m.points)} device nodes, {len(m.edges)} edges, "
        f"{dual['n_clipped']} clipped cells; tree {t_tree:.1f} s, total {m.build_time_s:.1f} s")
    return m


def _dual(tree, ids, LO, HI, leaf_label, label, h, log):
    """Voronoi cells of all leaf corners: per node and per label the volume,
    and per (node, neighbour, label) the facet area."""
    shape = np.array(label.shape)
    nshape = tuple(shape + 1)
    corners = np.concatenate([np.stack([np.where(o[a], HI[:, a], LO[:, a]) for a in range(3)], axis=1)
                              for o in itertools.product((0, 1), repeat=3)])
    is_node = np.zeros(nshape, bool)
    is_node[tuple(corners.T)] = True
    node_idx = np.argwhere(is_node)                                     # C order
    N = len(node_idx)
    node_id = np.full(nshape, -1, dtype=np.int64)
    node_id[tuple(node_idx.T)] = np.arange(N)
    P = node_idx * h                                                    # nm, relative to the grid origin
    dom_hi = shape * h

    # nearest node along each axis direction (grid index), -1 if none
    nb = np.full((N, 3, 2), -1, dtype=np.int64)
    for a in range(3):
        idx = np.where(is_node, np.arange(nshape[a]).reshape([-1 if b == a else 1 for b in range(3)]), -1)
        big = nshape[a] + 1
        fwd = np.where(idx >= 0, idx, big)
        nxt = np.flip(np.minimum.accumulate(np.flip(fwd, axis=a), axis=a), axis=a)
        sl = [slice(None)] * 3
        sl[a] = slice(1, None)
        nxt_strict = np.full(nshape, big)
        dst = [slice(None)] * 3
        dst[a] = slice(0, -1)
        nxt_strict[tuple(dst)] = nxt[tuple(sl)]
        bwd = np.where(idx >= 0, idx, -1)
        prv = np.maximum.accumulate(bwd, axis=a)
        prv_strict = np.full(nshape, -1)
        dst2 = [slice(None)] * 3
        dst2[a] = slice(1, None)
        sl2 = [slice(None)] * 3
        sl2[a] = slice(0, -1)
        prv_strict[tuple(dst2)] = prv[tuple(sl2)]
        f = nxt_strict[tuple(node_idx.T)]
        b = prv_strict[tuple(node_idx.T)]
        nb[:, a, 1] = np.where(f < big, f, -1)
        nb[:, a, 0] = b
    # half widths (nm) of the candidate box; a node on the domain wall has 0
    half = np.zeros((N, 3, 2))
    ok = np.ones(N, bool)
    for a in range(3):
        on_lo = node_idx[:, a] == 0
        on_hi = node_idx[:, a] == shape[a]
        lo_ok = (nb[:, a, 0] >= 0) | on_lo
        hi_ok = (nb[:, a, 1] >= 0) | on_hi
        ok &= lo_ok & hi_ok
        half[:, a, 0] = np.where(nb[:, a, 0] >= 0, (node_idx[:, a] - nb[:, a, 0]) * h[a] / 2, 0.0)
        half[:, a, 1] = np.where(nb[:, a, 1] >= 0, (nb[:, a, 1] - node_idx[:, a]) * h[a] / 2, 0.0)
    kd = cKDTree(P)
    for o in itertools.product((0, 1), repeat=3):
        c = P + np.stack([np.where(o[a], half[:, a, 1], -half[:, a, 0]) for a in range(3)], axis=1)
        d_own = np.linalg.norm(c - P, axis=1)
        d_min, _ = kd.query(c, k=1)
        ok &= d_min >= d_own - 1e-7 * (1.0 + d_own)
    box = np.flatnonzero(ok)
    clipped = np.flatnonzero(~ok)

    vol = {}                    # (node, label) -> nm^3
    facet = {}                  # (node, nbr node, label) -> nm^2 (from the node's own cell)

    # fast path: octants
    def lab_at(pt_units):
        v = np.clip(np.floor(pt_units).astype(np.int64), 0, shape - 1)
        return label[v[:, 0], v[:, 1], v[:, 2]]

    octant_lab = {}
    for o in itertools.product((0, 1), repeat=3):
        w = np.stack([half[box, a, o[a]] for a in range(3)], axis=1)          # nm
        sign = np.array([1 if o[a] else -1 for a in range(3)])
        cen_units = node_idx[box] + sign * (w / 2) / h
        lab = lab_at(cen_units)
        octant_lab[o] = lab
        v = w.prod(axis=1)
        for n_, l_, v_ in zip(box[v > 0], lab[v > 0], v[v > 0]):
            vol[(n_, l_)] = vol.get((n_, l_), 0.0) + v_
    for a in range(3):
        others = [b for b in range(3) if b != a]
        for side in (0, 1):
            q = nb[box, a, side]
            has = q >= 0
            qi = node_idx[box].copy()
            qi[:, a] = q
            qid = np.full(len(box), -1, dtype=np.int64)
            qid[has] = node_id[tuple(qi[has].T)]
            for o in itertools.product((0, 1), repeat=3):
                if o[a] != side:
                    continue
                part = half[box, others[0], o[others[0]]] * half[box, others[1], o[others[1]]]
                lab = octant_lab[o]
                sel = has & (part > 0)
                for n_, j_, l_, a_ in zip(box[sel], qid[sel], lab[sel], part[sel]):
                    facet[(n_, j_, l_)] = facet.get((n_, j_, l_), 0.0) + a_

    # transition nodes: exact clipping
    leaf_of = tree.owner
    LOh, HIh = LO * h, HI * h
    leaf_index = {int(i): k for k, i in enumerate(ids)}
    for n_ in clipped:
        p = P[n_]
        R = 2.0 * np.max(half[n_]) + 4.0 * h.max()
        while True:
            cand = [j for j in kd.query_ball_point(p, R) if j != n_]
            lo = np.maximum(p - R, 0.0)
            hi = np.minimum(p + R, dom_hi)
            faces, complete = voronoi_cell(p, [(j, P[j]) for j in cand], lo, hi)
            vmax = max(np.linalg.norm(np.asarray(v) - p) for f in faces.values() for v in f)
            if complete or 2 * vmax <= R * (1 + 1e-9) or (np.all(lo <= 0) and np.all(hi >= dom_hi)):
                break
            R *= 2.0
        pts = np.concatenate([np.asarray(f) for f in faces.values()])
        blo = np.floor(pts.min(0) / h + 1e-9).astype(int)
        bhi = np.ceil(pts.max(0) / h - 1e-9).astype(int)
        blo, bhi = np.clip(blo, 0, shape - 1), np.clip(np.maximum(bhi, blo + 1), 1, shape)
        sub = leaf_of[blo[0]:bhi[0], blo[1]:bhi[1], blo[2]:bhi[2]]
        leaves_here = np.unique(sub)
        labs = {int(leaf_label[leaf_index[int(L)]]) for L in leaves_here}
        nbr_faces = {k: f for k, f in faces.items() if not isinstance(k, tuple)}
        if len(labs) == 1:
            (l_,) = labs
            vol[(n_, l_)] = vol.get((n_, l_), 0.0) + _volume(faces)
            for j, f in nbr_faces.items():
                a_ = float(np.linalg.norm(_area_vec(f)))
                if a_ > _FACET_REL * float((P[j] - p) @ (P[j] - p)):
                    facet[(n_, j, l_)] = facet.get((n_, j, l_), 0.0) + a_
            continue
        for L in leaves_here:
            k = leaf_index[int(L)]
            l_ = int(leaf_label[k])
            part = _clip_to_box(faces, LOh[k], HIh[k])
            if len(part) < 4:
                continue
            v = _volume(part)
            if v > 1e-12:
                vol[(n_, l_)] = vol.get((n_, l_), 0.0) + v
            for j, f in part.items():
                if isinstance(j, tuple):
                    continue
                a_ = float(np.linalg.norm(_area_vec(f)))
                if a_ > _FACET_REL * float((P[j] - p) @ (P[j] - p)):
                    facet[(n_, j, l_)] = facet.get((n_, j, l_), 0.0) + a_
    if log:
        log(f"  dual: {N} nodes, {len(box)} box cells, {len(clipped)} clipped")
    return dict(N=N, node_idx=node_idx, P=P, vol=vol, facet=facet, n_clipped=len(clipped),
                box_nodes=box, clipped_nodes=clipped)


def _assemble(dual, materials, h, node_net, mat, origin, face_contacts, shape):
    N, P, node_idx = dual["N"], dual["P"], dual["node_idx"]
    kinds = np.array([m["kind"] for m in materials] + [GAS])
    eps = np.array([mat.eps if m["kind"] == SEMI else EPS0 * float(m.get("eps_r", 1.0)) for m in materials]
                   + [0.0])
    nm2, nm3, cm = 1e-14, 1e-21, 1e-7
    cv = np.zeros(N)
    cv_semi = np.zeros(N)
    touch = {}                                           # conductor label -> node mask
    for (n_, l_), v in dual["vol"].items():
        k = kinds[l_]
        if k == SEMI:
            cv[n_] += v
            cv_semi[n_] += v
        elif k == INSULATOR:
            cv[n_] += v
        elif k == CONDUCTOR:
            touch.setdefault(materials[l_]["contact"], set()).add(n_)
    # facets: per undirected edge, area from each side (device labels only)
    fa, fs, fg = {}, {}, {}
    for (n_, j_, l_), a in dual["facet"].items():
        if kinds[l_] not in (SEMI, INSULATOR):
            continue
        key = (min(n_, j_), max(n_, j_), 0 if n_ < j_ else 1)
        fa[key] = fa.get(key, 0.0) + a
        if kinds[l_] == SEMI:
            fs[key] = fs.get(key, 0.0) + a
        fg[key] = fg.get(key, 0.0) + eps[l_] * a
    pairs = sorted({(i, j) for (i, j, _) in fa})
    side_area = np.array([[fa.get((i, j, s), np.nan) for s in (0, 1)] for i, j in pairs]).reshape(-1, 2)
    asym = np.nanmax(np.abs(side_area[:, 0] - side_area[:, 1]) / np.maximum(np.nanmax(side_area, axis=1), 1e-30)) \
        if len(pairs) else 0.0
    one_sided = int(np.isnan(side_area).any(axis=1).sum())

    # each edge's facet is the mean of what its two cells computed (they
    # agree to round-off; the gate checks it)
    def mean2(d):
        return np.array([np.mean([d.get((i, j, s), 0.0) for s in (0, 1) if (i, j, s) in fa]) for i, j in pairs])

    A, As, G = (mean2(fa), mean2(fs), mean2(fg)) if pairs else (np.zeros(0),) * 3
    e = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    L = np.linalg.norm(P[e[:, 1]] - P[e[:, 0]], axis=1) if len(e) else np.zeros(0)

    dev = cv > 0
    keep_e = (A > 1e-12) & dev[e[:, 0]] & dev[e[:, 1]]
    new_id = np.full(N, -1, dtype=np.int64)
    new_id[dev] = np.arange(int(dev.sum()))
    edges = new_id[e[keep_e]]
    A, As, G, L = A[keep_e], As[keep_e], G[keep_e], L[keep_e]

    contact_nodes = {name: np.zeros(N, bool) for name in touch}
    for name, s in touch.items():
        contact_nodes[name][list(s)] = True
    for name, (a, side) in (face_contacts or {}).items():
        on = node_idx[:, a] == (0 if side == "min" else shape[a])
        contact_nodes.setdefault(name, np.zeros(N, bool))
        contact_nodes[name] |= on
    names = list(contact_nodes)
    owner = np.full(N, -1)
    for c, name in enumerate(names):
        mask = contact_nodes[name] & dev
        clash = mask & (owner >= 0)
        if clash.any():
            raise ValueError(f"{int(clash.sum())} node(s) touch both contact {names[owner[clash][0]]!r} and {name!r}")
        owner[mask] = c
    b_idx = np.concatenate([new_id[contact_nodes[nm] & dev] for nm in names]) if names else np.zeros(0, int)
    b_type = np.array(sum([[f"contact:{nm}"] * int((contact_nodes[nm] & dev).sum()) for nm in names], []),
                      dtype=object)
    order = np.argsort(b_idx, kind="stable")              # the solver pairs these with psi[is_contact]
    b_idx, b_type = b_idx[order], list(b_type[order])

    pts_nm = P[dev] + origin
    semi_node = cv_semi[dev] > 0
    net = np.asarray(node_net, float)[tuple(node_idx[dev].T)]
    contacts = []
    for nm in names:
        p = pts_nm[new_id[contact_nodes[nm] & dev]] * cm
        contacts.append(SimpleNamespace(name=nm, x_range_cm=(float(p[:, 0].min()), float(p[:, 0].max()))
                                        if len(p) else (0.0, 0.0)))
    return OctreeMesh(
        points=pts_nm * cm, triangles=np.zeros((0, 3), dtype=int), edges=edges,
        edge_weight=A * nm2 / (L * cm), edge_g=G * nm2 / (L * cm),
        facet_length=A * nm2, facet_length_semi=As * nm2,
        cv_area=cv[dev] * nm3, cv_area_semi=cv_semi[dev] * nm3,
        Cdop=np.where(semi_node, net, 0.0), ni_arr=np.where(semi_node, mat.ni, 0.0), is_insulator=~semi_node,
        dEi_arr=np.zeros(int(dev.sum())), Eg_arr=np.full(int(dev.sum()), mat.Eg_eV),
        ec_off_arr=np.full(int(dev.sum()), mat.Vt * np.log(mat.Nc / mat.ni)),
        boundary_point_index=b_idx, boundary_bc_type=b_type, boundary_edges=set(),
        domain=SimpleNamespace(contacts=contacts), h_min_cm=float(L.min() * cm) if len(L) else 0.0,
        grid_index=node_idx[dev], node_id=new_id, dim=3,
        quality=dict(facet_asymmetry=float(asym), one_sided_facets=one_sided),
        _cv_all=cv, _cv_semi_all=cv_semi, _vol=dual["vol"], _edges_all=e[keep_e], _dual=dual)


# --- the mandatory gate ---------------------------------------------------------------
def check_octree_mesh(m, label, materials, h, log=print, vol_rtol=1e-9, facet_rtol=1e-6):
    """Raise if the mesh is not an admissible box-FV mesh; called by
    build_octree_mesh on every build (never optional)."""
    errs = []
    if not (m.cv_area > 0).all():
        errs.append(f"{int((m.cv_area <= 0).sum())} non-positive control volumes")
    if (m.edge_weight < 0).any() or (m.edge_g < 0).any():
        errs.append("negative edge coupling")
    # per-label volume sums == voxel volumes (device labels and conductors)
    vox = float(np.prod(h))
    by_label = {}
    for (n_, l_), v in m._vol.items():
        by_label[l_] = by_label.get(l_, 0.0) + v
    for l_, m_ in enumerate(materials):
        want = float((label == l_).sum()) * vox
        got = by_label.get(l_, 0.0)
        if abs(got - want) > vol_rtol * max(want, 1.0) + 1e-6:
            errs.append(f"material {m_['name']!r}: dual volume {got:.6g} nm^3 != voxel volume {want:.6g}")
    if m.quality["facet_asymmetry"] > facet_rtol:
        errs.append(f"facet area differs between the two sides by {m.quality['facet_asymmetry']:.2e} (relative)")
    if m.quality["one_sided_facets"]:
        errs.append(f"{m.quality['one_sided_facets']} facet(s) seen from one side only")
    # conformity: semiconductor facet area only between semiconductor nodes
    semi = m.cv_area_semi > 0
    bad = (m.facet_length_semi > 0) & ~(semi[m.edges[:, 0]] & semi[m.edges[:, 1]])
    if bad.any():
        errs.append(f"{int(bad.sum())} edge(s) carry semiconductor facet area but join a non-semiconductor node")
    deg = np.bincount(m.edges.ravel(), minlength=len(m.points))
    if (deg == 0).any():
        errs.append(f"{int((deg == 0).sum())} isolated node(s)")
    for c in m.domain.contacts:
        if not (np.array(m.boundary_bc_type) == f"contact:{c.name}").any():
            errs.append(f"contact {c.name!r} has no nodes")
    if errs:
        raise RuntimeError("octree mesh failed the quality gate:\n  " + "\n  ".join(errs))
    log(f"  mesh gate: OK (volumes exact per material, facet symmetry {m.quality['facet_asymmetry']:.1e}, "
        f"couplings >= 0, conforming, no isolated nodes)")
