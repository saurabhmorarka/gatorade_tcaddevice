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
def target_spacing(label, kinds, cell_net, h, h_fine, h_fine_other, h_max, growth,
                   growth_insulator=None, h_max_insulator=None, fine_max_doping=None):
    """Per-axis target leaf size (units) at every voxel.

    label      int array (nx, ny, nz), -1 = gas, else index into kinds
    kinds      list of SEMI / INSULATOR / CONDUCTOR per label
    cell_net   net doping per voxel (sign changes = junctions)
    h          voxel size per axis (nm)
    h_fine     spacing (nm) wanted at semiconductor interfaces and junctions
    h_fine_other  spacing at the other device interfaces (insulator /
               insulator, insulator / conductor)
    The spacing grows as h_fine + (growth - 1) * distance, capped at h_max;
    gas and conductor voxels only need what their labels force.
    growth_insulator, h_max_insulator (default growth, h_max): the same
    inside insulators, which only carry the electrostatics (capacitance):
    fine at the semiconductor interface like the silicon side, coarsening
    much faster away from it. Layer thicknesses stay exact either way (the
    tree splits at every label change).
    fine_max_doping (cm^-3, default None = everywhere): a semiconductor
    surface takes h_fine only where the semiconductor is doped below this -
    the channel and lightly doped body, not a degenerate S/D's dielectric
    surfaces (no inversion layer to resolve), which take h_fine_other.
    Junctions and metal contacts keep h_fine (a sub-grid Schottky contact's
    current moved 1.9% without it, planar_tiny)."""
    h = np.asarray(h, float)
    kind = np.array(list(kinds) + [GAS])[label]
    semi = kind == SEMI
    dev = semi | (kind == INSULATOR)
    big = 1e9

    def dist(seed):
        if not seed.any():
            return np.full(label.shape, big)
        return ndimage.distance_transform_edt(~seed, sampling=h)

    # Per axis: the spacing ALONG axis a is graded from the interfaces and
    # junctions crossed along a (faces normal to a) - the Si/oxide surface
    # needs fine y, an STI sidewall fine z, a junction fine spacing across
    # it; an isotropic distance would put 0.5 nm y-spacing down the whole
    # sidewall (the rectilinear mesher's per-axis lines do this right).
    s = np.sign(cell_net) * semi
    ins = kind == INSULATOR
    g_ins = growth if growth_insulator is None else growth_insulator
    hmax_ins = h_max if h_max_insulator is None else h_max_insulator
    tgt = np.empty((3,) + label.shape)
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a], sl1[a] = slice(0, -1), slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        seed_s = np.zeros(label.shape, bool)
        hit = semi[sl0] != semi[sl1]
        if fine_max_doping is not None:          # (a metal contact keeps h_fine: Schottky contacts)
            keep = (semi & (np.abs(cell_net) < fine_max_doping)) | (kind == CONDUCTOR)
            hit &= keep[sl0] | keep[sl1]
        hit |= (s[sl0] * s[sl1]) < 0
        seed_s[sl0] |= hit
        seed_s[sl1] |= hit
        seed_o = np.zeros(label.shape, bool)
        hit = (label[sl0] != label[sl1]) & (dev[sl0] | dev[sl1])
        seed_o[sl0] |= hit
        seed_o[sl1] |= hit
        g = np.where(ins, g_ins, growth) - 1.0
        t = np.minimum(h_fine[a] + g * dist(seed_s), h_fine_other[a] + g * dist(seed_o))
        t = np.minimum(t, np.where(ins, hmax_ins[a], h_max[a]))
        tgt[a] = np.where(dev, t, big) / h[a]
    return np.maximum(tgt, 1.0)


# --- the tree -------------------------------------------------------------------------
class BoxTree:
    """Leaves as integer boxes [lo, hi) in voxel units, with an owner array
    (leaf id per voxel) for point location and neighbour lookup."""

    def __init__(self, label, tgt, root, semi_labels=()):
        self.label = label
        # semi_label[l] for l in -1..: is label l a semiconductor (index -1 = gas)
        self.semi_label = np.zeros(int(label.max()) + 2, bool)
        for l in semi_labels:
            self.semi_label[l] = True
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

    def has_semi_nbr(self, i):
        lo = np.maximum(self.lo[i] - 1, 0)
        hi = np.minimum(self.hi[i] + 1, self.shape)
        return bool(self.semi_label[self.label[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]].any())

    def compact(self):
        """Drop dead boxes (the parents of split leaves) and renumber, so
        refine() can restart from the current leaves (the Rust refine takes
        its start boxes as roots)."""
        ids = np.flatnonzero(self.alive)
        new = np.full(len(self.alive), -1, dtype=np.int32)
        new[ids] = np.arange(len(ids), dtype=np.int32)
        self.lo = [self.lo[i] for i in ids]
        self.hi = [self.hi[i] for i in ids]
        self.alive = [True] * len(ids)
        self.owner = new[self.owner]

    def split_for_stray(self, nodes, is_conductor):
        """Splits that put a metal-surface node next to each `nodes` entry
        (voxel-grid node indices of device nodes whose Voronoi cell reaches
        into a metal they are not on). For each node: the nearest device
        leaves (closed box containing the node, then one and two voxels
        further) that share a face with a conductor and whose tangential
        extent strictly contains the node's coordinate get split there;
        face match then carries the split into the metal leaf, so the
        surface gains a node in line with the offender. Only geometric
        facts are used (no orientation or layer assumptions), so it holds
        for any gate / dielectric / contact shape. Returns the number of
        splits made."""
        n_split = 0
        sh = self.shape
        for q in np.asarray(nodes, dtype=np.int64):
            made = 0
            for ring in (1, 2, 3, 4, 6):
                lo = np.maximum(q - ring, 0)
                hi = np.minimum(q + ring, sh)
                cand = np.unique(self.owner[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
                made = 0
                for i in cand:
                    lab = self.label[tuple(self.lo[i])]
                    if not self.alive[i] or lab < 0 or is_conductor[lab]:
                        continue
                    for a in range(3):
                        faces = [self.face_ids(i, a, side) for side in (-1, 1)]
                        if not any(f is not None and is_conductor[f[1]].any() for f in faces):
                            continue
                        # split at q in each tangential axis it lies strictly inside;
                        # one child suffices: both carry the new plane
                        for b in range(3):
                            if b != a and self.lo[i][b] < q[b] < self.hi[i][b]:
                                i = self.split(i, b, int(q[b]))[0]
                                made += 1
                        break
                if made:
                    n_split += made
                    break
            if not made:
                # every nearby conductor-facing leaf already has a plane at q:
                # halve the nearest ones along their tangential axes instead
                # (refines the metal surface there, so the next pass differs)
                lo = np.maximum(q - 2, 0)
                hi = np.minimum(q + 2, sh)
                for i in np.unique(self.owner[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]):
                    lab = self.label[tuple(self.lo[i])]
                    if not self.alive[i] or lab < 0 or is_conductor[lab]:
                        continue
                    for a in range(3):
                        faces = [self.face_ids(i, a, side) for side in (-1, 1)]
                        if not any(f is not None and is_conductor[f[1]].any() for f in faces):
                            continue
                        for b in range(3):
                            if b != a and self.hi[i][b] - self.lo[i][b] >= 2:
                                i = self.split(i, b, int((self.lo[i][b] + self.hi[i][b]) // 2))[0]
                                n_split += 1
                        break
        return n_split

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
        """Face match only across interfaces with a semiconductor or a
        conductor on one side (`semi_label` marks both): carrier facets
        must join semiconductor nodes, and a terminal surface must carry
        nodes. Elsewhere (insulator / insulator, gas) the per-material
        split is exact by clipping, and matching would spread refinement
        for nothing."""
        my = self.label[self.lo[i][0], self.lo[i][1], self.lo[i][2]]
        if not self.semi_label[my] and not self.has_semi_nbr(i):
            return None
        for axis in range(3):
            for side in (-1, 1):
                f = self.face_ids(i, axis, side)
                if f is None:
                    continue
                own, lab = f
                diff = lab != my
                if not self.semi_label[my]:
                    diff &= self.semi_label[lab]
                for n in np.unique(own[diff]):
                    for b in range(3):
                        if b == axis:
                            continue
                        if self.lo[n][b] > self.lo[i][b]:
                            return b, int(self.lo[n][b])
                        if self.hi[n][b] < self.hi[i][b]:
                            return b, int(self.hi[n][b])
        return None

    def refine(self, log=None, use_native=True):
        from core.native import native
        if use_native and native is not None and hasattr(native, "boxtree_refine"):
            lo, hi, alive, owner, n_split = native.boxtree_refine(
                np.ascontiguousarray(self.label, dtype=np.int32), np.ascontiguousarray(self.tgt, dtype=float),
                np.ascontiguousarray(self.semi_label[:-1]), np.array(self.lo, dtype=np.int64),
                np.array(self.hi, dtype=np.int64))
            self.lo, self.hi = list(lo), list(hi)
            self.alive = list(alive)
            self.owner = owner.astype(np.int32)
            if log:
                log(f"  tree: {int(np.sum(self.alive))} leaves after {n_split} splits")
            return
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
                      h_max=(8.0, 8.0, 8.0), growth=1.35, growth_insulator=2.0, h_max_insulator=(16.0, 16.0, 16.0),
                      fine_max_doping=1e19, root=None, log=print):
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
    tgt = target_spacing(label, kinds, cell_net, h, h_fine, h_fine_other, h_max, growth,
                         growth_insulator, h_max_insulator, fine_max_doping)
    if root is None:
        root = [max(1, int(round(2 * h_max[a] / h[a]))) for a in range(3)]
    # face match across semiconductor interfaces (carrier facets must join
    # semiconductor nodes) and conductor surfaces (a terminal is a Dirichlet
    # surface: it must carry nodes, not lie between them)
    tree = BoxTree(label, tgt, root, semi_labels=[k for k, m in enumerate(materials) if m["kind"] in (SEMI, CONDUCTOR)])
    tree.refine(log=log)
    t_tree = time.perf_counter() - t0
    # No device node may have Voronoi volume inside a metal it is not on
    # (that volume would be dropped, and the flux across that piece of the
    # metal surface lost). Exact test on the dual; where it fails, split
    # next to the offender and rebuild. Gated in check_octree_mesh.
    is_conductor = np.array([m["kind"] == CONDUCTOR for m in materials] + [False])     # index -1 = gas
    for _ in range(_STRAY_PASSES):
        ids, LO, HI = tree.leaves()
        leaf_label = label[LO[:, 0], LO[:, 1], LO[:, 2]]
        dual = _dual(tree, ids, LO, HI, leaf_label, label, h, log)
        bad = _stray_nodes(dual, label, kinds, h)
        if not len(bad) or not tree.split_for_stray(dual["node_idx"][bad], is_conductor):
            break
        if log:
            log(f"  stray metal volume at {len(bad)} node(s): split next to them, refine again")
        tree.compact()
        tree.refine(log=log)
    m = _assemble(dual, materials, h, node_net, mat, np.asarray(origin_nm, float), face_contacts, label)
    m.leaves = (LO, HI, leaf_label)
    m.tree_time_s = t_tree
    m.build_time_s = time.perf_counter() - t0
    check_octree_mesh(m, label, materials, h, log=log)
    log(f"  octree mesh: {len(m.points)} device nodes, {len(m.edges)} edges, "
        f"{m.n_clipped} clipped cells; tree {t_tree:.1f} s, total {m.build_time_s:.1f} s")
    return m


_STRAY_PASSES = 6


def _voxels_around(label, node_idx):
    """(N, 8) labels of the voxels around each grid node (-1 outside)."""
    padded = np.pad(np.asarray(label), 1, constant_values=-1)
    return np.stack([padded[node_idx[:, 0] + a, node_idx[:, 1] + b, node_idx[:, 2] + c]
                     for a in (0, 1) for b in (0, 1) for c in (0, 1)], axis=1)


def _stray_parts(dual, label, kinds, h):
    """Mask over the dual's volume parts: metal volume in the cell of a
    device node that is not on that metal."""
    VN, VL, VV = dual["vol"]
    kinds = np.array(list(kinds) + [GAS])
    N = dual["N"]
    dev = np.bincount(VN, weights=VV * np.isin(kinds[VL], (SEMI, INSULATOR)), minlength=N) > 0
    around = _voxels_around(label, dual["node_idx"])
    on_it = (around[VN] == VL[:, None]).any(axis=1)
    return (kinds[VL] == CONDUCTOR) & dev[VN] & ~on_it & (VV > 1e-9 * float(np.prod(h)))


def _stray_nodes(dual, label, kinds, h):
    return np.unique(dual["vol"][0][_stray_parts(dual, label, kinds, h)])


def _axis_neighbours(is_node, node_idx, shape):
    """Grid index of the nearest node along -a and +a from every node (-1
    if none)."""
    nshape = is_node.shape
    N = len(node_idx)
    nb = np.full((N, 3, 2), -1, dtype=np.int64)
    for a in range(3):
        pos = np.arange(nshape[a]).reshape([-1 if b == a else 1 for b in range(3)])
        big = nshape[a] + 1
        nxt = np.flip(np.minimum.accumulate(np.flip(np.where(is_node, pos, big), axis=a), axis=a), axis=a)
        prv = np.maximum.accumulate(np.where(is_node, pos, -1), axis=a)
        lo_sl = [slice(None)] * 3
        hi_sl = [slice(None)] * 3
        lo_sl[a], hi_sl[a] = slice(0, -1), slice(1, None)
        nxt_s = np.full(nshape, big)
        nxt_s[tuple(lo_sl)] = nxt[tuple(hi_sl)]
        prv_s = np.full(nshape, -1)
        prv_s[tuple(hi_sl)] = prv[tuple(lo_sl)]
        f = nxt_s[tuple(node_idx.T)]
        nb[:, a, 1] = np.where(f < big, f, -1)
        nb[:, a, 0] = prv_s[tuple(node_idx.T)]
    return nb


def _dual(tree, ids, LO, HI, leaf_label, label, h, log):
    """Voronoi cells of all leaf corners, as flat arrays: volumes
    (node, label, nm^3) and facet parts (node, neighbour, label, nm^2),
    each seen from the node's own cell (both cells of an edge report its
    facet; the gate compares them)."""
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
    P = node_idx * h                                                    # nm from the grid origin
    dom_hi = shape * h
    nb = _axis_neighbours(is_node, node_idx, shape)
    half = np.zeros((N, 3, 2))
    ok = np.ones(N, bool)
    for a in range(3):
        ok &= ((nb[:, a, 0] >= 0) | (node_idx[:, a] == 0)) & ((nb[:, a, 1] >= 0) | (node_idx[:, a] == shape[a]))
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
    VN, VL, VV, FN, FJ, FL, FA = [], [], [], [], [], [], []

    # fast path: the box cell's octants, each inside one leaf
    octant_lab = {}
    for o in itertools.product((0, 1), repeat=3):
        w = np.stack([half[box, a, o[a]] for a in range(3)], axis=1)
        sign = np.array([1 if o[a] else -1 for a in range(3)])
        cen = node_idx[box] + sign * (w / 2) / h
        v = np.clip(np.floor(cen).astype(np.int64), 0, shape - 1)
        lab = label[v[:, 0], v[:, 1], v[:, 2]]
        octant_lab[o] = lab
        vol = w.prod(axis=1)
        s = vol > 0
        VN.append(box[s]); VL.append(lab[s]); VV.append(vol[s])
    for a in range(3):
        others = [b for b in range(3) if b != a]
        for side in (0, 1):
            q = nb[box, a, side]
            has = q >= 0
            qi = node_idx[box].copy()
            qi[:, a] = np.where(has, q, 0)
            qid = np.where(has, node_id[tuple(qi.T)], -1)
            for o in itertools.product((0, 1), repeat=3):
                if o[a] != side:
                    continue
                part = half[box, others[0], o[others[0]]] * half[box, others[1], o[others[1]]]
                s = has & (part > 0)
                FN.append(box[s]); FJ.append(qid[s]); FL.append(octant_lab[o][s]); FA.append(part[s])

    # transition nodes: exact clipping (Rust kernel, numpy fallback)
    if len(clipped):
        from core.native import native
        leaf_of_owner = np.full(len(tree.lo), -1, dtype=np.int64)
        leaf_of_owner[ids] = np.arange(len(ids))
        todo = clipped
        R = 2.0 * half[todo].reshape(len(todo), -1).max(axis=1) + 4.0 * h.max()
        while len(todo):
            if native is not None and hasattr(native, "voronoi_cells"):
                if hasattr(native, "ball_query"):         # 4.4 s -> ~0.1 s on the full planar device
                    ptr, idx = native.ball_query(np.ascontiguousarray(P), todo.astype(np.int64), R.astype(float),
                                                 float(np.median(R)))
                else:
                    lists = kd.query_ball_point(P[todo], R)
                    ptr = np.zeros(len(todo) + 1, dtype=np.int64)
                    ptr[1:] = np.cumsum([len(x) for x in lists])
                    idx = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64, count=int(ptr[-1]))
                comp, vp, vl, vv, fp, fn, fl, fa = native.voronoi_cells(
                    np.ascontiguousarray(P), todo.astype(np.int64), ptr, idx, R.astype(float), dom_hi.astype(float),
                    h.astype(float), np.ascontiguousarray(tree.owner, dtype=np.int32),
                    np.ascontiguousarray(LO, dtype=np.int64), np.ascontiguousarray(HI, dtype=np.int64),
                    leaf_label.astype(np.int64), leaf_of_owner)
                done = np.asarray(comp, bool)
                rows_v = np.repeat(np.arange(len(todo)), np.diff(vp))
                rows_f = np.repeat(np.arange(len(todo)), np.diff(fp))
                VN.append(todo[rows_v]); VL.append(vl); VV.append(vv)
                FN.append(todo[rows_f]); FJ.append(fn); FL.append(fl); FA.append(fa)
            else:
                done = np.ones(len(todo), bool)
                for r, n_ in enumerate(todo):
                    _clip_one_py(n_, P, kd, R[r], dom_hi, h, shape, tree.owner, ids, LO, HI, leaf_label,
                                 VN, VL, VV, FN, FJ, FL, FA)
            todo, R = todo[~done], 2.0 * R[~done]
    cat = (lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt))
    if log:
        log(f"  dual: {N} nodes, {len(box)} box cells, {len(clipped)} clipped")
    return dict(N=N, node_idx=node_idx, P=P, n_clipped=len(clipped), box_nodes=box, clipped_nodes=clipped,
                vol=(cat(VN, np.int64), cat(VL, np.int64), cat(VV, float)),
                facet=(cat(FN, np.int64), cat(FJ, np.int64), cat(FL, np.int64), cat(FA, float)))


def _clip_one_py(n_, P, kd, R, dom_hi, h, shape, owner, ids, LO, HI, leaf_label, VN, VL, VV, FN, FJ, FL, FA):
    """numpy fallback (and reference) for one transition node."""
    p = P[n_]
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
    blo = np.clip(np.floor(pts.min(0) / h + 1e-9).astype(int), 0, shape - 1)
    bhi = np.clip(np.maximum(np.ceil(pts.max(0) / h - 1e-9).astype(int), blo + 1), 1, shape)
    leaves_here = np.unique(owner[blo[0]:bhi[0], blo[1]:bhi[1], blo[2]:bhi[2]])
    row = {int(i): k for k, i in enumerate(ids)}
    rows = [row[int(L)] for L in leaves_here]
    labs = {int(leaf_label[k]) for k in rows}

    def emit(part, l_):
        v = _volume(part)
        if v > 1e-12:
            VN.append(np.array([n_])); VL.append(np.array([l_])); VV.append(np.array([v]))
        for j, f in part.items():
            if isinstance(j, tuple):
                continue
            a_ = float(np.linalg.norm(_area_vec(f)))
            if a_ > _FACET_REL * float((P[j] - p) @ (P[j] - p)):
                FN.append(np.array([n_])); FJ.append(np.array([j])); FL.append(np.array([l_]))
                FA.append(np.array([a_]))

    if len(labs) == 1:
        emit(faces, labs.pop())
        return
    for k in rows:
        part = _clip_to_box(faces, LO[k] * h, HI[k] * h)
        if len(part) >= 4:
            emit(part, int(leaf_label[k]))


def _assemble(dual, materials, h, node_net, mat, origin, face_contacts, label):
    shape = label.shape
    N, P, node_idx = dual["N"], dual["P"], dual["node_idx"]
    kinds = np.array([m["kind"] for m in materials] + [GAS])
    semi_mat = [m.get("material") or mat if m["kind"] == SEMI else None for m in materials]
    eps = np.array([semi_mat[k].eps if m["kind"] == SEMI else EPS0 * float(m.get("eps_r", 1.0))
                    for k, m in enumerate(materials)] + [0.0])
    nm2, nm3, cm = 1e-14, 1e-21, 1e-7
    VN, VL, VV = dual["vol"]
    kv = kinds[VL]
    cv = np.bincount(VN, weights=VV * np.isin(kv, (SEMI, INSULATOR)), minlength=N)
    cv_semi = np.bincount(VN, weights=VV * (kv == SEMI), minlength=N)
    label_vol = np.bincount(VL + 1, weights=VV, minlength=len(materials) + 1)       # index 0 = gas
    # A contact node lies ON its conductor (one of its 8 voxels is that
    # metal). Not "its Voronoi cell reaches the metal": a hanging node's
    # cell can poke into the metal past the surface nodes, and that rule
    # pinned insulator nodes 0.5 nm above the channel to the gate
    # potential (a local 3x Cox; Id off by up to 16% with the tree layout).
    around = _voxels_around(label, node_idx)
    touch = {}
    for l_, m_ in enumerate(materials):
        if m_["kind"] == CONDUCTOR:
            touch.setdefault(m_["contact"], []).append(np.flatnonzero((around == l_).any(axis=1)))
    touch = {k: np.unique(np.concatenate(v)) for k, v in touch.items()}

    FN, FJ, FL, FA = dual["facet"]
    # Geometric area of every facet from each side, whatever material it is
    # labelled with: a facet lying exactly ON a metal surface is labelled by
    # sampling one side of that plane, and the two cells can sample
    # different sides (one says metal, the other the epi under it). Such a
    # side then takes the other side's device parts below.
    g_ok = FJ >= 0
    g_key = (np.minimum(FN, FJ) * N + np.maximum(FN, FJ))[g_ok] * 2 + (FN > FJ)[g_ok]
    g_u, g_inv = np.unique(g_key, return_inverse=True)
    g_area = np.bincount(g_inv, weights=FA[g_ok])
    kf = kinds[FL]
    d = np.isin(kf, (SEMI, INSULATOR)) & (FJ >= 0)
    FN, FJ, FL, FA, kf = FN[d], FJ[d], FL[d], FA[d], kf[d]
    lo, hi = np.minimum(FN, FJ), np.maximum(FN, FJ)
    side = (FN > FJ).astype(np.int64)
    key = (lo * N + hi) * 2 + side
    ukey, inv = np.unique(key, return_inverse=True)
    sA = np.bincount(inv, weights=FA)
    sS = np.bincount(inv, weights=FA * (kf == SEMI))
    sG = np.bincount(inv, weights=FA * eps[FL])
    pk, sd = ukey // 2, ukey % 2
    upair, pinv = np.unique(pk, return_inverse=True)
    A2 = np.full((len(upair), 2), np.nan)
    S2 = np.full((len(upair), 2), np.nan)
    G2 = np.full((len(upair), 2), np.nan)
    A2[pinv, sd], S2[pinv, sd], G2[pinv, sd] = sA, sS, sG
    T2 = np.full((len(upair), 2), np.nan)                     # geometric area per side (all labels)
    for sd_ in (0, 1):
        k_ = np.searchsorted(g_u, upair * 2 + sd_)
        k_ = np.minimum(k_, len(g_u) - 1)
        hit = g_u[k_] == upair * 2 + sd_
        T2[hit, sd_] = g_area[k_[hit]]
    for sd_ in (0, 1):                                        # a side that saw only metal: mirror the other
        miss = np.isnan(A2[:, sd_]) & ~np.isnan(A2[:, 1 - sd_]) & ~np.isnan(T2).any(axis=1)
        miss &= np.abs(T2[:, 0] - T2[:, 1]) <= 1e-6 * np.maximum(T2.max(axis=1), 1e-30)
        A2[miss, sd_], S2[miss, sd_], G2[miss, sd_] = A2[miss, 1 - sd_], S2[miss, 1 - sd_], G2[miss, 1 - sd_]
    both = ~np.isnan(A2).any(axis=1)
    one_sided = int((~both).sum())
    asym = float(np.max(np.abs(A2[both, 0] - A2[both, 1]) / np.maximum(A2[both].max(axis=1), 1e-30))) \
        if both.any() else 0.0
    A, As, G = np.nanmean(A2, axis=1), np.nanmean(S2, axis=1), np.nanmean(G2, axis=1)
    e = np.stack([upair // N, upair % N], axis=1)
    L = np.linalg.norm(P[e[:, 1]] - P[e[:, 0]], axis=1)

    dev = cv > 0
    keep_e = (A > 0) & dev[e[:, 0]] & dev[e[:, 1]]
    new_id = np.full(N, -1, dtype=np.int64)
    new_id[dev] = np.arange(int(dev.sum()))
    edges = new_id[e[keep_e]]
    A, As, G, L = A[keep_e], As[keep_e], G[keep_e], L[keep_e]

    contact_nodes = {}
    for name, s in touch.items():
        mask = np.zeros(N, bool)
        mask[s] = True
        contact_nodes[name] = mask
    for name, (a, side_) in (face_contacts or {}).items():
        on = node_idx[:, a] == (0 if side_ == "min" else shape[a])
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
    # metal volume in the cells of device nodes not on that metal (gated)
    stray = _stray_parts(dual, label, kinds[:-1], h)
    stray_vol, stray_nodes = float(VV[stray].sum()), int(len(np.unique(VN[stray])))
    b_idx = np.concatenate([new_id[contact_nodes[nm] & dev] for nm in names]) if names else np.zeros(0, int)
    b_type = np.array(sum([[f"contact:{nm}"] * int((contact_nodes[nm] & dev).sum()) for nm in names], []),
                      dtype=object)
    order = np.argsort(b_idx, kind="stable")              # the solver pairs these with psi[is_contact]
    b_idx, b_type = b_idx[order], list(b_type[order])

    pts_nm = P[dev] + origin
    semi_node = cv_semi[dev] > 0
    # band structure per node: the semiconductor filling most of its control
    # volume (a Si / SiGe interface node takes the larger share)
    from core.materials import delta_Ei_of
    sl = np.flatnonzero([m["kind"] == SEMI for m in materials])
    vol_l = np.zeros((N, len(materials)))
    s_ = kv == SEMI
    np.add.at(vol_l, (VN[s_], VL[s_]), VV[s_])
    k_node = np.where(vol_l[:, sl].max(axis=1) > 0, sl[np.argmax(vol_l[:, sl], axis=1)], -1)[dev] \
        if len(sl) else np.full(int(dev.sum()), -1)
    def per_node(f, fill):
        tab = np.array([f(semi_mat[k]) if semi_mat[k] is not None else fill for k in range(len(materials))])
        return np.where(k_node >= 0, tab[np.maximum(k_node, 0)], fill)
    ni_node = per_node(lambda m_: m_.ni, 0.0)
    dEi_node = per_node(lambda m_: delta_Ei_of(m_, mat), 0.0)
    Eg_node = per_node(lambda m_: m_.Eg_eV, mat.Eg_eV)
    ecoff_node = per_node(lambda m_: m_.Vt * np.log(m_.Nc / m_.ni), mat.Vt * np.log(mat.Nc / mat.ni))
    net = np.asarray(node_net, float)[tuple(node_idx[dev].T)]
    contacts = []
    for nm in names:
        p = pts_nm[new_id[contact_nodes[nm] & dev]] * cm
        contacts.append(SimpleNamespace(name=nm, x_range_cm=(float(p[:, 0].min()), float(p[:, 0].max()))
                                        if len(p) else (0.0, 0.0)))
    n_dev = int(dev.sum())
    return OctreeMesh(
        points=pts_nm * cm, triangles=np.zeros((0, 3), dtype=int), edges=edges,
        edge_weight=A * nm2 / (L * cm), edge_g=G * nm2 / (L * cm),
        facet_length=A * nm2, facet_length_semi=As * nm2,
        cv_area=cv[dev] * nm3, cv_area_semi=cv_semi[dev] * nm3,
        Cdop=np.where(semi_node, net, 0.0), ni_arr=np.where(semi_node, ni_node, 0.0), is_insulator=~semi_node,
        dEi_arr=np.where(semi_node, dEi_node, 0.0), Eg_arr=Eg_node, ec_off_arr=ecoff_node,
        boundary_point_index=b_idx, boundary_bc_type=b_type, boundary_edges=set(),
        domain=SimpleNamespace(contacts=contacts), h_min_cm=float(L.min() * cm) if len(L) else 0.0,
        grid_index=node_idx[dev], node_id=new_id, dim=3,
        quality=dict(facet_asymmetry=asym, one_sided_facets=one_sided, stray_metal_volume=stray_vol,
                     stray_metal_nodes=stray_nodes),
        label_volume=label_vol, n_clipped=dual["n_clipped"])


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
    for l_, m_ in enumerate(materials):
        want = float((label == l_).sum()) * vox
        got = float(m.label_volume[l_ + 1])
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
    if m.quality["stray_metal_nodes"]:
        errs.append(f"{m.quality['stray_metal_nodes']} device node(s) have {m.quality['stray_metal_volume']:.3g} nm^3 "
                    f"of Voronoi cell inside a metal they are not on")
    deg = np.bincount(m.edges.ravel(), minlength=len(m.points))
    if (deg == 0).any():
        errs.append(f"{int((deg == 0).sum())} isolated node(s)")
    for c in m.domain.contacts:
        if not (np.array(m.boundary_bc_type) == f"contact:{c.name}").any():
            errs.append(f"contact {c.name!r} has no nodes")
    if errs:
        raise RuntimeError("octree mesh failed the quality gate:\n  " + "\n  ".join(errs))
    log(f"  mesh gate: OK (volumes exact per material, facet symmetry {m.quality['facet_asymmetry']:.1e}, "
        f"couplings >= 0, conforming, no isolated nodes, no stray metal volume)")
