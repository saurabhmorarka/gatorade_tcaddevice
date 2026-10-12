"""Transition-cell study for a 2:1-balanced octree box-FV mesh.

(a) centre-fan tets (2D quadtree analogue): every leaf cube gets a centre
    node, each face is triangulated conforming to its neighbour (Kuhn-style
    diagonal per fine face square), and tets = centre + face triangle.
    Check summed P1 cotangent couplings per edge (= circumcentric box-FV).
(b) Voronoi dual of the octree vertex set: exact cell of every vertex by
    half-space clipping; check closure, volume sum, area symmetry, and
    distance from non-cube cells to the nearest level change.
"""
import itertools
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

DIRS = [d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]


def build_octree(N, s0, n_seeds, rng):
    """Leaves {(x,y,z,s)} in a cube of side N, root cells size s0, random
    refinement seeds down to size 1, then full (26-neighbour) 2:1 balance."""
    leaves = {(x, y, z, s0) for x in range(0, N, s0) for y in range(0, N, s0) for z in range(0, N, s0)}

    def find(p):
        for (x, y, z, s) in ():
            pass
        s = s0
        while s >= 1:
            o = tuple(c - c % s for c in p)
            if o + (s,) in leaves:
                return o + (s,)
            s //= 2
        return None

    def split(c):
        x, y, z, s = c
        leaves.remove(c)
        h = s // 2
        for d in itertools.product((0, h), repeat=3):
            leaves.add((x + d[0], y + d[1], z + d[2], h))

    for _ in range(n_seeds):
        p = tuple(rng.randrange(N) for _ in range(3))
        while True:
            c = find(p)
            if c[3] == 1:
                break
            split(c)
    changed = True
    while changed:
        changed = False
        for c in list(leaves):
            if c not in leaves:
                continue
            x, y, z, s = c
            for d in DIRS:
                p = tuple(v + (s if dd > 0 else (-1 if dd < 0 else 0)) for v, dd in zip((x, y, z), d))
                if not all(0 <= v < N for v in p):
                    continue
                n = find(p)
                if n[3] > 2 * s:
                    split(n)
                    changed = True
    return leaves


def vertices(leaves):
    V = set()
    for (x, y, z, s) in leaves:
        for d in itertools.product((0, s), repeat=3):
            V.add((x + d[0], y + d[1], z + d[2]))
    return np.array(sorted(V), float)


# ---------------------------------------------------------------- (b) Voronoi
TOL = 1e-9


def clip_poly(faces, n, c, label):
    """faces: {label: [v,...]} convex polyhedron; keep n.x <= c."""
    out, cut_pts = {}, []
    for lab, poly in faces.items():
        new = []
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            da, db = n @ a - c, n @ b - c
            if da <= TOL:
                new.append(a)
                if abs(da) <= TOL:
                    cut_pts.append(a)
            if (da < -TOL and db > TOL) or (da > TOL and db < -TOL):
                t = da / (da - db)
                q = a + t * (b - a)
                new.append(q)
                cut_pts.append(q)
        if len(new) >= 3:
            out[lab] = new
    if len(cut_pts) >= 3:
        P = np.array(cut_pts)
        P = np.unique(np.round(P, 9), axis=0)
        if len(P) >= 3:
            cen = P.mean(0)
            u = P[0] - cen
            if np.linalg.norm(u) < 1e-12:
                u = P[1] - cen
            u /= np.linalg.norm(u)
            w = np.cross(n / np.linalg.norm(n), u)
            ang = np.arctan2((P - cen) @ w, (P - cen) @ u)
            out[label] = list(P[np.argsort(ang)])
    return out


def poly_area_vec(poly):
    P = np.array(poly)
    return 0.5 * np.cross(P - P[0], np.roll(P, -1, 0) - P[0]).sum(0)


def voronoi_cell(p, nbrs, box_lo, box_hi):
    lo, hi = np.array(box_lo, float), np.array(box_hi, float)
    C = [np.array(v, float) for v in itertools.product(*zip(lo, hi))]
    # cube faces of the box (each ordered)
    faces = {}
    idx = {v: i for i, v in enumerate(itertools.product((0, 1), repeat=3))}
    for a in range(3):
        for side in (0, 1):
            pts = []
            b, cc = [k for k in range(3) if k != a]
            for (u, v) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                key = [0, 0, 0]
                key[a], key[b], key[cc] = side, u, v
                pts.append(C[idx[tuple(key)]])
            out = np.zeros(3); out[a] = 1 if side else -1
            if poly_area_vec(pts) @ out < 0:
                pts = pts[::-1]
            faces[("B", a, side)] = pts
    for j, q in nbrs:
        n = q - p
        faces = clip_poly(faces, n, (q @ q - p @ p) / 2, j)
    return faces


def study_voronoi(leaves, N):
    X = vertices(leaves)
    tree = cKDTree(X)
    # local leaf size at each vertex = min size of leaves touching it
    size_at = defaultdict(lambda: 10 ** 9)
    smax_at = defaultdict(int)
    for (x, y, z, s) in leaves:
        for d in itertools.product((0, s), repeat=3):
            k = (x + d[0], y + d[1], z + d[2])
            size_at[k] = min(size_at[k], s)
            smax_at[k] = max(smax_at[k], s)
    key = [tuple(int(v) for v in x) for x in X]
    vol = np.zeros(len(X))
    area = {}
    worst_close = 0.0
    nonbox = []
    for i, p in enumerate(X):
        R = 2.0 * smax_at[key[i]]
        while True:
            nb = [j for j in tree.query_ball_point(p, R) if j != i]
            faces = voronoi_cell(p, [(j, X[j]) for j in nb], np.maximum(p - R, 0), np.minimum(p + R, N))
            # security radius: every cell vertex within R/2 of p
            vmax = max(np.linalg.norm(np.array(v) - p) for f in faces.values() for v in f)
            if 2 * vmax <= R + 1e-9:
                break
            R *= 2
        close = np.zeros(3)
        V = 0.0
        for lab, f in faces.items():
            av = poly_area_vec(f)
            close += av
            V += np.dot(np.array(f[0]), av) / 3
            if not isinstance(lab, tuple):
                a = np.linalg.norm(av)
                if a > 1e-12:
                    area[(i, lab)] = a
        worst_close = max(worst_close, np.abs(close).max())
        vol[i] = V
        # is the cell an axis-aligned box with only axis neighbours?
        nl = [lab for lab in faces if not isinstance(lab, tuple) and np.linalg.norm(poly_area_vec(faces[lab])) > 1e-12]
        offs = [X[j] - p for j in nl]
        if any(np.count_nonzero(np.abs(o) > 1e-9) != 1 for o in offs):
            nonbox.append(i)
    asym = max(abs(area[(i, j)] - area.get((j, i), 0)) for (i, j) in area)
    return X, key, size_at, smax_at, vol, area, worst_close, asym, nonbox


def level_change_dist(leaves, X, idx_list, size_at, smax_at, key):
    """Chebyshev distance (in units of the local fine size) from vertex to the
    nearest vertex where touching leaves have different sizes."""
    trans = np.array([X[i] for i in range(len(X)) if size_at[key[i]] != smax_at[key[i]]])
    if len(trans) == 0:
        return []
    t = cKDTree(trans)
    d, _ = t.query(X[idx_list], p=np.inf)
    return d


# ------------------------------------------------------- (a) centre fan tets
def cot_couplings(tets, P):
    """Summed P1 stiffness off-diagonals (-K_ij) per edge: sum over tets of
    l_kl cot(theta_kl) / 6 (opposite edge kl, dihedral theta_kl)."""
    W = defaultdict(float)
    for t in tets:
        v = P[list(t)]
        vol6 = abs(np.linalg.det(np.array([v[1] - v[0], v[2] - v[0], v[3] - v[0]])))
        if vol6 < 1e-12:
            continue
        # grad of barycentric: K_ij = V * g_i . g_j
        M = np.array([v[1] - v[0], v[2] - v[0], v[3] - v[0]])
        Ginv = np.linalg.inv(M)  # columns: gradients of lambda_1..3
        g = [-(Ginv[:, 0] + Ginv[:, 1] + Ginv[:, 2]), Ginv[:, 0], Ginv[:, 1], Ginv[:, 2]]
        V = vol6 / 6
        for a, b in itertools.combinations(range(4), 2):
            e = tuple(sorted((t[a], t[b])))
            W[e] += -V * g[a] @ g[b]
    return W


def centre_fan(leaves):
    pts = {}

    def node(c):
        if c not in pts:
            pts[c] = len(pts)
        return pts[c]

    Vset = set()
    for (x, y, z, s) in leaves:
        for d in itertools.product((0, s), repeat=3):
            Vset.add((2 * (x + d[0]), 2 * (y + d[1]), 2 * (z + d[2])))
    tets = []
    for (x, y, z, s) in leaves:
        o = np.array([2 * x, 2 * y, 2 * z])
        S = 2 * s
        c = node(tuple(o + S // 2))
        for a in range(3):
            b, cc = [k for k in range(3) if k != a]
            for side in (0, S):
                # face sub-grid: 2x2 if the face centre is a vertex (fine neighbour) else 1x1
                fc = o.copy()
                fc[a] += side
                fc[b] += S // 2
                fc[cc] += S // 2
                n = 2 if tuple(fc) in Vset else 1
                h = S // n
                for i in range(n):
                    for j in range(n):
                        def q(u, v):
                            r = o.copy()
                            r[a] += side
                            r[b] += i * h + u * h
                            r[cc] += j * h + v * h
                            return tuple(r)
                        # square corners; if an edge midpoint hangs on a 1x1 face, split that edge
                        corners = [q(0, 0), q(1, 0), q(1, 1), q(0, 1)]
                        ring = []
                        for k in range(4):
                            A, B = np.array(corners[k]), np.array(corners[(k + 1) % 4])
                            ring.append(tuple(A))
                            mid = tuple((A + B) // 2)
                            if mid in Vset:
                                ring.append(mid)
                        # triangulate the face polygon: diagonal along global (+,+) Kuhn direction
                        if len(ring) == 4:
                            # use the diagonal through the corner with min (b+cc) (consistent globally)
                            sums = [r[b] + r[cc] for r in ring]
                            k0 = int(np.argmin(sums))
                            r = ring[k0:] + ring[:k0]
                            tris = [(r[0], r[1], r[2]), (r[0], r[2], r[3])]
                        else:
                            # fan from the face centre point of this square
                            fcp = tuple((np.array(corners[0]) + np.array(corners[2])) // 2)
                            tris = [(fcp, ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))]
                        for t in tris:
                            tets.append((c, node(t[0]), node(t[1]), node(t[2])))
    P = np.zeros((len(pts), 3))
    for k, i in pts.items():
        P[i] = np.array(k) / 2.0
    return tets, P


if __name__ == "__main__":
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    N, s0 = 16, 8
    for trial in range(3):
        leaves = build_octree(N, s0, n_seeds=6, rng=rng)
        sizes = sorted({c[3] for c in leaves})
        # (a)
        tets, P = centre_fan(leaves)
        W = cot_couplings(tets, P)
        w = np.array(list(W.values()))
        neg = w < -1e-9
        # (b)
        X, key, size_at, smax_at, vol, area, clos, asym, nonbox = study_voronoi(leaves, N)
        d = level_change_dist(leaves, X, nonbox, size_at, smax_at, key)
        print(f"trial {trial}: {len(leaves)} leaves sizes {sizes}, {len(X)} vertices")
        print(f"  (a) centre fan: {len(tets)} tets, {len(w)} edges, negative couplings {neg.sum()} "
              f"(min {w.min():+.3f}, sum-of-neg {w[neg].sum():+.3f})")
        print(f"  (b) Voronoi: vol sum {vol.sum():.6f} (box {N**3}), min cell vol {vol.min():.4f}, "
              f"closure err {clos:.1e}, area asym {asym:.1e}, faces {len(area)//2}")
        if len(nonbox):
            print(f"      non-box cells {len(nonbox)}; Chebyshev dist to nearest level-change vertex: "
                  f"max {np.max(d):.1f} units (fine size 1)")
            fine = [i for i in nonbox]
            rel = [d[k] / size_at[key[i]] for k, i in enumerate(nonbox)]
            print(f"      ... in units of the vertex's own finest leaf size: max {max(rel):.2f}")
