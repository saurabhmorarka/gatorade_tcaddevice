"""Level-set toolkit for the virtual fab: re-distancing, column surface
heights and 2D mask distance fields.

A region is the set {phi < 0} of a level-set function phi sampled at the
grid nodes. Set operations (min/max) keep the zero set exact but leave phi
a non-distance away from it; steps that need true distances (conformal
deposition, isotropic etch, distance-based doping) call redistance() on the
field they need, when they need it.

redistance() is a closest-point method:
  1. seeds = nodes with a sign change to any of their 3^n - 1 neighbours
     (the diagonal ones put closest points on sharp edges and corners)
  2. each seed's closest surface point by Newton projection onto the
     trilinear interpolant of phi (sub-voxel, ~1e-3 nm on smooth surfaces)
  3. every node takes its nearest seed's closest point (scipy EDT), then
     slides along the tangent plane and re-projects, keeping whichever
     surface point is closer
Accuracy on 1/0.5/1 nm voxels: ~0.02 nm on smooth surfaces. Sharp edges
and corners are resolved to under half a voxel (a sampled field cannot hold
a sharp edge; every level-set method chamfers it at voxel scale).

Steps 2-3 run in Rust for 3D fields when tcad_native is built
(native/src/levelset.rs, the same algorithm; TCAD_NATIVE=0 forces numpy).
"""
import itertools

import numpy as np
from scipy import ndimage

from core.native import native

BIG = 1.0e30


def _sample(arrs, p):
    return [ndimage.map_coordinates(a, p.T, order=1, mode="nearest") for a in arrs]


def _project(phi, grads, p, h, n_iter):
    """Newton-project points p (index coordinates) onto phi's trilinear zero set."""
    for _ in range(n_iter):
        f, *g = _sample([phi] + grads, p)
        g = np.stack(g, -1)
        p = p - ((f / np.maximum((g * g).sum(-1), 1e-30))[:, None] * g) / h
    return p


def seed_mask(phi):
    """Nodes with a sign change (phi < 0 vs >= 0) to any neighbour,
    diagonals included."""
    inside = phi < 0
    seed = np.zeros(phi.shape, bool)
    for off in itertools.product((-1, 0, 1), repeat=phi.ndim):
        if off <= (0,) * phi.ndim:
            continue
        s0 = tuple(slice(max(0, -o), n - max(0, o)) for n, o in zip(phi.shape, off))
        s1 = tuple(slice(max(0, o), n - max(0, -o)) for n, o in zip(phi.shape, off))
        x = inside[s0] != inside[s1]
        seed[s0] |= x
        seed[s1] |= x
    return seed


def _straddles(phi, p):
    """True where the voxel holding index-space point p has corners on both
    sides (phi < 0 and phi >= 0): a real interface passes through it. Set
    operations leave phi = 0 on surfaces a region merely touches; the
    interpolant is zero there too, and without this test a projected point
    would land on such a phantom sheet."""
    shape = np.array(phi.shape)
    i0 = np.clip(np.floor(p).astype(np.int64), 0, shape - 2)
    neg = np.zeros(len(p), bool)
    pos = np.zeros(len(p), bool)
    for off in itertools.product((0, 1), repeat=phi.ndim):
        v = phi[tuple((i0 + np.array(off)).T)]
        neg |= v < 0
        pos |= v >= 0
    return neg & pos


def redistance(phi, h, band=None, slide=2):
    """Signed distance (same units as h) to the zero set of phi; the sign of
    phi is kept. Nodes farther than `band` from the surface get the plain
    voxel-accurate EDT distance (only the band is refined). A field with no
    surface (all one sign) returns +-BIG."""
    phi = np.asarray(phi, dtype=float)
    h = np.asarray(h, dtype=float)
    sgn = np.where(phi < 0, -1.0, 1.0)
    seed = seed_mask(phi)
    if not seed.any():
        return sgn * BIG
    si = np.argwhere(seed)
    fast = native is not None and phi.ndim == 3
    if fast:
        phi = np.ascontiguousarray(phi)
        cp = native.ls_project(phi, h, np.ascontiguousarray(si, dtype=float), 4)
    else:
        grads = list(np.gradient(phi, *h))
        cp = _project(phi, grads, si.astype(float), h, 4)
    dist, ind = ndimage.distance_transform_edt(~seed, sampling=h, return_indices=True)
    lin = np.full(phi.shape, -1, dtype=np.int64)
    lin[tuple(si.T)] = np.arange(len(si))
    owner = lin[tuple(ind)]
    todo = np.ones(phi.shape, bool) if band is None else dist <= band
    xi = np.argwhere(todo).astype(float)
    p = cp[owner[todo]]
    if fast:
        out = dist
        out[todo] = native.ls_refine(phi, h, np.ascontiguousarray(xi), np.ascontiguousarray(p), slide)
        return sgn * out
    best = np.linalg.norm((xi - p) * h, axis=1)
    for _ in range(slide):
        _, *g = _sample([phi] + grads, p)
        n = np.stack(g, -1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        v = (xi - p) * h
        q = p + (v - (v * n).sum(1, keepdims=True) * n) / h
        q = _project(phi, grads, q, h, 3)
        f, *g = _sample([phi] + grads, q)
        resid = np.abs(f) / np.maximum(np.linalg.norm(np.stack(g, -1), axis=1), 1e-30)
        dq = np.linalg.norm((xi - q) * h, axis=1)
        better = (dq < best) & (resid < 1e-3 * h.min()) & _straddles(phi, q)
        best = np.where(better, dq, best)
        p = np.where(better[:, None], q, p)
    out = dist
    out[todo] = best
    return sgn * out


def local_distance(phi, h, pad):
    """redistance() inside the bounding box of {phi < 0} grown by `pad`
    (same units as h); everywhere else +pad. For callers that only need
    distances up to `pad` from a compact region (an implant blocker's
    footprint): much less work than the whole domain."""
    phi = np.asarray(phi, dtype=float)
    h = np.asarray(h, dtype=float)
    idx = np.argwhere(phi < 0)
    out = np.full(phi.shape, float(pad))
    if not len(idx):
        return out
    m = np.ceil(pad / h).astype(int) + 2
    lo = np.maximum(idx.min(0) - m, 0)
    hi = np.minimum(idx.max(0) + m + 1, phi.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    d = redistance(phi[box], h, band=pad)
    out[box] = np.minimum(d, pad)
    return out


def column_top(phi, y, axis=1):
    """Sub-voxel y of the first node from the top (smallest y) of each column
    that is inside phi (phi < 0), by linear interpolation to the node above.
    Returns an array over the other two axes; +inf where the column has no
    inside node, y[0] where it is inside at the top of the domain."""
    phi = np.moveaxis(np.asarray(phi, float), axis, -1)
    inside = phi < 0
    has = inside.any(-1)
    j = np.argmax(inside, axis=-1)
    jm = np.maximum(j - 1, 0)
    f1 = np.take_along_axis(phi, j[..., None], -1)[..., 0]
    f0 = np.take_along_axis(phi, jm[..., None], -1)[..., 0]
    hy = y[1] - y[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(j > 0, f0 / (f0 - f1), 0.0)
    yt = y[jm] + np.where(j > 0, frac, 0.0) * hy
    return np.where(has, yt, np.inf)


def polygon_sdf(px, pz, poly):
    """Exact signed distance (negative inside) from points (px, pz) to a
    closed polygon [(x, z), ...]."""
    poly = np.asarray(poly, dtype=float)
    a = poly
    b = np.roll(poly, -1, axis=0)
    d = np.full(px.shape, np.inf)
    inside = np.zeros(px.shape, bool)
    for (ax, az), (bx, bz) in zip(a, b):
        ex, ez = bx - ax, bz - az
        wx, wz = px - ax, pz - az
        t = np.clip((wx * ex + wz * ez) / max(ex * ex + ez * ez, 1e-300), 0.0, 1.0)
        d = np.minimum(d, np.hypot(wx - t * ex, wz - t * ez))
        crosses = ((az > pz) != (bz > pz)) & (px < ax + (pz - az) * ex / np.where(ez == 0, 1e-300, ez))
        inside ^= crosses
    return np.where(inside, -d, d)


def rect_poly(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
