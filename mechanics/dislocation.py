"""Edge dislocations as a Volterra cut (2D, line along z).

A dislocation with Burgers vector b is the displacement jump b across a
cut surface that ends on the dislocation line. In an eigenstrain solver the
cut is one layer of pixels with stress-free strain

    eps* = sym(b (x) n) / t

(n the cut's unit normal, t the layer thickness along n): the layer is
stretched or sheared by exactly b. The cut runs from the core to a free
surface, so only the core is a real defect; the exact cut position does
not matter beyond the layer's own pixels (Volterra).

Sign convention: b = -b_x x_hat on a vertical cut (n = x_hat) REMOVES a
half plane of thickness b_x between the core and the surface: the material
on both sides is pulled together, the classic missing-half-plane edge
dislocation of an SMT S/D, which leaves the channel next to it in tension.
For Si, b = a/2<110> = 0.384 nm.
"""
import numpy as np

B_SI = 5.431 / np.sqrt(2) / 10.0          # a/2<110>, nm


def edge_dislocation(grid, core, cut_dir, b, labels=None, solid=None):
    """(mask (ny, nx), eps* (ny*nx, 6)) for an edge dislocation whose cut
    runs from `core` (x, y nm) along `cut_dir` (unit vector, toward the
    surface) to the end of the solid. b: Burgers vector (bx, by) nm.
    Only pixels where solid (a bool (ny, nx), default labels != 0) is True
    get the eigenstrain."""
    X, Y = grid.coords()
    h = grid.h
    d = np.asarray(cut_dir, float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])                       # cut normal
    rx, ry = X - core[0], Y - core[1]
    along = rx * d[0] + ry * d[1]
    across = rx * n[0] + ry * n[1]
    # pixel thickness along n of a square pixel: h (|n_x| + |n_y|) / max... use the
    # band |across| < t/2 with t = h (|n_x| + |n_y|), so each column of the cut
    # is crossed by about one pixel
    t = h * (abs(n[0]) + abs(n[1]))
    mask = (along >= 0) & (across >= -t / 2) & (across < t / 2)      # half-open: one pixel per column
    if solid is None and labels is not None:
        solid = labels != 0
    if solid is not None:
        mask &= solid
    bb = np.asarray(b, float)
    e = np.zeros(6)
    e[0] = bb[0] * n[0] / t
    e[1] = bb[1] * n[1] / t
    e[5] = (bb[0] * n[1] + bb[1] * n[0]) / t          # engineering shear
    eps = np.zeros((mask.size, 6))
    eps[mask.ravel()] = e
    return mask, eps


def edge_dislocation_infinite(x, y, b, E, nu):
    """Analytic stress (sxx, syy, sxy) of an edge dislocation (b along x,
    line along z, glide plane y = 0, extra half plane at y > 0 for b > 0)
    in an infinite isotropic medium, plane strain (Hirth & Lothe 3-43).
    Here y points UP (the textbook frame); the solver's y points down."""
    mu = E / (2 * (1 + nu))
    D = mu * b / (2 * np.pi * (1 - nu))
    r4 = (x ** 2 + y ** 2) ** 2
    sxx = -D * y * (3 * x ** 2 + y ** 2) / r4
    syy = D * y * (x ** 2 - y ** 2) / r4
    sxy = D * x * (x ** 2 - y ** 2) / r4
    return sxx, syy, sxy
