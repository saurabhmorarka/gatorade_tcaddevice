"""2D linear elastic stress through a process flow (plane strain).

The structure is a uniform pixel grid (one material label per pixel, the
fab's voxel picture of an x-y slice): x along the channel, y the depth,
positive down. Each pixel is a bilinear (Q1) element, 2x2 Gauss.
Plane strain: eps_zz = eps_yz = eps_xz = 0 (a wide planar device), so
sigma_zz follows from the full 6x6 stiffness. Boundaries: the side walls
are symmetry planes (rollers, u_x = 0), the bottom is held (u_y = 0), the
top is free.

Process history. Stress is tracked step by step, purely elastically:
  * birth (deposit / epitaxy): new pixels join stress-free in the CURRENT
    shape. If the material is a stressor, its stress-free strain eps*
    (lattice mismatch, or eps* = -S sigma0 for an intrinsic film stress
    sigma0) is switched on at birth, and the structure re-equilibrates.
  * death (etch / CMP / pull): removed pixels release the forces their
    stress exerted on the rest, and the structure re-equilibrates. This is
    how a dummy-gate pull or a cap strip moves the channel stress.
  * anneal (relax): chosen pixels give up a fraction r of their current
    stress (their stress-free reference moves by r*S*sigma): the elastic
    picture of stress memorization (SMT).
Everything is linear, so every increment is split EXACTLY by stressor:
the stress is carried as one field per stressor and the total is their
sum (one factorization, one right-hand side per stressor).

Removed / not-yet-deposited pixels are a very soft "void" (stiffness
VOID_FACTOR x silicon), which keeps the matrix non-singular; their stress
is reported as 0.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PLANE = [0, 1, 5]          # Voigt rows of the in-plane components (xx, yy, xy)
VOID_FACTOR = 1e-7


def _B_matrices(h):
    """B (3x8) at the 2x2 Gauss points and the centre, for a square
    element of side h; node order (x0,y0), (x1,y0), (x1,y1), (x0,y1),
    dofs (ux, uy) per node."""
    g = 1.0 / np.sqrt(3.0)
    pts = [(-g, -g), (g, -g), (g, g), (-g, g), (0.0, 0.0)]
    sx = np.array([-1, 1, 1, -1], float)
    sy = np.array([-1, -1, 1, 1], float)
    out = []
    for xi, eta in pts:
        dNx = sx * (1 + sy * eta) / 4 * (2.0 / h)
        dNy = sy * (1 + sx * xi) / 4 * (2.0 / h)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNx
        B[1, 1::2] = dNy
        B[2, 0::2] = dNy
        B[2, 1::2] = dNx
        out.append(B)
    return np.array(out[:4]), out[4]


class Grid2D:
    """ny x nx pixels of side h (nm); pixel (j, i) spans x in [x0 + i h,
    x0 + (i+1) h], y in [y0 + j h, y0 + (j+1) h]."""

    def __init__(self, nx, ny, h, x0=0.0, y0=0.0):
        self.nx, self.ny, self.h, self.x0, self.y0 = nx, ny, float(h), float(x0), float(y0)
        self.xc = x0 + (np.arange(nx) + 0.5) * h
        self.yc = y0 + (np.arange(ny) + 0.5) * h
        j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        n0 = j * (nx + 1) + i
        nodes = np.stack([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1], axis=-1).reshape(-1, 4)
        self.edofs = np.empty((nodes.shape[0], 8), dtype=np.int64)
        self.edofs[:, 0::2] = 2 * nodes
        self.edofs[:, 1::2] = 2 * nodes + 1
        self.ndof = 2 * (nx + 1) * (ny + 1)
        Bg, self.Bc = _B_matrices(h)
        self.Bg = Bg                                          # (4, 3, 8)
        self.detJ = h * h / 4.0
        # G[a, p, q, b] = sum_g B_g[p, a] B_g[q, b] detJ  ->  Ke = G : D
        self.G = np.einsum("gpa,gqb->apqb", Bg, Bg) * self.detJ
        self.Bsum = Bg.sum(axis=0) * self.detJ                # (3, 8): int B dA (B is linear)
        # Dirichlet dofs
        jj, ii = np.meshgrid(np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
        node = jj * (nx + 1) + ii
        fixed = np.zeros(self.ndof, bool)
        fixed[2 * node[:, 0]] = True                     # left wall u_x = 0
        fixed[2 * node[:, -1]] = True                    # right wall u_x = 0
        fixed[2 * node[-1, :] + 1] = True                # bottom u_y = 0
        self.free = ~fixed

    def coords(self):
        return np.meshgrid(self.xc, self.yc)


class StressHistory2D:
    """Elastic stress of a pixel structure through a sequence of process
    steps. materials: label -> mechanics.elastic.ElasticMaterial (label 0 is
    void and must not be in it)."""

    def __init__(self, grid, materials, void_ref=None):
        self.g = grid
        self.mats = dict(materials)
        ref = void_ref if void_ref is not None else next(iter(self.mats.values())).C
        self.C_void = VOID_FACTOR * ref
        self.labels = np.zeros((grid.ny, grid.nx), dtype=np.int32)
        self.sigma = {}            # stressor -> (n_elem, 4 Gauss points, 6) GPa
        self.u = {}                # stressor -> (ndof,) nm
        self.history = []          # [(step label, {stressor: sigma field copy})]
        self.log = []

    # -- assembly -----------------------------------------------------------
    def _C_elems(self, labels):
        lab = labels.ravel()
        Ce = np.broadcast_to(self.C_void, (lab.size, 6, 6)).copy()
        for k, m in self.mats.items():
            Ce[lab == k] = m.C
        return Ce

    def _stiffness(self, Ce):
        D = Ce[:, PLANE][:, :, PLANE]
        Ke = np.einsum("apqb,epq->eab", self.g.G, D)
        ed = self.g.edofs
        rows = np.repeat(ed, 8, axis=1).ravel()
        cols = np.tile(ed, (1, 8)).ravel()
        K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(self.g.ndof, self.g.ndof))
        return K

    def _strain(self, u):
        """Engineering strain (n_elem, 4, 6) at the Gauss points (plane
        strain: zz, yz, xz are 0)."""
        ue = u[self.g.edofs]
        e3 = np.einsum("gpa,ea->egp", self.g.Bg, ue)
        e = np.zeros(e3.shape[:2] + (6,))
        e[..., 0], e[..., 1], e[..., 5] = e3[..., 0], e3[..., 1], e3[..., 2]
        return e

    def _force(self, sig_e, mask):
        """Internal force sum over masked elements of int B^T sigma dA, from
        the Gauss-point stresses (n_elem, 4, 6): exactly the force the
        stiffness K produces, so removing material releases exactly what
        it held."""
        f = np.zeros(self.g.ndof)
        if mask.any():
            fe = np.einsum("gpa,egp->ea", self.g.Bg, sig_e[mask][..., PLANE]) * self.g.detJ
            np.add.at(f, self.g.edofs[mask].ravel(), fe.ravel())
        return f

    # -- one process step -----------------------------------------------------
    def step(self, name, labels, stressors=None, relax=None):
        """Advance to the structure `labels` (ny, nx).

        stressors: {stressor name: (mask (ny, nx) bool, eps* (6,) Voigt or
            (n, 6))} - stress-free strains switched on in this step, on
            pixels that are alive after it (normally the ones born now).
        relax: (mask, r) - anneal: those pixels give up a fraction r of
            their stress (all stressors alike)."""
        old = self.labels.ravel()
        new = np.asarray(labels, np.int32).ravel()
        died = (old != 0) & (new != old)
        alive = new != 0
        Ce = self._C_elems(new.reshape(self.labels.shape))
        rhs = {}
        for s, sig in self.sigma.items():          # forces released by removed pixels
            f = self._force(sig, died)
            sig[died] = 0.0
            sig[~alive] = 0.0
            rhs[s] = f
        if relax is not None:
            rmask, r = relax
            rmask = np.asarray(rmask).ravel() & alive & ~died
            for s, sig in self.sigma.items():
                drop = r * sig * rmask[:, None, None]
                rhs[s] = rhs[s] + self._force(drop, rmask)
                sig -= drop
        for s, (mask, eps) in (stressors or {}).items():
            mask = np.asarray(mask).ravel() & alive
            eps = np.broadcast_to(np.asarray(eps, float), (mask.size, 6)) if np.ndim(eps) == 1 \
                else np.asarray(eps, float).reshape(-1, 6)
            s0 = -np.einsum("eij,ej->ei", Ce, eps) * mask[:, None]   # sigma = C(0 - eps*)
            s0 = np.repeat(s0[:, None, :], 4, axis=1)
            self.sigma.setdefault(s, np.zeros((new.size, 4, 6)))
            self.u.setdefault(s, np.zeros(self.g.ndof))
            self.sigma[s] += s0
            rhs[s] = rhs.get(s, np.zeros(self.g.ndof)) - self._force(s0, mask)
        self.labels = new.reshape(self.labels.shape).copy()
        names = [s for s in rhs if np.any(rhs[s])]
        if names:
            K = self._stiffness(Ce)
            fr = self.g.free
            lu = spla.splu(K[fr][:, fr].tocsc())
            F = np.stack([rhs[s][fr] for s in names], axis=1)
            X = lu.solve(F)
            for k, s in enumerate(names):
                du = np.zeros(self.g.ndof)
                du[fr] = X[:, k]
                self.u[s] += du
                ds = np.einsum("eij,egj->egi", Ce, self._strain(du))
                self.sigma[s] += ds * alive[:, None, None]
        # recorded per step: the element (Gauss-point mean) stress
        self.history.append((name, {s: v.mean(axis=1) for s, v in self.sigma.items()}))
        return self

    # -- results --------------------------------------------------------------
    def total(self, sigma=None):
        """Total element stress (n_elem, 6): the current state, or a
        recorded step's {stressor: field}."""
        sigma = self.sigma if sigma is None else sigma
        if not sigma:
            return np.zeros((self.labels.size, 6))
        return sum(v.mean(axis=1) if v.ndim == 3 else v for v in sigma.values())

    def field(self, comp, stressor=None, step=-1):
        """(ny, nx) map of one Voigt component (0..5 or 'xx', ...) at a
        recorded step, total or one stressor."""
        k = VOIGT[comp] if isinstance(comp, str) else comp
        sig = self.history[step][1]
        s = self.total(sig) if stressor is None else sig.get(stressor, np.zeros((self.labels.size, 6)))
        return s[:, k].reshape(self.labels.shape)

    def probe(self, mask, step=-1):
        """Mean stress (6,) in the masked pixels, total and per stressor."""
        m = np.asarray(mask).ravel()
        sig = self.history[step][1]
        per = {s: v[m].mean(axis=0) for s, v in sig.items()}
        return (sum(per.values()) if per else np.zeros(6)), per


VOIGT = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}
