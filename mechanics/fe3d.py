"""3D linear elastic stress through a process flow, on a voxel grid.

The 3D counterpart of mechanics/fe2d.py: one trilinear hex (Q1, 2x2x2
Gauss) per voxel of the fab grid (x along the channel, y depth positive
down, z across the width). Boundaries: the x and z walls are symmetry
planes (rollers: u_x = 0 on the x walls, u_z = 0 on the z walls), the
bottom is held (u_y = 0), the top is free.

Process history, exactly as in 2D: birth (stress-free in the current
shape, plus the stressor's eps*), death (releases the internal force the
removed voxels exerted), anneal (relax a fraction of the stress), one
stress field per stressor (exact superposition).

Differences from 2D, for size (a FinFET cell at 1 nm is ~2e5 voxels,
~6e5 unknowns):
  * void voxels are not in the system at all: only nodes touching a solid
    voxel are unknowns (no soft void stiffness, which would ruin CG);
  * the stiffness is assembled in 3x3 node blocks straight from the
    element tables (one table per material: the grid is uniform);
  * the solve is CG with PETSc's GAMG (rigid-body near-null space from the
    node coordinates) when petsc4py is available, else a direct solve.
"""
import numpy as np
import scipy.sparse as sp

try:
    from petsc4py import PETSc
except ImportError:          # the .venv (tests) has no PETSc: direct solves on small problems
    PETSc = None

CG_RTOL = 1e-9


def _hex_B(h):
    """B (8 Gauss points, 6, 24) of a cube of side h; corners a = 4 di + 2 dj
    + dk; dofs (ux, uy, uz) per corner; Voigt (xx, yy, zz, yz, xz, xy) with
    engineering shear."""
    g = 1.0 / np.sqrt(3.0)
    corners = np.array([(di, dj, dk) for di in (0, 1) for dj in (0, 1) for dk in (0, 1)], float)
    s = 2 * corners - 1
    pts = [(a * g, b * g, c * g) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    Bs = []
    for xi, eta, zeta in pts:
        dx = s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta) / 8 * (2 / h)
        dy = s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zeta) / 8 * (2 / h)
        dz = s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) / 8 * (2 / h)
        B = np.zeros((6, 24))
        B[0, 0::3] = dx
        B[1, 1::3] = dy
        B[2, 2::3] = dz
        B[3, 1::3] = dz
        B[3, 2::3] = dy
        B[4, 0::3] = dz
        B[4, 2::3] = dx
        B[5, 0::3] = dy
        B[5, 1::3] = dx
        Bs.append(B)
    return np.array(Bs), corners.astype(int)


class Grid3D:
    """nx x ny x nz voxels of side h (nm), lower corner lo."""

    def __init__(self, shape, h, lo=(0.0, 0.0, 0.0)):
        self.shape = tuple(int(v) for v in shape)
        self.h = float(h)
        self.lo = np.asarray(lo, float)
        nx, ny, nz = self.shape
        self.nshape = (nx + 1, ny + 1, nz + 1)
        self.n_nodes = int(np.prod(self.nshape))
        self.Bg, corners = _hex_B(self.h)
        self.detJ = (self.h / 2) ** 3
        i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        base = (i * (ny + 1) + j) * (nz + 1) + k
        off = (corners[:, 0] * (ny + 1) + corners[:, 1]) * (nz + 1) + corners[:, 2]
        self.enodes = (base.ravel()[:, None] + off[None, :]).astype(np.int64)     # (n_elem, 8)
        self.G = np.einsum("gpa,gqb->apqb", self.Bg, self.Bg) * self.detJ         # (24, 6, 6, 24)
        ni, nj, nk = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
        self.node_ijk = np.stack([ni.ravel(), nj.ravel(), nk.ravel()], axis=1)

    def centres(self):
        nx, ny, nz = self.shape
        ax = [self.lo[a] + (np.arange(n) + 0.5) * self.h for a, n in enumerate(self.shape)]
        return np.meshgrid(*ax, indexing="ij")


class StressHistory3D:
    """Elastic stress through process steps on a Grid3D. materials: label
    -> ElasticMaterial (label 0 = void, not in the dict)."""

    def __init__(self, grid, materials):
        self.g = grid
        self.mats = dict(materials)
        self.labels = np.zeros(grid.shape, np.int32)
        self.sigma = {}                    # stressor -> (n_elem, 8, 6) GPa (Gauss points)
        self.log = []
        self._Ke = {}
        self.pending = {}                  # stressor -> queued force (n_nodes*3)
        self.n_solves = 0

    def _Ke_table(self, lab_ids):
        for k in lab_ids:
            if k not in self._Ke:
                self._Ke[k] = np.einsum("apqb,pq->ab", self.g.G, self.mats[k].C)
        return self._Ke

    def _strain(self, u, elems):
        ue = u.reshape(-1, 3)[self.g.enodes[elems]].reshape(len(elems), 24)
        return np.einsum("gpa,ea->egp", self.g.Bg, ue)

    def _force(self, sig_el, elems):
        """Global (n_nodes*3) internal force int B^T sigma of elements
        `elems` with Gauss stresses sig_el (len(elems), 8, 6)."""
        f = np.zeros(3 * self.g.n_nodes)
        if len(elems):
            fe = np.einsum("gpa,egp->ea", self.g.Bg, sig_el) * self.g.detJ
            dofs = (3 * self.g.enodes[elems][:, :, None] + np.arange(3)).reshape(len(elems), 24)
            f += np.bincount(dofs.ravel(), weights=fe.ravel(), minlength=f.size)
        return f

    # -- assembly + solve ---------------------------------------------------------
    def _system(self, alive):
        """Block-assembled stiffness on the active nodes. Returns (K csr
        scalar, active node ids, fixed local dofs)."""
        g = self.g
        elems = np.nonzero(alive)[0]
        lab = self.labels.ravel()[elems]
        en = g.enodes[elems]
        act, local = np.unique(en, return_inverse=True)
        local = local.reshape(en.shape)
        nn = len(act)
        keys = (local[:, :, None].astype(np.int64) * nn + local[:, None, :]).ravel()
        ukeys, pos = np.unique(keys, return_inverse=True)
        nb = len(ukeys)
        rows, cols = ukeys // nn, ukeys % nn
        Ke = self._Ke_table(np.unique(lab))
        ids = np.unique(lab)
        Kt = np.zeros((ids.max() + 1, 8, 8, 3, 3))
        for k in ids:
            Kt[k] = Ke[k].reshape(8, 3, 8, 3).transpose(0, 2, 1, 3)
        data = np.empty((nb, 3, 3))
        for c in range(3):
            for d in range(3):
                data[:, c, d] = np.bincount(pos, weights=Kt[:, :, :, c, d][lab].ravel(), minlength=nb)
        order = np.lexsort((cols, rows))
        rows, cols, data = rows[order], cols[order], data[order]
        indptr = np.searchsorted(rows, np.arange(nn + 1))
        K = sp.bsr_matrix((data, cols, indptr), shape=(3 * nn, 3 * nn)).tocsr()
        ijk = g.node_ijk[act]
        nx, ny, nz = g.shape
        fixed = np.zeros((nn, 3), bool)
        fixed[(ijk[:, 0] == 0) | (ijk[:, 0] == nx), 0] = True
        fixed[(ijk[:, 2] == 0) | (ijk[:, 2] == nz), 2] = True
        fixed[ijk[:, 1] == ny, 1] = True
        return K, act, np.nonzero(fixed.ravel())[0], ijk * g.h + g.lo

    def _solve(self, K, rhs_list, fixed, coords):
        """Solve K u = f for each right-hand side (fixed dofs held at 0)."""
        n = K.shape[0]
        if PETSc is None or n < 30000:
            free = np.ones(n, bool)
            free[fixed] = False
            from scipy.sparse.linalg import splu
            lu = splu(K[free][:, free].tocsc())
            out = []
            for f in rhs_list:
                u = np.zeros(n)
                u[free] = lu.solve(f[free])
                out.append(u)
            return out
        K = K.tocsr()
        A = PETSc.Mat().createAIJ(size=K.shape, csr=(K.indptr.astype(PETSc.IntType),
                                                     K.indices.astype(PETSc.IntType), K.data))
        A.setBlockSize(3)
        A.zeroRowsColumns(fixed.astype(PETSc.IntType), diag=1.0)
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType("cg")
        ksp.getPC().setType("gamg")
        ksp.getPC().setCoordinates(coords)
        ksp.setTolerances(rtol=CG_RTOL, max_it=3000)
        ksp.setFromOptions()
        out = []
        for f in rhs_list:
            f = f.copy()
            f[fixed] = 0.0
            b = A.createVecLeft()
            b.setArray(f)
            x = A.createVecRight()
            ksp.solve(b, x)
            self.log.append(("cg", ksp.getIterationNumber(), ksp.getConvergedReason()))
            out.append(x.getArray().copy())
        return out

    # -- one process step ---------------------------------------------------------
    def step(self, name, labels, stressors=None, relax=None, flush=True):
        """Advance to `labels`. Forces (released by removed voxels, switched
        on by stressors, freed by an anneal) are queued; they are solved
        (flush) before any voxel is born into a structure with unsolved
        forces (the newcomer must be stress-free in the CURRENT shape),
        before an anneal, and at the end of the step if flush=True. With
        flush=False a run of removals is solved once, which is exact: a
        purely elastic structure with no births in between ends in the same
        state whatever order its parts are removed in.
        Returns True if a solve ran in this step."""
        old = self.labels.ravel()
        new = np.asarray(labels, np.int32).ravel()
        died = np.nonzero((old != 0) & (new != old))[0]
        born = (new != 0) & (new != old)
        solved = False
        for s, sig in self.sigma.items():                       # removals
            self._add(s, self._force(sig[died], died))
            sig[died] = 0.0
        mid = old.copy()
        mid[died] = 0
        if born.any() and self._pending():                       # settle before births
            self.labels = mid.reshape(self.g.shape)
            solved |= self.flush()
        self.labels = new.reshape(self.g.shape).copy()
        alive = new != 0
        if relax is not None:
            solved |= self.flush()
            rmask, r = relax
            re = np.nonzero(np.asarray(rmask).ravel() & alive & ~born)[0]
            for s, sig in self.sigma.items():
                drop = r * sig[re]
                sig[re] -= drop
                self._add(s, self._force(drop, re))
        for s, (mask, eps) in (stressors or {}).items():
            el = np.nonzero(np.asarray(mask).ravel() & alive)[0]
            if not len(el):
                continue
            eps = np.asarray(eps, float)
            e_el = np.broadcast_to(eps, (len(el), 6)) if eps.ndim == 1 else eps.reshape(-1, 6)[el]
            C = np.stack([self.mats[k].C for k in new[el]])
            s0 = np.repeat(-np.einsum("eij,ej->ei", C, e_el)[:, None, :], 8, axis=1)   # sigma = C(0 - eps*)
            sig = self.sigma.setdefault(s, np.zeros((new.size, 8, 6)))
            sig[el] += s0
            self._add(s, -self._force(s0, el))
        if flush:
            solved |= self.flush()
        return solved

    def _add(self, s, f):
        if np.any(f):
            self.pending[s] = self.pending.get(s, 0.0) + f

    def _pending(self):
        return any(np.any(f) for f in self.pending.values())

    def flush(self):
        """Solve the queued forces on the current structure; True if any."""
        names = [s for s, f in self.pending.items() if np.any(f)]
        if not names:
            self.pending = {}
            return False
        new = self.labels.ravel()
        alive = new != 0
        K, act, fixed, coords = self._system(alive)
        dof = (3 * act[:, None] + np.arange(3)).ravel()
        sols = self._solve(K, [self.pending[s][dof] for s in names], fixed, coords)
        el_alive = np.nonzero(alive)[0]
        Cs = np.stack([self.mats[k].C for k in new[el_alive]])
        for s, du_act in zip(names, sols):
            du = np.zeros(3 * self.g.n_nodes)
            du[dof] = du_act
            self.sigma[s][el_alive] += np.einsum("eij,egj->egi", Cs, self._strain(du, el_alive))
        self.pending = {}
        self.n_solves += 1
        return True

    # -- results ------------------------------------------------------------------
    def element_stress(self, stressor=None):
        """(n_elem, 6) Gauss-mean stress, total or one stressor."""
        if stressor is not None:
            return self.sigma[stressor].mean(axis=1)
        if not self.sigma:
            return np.zeros((self.labels.size, 6))
        return sum(v.mean(axis=1) for v in self.sigma.values())

    def probe(self, mask):
        m = np.asarray(mask).ravel()
        per = {s: v[m].mean(axis=(0, 1)) for s, v in self.sigma.items()}
        return (sum(per.values()) if per else np.zeros(6)), per
