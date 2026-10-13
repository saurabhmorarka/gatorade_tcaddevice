"""Stateful linear solver for a SEQUENCE of Newton systems J*x = b that share
one sparsity pattern (same mesh, same contacts) while their values drift
slowly - the situation every bias sweep is in, and the reason this module
exists (plans/3d_plan.md, "Part 2b").

Why not a plain spsolve per Newton iteration
--------------------------------------------
Measured on the 2D MOSFET (14k unknowns): one SuperLU factorization costs
~56 ms, one solve WITH an existing factorization ~1.3 ms. On a real 3D
Jacobian (84k unknowns) SuperLU needs ~6.7 s per factorization, MUMPS ~0.8-1.1
s, and an ILU(1)-preconditioned GMRES ~0.3 s. So the dominant cost is
(re)factoring, and the cure is to factor rarely:

  * Each system is solved by GMRES preconditioned with the MOST RECENT
    factorization (possibly from an earlier Newton iteration or an earlier
    bias point). Consecutive Newton Jacobians differ little, so GMRES
    converges in a handful of iterations, each costing one triangular
    solve.
  * The solver refactors only when that stops paying off: when GMRES fails,
    or when the time spent in Krylov iterations since the last
    factorization exceeds the time that factorization took (a self-tuning
    rule - no per-device threshold to hand-pick). A reused
    preconditioner gets an iteration budget of a few times what the fresh
    one needed; past it, refactoring is cheaper than grinding on.
  * PETSc refactorizations keep the KSP and redo only the numeric phase
    (the ordering and symbolic fill pattern belong to the sparsity
    pattern, which a sweep never changes).

Each system is solved to a caller-chosen relative residual (`rtol`; default
1e-9 - below that, double precision stalls on these Jacobians: measured
~1e-9 floor on the 2D MOSFET's). Newton passes a loose tolerance far from
convergence and a tight one near it (inexact Newton, Eisenstat & Walker
1996), which keeps Newton's fast local convergence while making most
early solves a few Krylov iterations. Convergence is still judged by the
Newton solver on the TRUE residual/update, so converged results are
unchanged to within the Newton tolerance.

GMRES here is right-preconditioned, so its residual is the true residual of
the (equilibrated) system - not the preconditioned residual, which can look
converged while the true one is not (scipy's own gmres also reports
convergence flags unreliably here, which is why it is not used).

Equilibration
-------------
Ruiz row/column scaling (core/jacobian_scaling.py's rationale applies
unchanged) is done directly on the CSR value array with a cached column
permutation, instead of building diag()@J@diag() products - those sparse
products re-sort indices and were ~10% of a 2D Newton iteration on their
own. The scale vectors are frozen at factorization time, so a reused
preconditioner and the new matrix are always scaled identically.

Backends
--------
  "superlu"     scipy splu + our GMRES. Always available. Uses a METIS
                nested-dissection ordering when pymetis is installed (pip
                wheel; 2.5x faster in 2D, 4x in 3D than SuperLU's COLAMD).
  "umfpack"     UMFPACK with symbolic-analysis reuse (conda-forge
                scikit-umfpack). Opt-in only: its factorization is the
                fastest measured in isolation (18 ms on the 2D MOSFET), but
                inside the reuse+GMRES loop each preconditioner solve costs
                more, and the full curve was slower than METIS-SuperLU
                (4.1 s vs 3.2 s; 6.0 s with unrestricted BLAS threads).
  "scipy_ilu"   scipy spilu (ILUTP) + scipy GMRES. Pure-scipy 3D fallback.
  "petsc_lu"    PETSc MUMPS LU + PETSc GMRES (C), preconditioner reused
                across systems. Needs petsc4py (environment.yml).
  "petsc_ilu"   PETSc ILU(k) + PETSc GMRES. The fastest measured option on
                real 3D drift-diffusion Jacobians. Needs petsc4py.
  "auto"        superlu below AUTO_ITERATIVE_MIN unknowns; above it
                petsc_ilu if petsc4py imports, else scipy_ilu. Override with
                backend= or the TCAD_LINSOLVE environment variable.
"""
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

AUTO_ITERATIVE_MIN = 60000   # unknowns; 2D-sized systems stay on direct LU
REUSE_ITS_FACTOR = 3         # a reused preconditioner may take this x the fresh one's iterations
REUSE_ITS_MIN = 20           # ... but never fewer than this (see LinearSolver._krylov)

# PETSc (conda-forge) is built on MPICH, whose default libfabric provider
# busy-polls from a progress thread as soon as MPI initializes: measured, an
# IDLE process burned ~2 cores (user 1.0 s + sys 3.1 s per 2 s of sleep)
# after `from petsc4py import PETSc`, so every parallel sweep worker was
# silently stealing a second core from the others. All solves here are
# single-process (COMM_SELF), so the provider is irrelevant to them; "tcp"
# does not poll (idle cost ~0.08 s). Must be set before MPI initializes.
os.environ.setdefault("FI_PROVIDER", "tcp")

try:  # optional fast backend (conda env from environment.yml)
    from petsc4py import PETSc  # noqa: F401
    HAVE_PETSC = True
except Exception:  # pragma: no cover - depends on the environment
    HAVE_PETSC = False
try:  # optional (conda-forge scikit-umfpack; no pip wheels)
    import scikits.umfpack as _umf
    HAVE_UMFPACK = True
except Exception:  # pragma: no cover
    HAVE_UMFPACK = False
try:  # pip wheel available; strongly recommended (2.5-4x faster SuperLU)
    import pymetis as _metis
    HAVE_METIS = True
except Exception:  # pragma: no cover
    HAVE_METIS = False

BACKENDS = ("auto", "superlu", "umfpack", "scipy_ilu", "petsc_lu", "petsc_ilu")


def resolve_backend(backend, n):
    backend = backend or os.environ.get("TCAD_LINSOLVE", "auto")
    if backend not in BACKENDS:
        raise ValueError(f"linear solver backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "auto":
        if n < AUTO_ITERATIVE_MIN:
            return "superlu"
        return "petsc_ilu" if HAVE_PETSC else "scipy_ilu"
    if backend == "umfpack" and not HAVE_UMFPACK:
        raise ImportError("backend 'umfpack' needs scikit-umfpack (conda env from environment.yml)")
    if backend.startswith("petsc") and not HAVE_PETSC:
        raise ImportError(f"backend {backend!r} needs petsc4py - create the conda env from "
                          "environment.yml, or use backend='superlu'/'scipy_ilu'")
    return backend


class _Pattern:
    """Cached index structure of one CSR sparsity pattern: row id of every
    stored entry, and the permutation that visits entries column by column
    (so column maxima are a reduceat, not a slow ufunc.at scatter)."""

    def __init__(self, A):
        self.n = A.shape[0]
        self.indptr = A.indptr.copy()
        self.indices = A.indices.copy()
        self.row_of = np.repeat(np.arange(self.n), np.diff(A.indptr))
        self.col_perm = np.argsort(A.indices, kind="stable")
        counts = np.bincount(A.indices, minlength=self.n)
        self.col_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.row_nonempty = np.diff(A.indptr) > 0
        self.col_nonempty = counts > 0

    def nd_permutation(self):
        """(perm, data_perm, csc_indptr, csc_indices) for factoring the
        symmetrically permuted matrix A[perm][:, perm] with SuperLU's
        NATURAL column order: perm is a METIS nested-dissection ordering
        of the graph of A+A^T (computed once per pattern), and data_perm
        maps this pattern's CSR value array straight onto the permuted
        matrix's CSC value array, so each refactorization is one gather
        instead of a sparse fancy-index (which re-sorts).

        Measured: SuperLU's own COLAMD ordering gives 50 ms (2D MOSFET,
        14k unknowns) / 6.7 s (real 3D Jacobian, 84k); METIS-ND + NATURAL
        gives 20 ms / 1.8 s with the same residual (~1e-12)."""
        if getattr(self, "_nd", None) is None:
            G = sp.csr_matrix((np.ones(len(self.indices)), self.indices, self.indptr),
                              shape=(self.n, self.n))
            G = (G + G.T).tocsr()
            G.setdiag(0)
            G.eliminate_zeros()
            if hasattr(_metis, "CSRAdjacency"):     # pymetis >= 2025 API
                perm, _ = _metis.nested_dissection(_metis.CSRAdjacency(G.indptr, G.indices))
            else:
                perm, _ = _metis.nested_dissection(xadj=G.indptr, adjncy=G.indices)
            perm = np.asarray(perm)
            tag = sp.csr_matrix((np.arange(1, len(self.indices) + 1, dtype=float), self.indices,
                                 self.indptr), shape=(self.n, self.n))
            Pm = tag[perm][:, perm].tocsc()
            Pm.sort_indices()
            self._nd = (perm, Pm.data.astype(np.int64) - 1, Pm.indptr.copy(), Pm.indices.copy())
        return self._nd

    def block_map(self, bs):
        """(perm, bsr_indptr, bsr_indices, value_map) to hand this pattern
        to PETSc as a node-major BAIJ matrix with bs x bs blocks: perm maps
        node-major position -> field-major index, and value_map gathers the
        flattened row-major block values from the CSR value array (-1 =
        structural zero inside a block)."""
        if getattr(self, "_blk", None) is None or self._blk[0] != bs:
            nn = self.n // bs
            perm = (np.arange(nn)[:, None] + nn * np.arange(bs)[None, :]).ravel()
            tag = sp.csr_matrix((np.arange(1, len(self.indices) + 1, dtype=float), self.indices,
                                 self.indptr), shape=(self.n, self.n))
            Pm = tag[perm][:, perm].tocsr()
            B = sp.bsr_matrix(Pm, blocksize=(bs, bs))
            B.sort_indices()
            vmap = B.data.reshape(-1).astype(np.int64) - 1
            I = PETSc.IntType if HAVE_PETSC else np.int64
            self._blk = (bs, perm, B.indptr.astype(I), B.indices.astype(I),
                         (len(vmap), np.flatnonzero(vmap >= 0), vmap[vmap >= 0]))
        return self._blk[1:]

    def block_values(self, data, bs):
        """This pattern's CSR values as the flattened BAIJ block values of
        block_map(bs) (structural zeros inside a block = 0)."""
        _, _, _, (size, pos, src) = self.block_map(bs)
        out = np.zeros(size)
        out[pos] = data[src]
        return out

    def matches(self, A):
        return (A.shape[0] == self.n and A.nnz == len(self.indices)
                and np.array_equal(A.indptr, self.indptr) and np.array_equal(A.indices, self.indices))

    def row_max(self, absdata):
        out = np.zeros(self.n)
        nz = self.row_nonempty
        out[nz] = np.maximum.reduceat(absdata, self.indptr[:-1][nz])
        return out

    def col_max(self, absdata):
        out = np.zeros(self.n)
        nz = self.col_nonempty
        out[nz] = np.maximum.reduceat(absdata[self.col_perm], self.col_start[nz])
        return out

    def ruiz(self, data, n_passes=2):
        """(row_scale, col_scale) for Ruiz equilibration of the matrix with
        these CSR values (same result as core/jacobian_scaling.equilibrate)."""
        r_tot = np.ones(self.n)
        c_tot = np.ones(self.n)
        a = np.abs(data)
        for _ in range(n_passes):
            rm = self.row_max(a)
            r = np.where(rm > 0, 1.0 / np.sqrt(rm), 1.0)
            a = a * r[self.row_of]
            r_tot *= r
            cm = self.col_max(a)
            c = np.where(cm > 0, 1.0 / np.sqrt(cm), 1.0)
            a = a * c[self.indices]
            c_tot *= c
        return r_tot, c_tot


class LinearSolver:
    """See the module docstring. One instance per (mesh, sweep); safe to
    reuse across bias points and even across a change of sparsity pattern
    (it detects that and refactors)."""

    def __init__(self, backend=None, rtol=1e-9, max_krylov=40, max_krylov_ilu=300, ilu_levels=None,
                 ilu_fill=10.0, block_size=None):
        """block_size: number of coupled unknowns per mesh node, laid out
        FIELD-major (all of field 0, then all of field 1, ... - the layout
        of every Newton solver here: U = [psi; phin; phip]). Given, the
        petsc_ilu backend reorders node-major and factors point-block ILU
        on bs x bs blocks (PETSc BAIJ) - measured on the real 3D MOSFET
        Jacobian: block ILU(2) setup+solve 0.23 s vs scalar ILU(1) 0.47 s.
        ilu_levels: default 2 with blocks, 1 without."""
        self.block_size = block_size
        self.backend_req = backend
        self.backend = None
        self.rtol = rtol
        self.max_krylov = max_krylov          # with a (lagged) exact LU preconditioner
        self.max_krylov_ilu = max_krylov_ilu  # with an ILU preconditioner (needs many more)
        self.ilu_levels = ilu_levels if ilu_levels is not None else (2 if block_size else 1)
        self.ilu_fill = ilu_fill
        self._pat = None
        self._prec = None          # backend-specific preconditioner handle
        self._ksp_kept = None      # PETSc KSP + preconditioner matrix kept for this pattern
        self._coarse = None        # (n, k) coarse space of the ILU backends, see set_coarse
        self._fresh_its = 0        # Krylov iterations of the solve right after the last factorization
        self._scale = None         # (row_scale, col_scale) frozen at factor time
        self._t_factor = 0.0
        self._krylov_time_since_factor = 0.0
        self.stats = dict(systems=0, factorizations=0, krylov_its=0, fallbacks=0,
                          t_factor=0.0, t_solve=0.0)

    # -- public ---------------------------------------------------------
    def max_rtol(self, n):
        """Loosest relative residual an inexact-Newton caller should ask
        this backend for. With an (lagged) exact LU preconditioner a loose
        residual still gives a good Newton direction. With ILU it is capped
        tighter: these Jacobians are ill-conditioned (quasi-Fermi levels of
        a carrier with ~zero density are nearly undetermined), and an
        ILU-GMRES solution at rtol=1e-4 once threw an already-converged
        equilibrium to |F|=3e5 (extruded 3D MOSFET, before the quasi-Fermi
        box of DEVELOPMENT_LOG 55). Measured since, on the full planar 3D
        I-V (equilibrium, drain ramp, Id-Vg walk; DEVELOPMENT_LOG 56):
        1e-8 / 1e-6 / 1e-4 take 58 / 62 / 68 drain-ramp and 92 / 92 / 95
        walk Newton iterations, same currents, and 1e-6 already gets most
        of the Krylov saving - kept as the cap, with margin for harder
        devices."""
        be = self.backend or resolve_backend(self.backend_req, n)
        return 1e-6 if be in ("petsc_ilu", "scipy_ilu") else 1e-4

    def solve(self, J, b, rtol=None):
        """x with ||J x - b|| <= rtol*||b|| (relative residual of the
        equilibrated system; default self.rtol). An inexact-Newton caller
        passes a loose rtol far from convergence and a tight one close to
        it - see solver2d/newton_solver_qf_2d.py."""
        rtol = self.rtol if rtol is None else max(rtol, self.rtol)
        A = J.tocsr() if not sp.isspmatrix_csr(J) else J
        if not A.has_sorted_indices:
            A = A.copy()
            A.sort_indices()
        if self.backend is None:
            self.backend = resolve_backend(self.backend_req, A.shape[0])
        if self._pat is None or not self._pat.matches(A):
            self._pat = _Pattern(A)
            self._forget_pattern()
        self.stats["systems"] += 1
        t0 = time.perf_counter()
        if self._prec is None:
            x = self._factor_and_solve(A, b, rtol)
        else:
            x, its, ok = self._krylov(A, b, rtol)
            self.stats["krylov_its"] += its
            self._krylov_time_since_factor += time.perf_counter() - t0
            if not ok:
                self.stats["fallbacks"] += 1
                x = self._factor_and_solve(A, b, rtol)
            elif self._krylov_time_since_factor > self._t_factor:
                self._drop_prec()   # refactor at the NEXT system (this x is good)
        self.stats["t_solve"] += time.perf_counter() - t0
        return x

    def set_coarse(self, Z):
        """Coarse space for the ILU backends: an (n, k) array of near-null
        directions of the coming Jacobians (unscaled unknowns), or None.
        ILU is local; a direction along which the matrix is nearly singular
        over a large region (a Schottky contact's S/D island: its majority
        quasi-Fermi level is tied to the metal only through the contact
        conductance, contacts/subgrid.py coarse_space) leaves GMRES to
        find it, ~20 extra iterations on the planar 3D device. With Z the
        preconditioner becomes two-level (multiplicative): a Galerkin
        correction on span(Z), then ILU on the remaining residual
        (DEVELOPMENT_LOG 65). With a direct backend it applies to the
        reused (lagged) factorization."""
        same = (Z is None and self._coarse is None) or (
            Z is not None and self._coarse is not None and Z.shape == self._coarse.shape
            and np.array_equal(Z, self._coarse))
        if not same:
            self._coarse = None if Z is None else np.array(Z, float, copy=True)
            self._forget_pattern()     # the PETSc KSP changes preconditioner type

    def _coarse_scaled(self, As):
        """(Zs, AZs): the coarse space in the scaled unknowns (x = c * y)
        with unit columns, and the scaled matrix applied to it."""
        Zs = self._coarse / self._scale[1][:, None]
        Zs = Zs / np.linalg.norm(Zs, axis=0)
        return Zs, As @ Zs

    def summary(self):
        s = self.stats
        return (f"{self.backend}: {s['systems']} systems, {s['factorizations']} factorizations, "
                f"{s['krylov_its']} Krylov its, {s['fallbacks']} fallbacks, "
                f"factor {s['t_factor']:.2f}s / total {s['t_solve']:.2f}s")

    # -- internals ------------------------------------------------------
    def _drop_prec(self):
        """Mark the preconditioner stale: the next system refactors (PETSc
        keeps its KSP for a numeric-only refactorization; _forget_pattern
        releases it)."""
        self._prec = None
        self._krylov_time_since_factor = 0.0

    def _forget_pattern(self):
        self._drop_prec()
        if self._ksp_kept is not None:
            self._ksp_kept["ksp"].destroy()
            self._ksp_kept = None

    def _scaled(self, A, b, scale):
        r, c = scale
        data = A.data * r[self._pat.row_of] * c[self._pat.indices]
        As = sp.csr_matrix((data, self._pat.indices, self._pat.indptr), shape=A.shape)
        return As, r * b

    def _factor_and_solve(self, A, b, rtol):
        self._drop_prec()
        t0 = time.perf_counter()
        self._scale = self._pat.ruiz(A.data)
        As, bs = self._scaled(A, b, self._scale)
        if self.backend in ("superlu", "umfpack"):
            lu = self._superlu(As) if self.backend == "superlu" else self._umfpack(As)
            y = lu.solve(bs)
            # exact now; reused (lagged) as GMRES preconditioner it is two-level
            self._prec = dict(lu=lu if self._coarse is None else _TwoLevel(lu.solve, *self._coarse_scaled(As)))
        elif self.backend == "scipy_ilu":
            ilu = spla.spilu(As.tocsc(), fill_factor=self.ilu_fill, drop_tol=1e-6)
            if self._coarse is not None:
                ilu = _TwoLevel(ilu.solve, *self._coarse_scaled(As))
            self._prec = dict(lu=ilu)
            y, ok, its = gmres_right(As, bs, ilu.solve, rtol, self.max_krylov_ilu)
            if not ok:
                raise RuntimeError("scipy ILU-GMRES failed on a freshly built preconditioner")
        else:
            self._prec = self._petsc_setup(As)
            y, its, ok = self._petsc_solve(As, bs, rtol, reuse=False)
            if not ok:
                raise RuntimeError(f"PETSc {self.backend} failed on a freshly built preconditioner")
        self._fresh_its = its if self.backend in ("scipy_ilu", "petsc_ilu") else 0
        self._t_factor = time.perf_counter() - t0
        self.stats["factorizations"] += 1
        self.stats["t_factor"] += self._t_factor
        return self._scale[1] * y

    def _krylov(self, A, b, rtol):
        """Solve with the reused (stale) preconditioner; an ILU one on an
        iteration budget of REUSE_ITS_FACTOR x what the fresh one needed (at least
        REUSE_ITS_MIN): past that a refactorization is cheaper than grinding
        on - measured on the full planar 3D device, a reused ILU that ran
        into the 300-iteration limit cost ~1.5 s before the fallback
        refactored anyway, against 0.08 s for the refactorization."""
        As, bs = self._scaled(A, b, self._scale)
        maxit = self.max_krylov if self.backend in ("superlu", "umfpack", "petsc_lu") else self.max_krylov_ilu
        if self.backend in ("scipy_ilu", "petsc_ilu"):   # (measured there; direct-LU reuse untouched)
            maxit = min(maxit, max(REUSE_ITS_MIN, REUSE_ITS_FACTOR * self._fresh_its))
        if self.backend in ("superlu", "umfpack", "scipy_ilu"):
            y, ok, its = gmres_right(As, bs, self._prec["lu"].solve, rtol, maxit)
        else:
            y, its, ok = self._petsc_solve(As, bs, rtol, reuse=True, maxit=maxit)
        return self._scale[1] * y, its, ok

    def _superlu(self, As):
        if not HAVE_METIS:
            return spla.splu(As.tocsc())
        perm, data_perm, ip, ix = self._pat.nd_permutation()
        Ap = sp.csc_matrix((As.data[data_perm], ix, ip), shape=As.shape)
        lu = spla.splu(Ap, permc_spec="NATURAL", diag_pivot_thresh=0.1)

        class _PermutedLU:
            def solve(self_, rhs):
                out = np.empty_like(rhs)
                out[perm] = lu.solve(rhs[perm])
                return out
        return _PermutedLU()

    def _umfpack(self, As):
        """UMFPACK numeric factorization, reusing the symbolic analysis
        (ordering) for as long as the pattern is unchanged. Measured 18 ms
        vs SuperLU-COLAMD's 50 ms on the 2D MOSFET Jacobian."""
        Ac = As.tocsc()
        Ac.sort_indices()
        pat = self._pat
        if getattr(pat, "_umf_ctx", None) is None:
            pat._umf_ctx = _umf.UmfpackContext("di")
            pat._umf_ctx.symbolic(Ac)
        ctx = pat._umf_ctx
        ctx.numeric(Ac)

        class _UmfLU:
            def solve(self_, rhs):
                return ctx.solve(_umf.UMFPACK_A, Ac, rhs, autoTranspose=True)
        return _UmfLU()

    def _use_blocks(self):
        bs = self.block_size
        return bool(bs) and bs > 1 and self.backend == "petsc_ilu" and self._pat.n % bs == 0

    def _petsc_csr(self, As):
        if self._use_blocks():
            _, bip, bix, _ = self._pat.block_map(self.block_size)
            return bip, bix, self._pat.block_values(As.data, self.block_size)
        I = PETSc.IntType
        if getattr(self._pat, "_csr_I", None) is None:
            self._pat._csr_I = (self._pat.indptr.astype(I), self._pat.indices.astype(I))
        return (*self._pat._csr_I, As.data)

    def _petsc_mat(self, As):
        csr = self._petsc_csr(As)
        if self._use_blocks():
            return PETSc.Mat().createBAIJ(size=As.shape, bsize=self.block_size, csr=csr, comm=PETSc.COMM_SELF)
        return PETSc.Mat().createAIJ(size=As.shape, csr=csr, comm=PETSc.COMM_SELF)

    def _petsc_setup(self, As):
        """The KSP with a preconditioner built from As. While the sparsity
        pattern is unchanged the KSP and its preconditioner matrix are kept
        and only refilled: PETSc then redoes just the NUMERIC factorization,
        reusing the ordering and symbolic (fill-pattern) phase - measured on
        the full planar 3D Jacobian, block ILU(2) 81 ms instead of 176 ms."""
        kept = self._ksp_kept
        if kept is not None:
            kept["P"].setValuesBlockedCSR(*self._petsc_csr(As)) if self._use_blocks() \
                else kept["P"].setValuesCSR(*self._petsc_csr(As))
            kept["P"].assemble()
            kept["ksp"].setOperators(kept["P"], kept["P"])
            if kept["two_level"] is not None:
                self._petsc_two_level(kept, As)
            kept["ksp"].setUp()
            return kept
        M = self._petsc_mat(As)
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(M, M)
        maxit = self.max_krylov if self.backend == "petsc_lu" else self.max_krylov_ilu
        ksp.setType("gmres")
        ksp.setGMRESRestart(maxit)
        ksp.setPCSide(PETSc.PC.Side.RIGHT)          # residual = true (unpreconditioned) residual
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setTolerances(rtol=self.rtol, atol=1e-50, max_it=maxit)
        pc = ksp.getPC()
        if self.backend == "petsc_lu":
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
        elif self._coarse is None:
            pc.setType("ilu")
            pc.setFactorLevels(self.ilu_levels)
            pc.setFactorOrdering("rcm")
        else:
            ilu = PETSc.PC().create(comm=PETSc.COMM_SELF)
            ilu.setType("ilu")
            ilu.setFactorLevels(self.ilu_levels)
            ilu.setFactorOrdering("rcm")
            pc.setType("python")
            pc.setPythonContext(_PetscTwoLevel(ilu, As.shape[0]))
        self._ksp_kept = dict(ksp=ksp, P=M, two_level=pc.getPythonContext() if self._coarse is not None
                              and self.backend == "petsc_ilu" else None)
        if self._ksp_kept["two_level"] is not None:
            self._petsc_two_level(self._ksp_kept, As)
        ksp.setUp()
        return self._ksp_kept

    def _petsc_two_level(self, kept, As):
        """(Re)factor the ILU inside the two-level preconditioner and refresh
        its coarse space for this factorization's matrix and scaling."""
        ctx = kept["two_level"]
        ctx.ilu.setOperators(kept["P"], kept["P"])
        ctx.ilu.setUp()
        Zs, AZs = self._coarse_scaled(As)
        perm = self._pat.block_map(self.block_size)[0] if self._use_blocks() else None
        ctx.set(Zs if perm is None else Zs[perm], AZs if perm is None else AZs[perm])

    def _petsc_solve(self, As, bs, rtol, reuse, maxit=None):
        ksp = self._prec["ksp"]
        if maxit is None:
            maxit = self.max_krylov if self.backend == "petsc_lu" else self.max_krylov_ilu
        ksp.setTolerances(rtol=rtol, max_it=maxit)
        if reuse:
            A_new = self._petsc_mat(As)
            ksp.setOperators(A_new, self._prec["P"])   # operator = new J, preconditioner = old
        perm = self._pat.block_map(self.block_size)[0] if self._use_blocks() else None
        rhs = bs[perm] if perm is not None else bs.copy()
        bv = PETSc.Vec().createWithArray(rhs, comm=PETSc.COMM_SELF)
        xv = bv.duplicate()
        ksp.solve(bv, xv)
        y = xv.getArray().copy()
        if perm is not None:
            out = np.empty_like(y)
            out[perm] = y
            y = out
        ok = ksp.getConvergedReason() > 0 and np.all(np.isfinite(y))
        return y, ksp.getIterationNumber(), ok


class _TwoLevel:
    """Multiplicative two-level preconditioner: Galerkin correction on
    span(Z), then the fine preconditioner on what is left -
    y = Z E^-1 Z^T v,  y += M^-1 (v - A y),  E = Z^T A Z. A Z is formed
    once per factorization, so an application costs one fine solve and
    two thin (n x k) products, no extra matrix-vector product."""

    def __init__(self, fine, Z, AZ):
        self.fine = fine
        self.set(Z, AZ)

    def set(self, Z, AZ):
        self.Z, self.AZ = Z, AZ
        self.Einv = np.linalg.pinv(Z.T @ AZ)   # pinv: a degenerate space must not break a solve

    def solve(self, v):
        w = self.Einv @ (self.Z.T @ v)
        return self.Z @ w + self.fine(v - self.AZ @ w)


class _PetscTwoLevel:
    """_TwoLevel as a PETSc python PC around a PETSc ILU PC, working in
    place on PETSc vectors: Z and A Z live on the coarse space's support
    only (the S/D islands), so beyond the ILU apply an application is one
    vector copy (in C) and a few support-sized updates."""

    def __init__(self, ilu, n):
        self.ilu = ilu
        self._r = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)

    def set(self, Z, AZ):
        self.zi = np.flatnonzero(np.any(Z != 0.0, axis=1))
        self.ai = np.flatnonzero(np.any(AZ != 0.0, axis=1))
        self.Zt = np.ascontiguousarray(Z[self.zi].T)
        self.AZ = np.ascontiguousarray(AZ[self.ai])
        self.Einv = np.linalg.pinv(Z.T @ AZ)   # pinv: a degenerate space must not break a solve

    def apply(self, pc, x, y):
        w = self.Einv @ (self.Zt @ x.getArray(readonly=True)[self.zi])
        x.copy(self._r)
        with self._r as r:
            r[self.ai] -= self.AZ @ w
        self.ilu.apply(self._r, y)
        with y as ya:
            ya[self.zi] += w @ self.Zt


def gmres_right(A, b, prec_solve, rtol, maxit):
    """Right-preconditioned GMRES (no restart), classical Gram-Schmidt with
    re-orthogonalization (two BLAS-2 passes - faster than modified GS in
    numpy and as stable). Returns (x, ok, iterations); ok means the TRUE
    residual ||b - A x|| <= rtol*||b|| (re-checked explicitly at the end)."""
    n = len(b)
    beta = np.linalg.norm(b)
    if beta == 0.0:
        return np.zeros(n), True, 0
    V = np.empty((maxit + 1, n))
    Z = np.empty((maxit, n))
    H = np.zeros((maxit + 1, maxit))
    cs = np.zeros(maxit)
    sn = np.zeros(maxit)
    g = np.zeros(maxit + 1)
    g[0] = beta
    V[0] = b / beta
    k_done = 0
    for k in range(maxit):
        Z[k] = prec_solve(V[k])
        w = A @ Z[k]
        Vk = V[:k + 1]
        h = Vk @ w
        w = w - h @ Vk
        h2 = Vk @ w
        w = w - h2 @ Vk
        h = h + h2
        hn = np.linalg.norm(w)
        col = np.zeros(k + 2)
        col[:k + 1] = h
        col[k + 1] = hn
        for i in range(k):  # apply previous Givens rotations
            t = cs[i] * col[i] + sn[i] * col[i + 1]
            col[i + 1] = -sn[i] * col[i] + cs[i] * col[i + 1]
            col[i] = t
        d = np.hypot(col[k], col[k + 1])
        cs[k], sn[k] = (1.0, 0.0) if d == 0.0 else (col[k] / d, col[k + 1] / d)
        col[k] = d
        col[k + 1] = 0.0
        H[:k + 2, k] = col
        g[k + 1] = -sn[k] * g[k]
        g[k] = cs[k] * g[k]
        k_done = k + 1
        if not np.isfinite(g[k + 1]):
            return np.zeros(n), False, k_done
        if abs(g[k + 1]) <= rtol * beta or hn == 0.0:
            break
        V[k + 1] = w / hn
    y = np.linalg.solve(np.triu(H[:k_done, :k_done]), g[:k_done]) if k_done else np.zeros(0)
    x = y @ Z[:k_done]
    ok = bool(np.all(np.isfinite(x))) and np.linalg.norm(b - A @ x) <= 2.0 * rtol * beta
    return x, ok, k_done
