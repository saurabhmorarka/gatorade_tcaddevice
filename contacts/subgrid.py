"""Sub-grid Schottky contact for 2D/3D device meshes (plans/schottky_plan.md
section 8).

At S/D doping the barrier is 0.5-1 nm thick; a device mesh cannot resolve
it (the explicitly resolved model is off 1.8x at 1 nm spacing, 15x at
2 nm). So the barrier is sub-grid:

  1. Table. The validated 1D model (contacts/newton_schottky_1d.py: TE +
     nonlocal tunneling, resolved on a 0.05 nm mesh) is solved once per
     (material, contact parameters, majority carrier) and tabulated as the
     current density from the metal into the semiconductor

         J(N, u),  u = V_metal - phi_maj(x_ref),

     phi_maj the majority quasi-Fermi potential just past the depletion
     region (x_ref). The barrier, its tunneling and its band bending are
     inside J; the neutral bulk beyond x_ref is left to the device mesh.
     Cached on disk (a hash of every parameter).

  2. Boundary condition. The semiconductor nodes touching the metal become
     ordinary free nodes (Poisson keeps them neutral: the barrier dipole
     is sub-grid). Their majority carrier gets a sink/source
     J(N_i, V - phi_i) A_i / (q cv_i), the minority carrier a thermionic
     Robin flux v (c - c(phi = V)) A_i / cv_i, A_i the node's metal
     interface area. Both enter newton_solve_2d through its `generation`
     hook (nodal, analytic Jacobian).

Valid wherever the interface's curvature radius is much larger than the
barrier width (1-4 nm): planar, FinFET and nanosheet contacts alike.
"""
import hashlib
import json
import os
import warnings
from dataclasses import asdict

import numpy as np
import scipy.sparse as sp

from core.params import Q
from contacts.schottky import SchottkyContact, te_velocity
from contacts import newton_schottky_1d as ns

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", ".cache",
                         "schottky_tables")
TABLE_VERSION = 3
LOG10_N_STEP = 0.125
U_GRID = np.round(np.arange(-0.6, 0.6001, 0.01), 4)
# metal biases of the tabulating 1D sweeps: 25 mV steps where a contact
# works (0.1 V gaps missed the bend at flat band: 41% off at -0.25 V, PMOS)
V_SWEEP = tuple(np.round(np.r_[0.01, np.arange(0.025, 0.401, 0.025), 0.5, 0.6, 0.8, 1.0], 4).tolist())
# cm^-3: a contact's doped island for coarse_space (1e18 and 1e19 measured
# the same on the planar device; 1e20 cuts the island to the contact implant)
COARSE_DOPING = 1e18


def _material_key(mat):
    return {k: float(getattr(mat, k)) for k in ("T", "eps_r", "ni", "mu_n", "mu_p", "tau_n", "tau_p", "chi_eV",
                                                "Eg_eV", "Nc", "Nv")}


class SchottkyTable:
    """J(N, u) for one contact on one semiconductor, majority carrier
    `carrier` ("n" or "p"). Tabulated over log10|N| on a fixed step and
    u on U_GRID as g = asinh(J / J_s(N)), J_s = G0 Vt with G0 the zero-bias
    conductance (1 / rho_c): g is then linear through u = 0 and
    logarithmic where J is exponential, so linear interpolation is
    accurate. (A fixed scale failed: at 7e20 J is ~1e8 x the scale per 10
    mV, and asinh-linear interpolation gave J(0.01 V) 1300x too small.)
    Bilinear in (log10 N, u); log J_s linear in log10 N; linear
    extrapolation in u beyond the grid."""

    def __init__(self, mat, contact: SchottkyContact, carrier, log10_N, g, log_Js):
        self.mat, self.contact, self.carrier = mat, contact, carrier
        self.log10_N = np.asarray(log10_N, float)
        self.g = np.asarray(g, float)              # (len(log10_N), len(U_GRID))
        self.log_Js = np.asarray(log_Js, float)    # (len(log10_N),)
        self.u = U_GRID

    @classmethod
    def build(cls, mat, contact, carrier, N_min, N_max, cache=True, log=None):
        lo = np.floor(np.log10(N_min) / LOG10_N_STEP) * LOG10_N_STEP
        hi = np.ceil(np.log10(N_max) / LOG10_N_STEP) * LOG10_N_STEP
        log10_N = np.round(np.arange(lo, hi + LOG10_N_STEP / 2, LOG10_N_STEP), 4)
        key = json.dumps(dict(v=TABLE_VERSION, mat=_material_key(mat), contact=asdict(contact), carrier=carrier,
                              log10_N=log10_N.tolist(), u=U_GRID.tolist(), V=V_SWEEP,
                              path=ns.PATH_LENGTH, dE=ns.DE_PER_VT), sort_keys=True)
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, f"{h}.npz")
        if cache and os.path.exists(path):
            z = np.load(path)
            return cls(mat, contact, carrier, z["log10_N"], z["g"], z["log_Js"])
        if log:
            log(f"Schottky table: {len(log10_N)} dopings 1e{log10_N[0]:.2f}-1e{log10_N[-1]:.2f}, carrier {carrier}")
        rows = [_tabulate_one(mat, contact, carrier, 10.0 ** l) for l in log10_N]
        g = np.array([r[0] for r in rows])
        log_Js = np.log(np.array([r[1] for r in rows]))
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(path, log10_N=log10_N, g=g, log_Js=log_Js)
        return cls(mat, contact, carrier, log10_N, g, log_Js)

    def __call__(self, N_abs, u):
        """(J, dJ/du) at doping magnitudes N_abs and drops u (arrays)."""
        l = np.clip(np.log10(np.maximum(N_abs, 1.0)), self.log10_N[0], self.log10_N[-1])
        if len(self.log10_N) > 1:
            fi = (l - self.log10_N[0]) / LOG10_N_STEP
            i0 = np.clip(np.floor(fi).astype(int), 0, len(self.log10_N) - 2)
            w = fi - i0
        else:
            i0, w = np.zeros(len(np.atleast_1d(l)), int), np.zeros(len(np.atleast_1d(l)))
        du = U_GRID[1] - U_GRID[0]
        fu = (np.asarray(u, float) - U_GRID[0]) / du
        k0 = np.clip(np.floor(fu).astype(int), 0, len(U_GRID) - 2)
        t = fu - k0                                  # may be < 0 or > 1: linear extrapolation

        def at(i):
            gi = self.g[i]
            a, b = gi[np.arange(len(k0)), k0], gi[np.arange(len(k0)), k0 + 1]
            return a + t * (b - a), (b - a) / du

        ga, sa = at(i0)
        if len(self.log10_N) > 1:
            gb, sb = at(i0 + 1)
            g = (1 - w) * ga + w * gb
            s = (1 - w) * sa + w * sb
            Js = np.exp((1 - w) * self.log_Js[i0] + w * self.log_Js[i0 + 1])
        else:
            g, s, Js = ga, sa, np.exp(self.log_Js[i0])
        return Js * np.sinh(g), Js * np.cosh(g) * s


def _tabulate_one(mat, contact, carrier, N):
    """g(u) on U_GRID for one doping: 1D sweeps of the resolved model on a
    slab metal | N | ohmic, read at x_ref past the depletion region."""
    sign = 1.0 if carrier == "n" else -1.0
    phi_b = contact.barriers(mat.chi_eV, mat.Eg_eV)[0 if carrier == "n" else 1]
    W = np.sqrt(2 * mat.eps * max(phi_b, 0.05) / (Q * N))       # depletion width estimate
    x_ref = 1.5 * W + 3e-7
    L = max(40e-7, 4 * x_ref)
    x = ns.contact_grid(L, h0=0.05e-7, h_max=max(0.5e-7, L / 300))
    Cdop = np.full(len(x), sign * N)
    ends = {"left": contact, "right": None}
    i_ref = int(np.searchsorted(x, x_ref))
    us, Js = [0.0], [0.0]
    for direction in (1.0, -1.0):
        targets = [direction * v for v in V_SWEEP]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs = ns.sweep(x, Cdop, mat, ends, targets, dV_max=0.05)
        for V, r in zip(targets, rs):
            if r is None:
                continue
            phi = r["phin"] if carrier == "n" else r["phip"]
            us.append(V - phi[i_ref])
            Js.append(r["J_left"])
    us, Js = np.array(us), np.array(Js)
    o = np.argsort(us)
    us, Js = us[o], Js[o]
    # zero-bias conductance from the two samples nearest u = 0
    k0 = int(np.flatnonzero(us == 0.0)[0])
    G0 = 0.5 * (Js[k0 + 1] / us[k0 + 1] + Js[k0 - 1] / us[k0 - 1])
    J_s = abs(G0) * mat.Vt
    g = np.arcsinh(Js / J_s)
    # Monotone in u by physics; linear in g beyond the sampled range.
    gi = np.interp(U_GRID, us, g)
    for side in (U_GRID < us[0], U_GRID > us[-1]):
        if side.any():
            a, b = (0, 1) if side[0] else (-2, -1)
            slope = (g[b] - g[a]) / (us[b] - us[a])
            u0, g0 = (us[0], g[0]) if side[0] else (us[-1], g[-1])
            gi[side] = g0 + slope * (U_GRID[side] - u0)
    return gi, J_s


def contact_material(mesh, mat, nodes):
    """The semiconductor Material a contact touches: the per-node material
    (mesh.node_materials / node_material_index) most of its nodes carry,
    else the base `mat`. A p+ SiGe S/D contact must be tabulated with
    SiGe's gap and band densities: with the base silicon its hole barrier
    read Eg(Si) - Phi_Bn = 0.67 eV instead of Eg(SiGe) - Phi_Bn = 0.30 eV
    (DEVELOPMENT_LOG 82). A single-material mesh is unchanged."""
    mats = getattr(mesh, "node_materials", None)
    idx = getattr(mesh, "node_material_index", None)
    if mats is None or idx is None:
        return mat
    k = np.bincount(np.asarray(idx)[nodes]).argmax()
    m = mats[k] if k > 0 else mat
    if k > 0 and _material_key(m) == _material_key(mat):
        return mat                            # the same semiconductor as the base: keep the base object
    return m


class SubgridContacts:
    """The sub-grid Schottky boundary condition for a set of contacts on a
    2D/3D mesh. contacts: {name: (SchottkyTable, metal-side info)} built by
    `setup`. Use `.generation(bias)` as newton_solve_2d's generation hook
    and `.terminal_current(result, name, bias)` for the contact current."""

    def __init__(self, mesh, mat, specs, area):
        self.mesh, self.mat = mesh, mat
        self.specs = specs                    # name -> dict(table, nodes, carrier)
        self.area = area                      # (N,) metal interface area per node
        cv = mesh.cv_area_semi if getattr(mesh, "cv_area_semi", None) is not None else mesh.cv_area
        self.cv = np.asarray(cv, float)
        self.N = len(mesh.points)
        self.nodes = np.concatenate([s["nodes"] for s in specs.values()]) if specs else np.zeros(0, int)

    @classmethod
    def setup(cls, mesh, mat, contacts, area, log=None, cache=True):
        """contacts: {name: SchottkyContact}; area: (N,) interface areas
        (see voxel_interface_areas / boundary_interface_areas)."""
        bc = np.array(mesh.boundary_bc_type)
        is_ins = mesh.is_insulator if mesh.is_insulator is not None else np.zeros(len(mesh.points), bool)
        specs = {}
        for name, c in contacts.items():
            pts = mesh.boundary_point_index[bc == f"contact:{name}"]
            semi = pts[~is_ins[pts] & (area[pts] > 0)]
            if not len(semi):
                raise ValueError(f"Schottky contact {name!r} touches no semiconductor node with an interface area")
            Nd = mesh.Cdop[semi]
            carrier = "n" if np.median(Nd) >= 0 else "p"
            Nabs = np.abs(Nd)
            cmat = contact_material(mesh, mat, semi)
            table = SchottkyTable.build(cmat, c, carrier, max(Nabs.min(), 1e15), Nabs.max(), cache=cache, log=log)
            specs[name] = dict(table=table, nodes=semi, carrier=carrier, contact=c, mat=cmat)
        return cls(mesh, mat, specs, area)

    def metal_psi0(self, name):
        """The metal's electrostatic potential at zero bias in the solver's
        psi reference (n = ni exp((psi - phin + dEi)/Vt)): the interface
        value of the resolved 1D model, -Phi_Bn + (Ec - Ei) - dEi, median
        over the contact's semiconductor nodes."""
        s = self.specs[name]
        i = s["nodes"]
        mat = self.mat
        ec_off = getattr(self.mesh, "ec_off_arr", None)
        ec_off = np.full(self.N, mat.Vt * np.log(mat.Nc / mat.ni)) if ec_off is None else ec_off
        dEi = getattr(self.mesh, "dEi_arr", None)
        dEi = np.zeros(self.N) if dEi is None else dEi
        phi_bn = s["contact"].barriers(mat.chi_eV, mat.Eg_eV)[0]
        return float(np.median(ec_off[i] - dEi[i])) - phi_bn

    def released(self):
        """Mesh node indices the solver must NOT treat as Dirichlet."""
        return self.nodes

    def coarse_space_cached(self):
        if not hasattr(self, "_coarse"):
            self._coarse = self.coarse_space()
        return self._coarse

    def coarse_space(self):
        """Near-null directions of the Newton Jacobian that a contact-sink
        boundary creates, for the linear solver's coarse correction
        (core/linsolve.py, LinearSolver.set_coarse): one per contact, its
        majority quasi-Fermi level shifted uniformly over the contact's
        doped island (majority-type nodes with |Cdop| > COARSE_DOPING
        connected to the contact through the mesh). An ohmic contact pins
        that level; this one ties it to the metal only through the contact
        conductance, far below the island's own, so the shift costs almost
        nothing and ILU cannot see it (DEVELOPMENT_LOG 65). (3N, k) array
        in U = (psi, phin, phip) order, or None."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        N, mesh = self.N, self.mesh
        ins = mesh.is_insulator if mesh.is_insulator is not None else np.zeros(N, bool)
        ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
        cols = []
        for s in self.specs.values():
            sign = 1.0 if s["carrier"] == "n" else -1.0
            ok = ~ins & (sign * np.asarray(mesh.Cdop) > COARSE_DOPING)
            seeds = s["nodes"][ok[s["nodes"]]]
            if not len(seeds):
                continue
            e = ok[ii] & ok[jj]
            G = coo_matrix((np.ones(int(e.sum())), (ii[e], jj[e])), shape=(N, N))
            _, lab = connected_components(G, directed=False)
            island = ok & np.isin(lab, np.unique(lab[seeds]))
            z = np.zeros(3 * N)
            z[(1 if s["carrier"] == "n" else 2) * N + np.flatnonzero(island)] = 1.0
            if not any(np.array_equal(z, c) for c in cols):   # contacts sharing one island
                cols.append(z)
        return np.array(cols).T if cols else None

    def _rates(self, psi, phin, phip, bias, jacobian):
        """Per-node particle rates per volume (cm^-3 s^-1, > 0 = generation)
        and, if jacobian, the sparse derivatives w.r.t. U = (psi, phin, phip).
        Also returns the per-node conventional current from the metal into
        the semiconductor (A per node) per contact."""
        N, mat = self.N, self.mat
        Vt = mat.Vt
        dEi = getattr(self.mesh, "dEi_arr", None)
        dEi = np.zeros(N) if dEi is None else dEi
        ni = self.mesh.ni_arr
        Gn, Gp = np.zeros(N), np.zeros(N)
        rows_n, cols_n, vals_n, rows_p, cols_p, vals_p = [], [], [], [], [], []
        I_node = {}
        for name, s in self.specs.items():
            i = s["nodes"]
            V = bias.get(name, 0.0)
            A, cv = self.area[i], self.cv[i]
            k = A / cv
            n = ni[i] * np.exp((psi[i] - phin[i] + dEi[i]) / Vt)
            p = ni[i] * np.exp((phip[i] - psi[i] - dEi[i]) / Vt)
            c = s["contact"]
            cm = s.get("mat", mat)
            vn = te_velocity(c.richardson_n, cm.T, cm.Nc)
            vp = te_velocity(c.richardson_p, cm.T, cm.Nv)
            if s["carrier"] == "n":
                J, dJ = s["table"](np.abs(self.mesh.Cdop[i]), V - phin[i])
                # J > 0: conventional current metal -> semiconductor = electrons leaving.
                Gn[i] += -J / Q * k
                # minority holes: flux into the metal vp (p - p(phip = V))
                pV = ni[i] * np.exp((V - psi[i] - dEi[i]) / Vt)
                Gp[i] += -vp * (p - pV) * k
                I = (J - Q * vp * (p - pV)) * A
                if jacobian:
                    rows_n += [i]; cols_n += [N + i]; vals_n += [dJ / Q * k]          # du/dphin = -1
                    rows_p += [i, i]; cols_p += [i, 2 * N + i]
                    vals_p += [-vp * k * (-p + pV) / Vt, -vp * k * p / Vt]
            else:
                J, dJ = s["table"](np.abs(self.mesh.Cdop[i]), V - phip[i])
                # J > 0: holes flow metal -> semiconductor.
                Gp[i] += J / Q * k
                nV = ni[i] * np.exp((psi[i] - V + dEi[i]) / Vt)
                Gn[i] += -vn * (n - nV) * k
                I = (J + Q * vn * (n - nV)) * A
                if jacobian:
                    rows_p += [i]; cols_p += [2 * N + i]; vals_p += [-dJ / Q * k]
                    rows_n += [i, i]; cols_n += [i, N + i]
                    vals_n += [-vn * k * (n - nV) / Vt, -vn * k * (-n / Vt)]
            I_node[name] = I
        if not jacobian:
            return Gn, Gp, None, None, I_node

        def mat_(r, c_, v):
            if not r:
                return sp.csr_matrix((N, 3 * N))
            return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c_))), shape=(N, 3 * N))
        return Gn, Gp, mat_(rows_n, cols_n, vals_n), mat_(rows_p, cols_p, vals_p), I_node

    def generation(self, bias, extra=None):
        """newton_solve_2d's generation hook for the given contact biases
        ({name: V}), optionally summed with another hook `extra`."""
        def g(psi, phin, phip, jacobian=True):
            Gn, Gp, dGn, dGp, _ = self._rates(psi, phin, phip, bias, jacobian)
            if extra is not None:
                a = extra(psi, phin, phip, jacobian=jacobian)
                Gn, Gp = Gn + a[0], Gp + a[1]
                if jacobian:
                    dGn = dGn + a[2] if a[2] is not None else dGn
                    dGp = dGp + a[3] if a[3] is not None else dGp
            return Gn, Gp, dGn, dGp
        # relaxes toward the metal Fermi level: inside the box (and so does an
        # extra hook that says so, e.g. an SRH-form tunneling rate)
        g.keeps_qf_box = extra is None or bool(getattr(extra, "keeps_qf_box", False))
        return g

    def terminal_current(self, result, name, bias):
        """Current (A, or A/cm in 2D) from the contact into the device."""
        *_, I_node = self._rates(result["psi"], result["phin"], result["phip"], bias, False)
        return float(np.sum(I_node[name]))


def voxel_interface_areas(cell_label, kinds, contact_of_label, axes_cm, mesh, contact_names):
    """Metal/semiconductor interface area (cm^2 in 3D) per mesh node, from a
    cell-labelled rectilinear grid (the fab voxels or the rectilinear
    mesh's cells): every face between a semiconductor cell and a metal
    cell of contact k is split into 2^(d-1) equal parts, each given to the
    nearest semiconductor node of contact k. The total is the exact
    interface area; on the rectilinear mesh (nodes at the cell corners) the
    split is exactly the dual-cell partition, on the octree the nearest
    node's dual cell.

    cell_label (shape of cells) int, -1 = gas; kinds[label] in
    {"semi", "insulator", "conductor"} (the meshers' kinds); contact_of_label[label] the
    contact name of a conductor label (else None); axes_cm the grid node
    coordinates per axis; mesh.points in cm."""
    from scipy.spatial import cKDTree
    lab = np.asarray(cell_label)
    d = lab.ndim
    axes = [np.asarray(a, float) for a in axes_cm]
    kinds = np.asarray(list(kinds) + ["gas"])
    k = np.where(lab >= 0, lab, len(kinds) - 1)
    is_semi = kinds[k] == "semi"
    bc = np.array(mesh.boundary_bc_type)
    is_ins = mesh.is_insulator if mesh.is_insulator is not None else np.zeros(len(mesh.points), bool)
    area = np.zeros(len(mesh.points))
    centers = [0.5 * (a[1:] + a[:-1]) for a in axes]
    widths = [np.diff(a) for a in axes]
    for name in contact_names:
        metal_labels = [l for l, cn in enumerate(contact_of_label) if cn == name]
        if not metal_labels:
            continue
        nodes = mesh.boundary_point_index[bc == f"contact:{name}"]
        nodes = nodes[~is_ins[nodes]]
        if not len(nodes):
            continue
        tree = cKDTree(mesh.points[nodes])
        is_metal = np.isin(lab, metal_labels)
        for a in range(d):
            sl0 = [slice(None)] * d
            sl1 = [slice(None)] * d
            sl0[a], sl1[a] = slice(0, -1), slice(1, None)
            face = (is_semi[tuple(sl0)] & is_metal[tuple(sl1)]) | (is_metal[tuple(sl0)] & is_semi[tuple(sl1)])
            idx = np.argwhere(face)
            if not len(idx):
                continue
            others = [b for b in range(d) if b != a]
            fa = np.ones(len(idx))
            for b in others:
                fa = fa * widths[b][idx[:, b]]
            # face position: on the grid plane between the two cells along a
            pos = np.empty((len(idx), d))
            pos[:, a] = axes[a][idx[:, a] + 1]
            for b in others:
                pos[:, b] = centers[b][idx[:, b]]
            n_parts = 2 ** (d - 1)
            import itertools
            for o in itertools.product((-0.25, 0.25), repeat=d - 1):
                q = pos.copy()
                for b, ob in zip(others, o):
                    q[:, b] = pos[:, b] + ob * widths[b][idx[:, b]]
                _, j = tree.query(q)
                np.add.at(area, nodes[j], fa / n_parts)
    return area


def boundary_interface_areas(mesh, contact_names):
    """2D meshes: contact length (cm, per unit depth) per node, half of
    each contact boundary edge to each end."""
    bc = np.array(mesh.boundary_bc_type)
    area = np.zeros(len(mesh.points))
    for name in contact_names:
        on = np.zeros(len(mesh.points), bool)
        on[mesh.boundary_point_index[bc == f"contact:{name}"]] = True
        for i, j in mesh.boundary_edges:
            if on[i] and on[j]:
                L = float(np.linalg.norm(mesh.points[i] - mesh.points[j]))
                area[i] += L / 2
                area[j] += L / 2
    return area
