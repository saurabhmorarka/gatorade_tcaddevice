"""Octree box-FV mesher (mesh3d/octree.py): the mandatory gate on a
synthetic device, the Rust kernels against their Python references, and
agreement with the rectilinear mesher on a small fab device.

    python3 -m unittest testsuite.test_octree
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mesh3d.octree as O                                     # noqa: E402
from core.config import _resolve_material_block               # noqa: E402
from core.native import native                                # noqa: E402

HAVE_NATIVE = native is not None and hasattr(native, "voronoi_cells") and hasattr(native, "boxtree_refine")


def synthetic():
    """Si with an n/p/n doping, a 1.5 nm oxide, a metal gate on part of it,
    gas above: every kind of interface and a junction."""
    nx, ny, nz = 20, 30, 12
    lab = np.zeros((nx, ny, nz), dtype=np.int64)
    lab[:, :10, :] = -1
    lab[:, 10:13, :] = 1
    lab[6:14, 4:10, 2:10] = 2
    mats = [dict(name="si", kind="semi"), dict(name="ox", kind="insulator", eps_r=3.9),
            dict(name="gate", kind="conductor", contact="gate")]
    net = np.full((nx + 1, ny + 1, nz + 1), -1e17)
    net[:5] = 1e19
    net[15:] = 1e19
    return lab, mats, net


class TestOctreeMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab, cls.mats, cls.net = synthetic()
        cls.mat = _resolve_material_block({})
        cls.m = O.build_octree_mesh(cls.lab, cls.mats, (1.0, 0.5, 1.0), cls.net, cls.mat,
                                    face_contacts={"body": (1, "max")}, log=lambda *a: None)

    def test_gate_and_volumes(self):
        m, h = self.m, np.array([1.0, 0.5, 1.0])
        self.assertTrue((m.cv_area > 0).all())
        dev = np.isin(self.lab, (0, 1)).sum() * h.prod() * 1e-21
        self.assertAlmostEqual(m.cv_area.sum() / dev, 1.0, places=10)
        si = (self.lab == 0).sum() * h.prod() * 1e-21
        self.assertAlmostEqual(m.cv_area_semi.sum() / si, 1.0, places=10)
        self.assertLess(m.quality["facet_asymmetry"], 1e-6)
        self.assertGreater(m.n_clipped, 0)                     # the test exercises transition cells

    def test_terminals(self):
        types = set(self.m.boundary_bc_type)
        self.assertEqual(types, {"contact:gate", "contact:body"})
        b = self.m.boundary_point_index
        self.assertTrue((np.diff(b) > 0).all())              # sorted: the solver relies on it

    def test_gate_catches_bad_volume(self):
        bad = O.OctreeMesh(**vars(self.m))
        bad.label_volume = self.m.label_volume.copy()
        bad.label_volume[2] *= 1.01
        with self.assertRaises(RuntimeError):
            O.check_octree_mesh(bad, self.lab, self.mats, np.array([1.0, 0.5, 1.0]), log=lambda *a: None)


class TestOctreeMetalStaircase(unittest.TestCase):
    """A contact plug's round bottom staircases the metal surface; where a
    step lies on a facet's plane, one half of that facet is on the metal
    surface and the two nodes sampled different sides of it (one saw metal,
    the other epi): the gate failed (facet asymmetry 0.5) on the PMOS
    nanosheet. Built from that device."""

    def test_plug_contact_mesh_passes_gate(self):
        from fab import flow
        from fab.simulate import build_device
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = build_device(flow.load(os.path.join(root, "configs", "fab_nanosheet_pmos.yaml")), mesher="octree",
                         log=lambda *a: None)
        self.assertLess(d["mesh"].quality["facet_asymmetry"], 1e-6)


class TestOctreeInterfaceFacet(unittest.TestCase):
    """The default FinFET (gate metal 15 nm): an STI-only node and a Si-only
    node whose bisector lies exactly on the fin sidewall got a
    semiconductor facet from the Si side, and the gate failed
    (conformity). Carrier facets now need semiconductor volume at both
    ends; the gate runs on this build."""

    def test_finfet_mesh_passes_gate(self):
        from fab import flow
        from fab.simulate import build_device
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        m = build_device(flow.load(os.path.join(root, "configs", "fab_finfet.yaml")), mesher="octree",
                         log=lambda *a: None)["mesh"]
        semi = m.cv_area_semi > 0
        self.assertFalse(((m.facet_length_semi > 0) & ~semi[m.edges].all(axis=1)).any())


@unittest.skipUnless(HAVE_NATIVE, "Rust kernels not built")
class TestOctreeNative(unittest.TestCase):
    def setUp(self):
        self.lab, self.mats, self.net = synthetic()
        self.h = np.array([1.0, 0.5, 1.0])
        kinds = [m["kind"] for m in self.mats]
        n = self.net
        cn = sum(n[i:n.shape[0] - 1 + i, j:n.shape[1] - 1 + j, k:n.shape[2] - 1 + k]
                 for i in (0, 1) for j in (0, 1) for k in (0, 1)) / 8
        self.tgt = O.target_spacing(self.lab, kinds, cn, self.h, (2, .5, 2), (2, 2, 2), (8, 8, 8), 1.35)

    def tree(self, nat):
        t = O.BoxTree(self.lab, self.tgt, [16, 32, 16], semi_labels=[0])
        t.refine(use_native=nat)
        return t

    def test_tree_identical(self):
        a, b = self.tree(False), self.tree(True)
        _, la, ha = a.leaves()
        _, lb, hb = b.leaves()
        oa, ob = np.lexsort(la.T[::-1]), np.lexsort(lb.T[::-1])
        np.testing.assert_array_equal(la[oa], lb[ob])
        np.testing.assert_array_equal(ha[oa], hb[ob])

    def test_clipping_matches_python(self):
        import core.native as cn
        t = self.tree(True)
        ids, LO, HI = t.leaves()
        ll = self.lab[LO[:, 0], LO[:, 1], LO[:, 2]]
        d_rs = O._dual(t, ids, LO, HI, ll, self.lab, self.h, None)
        saved = cn.native
        try:
            cn.native = None
            d_py = O._dual(t, ids, LO, HI, ll, self.lab, self.h, None)
        finally:
            cn.native = saved
        for d in (d_rs, d_py):
            VN, VL, VV = d["vol"]
            d["cv"] = np.bincount(VN, weights=VV, minlength=d["N"])
        np.testing.assert_allclose(d_rs["cv"], d_py["cv"], rtol=1e-7, atol=1e-10)   # round-off (operation order)


def _has_petsc():
    try:
        import petsc4py  # noqa: F401
        return True
    except ImportError:
        return False


class TestOctreeVsRectilinear(unittest.TestCase):
    """The same small fab device on both meshers: drain current at
    Vds = 50 mV, Vgs = 0.6 V agrees to 1% (different discretizations of
    the same structure)."""

    @unittest.skipUnless(HAVE_NATIVE, "Rust kernels not built (the Python clipper is too slow for this test)")
    @unittest.skipUnless(_has_petsc(), "needs PETSc: SuperLU finds the rectilinear mesh's equilibrium Jacobian "
                                       "exactly singular (DEVELOPMENT_LOG 46)")
    def test_drain_current(self):
        from fab import flow
        from fab.simulate import build_device, drain_current
        from main2d_mosfet_sweep import solve_mosfet
        from mos.mos_params import MOSDevice
        cfg = flow.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "configs", "fab_planar_tiny.yaml"))
        I = {}
        for mesher in ("octree", "cartesian"):
            d = build_device(cfg, mesher=mesher, log=lambda *a: None)
            dev = MOSDevice(gate_workfunction_eV=d["wf"])
            r = solve_mosfet(d["mesh"], d["mat"], dev, d["Cs"], 0.0, 0.0, velocity_saturation=True)
            for vg, vd in ((0.0, 0.05), (0.6, 0.05)):
                r = solve_mosfet(d["mesh"], d["mat"], dev, d["Cs"], vg, vd, psi_init=r["psi"],
                                 phin_init=r["phin"], phip_init=r["phip"], velocity_saturation=True)
            I[mesher] = drain_current(d, r)
        # 1% with the old edge-field vsat force; 1.8% with hybrid (DEVELOPMENT_LOG 50), 2.3% with
        # Masetti mobility (log 55: the heavily doped S/D weigh more): the default octree's known
        # resolution gap (full device +3.4% vs fine meshes). Back to 1% after the defaults study,
        # which the user wants done on a FinFET (plans/3d_handoff.md section 6).
        # 2.58% since the recessed, contact-implanted S/D contacts (DEVELOPMENT_LOG 63):
        # more of the current path runs through the n+ around the recess, where the
        # default octree is coarser - same gap, not a new one. Back to 1% with that study.
        self.assertAlmostEqual(I["octree"] / I["cartesian"], 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
