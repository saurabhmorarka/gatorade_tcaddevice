"""Tests for the VTK export (core/vtk_io.py, solver2d/vtk_export.py): the
file format round-trips exactly, device cells carry the right regions and
materials, the exported electric field is exact for a linear potential (2D
triangles and extruded wedges), and - when ParaView is installed - ParaView
itself reads the files.

Run with:  python3 testsuite/test_vtk.py [-v]
"""
import glob
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

TCAD1D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TCAD1D_ROOT)

from core import vtk_io  # noqa: E402

PVPYTHON = (glob.glob("/Applications/ParaView*/Contents/bin/pvpython") + [None])[0]


def _mosfet():
    cwd = os.getcwd()
    os.chdir(TCAD1D_ROOT)
    try:
        import main2d_mosfet_sweep as M
        return M._context(M.DEFAULT_PATH)
    finally:
        os.chdir(cwd)


class TestVtkFormat(unittest.TestCase):
    def roundtrip(self, compress):
        rng = np.random.default_rng(0)
        pts = rng.random((7, 3))
        tri = np.array([[0, 1, 2], [2, 3, 4]])
        quad = np.array([[3, 4, 5, 6]])
        pd = {"f64": rng.random(7), "vec": rng.random((7, 3)), "i32": np.arange(7, dtype=np.int32),
              "flag": np.array([True, False] * 3 + [True])}
        cd = {"rid": np.array([0, 1, 1], dtype=np.int32), "E": rng.random((3, 3))}
        fd = {"names": ["a", "bb", "ccc"], "label": "Vg=1.0V", "bias": np.array([0.5, -1.0])}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.vtu")
            vtk_io.write_vtu(path, pts, [tri, quad], [vtk_io.VTK_TRIANGLE, vtk_io.VTK_QUAD],
                             point_data=pd, cell_data=cd, field_data=fd, compress=compress)
            d = vtk_io.read_vtu(path)
        np.testing.assert_array_equal(d["points"], pts)
        np.testing.assert_array_equal(d["connectivity"], [0, 1, 2, 2, 3, 4, 3, 4, 5, 6])
        np.testing.assert_array_equal(d["offsets"], [3, 6, 10])
        np.testing.assert_array_equal(d["types"], [5, 5, 9])
        for k, v in pd.items():
            np.testing.assert_array_equal(d["point_data"][k], v)
        for k, v in cd.items():
            np.testing.assert_array_equal(d["cell_data"][k], v)
        self.assertEqual(d["field_data"]["names"], ["a", "bb", "ccc"])
        self.assertEqual(d["field_data"]["label"], ["Vg=1.0V"])
        np.testing.assert_array_equal(d["field_data"]["bias"], [0.5, -1.0])

    def test_raw(self):
        self.roundtrip(False)

    def test_zlib(self):
        self.roundtrip(True)

    def test_zlib_multiblock(self):
        old = vtk_io._BLOCK
        vtk_io._BLOCK = 40          # force several compressed blocks per array
        try:
            self.roundtrip(True)
        finally:
            vtk_io._BLOCK = old

    def test_pvd(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.pvd")
            vtk_io.write_pvd(path, [dict(file="a.vtu", timestep=-0.5, name="Vg=-0.5"),
                                    dict(file="b.vtu", timestep=1.0)])
            got = vtk_io.read_pvd(path)
        self.assertEqual([(t, os.path.basename(f), n) for t, f, n in got],
                         [(-0.5, "a.vtu", "Vg=-0.5"), (1.0, "b.vtu", None)])

    def test_length_check(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            vtk_io.write_vtu(os.path.join(tmp, "x.vtu"), np.zeros((3, 3)), np.array([[0, 1, 2]]),
                             vtk_io.VTK_TRIANGLE, point_data={"bad": np.zeros(2)})


class TestDeviceExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mesh3d.extrude import extrude_mesh
        cls.mesh, cls.mat, _, _, cls.eq, _ = _mosfet()
        cls.mesh3 = extrude_mesh(cls.mesh, cls.mat, 0.1e-4, 4)

    def test_regions_and_materials(self):
        from solver2d.vtk_export import device_cells
        geo = device_cells(self.mesh, self.mat)
        self.assertEqual(geo["region_names"], ["substrate", "source", "drain", "oxide"])
        self.assertEqual(geo["material_names"], ["Silicon", "SiO2"])
        self.assertTrue(np.all(geo["region_id"] >= 0), "every triangle lies in a region")
        oxide = geo["region_id"] == 3
        cy = self.mesh.points[geo["cells"]][:, :, 1].mean(axis=1)
        self.assertTrue(np.all(cy[oxide] < 0) and np.all(cy[~oxide] > 0), "oxide cells sit above y=0")
        self.assertTrue(np.all(geo["material_id"][oxide] == 1) and np.all(geo["material_id"][~oxide] == 0))

    def test_wedges(self):
        from solver2d.vtk_export import device_cells
        geo2, geo3 = device_cells(self.mesh, self.mat), device_cells(self.mesh3, self.mat)
        self.assertEqual(geo3["cell_type"], vtk_io.VTK_WEDGE)
        self.assertEqual(len(geo3["cells"]), 3 * len(geo2["cells"]))
        np.testing.assert_array_equal(geo3["region_id"], np.tile(geo2["region_id"], 3))

    def test_field_exact_for_linear_potential(self):
        from solver2d.vtk_export import device_cells, cell_field
        a, b, c = 2.0e4, -3.0e4, 5.0e4      # V/cm
        for mesh in (self.mesh, self.mesh3):
            P = mesh.points
            z = P[:, 2] if P.shape[1] == 3 else 0.0
            psi = a * P[:, 0] + b * P[:, 1] + c * z
            E = cell_field(mesh, device_cells(mesh, self.mat), psi)
            want = [-a, -b, -c if P.shape[1] == 3 else 0.0]
            np.testing.assert_allclose(E, np.broadcast_to(want, E.shape), rtol=1e-6, atol=1e-6 * abs(c))

    def test_state_roundtrip(self):
        from solver2d.vtk_export import write_sweep
        with tempfile.TemporaryDirectory() as tmp:
            pvd = write_sweep(tmp, "eq", self.mesh3, self.mat,
                              [dict(state=self.eq3(), bias=0.0, label="eq", contact_bias={"gate": 0.0})])
            (_, f, _), = vtk_io.read_pvd(pvd)
            d = vtk_io.read_vtu(f)
        ins = self.mesh3.is_insulator
        np.testing.assert_array_equal(d["point_data"]["ElectrostaticPotential"], self.eq3()["psi"])
        np.testing.assert_array_equal(d["point_data"]["eDensity"][~ins], self.eq3()["n"][~ins])
        self.assertTrue(np.all(np.isnan(d["point_data"]["eDensity"][ins])), "no carriers in the oxide")
        self.assertEqual(d["field_data"]["contact_names"], ["source", "body", "gate", "drain"])
        Ec, Ev = d["point_data"]["ConductionBandEnergy"], d["point_data"]["ValenceBandEnergy"]
        np.testing.assert_allclose((Ec - Ev)[~ins], self.mat.Eg_eV)

    def eq3(self):
        return {k: np.tile(v, 4) for k, v in self.eq.items() if k in ("psi", "n", "p", "phin", "phip")}


@unittest.skipIf(PVPYTHON is None, "ParaView not installed")
class TestParaViewReads(unittest.TestCase):
    def test_paraview_reads_export(self):
        from solver2d.vtk_export import write_state
        mesh, mat, _, _, eq, _ = _mosfet()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "eq.vtu")
            write_state(path, mesh, mat, eq, label="eq")
            script = ("from paraview.simple import *\n"
                      f"r = XMLUnstructuredGridReader(FileName=[{path!r}]); r.UpdatePipeline()\n"
                      "di = r.GetDataInformation()\n"
                      "print('RESULT', di.GetNumberOfPoints(), di.GetNumberOfCells(),"
                      " len(r.PointData.keys()), len(r.CellData.keys()))\n")
            out = subprocess.run([PVPYTHON, "-c", script], capture_output=True, text=True, timeout=120)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")), None)
        self.assertIsNotNone(line, out.stdout + out.stderr)
        n_pts, n_cells, n_pd, n_cd = map(int, line.split()[1:])
        self.assertEqual((n_pts, n_cells), (len(mesh.points), len(mesh.triangles)))
        # 12 scalar point fields + 4 nodal vectors (E, e/h/total current density)
        self.assertEqual((n_pd, n_cd), (16, 4))


if __name__ == "__main__":
    unittest.main()
