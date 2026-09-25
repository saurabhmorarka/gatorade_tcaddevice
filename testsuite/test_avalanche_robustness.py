"""Fast subset of avalanche/robustness_matrix.py: three of its hardest
device x mesh cells, each traced through breakdown and held to the same
PASS criteria (reaches J_STOP, every point conserves current to CONS_TOL)
plus its breakdown voltage and no voltage snapback.

Expected BVs are this solver's own values (avalanche/robustness_matrix.py,
session 23), NOT an external reference - the external checks (ionization
integral ~1 at BV, mesh convergence) live in main_avalanche.py and the full
matrix. Run the full matrix after any change to the avalanche physics or
numerics: python3 -m avalanche.robustness_matrix
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avalanche.robustness_matrix import DEVICES, MESHES, build_device, run_one  # noqa: E402

# (device, mesh, expected BV at |J| = 1 A/cm^2 in volts)
CASES = [
    ("p+n flat 1e21/1e17", "very coarse (3.0, 1.25)", -14.072),
    ("n+p gaussian n+ / flat p 5e17", "very coarse (3.0, 1.25)", -8.883),
    ("log-graded both sides", "coarse (1.0, 1.15)", -17.576),
]
BV_RTOL = 2e-3


class TestAvalancheRobustness(unittest.TestCase):
    pass


def _make(device, mesh, bv_expected):
    def test(self):
        mat, dev, g = build_device(*DEVICES[device], *MESHES[mesh])
        r = run_one(mat, g)
        self.assertEqual(r["status"], "reached_J_stop", f"{device} / {mesh}: trace stopped early ({r['status']})")
        self.assertTrue(r["passed"], f"{device} / {mesh}: current non-conservation {r['max_nonconservation']:.1e}")
        self.assertLess(r["snapback_mV"], 1.0, f"{device} / {mesh}: {r['snapback_mV']:.0f} mV voltage snapback")
        self.assertAlmostEqual(r["BV"] / bv_expected, 1.0, delta=BV_RTOL,
                               msg=f"{device} / {mesh}: BV {r['BV']:.4f} V, expected {bv_expected} V")
    return test


for _dev, _mesh, _bv in CASES:
    _name = "test_" + "".join(ch if ch.isalnum() else "_" for ch in f"{_dev}_{_mesh}").strip("_").lower()
    setattr(TestAvalancheRobustness, _name, _make(_dev, _mesh, _bv))


if __name__ == "__main__":
    unittest.main()
