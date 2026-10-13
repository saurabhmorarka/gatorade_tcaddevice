"""Regression tests for the 2D drivers: rerun each one end to end (in a
subprocess, writing into a temporary --out-dir so the tracked out/ files are
never touched) and compare its CSV output against testsuite/golden/2d_*.json.

The 1D suite (test_examples.py) compares a handful of summary scalars; here
every swept bias point is compared, because the 2D work that this suite
guards (the 3D performance core - linear-solver backends, predictors,
assembly) changes HOW each point is solved, and a regression there shows up
as one bad point in the middle of a curve rather than as a shifted constant.

Tolerances are per example, set by how tightly that driver actually
converges: the diode/MOS-cap/MOSFET sweeps reach res_norm ~1e-8 or better and
reproduce bit-for-bit run to run, so rtol=1e-3 is loose; the BTBT/TAT sweep
stops at res_norm ~2e-5 (its outer generation lagging), and its committed CSVs
already drift ~1% between code versions, so it gets rtol=2e-2. Values whose
magnitude is below the example's `floor` (numerical noise such as a source
current of 1e-17 A/um, or Id at Vds=0) are only checked against that floor.
The MOSFET's floor (1e-13 A/um) is measured, not guessed: an off-state Id of
~7e-14 A/um changes by ~5e-4 relative (~4e-17 A/um absolute) when an already
converged state is polished by further exact Newton iterations that move psi
by only ~1e-16 V - the contact-current sum's own round-off - so relative
agreement below ~1e-13 A/um is not meaningful at rtol=1e-3. The diode's
floor (1e-9 A/cm^2) sits three decades below its smallest physical current
(~5e-6 A/cm^2 at +-0.1 V); its Va=0 point is exactly zero physically and
comes out as +-1e-12-level round-off. The BTBT sweep is the same MOSFET
structure and gets the same 1e-13 A/um floor: its ~8e-15 A/um off-state
drain current was measured to differ by ~2e-16 A/um between the pip venv
(Accelerate BLAS) and the conda env (OpenBLAS) - the same round-off floor.

Quick by default: the two long drivers run a reduced copy of their config
(MOSFET: every third bias point; BTBT: the leakage side, Vg and Vsub from
0 to -2 V at 0.4 V) and those points are compared against the full-sweep
golden files - a converged bias
point does not depend on the continuation path to it, within each
example's rtol. TCAD_FULL_TESTS=1 runs and compares the full sweeps.

Run with:      python3 testsuite/test_2d_examples.py [-v]
Capture with:  python3 testsuite/test_2d_examples.py --capture [name ...]
(capture only after independently verifying the new numbers - see README.md)
"""
import csv
import json
import yaml
import os
import subprocess
import sys
import tempfile
import time
import unittest

TCAD1D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(TCAD1D_ROOT, "testsuite", "golden")

# name -> driver command (relative to TCAD1D_ROOT), CSV files, key columns
# (identify a row), value columns (compared), rtol, absolute noise floor.
EXAMPLES_2D = {
    "diode_2d": dict(cmd=["main2d_sweep.py"], csvs=["iv_sweep.csv"],
                     keys=["Va_V"], values=["J_anode_A_cm2"], rtol=1e-3, floor=1e-9),
    "mos_2d": dict(cmd=["main2d_mos_sweep.py"], csvs=["cv_sweep.csv"],
                   keys=["VG_V"], values=["Qs_C_cm2", "C_lf_F_cm2"], rtol=1e-3, floor=1e-14),
    "mosfet_2d": dict(cmd=["main2d_mosfet_sweep.py"], csvs=["ids_vgs.csv", "ids_vds.csv"],
                      keys=["Vds_V", "Vgs_V"], values=["Id_A_per_um"], rtol=1e-3, floor=1e-13,
                      quick=dict(config="configs/input_mosfet_2d.yaml",
                                 set={"bias": dict(vgs_step_V=0.15, vds_step_V=0.3)},
                                 rows={"ids_vgs.csv": 26, "ids_vds.csv": 24})),
    "btbt_2d": dict(cmd=["-m", "btbt.main_btbt2d_sweep"], csvs=["idvg_btbt.csv", "idvsub_btbt.csv"],
                    keys=["model", "bias_V"],
                    values=["drain_A_per_um", "body_A_per_um", "source_A_per_um"],
                    rtol=2e-2, floor=1e-13,
                    quick=dict(config="configs/input_mosfet_2d_btbt.yaml",
                               # leakage side only: at Vg > 0 all four models agree (on-state,
                               # covered by mosfet_2d); Vg = -2 V keeps nonlocal GIDL, which
                               # turns on below -1.6 V. The walk is 0.1 V steps whatever the
                               # spacing, so the range is what costs.
                               set={"btbt": dict(vgs_start_V=0.0, vgs_step_V=0.4,
                                                 vsub_stop_V=-2.0, vsub_step_V=0.4)},
                               rows={"idvg_btbt.csv": 24, "idvsub_btbt.csv": 24})),
}
FULL = os.environ.get("TCAD_FULL_TESTS", "") not in ("", "0")


def run_example(name, quick=False):
    """Run one 2D driver into a temp dir; return ({csv: [row dicts]}, wall seconds).
    quick: the example's reduced sweep (spec["quick"]), if it has one."""
    spec = EXAMPLES_2D[name]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = []
        if quick and "quick" in spec:
            with open(os.path.join(TCAD1D_ROOT, spec["quick"]["config"])) as f:
                c = yaml.safe_load(f)
            for section, values in spec["quick"]["set"].items():
                c[section].update(values)
            cfg = [os.path.join(tmp, "quick.yaml")]
            with open(cfg[0], "w") as f:
                yaml.safe_dump(c, f)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, *spec["cmd"], *cfg, "--out-dir", tmp],
                              cwd=TCAD1D_ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"{name} driver failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        out = {}
        for fname in spec["csvs"]:
            with open(os.path.join(tmp, fname), newline="") as f:
                out[fname] = list(csv.DictReader(f))
    return out, wall


def _row_key(spec, row):
    # a key column may be absent from one of the example's CSVs (ids_vds.csv
    # has Vgs_V/Vds_V in the other order - DictReader makes order irrelevant)
    return "|".join(f"{k}={row[k]}" for k in spec["keys"] if k in row)


def extract(name, csvs):
    spec = EXAMPLES_2D[name]
    return {fname: {_row_key(spec, r): {v: float(r[v]) for v in spec["values"] if v in r}
                    for r in rows}
            for fname, rows in csvs.items()}


def compare(name, golden, new, quick=False):
    """List of human-readable mismatch strings (empty = pass). quick: the
    new rows are a reduced sweep - each must be a golden row, and their
    number must be the expected one (a point that failed to converge is
    missing from the CSV)."""
    spec = EXAMPLES_2D[name]
    rtol, floor = spec["rtol"], spec["floor"]
    bad = []
    if quick:
        for fname, nrows in new.items():
            want = spec["quick"]["rows"][fname]
            if len(nrows) != want:
                bad.append(f"{fname}: {len(nrows)} rows, expected {want} (point did not converge?)")
            bad += [f"{fname} [{k}]: not a golden bias point" for k in nrows if k not in golden[fname]]
        golden = {f: {k: v for k, v in g.items() if k in new.get(f, {})} for f, g in golden.items()}
    for fname, grows in golden.items():
        nrows = new.get(fname, {})
        for key, gvals in grows.items():
            if key not in nrows:
                bad.append(f"{fname} [{key}]: missing (point did not converge?)")
                continue
            for col, g in gvals.items():
                x = nrows[key][col]
                if abs(g) < floor and abs(x) < floor:
                    continue
                if abs(x - g) > rtol * max(abs(g), floor):
                    bad.append(f"{fname} [{key}] {col}: golden {g:.6e} new {x:.6e} "
                               f"(rel {abs(x - g) / max(abs(g), floor):.2e} > {rtol:g})")
    return bad


def capture(names):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name in names:
        csvs, wall = run_example(name)
        path = os.path.join(GOLDEN_DIR, f"2d_{name}.json")
        with open(path, "w") as f:
            json.dump(extract(name, csvs), f, indent=1, sort_keys=True)
        print(f"captured {path} ({wall:.1f}s)")


class Test2DExamples(unittest.TestCase):
    pass


def _make_test(name):
    def test(self):
        path = os.path.join(GOLDEN_DIR, f"2d_{name}.json")
        if not os.path.exists(path):
            self.skipTest(f"no golden file {path} (run with --capture {name})")
        with open(path) as f:
            golden = json.load(f)
        quick = not FULL and "quick" in EXAMPLES_2D[name]
        csvs, wall = run_example(name, quick)
        print(f"\n  {name}{' (quick)' if quick else ''}: {wall:.1f}s wall-clock", end="")
        bad = compare(name, golden, extract(name, csvs), quick)
        self.assertFalse(bad, f"{len(bad)} mismatch(es):\n  " + "\n  ".join(bad[:30]))
    return test


for _name in EXAMPLES_2D:
    setattr(Test2DExamples, f"test_{_name}", _make_test(_name))


if __name__ == "__main__":
    if "--capture" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--capture"]
        capture(rest or list(EXAMPLES_2D))
    else:
        unittest.main()
