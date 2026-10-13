"""Run the test suite in parallel: one process per test class (per test
for the classes recorded as slow), longest first, on the performance
cores. Same tests and the same pass/fail as `python -m unittest discover
-s testsuite`, in a fraction of the wall time.

  python3 testsuite/run.py                 # quick tier (the default)
  TCAD_FULL_TESTS=1 python3 testsuite/run.py   # full sweeps (before a merge)
  python3 testsuite/run.py -k schottky     # only ids containing "schottky"
  python3 testsuite/run.py --record        # rewrite testsuite/timings.json

Run it with the interpreter whose packages the tests need (the conda env
for the PETSc / octree tests: those skip themselves elsewhere).

Scheduling uses testsuite/timings.json (seconds per test id, from the last
--record run): a class whose tests add up to more than SPLIT_S runs one
process per test (unless it has a setUpClass, which that would repeat),
and the longest jobs start first. A missing or stale
file only makes the schedule worse, never the result.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TIMINGS = os.path.join(HERE, "timings.json")
SPLIT_S = 10.0          # s: a class slower than this runs one process per test
DEFAULT_S = 1.0         # s: assumed for a test missing from timings.json
# the jobs are the parallelism: no BLAS/OpenMP threads inside each one
SINGLE_THREAD = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")


def test_ids():
    """(test ids, the classes that share a setUpClass between their tests)."""
    def walk(suite):
        for t in suite:
            if isinstance(t, unittest.TestSuite):
                yield from walk(t)
            else:
                yield t
    sys.path.insert(0, ROOT)
    tests = list(walk(unittest.TestLoader().discover(HERE, top_level_dir=HERE)))
    shared = {"testsuite." + t.id().rsplit(".", 1)[0] for t in tests
              if "setUpClass" in vars(type(t))}
    return ["testsuite." + t.id() for t in tests], shared


def jobs(ids, timings, shared=()):
    """[(estimated seconds, [test ids])], longest first. A class in
    `shared` stays one process: splitting it repeats its setUpClass (a 3D
    device or fab build) per test - measured slower: 72 s wall kept
    together, 108 s splitting all, 87 s splitting TestWindow + TestPlanar."""
    by_class = {}
    for i in ids:
        if "_FailedTest" in i or "ModuleImportFailure" in i:
            by_class.setdefault(i, []).append(i)      # an import error: report it as its own job
            continue
        by_class.setdefault(i.rsplit(".", 1)[0], []).append(i)
    out = []
    for cls, tests in by_class.items():
        est = [timings.get(t, DEFAULT_S) for t in tests]
        if sum(est) > SPLIT_S and len(tests) > 1 and cls not in shared:
            out += [(e, [t]) for e, t in zip(est, tests)]
        else:
            out.append((sum(est), [cls] if len(tests) > 1 else tests))
    return sorted(out, key=lambda j: -j[0])


def run_job(names):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", "-v", *names], cwd=ROOT,
                       capture_output=True, text=True, env=dict(os.environ, **SINGLE_THREAD))
    wall = time.perf_counter() - t0
    ran = re.search(r"Ran (\d+) test", p.stderr)
    skipped = re.search(r"skipped=(\d+)", p.stderr)
    return dict(names=names, ok=p.returncode == 0, wall=wall, out=p.stdout + p.stderr,
                ran=int(ran.group(1)) if ran else 0, skipped=int(skipped.group(1)) if skipped else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jobs", type=int, default=None, help="parallel processes (default: performance cores)")
    ap.add_argument("-k", default=None, help="only test ids containing this (case-insensitive)")
    ap.add_argument("--record", action="store_true", help="rewrite timings.json from this run (one process per test)")
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    from core.parallel import performance_cores
    n_jobs = a.jobs or performance_cores()
    ids, shared = test_ids()
    if a.k:
        ids = [i for i in ids if a.k.lower() in i.lower()]
    try:
        with open(TIMINGS) as f:
            timings = json.load(f)
    except (OSError, ValueError):
        timings = {}
    todo = [(timings.get(i, DEFAULT_S), [i]) for i in ids] if a.record else jobs(ids, timings, shared)
    print(f"{len(ids)} tests in {len(todo)} processes, {n_jobs} at a time"
          f"{' (TCAD_FULL_TESTS)' if os.environ.get('TCAD_FULL_TESTS', '') not in ('', '0') else ''}")
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(n_jobs) as ex:
        for r in ex.map(run_job, [names for _, names in todo]):
            results.append(r)
            label = r["names"][0] if len(r["names"]) == 1 else f"{len(r['names'])} tests"
            print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['wall']:6.1f}s  {label}", flush=True)
    wall = time.perf_counter() - t0
    failed = [r for r in results if not r["ok"]]
    for r in failed:
        print("\n" + "=" * 70 + f"\nFAILED: {' '.join(r['names'])}\n" + r["out"][-4000:])
    ran = sum(r["ran"] for r in results)
    skipped = sum(r["skipped"] for r in results)
    print(f"\nRan {ran} tests ({skipped} skipped) in {wall:.1f}s wall, "
          f"{sum(r['wall'] for r in results):.1f}s of processes: "
          f"{'FAILED (' + str(len(failed)) + ' processes)' if failed else 'OK'}")
    if a.record and not failed:
        timings.update({r["names"][0]: round(r["wall"], 2) for r in results})
        with open(TIMINGS, "w") as f:
            json.dump(dict(sorted(timings.items())), f, indent=1)
        print(f"recorded {TIMINGS}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
