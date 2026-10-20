"""Gate-length sweep of the 3D FinFET (NMOS / PMOS) with the ballistic
mobility on and off and (PMOS) the process stress on and off: runs
`python -m fab simulate` per case (octree, fast vsat) and reads the I-V
(docs/ballistic.md, DEVELOPMENT_LOG 84).

The device is configs/fab_finfet{,_pmos}.yaml with Lg changed and the
contacted poly pitch cpp = Lg + 33 nm (the default 48 nm at Lg 15: the
spacers, S/D epi and contacts keep their size). Per case: Vt,sat by
constant current 100 nA * W/L (W = the template's effective width, the
currents are per um of it), Ion at |Vgs - Vt,sat| = 0.45 V, |Vds| = 0.7 V
(interpolated on the 0.05 V gate grid, linear in Id), Idlin at the same
overdrive above Vt,lin.

  ~/miniconda3/envs/tcad/bin/python scripts/ballistic/finfet_sweep.py --out out/ballistic_finfet \
      --types nmos --lg 12 15 20 30 --cases off bal
  ... --report            (tables from the finished runs)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
OVERDRIVE = 0.45


def write_config(kind, Lg, out):
    tpl = os.path.join(ROOT, "configs", "fab_finfet.yaml" if kind == "nmos" else "fab_finfet_pmos.yaml")
    cfg = yaml.safe_load(open(tpl))
    cfg["params"]["Lg"] = f"{Lg:g}nm"
    cfg["params"]["cpp"] = f"{Lg + 33:g}nm"
    path = os.path.join(out, "cfg", f"finfet_{kind}_Lg{Lg:g}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yaml.safe_dump(cfg, open(path, "w"))
    return path


def run(kind, Lg, tag, out, extra=()):
    cfg = os.path.join(out, "cfg", f"finfet_{kind}_Lg{Lg:g}.yaml")
    d = os.path.join(out, f"{kind}_Lg{Lg:g}_{tag}")
    if os.path.exists(os.path.join(d, "ids_vgs.csv")):
        return d, 0.0, "cached"
    sgn = 1 if kind == "nmos" else -1
    cmd = [PY, "-m", "fab", "simulate", cfg, "--out", d, "--mesher", "octree",
           "--vds", str(sgn * 0.05), str(sgn * 0.7), "--vgs-step", "0.05"]
    cmd += ["--vgs-max", "0.85"] if kind == "nmos" else ["--vgs-min", "-0.85", "--vgs-max", "0"]
    if "bal" in tag:
        cmd += ["--ballistic"]
    if tag.startswith("str"):
        st = os.path.join(out, "stress", f"finfet_{kind}_Lg{Lg:g}.npz")
        cmd += ["--stress", st] if os.path.exists(st) else ["--stress"]
    cmd += list(extra)
    os.makedirs(d, exist_ok=True)
    t0 = time.perf_counter()
    with open(os.path.join(d, "run.log"), "w") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONPATH=ROOT))
    if tag.startswith("str"):
        # keep the solved stress field for the other stress case of this Lg
        src = os.path.join(d, "stress_field.npz")
        dst = os.path.join(out, "stress", f"finfet_{kind}_Lg{Lg:g}.npz")
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            import shutil
            shutil.copy(src, dst)
    return d, time.perf_counter() - t0, rc


def curves(d):
    C = {}
    with open(os.path.join(d, "ids_vgs.csv")) as fh:
        r = csv.reader(fh)
        next(r)
        for vds, vgs, i in r:
            C.setdefault(float(vds), []).append((float(vgs), abs(float(i))))
    return {k: np.array(sorted(v)) for k, v in C.items()}


def metrics(d, Lg, kind):
    """Vt,sat / Vt,lin (constant current 100 nA W/L), Ion and Idlin at 0.45 V overdrive."""
    sgn = 1 if kind == "nmos" else -1
    C = curves(d)
    I_cc = 1e-7 / (Lg * 1e-3)
    out = {}
    for vds, name in ((sgn * 0.7, "sat"), (sgn * 0.05, "lin")):
        if vds not in C:
            out["missing"] = out.get("missing", []) + [vds]
            continue
        a = C[vds]
        a = a[np.argsort(sgn * a[:, 0])]
        a = a[sgn * a[:, 0] >= 0]
        v, I = sgn * a[:, 0], a[:, 1]
        k = np.flatnonzero(I >= I_cc)
        if not len(k) or k[0] == 0:
            continue
        k = k[0]
        vt = v[k - 1] + (v[k] - v[k - 1]) * (np.log(I_cc) - np.log(I[k - 1])) / (np.log(I[k]) - np.log(I[k - 1]))
        out[f"Vt_{name}"] = float(sgn * vt)
        target = vt + OVERDRIVE
        if target <= v[-1]:
            out["Ion" if name == "sat" else "Idlin"] = float(np.interp(target, v, I)) * 1e6
        out[f"n_{name}"] = int(len(v))
    if "Vt_sat" in out and "Vt_lin" in out:
        out["DIBL"] = abs(out["Vt_lin"] - out["Vt_sat"]) / 0.65 * 1e3
    log = open(os.path.join(d, "run.log")).read()
    for line in log.splitlines():
        if "total" in line and " s" in line and "time" not in out:
            pass
        if "channel sigma" in line:
            out["stress_line"] = line.strip()
    out["log_tail"] = log.strip().splitlines()[-1] if log.strip() else ""
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/ballistic_finfet")
    ap.add_argument("--lg", type=float, nargs="*", default=[12, 15, 20, 30])
    ap.add_argument("--types", nargs="*", default=["nmos"])
    ap.add_argument("--cases", nargs="*", default=["off", "bal"])
    ap.add_argument("--parallel", type=int, default=3, help="simulations at once (each runs 2 curve workers)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="extra fab simulate arguments")
    ap.add_argument("--tag-suffix", default="")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    jobs = []
    for kind in a.types:
        for Lg in a.lg:
            write_config(kind, Lg, out)
            for tag in a.cases:
                jobs.append((kind, Lg, tag + a.tag_suffix))
    if not a.report:
        # stress cases first per Lg: "str" solves the field, "str_bal" reuses it
        order = sorted(jobs, key=lambda j: (not j[2].startswith("str"), "bal" in j[2], -j[1]))
        first = [j for j in order if j[2].startswith("str") and "bal" not in j[2]]
        rest = [j for j in order if j not in first]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(a.parallel) as ex:
            for batch in (first, rest):
                for (kind, Lg, tag), (d, dt, rc) in zip(batch, ex.map(lambda j: run(*j, out, a.extra), batch)):
                    print(f"{kind} Lg {Lg:g} {tag}: rc {rc}, {dt:.0f} s", flush=True)
        print(f"total {time.perf_counter() - t0:.0f} s")
    res = []
    for kind, Lg, tag in jobs:
        d = os.path.join(out, f"{kind}_Lg{Lg:g}_{tag}")
        if os.path.exists(os.path.join(d, "ids_vgs.csv")):
            res.append(dict(kind=kind, Lg=Lg, tag=tag, **metrics(d, Lg, kind)))
    json.dump(res, open(os.path.join(out, f"results{a.tag_suffix}.json"), "w"), indent=1)
    for r in res:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
