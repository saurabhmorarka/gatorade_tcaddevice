"""Series-resistance partition of a 3D transistor in the linear regime
(DEVELOPMENT_LOG 87, docs/resistance_partition.md).

Method (quasi-Fermi dissipation). Current is conserved, so the electrical
power the drain delivers, I * Vds, is dissipated somewhere in the device.
Every mesh edge dissipates  I_e (phi_i - phi_j)  (Scharfetter-Gummel edge
current times the drop of the carrier's quasi-Fermi potential, electrons and
holes both), and every sub-grid Schottky contact node dissipates
I_node (V_contact - phi_node), the drop across the barrier from the metal to
the quasi-Fermi level just past the depletion region. The sum is exactly
I * Vds (checked: `closure`), and a region's resistance is
R = P_region / I^2: additive, no choice of cut planes, no assumption that the
equipotentials are flat. Regions are x-slabs about the gate:

  channel  between the gate edges (xg0 .. xg1)
  spacer   under the spacer on each side (the extension / fin under it)
  epi      beyond the spacer up to the contact (epi, fin and sub-fin Si)
  contact  the sub-grid metal / semiconductor barrier of the contact

each for the source and the drain side. Resistances are per fin (FinFET) or
per stack (nanosheet) of the real device (the half-width solve is doubled),
and in Ohm um with the template's effective width W (as Id / W).

  ~/miniconda3/envs/tcad/bin/python scripts/resistance/partition.py --out out/rpart
  ... --devices finfet_pmos --ballistic off --mesher octree --set cpp=36
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fab.partition import REGIONS, partition      # (the same partition every `fab simulate` run writes)

DEVICES = {"finfet_nmos": "configs/fab_finfet.yaml", "finfet_pmos": "configs/fab_finfet_pmos.yaml",
           "nanosheet_nmos": "configs/fab_nanosheet.yaml", "nanosheet_pmos": "configs/fab_nanosheet_pmos.yaml"}


def _num(v):
    try:
        return float(v)
    except ValueError:
        return v


def run_case(task):
    import warnings
    warnings.simplefilter("ignore")
    from fab import simulate as S
    from main2d_mosfet_sweep import continuation
    t0 = time.perf_counter()
    cfg = yaml.safe_load(open(os.path.join(ROOT, DEVICES[task["device"]])))
    for k, v in task["set"].items():
        cfg["params"][k] = v
    tag = task["tag"]
    path = os.path.join(task["out"], "cfg", f"{tag}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yaml.safe_dump(cfg, open(path, "w"))
    sgn = -1.0 if cfg["params"]["type"] == "pmos" else 1.0
    vds, vgs = sgn * task["vds"], sgn * task["vgs"]
    bal = None if task["ballistic"] == "off" else tuple(sorted({"enabled": True}.items()))
    mob = tuple(sorted({"mode": "fast", "orientation": "auto"}.items()))
    d = S._context(path, None, True, task["mesher"], None, None, None, mob, bal)
    d["vsat"] = True
    ramp, it, _ = S._ramp(d, [vds])
    if vds not in ramp:
        return dict(task=task, failed="drain ramp")
    st, it2, _ = continuation(S._solver_at(d, None, vds, start=ramp[vds]), 0.0, ramp[vds], [vgs / 2, vgs], **S.GATE_WALK)
    if vgs not in st:
        return dict(task=task, failed=f"gate walk to {vgs}")
    r = st[vgs]
    out = partition(d, r, vds)
    out.update(task=task, Vgs=vgs, nodes=len(d["mesh"].points), time_s=time.perf_counter() - t0,
               Id_uA_um=out["I_A"] / out["W_um"] * 1e6)
    if task.get("save_state"):
        np.savez_compressed(os.path.join(task["out"], f"{tag}_state.npz"), **{k: r[k] for k in ("psi", "n", "p", "phin", "phip")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/rpart")
    ap.add_argument("--devices", nargs="*", default=list(DEVICES))
    ap.add_argument("--ballistic", nargs="*", default=["off", "on"])
    ap.add_argument("--mesher", default="octree", choices=("cartesian", "octree"))
    ap.add_argument("--vds", type=float, default=0.05, help="|Vds| (V)")
    ap.add_argument("--vgs", type=float, default=0.7, help="|Vgs| (V)")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V", help="template parameter overrides")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--save-state", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    st = {k: _num(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    tasks = [dict(device=dv, ballistic=b, mesher=a.mesher, vds=a.vds, vgs=a.vgs, set=st, out=out, save_state=a.save_state,
                  tag=f"{dv}_bal{b}_{a.mesher}{a.tag}") for dv in a.devices for b in a.ballistic]
    from core.parallel import performance_cores
    res = []
    with ProcessPoolExecutor(min(a.workers or performance_cores(), len(tasks))) as ex:
        for r in ex.map(run_case, tasks):
            res.append(r)
            if "failed" in r:
                print(r["task"]["tag"], "FAILED", r["failed"], flush=True)
                continue
            print(f"{r['task']['tag']:44s} Id {r['Id_uA_um']:7.1f} uA/um  Rtot {r['R_total'] / 1e3:6.2f} kOhm  "
                  + "  ".join(f"{g} {r['R_' + g] / 1e3:5.2f}" for g in REGIONS) + f"  closure {r['closure']:.4f}  "
                  f"{r['time_s']:.0f} s", flush=True)
            fn = os.path.join(out, r["task"]["tag"] + ".json")
            json.dump(r, open(fn, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
