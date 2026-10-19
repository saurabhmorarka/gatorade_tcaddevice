import sys, time, pickle, os, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.getcwd())
import numpy as np
from fab import simulate as S
from main2d_mosfet_sweep import continuation
cfg = sys.argv[1]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
d = S._context(cfg, None, True, "octree", None, None); d["vsat"] = True
print("nodes", d["mesh"].n_nodes if hasattr(d["mesh"], "n_nodes") else len(d["mesh"].cv_area), flush=True)
ramp, it, _ = S._ramp(d, [0.7]); base = ramp[0.7]; print("ramp its", it, flush=True)
inner = S._solver_at(d, None, 0.7, start=base)
def solve(p, g):
    t = time.perf_counter(); r = inner(p, g)
    print(f"  Vgs {p:.4f} its {r['iters']:3d} conv {r.get('converged')} {time.perf_counter()-t:6.1f}s "
          f"res {r.get('residual', r.get('res', ''))}", flush=True)
    return r
def onp(v, r):
    print(f"POINT {v:.2f} I={S.drain_current(d, r):.4e}", flush=True)
    pickle.dump({k: r[k] for k in ("psi", "phin", "phip")}, open(f"{out}/vgs{v:.2f}.pkl", "wb"))
vgs = list(np.round(np.arange(0.05, 0.8001, 0.05), 6))
st, it, _ = continuation(solve, 0.0, base, vgs, on_point=onp, log=print, **S.GATE_WALK)
print("DONE", it, sorted(st))
