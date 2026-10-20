"""Where does each mobility term act on the I-V? 2D long-channel NMOS
(configs/input_mosfet_2d.yaml, Lg 0.5 um, tox 5 nm), cumulative stack.
Run from the mobility-stack worktree root."""
import copy, os, sys, warnings, json
from multiprocessing import Pool
import numpy as np
import yaml

WT = os.getcwd()
sys.path.insert(0, WT)
OUT = sys.argv[1]
VGS = [round(v, 2) for v in np.arange(0.0, 2.01, 0.1)]
VDS_SWEEP = [round(v, 2) for v in np.arange(0.0, 2.01, 0.1)]

CASES = {
    "bulk (Masetti)": "off",
    "+ phonon": {"mode": "fast", "delta_scale_n": 1e6, "delta_scale_p": 1e6, "interface_charge": 0},
    "+ surface roughness": {"mode": "fast", "interface_charge": 0},
    "+ Coulomb (Qit 1e11)": {"mode": "fast", "interface_charge": 1e11},
    "+ Coulomb (Qit 3e11)": {"mode": "fast", "interface_charge": 3e11},
}


def cfg_path(name, mob, vsat):
    import main2d_mosfet_sweep as m
    cfg = yaml.safe_load(open(m.DEFAULT_PATH))
    cfg.setdefault("physics", {})["mobility"] = mob
    cfg["physics"]["velocity_saturation"] = vsat
    p = os.path.join(OUT, f"cfg_{abs(hash((name, vsat)))}.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    return p


def run(task):
    name, mob, vsat, kind = task
    os.chdir(WT)
    import main2d_mosfet_sweep as m
    from solver2d.current import contact_current
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh, mat, dev, Cs, eq, vs = m._context(cfg_path(name, mob, vsat))
    gate = next(c for c in mesh.domain.contacts if c.name == "gate")
    x0, x1 = gate.x_range_cm
    L = x1 - x0
    P = mesh.points
    xm0, xm1 = x0 + 0.25 * L, x1 - 0.25 * L
    col = (P[:, 0] > xm0) & (P[:, 0] < xm1) & (P[:, 1] >= 0.0) & (P[:, 1] < 30e-7) & ~mesh.is_insulator

    def solve(vg, vd, g):
        return m.solve_mosfet(mesh, mat, dev, Cs, vg, vd, psi_init=g["psi"], phin_init=g["phin"],
                              phip_init=g["phip"], cold_retry=False, velocity_saturation=vs)
    res = {}
    if kind == "idvg":
        Vds = 0.05
        ramp, _, _ = m.continuation(lambda p_, g: solve(0.0, p_, g), 0.0, eq, [Vds])
        st, _, _ = m.continuation(lambda p_, g: solve(p_, Vds, g), 0.0, ramp[Vds], VGS)
        for v in VGS:
            r = st.get(v)
            if r is None:
                continue
            w = (r["n"] * mesh.cv_area_semi)[col]
            sm_ = getattr(mesh, "surface_mobility", None)
            if sm_ is not None:
                mu = sm_.nodal(mat, r["psi"], r["n"], r["p"])[0]
            else:
                from solver2d.newton_solver_qf_2d import _mesh_mobility_nodal
                mu = _mesh_mobility_nodal(mesh, mat)[0]
            res[v] = dict(Id=abs(contact_current(mesh, mat, r, "drain")) * 1e-4 * 1e6,  # uA/um
                          Ninv=float(w.sum() / (xm1 - xm0)), mu=float((mu[col] * w).sum() / w.sum()))
    else:   # Id-Vd at Vg = 1.5
        Vg = 1.5
        gv, _, _ = m.continuation(lambda p_, g: solve(p_, 0.0, g), 0.0, eq, [Vg])
        st, _, _ = m.continuation(lambda p_, g: solve(Vg, p_, g), 0.0, gv[Vg], VDS_SWEEP)
        for v in VDS_SWEEP:
            r = st.get(v)
            if r is not None:
                res[v] = dict(Id=abs(contact_current(mesh, mat, r, "drain")) * 1e-4 * 1e6)
    return name, vsat, kind, res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tasks = [(n, mob, False, "idvg") for n, mob in CASES.items()]
    tasks += [(n, mob, vs, "idvd") for n, mob in CASES.items() if n in ("bulk (Masetti)", "+ surface roughness")
              for vs in (False, True)]
    with Pool(len(tasks)) as p:
        R = p.map(run, tasks)
    json.dump([[n, vs, k, {str(a): b for a, b in r.items()}] for n, vs, k, r in R],
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("done", [(n, vs, k, len(r)) for n, vs, k, r in R])
