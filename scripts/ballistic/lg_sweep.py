"""Gate-length sweep of the 2D planar NMOS / PMOS with the ballistic
mobility on and off, process stress on and off (docs/ballistic.md,
DEVELOPMENT_LOG 84; the criteria are mobility_plan.md section 4.3).

Every device is the 30 nm configs/input_mosfet_2d_stress_{nmos,pmos}.yaml
cell with the gate (and the planar_flow stress cell) stretched to Lg:
S/D 44 nm + spacer 8 nm on each side, the same doping, oxide and mesh
rules. Per case:

  * saturation: |Vds| 0.7 V, gate walk; Vt,sat by constant current
    100 nA * W/L; Ion at |Vgs - Vt,sat| = 0.45 V; the "source velocity"
    Ion / (Cox (Vgs - Vt)); the sensitivity S = dln Ion / dln mu_low
    (+10% low-field mobility everywhere, one extra solve);
  * linear: |Vds| 50 mV, Vt,lin by the same constant current; at overdrive
    0.4 and 0.6 V the apparent mobility mu_app = L Id / (Q_inv dphi), with
    Q_inv the inversion charge per area under the gate and dphi the
    carrier quasi-Fermi drop along the surface between the gate edges
    (intrinsic channel: no series resistance).

  ~/miniconda3/envs/tcad/bin/python scripts/ballistic/lg_sweep.py --out out/ballistic_lg
  ... --lg 15 30 --types nmos --cases off,on      (a subset)
"""
import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SD, SPACER = 44.0, 8.0          # nm, planar_flow GEOM
EPS0, EPS_OX = 8.8541878128e-14, 3.9
TOX_CM = 1e-7
Q = 1.602176634e-19
OVERDRIVE = 0.45
LIN_OVERDRIVES = (0.4, 0.6)
EXT_NM, BODY_CM3, DEEP_NM = 5.0, 6e18, 25.0


def device_config(kind, Lg, stress_file=None, ballistic=None, ext_nm=EXT_NM, body_cm3=BODY_CM3,
                  deep_nm=DEEP_NM, vsat=True):
    """The template config with the gate stretched to Lg (nm). Both types get
    a shallow S/D extension (ext_nm deep, 1e20) from the gate edge under the
    spacer and a deep S/D outside the spacer (NMOS n+ Si deep_nm deep; PMOS
    the template's SiGe, 40 nm), a body of body_cm3: a scalable planar
    device down to Lg 12 nm (the template's 25 nm deep junction at the gate
    edge punches through below ~20 nm)."""
    tpl = os.path.join(ROOT, "configs", f"input_mosfet_2d_stress_{kind}.yaml")
    cfg = yaml.safe_load(open(tpl))
    cfg = copy.deepcopy(cfg)
    a = SD + SPACER                       # gate left edge (nm)
    b = a + Lg
    W = 2 * a + Lg
    um = lambda v: round(v * 1e-3, 6)
    g = cfg["geometry"]
    g["width_um"] = um(W)
    g["substrate"]["concentration_cm3"] = float(body_cm3)
    t = "n" if kind == "nmos" else "p"
    regs = {r["name"]: r for r in g["regions"]}
    ext = lambda name, x0, x1: dict(name=name, doping_type=t, concentration_cm3=1.0e20,
                                    x_range_um=[um(x0), um(x1)], y_range_um=[0.0, um(ext_nm)])
    deep = lambda name, x0, x1: dict(regs[name], x_range_um=[um(x0), um(x1)],
                                     **({"y_range_um": [0.0, um(deep_nm)]} if kind == "nmos" else {}))
    g["regions"] = [ext("source_ext", 0.0, a), ext("drain_ext", b, W),
                    deep("source", 0.0, SD), deep("drain", b + SPACER, W),
                    dict(regs["oxide"], x_range_um=[um(a), um(b)])]
    g["mesas"][0]["x_range_um"] = [um(a), um(b)]
    for c in cfg["contacts"]:
        if c["name"] == "gate":
            c["x_range_um"] = [um(a), um(b)]
        elif c["name"] == "drain":
            c["x_range_um"] = [um(W - 30), um(W - 14)]
        elif c["name"] == "body":
            c["x_range_um"] = [0.0, um(W)]
    cfg["mesh"]["refine_boxes"][0]["x_range_um"] = [um(a - 12), um(b + 12)]
    if stress_file:
        cfg["stress"] = dict(cfg["stress"], file=stress_file)
    else:
        cfg.pop("stress", None)
    if not vsat:
        cfg["physics"]["velocity_saturation"] = False
    if ballistic is not None:
        cfg["ballistic"] = dict(ballistic, enabled=True)
    return cfg


def stress_field_file(kind, Lg, out):
    """Solve the planar_flow stress for a cell with this Lg once; .npz path."""
    from mechanics.planar_flow import DEFAULTS, GEOM, run_flow
    from mechanics.stress_field import StressField
    flow = "nmos_smt" if kind == "nmos" else "pmos_gate_last"
    path = os.path.join(out, "stress", f"{flow}_Lg{Lg:g}.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        b = run_flow(flow, geom=dict(GEOM, Lg=Lg), par=dict(DEFAULTS))
        StressField.from_planar_flow(b, flow=flow, par=dict(DEFAULTS)).save(path)
    return path


def _surface_nodes(mesh):
    P = mesh.points * 1e7
    semi = ~mesh.is_insulator
    top = semi & (np.abs(P[:, 1]) < 1e-6)
    return np.flatnonzero(top), P


def channel_probe(mesh, r, kind, gx):
    """(Q_inv C/cm^2 under the gate, quasi-Fermi drop V between the gate
    edges at the surface, sign-adjusted)."""
    from solver2d.newton_solver_qf_2d import _mesh_semi_geometry
    P = mesh.points * 1e7
    c = r["n"] if kind == "nmos" else r["p"]
    phi = r["phin"] if kind == "nmos" else r["phip"]
    cv = mesh.cv_area_semi if getattr(mesh, "cv_area_semi", None) is not None else mesh.cv_area
    sel = (~mesh.is_insulator) & (P[:, 0] >= gx[0]) & (P[:, 0] <= gx[1]) & (P[:, 1] < 15.0)
    L_cm = (gx[1] - gx[0]) * 1e-7
    Qinv = Q * float(np.sum(c[sel] * cv[sel])) / L_cm
    top, _ = _surface_nodes(mesh)
    xs = P[top, 0]
    i0 = top[np.argmin(np.abs(xs - gx[0]))]
    i1 = top[np.argmin(np.abs(xs - gx[1]))]
    return Qinv, abs(float(phi[i1] - phi[i0]))


def channel_mobility(mesh, mat, r, kind, gx):
    """The carrier-weighted low-field mobility (ballistic term removed) of
    the channel edges along x under the gate: the mu of Matthiessen's
    prediction mu_bal/(mu + mu_bal) for the apparent-mobility ratio."""
    from solver2d.newton_solver_qf_2d import low_field_mobility
    b = getattr(mesh, "edge_mu_ballistic", None)
    mesh.edge_mu_ballistic = None
    try:
        mu = low_field_mobility(mesh, mat)[0 if kind == "nmos" else 1]
    finally:
        mesh.edge_mu_ballistic = b
    P = mesh.points * 1e7
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    d = P[jj] - P[ii]
    xm, ym = 0.5 * (P[ii, 0] + P[jj, 0]), 0.5 * (P[ii, 1] + P[jj, 1])
    sel = (np.abs(d[:, 1]) < 1e-9) & (xm > gx[0]) & (xm < gx[1]) & (ym >= 0) & (ym < 10)
    sel &= ~(mesh.is_insulator[ii] | mesh.is_insulator[jj])
    c = r["n"] if kind == "nmos" else r["p"]
    wt = 0.5 * (c[ii] + c[jj])[sel]
    return float(np.sum(wt * mu[sel]) / np.sum(wt))


def run_case(task):
    """One (kind, Lg, stress, ballistic) device: saturation and linear walks."""
    import warnings
    warnings.simplefilter("ignore")
    import main2d_mosfet_sweep as M
    from solver2d.newton_solver_qf_2d import set_edge_mobility_factor
    from solver2d.stress_mobility import channel_report
    t0 = time.perf_counter()
    kind, Lg = task["kind"], task["Lg"]
    sgn = 1.0 if kind == "nmos" else -1.0
    cfg = device_config(kind, Lg, task.get("stress_file"), task.get("ballistic"), **task.get("dev", {}))
    path = os.path.join(task["out"], "cfg", f"{kind}_Lg{Lg:g}_{task['tag']}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yaml.safe_dump(cfg, open(path, "w"))
    mesh, mat, dev, Cs, eq, vsat = M._context(path)
    gate = next(c for c in M.build_from_config(cfg)[0].contacts if c.name == "gate")
    gx = (gate.x_range_cm[0] * 1e7, gate.x_range_cm[1] * 1e7)
    L_um = Lg * 1e-3
    I_cc = 1e-7 / L_um                          # A/um: 100 nA * W/L
    Cox = EPS0 * EPS_OX / TOX_CM
    out = dict(kind=kind, Lg=Lg, tag=task["tag"], n_nodes=len(mesh.points))
    if getattr(mesh, "stress_report", None) is not None:
        f_n, f_p = mesh.edge_mu0_factor
        ch = channel_report(mesh, f_n, f_p, mesh.stress_field, box_nm=((gx[0] + 1, gx[1] - 1), (0.0, 3.0)))
        if ch:
            out.update(sigma_ch_MPa=ch["sigma_MPa"], gamma_n=ch["f_n"], gamma_p=ch["f_p"])
    if getattr(mesh, "ballistic_report", None):
        out["mu_bal"] = mesh.ballistic_report["mu_bal_n" if kind == "nmos" else "mu_bal_p"]
    iters = 0
    failed = []

    def at(Vgs, Vds):
        return lambda p, g: M.solve_mosfet(mesh, mat, dev, Cs, *((p, Vds) if Vgs is None else (Vgs, p)),
                                            psi_init=g["psi"], phin_init=g["phin"], phip_init=g["phip"],
                                            cold_retry=False, velocity_saturation=vsat)

    def walk(Vds, vgs_targets):
        nonlocal iters
        ramp, it, _ = M.continuation(at(0.0, None), 0.0, eq, [Vds])
        iters += it
        if Vds not in ramp:
            failed.append(f"drain ramp {Vds}")
            return {}, None
        up = [v for v in vgs_targets if sgn * v > 0]
        down = sorted((v for v in vgs_targets if sgn * v < 0), key=lambda v: -sgn * v)
        st = {0.0: ramp[Vds]}
        for tg in (up, down):
            s, it, _ = M.continuation(at(None, Vds), 0.0, ramp[Vds], tg, psi_only=True)
            iters += it
            st.update(s)
        missing = [v for v in vgs_targets if v not in st]
        if missing:
            failed.append(f"Vds {Vds}: Vgs {missing[0]:+.3f}..")
        return st, ramp[Vds]

    def vt_cc(states):
        v = np.array(sorted(states, key=lambda x: sgn * x))
        I = np.array([abs(M._id_per_um(mesh, mat, states[x])) for x in v])
        k = np.flatnonzero(I >= I_cc)
        if not len(k) or k[0] == 0:
            return None
        k = k[0]
        a, b = np.log(I[k - 1]), np.log(I[k])
        return float(v[k - 1] + (v[k] - v[k - 1]) * (np.log(I_cc) - a) / (b - a))

    def solve_from(states, Vds, vg):
        nonlocal iters
        near = min(states, key=lambda x: abs(x - vg))
        st, it, _ = M.continuation(at(None, Vds), near, states[near], [vg], psi_only=True, h_init=0.02)
        iters += it
        if vg not in st:
            failed.append(f"Vds {Vds}: Vgs {vg:+.3f}")
            return None
        return st[vg]

    grid = [round(sgn * v, 4) for v in np.arange(-0.5, 1.20001, 0.05) if abs(v) > 1e-9]
    # saturation
    Vd = sgn * 0.7
    sat, _ = walk(Vd, grid)
    vt = vt_cc(sat) if sat else None
    out["Vt_sat"] = vt
    if vt is not None:
        vg = round(vt + sgn * OVERDRIVE, 6)
        r = solve_from(sat, Vd, vg)
        if r is not None:
            Ion = abs(M._id_per_um(mesh, mat, r))
            out["Ion_uA_um"] = Ion * 1e6
            # source velocity: Ion / (W Cox Vov), cm/s
            out["v_src_cm_s"] = Ion * 1e4 / (Cox * OVERDRIVE)
            # S = dln Ion / dln mu_low: +10% low-field mobility everywhere
            set_edge_mobility_factor(mesh, "perturb", np.full(len(mesh.edges), 1.1))
            r2 = M.solve_mosfet(mesh, mat, dev, Cs, vg, Vd, psi_init=r["psi"], phin_init=r["phin"],
                                phip_init=r["phip"], cold_retry=False, velocity_saturation=vsat)
            iters += r2["iters"]
            if M._converged(r2):
                out["S_mu"] = float(np.log(abs(M._id_per_um(mesh, mat, r2)) / Ion) / np.log(1.1))
            set_edge_mobility_factor(mesh, "perturb", None)
    # linear
    Vl = sgn * 0.05
    lin, _ = walk(Vl, grid)
    vtl = vt_cc(lin) if lin else None
    out["Vt_lin"] = vtl
    if vt is not None and vtl is not None:
        out["DIBL_mV_V"] = abs(vtl - vt) / 0.65 * 1e3
    if vtl is not None:
        for ov in LIN_OVERDRIVES:
            vg = round(vtl + sgn * ov, 6)
            r = solve_from(lin, Vl, vg)
            if r is None:
                continue
            Id = abs(M._id_per_um(mesh, mat, r)) * 1e4        # A/cm of width
            Qi, dphi = channel_probe(mesh, r, kind, gx)
            out[f"Idlin_{ov}"] = Id * 1e2                      # uA/um
            out[f"mu_app_{ov}"] = Id * (gx[1] - gx[0]) * 1e-7 / (Qi * dphi)
            out[f"Qinv_{ov}"] = Qi
            out[f"mu_ch_{ov}"] = channel_mobility(mesh, mat, r, kind, gx)
    out.update(newton_iters=iters, failed=failed, time_s=time.perf_counter() - t0)
    return out


CASES = {   # tag: (stress, ballistic options)
    "off": (False, None),
    "bal": (False, {}),
    "str": (True, None),
    "str_bal": (True, {}),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/ballistic_lg")
    ap.add_argument("--lg", type=float, nargs="*", default=[12, 15, 20, 30, 50, 100])
    ap.add_argument("--types", nargs="*", default=["nmos", "pmos"])
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--k", type=float, default=None, help="ballistic k (cm2/(V s nm)) for both carriers")
    ap.add_argument("--placement", default=None, choices=("lowfield", "total"),
                    help="ballistic placement (default: the model's, lowfield)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--ext-nm", type=float, default=EXT_NM, help="S/D extension depth (nm)")
    ap.add_argument("--no-vsat", action="store_true", help="velocity saturation off (diagnosis)")
    ap.add_argument("--body", type=float, default=BODY_CM3, help="body doping (cm^-3)")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    t0 = time.perf_counter()
    tasks = []
    for kind in args.types:
        for Lg in args.lg:
            sf = None
            for tag in args.cases.split(","):
                stress, bal = CASES[tag]
                if stress and sf is None:
                    sf = stress_field_file(kind, Lg, out)
                if bal is not None and args.k is not None:
                    bal = dict(bal, k=args.k)
                if bal is not None and args.placement:
                    bal = dict(bal, placement=args.placement)
                tasks.append(dict(kind=kind, Lg=Lg, tag=tag + args.tag_suffix, out=out,
                                  stress_file=sf if stress else None, ballistic=bal,
                                  dev=dict(ext_nm=args.ext_nm, body_cm3=args.body, vsat=not args.no_vsat)))
    print(f"{len(tasks)} cases; stress fields {time.perf_counter() - t0:.0f} s", flush=True)
    from core.parallel import performance_cores
    res = []
    with ProcessPoolExecutor(args.workers or performance_cores()) as ex:
        for r in ex.map(run_case, sorted(tasks, key=lambda t: -t["Lg"])):
            res.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    fn = os.path.join(out, f"results{args.tag_suffix}.json")
    old = json.load(open(fn)) if os.path.exists(fn) else []
    keep = [o for o in old if (o["kind"], o["Lg"], o["tag"]) not in {(r["kind"], r["Lg"], r["tag"]) for r in res}]
    json.dump(keep + res, open(fn, "w"), indent=1, default=float)
    print(f"total {time.perf_counter() - t0:.0f} s -> {fn}")


if __name__ == "__main__":
    main()
