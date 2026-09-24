"""2D NMOS band-to-band / trap-assisted tunneling leakage: GIDL and drain-body
junction tunneling (configs/input_mosfet_2d_btbt.yaml).

Two sweeps at Vds = 1 V separate the two leakage components:
  Id-Vg   (Vsub = 0, Vg from +1.5 down to -2 V): GIDL. Lowering Vg raises
          Vdg and bends the bands at the drain surface under the gate edge.
  Id-Vsub (Vg = 0, Vsub from 0 down to -3 V): drain-body junction
          tunneling. The reverse body bias raises Vdb across the whole
          junction, independent of the gate.

Each sweep is run with four models (btbt/):
  none       SRH only (the solver as shipped)
  local_1d   the 1D local models exactly as shipped in tat/: Kane with its
             F_sat = 9e5 V/cm cap, plus Hurkx trap-assisted tunneling
  local      the same local Kane + Hurkx, Kane uncapped
  nonlocal   Kane along complete Eg field-line paths, plus Hurkx with Gamma
             from the half-gap field-line paths (btbt/paths2d.py)

The terminal currents come from the solver. The generation is also
integrated by mechanism (Kane band-to-band vs Hurkx trap-assisted) and by
location on the drain side: SURFACE (the path, or the node, within
SURFACE_DEPTH of the Si/SiO2 interface - the gate-controlled GIDL region)
vs BULK (the drain-body junction). That split shows directly which
component each sweep turns on.

Usage: python3 -m btbt.main_btbt2d_sweep [config] [--workers N]
"""
import argparse
import csv
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm

from main2d_mosfet_sweep import build_from_config, gate_bc, continuation
from mesh2d.mesh2d import build_mesh2d
from solver2d.current import contact_current
from solver2d.newton_solver_qf_2d import newton_solve_2d, _mesh_semi_geometry, _mesh_ni_edge_g
from tat.tat import KaneBTBTModel, HurkxTATModel
from btbt.geom2d import TriGeom
from btbt.local2d import LocalTunneling2D
from btbt.paths2d import NonlocalTunneling2D, integrate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "configs", "input_mosfet_2d_btbt.yaml")
SURFACE_DEPTH = 2.0e-6          # cm (20 nm)
OUTER_TOL = 1e-2                # nonlocal outer (lagged) iteration: relative change in total generation
OUTER_ABS = 1e-19               # A/um - changes below this are converged regardless
OUTER_MAX = 8
MODELS = ("none", "local_1d", "local", "nonlocal")
LABELS = {"none": "SRH only (no tunneling)",
          "local_1d": "local Kane(F_sat cap)+Hurkx - 1D model as shipped",
          "local": "local Kane+Hurkx, uncapped",
          "nonlocal": "nonlocal field-line Kane+Hurkx"}
COLORS = {"none": "#6c757d", "local_1d": "#e8590c", "local": "#d6336c", "nonlocal": "#1f6feb"}
COMP_STYLE = {"kane_surface": ("#1f6feb", "-", "Kane BTBT, surface (GIDL)"),
              "tat_surface": ("#1f6feb", "--", "trap-assisted, surface (GIDL)"),
              "kane_bulk": ("#2f9e44", "-", "Kane BTBT, bulk junction"),
              "tat_bulk": ("#2f9e44", "--", "trap-assisted, bulk junction")}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


# --- worker side -------------------------------------------------------------

_CTX = {}


def _context(config_path):
    if config_path not in _CTX:
        cfg = load_config(config_path)
        domain, mat, dev, Cs, mesh_opts, _, _ = build_from_config(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = build_mesh2d(domain, mat=mat, **mesh_opts)
        vsat = bool((cfg.get("physics") or {}).get("velocity_saturation", False))
        geom = TriGeom(mesh, mat)
        _, cv_semi, _ = _mesh_semi_geometry(mesh)
        ni_arr, _ = _mesh_ni_edge_g(mesh, mat)
        gate = next(c for c in domain.contacts if c.name == "gate")
        x_mid = 0.5 * (gate.x_range_cm[0] + gate.x_range_cm[1])
        ctx = dict(cfg=cfg, mesh=mesh, mat=mat, dev=dev, Cs=Cs, vsat=vsat, geom=geom, cv=cv_semi, ni=ni_arr,
                   x_mid=x_mid)
        ctx["eq"] = _solve(ctx, None, 0.0, 0.0, 0.0, None)
        _CTX[config_path] = ctx
    return _CTX[config_path]


def make_model(ctx, name):
    kane = KaneBTBTModel.si_kane_quadratic()
    kane_uncapped = KaneBTBTModel(A=kane.A, B=kane.B, P=kane.P, F_sat_V_cm=np.inf)
    hurkx = HurkxTATModel()
    if name == "none":
        return None
    if name == "local_1d":
        return LocalTunneling2D(ctx["geom"], ctx["mat"], ctx["ni"], kane=kane, hurkx=hurkx)
    if name == "local":
        return LocalTunneling2D(ctx["geom"], ctx["mat"], ctx["ni"], kane=kane_uncapped, hurkx=hurkx)
    if name == "nonlocal":
        return NonlocalTunneling2D(ctx["geom"], ctx["mat"], ctx["ni"], ctx["cv"], kane=kane_uncapped, hurkx=hurkx,
                                   surface_depth_cm=SURFACE_DEPTH, x_mid_cm=ctx["x_mid"])
    raise ValueError(name)


def _newton(ctx, model, Vg, Vds, Vsub, init):
    psi_bc, phin_bc, phip_bc = gate_bc(ctx["dev"], ctx["mat"], ctx["Cs"], Vg)
    kw = {} if init is None else dict(psi_init=init["psi"], phin_init=init["phin"], phip_init=init["phip"],
                                      cold_retry=False)
    return newton_solve_2d(ctx["mesh"], ctx["mat"], {"source": 0.0, "drain": Vds, "gate": Vg, "body": Vsub},
                           psi_bc_override={"gate": psi_bc}, phin_bc_override={"gate": phin_bc},
                           phip_bc_override={"gate": phip_bc}, maxiter=40,
                           velocity_saturation=ctx["vsat"], generation=model, **kw)


def terminal(ctx, r):
    m, mat = ctx["mesh"], ctx["mat"]
    return {k: contact_current(m, mat, r, k) * 1e-4 for k in ("drain", "body", "source")}


_FROZEN = ("Gn_k", "Gp_k", "Gam_n", "Gam_p", "kane_info")


def _gen_totals(ctx, model, r):
    """q x total Kane pair generation and trap-assisted generation, A/um."""
    cv = ctx["cv"]
    return (integrate(model.Gp_k, cv, slice(None)),
            integrate(model.tat_rate(r["psi"], r["phin"], r["phip"]), cv, slice(None)))


def _solve(ctx, model, Vg, Vds, Vsub, init):
    """One bias point. The nonlocal model runs the lagged outer iteration:
    Newton with frozen paths -> retrace the paths at the new solution ->
    if the total Kane and trap-assisted generation changed by more than
    OUTER_TOL, Newton again with the new paths, and so on. When the
    retraced source matches the frozen one the solution is already
    self-consistent, so no extra Newton solve is spent confirming it (at a
    typical continuation step the previous bias point's paths are already
    within tolerance)."""
    if not isinstance(model, NonlocalTunneling2D):
        r = _newton(ctx, model, Vg, Vds, Vsub, init)
        r["outer"] = 0
        return r
    # Paths traced on the continuation's (secant-predicted) initial guess -
    # usually already within OUTER_TOL of the converged ones, so most points
    # need a single Newton solve.
    src = init if init is not None else ctx["eq"]
    model.prepare(src["psi"], src["phin"], src["phip"])
    r = _newton(ctx, model, Vg, Vds, Vsub, init)
    iters = r["iters"]
    k = 0
    for k in range(1, OUTER_MAX + 1):
        if r["res_norm"] > 1e-4:
            break
        old = _gen_totals(ctx, model, r)
        model.prepare(r["psi"], r["phin"], r["phip"])
        new = _gen_totals(ctx, model, r)
        if all(abs(a - b) <= max(OUTER_TOL * abs(b), OUTER_ABS) for a, b in zip(old, new)):
            break
        r = _newton(ctx, model, Vg, Vds, Vsub, r)
        iters += r["iters"]
    r["iters"] = iters
    r["outer"] = k
    r["frozen"] = {key: getattr(model, key) for key in _FROZEN}
    return r


class _Scaled:
    """model's generation times lam (and its Jacobian)."""
    def __init__(self, model):
        self.model, self.lam = model, 0.0

    def __call__(self, psi, phin, phip, jacobian=False):
        Gn, Gp, dn, dp = self.model(psi, phin, phip, jacobian)
        s = self.lam
        return s * Gn, s * Gp, None if dn is None else s * dn, None if dp is None else s * dp


def turn_on(ctx, model, log=None):
    """Equilibrium with a LOCAL model switched on gradually (strength
    10^-9 -> 1, continuation in log10 of the strength). The local Kane model
    generates carriers even in equilibrium wherever the field is high (the
    n+/p 1e18 junction corner), and switching it on in one jump from the
    no-tunneling equilibrium fails Newton's first bias step."""
    sc = _Scaled(model)
    sc.lam = 1e-9

    def at(p, g):
        sc.lam = 10.0 ** p
        return _newton(ctx, sc, 0.0, 0.0, 0.0, g)
    r = _newton(ctx, sc, 0.0, 0.0, 0.0, ctx["eq"])
    out, _, _ = continuation(at, -9.0, r, [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0], log=log)
    r = out.get(0.0) or r
    r["outer"] = 0
    return r


def components(ctx, model, r):
    """Drain-side generation integrated by mechanism and location, A/um."""
    g, cv = ctx["geom"], ctx["cv"]
    P = g.points
    drain = P[:, 0] > ctx["x_mid"]
    surf = P[:, 1] < SURFACE_DEPTH
    out = dict(kane_surface=0.0, kane_bulk=0.0, tat_surface=0.0, tat_bulk=0.0)
    if model is None:
        return out
    if isinstance(model, LocalTunneling2D):
        Gk, Gt, _ = model.parts(r["psi"], r["phin"], r["phip"])
        out["kane_surface"] = integrate(Gk, cv, drain & surf)
        out["kane_bulk"] = integrate(Gk, cv, drain & ~surf)
    else:
        for key, v in r["frozen"].items():
            setattr(model, key, v)
        info = model.kane_info
        if info is not None and len(info["pair"]):
            from core.params import Q
            ds = info["drain_side"]
            out["kane_surface"] = float(Q * np.sum(info["pair"][ds & info["surface"]]) * 1e-4)
            out["kane_bulk"] = float(Q * np.sum(info["pair"][ds & ~info["surface"]]) * 1e-4)
        Gt = model.tat_rate(r["psi"], r["phin"], r["phip"])
    out["tat_surface"] = integrate(Gt, cv, drain & surf)
    out["tat_bulk"] = integrate(Gt, cv, drain & ~surf)
    return out


def _generation_maps(ctx, model, r):
    """Per-node maps for the figure: total hole generation (Kane + trap-
    assisted, cm^-3 s^-1), and for the nonlocal model the traced paths of
    the strongest Kane starts."""
    if model is None:
        return None
    if isinstance(model, LocalTunneling2D):
        Gk, Gt, F = model.parts(r["psi"], r["phin"], r["phip"])
        return dict(G=Gk + Gt, F=F)
    for key, v in r["frozen"].items():
        setattr(model, key, v)
    info = model.kane_info
    pick = []
    if info is not None and len(info["pair"]):
        P = ctx["geom"].points
        order = np.argsort(-info["pair"])
        for j in order:
            if info["pair"][j] < 1e-6 * info["pair"][order[0]] or len(pick) >= 24:
                break
            q = P[info["start"][j]]
            if all(np.hypot(*(q - P[s])) > 3e-7 for s in pick):
                pick.append(info["start"][j])
    Gt = model.tat_rate(r["psi"], r["phin"], r["phip"])
    G = model.Gp_k + Gt
    model.prepare(r["psi"], r["phin"], r["phip"], record=np.array(pick, dtype=int))
    info = model.kane_info or {}
    polys = info.get("poly", [])
    kane = {k: info[k] for k in ("start", "end", "pair", "l")} if "pair" in info else None
    return dict(G=G, Ge=model.Gn_k + Gt, polys=polys, kane=kane)


def run_curve(task):
    t0 = time.perf_counter()
    ctx = _context(task["config"])
    model = make_model(ctx, task["model"])
    Vds = task["Vds"]
    log = []

    def at(which, Vg=0.0, Vd=Vds, Vs=0.0):
        def f(p, g):
            args = dict(Vg=Vg, Vd=Vd, Vs=Vs)
            args[which] = p
            if "frozen" in holder:
                g = dict(g, frozen=holder["frozen"])
            r = _solve(ctx, model, args["Vg"], args["Vd"], args["Vs"], g)
            if r["res_norm"] < 1e-4 and "frozen" in r:
                holder["frozen"] = r["frozen"]
            return r
        return f

    holder = {}
    # equilibrium (tunneling on) -> drain ramp at Vg = 0
    r0 = turn_on(ctx, model, log.append) if isinstance(model, LocalTunneling2D) else \
        _solve(ctx, model, 0.0, 0.0, 0.0, ctx["eq"])
    if "frozen" in r0:
        holder["frozen"] = r0["frozen"]
    ramp, it_tot, _ = continuation(at("Vd"), 0.0, r0, [Vds], log=log.append)
    if Vds not in ramp:
        return dict(task=task, points={}, log=log, time_s=time.perf_counter() - t0, maps={})
    base = ramp[Vds]
    vals = np.asarray(task["values"])
    if task["kind"] == "idvg":
        up, i1, _ = continuation(at("Vg"), 0.0, base, sorted(v for v in vals if v > 0), log=log.append)
        holder.pop("frozen", None)
        if "frozen" in base:
            holder["frozen"] = base["frozen"]
        dn, i2, _ = continuation(at("Vg"), 0.0, base, sorted((v for v in vals if v < 0), reverse=True),
                                 log=log.append)
        states = {0.0: base, **up, **dn}
    else:
        dn, i2, _ = continuation(at("Vs"), 0.0, base, sorted((v for v in vals if v < 0), reverse=True),
                                 log=log.append)
        states = {0.0: base, **dn}
    wanted = set(map(float, vals)) | {0.0}
    points = {}
    for p, r in sorted(states.items()):
        if p not in wanted:
            continue
        points[p] = dict(**terminal(ctx, r), **components(ctx, model, r), iters=r["iters"], outer=r["outer"],
                         res_norm=r["res_norm"])
    maps = {}
    for p_map in task.get("map_at", []):
        if p_map in states:
            r = states[p_map]
            maps[p_map] = dict(bias=p_map, psi=r["psi"], phin=r["phin"], phip=r["phip"],
                               gen=_generation_maps(ctx, model, r))
    return dict(task=task, points=points, log=log, time_s=time.perf_counter() - t0, maps=maps)


# --- plotting ------------------------------------------------------------------

def _curve(res):
    pts = res["points"]
    v = np.array(sorted(pts))
    return v, {k: np.array([pts[x][k] for x in v]) for k in next(iter(pts.values()))}


def plot_sweep(results, kind, xlabel, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    ax = axes[0]
    for res in results:
        m = res["task"]["model"]
        if not res["points"]:
            continue
        v, c = _curve(res)
        ax.semilogy(v, np.abs(c["drain"]), "o-", ms=3, color=COLORS[m], label=f"Id: {LABELS[m]}")
        if m == "nonlocal":
            ax.semilogy(v, np.abs(c["body"]), ":", color=COLORS[m], label="|Ib|: nonlocal")
    ax.set_xlabel(xlabel); ax.set_ylabel("|I| (A/um)"); ax.set_title(title)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5, loc="best")
    for ax, m in zip(axes[1:], ("nonlocal", "local")):
        res = next((r for r in results if r["task"]["model"] == m and r["points"]), None)
        if res is None:
            continue
        v, c = _curve(res)
        ax.semilogy(v, np.abs(c["drain"] - _baseline(results, v)), "k-", lw=2.2, alpha=0.35,
                    label="Id - Id(SRH only)")
        for comp, (col, ls, lbl) in COMP_STYLE.items():
            pos = np.where(c[comp] > 0, c[comp], np.nan)
            neg = np.where(c[comp] < 0, -c[comp], np.nan)
            ax.semilogy(v, pos, ls, color=col, label=lbl)
            if np.any(np.isfinite(neg)):
                ax.semilogy(v, neg, ls, color=col, marker="x", ms=5, lw=0.8,
                            label=lbl + ": NEGATIVE (net trap recombination)")
        ax.set_xlabel(xlabel); ax.set_ylabel("q x integrated generation, drain side (A/um)")
        ax.set_title(f"{LABELS[m]}: components")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
        lo = max(1e-22, np.nanmin(np.abs(c["drain"])) * 1e-4)
        ax.set_ylim(lo, None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)


def _baseline(results, v):
    res = next(r for r in results if r["task"]["model"] == "none")
    vb, cb = _curve(res)
    return np.interp(v, vb, cb["drain"])


def plot_maps(ctx, results, out_path):
    mesh = ctx["mesh"]
    P = mesh.points * 1e4
    T = ctx["geom"].triangles[ctx["geom"].tri_semi]
    cases = [(r["task"]["kind"], r) for r in results if r["maps"] and r["task"]["model"] in ("local", "nonlocal")]
    rows = [("idvg", "Vg = {b:g} V, Vsub = 0 (GIDL)"), ("idvsub", "Vg = 0, Vsub = {b:g} V (junction)")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    norm = LogNorm(1e16, 1e26)
    for i, (kind, ttl) in enumerate(rows):
        for j, m in enumerate(("local", "nonlocal")):
            ax = axes[i, j]
            res = next((r for k, r in cases if k == kind and r["task"]["model"] == m), None)
            if res is None:
                ax.set_visible(False)
                continue
            mp = res["maps"][min(res["maps"])]
            gen = mp["gen"]
            G = np.clip(gen["G"], 1e10, None)
            tp = ax.tripcolor(P[:, 0], P[:, 1], T, G, norm=norm, cmap="magma", shading="gouraud")
            ax.tricontour(P[:, 0], P[:, 1], ctx["geom"].triangles, mesh.Cdop, levels=[0.0], colors="w",
                          linewidths=0.8)
            ax.tricontour(P[:, 0], P[:, 1], T, mp["psi"], levels=12, colors="c", linewidths=0.4,
                          alpha=0.6)
            for poly in gen.get("polys", []):
                if len(poly) > 1:
                    q = poly * 1e4
                    ax.plot(q[:, 0], q[:, 1], "-", color="#7CFC00", lw=1.0)
                    ax.plot(q[0, 0], q[0, 1], "o", color="#7CFC00", ms=2.5)
                    ax.plot(q[-1, 0], q[-1, 1], "s", color="#00FFFF", ms=2.5)
            ax.axhspan(-0.003, 0.0, xmin=0, xmax=1, color="none")
            ax.fill_between([0.3, 0.8], -0.012, 0.0, color="#bbbbbb")
            ax.text(0.62, -0.006, "gate / oxide", fontsize=8, va="center")
            ax.set_xlim(0.6, 1.1); ax.set_ylim(0.25, -0.015)
            ax.set_aspect("equal")
            b = mp["bias"]
            what = "Kane + trap-assisted hole generation" if m == "nonlocal" else "Kane + trap-assisted generation"
            ax.set_title(f"{LABELS[m]}\n{ttl.format(b=b)}: {what}", fontsize=9)
            ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
            fig.colorbar(tp, ax=ax, label="G (cm^-3 s^-1)", shrink=0.8)
    fig.text(0.5, 0.005, "white: metallurgical junction; cyan: equipotentials; green: traced Kane tunneling "
             "paths (dot = hole start, square = electron end)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=140); plt.close(fig)


def _cut(ctx, mp, x_um, y_um):
    """psi, phin, phip along a vertical cut x = x_um through the silicon."""
    g = ctx["geom"]
    pts = np.column_stack([np.full_like(y_um, x_um), y_um]) * 1e-4
    tri = g.locate(pts, np.full(len(pts), -1))
    ok = (tri >= 0)
    ok[ok] &= g.tri_semi[tri[ok]]
    out = {k: np.full(len(pts), np.nan) for k in ("psi", "phin", "phip")}
    for k in out:
        out[k][ok] = g.interp(mp[k], pts[ok], tri[ok])
    return out


def plot_band_figure(ctx, res, cut_x_um, xlim, ylim, cut_y, title, bias_label, out_path):
    """One row per saved bias: (1) |E| with electric field lines and the
    traced Kane tunneling paths, (2) hole generation (Kane + trap-assisted),
    (3) the band diagram along the vertical cut x = cut_x_um with the
    strongest Kane tunneling transition near the cut drawn as a horizontal
    arrow from the valence band (hole left behind) to the conduction band
    (electron arrives)."""
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    g, mesh, mat = ctx["geom"], ctx["mesh"], ctx["mat"]
    Eg = mat.Eg_eV
    P = g.points * 1e4
    Ts = g.triangles[g.tri_semi]
    tri_s = Triangulation(P[:, 0], P[:, 1], g.triangles, mask=~g.tri_semi)
    biases = sorted(res["maps"], reverse=True)
    aspect = (xlim[1] - xlim[0]) / abs(ylim[0] - ylim[1])
    fig, axes = plt.subplots(len(biases), 3, figsize=(7.5 * 2 + 6, 7.0 * 2 / aspect * len(biases) / 2 + 1.8),
                             squeeze=False, gridspec_kw=dict(width_ratios=[1.25, 1.25, 1.0]))
    xs = np.linspace(*xlim, 220)
    ys = np.linspace(min(ylim), max(ylim), 160)
    X, Y = np.meshgrid(xs, ys)
    for row, b in zip(axes, biases):
        mp = res["maps"][b]
        gen = mp["gen"]
        gn = g.node_grad(mp["psi"])                           # V/cm; E = -grad psi
        Emag = np.hypot(gn[:, 0], gn[:, 1])
        ax = row[0]
        tp = ax.tripcolor(P[:, 0], P[:, 1], Ts, np.clip(Emag, 1e4, None), norm=LogNorm(1e4, 3e6),
                          cmap="viridis", shading="gouraud")
        Ex = LinearTriInterpolator(tri_s, -gn[:, 0])(X, Y)
        Ey = LinearTriInterpolator(tri_s, -gn[:, 1])(X, Y)
        ax.streamplot(xs, ys, Ex.filled(np.nan), Ey.filled(np.nan), color="w", linewidth=0.5, density=1.3,
                      arrowsize=0.6)
        fig.colorbar(tp, ax=ax, label="|E| (V/cm)", shrink=0.8, pad=0.02)
        ax2 = row[1]
        tp2 = ax2.tripcolor(P[:, 0], P[:, 1], Ts, np.clip(gen["G"], 1e10, None), norm=LogNorm(1e16, 1e26),
                            cmap="magma", shading="gouraud")
        fig.colorbar(tp2, ax=ax2, label="G, holes (cm^-3 s^-1)", shrink=0.8, pad=0.02)
        for a in (ax, ax2):
            a.tricontour(P[:, 0], P[:, 1], g.triangles, mesh.Cdop, levels=[0.0], colors="#ff5555",
                         linewidths=1.0)
            for poly in gen.get("polys", []):
                if len(poly) > 1:
                    q = poly * 1e4
                    a.plot(q[:, 0], q[:, 1], "-", color="#7CFC00", lw=1.2)
                    a.plot(q[0, 0], q[0, 1], "o", color="#7CFC00", ms=3)
                    a.plot(q[-1, 0], q[-1, 1], "s", color="#00FFFF", ms=3)
            a.axvline(cut_x_um, color="#ffcc00", ls="--", lw=1)
            a.fill_between([0.3, 0.8], min(ylim) - 1, 0.0, color="#bbbbbb")
            a.set_xlim(*xlim); a.set_ylim(*ylim); a.set_aspect("equal")
            a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
        ax.set_title(f"{bias_label.format(b=b)}: |E| and electric field lines", fontsize=10)
        ax2.set_title(f"{bias_label.format(b=b)}: hole generation (Kane + trap-assisted)", fontsize=10)
        # band diagram
        ax3 = row[2]
        yc = np.linspace(*cut_y, 600)
        c = _cut(ctx, mp, cut_x_um, yc)
        Ec, Ev = -c["psi"] + Eg / 2, -c["psi"] - Eg / 2
        ynm = yc * 1e3
        ax3.plot(ynm, Ec, "k-", lw=1.6, label="Ec")
        ax3.plot(ynm, Ev, "k-", lw=1.6, label="Ev")
        ax3.plot(ynm, -c["phin"], "b--", lw=1, label="Efn")
        ax3.plot(ynm, -c["phip"], "r--", lw=1, label="Efp")
        k = gen.get("kane")
        drawn = False
        if k is not None and len(k["pair"]):
            sx = g.points[k["start"], 0] * 1e4
            sy, ey = g.points[k["start"], 1] * 1e7, k["end"][:, 1] * 1e7
            lo, hi = cut_y[0] * 1e3, cut_y[1] * 1e3
            near = np.flatnonzero((np.abs(sx - cut_x_um) < 0.01) & (np.minimum(sy, ey) >= lo)
                                  & (np.maximum(sy, ey) <= hi))
            if len(near):
                j = near[np.argmax(k["pair"][near])]
                y0, y1 = sy[j], ey[j]
                E0 = -mp["psi"][k["start"][j]] - Eg / 2
                ax3.annotate("", xy=(y1, E0), xytext=(y0, E0),
                             arrowprops=dict(arrowstyle="->", color="#2f9e44", lw=2.2))
                ax3.text(0.5 * (y0 + y1), E0 + 0.08, f"band-to-band tunneling\nl = {k['l'][j] * 1e7:.1f} nm",
                         color="#2f9e44", ha="center", fontsize=8.5)
                drawn = True
        valid = np.isfinite(c["psi"])
        bend = np.nanmax(c["psi"][valid]) - np.nanmin(c["psi"][valid]) if valid.any() else np.nan
        ax3.text(0.02, 0.03, f"band bending along cut = {bend:.2f} V (Eg = {Eg:.2f})"
                 + ("" if drawn else "\nno Kane path near the cut: bands bend < Eg within reach"),
                 transform=ax3.transAxes, fontsize=8.5)
        ax3.set_xlim(cut_y[0] * 1e3, cut_y[1] * 1e3)
        ax3.set_xlabel(f"depth y (nm) along x = {cut_x_um:g} um"); ax3.set_ylabel("energy (eV)")
        ax3.set_title(f"{bias_label.format(b=b)}: band diagram along the cut", fontsize=9)
        ax3.grid(alpha=0.3); ax3.legend(fontsize=8, loc="upper right")
    fig.suptitle(title + "\ngreen lines: traced Kane tunneling paths (dot = hole left behind, square = electron "
                 "arrives); red: metallurgical junction; yellow dashed: band-diagram cut; grey: gate oxide",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


def _grid(start, stop, step):
    n = int(round(abs(stop - start) / step))
    return [round(start + np.sign(stop - start) * k * step, 6) for k in range(n + 1)]


def main():
    ap = argparse.ArgumentParser(description="2D NMOS GIDL / junction tunneling leakage")
    ap.add_argument("config", nargs="?", default=DEFAULT_PATH)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    cfg = load_config(args.config)
    b = cfg.get("btbt") or {}
    Vds = float(b.get("vds_V", 1.0))
    vgs = _grid(b.get("vgs_start_V", 1.0), b.get("vgs_stop_V", -2.0), b.get("vgs_step_V", 0.1))
    vsub = _grid(b.get("vsub_start_V", 0.0), b.get("vsub_stop_V", -3.0), b.get("vsub_step_V", 0.1))
    models = args.models.split(",")
    def maps_at(vals, m):
        vals = sorted(vals)
        if m != "nonlocal":
            return [vals[0]]
        return sorted({vals[0], vals[len(vals) // 3], vals[len(vals) // 2]})
    tasks = ([dict(kind="idvg", model=m, Vds=Vds, values=vgs, map_at=maps_at([v for v in vgs if v < 0], m),
                   config=args.config) for m in models] +
             [dict(kind="idvsub", model=m, Vds=Vds, values=vsub, map_at=maps_at([v for v in vsub if v < 0], m),
                   config=args.config) for m in models])
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        results = list(pool.map(run_curve, tasks))
    wall = time.perf_counter() - t0

    out_dir = os.path.join(ROOT, "out", "btbt", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    for res in results:
        t = res["task"]
        pts = res["points"]
        n_ok = sum(p["res_norm"] < 1e-4 for p in pts.values())
        outer = [p["outer"] for p in pts.values()]
        print(f"  {t['kind']:6s} {t['model']:9s}: {n_ok}/{len(t['values'])} points converged, "
              f"{sum(p['iters'] for p in pts.values())} Newton its"
              + (f", outer passes {min(outer)}-{max(outer)}" if outer and max(outer) else "")
              + f", {res['time_s']:.1f}s")
        for line in res["log"]:
            print("   ", line)
    print(f"all curves: {wall:.1f}s wall-clock ({len(tasks)} curves in parallel)")

    for kind, xlabel, fname, title in (
            ("idvg", "Vgs (V)", "idvg_btbt", f"Id-Vg, Vds = {Vds:g} V, Vsub = 0"),
            ("idvsub", "Vsub (V)", "idvsub_btbt", f"Id-Vsub, Vds = {Vds:g} V, Vg = 0")):
        rs = [r for r in results if r["task"]["kind"] == kind]
        with open(os.path.join(out_dir, fname + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            cols = ["drain", "body", "source", "kane_surface", "kane_bulk", "tat_surface", "tat_bulk", "iters",
                    "outer", "res_norm"]
            w.writerow(["model", "bias_V"] + [c + ("_A_per_um" if c in ("drain", "body", "source") or
                                                   c.startswith(("kane", "tat")) else "") for c in cols])
            for r in rs:
                for p, d in sorted(r["points"].items()):
                    w.writerow([r["task"]["model"], f"{p:.4f}"] + [f"{d[c]:.6e}" if isinstance(d[c], float) else d[c]
                                                                   for c in cols])
        plot_sweep(rs, kind, xlabel, title, os.path.join(out_dir, fname + ".png"))
    ctx = _context(args.config)
    plot_maps(ctx, results, os.path.join(out_dir, "btbt_generation_maps.png"))
    nl = {r["task"]["kind"]: r for r in results if r["task"]["model"] == "nonlocal" and r["maps"]}
    if "idvg" in nl:
        plot_band_figure(ctx, nl["idvg"], 0.785, (0.70, 0.86), (0.06, -0.004), (0.0, 0.06),
                         f"GIDL: gate-drain overlap, Vds = {Vds:g} V, Vsub = 0 (nonlocal model)",
                         "Vg = {b:g} V", os.path.join(out_dir, "gidl_fields_bands.png"))
    if "idvsub" in nl:
        plot_band_figure(ctx, nl["idvsub"], 0.95, (0.72, 1.1), (0.24, 0.08), (0.10, 0.24),
                         f"Drain-body junction tunneling, Vds = {Vds:g} V, Vg = 0 (nonlocal model)",
                         "Vsub = {b:g} V", os.path.join(out_dir, "junction_fields_bands.png"))
    print(f"wrote {out_dir}/idvg_btbt.png, idvsub_btbt.png, btbt_generation_maps.png, "
          "gidl_fields_bands.png, junction_fields_bands.png (+ csv)")


if __name__ == "__main__":
    main()
