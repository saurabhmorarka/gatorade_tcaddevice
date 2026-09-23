"""2D planar NMOS transistor driver: same oxide+gate stack as the MOS
capacitor, now with n+ source/drain regions connected to their own ohmic
contacts, so a real channel current exists (modulated by the gate) -
requires the full coupled Poisson+continuity solver
(solver2d/newton_solver_qf_2d.py, the same one the 2D diode uses), NOT the
MOS capacitor's Poisson-only solver.

Produces the two standard MOSFET characterization curves plus the usual
figures of merit (solver2d/mosfet_metrics.py):
  Id-Vg (transfer): at a low and a high Vds, linear + log axes, annotated
      with SS, Vt (constant-current and max-gm), DIBL, Ion/Ioff.
  Id-Vd (output): a family of fixed Vgs.

Speed: every curve is an independent continuation (equilibrium anchor ->
bias ramp -> sweep) run in its own worker process, and each continuation
step starts from a secant (two-point extrapolation) predictor, so a bias
point typically costs ~3-5 Newton iterations; a step that fails is
retried at half the bias step instead of cold-restarting.

The gate contact sits on the oxide (an insulator node, ni=0) so its
Dirichlet BC can't use the generic ohmic mass-action relation
newton_solve_2d applies to every other contact by default - this driver
computes it explicitly (mos.mos_analytic.flatband_voltage, same formula
the MOS capacitor's solver uses) and passes it through newton_solve_2d's
psi_bc_override/phin_bc_override/phip_bc_override.

Usage: python3 main2d_mosfet_sweep.py [configs/input_mosfet_2d.yaml]
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

from core import structure_io as sio
from core.config import _resolve_material_block
from core.field_save import resolve_save_points
from core.physics import equilibrium_bulk_potential
from mesh2d.config2d import build_domain_from_config
from mesh2d.mesh2d import build_mesh2d
from mos import mos_analytic as man
from mos.mos_params import MOSDevice
from solver2d.current import contact_current
from solver2d.mosfet_metrics import mosfet_metrics
from solver2d.efield2d import electric_field_2d
from solver2d.newton_solver_qf_2d import newton_solve_2d

_CM_TO_UM = 1.0e4
_UM_TO_CM = 1.0e-4

DEFAULT_PATH = os.path.join("configs", "input_mosfet_2d.yaml")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_from_config(cfg):
    domain = build_domain_from_config(cfg)
    material = _resolve_material_block(cfg.get("material") or {})

    substrate = cfg["geometry"]["substrate"]
    Na = float(substrate["concentration_cm3"])
    Cdop_substrate = -Na if substrate["doping_type"] == "p" else Na

    oxide_region = next(r for r in domain.regions if r.kind == "insulator")
    t_ox_cm = oxide_region.y_range_cm[1] - oxide_region.y_range_cm[0]
    dev = MOSDevice(Na=Na, t_ox=t_ox_cm, eps_ox_r=oxide_region.eps_r,
                     gate_workfunction_eV=(cfg.get("gate") or {}).get("workfunction_eV"))

    mesh_cfg = cfg.get("mesh") or {}
    mesh_opts = dict(
        h_min_cm=float(mesh_cfg.get("h_min_um", 0.2)) * _UM_TO_CM,
        h_max_cm=float(mesh_cfg.get("h_max_um", 2.0)) * _UM_TO_CM,
        growth=float(mesh_cfg.get("growth", 1.3)),
        min_angle_deg=int(mesh_cfg.get("min_angle_deg", 32)),
        mesh_style=mesh_cfg.get("style", "unstructured"),
        refine_boxes=tuple(
            (b["x_range_um"][0] * _UM_TO_CM, b["x_range_um"][1] * _UM_TO_CM,
             b["y_range_um"][0] * _UM_TO_CM, b["y_range_um"][1] * _UM_TO_CM, b["h_um"] * _UM_TO_CM)
            for b in mesh_cfg.get("refine_boxes") or []),
        interface_h_cm=float(mesh_cfg["interface_h_um"]) * _UM_TO_CM if "interface_h_um" in mesh_cfg else None,
        junction_h_cm=float(mesh_cfg["junction_h_um"]) * _UM_TO_CM if "junction_h_um" in mesh_cfg else None,
    )

    bias_cfg = cfg.get("bias") or {}
    output_cfg = cfg.get("output") or {}
    return domain, material, dev, Cdop_substrate, mesh_opts, bias_cfg, output_cfg


def gate_bc(dev, mat, Cdop_substrate, Vgs):
    """(psi_bc, phin_bc, phip_bc) for the gate contact - an ideal metal
    sitting directly on the oxide (see mos/mos_solver.py's Cdop_gate=None
    convention). phin/phip are moot where ni=0 (the whole oxide, the gate
    node included) but still need a finite value for the Dirichlet row -
    Vgs, matching the "VG on the insulator side" convention used
    throughout the 1D and 2D MOS capacitor solvers."""
    psi_bulk = equilibrium_bulk_potential(mat, Cdop_substrate)
    V_FB = man.flatband_voltage(dev, mat, Cdop_substrate)
    psi_bc = psi_bulk + (Vgs - V_FB)
    return psi_bc, Vgs, Vgs



SWEEP_MAXITER = 40      # per continuation step (a failed step is re-tried at half the bias step)
MAX_HALVINGS = 8
FAST_ITERS = 4          # a step converging this fast doubles the next step
H_INIT = 0.1            # V - first continuation step (a full jump to a far target just fails)


def solve_mosfet(mesh, mat, dev, Cdop_substrate, Vgs, Vds, psi_init=None, phin_init=None, phip_init=None,
                  verbose=False, maxiter=SWEEP_MAXITER, cold_retry=True, velocity_saturation=False):
    psi_bc, phin_bc, phip_bc = gate_bc(dev, mat, Cdop_substrate, Vgs)
    bias_by_contact = {"source": 0.0, "drain": Vds, "gate": Vgs, "body": 0.0}
    return newton_solve_2d(
        mesh, mat, bias_by_contact,
        psi_init=psi_init, phin_init=phin_init, phip_init=phip_init,
        psi_bc_override={"gate": psi_bc}, phin_bc_override={"gate": phin_bc},
        phip_bc_override={"gate": phip_bc}, verbose=verbose, maxiter=maxiter,
        cold_retry=cold_retry, velocity_saturation=velocity_saturation,
    )


def _converged(r):
    return r["res_norm"] < 1e-4


_FIELDS = ("psi", "phin", "phip")


def _predict(prev, cur, p_next):
    """Secant predictor: extrapolate (psi, phin, phip) linearly in the bias
    from the last two converged points. Only the change is extrapolated,
    capped at 0.5 V per node so a quasi-Fermi level that is numerically
    undetermined (a carrier with ~0 density, e.g. phip in an n+ region)
    can't be flung somewhere absurd."""
    if prev is None:
        return {k: cur[1][k] for k in _FIELDS}
    (p0, s0), (p1, s1) = prev, cur
    a = (p_next - p1) / (p1 - p0)
    return {k: s1[k] + np.clip(a * (s1[k] - s0[k]), -0.5, 0.5) for k in _FIELDS}


def continuation(solve_at, p0, s0, targets, prev=None, log=None):
    """Walk the bias parameter from p0 (converged state s0) through each
    value in `targets` (in order), with a secant predictor and adaptive
    sub-stepping: a step that fails is retried at half size (up to
    MAX_HALVINGS times), a step that converges in <= FAST_ITERS iterations
    doubles the next step (never overshooting the next requested target).
    Returns ({target: state}, n_newton_iterations, last (prev, cur) pair);
    stops early (keeping what converged) if a step can't be completed."""
    out = {}
    cur = (p0, s0)
    total_it = 0
    h = H_INIT
    for tgt in targets:
        while abs(tgt - cur[0]) > 1e-12:
            remaining = tgt - cur[0]
            h = np.sign(remaining) * min(abs(h), abs(remaining))
            for _ in range(MAX_HALVINGS + 1):
                p_next = cur[0] + h
                r = solve_at(p_next, _predict(prev, cur, p_next))
                total_it += r["iters"]
                if _converged(r):
                    break
                h *= 0.5
            else:
                if log:
                    log(f"    step to {p_next:+.4f} failed after {MAX_HALVINGS} halvings - curve stopped")
                return out, total_it, (prev, cur)
            prev, cur = cur, (p_next, r)
            if r["iters"] <= FAST_ITERS:
                h *= 2.0
        out[float(tgt)] = cur[1]
    return out, total_it, (prev, cur)


# --- worker side (one process per curve) -----------------------------------

_CTX = {}


def _context(config_path):
    if config_path not in _CTX:
        cfg = load_config(config_path)
        domain, mat, dev, Cs, mesh_opts, bias_cfg, output_cfg = build_from_config(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = build_mesh2d(domain, mat=mat, **mesh_opts)
        vsat = bool((cfg.get("physics") or {}).get("velocity_saturation", False))
        eq = solve_mosfet(mesh, mat, dev, Cs, 0.0, 0.0, velocity_saturation=vsat)
        _CTX[config_path] = (mesh, mat, dev, Cs, eq, vsat)
    return _CTX[config_path]


def _id_per_um(mesh, mat, r):
    return contact_current(mesh, mat, r, "drain") * 1e-4   # A/cm of depth -> A/um of width


def run_curve(task):
    """One Id-Vg (task['kind']=='idvg', fixed task['Vds']) or Id-Vd
    ('idvd', fixed task['Vgs']) curve, from the equilibrium anchor."""
    t0 = time.perf_counter()
    mesh, mat, dev, Cs, eq, vsat = _context(task["config"])
    log_lines = []
    log = log_lines.append

    def at(Vgs, Vds):
        return lambda p, g: solve_mosfet(mesh, mat, dev, Cs, *((p, Vds) if Vgs is None else (Vgs, p)),
                                          psi_init=g["psi"], phin_init=g["phin"], phip_init=g["phip"],
                                          cold_retry=False, velocity_saturation=vsat)
    total_it = 0
    if task["kind"] == "idvg":
        Vds = task["Vds"]
        # ramp the drain at Vgs=0 (off, the gentlest place to raise Vds)
        ramp, it, _ = continuation(at(0.0, None), 0.0, eq, [Vds], log=log)
        total_it += it
        if Vds not in ramp:
            return dict(task=task, points={}, newton_iters=total_it, log=log_lines,
                        time_s=time.perf_counter() - t0, last_state=None, last_bias=None)
        base = ramp[Vds]
        vgs = np.asarray(task["values"])
        up, it, _ = continuation(at(None, Vds), 0.0, base, sorted(v for v in vgs if v > 0), log=log)
        total_it += it
        down, it, _ = continuation(at(None, Vds), 0.0, base, sorted((v for v in vgs if v < 0), reverse=True), log=log)
        total_it += it
        states = {0.0: base, **up, **down}
    else:
        Vgs = task["Vgs"]
        ramp, it, _ = continuation(at(None, 0.0), 0.0, eq, [Vgs], log=log)
        total_it += it
        if Vgs not in ramp:
            return dict(task=task, points={}, newton_iters=total_it, log=log_lines,
                        time_s=time.perf_counter() - t0, last_state=None, last_bias=None)
        vds = sorted(v for v in task["values"] if v > 0)
        states, it, _ = continuation(at(Vgs, None), 0.0, ramp[Vgs], vds, log=log)
        total_it += it
        states[0.0] = ramp[Vgs]
    points = {p: dict(Id=_id_per_um(mesh, mat, r), iters=r["iters"], res_norm=r["res_norm"])
              for p, r in sorted(states.items()) if p in set(map(float, task["values"])) or p == 0.0}
    last = max(p for p in states if p in points)
    return dict(task=task, points=points, newton_iters=total_it, log=log_lines,
                time_s=time.perf_counter() - t0,
                last_state={k: states[last][k] for k in ("psi", "n", "p", "phin", "phip")}, last_bias=last)


# --- plotting ---------------------------------------------------------------

def plot_transfer(curves, m, out_path, L_um):
    lo, hi = curves
    fig, (axl, axg) = plt.subplots(1, 2, figsize=(14, 5.5))
    cols = {"lo": "#1f6fb4", "hi": "#d1492e"}
    for tag, c in (("lo", lo), ("hi", hi)):
        lbl = f"Vds = {c['Vds']:g} V"
        axl.plot(c["Vgs"], c["Id"] * 1e6, "o-", ms=3, color=cols[tag], label=lbl)
        axg.semilogy(c["Vgs"], c["Id"], "o-", ms=3, color=cols[tag], label=lbl)

    # linear axes: max-gm tangent -> Vt_lin
    vg = np.asarray(lo["Vgs"]); idl = np.asarray(lo["Id"])
    k = int(np.argmin(np.abs(vg - m["vgs_gm_max_lo"])))
    x = np.linspace(m["Vt_lin"] + 0.5 * lo["Vds"], vg.max(), 50)
    axl.plot(x, (idl[k] + m["gm_max_lo"] * (x - vg[k])) * 1e6, "--", color="#555", lw=1)
    axl.annotate(f"max-gm extrapolation\nVt,lin = {m['Vt_lin']:.3f} V\ngm,max = {m['gm_max_lo'] * 1e6:.1f} uS/um",
                 xy=(m["Vt_lin"] + 0.5 * lo["Vds"], 0), xytext=(0.02, 0.6), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9)
    axl.set_xlabel("Vgs (V)"); axl.set_ylabel("Id (uA/um)"); axl.set_title("Id-Vg, linear")
    axl.grid(alpha=0.3); axl.legend(loc="upper left")

    # log axes: SS window, constant-current Vt, DIBL shift, Ion/Ioff.
    # Labels sit in fixed free corners of the axes, arrows point at the
    # feature, so nothing overlaps the curves.
    axg.axhline(m["I_cc"], color="#777", lw=0.8, ls=":")
    axg.annotate(f"I_cc = 100 nA x W/L\n= {m['I_cc']:.1e} A/um", xy=(max(lo["Vgs"]), m["I_cc"]),
                 xytext=(0.80, 0.42), textcoords="axes fraction", fontsize=8, color="#555",
                 arrowprops=dict(arrowstyle="->", color="#999"))
    ss_pos = {"lo": (0.66, 0.18), "hi": (0.03, 0.62)}
    for tag, c in (("lo", lo), ("hi", hi)):
        a_, b_ = m[f"SS_{tag}_window"]
        ia, ib = np.interp([a_, b_], c["Vgs"], c["Id"])
        axg.plot([a_, b_], [ia, ib], "-", lw=5, alpha=0.35, color=cols[tag])
        axg.annotate(f"SS = {m[f'SS_{tag}']:.1f} mV/dec\n(Vds = {c['Vds']:g} V)",
                     xy=(0.5 * (a_ + b_), np.sqrt(ia * ib)), xytext=ss_pos[tag], textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color=cols[tag]), color=cols[tag], fontsize=9)
    vlo, vhi = m["Vt_cc_lo"], m["Vt_cc_hi"]
    axg.annotate("", xy=(vhi, m["I_cc"]), xytext=(vlo, m["I_cc"]),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=1.2, shrinkA=0, shrinkB=0))
    axg.annotate(f"DIBL = {m['DIBL_mV_per_V']:.0f} mV/V\nVt,cc: {vlo:.3f} V -> {vhi:.3f} V",
                 xy=(0.5 * (vlo + vhi), m["I_cc"]), xytext=(0.03, 0.80), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
    axg.annotate(f"Ion = {m['Ion'] * 1e6:.0f} uA/um", xy=(max(hi["Vgs"]), m["Ion"]),
                 xytext=(0.55, 0.93), textcoords="axes fraction", arrowprops=dict(arrowstyle="->"), fontsize=9)
    axg.annotate(f"Ioff = {m['Ioff']:.1e} A/um\nIon/Ioff = {m['Ion_Ioff']:.1e}", xy=(0.0, m["Ioff"]),
                 xytext=(0.03, 0.35), textcoords="axes fraction", arrowprops=dict(arrowstyle="->"), fontsize=9)
    axg.set_xlabel("Vgs (V)"); axg.set_ylabel("Id (A/um)"); axg.set_title("Id-Vg, log")
    axg.grid(alpha=0.3, which="both"); axg.legend(loc="lower right")
    fig.suptitle(f"2D NMOS, Lg = {L_um:g} um", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_output(out_curves, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(out_curves):
        ax.plot(c["Vds"], c["Id"] * 1e6, "o-", ms=3, color=cmap(i / max(1, len(out_curves) - 1)),
                label=f"Vgs = {c['Vgs']:g} V")
    ax.set_xlabel("Vds (V)"); ax.set_ylabel("Id (uA/um)"); ax.set_title("Id-Vd (output)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)


def _grid(start, stop, step):
    return [round(float(v), 6) for v in np.arange(start, stop + 0.5 * step, step)]


def main():
    parser = argparse.ArgumentParser(description="gatorade_tcaddevice 2D planar NMOS driver")
    parser.add_argument("config", nargs="?", default=DEFAULT_PATH)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    domain, mat, dev, Cdop_substrate, mesh_opts, bias_cfg, output_cfg = build_from_config(cfg)
    gate = next(c for c in domain.contacts if c.name == "gate")
    L_um = (gate.x_range_cm[1] - gate.x_range_cm[0]) * _CM_TO_UM

    vgs_list = _grid(bias_cfg.get("vgs_start_V", -0.3), bias_cfg.get("vgs_stop_V", 1.5),
                     bias_cfg.get("vgs_step_V", 0.05))
    vds_transfer = [float(v) for v in bias_cfg.get("vds_transfer_V", [0.05, 1.0])]
    vds_list = _grid(bias_cfg.get("vds_start_V", 0.0), bias_cfg.get("vds_stop_V", 1.5),
                     bias_cfg.get("vds_step_V", 0.1))
    vgs_output = [float(v) for v in bias_cfg.get("vgs_for_output_V", [0.9, 1.1, 1.3, 1.5])]

    tasks = ([dict(kind="idvg", Vds=v, values=vgs_list, config=args.config) for v in vds_transfer] +
             [dict(kind="idvd", Vgs=v, values=vds_list, config=args.config) for v in vgs_output])

    # One BLAS thread per worker - the workers are the parallelism.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        results = list(pool.map(run_curve, tasks))
    wall = time.perf_counter() - t0

    out_dir = os.path.join("out", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    for res in results:
        t = res["task"]
        name = f"Id-Vg @ Vds={t['Vds']:g}V" if t["kind"] == "idvg" else f"Id-Vd @ Vgs={t['Vgs']:g}V"
        n_ok = sum(_converged(p) for p in res["points"].values())
        print(f"  {name:22s}: {n_ok}/{len(res['points'])} points converged, "
              f"{res['newton_iters']} Newton iterations, {res['time_s']:.1f}s")
        for line in res["log"]:
            print(line)
    vsat_on = bool((cfg.get("physics") or {}).get("velocity_saturation", False))
    print(f"all curves: {wall:.1f}s wall-clock ({len(tasks)} curves in parallel), "
          f"velocity saturation {'ON' if vsat_on else 'OFF'}")

    transfer = []
    for res in results[:len(vds_transfer)]:
        pts = res["points"]
        v = np.array(sorted(pts))
        transfer.append(dict(Vds=res["task"]["Vds"], Vgs=v, Id=np.array([pts[x]["Id"] for x in v])))
    output = []
    for res in results[len(vds_transfer):]:
        pts = res["points"]
        v = np.array(sorted(pts))
        output.append(dict(Vgs=res["task"]["Vgs"], Vds=v, Id=np.array([pts[x]["Id"] for x in v])))

    with open(os.path.join(out_dir, "ids_vgs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Vds_V", "Vgs_V", "Id_A_per_um"])
        for c in transfer:
            for vg, i in zip(c["Vgs"], c["Id"]):
                w.writerow([f"{c['Vds']:.4f}", f"{vg:.4f}", f"{i:.6e}"])
    with open(os.path.join(out_dir, "ids_vds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Vgs_V", "Vds_V", "Id_A_per_um"])
        for c in output:
            for vd, i in zip(c["Vds"], c["Id"]):
                w.writerow([f"{c['Vgs']:.4f}", f"{vd:.4f}", f"{i:.6e}"])

    if len(transfer) >= 2:
        m = mosfet_metrics(transfer[0], transfer[-1], L_um)
        print(f"  SS (Vds={transfer[0]['Vds']:g}V) = {m['SS_lo']:.1f} mV/dec, "
              f"SS (Vds={transfer[-1]['Vds']:g}V) = {m['SS_hi']:.1f} mV/dec")
        print(f"  Vt,cc = {m['Vt_cc_lo']:.3f} V -> {m['Vt_cc_hi']:.3f} V, DIBL = {m['DIBL_mV_per_V']:.1f} mV/V, "
              f"Vt,lin (max gm) = {m['Vt_lin']:.3f} V")
        print(f"  Ion = {m['Ion'] * 1e6:.1f} uA/um, Ioff = {m['Ioff']:.2e} A/um, Ion/Ioff = {m['Ion_Ioff']:.2e}")
        plot_transfer(transfer, m, os.path.join(out_dir, "ids_vgs.png"), L_um)
    plot_output(output, os.path.join(out_dir, "ids_vds.png"))
    print(f"wrote {out_dir}/ids_vgs.png, ids_vds.png, ids_vgs.csv, ids_vds.csv")

    # Structure+fields JSON: the last point of the low-Vds transfer curve.
    structure_file = output_cfg.get("structure_file")
    if structure_file:
        mesh = _context(args.config)[0]
        res = results[0]
        r = res["last_state"]
        Ex, Ey = electric_field_2d(mesh.points, mesh.triangles, r["psi"])
        fields = {k: r[k] for k in ("psi", "n", "p", "phin", "phip")}
        fields["Ex"] = Ex; fields["Ey"] = Ey
        regions = [dict(name=rg.name, x_range_um=[rg.x_range_cm[0] * _CM_TO_UM, rg.x_range_cm[1] * _CM_TO_UM],
                        y_range_um=[rg.y_range_cm[0] * _CM_TO_UM, rg.y_range_cm[1] * _CM_TO_UM],
                        kind=rg.kind, doping_type=rg.doping_type) for rg in domain.regions]
        boundary = [{"point_index": int(i), "bc_type": b}
                    for i, b in zip(mesh.boundary_point_index, mesh.boundary_bc_type)]
        material_dict = dict(eps_r=mat.eps_r, ni=mat.ni, mu_n=mat.mu_n, mu_p=mat.mu_p,
                              tau_n=mat.tau_n, tau_p=mat.tau_p, chi_eV=mat.chi_eV, Eg_eV=mat.Eg_eV)
        points_um = mesh.points * _CM_TO_UM
        doc = sio.build_structure(
            device="mosfet2d", material=material_dict, regions=regions,
            x_um=points_um[:, 0], y_um=points_um[:, 1], doping_cm3=mesh.Cdop,
            bias_points=[{"label": f"Vgs={res['last_bias']:+.3f}V (Vds={res['task']['Vds']}V)",
                          "bias": float(res["last_bias"]), "fields": fields}],
            dim=2, mesh2d={"triangles": mesh.triangles.tolist(), "boundary": boundary})
        sio.write_structure(os.path.join(out_dir, structure_file), doc)


if __name__ == "__main__":
    main()
