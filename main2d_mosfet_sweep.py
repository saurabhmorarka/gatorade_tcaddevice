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
from core.parallel import performance_cores
from core.physics import equilibrium_bulk_potential
from mesh2d.config2d import build_domain_from_config
from mesh2d.mesh2d import build_mesh2d
from mos import mos_analytic as man
from mos.mos_params import MOSDevice
from solver2d.current import contact_current
from solver2d.mosfet_metrics import mosfet_metrics_signed
from solver2d.efield2d import electric_field_2d
from solver2d.newton_solver_qf_2d import newton_solve_2d, vsat_force

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
                  verbose=False, maxiter=SWEEP_MAXITER, cold_retry=True, velocity_saturation=False,
                  vsat_fallback=True, schottky=None, quantum=None, quantum_update=None, quantum_tol=3e-3,
                  generation=None):
    psi_bc, phin_bc, phip_bc = gate_bc(dev, mat, Cdop_substrate, Vgs)
    bias_by_contact = {"source": 0.0, "drain": Vds, "gate": Vgs, "body": 0.0}
    return newton_solve_2d(
        mesh, mat, bias_by_contact,
        psi_init=psi_init, phin_init=phin_init, phip_init=phip_init,
        psi_bc_override={"gate": psi_bc}, phin_bc_override={"gate": phin_bc},
        phip_bc_override={"gate": phip_bc}, verbose=verbose, maxiter=maxiter,
        cold_retry=cold_retry, velocity_saturation=velocity_saturation, vsat_fallback=vsat_fallback,
        schottky=schottky, quantum=quantum, quantum_update=quantum_update, quantum_tol=quantum_tol,
        generation=generation,
    )


def _converged(r):
    # not r.get("converged", r["res_norm"] < 1e-4): the default is evaluated
    # eagerly, and an early-exit result (a linear-solver breakdown) has no res_norm
    return r["converged"] if "converged" in r else r["res_norm"] < 1e-4


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


def _predict_psi(prev, cur, p_next):
    """Secant predictor for psi only; the quasi-Fermi levels stay at the
    last converged state. For a GATE walk: the gate moves psi, while phin
    and phip stay pinned by the unchanged source/drain contacts, and their
    extrapolated change is mostly noise from the undetermined regions -
    on the 3D planar device at Vds = 1 V the full secant guess cost up to
    15 Newton iterations per step, this one 5 (DEVELOPMENT_LOG 57). Not
    for a drain ramp, whose contact quasi-Fermi levels move: there the
    full secant is 6x cheaper."""
    g = _predict(prev, cur, p_next)
    return {"psi": g["psi"], "phin": cur[1]["phin"], "phip": cur[1]["phip"]}


def continuation(solve_at, p0, s0, targets, prev=None, log=None, h_init=H_INIT, fast_iters=FAST_ITERS,
                 psi_only=False, on_point=None):
    """Walk the bias parameter from p0 (converged state s0) through each
    value in `targets` (in order), with a secant predictor (psi only if
    psi_only: for a gate walk, see _predict_psi) and adaptive sub-stepping
    from a first step h_init: a step that fails is retried at half size (up
    to MAX_HALVINGS times), a step that converges in <= fast_iters iterations
    doubles the next step (never overshooting the next requested target).
    on_point(target, state), if given, is called as each target converges.
    Returns ({target: state}, n_newton_iterations, last (prev, cur) pair);
    stops early (keeping what converged) if a step can't be completed."""
    out = {}
    cur = (p0, s0)
    total_it = 0
    h = h_init
    for tgt in targets:
        while abs(tgt - cur[0]) > 1e-12:
            remaining = tgt - cur[0]
            h = np.sign(remaining) * min(abs(h), abs(remaining))
            for _ in range(MAX_HALVINGS + 1):
                p_next = cur[0] + h
                r = solve_at(p_next, (_predict_psi if psi_only else _predict)(prev, cur, p_next))
                total_it += r["iters"]
                if _converged(r):
                    break
                h *= 0.5
            else:
                if log:
                    log(f"    step to {p_next:+.4f} failed after {MAX_HALVINGS} halvings - curve stopped")
                return out, total_it, (prev, cur)
            prev, cur = cur, (p_next, r)
            # step_iters: a solve made of several Newton solves (the
            # Schrodinger-Poisson loop, quantum/device.py) reports the
            # first one's count - what the bias step itself cost
            if r.get("step_iters", r["iters"]) <= fast_iters:
                h *= 2.0
        out[float(tgt)] = cur[1]
        if on_point is not None:
            on_point(float(tgt), cur[1])
    return out, total_it, (prev, cur)


# --- worker side (one process per curve) -----------------------------------

_CTX = {}


DEFAULT_MOBILITY = "fast"


def mobility_options(cfg, override=None):
    """The inversion-layer mobility of a config (docs/mobility.md): the
    physics: mobility: entry - a mode string (fast | exact | off) or an
    options dict (solver2d/surface_mobility.MobilityModel) - or the
    override (a mode from --mobility). Absent: DEFAULT_MOBILITY. Returns a
    dict, or None for off."""
    m = (cfg.get("physics") or {}).get("mobility", DEFAULT_MOBILITY)
    m = dict(m) if isinstance(m, dict) else {"mode": m}
    if override:
        m["mode"] = override
    return None if m.get("mode", "fast") in ("off", None, False) else m


def _context(config_path, extrude=None, stress_file=None, mobility=None):
    """extrude=(width_um, nz): run the SAME device as a 3D mesh extruded in
    z (mesh3d/extrude.py) - the 3D performance testbed and the first rung
    of the 3D validation ladder (Id per um of width must match 2D).
    A config with a stress: block gets the process stress -> mobility
    factors (solver2d/stress_mobility.py) on its mesh; stress_file is the
    field main() solved once and saved (else it is solved here)."""
    key = (config_path, tuple(extrude) if extrude else None, stress_file, mobility)
    if key not in _CTX:
        cfg = load_config(config_path)
        domain, mat, dev, Cs, mesh_opts, bias_cfg, output_cfg = build_from_config(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = build_mesh2d(domain, mat=mat, **mesh_opts)
        mob = mobility_options(cfg, mobility)
        if mob is not None and not extrude:
            # before equilibrium: an interface charge is electrostatics
            from solver2d.surface_mobility import attach_mobility
            attach_mobility(mesh, mat, mob)
        vsat = (cfg.get("physics") or {}).get("velocity_saturation", False)   # bool or a force name
        vsat_force(vsat)                                                       # validate early
        eq = solve_mosfet(mesh, mat, dev, Cs, 0.0, 0.0, velocity_saturation=vsat)
        if extrude:
            from mesh3d.extrude import extrude_mesh
            width_um, nz = extrude
            mesh = extrude_mesh(mesh, mat, width_um * _UM_TO_CM, int(nz))
            g = {k: np.tile(eq[k], int(nz)) for k in _FIELDS}   # z-invariant: exact start
            if mob is not None:
                from solver2d.surface_mobility import attach_mobility
                attach_mobility(mesh, mat, mob)
            eq = solve_mosfet(mesh, mat, dev, Cs, 0.0, 0.0, psi_init=g["psi"], phin_init=g["phin"],
                              phip_init=g["phip"], velocity_saturation=vsat)
        if cfg.get("stress") and stress_file != "off":
            # after equilibrium: it carries no current, so the mobility does not enter it
            from solver2d.stress_mobility import PiezoModel, apply_stress_mobility, field_from_config
            st = dict(cfg["stress"], **({"file": stress_file} if stress_file else {}))
            field = field_from_config(st, domain, log=None)
            mesh.stress_report = apply_stress_mobility(mesh, mat, field, PiezoModel.from_options(st), log=None)
            mesh.stress_field = field
        _CTX[key] = (mesh, mat, dev, Cs, eq, vsat)
    return _CTX[key]


def _id_per_um(mesh, mat, r):
    I = contact_current(mesh, mat, r, "drain")
    width_cm = getattr(mesh, "width_cm", None)
    if width_cm:                      # 3D: total current (A) over the extruded width
        return I / (width_cm * _CM_TO_UM)
    return I * 1e-4                   # 2D: A/cm of depth -> A/um of width


def run_curve(task):
    """One Id-Vg (task['kind']=='idvg', fixed task['Vds']) or Id-Vd
    ('idvd', fixed task['Vgs']) curve, from the equilibrium anchor."""
    t0 = time.perf_counter()
    mesh, mat, dev, Cs, eq, vsat = _context(task["config"], task.get("extrude"), task.get("stress_file"),
                                            task.get("mobility"))
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
    points = {p: dict(Id=_id_per_um(mesh, mat, r), iters=r["iters"], res_norm=r["res_norm"],
                      converged=_converged(r))
              for p, r in sorted(states.items()) if p in set(map(float, task["values"])) or p == 0.0}
    last = max(p for p in states if p in points)
    vtk_entries = []
    if task.get("vtk") == "all":
        # each worker writes its own states (files are named by bias, so
        # segments of one curve never collide); main() writes the .pvd
        from solver2d.vtk_export import write_states
        vtk_entries = write_states(task["vtk_dir"], curve_stem(task), mesh, mat,
                                   [dict(state=states[p], bias=p, **_vtk_meta(task, p)) for p in points])
    return dict(task=task, points=points, newton_iters=total_it, log=log_lines,
                time_s=time.perf_counter() - t0,
                last_state={k: states[last][k] for k in ("psi", "n", "p", "phin", "phip")}, last_bias=last,
                vtk_entries=vtk_entries)


def curve_stem(task):
    return f"idvg_Vds{task['Vds']:g}" if task["kind"] == "idvg" else f"idvd_Vgs{task['Vgs']:g}"


def _vtk_meta(task, p):
    """VTK label and contact biases of the point at swept value p."""
    Vgs, Vds = (p, task["Vds"]) if task["kind"] == "idvg" else (task["Vgs"], p)
    return dict(label=f"Vgs={Vgs:+.3f}V Vds={Vds:+.3f}V", contact_bias={"gate": Vgs, "drain": Vds})


def split_task(task, curve_index, chunk_points):
    """Split one curve task into segments of at most chunk_points bias
    values, each in its own continuation order (Id-Vg: the Vgs<0 branch and
    the Vgs>0 branch walk away from 0 separately, so they split separately).
    Every segment is a complete run_curve task on its own: it repeats the
    cheap anchor (equilibrium -> drain/gate ramp) and walks from the anchor
    to its first value with adaptive steps whose states are not recorded.
    That duplicated walk is the price of using every core instead of one
    per curve - the long Id-Vg curves otherwise set the wall-clock time."""
    vals = sorted(float(v) for v in task["values"])
    if task["kind"] == "idvg":
        branches = [[v for v in vals if v > 0], sorted((v for v in vals if v < 0), reverse=True)]
    else:
        branches = [[v for v in vals if v > 0]]
    segs = []
    for br in branches:
        if not br:
            continue
        n = len(br) if chunk_points <= 0 else chunk_points
        k = -(-len(br) // n)                          # number of chunks
        size = -(-len(br) // k)                        # balanced chunk size
        for c in range(k):
            chunk = br[c * size:(c + 1) * size]
            if chunk:
                segs.append(dict(task, values=chunk + [0.0], curve=curve_index))
    return segs or [dict(task, curve=curve_index)]


def merge_segments(tasks, segments, seg_results):
    """Reassemble per-curve results (same shape run_curve returns for a
    whole curve) from the segment results."""
    merged = []
    for i, task in enumerate(tasks):
        parts = [r for s, r in zip(segments, seg_results) if s["curve"] == i]
        points, log, its, vtk_entries = {}, [], 0, {}
        for r in parts:
            for p, v in r["points"].items():
                points.setdefault(p, v)
            log += r["log"]
            its += r["newton_iters"]
            for e in r.get("vtk_entries", []):
                vtk_entries.setdefault(e["timestep"], e)
        with_state = [r for r in parts if r["last_state"] is not None]
        last = max(with_state, key=lambda r: r["last_bias"]) if with_state else None
        merged.append(dict(task=task, points=points, newton_iters=its, log=log,
                           time_s=max(r["time_s"] for r in parts),
                           last_state=last["last_state"] if last else None,
                           last_bias=last["last_bias"] if last else None,
                           vtk_entries=list(vtk_entries.values())))
    return merged


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


def _prepare_stress(st, domain, out_dir):
    """Solve a config's stress: block once (unless it names a file) and
    save it to <out_dir>/stress_field.npz; returns the file."""
    if st.get("file"):
        return st["file"]
    from solver2d.stress_mobility import field_from_config
    t0 = time.perf_counter()
    field = field_from_config(dict(st, align="none"), domain, log=None)
    os.makedirs(out_dir, exist_ok=True)
    path = field.save(os.path.join(out_dir, "stress_field.npz"))
    print(f"process stress ({st.get('source', 'planar_flow')} {st.get('flow', '')}): "
          f"{time.perf_counter() - t0:.1f} s -> {path}")
    return path


def _report_stress(mesh, domain):
    from solver2d.stress_mobility import channel_report
    rep = mesh.stress_report
    f_n, f_p = mesh.edge_mu0_factor
    gate = next(c for c in domain.contacts if c.name == "gate")
    gx = np.array(gate.x_range_cm) * 1e7
    ch = channel_report(mesh, f_n, f_p, mesh.stress_field, box_nm=((gx[0] + 2, gx[1] - 2), (0.0, 3.0)))
    print(f"stress mobility ({rep['mode']}, doping factor {'on' if rep['doping_factor'] else 'off'}): "
          f"{rep['n_si_edges']} silicon edges, frame check {100 * (1 - rep['frame_mismatch_exact']):.1f}% on silicon")
    if ch:
        print(f"  channel (top 3 nm under the gate, x edges): sigma xx/yy/zz/yz/xz/xy = {ch['sigma_MPa']} MPa, "
              f"mu_n x {ch['f_n']:.3f}, mu_p x {ch['f_p']:.3f}")


def _grid(start, stop, step):
    return [round(float(v), 6) for v in np.arange(start, stop + 0.5 * step, step)]


def main():
    parser = argparse.ArgumentParser(description="gatorade_tcaddevice 2D planar NMOS driver")
    parser.add_argument("config", nargs="?", default=DEFAULT_PATH)
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: out/<config name>)")
    parser.add_argument("--workers", type=int, default=performance_cores())
    parser.add_argument("--chunk-points", type=int, default=0,
                        help="split each curve into segments of at most this many bias points, "
                             "run as separate parallel tasks (0 = one task per curve)")
    parser.add_argument("--extrude-um", type=float, default=None,
                        help="run the device as a 3D mesh extruded over this width (um)")
    parser.add_argument("--nz", type=int, default=6, help="number of z layers with --extrude-um")
    parser.add_argument("--mobility", choices=("fast", "exact", "off"), default=None,
                        help="inversion-layer mobility (docs/mobility.md), overriding the config's "
                             "physics: mobility: (default fast)")
    parser.add_argument("--no-stress", action="store_true",
                        help="ignore the config's stress: block (the unstressed reference run)")
    parser.add_argument("--vtk", nargs="?", const="last", choices=["last", "all"], default=None,
                        help="also write VTK (<out-dir>/vtk/*.vtu + one .pvd per curve, for ParaView): "
                             "'last' (default) = the last point of each curve, 'all' = every bias point "
                             "(written by the workers; ~2 MB per point for a 3D mesh)")
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

    extrude = (args.extrude_um, args.nz) if args.extrude_um else None
    out_dir = os.path.join("out", os.path.splitext(os.path.basename(args.config))[0])
    if args.out_dir:
        out_dir = args.out_dir
    vtk_dir = os.path.join(out_dir, "vtk")
    tasks = ([dict(kind="idvg", Vds=v, values=vgs_list, config=args.config, extrude=extrude)
              for v in vds_transfer] +
             [dict(kind="idvd", Vgs=v, values=vds_list, config=args.config, extrude=extrude)
              for v in vgs_output])
    for t in tasks:
        t.update(vtk=args.vtk, vtk_dir=vtk_dir, mobility=args.mobility)
    mob = mobility_options(cfg, args.mobility)
    print("inversion-layer mobility: " + ("off (bulk only)" if mob is None else
                                          ", ".join(f"{k} {v}" for k, v in mob.items())))
    stress_file = None
    if cfg.get("stress") and args.no_stress:
        stress_file = "off"
        for t in tasks:
            t["stress_file"] = "off"
        print("stress: block ignored (--no-stress): unstressed mobility")
    elif cfg.get("stress"):
        # the process stress is solved once here and loaded by every worker
        stress_file = _prepare_stress(cfg["stress"], domain, out_dir)
        for t in tasks:
            t["stress_file"] = stress_file
        mesh_s = _context(args.config, extrude, stress_file, args.mobility)[0]
        _report_stress(mesh_s, domain)

    # One BLAS thread per worker - the workers are the parallelism.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    t0 = time.perf_counter()
    segments = [seg for i, t in enumerate(tasks) for seg in split_task(t, i, args.chunk_points)]
    # Longest segments first, so the pool never ends waiting on one big
    # segment that started last.
    order = sorted(range(len(segments)), key=lambda k: -len(segments[k]["values"]))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(segments))) as pool:
        seg_results = dict(zip(order, pool.map(run_curve, [segments[k] for k in order])))
    results = merge_segments(tasks, segments, [seg_results[k] for k in range(len(segments))])
    wall = time.perf_counter() - t0

    os.makedirs(out_dir, exist_ok=True)
    for res in results:
        t = res["task"]
        name = f"Id-Vg @ Vds={t['Vds']:g}V" if t["kind"] == "idvg" else f"Id-Vd @ Vgs={t['Vgs']:g}V"
        n_ok = sum(_converged(p) for p in res["points"].values())
        print(f"  {name:22s}: {n_ok}/{len(res['points'])} points converged, "
              f"{res['newton_iters']} Newton iterations, {res['time_s']:.1f}s")
        for line in res["log"]:
            print(line)
    force = vsat_force((cfg.get("physics") or {}).get("velocity_saturation", False))
    print(f"all curves: {wall:.1f}s wall-clock ({len(tasks)} curves as {len(segments)} parallel segments), "
          f"velocity saturation {'ON (' + force + ')' if force else 'OFF'}")

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
        m = mosfet_metrics_signed(transfer[0], transfer[-1], L_um)
        print(f"  SS (Vds={transfer[0]['Vds']:g}V) = {m['SS_lo']:.1f} mV/dec, "
              f"SS (Vds={transfer[-1]['Vds']:g}V) = {m['SS_hi']:.1f} mV/dec")
        print(f"  Vt,cc = {m['Vt_cc_lo']:.3f} V -> {m['Vt_cc_hi']:.3f} V, DIBL = {m['DIBL_mV_per_V']:.1f} mV/V, "
              f"Vt,lin (max gm) = {m['Vt_lin']:.3f} V")
        print(f"  Ion = {m['Ion'] * 1e6:.1f} uA/um, Ioff = {m['Ioff']:.2e} A/um, Ion/Ioff = {m['Ion_Ioff']:.2e}")
        plot_transfer(transfer, m, os.path.join(out_dir, "ids_vgs.png"), L_um)
    plot_output(output, os.path.join(out_dir, "ids_vds.png"))
    print(f"wrote {out_dir}/ids_vgs.png, ids_vds.png, ids_vgs.csv, ids_vds.csv")

    if args.vtk:
        from solver2d.vtk_export import write_states, write_pvd
        n_files = 0
        for res in results:
            entries = res["vtk_entries"]
            if args.vtk == "last" and res["last_state"] is not None:
                mesh_v, mat_v = _context(args.config, extrude, stress_file, args.mobility)[:2]
                entries = write_states(vtk_dir, curve_stem(res["task"]), mesh_v, mat_v,
                                       [dict(state=res["last_state"], bias=res["last_bias"],
                                             **_vtk_meta(res["task"], res["last_bias"]))])
            if entries:
                write_pvd(vtk_dir, curve_stem(res["task"]), entries)
                n_files += len(entries)
        print(f"wrote {vtk_dir}/*.pvd ({len(results)} curves, {n_files} states)")

    # Structure+fields JSON: the last point of the low-Vds transfer curve.
    structure_file = output_cfg.get("structure_file")
    if structure_file and not extrude:
        mesh = _context(args.config, None, stress_file, args.mobility)[0]
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
