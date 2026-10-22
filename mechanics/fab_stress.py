"""3D process stress of a virtual-fab device (FinFET, nanosheet, planar),
step by step through its real flow (fab/), on the fab's own voxels.

  python3 -m mechanics.fab_stress configs/fab_finfet_pmos.yaml
  python3 -m mechanics.fab_stress configs/fab_nanosheet_pmos.yaml --set sige_sl_x=0.25 gate_GPa=0

Every fab step is snapshotted (fab.flow.build on_step), coarsened to the
stress grid (default 1 nm: the majority region of each 2x2x2 block of
0.5 nm voxels) and handed to mechanics.fe3d.StressHistory3D. A solve runs
only where the stress can change (a stressor born, stressed material
removed); depositing an unstressed film or a mask/doping step costs
nothing.

Stressors, from the fab's region names (rough defaults; --set overrides):
  sd_epi (SiGe)     lattice mismatch, x = epi_x (0.5: the PMOS Si0.5Ge0.5 S/D)
  sd_epi_seed       the bilayer epi's low-Ge seed, x from its material name
  (SiGe0.2)         (fab param epi_seed_x; epi_seed_thickness=0 drops it)
  sige  (SiGe)      nanosheet superlattice, x = sige_sl_x (0.30): grown
                    pseudomorphic on the Si lattice, so the SiGe is born
                    compressed (mismatch eps*); relaxes sideways at the stack
                    etch, released at channel release. sige_sl_strain=0 grows
                    it stress-free instead (what-if, for teaching)
  sd_epi (Si, NMOS) Si:P: heavy P (~2e21 cm-3) shrinks the lattice; eps* =
                    sip_strain_pct (-0.4 %, i.e. born tensile; 0 = off).
                    Literature: 1.75e21 P ~ 0.8 at.% C_sub, 2.8e21 ~ 1.8 at.%
                    C_sub (-0.27 .. -0.62 %); Si:C 1.65 % C ~ -0.57 %
  S/D dislocations  opt-in (sd_disl=1: one per S/D side; -1 = NMOS only):
                    edge dislocations, b = a/2<110> along the channel, a
                    missing half plane above a core disl_depth (25 nm) below
                    the channel top, in the epi next to the channel
                    (sd_dislocations, mechanics.dislocation)
  gate (metal)      intrinsic stress gate_GPa (-1.0, TiN)
  *_contact         intrinsic stress contact_GPa (-1.0, TiN fill; W would be +1)
  sti               sti_GPa (0 = off; e.g. -0.3 HDP oxide)
  spacer            spacer_GPa (0 = off)
Crystal: Si and SiGe anisotropic, (001) wafer, <110> channel (FinFET side
walls (110)); the rest isotropic (core/material_db.py).

Channel: the silicon within 1.5 nm of the final gate dielectric, under the
gate (per sheet for a nanosheet). Reported after every solved step: mean
sigma_xx (channel), sigma_yy (vertical), sigma_zz (width), total and per
stressor, plus the linear piezoresistance estimate of the mobility change.
"""
import argparse
import os
import time

import numpy as np
from scipy import ndimage

from mechanics import elastic as el
from mechanics.dislocation import B_SI, edge_dislocation_3d
from mechanics.fe3d import Grid3D, StressHistory3D
from mechanics.piezo import mobility_change

DEFAULTS = dict(epi_x=0.5, sige_sl_x=0.30, sige_sl_strain=1.0, gate_GPa=-1.0, contact_GPa=-1.0, sti_GPa=0.0,
                spacer_GPa=0.0, sip_strain_pct=-0.4, sd_disl=0.0, disl_depth=25.0, disl_offset=0.5,
                disl_b=B_SI, disl_angle=90.0, channel_dir=110)
CRYSTAL = ("Si", "SiGe")          # materials a dislocation cut runs through (Si, SiGe, SiGe0.2, ...)
SI_LIKE = {"Si"}


def snapshots(config, factor=2, log=print, fab_set=None):
    """[(step label, coarse labels (region ids, 0 void))], {id: (name,
    material)}, coarse Grid3D, fab params. fab_set: fab template
    parameters to override ({"epi_seed_thickness": 0})."""
    from fab import flow
    names = {}
    snaps = []

    def on_step(i, label, st):
        lab = st.cell_labels()
        ids = np.zeros(lab.shape, np.int32)
        for k, r in enumerate(st.regions):
            gid = names.setdefault(r.name, (len(names) + 1, r.material))[0]
            ids[lab == k] = gid
        ids = _coarsen(ids, _factors(st.grid.h, factor))
        # a step that only carves a layer out of a region born in the step
        # before (the bilayer epi's seed out of the just-grown epi: parent ->
        # child relabels, the solid unchanged) is part of that birth: fold
        # it into the previous snapshot
        if len(snaps) >= 2:
            prev, before = snaps[-1][1], snaps[-2][1]
            changed = ids != prev
            pairs = {names[r.name][0]: names[r.parent][0] for r in st.regions
                     if r.parent is not None and r.parent in names}
            if changed.any() and pairs and ((ids != 0) == (prev != 0)).all():
                fold = True
                for child, parent in pairs.items():
                    m = changed & (ids == child)
                    fold &= bool((prev[m] == parent).all() and (before[m] != parent).all())
                fold &= bool(np.isin(ids[changed], list(pairs)).all())
                if fold:
                    snaps[-1] = (f"{snaps[-1][0]} + {label}", ids)
                    snapshots.grid = st.grid
                    return
        snaps.append((label, ids))
        snapshots.grid = st.grid

    cfg = flow.load(config)
    if fab_set:
        cfg.setdefault("params", {}).update(fab_set)
    st, _ = flow.build(cfg, on_step=on_step, log=lambda *a: None)
    gfab = st.grid
    h = float(gfab.h[0]) * factor
    shape = snaps[-1][1].shape
    grid = Grid3D(shape, h, gfab.lo)
    regions = {gid: (name, mat) for name, (gid, mat) in names.items()}
    return snaps, regions, grid, st


def _factors(h, factor):
    """Per-axis coarsening of fab voxels h to cubic stress voxels of
    factor * h[0] (a finer fab voxel across the fin, e.g. 0.25 nm in z,
    gets a larger factor)."""
    fs = [factor * h[0] / ha for ha in h]
    if any(abs(f - round(f)) > 1e-9 for f in fs):
        raise ValueError(f"fab voxel {tuple(h)} does not coarsen to a cubic {factor * h[0]:g} nm stress voxel")
    return tuple(int(round(f)) for f in fs)


def _coarsen(ids, f):
    fx, fy, fz = (f, f, f) if np.isscalar(f) else f
    if fx == fy == fz == 1:
        return ids
    nx, ny, nz = (s // fa * fa for s, fa in zip(ids.shape, (fx, fy, fz)))
    b = ids[:nx, :ny, :nz].reshape(nx // fx, fx, ny // fy, fy, nz // fz, fz)
    b = b.transpose(0, 2, 4, 1, 3, 5).reshape(nx // fx, ny // fy, nz // fz, fx * fy * fz)
    f3 = fx * fy * fz
    # Solid if at least half the children are solid; then the most common
    # solid region, ties to the OLDER region (smaller id). Plain majority
    # with void as a candidate flickered: a 4 SiGe / 4 void block read as
    # void, then as SiGe once ILD filled its void half - a false SiGe
    # "birth" (and its mismatch stress) in the ILD step.
    vals = [v for v in np.unique(b) if v != 0]
    out = np.zeros(b.shape[:3], np.int32)
    if not vals:
        return out
    counts = np.stack([(b == v).sum(axis=-1) for v in vals], axis=-1)     # ascending id: argmax -> older
    solid = (b != 0).sum(axis=-1) * 2 >= f3
    out[solid] = np.asarray(vals, np.int32)[np.argmax(counts, axis=-1)][solid]
    return out


def _ge_fraction(name, mat, par):
    """Ge fraction of a SiGe region: from the material name ("SiGe0.2", the
    bilayer epi's seed), else the superlattice / S/D fill parameter."""
    if len(mat) > 4:
        return float(mat[4:])
    return par["sige_sl_x"] if name == "sige" else par["epi_x"]


def _materials(regions, par):
    mats = {}
    R = el.channel_frame(par.get("channel_dir", 110))
    for gid, (name, mat) in regions.items():
        if mat.startswith("SiGe"):
            mats[gid] = el.material("SiGe", R=R, x_ge=_ge_fraction(name, mat, par))
        else:
            mats[gid] = el.material(mat, R=R)
    return mats


def _stressor_of(name, mat, par, m):
    """(stressor name, eps*) for voxels of region `name`, or None."""
    if mat.startswith("SiGe"):
        if name == "sige" and not par["sige_sl_strain"]:
            return None
        f = el.sige_mismatch(_ge_fraction(name, mat, par))
        sname = {"sige": "SiGe superlattice", "sd_epi_seed": "SiGe S/D seed"}.get(name, "SiGe S/D")
        return sname, np.array([f, f, f, 0, 0, 0])
    if name.startswith("sd_epi") and mat == "Si" and par["sip_strain_pct"]:
        f = par["sip_strain_pct"] / 100.0        # Si:P (NMOS S/D): a smaller lattice than Si, born tensile
        return "Si:P S/D", np.array([f, f, f, 0, 0, 0])
    if name == "gate" and par["gate_GPa"]:
        return "gate metal", el.film_eigenstrain(m, par["gate_GPa"])
    if name.endswith("_contact") and par["contact_GPa"]:
        return "contacts", el.film_eigenstrain(m, par["contact_GPa"])
    if name == "sti" and par["sti_GPa"]:
        return "STI", el.film_eigenstrain(m, par["sti_GPa"])
    if name in ("spacer", "inner_spacer") and par["spacer_GPa"]:
        return "spacers", el.film_eigenstrain(m, par["spacer_GPa"])
    return None


def sd_dislocations(labels, regions, grid, channel, par):
    """(mask, eps*, description) of the S/D edge dislocations (Volterra
    cuts, mechanics.dislocation.edge_dislocation_3d), or None.

    sd_disl dislocations per S/D side, at the gate-side face of the S/D epi:
      * the line runs across the width (z) through all the crystal (Si,
        SiGe) at that x and depth;
      * the core sits disl_offset inside the epi from its channel-side face,
        disl_depth nm below the channel top (the top of the final channel).
        The default, 25 nm, is the amorphization depth of a Ge PAI at
        25-30 keV (the dislocation-SMT of TSMC's FinFET patent, US9570587),
        the same pinch-off depth as the 2D planar flow. disl_depth < 0 puts
        it at the bottom of the epi there instead (the recess corner, where
        stacking faults in a recessed S/D epi start);
      * b = a/2<110> along the channel (disl_b nm), with the half plane
        between the core and the surface MISSING (the SMT / SPER type): it
        pulls the S/D together and stretches the channel;
      * the cut runs up from the core to the surface, vertical (disl_angle
        90) or inclined (54.7: a {111} plane), leaning away from the channel.
        The field does not depend on the cut (Volterra) as long as it ends
        on a free surface;
      * sd_disl = n > 1 puts n dislocations on a side, their cores evenly
        spaced in depth down to the core depth above.
    Inserted right after the S/D epi is born, while nothing sits on the
    epi top (the dummy gate and spacer are beside the cut, not above it)."""
    n_per = int(round(par["sd_disl"]))
    epi = np.isin(labels, [g for g, (n, m) in regions.items() if n.startswith("sd_epi")])
    if n_per <= 0 or not epi.any():
        return None
    cryst = np.isin(labels, [g for g, (n, m) in regions.items() if m.startswith(CRYSTAL)])
    X, Y, Z = grid.centres()
    h = grid.h
    xc = X[channel].mean()
    y_top = Y[channel].min() - h / 2
    th = np.radians(par["disl_angle"])
    mask = np.zeros(grid.shape, bool)
    eps = np.zeros((mask.size, 6))
    for side in (-1, 1):
        ex = X[epi & (np.sign(X - xc) == side)]
        if not len(ex):
            continue
        face = ex.max() + h / 2 if side < 0 else ex.min() - h / 2
        x_d = face + side * par["disl_offset"]
        if par["disl_depth"] < 0:
            col = epi & (np.abs(X - x_d) < h / 2)
            depth = Y[col].max() + h / 2 - y_top
        else:
            depth = par["disl_depth"]
        for k in range(n_per):
            core = (x_d, y_top + (k + 1) * depth / n_per, 0.0)
            cut = (side * np.cos(th), -np.sin(th), 0.0)            # up, leaning away from the channel
            # the cut normal has n_x > 0 for any upward cut, so b = -|b| x removes material
            m, e, _ = edge_dislocation_3d(grid, core, (0, 0, 1), cut, (-par["disl_b"], 0.0, 0.0), solid=cryst)
            mask |= m
            eps += e
    desc = f"S/D dislocations ({n_per}/side, core {depth:.0f} nm)"
    return mask, eps, desc


def channel_masks(labels, regions, grid):
    """{'channel' or 'sheet k': mask}: Si within 1.5 nm of the gate
    dielectric, under the gate's x extent."""
    gox = [g for g, (n, m) in regions.items() if n == "gate_oxide"]
    gate = [g for g, (n, m) in regions.items() if n == "gate"]
    si = [g for g, (n, m) in regions.items() if m in SI_LIKE]
    is_ox = np.isin(labels, gox)
    dist = ndimage.distance_transform_edt(~is_ox) * grid.h
    X, Y, Z = grid.centres()
    gx = X[np.isin(labels, gate)]
    x0, x1 = gx.min(), gx.max()
    ch = np.isin(labels, si) & (dist <= 1.5) & (X > x0 + 1) & (X < x1 - 1)
    out = {"channel": ch}
    sheets = [g for g, (n, m) in regions.items() if n == "sheets"]
    if sheets:
        lab_s, n = ndimage.label(np.isin(labels, sheets) & (X > x0) & (X < x1))
        comps = sorted(range(1, n + 1), key=lambda c: Y[lab_s == c].mean())
        for k, c in enumerate(comps):
            m = ch & (lab_s == c)
            if m.sum() > 10:
                out[f"sheet {k + 1} (top)" if k == 0 else f"sheet {k + 1}"] = m
    return out


def run(config, par=DEFAULTS, factor=2, fast=False, log=print, fab_set=None):
    """fast=False: solve after every step that changes the stress (the
    teaching table: every stress-changing step is listed). fast=True: solve
    only when the physics forces it - before material is born into a
    structure with unsolved forces, and at the end. Same final answer
    (exact: elastic removals/stressor switch-ons commute until the next
    birth), fewer solves; the table then lists those solves only."""
    t0 = time.perf_counter()
    snaps, regions, grid, st = snapshots(config, factor, fab_set=fab_set)
    t_fab = time.perf_counter() - t0
    mats = _materials(regions, par)
    sh = StressHistory3D(grid, mats)
    final = snaps[-1][1]
    probes = channel_masks(final, regions, grid)
    carrier = "p" if any(m == "SiGe" for n, m in regions.values() if n == "sd_epi") else "n"
    n_disl = par["sd_disl"] if par["sd_disl"] >= 0 else (1 if carrier == "n" else 0)   # -1: NMOS 1, PMOS 0
    par = dict(par, sd_disl=n_disl)
    disl_done = False
    rows = []
    prev = np.zeros(grid.shape, np.int32)
    for label, lab in snaps:
        born = (lab != 0) & (lab != prev)
        stressors = {}
        for gid in np.unique(lab[born]):
            name, mat = regions[gid]
            sdef = _stressor_of(name, mat, par, mats[gid])
            if sdef is not None:
                sname, eps = sdef
                m = born & (lab == gid)
                if sname in stressors:
                    m = m | stressors[sname][0]
                stressors[sname] = (m, eps)
        steps = [(label, stressors)]
        if n_disl > 0 and not disl_done and any(regions[g][0].startswith("sd_epi") for g in np.unique(lab[born])):
            d = sd_dislocations(lab, regions, grid, probes["channel"], par)
            if d is not None:                 # its own step right after the epi birth (no birth in it)
                steps.append((d[2], {"S/D dislocations": (d[0], d[1])}))
                disl_done = True
        for lbl, stt in steps:
            solved = sh.step(lbl, lab, stressors=stt, flush=not fast)
            if solved:          # fast mode solves before this step's births: the state before it
                rows.append(("before: " + lbl if fast else lbl, {k: sh.probe(m) for k, m in probes.items()}))
        prev = lab
    if sh.flush():
        rows.append(("(final)", {k: sh.probe(m) for k, m in probes.items()}))
    return dict(rows=rows, sh=sh, grid=grid, regions=regions, probes=probes, carrier=carrier,
                t_fab=t_fab, t_total=time.perf_counter() - t0, snaps=snaps)


def report(res, name):
    rows, carrier = res["rows"], res["carrier"]
    out = [f"\n=== {name}: channel stress (MPa, + tensile; xx along the channel, yy vertical, zz width)",
           f"{'step (solved steps only)':46s} {'sxx':>7s} {'syy':>7s} {'szz':>7s}"]
    for label, pr in rows:
        t = pr["channel"][0] * 1e3
        out.append(f"{label[:46]:46s} {t[0]:7.0f} {t[1]:7.0f} {t[2]:7.0f}")
    names = list(rows[-1][1]["channel"][1])
    if names:                   # the same steps, sigma_xx split by stressor (superposition)
        out.append(f"{'sigma_xx by stressor':46s} " + " ".join(f"{s[:12]:>12s}" for s in names))
        for label, pr in rows:
            per = pr["channel"][1]
            out.append(f"{label[:46]:46s} " + " ".join(f"{per[s][0] * 1e3 if s in per else 0.0:12.0f}" for s in names))
    pr = rows[-1][1]
    out.append(f"final, by stressor (channel){'':12s} {'sxx':>7s} {'syy':>7s} {'szz':>7s}   dmu/mu ({carrier}, linear piezo)")
    tot, per = pr["channel"]
    for s, v in per.items():
        out.append(f"  {s:42s} {v[0] * 1e3:7.0f} {v[1] * 1e3:7.0f} {v[2] * 1e3:7.0f}   {100 * mobility_change(v, carrier):+5.0f}%")
    out.append(f"  {'total':42s} {tot[0] * 1e3:7.0f} {tot[1] * 1e3:7.0f} {tot[2] * 1e3:7.0f}   "
               f"{100 * mobility_change(tot, carrier):+5.0f}%")
    if len(pr) > 2:
        out.append("per sheet (total):")
        for k, (t, _) in pr.items():
            if k != "channel":
                out.append(f"  {k:42s} {t[0] * 1e3:7.0f} {t[1] * 1e3:7.0f} {t[2] * 1e3:7.0f}   "
                           f"{100 * mobility_change(t, carrier):+5.0f}%")
    sh = res["sh"]
    out.append(f"({len(res['snaps'])} fab steps, {sh.n_solves} solves, grid {res['grid'].shape} at "
               f"{res['grid'].h:g} nm; fab {res['t_fab']:.0f} s, total {res['t_total']:.0f} s)")
    return "\n".join(out)


def plot(res, name, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, sh = res["grid"], res["sh"]
    s = sh.element_stress().reshape(g.shape + (6,)) * 1e3
    lab = sh.labels
    ch = res["probes"]["channel"]
    ix = int(np.round(np.nonzero(ch)[0].mean()))
    iz = int(np.nonzero(ch)[2].min())
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw=dict(width_ratios=[1.3, 1, 0.7]))
    ax = axs[0]
    rows = res["rows"]
    stressors = sorted({k for _, pr in rows for k in pr["channel"][1]})
    kk = np.arange(len(rows))
    for st_ in stressors:
        ax.plot(kk, [pr["channel"][1].get(st_, np.zeros(6))[0] * 1e3 for _, pr in rows], "-o", ms=3, label=st_)
    ax.plot(kk, [pr["channel"][0][0] * 1e3 for _, pr in rows], "k-o", lw=2.5, label="total")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(kk)
    ax.set_xticklabels([r[0][:34] for r in rows], rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("channel sigma_xx (MPa)")
    ax.set_title(f"{name}: channel stress, solved steps")
    ax.legend(fontsize=7.5)

    def show(ax, img, mask, extent, title, xl, yl):
        img = np.where(mask, img, np.nan)
        lim = np.nanpercentile(np.abs(img), 98)
        im = ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper", extent=extent,
                       interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        fig.colorbar(im, ax=ax, shrink=0.75)

    X0, Y0, Z0 = g.lo
    Lx, Ly, Lz = (np.array(g.shape) * g.h)
    show(axs[1], s[:, :, iz, 0].T, (lab[:, :, iz] != 0).T, [X0, X0 + Lx, Y0 + Ly, Y0],
         f"sigma_xx (MPa), x-y cut at z = {Z0 + (iz + 0.5) * g.h:.1f} nm", "x (nm)", "y (nm)")
    show(axs[2], s[ix, :, :, 0], lab[ix] != 0, [Z0, Z0 + Lz, Y0 + Ly, Y0],
         f"sigma_xx, y-z cut at x = {X0 + (ix + 0.5) * g.h:.1f} nm", "z (nm)", "y (nm)")
    for ax in axs[1:]:
        ax.set_ylim(min(60, Y0 + Ly), Y0)
    fig.tight_layout()
    path = os.path.join(out, f"{name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", default="out/stress3d")
    ap.add_argument("--factor", type=int, default=2, help="stress voxel = factor x fab voxel (default 2: 1 nm)")
    ap.add_argument("--fast", action="store_true",
                    help="solve only where the physics forces it (same final stress, fewer solves)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--fab", nargs="*", default=[],
                    help="fab template parameters, e.g. epi_seed_thickness=0 (single-layer S/D epi)")
    ap.add_argument("--tag", default="", help="suffix of the output names")
    a = ap.parse_args()
    fab_set = {}
    for kv in a.fab:
        k, v = kv.split("=")
        fab_set[k] = float(v)
    par = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        par[k] = float(v)
    os.makedirs(a.out, exist_ok=True)
    for cfg in a.configs:
        name = os.path.splitext(os.path.basename(cfg))[0] + a.tag
        res = run(cfg, par, a.factor, a.fast, fab_set=fab_set)
        txt = report(res, name)
        print(txt)
        print("  ->", plot(res, name, a.out))
        with open(os.path.join(a.out, name + ".txt"), "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
