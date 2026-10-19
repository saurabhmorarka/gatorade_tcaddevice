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
  sige  (SiGe)      nanosheet superlattice, x = sige_sl_x (0.30): grown
                    pseudomorphic on the Si lattice, so the SiGe is born
                    compressed (mismatch eps*); relaxes sideways at the stack
                    etch, released at channel release. sige_sl_strain=0 grows
                    it stress-free instead (what-if, for teaching)
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
from mechanics.fe3d import Grid3D, StressHistory3D
from mechanics.piezo import mobility_change

DEFAULTS = dict(epi_x=0.5, sige_sl_x=0.30, sige_sl_strain=1.0, gate_GPa=-1.0, contact_GPa=-1.0, sti_GPa=0.0,
                spacer_GPa=0.0)
SI_LIKE = {"Si"}


def snapshots(config, factor=2, log=print):
    """[(step label, coarse labels (region ids, 0 void))], {id: (name,
    material)}, coarse Grid3D, fab params."""
    from fab import flow
    names = {}
    snaps = []

    def on_step(i, label, st):
        lab = st.cell_labels()
        ids = np.zeros(lab.shape, np.int32)
        for k, r in enumerate(st.regions):
            gid = names.setdefault(r.name, (len(names) + 1, r.material))[0]
            ids[lab == k] = gid
        snaps.append((label, _coarsen(ids, factor)))
        snapshots.grid = st.grid

    cfg = flow.load(config)
    st, _ = flow.build(cfg, on_step=on_step, log=lambda *a: None)
    gfab = st.grid
    h = float(gfab.h[0]) * factor
    shape = snaps[-1][1].shape
    grid = Grid3D(shape, h, gfab.lo)
    regions = {gid: (name, mat) for name, (gid, mat) in names.items()}
    return snaps, regions, grid, st


def _coarsen(ids, f):
    if f == 1:
        return ids
    nx, ny, nz = (s // f * f for s in ids.shape)
    b = ids[:nx, :ny, :nz].reshape(nx // f, f, ny // f, f, nz // f, f)
    b = b.transpose(0, 2, 4, 1, 3, 5).reshape(nx // f, ny // f, nz // f, f ** 3)
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
    solid = (b != 0).sum(axis=-1) * 2 >= f ** 3
    out[solid] = np.asarray(vals, np.int32)[np.argmax(counts, axis=-1)][solid]
    return out


def _materials(regions, par):
    mats = {}
    for gid, (name, mat) in regions.items():
        if mat == "SiGe":
            x = par["sige_sl_x"] if name == "sige" else par["epi_x"]
            mats[gid] = el.material("SiGe", x_ge=x)
        else:
            mats[gid] = el.material(mat)
    return mats


def _stressor_of(name, mat, par, m):
    """(stressor name, eps*) for voxels of region `name`, or None."""
    if mat == "SiGe":
        if name == "sige" and not par["sige_sl_strain"]:
            return None
        x = par["sige_sl_x"] if name == "sige" else par["epi_x"]
        f = el.sige_mismatch(x)
        return ("SiGe superlattice" if name == "sige" else "SiGe S/D", np.array([f, f, f, 0, 0, 0]))
    if name == "gate" and par["gate_GPa"]:
        return "gate metal", el.film_eigenstrain(m, par["gate_GPa"])
    if name.endswith("_contact") and par["contact_GPa"]:
        return "contacts", el.film_eigenstrain(m, par["contact_GPa"])
    if name == "sti" and par["sti_GPa"]:
        return "STI", el.film_eigenstrain(m, par["sti_GPa"])
    if name in ("spacer", "inner_spacer") and par["spacer_GPa"]:
        return "spacers", el.film_eigenstrain(m, par["spacer_GPa"])
    return None


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


def run(config, par=DEFAULTS, factor=2, fast=False, log=print):
    """fast=False: solve after every step that changes the stress (the
    teaching table: every stress-changing step is listed). fast=True: solve
    only when the physics forces it - before material is born into a
    structure with unsolved forces, and at the end. Same final answer
    (exact: elastic removals/stressor switch-ons commute until the next
    birth), fewer solves; the table then lists those solves only."""
    t0 = time.perf_counter()
    snaps, regions, grid, st = snapshots(config, factor)
    t_fab = time.perf_counter() - t0
    mats = _materials(regions, par)
    sh = StressHistory3D(grid, mats)
    final = snaps[-1][1]
    probes = channel_masks(final, regions, grid)
    carrier = "p" if any(m == "SiGe" for n, m in regions.values() if n == "sd_epi") else "n"
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
        solved = sh.step(label, lab, stressors=stressors, flush=not fast)
        if solved:              # fast mode solves before this step's births: the state before it
            rows.append(("before: " + label if fast else label, {k: sh.probe(m) for k, m in probes.items()}))
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
    a = ap.parse_args()
    par = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        par[k] = float(v)
    os.makedirs(a.out, exist_ok=True)
    for cfg in a.configs:
        name = os.path.splitext(os.path.basename(cfg))[0]
        res = run(cfg, par, a.factor, a.fast)
        txt = report(res, name)
        print(txt)
        print("  ->", plot(res, name, a.out))
        with open(os.path.join(a.out, name + ".txt"), "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
