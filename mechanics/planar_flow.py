"""Channel stress through a planar process flow, step by step (2D plane
strain, x-y cut along the channel). A teaching driver for mechanics/:

  python3 -m mechanics.planar_flow                 # all three flows
  python3 -m mechanics.planar_flow --flow pmos_gate_last --out out/stress

Flows (one cell: STI | S/D | spacer | gate | spacer | S/D | STI, the side
walls are symmetry planes, i.e. a dense array of identical devices):

  pmos_gate_first - SiGe S/D (recess + epi), metal gate from the start
  pmos_gate_last  - the same, but a poly dummy gate is pulled after ILD CMP
                    and replaced by HfO2 / TiN (RMG): the "poly pull" boost
  nmos_smt        - stress memorization: S/D + poly gate amorphized, a
                    tensile nitride cap, anneal (the amorphized regions
                    recrystallize stress-free in the capped shape), cap strip

Stressors and their default inputs (rough literature values; every one can
be changed with --set NAME=VALUE):
  sige_x        0.5    Ge fraction of the S/D epi -> mismatch eps* = 1.97 %
  sti_GPa      -0.3    STI oxide intrinsic stress (HDP oxide, compressive)
  liner_GPa    +/-     CESL nitride: -2.5 (PMOS, compressive), +1.5 (NMOS)
  gate_GPa     -1.0    TiN metal gate
  contact_GPa  +1.0    W contact fill
  cap_GPa      +1.5    SMT cap nitride (NMOS)
  smt_disl      1      SMT edge dislocations (0 = none): one per S/D at the
                       gate-side edge, b = a/2<110> = 0.384 nm, a half plane
                       missing above the core (mechanics/dislocation.py)
  disl_depth   20      core depth (nm)
Isotropic films get eps* from their stress (mechanics.elastic.film_eigenstrain);
SiGe gets the lattice mismatch; Si/SiGe are anisotropic, (001) wafer,
<110> channel.

Prints the channel stress (sigma_xx along the channel, sigma_yy vertical,
sigma_zz across the width, MPa; + tensile) after every step, total and per
stressor, and writes a history plot and stress maps.
"""
import argparse
import os

import numpy as np
from scipy import ndimage

from mechanics import elastic as el
from mechanics.fe2d import Grid2D, StressHistory2D

GEOM = dict(h=1.0, Lg=30, tox=2, gate_h=40, spacer=8, sd=44, sti=20, sti_depth=80,
            recess=40, raise_=6, liner=10, contact_w=16, hk=2, y_top=-80, y_bot=160, smt_depth=25)
DEFAULTS = dict(sige_x=0.5, sti_GPa=-0.3, liner_pmos_GPa=-2.5, liner_nmos_GPa=1.5,
                gate_GPa=-1.0, contact_GPa=1.0, cap_GPa=1.5, smt_disl=1.0, disl_depth=20.0)

MATS = {1: "Si", 2: "SiGe", 3: "SiO2", 4: "SiO2", 5: "Poly", 6: "Si3N4", 7: "Si3N4",
        8: "SiO2", 9: "HfO2", 10: "TiN", 11: "W", 12: "Si3N4"}
NAMES = {1: "Si", 2: "SiGe", 3: "STI", 4: "gate oxide", 5: "poly", 6: "spacer", 7: "liner",
         8: "ILD", 9: "HfO2", 10: "TiN", 11: "W", 12: "SMT cap"}


class Builder:
    """Label-grid process steps on one cell, recording each into a
    StressHistory2D."""

    def __init__(self, geom, par):
        self.p, self.v = dict(geom), dict(par)
        p = self.p
        self.Lx = 2 * (p["sti"] + p["sd"] + p["spacer"]) + p["Lg"]
        nx = int(round(self.Lx / p["h"]))
        ny = int(round((p["y_bot"] - p["y_top"]) / p["h"]))
        self.g = Grid2D(nx, ny, p["h"], 0.0, p["y_top"])
        self.X, self.Y = self.g.coords()
        mats = {k: el.material(n, x_ge=par["sige_x"] if n == "SiGe" else None) for k, n in MATS.items()}
        self.mats = mats
        self.sh = StressHistory2D(self.g, mats, void_ref=mats[1].C)
        self.lab = np.zeros((ny, nx), np.int32)
        xc = self.Lx / 2
        self.xc = xc
        self.gate = (abs(self.X - xc) < p["Lg"] / 2)
        self.spacer = (abs(self.X - xc) >= p["Lg"] / 2) & (abs(self.X - xc) < p["Lg"] / 2 + p["spacer"])
        self.sd = (abs(self.X - xc) >= p["Lg"] / 2 + p["spacer"]) & (self.X > p["sti"]) & (self.X < self.Lx - p["sti"])
        self.sti_x = (self.X < p["sti"]) | (self.X > self.Lx - p["sti"])
        self.channel = (abs(self.X - xc) < p["Lg"] / 2 - 3) & (self.Y > 0) & (self.Y < 4)
        self.rows = []

    def film(self, k, sigma0):
        return el.film_eigenstrain(self.mats[k], sigma0)

    def step(self, name, stress=None, relax=None):
        st = {}
        for s, (k_or_mask, eps) in (stress or {}).items():
            mask = (self.lab == k_or_mask) if np.isscalar(k_or_mask) else k_or_mask
            st[s] = (mask, eps)
        self.sh.step(name, self.lab.copy(), stressors=st, relax=relax)
        tot, per = self.sh.probe(self.channel)
        self.rows.append((name, tot * 1e3, {s: v * 1e3 for s, v in per.items()}))

    def conformal(self, k, d, where=None):
        """Deposit material k conformally, thickness d, on all exposed
        surfaces (void pixels within d of solid)."""
        void = self.lab == 0
        dist = ndimage.distance_transform_edt(void) * self.p["h"]
        m = void & (dist <= d)
        if where is not None:
            m &= where
        self.lab[m] = k
        return m

    def fill_void(self, k, y_min):
        m = (self.lab == 0) & (self.Y > y_min)
        self.lab[m] = k
        return m

    def cmp(self, y_level):
        self.lab[self.Y < y_level] = 0


def run_flow(flow, geom=GEOM, par=DEFAULTS):
    b = Builder(geom, par)
    p, v, Y, X = b.p, b.v, b.Y, b.X
    pmos = flow.startswith("pmos")
    gate_last = flow == "pmos_gate_last"
    b.lab[Y > 0] = 1
    b.step("substrate")
    b.lab[b.sti_x & (Y > 0) & (Y < p["sti_depth"])] = 0
    b.step("STI trench etch")
    b.lab[b.sti_x & (Y > 0) & (Y < p["sti_depth"])] = 3
    b.step("STI fill + CMP", {"STI": (3, b.film(3, v["sti_GPa"]))})
    stack = b.gate & (Y < 0) & (Y > -p["gate_h"])
    b.lab[b.gate & (Y < 0) & (Y > -p["tox"])] = 4
    if gate_last or flow == "nmos_smt":
        b.lab[stack & (Y < -p["tox"])] = 5
        b.step("gate oxide + poly gate" if flow == "nmos_smt" else "gate oxide + dummy poly")
    else:
        b.lab[b.gate & (Y <= -p["tox"]) & (Y > -p["tox"] - p["hk"])] = 9
        b.lab[stack & (Y <= -p["tox"] - p["hk"])] = 10
        b.step("HfO2 + TiN metal gate", {"gate metal": (10, b.film(10, v["gate_GPa"]))})
    b.lab[b.spacer & (Y < 0) & (Y > -p["gate_h"])] = 6
    b.step("spacers")
    if pmos:
        rec = b.sd & (Y > 0) & (Y < p["recess"])
        b.lab[rec] = 0
        b.step("S/D recess etch")
        f = el.sige_mismatch(v["sige_x"])
        b.lab[b.sd & (Y > -p["raise_"]) & (Y < p["recess"])] = 2
        b.step(f"SiGe S/D epi (x={v['sige_x']:.2f}, f={100 * f:.2f}%)", {"SiGe S/D": (2, [f, f, f, 0, 0, 0])})
    else:
        amorph = (b.sd & (Y > 0) & (Y < p["smt_depth"])) | (b.lab == 5)
        b.step("S/D + gate amorphization (PAI)")
        cap = b.conformal(12, p["liner"] + 10)
        b.step("SMT nitride cap", {"SMT cap": (12, b.film(12, v["cap_GPa"]))})
        b.step("anneal: S/D + poly recrystallize", relax=(amorph, 1.0))
        if v["smt_disl"]:
            # SMT edge dislocations: one per S/D at the gate-side edge, core at
            # disl_depth, a half plane (a/2<110>) missing between core and surface
            from mechanics.dislocation import B_SI, edge_dislocation
            msk = np.zeros(b.lab.shape, bool)
            eps = np.zeros((b.lab.size, 6))
            for side in (-1, 1):
                x_d = b.xc + side * (p["Lg"] / 2 + p["spacer"]) + side * 0.5 * p["h"]
                m, e = edge_dislocation(b.g, (x_d, v["disl_depth"]), (0.0, -1.0), (-B_SI, 0.0),
                                        solid=(b.lab == 1))
                msk |= m
                eps += e
            b.step(f"SMT dislocations (b = a/2<110>, core {v['disl_depth']:.0f} nm)",
                   {"dislocations": (msk, eps)})
        b.lab[cap] = 0
        b.step("SMT cap strip")
    sig_l = v["liner_pmos_GPa"] if pmos else v["liner_nmos_GPa"]
    b.conformal(7, p["liner"])
    b.step(f"CESL liner ({sig_l:+.1f} GPa)", {"liner": (7, b.film(7, sig_l))})
    b.fill_void(8, -p["gate_h"] - 30)
    b.step("ILD deposit")
    if gate_last:
        b.cmp(-p["gate_h"])
        b.step("ILD CMP (open the dummy gate)")
        b.lab[b.lab == 5] = 0
        b.step("dummy poly pull")
        trench = (b.lab == 0) & b.gate & (Y > -p["gate_h"])
        b.conformal(9, p["hk"], where=trench | ((b.lab == 0) & (Y > -p["gate_h"])))
        b.step("HfO2 liner (RMG)")
        m = (b.lab == 0) & (Y > -p["gate_h"]) & (abs(X - b.xc) < p["Lg"] / 2)
        b.lab[m] = 10
        b.step("TiN gate fill + CMP", {"gate metal": (10, b.film(10, v["gate_GPa"]))})
    cx = p["sti"] + p["sd"] / 2
    ct = ((abs(X - cx) < p["contact_w"] / 2) | (abs(X - (b.Lx - cx)) < p["contact_w"] / 2))
    top_sd = -p["raise_"] if pmos else 0.0
    hole = ct & (Y < top_sd) & np.isin(b.lab, [7, 8])          # through ILD + liner only
    b.lab[hole] = 0
    b.step("contact etch")
    b.lab[hole] = 11
    b.step("W contact fill", {"contact": (11, b.film(11, v["contact_GPa"]))})
    return b


def report(b, flow):
    lines = [f"\n=== {flow}: channel stress (MPa, + tensile; xx along channel, yy vertical, zz width)",
             f"{'step':42s} {'sxx':>8s} {'syy':>8s} {'szz':>8s}   change in sxx"]
    prev = 0.0
    for name, tot, per in b.rows:
        lines.append(f"{name:42s} {tot[0]:8.0f} {tot[1]:8.0f} {tot[2]:8.0f}   {tot[0] - prev:+8.0f}")
        prev = tot[0]
    name, tot, per = b.rows[-1]
    lines.append("final, by stressor:")
    for s, v in per.items():
        lines.append(f"  {s:40s} {v[0]:8.0f} {v[1]:8.0f} {v[2]:8.0f}")
    return "\n".join(lines)


def plot(b, flow, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r[0] for r in b.rows]
    stressors = sorted({s for r in b.rows for s in r[2]})
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    k = np.arange(len(names))
    for s in stressors:
        ax.plot(k, [r[2].get(s, np.zeros(6))[0] for r in b.rows], "-o", ms=3, label=s)
    ax.plot(k, [r[1][0] for r in b.rows], "k-o", lw=2.5, label="total")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(k)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("channel sigma_xx (MPa, + tensile)")
    ax.set_title(f"{flow}: channel stress along the flow")
    ax.legend(fontsize=8)
    ax = axs[1]
    sxx = b.sh.field("xx") * 1e3
    sxx = np.where(b.lab == 0, np.nan, sxx)
    lim = np.nanpercentile(np.abs(sxx[b.lab == 1]), 99)
    im = ax.imshow(sxx, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper",
                   extent=[0, b.Lx, b.Y.max() + 0.5, b.Y.min() - 0.5], interpolation="nearest")
    edges = np.zeros_like(b.lab, bool)
    edges[:-1] |= b.lab[:-1] != b.lab[1:]
    edges[:, :-1] |= b.lab[:, :-1] != b.lab[:, 1:]
    yy, xx = np.nonzero(edges)
    ax.plot(b.X[yy, xx], b.Y[yy, xx], ",", color="k", alpha=0.6)
    ax.set_ylim(70, -60)
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.set_title("final sigma_xx (MPa)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out, f"{flow}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flow", nargs="*", default=["pmos_gate_first", "pmos_gate_last", "nmos_smt"])
    ap.add_argument("--out", default="out/stress")
    ap.add_argument("--set", nargs="*", default=[], help="override a stressor input, e.g. sige_x=0.3")
    a = ap.parse_args()
    par = dict(DEFAULTS)
    for kv in a.set:
        k, val = kv.split("=")
        par[k] = float(val)
    os.makedirs(a.out, exist_ok=True)
    import time
    for flow in a.flow:
        t = time.perf_counter()
        b = run_flow(flow, par=par)
        txt = report(b, flow)
        print(txt)
        print(f"({len(b.rows)} steps, {b.g.nx} x {b.g.ny} pixels, {time.perf_counter() - t:.1f} s)  "
              f"-> {plot(b, flow, a.out)}")
        with open(os.path.join(a.out, f"{flow}.txt"), "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
