"""Where the current goes at a Schottky S/D contact: band diagram with the
tunneling paths (the barrier thickness at each energy), the energy-resolved
current split into tunneling and emission over the barrier, and how both
change with the S/D doping. NMOS (n+ Si:P) and PMOS (p+ strained
Si0.5Ge0.5:B) reference contacts, TiN-like Schottky-Mott barriers
(DEVELOPMENT_LOG 60, Study A).

Run: .venv/bin/python -m contacts.plot_schottky_current [--out out/schottky]
"""
import argparse
import os

import numpy as np

from contacts.schottky import SchottkyContact
from contacts import newton_schottky_1d as ns
from contacts.study_ti_barriers import materials

V_SMALL = 0.02          # V, forward, inside the linear (rho_c) regime
DEVICES = (
    # label, material key, doping, contact, bias
    ("NMOS: TiN on n+ Si:P 7e20", "si", 7e20, SchottkyContact(barrier_n_eV=0.55), V_SMALL),
    ("PMOS: TiN on p+ Si0.5Ge0.5:B 5e20", "sige", -5e20, SchottkyContact(barrier_p_eV=0.20), -V_SMALL),
)
DOPINGS = np.logspace(18, np.log10(2e21), 14)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def solve_point(mat, doping, contact, V):
    x = ns.contact_grid(30e-7)
    r = ns.sweep(x, np.full(len(x), doping), mat, {"left": contact, "right": None}, [V])[0]
    return r, ns.spectrum(r)


def to_electron_energy(sp):
    """Holes are computed in hole energies (-Ev); plot in electron energy."""
    if sp["carrier"] == "n":
        return sp
    out = dict(sp)
    for k in ("U", "EF", "E_tun", "E_over", "barrier"):
        out[k] = -np.asarray(sp[k])
    return out


def band_panel(ax, sp, title):
    e = to_electron_energy(sp)
    s_nm = e["s"] * 1e7
    keep = s_nm <= 6.0
    ax.axvspan(-1.0, 0.0, color=MUTED, alpha=0.25, lw=0)
    ax.text(-0.5, 0.97, "metal", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK, fontsize=8)
    ax.plot([-1.0, 0.0], [0.0, 0.0], color=INK, lw=1.2)
    ax.text(-0.95, 0.012, "E$_F$ metal", color=INK, fontsize=7, va="bottom")
    band = "E$_c$" if sp["carrier"] == "n" else "E$_v$"
    ax.plot(s_nm[keep], e["U"][keep], color=INK, lw=2)
    ax.annotate(band, (s_nm[keep][-1], e["U"][keep][-1]), textcoords="offset points",
                xytext=(-4, 6 if sp["carrier"] == "p" else -12), ha="right", fontsize=8, color=INK)
    ax.plot(s_nm[keep], e["EF"][keep], color=INK, lw=1, ls="--")
    ax.annotate("semiconductor quasi-Fermi level", (s_nm[keep][-1], e["EF"][keep][-1]),
                textcoords="offset points", xytext=(-4, 4 if sp["carrier"] == "n" else -11), ha="right",
                fontsize=7, color=MUTED)
    # Tunneling paths: one horizontal bar per energy bin, from the metal to
    # its turning point; opacity ~ that bin's share of the current.
    j = np.abs(sp["j_tun"])
    jmax = j.max() if j.max() > 0 else 1.0
    for E, st, jj in zip(e["E_tun"], sp["s_turn"], j):
        if np.isfinite(st) and jj > 1e-3 * jmax:
            ax.plot([0.0, st * 1e7], [E, E], color=BLUE, lw=1.6, alpha=0.15 + 0.85 * jj / jmax,
                    solid_capstyle="butt")
    k = int(np.argmax(j))
    st_pk = sp["s_turn"][k] * 1e7
    ax.annotate(f"peak tunneling path\n{st_pk:.2f} nm thick", (st_pk, e["E_tun"][k]), textcoords="offset points",
                xytext=(18, 12 if sp["carrier"] == "n" else -40), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    phi = abs(sp["barrier"])
    ax.annotate(f"barrier {phi:.2f} eV", (0.0, e["barrier"]), textcoords="offset points",
                xytext=(10, 4 if sp["carrier"] == "n" else -12), fontsize=8, color=INK)
    ax.set_xlim(-1.0, 6.0)
    ax.set_xlabel("distance from the metal (nm)", color=INK)
    ax.set_ylabel("energy - E$_F$ metal (eV)", color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


FLOOR = 1e-5            # spectrum panel: log axis floor, relative to the peak


def spectrum_panel(ax, sp, ylim):
    e = to_electron_energy(sp)
    jt = np.abs(sp["j_tun"])
    jo = np.abs(sp["j_over"])
    scale = max(jt.max(), jo.max())
    for E, j, col in ((e["E_tun"], jt, BLUE), (e["E_over"], jo, ORANGE)):
        v = np.maximum(j / scale, FLOOR)
        order = np.argsort(E)
        ax.fill_betweenx(E[order], FLOOR, v[order], color=col, alpha=0.35, lw=0)
        ax.plot(v[order], E[order], color=col, lw=2)
    ax.axhline(e["barrier"], color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log")
    tot = abs(sp["J_tun"]) + abs(sp["J_over"])
    ft = abs(sp["J_tun"]) / tot
    ax.text(0.03, 0.5, f"tunneling {100 * ft:.3g}%\nover the barrier {100 * (1 - ft):.2g}%\n"
            f"|J| = {tot:.2e} A/cm$^2$", transform=ax.transAxes, ha="left", va="center", fontsize=8, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.set_ylim(*ylim)
    ax.set_xlim(FLOOR, 2.0)
    ax.set_xlabel("current per energy (relative to peak, log)", color=INK)
    ax.set_title("energy-resolved current", fontsize=10, color=INK, loc="left")


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/schottky")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = dict(zip(("si", "sige"), materials()))
    fig = plt.figure(figsize=(11, 12.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1.0], hspace=0.42, wspace=0.08)
    for row, (label, mk, dop, c, V) in enumerate(DEVICES):
        _, sp = solve_point(mats[mk], dop, c, V)
        e = to_electron_energy(sp)
        ax_b = fig.add_subplot(gs[row, 0])
        band_panel(ax_b, sp, f"{label}  (contact bias {V:+.2f} V)")
        jmax = max(np.abs(sp["j_tun"]).max(), np.abs(sp["j_over"]).max())
        E_over_vis = e["E_over"][np.abs(sp["j_over"]) > FLOOR * jmax]
        Es = np.concatenate([e["U"][e["s"] * 1e7 <= 6], e["E_tun"], E_over_vis, [0.0]])
        lo, hi = Es.min() - 0.05, Es.max() + 0.05
        ax_b.set_ylim(lo, hi)
        ax_s = fig.add_subplot(gs[row, 1], sharey=ax_b)
        spectrum_panel(ax_s, sp, (lo, hi))
        plt.setp(ax_s.get_yticklabels(), visible=False)
        for ax in (ax_b, ax_s):
            style(ax)
        if row == 0:
            from matplotlib.patches import Patch
            ax_s.legend(handles=[Patch(color=BLUE, alpha=0.6, label="tunneling (field emission)"),
                                 Patch(color=ORANGE, alpha=0.6, label="over the barrier (thermionic)")],
                        fontsize=8, frameon=True, framealpha=0.9, edgecolor="none", loc="upper right",
                        bbox_to_anchor=(1.0, 0.86))

    # Doping sweeps: tunneling share and barrier thickness, both contacts.
    ax_f = fig.add_subplot(gs[2, 0])
    ax_t = fig.add_subplot(gs[2, 1])
    for (label, mk, dop, c, V), col, name in zip(DEVICES, (BLUE, ORANGE), ("NMOS", "PMOS")):
        frac, thick = [], []
        for N in DOPINGS:
            _, sp = solve_point(mats[mk], np.sign(dop) * N, c, V)
            jt, jo = abs(sp["J_tun"]), abs(sp["J_over"])
            frac.append(jt / (jt + jo))
            w = np.abs(sp["j_tun"])
            ok = np.isfinite(sp["s_turn"]) & (w > 0)
            thick.append(np.sum(w[ok] * sp["s_turn"][ok]) / np.sum(w[ok]) * 1e7 if ok.any() else np.nan)
        ax_f.semilogx(DOPINGS, 100 * np.array(frac), color=col, lw=2, marker="o", ms=4)
        ax_f.annotate(name, (DOPINGS[0], 100 * frac[0]), textcoords="offset points", xytext=(4, 6), color=INK,
                      fontsize=8)
        ax_t.semilogx(DOPINGS, thick, color=col, lw=2, marker="o", ms=4)
        kpk = int(np.nanargmax(thick))
        ax_t.annotate(name, (DOPINGS[kpk], thick[kpk]), textcoords="offset points", xytext=(6, 4), color=INK,
                      fontsize=8)
    ax_f.set_xlabel("S/D doping at the contact (cm$^{-3}$)", color=INK)
    ax_f.set_ylabel("share of the current that tunnels (%)", color=INK)
    ax_f.set_ylim(-3, 103)
    ax_f.set_title("tunneling vs over the barrier, vs doping", fontsize=10, color=INK, loc="left")
    ax_t.set_xlabel("S/D doping (cm$^{-3}$)", color=INK)
    ax_t.set_ylabel("barrier thickness seen by the current (nm)", color=INK)
    ax_t.set_title("current-weighted tunneling distance", fontsize=10, color=INK, loc="left")
    for ax in (ax_f, ax_t):
        style(ax)
    fig.text(0.01, 0.005, "1D Schottky model (contacts/): thermionic-emission boundary + nonlocal WKB/Tsu-Esaki "
             "tunneling through the self-consistent barrier. NMOS Phi_Bn 0.55 eV, PMOS Phi_Bp 0.20 eV "
             "(TiN, Schottky-Mott).", fontsize=7.5, color=MUTED)
    path = os.path.join(args.out, "schottky_barrier_current.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print("figure:", path)


if __name__ == "__main__":
    main()
