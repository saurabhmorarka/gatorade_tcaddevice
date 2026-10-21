"""Series-resistance partition of a solved 3D transistor state - the default
by-product of every `fab simulate` run (DEVELOPMENT_LOG 87, docs/resistance_partition.md).

Method (quasi-Fermi dissipation). Current is conserved, so the power the
drain delivers, I * Vds, is dissipated somewhere. Every mesh edge dissipates
I_e (phi_i - phi_j) (Scharfetter-Gummel edge current times the drop of the
carrier's quasi-Fermi potential, electrons and holes both) and every
sub-grid Schottky contact node I_node (V_contact - phi_node). The sum is
I * Vds (`closure`, ~1), and a region's resistance is R = P_region / I^2:
additive, no cut planes, no flat-equipotential assumption. Regions are
x-slabs about the gate:

  channel  between the gate edges (xg0 .. xg1)
  spacer   under the spacer on each side (extension / fin under it)
  epi      beyond the spacer up to the contact (epi, fin, sub-fin Si)
  contact  the sub-grid metal / semiconductor barrier

It costs nothing extra: it reads a state the Id-Vg run has already solved.
Resistances are per fin / per stack of the real device (the half-width solve
is doubled) and, as R * W, in Ohm um with the template's effective width W.
"""
import json
import os

import numpy as np

REGIONS = ("channel", "spacer", "epi", "contact")
COLORS = {"channel": "#d95f02", "spacer": "#7570b3", "epi": "#1b9e77", "contact": "#e7298a"}
PART_VGS = 0.7        # |Vgs| (V) of the state the report is taken at (nearest solved point)


def region_of_x(x, xg0, xg1, spacer):
    """'channel' / 'spacer' / 'epi' of an x coordinate (nm), array-wise."""
    xm = 0.5 * (xg0 + xg1)
    half = 0.5 * (xg1 - xg0)
    dx = np.abs(np.asarray(x, float) - xm)
    return np.where(dx <= half, "channel", np.where(dx <= half + spacer, "spacer", "epi"))


def partition(d, r, vds, bins_nm=2.0):
    """The dissipation partition of the converged state r of device context d
    at drain bias vds: a dict with I_A, P_W, closure, R_total, R_<region> and
    R_<source|drain>_<region> (Ohm, whole device), per-contact diagnostics, and
    the x-resolved dissipation profile."""
    from solver2d.newton_solver_qf_2d import edge_currents
    from solver2d.current import contact_current
    mesh, mat, sk, p = d["mesh"], d["mat"], d.get("schottky"), d["params"]
    In, Ip = edge_currents(mesh, mat, r["psi"], r["n"], r["p"], r["phin"], r["phip"],
                           r.get("velocity_saturation", False), qs=r.get("qshift"))
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    Pe = In * (r["phin"][ii] - r["phin"][jj]) + Ip * (r["phip"][ii] - r["phip"][jj])
    X = mesh.points[:, 0] * 1e7
    xm = 0.5 * (X[ii] + X[jj])
    xg0, xg1, sp = float(p["xg0"]), float(p["xg1"]), float(p["spacer"])
    side = np.where(xm < 0.5 * (xg0 + xg1), "source", "drain")
    reg = region_of_x(xm, xg0, xg1, sp)
    P = {(s, g): float(Pe[(side == s) & (reg == g)].sum()) for s in ("source", "drain") for g in REGIONS[:3]}
    edges_x = np.arange(0.0, X.max() + bins_nm, bins_nm)
    prof = np.histogram(xm, bins=edges_x, weights=Pe)[0]
    ncon = {}
    if sk is not None:
        bias = {"source": 0.0, "drain": vds}
        *_, I_node = sk._rates(r["psi"], r["phin"], r["phip"], bias, False)
        for nm, s in sk.specs.items():
            i = s["nodes"]
            phi = r["phin"][i] if s["carrier"] == "n" else r["phip"][i]
            u = bias[nm] - phi
            P[(nm, "contact")] = float(np.sum(I_node[nm] * u))
            ncon[nm] = dict(nodes=int(len(i)), area_nm2=float(sk.area[i].sum() * 1e14),
                            u_mean_mV=float(np.average(u, weights=np.abs(I_node[nm]) + 1e-30) * 1e3),
                            N_mean=float(np.average(np.abs(mesh.Cdop[i]), weights=sk.area[i])))
    I = abs(contact_current(mesh, mat, r, "drain"))            # A, solved half
    half = 2.0 if d["half_z"] else 1.0
    tot = sum(P.values())
    out = dict(I_A=I * half, Vds=vds, P_W=tot, closure=tot / (abs(vds) * I) if I else None,
               R_total=abs(vds) / (I * half) if I else None, contacts=ncon)
    for (s, g), v in P.items():
        out[f"R_{s}_{g}"] = v / I ** 2 / half if I else None
    for g in REGIONS:
        out[f"R_{g}"] = sum(out.get(f"R_{s}_{g}") or 0.0 for s in ("source", "drain"))
    out["W_um"] = (d["W_nm"] or 1e3) * 1e-3
    out["profile_x_nm"] = edges_x[:-1].tolist()
    out["profile_P_W"] = (prof / I ** 2 / half).tolist()
    return out


def plot_partition(d, res, path, title=""):
    """Figure: (a) side view of the symmetry-plane cut with every semiconductor
    node coloured by the region the partition assigned it to (Schottky contact
    nodes marked), (b) the x-resolved dissipation (Ohm per nm) with the regions
    shaded, (c) the resistance of each region, source / drain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mesh, p = d["mesh"], d["params"]
    xg0, xg1, sp = float(p["xg0"]), float(p["xg1"]), float(p["spacer"])
    pts = mesh.points * 1e7
    semi = ~np.asarray(mesh.is_insulator, bool)
    zc = pts[semi, 2].max()
    cut = semi & (pts[:, 2] >= zc - 0.75)                 # the plane through the fin / sheet centre
    names = region_of_x(pts[cut, 0], xg0, xg1, sp)
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    a = fig.add_subplot(gs[0, :])
    for g in REGIONS[:3]:
        m = names == g
        a.scatter(pts[cut][m, 0], pts[cut][m, 1], s=3, c=COLORS[g], label={"epi": "epi / S-D (beyond spacer)"}.get(g, g))
    sk = d.get("schottky")
    if sk is not None:
        cn = np.concatenate([s["nodes"] for s in sk.specs.values()])
        cn = cn[pts[cn, 2] >= zc - 0.75]
        a.scatter(pts[cn, 0], pts[cn, 1], s=9, c=COLORS["contact"], marker="s", label="contact (Schottky nodes)")
    for x in (xg0 - sp, xg0, xg1, xg1 + sp):
        a.axvline(x, color="k", lw=0.6, ls=":")
    a.set_xlabel("x along the channel (nm)")
    a.set_ylabel("height (nm)")
    a.set_aspect("equal")
    a.legend(loc="upper right", fontsize=8, markerscale=3)
    a.set_title("region map: semiconductor nodes in the symmetry-plane cut, coloured by partition region", fontsize=9)
    b = fig.add_subplot(gs[1, 0])
    x = np.asarray(res["profile_x_nm"])
    pr = np.asarray(res["profile_P_W"])
    b.fill_between(x + 0.5, pr, step=None, color="0.4")
    for lo, hi, g in ((xg0, xg1, "channel"), (xg0 - sp, xg0, "spacer"), (xg1, xg1 + sp, "spacer")):
        b.axvspan(lo, hi, color=COLORS[g], alpha=0.25)
    b.set_xlabel("x (nm)")
    b.set_ylabel("dissipation / I^2  (Ohm per nm)")
    b.set_title("where the resistance sits (contact barrier not shown)", fontsize=9)
    c = fig.add_subplot(gs[1, 1])
    w = 0.38
    for k, g in enumerate(REGIONS):
        s_, d_ = res.get(f"R_source_{g}") or 0.0, res.get(f"R_drain_{g}") or 0.0
        c.bar(k - w / 2, s_, w, color=COLORS[g], alpha=0.65)
        c.bar(k + w / 2, d_, w, color=COLORS[g])
        c.text(k, max(s_, d_) * 1.02, f"{100 * (s_ + d_) / res['R_total']:.0f}%", ha="center", fontsize=8)
    c.set_xticks(range(len(REGIONS)), REGIONS)
    c.set_ylabel("R (Ohm)  light: source, dark: drain")
    c.set_title(f"R_total {res['R_total']:.0f} Ohm = {res['R_total'] * res['W_um']:.0f} Ohm um   closure {res['closure']:.3f}",
                fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(d, states, vds, out_dir, label=""):
    """Pick the solved state of `states` ({Vgs: state}) nearest |Vgs| = PART_VGS
    on the on-state side, partition it, write <out_dir>/partition_Vds<vds>.json
    and .png. Returns the result dict (or None when the device has no usable
    state / template parameters)."""
    p = d["params"]
    if not all(k in p for k in ("xg0", "xg1", "spacer")):
        return None
    sgn = -1.0 if vds < 0 else 1.0
    cand = [v for v in states if v * sgn > 0]
    if not cand:
        return None
    v = min(cand, key=lambda x: abs(abs(x) - PART_VGS))
    res = partition(d, states[v], vds)
    res.update(Vgs=v, Id_uA_um=res["I_A"] / res["W_um"] * 1e6)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"partition_Vds{vds:g}")
    json.dump(res, open(base + ".json", "w"), indent=1, default=float)
    plot_partition(d, res, base + ".png",
                   f"{label} Vds {vds:+g} V, Vgs {v:+.2f} V: Id {res['Id_uA_um']:.1f} uA/um")
    return res


def summary_line(res):
    R = res["R_total"]
    return (f"R_total {R:.0f} Ohm ({R * res['W_um']:.0f} Ohm um): "
            + ", ".join(f"{g} {100 * res['R_' + g] / R:.0f}%" for g in REGIONS) + f"  (closure {res['closure']:.3f})")
