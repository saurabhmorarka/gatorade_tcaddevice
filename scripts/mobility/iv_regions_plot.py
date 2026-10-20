import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1]; R = json.load(open(D + "/results.json"))
G = {n: r for n, vs, k, r in R if k == "idvg"}
V = {(n, vs): r for n, vs, k, r in R if k == "idvd"}
arr = lambda r, key: (np.array([float(k) for k in r]), np.array([v[key] for v in r.values()]))
names = list(G); cols = ["k", "C0", "C2", "C1", "C3"]
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
for n, c in zip(names, cols):
    vg, id_ = arr(G[n], "Id"); _, ni = arr(G[n], "Ninv"); _, mu = arr(G[n], "mu")
    ax[0, 0].semilogy(vg, id_, "o-", ms=3, color=c, label=n)
    ax[0, 1].plot(vg, id_, "o-", ms=3, color=c, label=n)
    s = ni > 1e10
    ax[0, 2].semilogx(ni[s], mu[s], "o-", ms=3, color=c, label=n)
# incremental effect of each term, vs Ninv (removes the Vt shift of Qit)
for i in range(1, len(names)):
    a, b = G[names[min(i - 1, 2)]], G[names[i]]     # each Coulomb case against the phonon + SR curve
    _, na = arr(a, "Ninv"); _, ia = arr(a, "Id"); _, nb = arr(b, "Ninv"); _, ib = arr(b, "Id")
    s = (nb > 1e9) & (nb < na.max())
    ref = np.exp(np.interp(np.log(nb[s]), np.log(na), np.log(ia)))
    ax[1, 0].semilogx(nb[s], ib[s] / ref, "o-", ms=3, color=cols[i], label=f"{names[i]} / {names[min(i - 1, 2)]}")
for (n, vs), r in V.items():
    vd, id_ = arr(r, "Id")
    ax[1, 1].plot(vd, id_, "-" if vs else "--", color="k" if n.startswith("bulk") else "C2",
                  label=f"{n}, vsat {'on' if vs else 'off'}")
for n in ("bulk (Masetti)", "+ surface roughness"):
    vd, a = arr(V[(n, False)], "Id"); _, b = arr(V[(n, True)], "Id")
    ax[1, 2].plot(vd[1:], b[1:] / a[1:], "o-", ms=3, color="k" if n.startswith("bulk") else "C2", label=n)
T = [("Id-Vg, Vds 50 mV (log)", "Vgs (V)", "Id (uA/um)"), ("Id-Vg, Vds 50 mV (linear)", "Vgs (V)", "Id (uA/um)"),
     ("mid-channel mu_eff vs inversion charge", "N_inv (cm^-2)", "mu_eff (cm^2/Vs)"),
     ("each term's incremental effect on Id (same N_inv)", "N_inv (cm^-2)", "Id ratio"),
     ("Id-Vd, Vgs 1.5 V: dashed vsat off, solid on", "Vds (V)", "Id (uA/um)"),
     ("velocity saturation: Id(on)/Id(off), Vgs 1.5 V", "Vds (V)", "Id ratio")]
for a, (t, x, y) in zip(ax.flat, T):
    a.set_title(t, fontsize=10); a.set_xlabel(x); a.set_ylabel(y); a.grid(True, which="both", alpha=0.3); a.legend(fontsize=7)
fig.suptitle("2D NMOS, Lg 0.5 um, tox 5 nm, Na 1e16 - cumulative mobility stack (fast mode)", fontsize=12)
fig.savefig(D + "/mobility_regions.png", dpi=110)
# numbers
for n in names:
    vg, id_ = arr(G[n], "Id"); _, ni = arr(G[n], "Ninv"); _, mu = arr(G[n], "mu")
    print(f"{n:24s}", " ".join(f"{v:.1f}:{i:.3g}:{m:.0f}" for v, i, m in zip(vg, id_, mu) if v in (0.2, 0.5, 0.8, 1.2, 2.0)))
