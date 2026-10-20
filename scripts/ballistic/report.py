"""Tables (markdown) and plots from scripts/ballistic/lg_sweep.py results.

  python3 scripts/ballistic/report.py out/ballistic_lg/results.json [--plot docs/img/ballistic_lg_2d.png]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

TAGS = ["off", "bal", "str", "str_bal"]
NAMES = {"off": "no stress, ballistic off", "bal": "no stress, ballistic ON",
         "str": "stress, ballistic off", "str_bal": "stress, ballistic ON"}


def _name(tag):
    base, _, suf = tag.partition("_tot") if "_tot" in tag else (tag, "", "")
    return NAMES.get(base, base) + (" (total)" if "_tot" in tag else "")


def _ref(tag):
    """The ballistic-off case a ballistic-on tag compares with."""
    return "str" if tag.startswith("str") else "off"


def load(paths):
    R = defaultdict(dict)
    for path in paths:
        for r in json.load(open(path)):
            R[(r["kind"], r["tag"])][float(r["Lg"])] = r
            if r["tag"] not in TAGS:
                TAGS.append(r["tag"])
    return R


def f(v, fmt="{:.0f}"):
    return "-" if v is None else fmt.format(v)


def ratio(d, a, b, key="Ion_uA_um"):
    try:
        return d[a][key] / d[b][key]
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def tables(R, kinds=("nmos", "pmos")):
    out = []
    for kind in kinds:
        out.append(f"\n### {kind.upper()}\n")
        out.append("| case | Lg nm | Vt,sat V | DIBL mV/V | Ion uA/um | v_src 1e6 cm/s | S = dlnIon/dlnmu | "
                   "mu_app(0.4) | mu_app ON/OFF | Matthiessen | sigma_xx MPa | gamma |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for tag in TAGS:
            d = R.get((kind, tag))
            if not d:
                continue
            ref = R.get((kind, _ref(tag))) if "bal" in tag else None
            for L in sorted(d):
                r = d[L]
                on_off = pred = None
                if ref and L in ref and r.get("mu_app_0.4") and ref[L].get("mu_app_0.4"):
                    on_off = r["mu_app_0.4"] / ref[L]["mu_app_0.4"]
                    mb, mc = r.get("mu_bal"), ref[L].get("mu_ch_0.4")
                    gg = r.get("gamma_n" if kind == "nmos" else "gamma_p")
                    if mb and gg:
                        mb = mb * gg ** 0.5              # mu_bal scaled by gamma^s, s = 0.5
                    if mb and mc:
                        pred = mb / (mc + mb)
                g = r.get("gamma_n" if kind == "nmos" else "gamma_p")
                out.append(f"| {_name(tag)} | {L:g} | {f(r.get('Vt_sat'), '{:+.3f}')} | {f(r.get('DIBL_mV_V'))} | "
                           f"{f(r.get('Ion_uA_um'))} | {f((r.get('v_src_cm_s') or 0) / 1e6 or None, '{:.2f}')} | "
                           f"{f(r.get('S_mu'), '{:.2f}')} | {f(r.get('mu_app_0.4'))} | {f(on_off, '{:.3f}')} | "
                           f"{f(pred, '{:.3f}')} | {f((r.get('sigma_ch_MPa') or [None])[0])} | {f(g, '{:.2f}')} |"
                           + (f" FAILED {r['failed']}" if r.get("failed") else ""))
        out.append("")
        out.append("| case | Ion(15)/Ion(30) | Ion(12)/Ion(15) | Ion(20)/Ion(30) | Ion(30)/Ion(100) |")
        out.append("|---|---|---|---|---|")
        for tag in TAGS:
            d = R.get((kind, tag))
            if not d:
                continue
            out.append(f"| {_name(tag)} | {f(ratio(d, 15.0, 30.0), '{:.3f}')} | {f(ratio(d, 12.0, 15.0), '{:.3f}')} | "
                       f"{f(ratio(d, 20.0, 30.0), '{:.3f}')} | {f(ratio(d, 30.0, 100.0), '{:.3f}')} |")
    return "\n".join(out)


def plot(R, path, kinds=("nmos", "pmos"), title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(kinds), 3, figsize=(15, 4.2 * len(kinds)), squeeze=False)
    sty = {"off": ("k", "o", "-"), "bal": ("C0", "s", "-"), "str": ("C3", "o", "--"), "str_bal": ("C1", "s", "--")}
    for i, kind in enumerate(kinds):
        for tag in TAGS:
            d = R.get((kind, tag))
            if not d:
                continue
            L = np.array(sorted(d))
            c, m, ls = sty[tag]
            Ion = np.array([d[l].get("Ion_uA_um") or np.nan for l in L])
            mu = np.array([d[l].get("mu_app_0.4") or np.nan for l in L])
            ax[i, 0].plot(L, Ion, color=c, marker=m, ls=ls, label=_name(tag))
            ax[i, 1].plot(L, mu, color=c, marker=m, ls=ls, label=_name(tag))
            if tag.startswith("str"):
                s = np.array([(d[l].get("sigma_ch_MPa") or [np.nan])[0] for l in L])
                g = np.array([d[l].get("gamma_n" if kind == "nmos" else "gamma_p") or np.nan for l in L])
                if tag == "str":
                    ax[i, 2].plot(L, s, color="C3", marker="o", label="channel sigma_xx (MPa)")
                    a2 = ax[i, 2].twinx()
                    a2.plot(L, g, color="C2", marker="^", ls=":", label="piezo factor gamma")
                    a2.set_ylabel("gamma (mu_low factor)")
                    a2.legend(loc="lower right", fontsize=8)
        ax[i, 0].set_ylabel(f"{kind.upper()} Ion (uA/um) at |Vgs-Vt,sat| = 0.45 V, |Vds| = 0.7 V")
        ax[i, 1].set_ylabel("apparent mobility (cm2/Vs), |Vds| 50 mV, overdrive 0.4 V")
        ax[i, 2].set_ylabel("channel sigma_xx (MPa)")
        for a in ax[i]:
            a.set_xscale("log")
            a.set_xlabel("gate length Lg (nm)")
            a.grid(alpha=0.3, which="both")
        ax[i, 0].legend(fontsize=8)
        ax[i, 2].legend(loc="upper right", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--kinds", nargs="*", default=["nmos", "pmos"])
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    R = load(a.results)
    print(tables(R, a.kinds))
    if a.plot:
        plot(R, a.plot, a.kinds, a.title)


if __name__ == "__main__":
    main()
