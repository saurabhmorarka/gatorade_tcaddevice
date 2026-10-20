"""Do the mobility terms still act on the same part of the I-V once the
ballistic term is on in a short channel? (docs/ballistic.md, DEVELOPMENT_LOG 85)

The cumulative stack of scripts/mobility/iv_regions.py (bulk Masetti,
+ Lombardi phonon, + surface roughness, + the Qit 3e11 cm^-2 sheet charge
alone (electrostatics), + its Coulomb scattering) on the scalable 2D planar NMOS of scripts/ballistic/lg_sweep.py
(1 nm oxide, body 6e18, 5 nm S/D extensions, no process stress) at Lg 500
(the long-channel reference), 30 and 15 nm, each with the ballistic term
off and on. Each step is compared at a FIXED Vgs: the mobility terms do not move
the electrostatics, so the Id ratio is the term's own effect (the sheet
charge is its own step). Per device and case:

  * Id-Vg at Vds 0.05 V and 0.7 V (Vgs 0..1.2 V);
  * Id-Vd at Vgs 1.0 V with velocity saturation on, and (bulk and the full
    stack only) with it off: the vsat term's region.

  ~/miniconda3/envs/tcad/bin/python scripts/ballistic/iv_regions_ballistic.py out/ballistic_iv
  ~/miniconda3/envs/tcad/bin/python scripts/ballistic/iv_regions_ballistic.py out/ballistic_iv --plot docs/img/ballistic_iv_regions.png
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LGS = (500.0, 30.0, 15.0)
VGS = [round(v, 3) for v in np.arange(0.05, 1.2001, 0.05)]
VDS = [round(v, 3) for v in np.arange(0.05, 1.0001, 0.05)]
VG_IDVD = 1.0
OVS = (0.05, 0.3, 0.6)          # overdrives of the summary table (V)
CASES = {   # cumulative, as scripts/mobility/iv_regions.py (Qit only on the gate dielectric)
    "bulk": "off",
    "+phonon": {"mode": "fast", "delta_scale_n": 1e6, "delta_scale_p": 1e6, "interface_charge": 0},
    "+SR": {"mode": "fast", "interface_charge": 0},
    # the 3e11 sheet charge alone (electrostatics: a -14 mV Vt shift at 1 nm oxide), its
    # scattering switched off by a huge mu1; then the Coulomb scattering on top
    "+Qit charge": {"mode": "fast", "interface_charge": 3e11, "coulomb_mu1": 1e12},
    "+Coulomb": {"mode": "fast", "interface_charge": 3e11},
}


def config(out, Lg, case, bal, vsat=True):
    from lg_sweep import device_config
    cfg = device_config("nmos", Lg, None, {} if bal else None, vsat=vsat)
    cfg.setdefault("physics", {})["mobility"] = CASES[case]
    p = os.path.join(out, "cfg", f"Lg{Lg:g}_{case}_{'bal' if bal else 'off'}{'' if vsat else '_novsat'}.yaml")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    yaml.safe_dump(cfg, open(p, "w"))
    return p


def run(task):
    warnings.simplefilter("ignore")
    import main2d_mosfet_sweep as M
    Lg, case, bal, kind, vsat, out = task
    t0 = time.perf_counter()
    mesh, mat, dev, Cs, eq, vs = M._context(config(out, Lg, case, bal, vsat))

    def at(Vgs, Vds):
        return lambda p, g: M.solve_mosfet(mesh, mat, dev, Cs, *((p, Vds) if Vgs is None else (Vgs, p)),
                                            psi_init=g["psi"], phin_init=g["phin"], phip_init=g["phip"],
                                            cold_retry=False, velocity_saturation=vs)
    res = {}
    if kind.startswith("idvg"):
        Vds = 0.05 if kind == "idvg_lin" else 0.7
        ramp, _, _ = M.continuation(at(0.0, None), 0.0, eq, [Vds])
        st, _, _ = M.continuation(at(None, Vds), 0.0, ramp[Vds], VGS, psi_only=True)
    else:
        gv, _, _ = M.continuation(at(None, 0.0), 0.0, eq, [VG_IDVD], psi_only=True)
        st, _, _ = M.continuation(at(VG_IDVD, None), 0.0, gv[VG_IDVD], VDS)
    for v, r in st.items():
        res[f"{v:.3f}"] = abs(M._id_per_um(mesh, mat, r)) * 1e6           # uA/um
    missing = [v for v in (VGS if kind.startswith("idvg") else VDS) if f"{v:.3f}" not in res]
    return dict(Lg=Lg, case=case, bal=bal, kind=kind, vsat=vsat, Id=res, missing=missing,
                time_s=time.perf_counter() - t0)


def tasks(out):
    T = []
    for Lg in LGS:
        for bal in (False, True):
            for case in CASES:
                for kind in ("idvg_lin", "idvg_sat", "idvd"):
                    T.append((Lg, case, bal, kind, True, out))
                if case in ("bulk", "+Coulomb"):
                    T.append((Lg, case, bal, "idvd", False, out))
    return T


def key(r):
    return f"{r['Lg']:g}|{r['case']}|{int(r['bal'])}|{r['kind']}|{int(r['vsat'])}"


# --- plot --------------------------------------------------------------------------------
def vt_cc(x, y, Lg):
    """Constant-current Vt (100 nA W/L, y in uA/um), log-interpolated."""
    I_cc = 0.1 / (Lg * 1e-3)
    k = np.flatnonzero(y >= I_cc)
    if not len(k) or k[0] == 0:
        return None
    k = k[0]
    a, b = np.log(y[k - 1]), np.log(y[k])
    return float(x[k - 1] + (x[k] - x[k - 1]) * (np.log(I_cc) - a) / (b - a))


def step_vgs(curve, Lg, a, b, bal, kind, vsat=True):
    """(x, Id change %) of going from case a to b at the same bias."""
    x0, y0 = curve(Lg, a, bal, kind, vsat)
    x1, y1 = curve(Lg, b, bal, kind, vsat)
    if x0 is None or x1 is None:
        return None, None
    xs = np.intersect1d(x0, x1)
    return xs, 100 * (y1[np.isin(x1, xs)] / y0[np.isin(x0, xs)] - 1)


def plot(out, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {key(r): r for r in json.load(open(os.path.join(out, "results.json")))}

    def curve(Lg, case, bal, kind, vsat=True):
        r = R.get(f"{Lg:g}|{case}|{int(bal)}|{kind}|{int(vsat)}")
        if r is None:
            return None, None
        x = np.array(sorted(float(k) for k in r["Id"]))
        return x, np.array([r["Id"][f"{v:.3f}"] for v in x])

    names = list(CASES)
    fig, ax = plt.subplots(3, len(LGS), figsize=(4.6 * len(LGS), 11.5))
    col = {"+phonon": "tab:blue", "+SR": "tab:orange", "+Qit charge": "tab:olive", "+Coulomb": "tab:green",
           "vsat": "tab:red"}
    for j, Lg in enumerate(LGS):
        for i, kind in enumerate(("idvg_lin", "idvg_sat")):
            a = ax[i, j]
            for bal, ls in ((False, "-"), (True, "--")):
                for k in range(1, len(names)):
                    x, d = step_vgs(curve, Lg, names[k - 1], names[k], bal, kind)
                    if d is not None:
                        a.plot(x, d, ls, color=col[names[k]], lw=0.8 if names[k] == "+Qit charge" else 1.5,
                               label=f"{names[k]}" + (" (ballistic on)" if bal else ""))
                vt = vt_cc(*curve(Lg, "+Coulomb", bal, kind), Lg)
                if vt is not None and not bal:
                    a.axvline(vt, color="k", ls=":", lw=0.8)
            a.axhline(0, color="k", lw=0.5)
            a.set_xlabel("Vgs (V)  (dotted: Vt,cc of the full stack)")
            a.set_ylabel("Id change by the term (%)")
            a.set_title(f"Lg {Lg:g} nm, Vds {'0.05' if kind == 'idvg_lin' else '0.7'} V")
            a.set_ylim(-70, 10)
            a.grid(alpha=0.3)
            if j == 0 and i == 0:
                a.legend(fontsize=7, ncol=2)
        a = ax[2, j]
        for bal, ls in ((False, "-"), (True, "--")):
            for case, c in (("bulk", "tab:gray"), ("+Coulomb", "tab:purple")):
                x0, y0 = curve(Lg, case, bal, "idvd", False)
                x1, y1 = curve(Lg, case, bal, "idvd", True)
                if x0 is None or x1 is None:
                    continue
                xs = np.intersect1d(x0, x1)
                d = 100 * (y1[np.isin(x1, xs)] / y0[np.isin(x0, xs)] - 1)
                a.plot(xs, d, ls, color=c, label=f"vsat on {'bulk' if case == 'bulk' else 'full stack'}"
                       + (" (ballistic on)" if bal else ""))
            for k in range(1, len(names)):
                xs, d = step_vgs(curve, Lg, names[k - 1], names[k], bal, "idvd")
                if d is not None:
                    a.plot(xs, d, ls, color=col[names[k]], lw=0.8)
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel("Vds (V)")
        a.set_ylabel("Id change (%)")
        a.set_title(f"Lg {Lg:g} nm, Id-Vd, Vgs {VG_IDVD} V: vsat (thick), terms (thin)")
        a.set_ylim(-70, 10)
        a.grid(alpha=0.3)
        if j == 0:
            a.legend(fontsize=7)
    fig.suptitle("2D planar NMOS: Id change of each cumulative mobility term at a fixed bias; "
                 "ballistic off (solid) vs on (dashed)")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print("wrote", png)


def summary(out):
    """Per Lg, ballistic off/on: each term's Id change at a fixed Vgs =
    Vt,cc (full stack, ballistic off) + OVS, at Vds 0.05 and 0.7, and vsat
    at Vds 1.0, Vgs 1.0."""
    R = {key(r): r for r in json.load(open(os.path.join(out, "results.json")))}

    def curve(Lg, case, bal, kind, vsat=True):
        r = R.get(f"{Lg:g}|{case}|{int(bal)}|{kind}|{int(vsat)}")
        x = np.array(sorted(float(k) for k in r["Id"]))
        return x, np.array([r["Id"][f"{v:.3f}"] for v in x])

    def I(Lg, case, bal, kind, v, vsat=True):
        r = R.get(f"{Lg:g}|{case}|{int(bal)}|{kind}|{int(vsat)}")
        return None if r is None else r["Id"].get(f"{v:.3f}")
    names = list(CASES)
    rows = []
    for Lg in LGS:
        vt_ref = {k: vt_cc(*curve(Lg, "+Coulomb", False, k), Lg) for k in ("idvg_lin", "idvg_sat")}
        for bal in (False, True):
            row = {"Lg": Lg, "bal": bal}
            for kind in ("idvg_lin", "idvg_sat"):
                x, y = curve(Lg, "+Coulomb", bal, kind)
                vt = vt_cc(x, y, Lg)
                row[f"Vt|{kind}"] = vt
                for k in range(1, len(names)):
                    xs, d = step_vgs(curve, Lg, names[k - 1], names[k], bal, kind)
                    for v in OVS:      # at Vgs = Vt(full stack, ballistic off) + v, the nearest grid point
                        j = int(np.argmin(np.abs(xs - (vt_ref[kind] + v))))
                        row[f"{names[k]}|{kind}|{v}"] = float(d[j])
                for v in OVS:
                    row[f"Id_full|{kind}|{v}"] = float(np.exp(np.interp(vt_ref[kind] + v, x, np.log(y))))
            for case in ("bulk", "+Coulomb"):
                a, b = I(Lg, case, bal, "idvd", 1.0, False), I(Lg, case, bal, "idvd", 1.0, True)
                row[f"vsat_{case}"] = None if a is None or b is None else 100 * (b / a - 1)
            rows.append(row)
    json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=1)
    for row in rows:
        print(f"Lg {row['Lg']:>4g} {'bal ON ' if row['bal'] else 'bal off'}: " + "  ".join(
            f"{n}@{v}:{'-' if row[f'{n}|idvg_lin|{v}'] is None else round(row[f'{n}|idvg_lin|{v}'], 1)}/"
            f"{'-' if row[f'{n}|idvg_sat|{v}'] is None else round(row[f'{n}|idvg_sat|{v}'], 1)}"
            for n in names[1:] for v in OVS)
            + f"  vsat bulk {row['vsat_bulk'] and round(row['vsat_bulk'], 1)} full {row['vsat_+Coulomb'] and round(row['vsat_+Coulomb'], 1)}")


if __name__ == "__main__":
    out = os.path.abspath(sys.argv[1])
    if "--plot" in sys.argv:
        plot(out, sys.argv[sys.argv.index("--plot") + 1])
        summary(out)
        sys.exit(0)
    os.makedirs(out, exist_ok=True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    from core.parallel import performance_cores
    t0 = time.perf_counter()
    T = tasks(out)
    res = []
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", performance_cores()))) as ex:
        for r in ex.map(run, sorted(T, key=lambda t: -t[0])):
            res.append(r)
            print(key(r), len(r["Id"]), "missing", r["missing"][:3], f"{r['time_s']:.0f} s", flush=True)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    print(f"total {time.perf_counter() - t0:.0f} s")
