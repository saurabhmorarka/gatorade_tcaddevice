"""Standard MOSFET figures of merit from simulated Id-Vg curves (currents in
A/um of device width, the usual per-width convention for a 2D cross-section):

  SS (mV/dec)   minimum subthreshold swing, min over the curve of
                dVgs/dlog10(Id) - the steepest part of the log curve. (A fit
                over the lowest-Vgs points would instead land on the
                drain-body junction leakage floor.)
  Vt_cc (V)     constant-current threshold: Vgs where Id = I_cc = 1e-7 A *
                W/L (per um width: 1e-7 / L_um A/um), interpolated in
                (Vgs, log10 Id).
  Vt_lin (V)    maximum-transconductance linear extrapolation: the tangent
                to Id(Vgs) at max gm hits Id=0 at Vt_lin + Vds/2 (the
                standard low-Vds definition).
  DIBL (mV/V)   -(Vt_cc(Vds_hi) - Vt_cc(Vds_lo)) / (Vds_hi - Vds_lo)
  Ion, Ioff     Id at the highest swept Vgs / at Vgs=0, both at Vds_hi
"""
import numpy as np


def ss_min(vgs, id_):
    """Returns (SS_mV_per_dec, (vgs_a, vgs_b)) for the steepest adjacent pair."""
    vgs, id_ = np.asarray(vgs), np.abs(np.asarray(id_))
    lg = np.log10(id_)
    dlg = np.diff(lg)
    dv = np.diff(vgs)
    ok = dlg > 0
    if not np.any(ok):
        return np.nan, (np.nan, np.nan)
    ss = np.where(ok, 1e3 * dv / np.where(ok, dlg, 1.0), np.inf)
    k = int(np.argmin(ss))
    return float(ss[k]), (float(vgs[k]), float(vgs[k + 1]))


def vt_constant_current(vgs, id_, i_cc):
    vgs, lg = np.asarray(vgs), np.log10(np.abs(np.asarray(id_)))
    above = np.where(lg >= np.log10(i_cc))[0]
    if len(above) == 0 or above[0] == 0:
        return np.nan
    k = above[0]
    t = (np.log10(i_cc) - lg[k - 1]) / (lg[k] - lg[k - 1])
    return float(vgs[k - 1] + t * (vgs[k] - vgs[k - 1]))


def vt_max_gm(vgs, id_, vds):
    """Returns (Vt_lin, gm_max, vgs_at_gm_max)."""
    vgs, id_ = np.asarray(vgs), np.asarray(id_)
    gm = np.gradient(id_, vgs)
    k = int(np.argmax(gm))
    vt = vgs[k] - id_[k] / gm[k] - 0.5 * vds
    return float(vt), float(gm[k]), float(vgs[k])


def mosfet_metrics(curve_lo, curve_hi, L_um):
    """curve_* = dict(Vds=, Vgs=array, Id=array A/um). Returns a dict."""
    i_cc = 1e-7 / L_um
    out = dict(I_cc=i_cc)
    for tag, c in (("lo", curve_lo), ("hi", curve_hi)):
        ss, win = ss_min(c["Vgs"], c["Id"])
        out[f"SS_{tag}"] = ss
        out[f"SS_{tag}_window"] = win
        out[f"Vt_cc_{tag}"] = vt_constant_current(c["Vgs"], c["Id"], i_cc)
    out["Vt_lin"], out["gm_max_lo"], out["vgs_gm_max_lo"] = vt_max_gm(
        curve_lo["Vgs"], curve_lo["Id"], curve_lo["Vds"])
    out["DIBL_mV_per_V"] = -1e3 * (out["Vt_cc_hi"] - out["Vt_cc_lo"]) / (curve_hi["Vds"] - curve_lo["Vds"])
    vg_hi, id_hi = np.asarray(curve_hi["Vgs"]), np.asarray(curve_hi["Id"])
    out["Ion"] = float(id_hi[np.argmax(vg_hi)])
    out["Ioff"] = float(np.interp(0.0, vg_hi, id_hi))
    out["Ion_Ioff"] = out["Ion"] / out["Ioff"]
    return out
