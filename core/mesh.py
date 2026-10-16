"""Nonuniform 1D mesh generator, shared by the diode and the MOS capacitor.

Both devices are built the same way: one or two semiconductor "regions"
(diode: p-side + n-side; MOS: just the substrate) meeting a hard interface
(the junction, or the oxide/substrate boundary) at x=0, each region meshed
outward from that interface with `_region_nodes` below. That function picks
the mesh spacing to resolve two effects that can each demand a fine mesh
independently of the other:

  1. Proximity to the interface itself, where the electrostatic potential
     bends sharply (depletion physics) even when the local doping is
     perfectly flat - the classic reason this project's meshes have always
     been finest at x=0 and geometrically coarsen away from it.
  2. The local doping PROFILE changing quickly (relevant once doping.type is
     "linear" or "gaussian" rather than "flat") - resolved via a
     |d(ln N)/dx| based limit, wherever it happens to fall in the region.

For a flat profile, effect (2) is inactive everywhere, and `_region_nodes`
reduces exactly to the original geometric-growth algorithm (bit-for-bit) -
so nothing about the existing flat-doping diode/MOS results changes.
"""
import numpy as np

from core.params import Material, Device
from core.doping_profiles import DopingProfile
from core.interfaces import Interface


def _debye_length(mat: Material, N: float) -> float:
    from core.params import Q
    return (mat.eps * mat.Vt / (Q * N)) ** 0.5


def _interface_concentration(profile: DopingProfile) -> float:
    """Doping right at the region's interface (depth=0) - see
    build_diode_grid's h_min comment for why this, not
    reference_concentration(), is what h_min must be sized from. The
    thickness argument to sample() doesn't matter here: at depth=0 every
    profile type evaluates to its depth=0 value regardless of thickness
    (a linear ramp's frac=0/thickness=0 either way), so a placeholder is
    fine - this runs before the region's actual thickness is resolved."""
    return float(profile.sample(np.array([0.0]), 1.0)[0])


def _one_sided_nodes(length: float, h_min: float, h_max: float, growth: float) -> np.ndarray:
    """Spacings starting at h_min, growing geometrically to h_max, then constant,
    covering total distance `length`. Returns cumulative node positions (0..length),
    NOT including the starting 0 (caller prepends it)."""
    xs = [0.0]
    h = h_min
    x = 0.0
    while x < length:
        x_next = x + h
        remaining_after = length - x_next
        if remaining_after <= 0.5 * h:
            # The final step would either overshoot or leave a tiny sliver
            # smaller than half a step (a huge, ill-conditioned
            # Scharfetter-Gummel flux coefficient q*D/h waiting to happen).
            # Land exactly on `length` instead of adding that sliver as its
            # own segment.
            xs.append(length)
            break
        xs.append(x_next)
        x = x_next
        h = min(h * growth, h_max)
    return np.array(xs)


def _graded_nodes(length: float, profile: DopingProfile, mat: Material,
                   h_min: float, h_max: float, growth: float,
                   gradient_safety_factor: float = 0.25, n_probe: int = 4000) -> np.ndarray:
    """Adaptive marching mesh for a region with a non-flat doping profile.
    Local target spacing is the tighter of:
      - the same geometric ramp _one_sided_nodes uses (h_min at the interface,
        growing to h_max), but with the LOCAL Debye length (from the doping
        value at that point) rather than a single region-wide value, and
      - gradient_safety_factor / |d(ln N)/dx|, so a cell never spans a region
        where the doping itself changes by more than ~25% (default) in log
        space - this is what actually resolves a gaussian implant peak or a
        linear grade, wherever in the region it falls.
    This is a heuristic, not a proof of mesh convergence - as with the MOS
    accumulation-layer mesh study (DEVELOPMENT_LOG.md), always sanity-check
    a new graded-doping case by re-running with a tighter mesh and confirming
    the result doesn't move."""
    xs_probe = np.linspace(0.0, length, n_probe)
    N_probe = np.clip(profile.sample(xs_probe, length), 1e10, None)
    dlogN_probe = np.gradient(np.log(N_probe), xs_probe)

    def target_h(x: float) -> float:
        N_here = np.interp(x, xs_probe, N_probe)
        dlnN_here = np.interp(x, xs_probe, dlogN_probe)
        L_D = _debye_length(mat, N_here)
        ramp = h_min + (h_max - h_min) * min(x / max(10.0 * L_D, 1e-30), 1.0)
        grad_limit = gradient_safety_factor / max(abs(dlnN_here), 1e-30)
        return float(np.clip(min(ramp, grad_limit), h_min, h_max))

    xs = [0.0]
    x = 0.0
    h_prev = None
    while x < length:
        h = target_h(x)
        if h_prev is not None:
            h = min(h, h_prev * growth)  # cap how fast spacing can grow step-to-step
        x_next = x + h
        remaining_after = length - x_next
        if remaining_after <= 0.5 * h:
            xs.append(length)
            break
        xs.append(x_next)
        x = x_next
        h_prev = h
    return np.array(xs)


def _region_nodes(length: float, profile: DopingProfile, mat: Material,
                   h_min: float, h_max: float, growth: float) -> np.ndarray:
    """Mesh nodes (0..length) for one region, dispatching to the exact
    original algorithm for flat doping, or the adaptive one otherwise."""
    if profile.type == "flat":
        return _one_sided_nodes(length, h_min, h_max, growth)
    return _graded_nodes(length, profile, mat, h_min, h_max, growth)


def build_diode_grid(mat: Material, dev: Device, growth: float = 1.06,
                      bulk_spacing_debye_factor: float = 5.0,
                      junction_spacing_debye_factor: float = 0.05,
                      avalanche_ii_refine: dict = None,
                      mat_n: Material = None):
    """Build the nonuniform grid and doping profile for a step (or graded)
    p-n junction diode.

    mat_n: optional SECOND material for the n-side (x>=0), making this a
        real heterojunction diode - `mat` is always the p-side (x<0)
        material. None (default): homojunction, `mat` used for both sides,
        100% backward-compatible (mirrors build_mos_grid's Cdop_gate=None
        pattern exactly - every existing call site passes one positional
        `mat` and gets today's exact behavior, byte-for-byte). When given,
        the mesh's own Debye-length sizing (h_min/h_max_p/h_max_n) already
        computes each side from its OWN material via separate L_D_p/L_D_n
        calls below, so a heterojunction's differing eps/doping is already
        correctly resolved per side with no further change - only the
        returned `mat_field`/`interfaces` are new.

    Returns dict with:
      x        : node positions, cm, x=0 at the junction, p-side negative
      Cdop     : net doping Nd-Na at each node, cm^-3
      junction_index : index of node closest to x=0
      Wp, Wn   : region lengths actually used
      mat_field: core.materials.MaterialField - MaterialField.uniform(mat, x)
                 for a homojunction (mat_n=None), or MaterialField.from_regions
                 (mat, mat_n, ...) for a heterojunction - every solver
                 function accepts either a plain Material (today's exact
                 path) or this MaterialField in place of it.
      interfaces: [] for a homojunction, or a single core.interfaces.Interface
                 at the junction node for a heterojunction (Qit_cm2=0.0,
                 electrically inert by default - same convention as
                 build_mos_grid's oxide/substrate interface).

    avalanche_ii_refine: opt-in extra tightening of h_min for the avalanche
        breakdown solver (newton_solver_avalanche.py), OFF by default (None)
        - every existing call site is bit-for-bit unchanged. The junction's
        Debye-length-tied h_min below resolves electrostatic curvature, but
        the impact-ionization MEAN FREE PATH 1/alpha(E) at breakdown fields
        is a different physical length scale that can be shorter for a
        lightly-doped, high-breakdown-voltage design. Pass a dict
        {"E_crit_V_cm": <a-priori peak-field estimate>, "ii_model": <an
        avalanche.AvalancheModel>, "cells_per_mfp": <optional, default 8.0>}
        to additionally cap h_min at 1/alpha(E_crit)/cells_per_mfp.
        E_crit_V_cm is only used to SIZE the mesh conservatively ahead of
        time - it is not solved for, and the field the solver actually
        converges to may differ.
    """
    p_profile = dev.p_profile if dev.p_profile is not None else DopingProfile.flat(dev.Na)
    n_profile = dev.n_profile if dev.n_profile is not None else DopingProfile.flat(dev.Nd)

    # mat_n_actual: the n-side's own material for every n-side-specific
    # quantity below (Debye length, diffusion length, mesh generation) -
    # defaults to `mat` (mat_n=None), so every quantity below is bit-for-bit
    # identical to before for a homojunction.
    mat_n_actual = mat_n if mat_n is not None else mat

    # h_min resolves the JUNCTION - it must be sized from the doping right
    # AT the interface (depth=0), not reference_concentration()'s
    # region-wide summary. Identical for a flat profile (sample(0) ==
    # concentration_cm3 == reference_concentration()), so every existing
    # flat-doping example is bit-for-bit unchanged; this only matters once
    # a graded profile makes the interface value genuinely different from
    # the region's other summary (e.g. a "linear" profile with
    # transition_um grading a light doping at the junction into a much
    # heavier one further out for low contact resistance - see
    # doping_profiles.py). Using the region-wide summary there would size
    # h_min off doping the mesh's finest cells never actually sit at.
    L_D_p = _debye_length(mat, _interface_concentration(p_profile))
    L_D_n = _debye_length(mat_n_actual, _interface_concentration(n_profile))
    L_D_min = min(L_D_p, L_D_n)

    # h_min (right at the junction) is shared, tied to the SHORTER Debye
    # length: the junction's field curvature on BOTH sides is set by
    # whichever side screens faster, so both sides need that same fine
    # spacing right at x=0. h_max (how coarse each side's BULK, far from
    # the junction, is allowed to get) must NOT be shared the same way -
    # for near-symmetric doping (L_D_p~L_D_n) it doesn't matter which is
    # used, but for strongly asymmetric doping (e.g. Na=1e17/Nd=1e21, a
    # ~100x Debye-length ratio) tying the LARGER side's bulk spacing to the
    # SMALLER side's Debye length wastes enormous numbers of needless
    # points across microns of quasi-neutral region that doesn't need
    # anywhere near that resolution - caught when a strongly asymmetric
    # example produced a 23,000-point mesh (h_max stuck at 0.65nm across a
    # 9.3um p-side) versus ~350 points for comparably-doped sides.
    h_min = junction_spacing_debye_factor * L_D_min
    if avalanche_ii_refine is not None:
        from avalanche.avalanche import ionization_coeffs
        E_probe = np.array([avalanche_ii_refine["E_crit_V_cm"]])
        alpha_n, alpha_p, _, _ = ionization_coeffs(E_probe, avalanche_ii_refine["ii_model"])
        ii_mfp = 1.0 / max(float(alpha_n[0]), float(alpha_p[0]), 1e-30)
        h_min = min(h_min, ii_mfp / avalanche_ii_refine.get("cells_per_mfp", 8.0))
    # h_max (the coarsest a cell is allowed to get, far out in the
    # quasi-neutral bulk) was tied to bulk_spacing_debye_factor*L_D - the
    # DEBYE length, an electrostatic SCREENING scale. That's the right
    # scale to resolve right at the junction (h_min's job), but the wrong
    # one for capping the BULK: the thing that actually needs resolving
    # far from the junction, once a bias is applied, is the injected
    # MINORITY CARRIER's spatial decay, set by its DIFFUSION length
    # (mat.Ln on the p-side, mat.Lp on the n-side - the two are typically
    # ~100x longer than the Debye length, e.g. Ln~1um vs L_D_p~13nm for
    # this project's default doping). A doping-profile gradient (graded/
    # gaussian regions) is already independently protected by
    # _graded_nodes's own gradient-based limiter regardless of h_max, so
    # for a flat profile there's no remaining reason to cap the bulk any
    # tighter than what resolves that diffusion-length-scale decay -
    # tying h_max to the Debye length instead was needlessly tight for
    # every example (not just an asymmetric one), and became actively
    # catastrophic for a heavily/degenerately doped side (Debye length
    # collapses with doping; diffusion length does not, since it depends
    # on mobility/lifetime, not doping). Confirmed empirically: relaxing
    # a doping-asymmetric example's mesh this way cut it from 3054 to 475
    # points with no change in solver behavior at several spot-checked
    # bias points. min_cells_per_diffusion_length=10 keeps a smooth
    # exponential decay resolved with a comfortable margin.
    min_cells_per_diffusion_length = 10.0
    h_max_p = max(bulk_spacing_debye_factor * L_D_p, mat.Ln / min_cells_per_diffusion_length)
    h_max_n = max(bulk_spacing_debye_factor * L_D_n, mat_n_actual.Lp / min_cells_per_diffusion_length)

    Wp = dev.Wp if dev.Wp is not None else max(dev.n_diffusion_lengths * mat.Ln, 20 * L_D_p)
    Wn = dev.Wn if dev.Wn is not None else max(dev.n_diffusion_lengths * mat_n_actual.Lp, 20 * L_D_n)

    x_n_side = _region_nodes(Wn, n_profile, mat_n_actual, h_min, h_max_n, growth)  # 0 .. Wn
    x_p_side = -_region_nodes(Wp, p_profile, mat, h_min, h_max_p, growth)[::-1]   # -Wp .. 0

    x = np.concatenate([x_p_side[:-1], x_n_side])
    x = np.unique(x)  # sorted, dedup the shared 0

    Cdop = np.where(x >= 0.0, n_profile.sample(x, Wn), -p_profile.sample(-x, Wp))
    junction_index = int(np.argmin(np.abs(x)))

    from core.materials import MaterialField
    if mat_n is not None:
        mat_field = MaterialField.from_regions(mat, mat_n, x, junction_index)
        interfaces = [Interface(node_index=junction_index, material_a="p-side", material_b="n-side")]
    else:
        mat_field = MaterialField.uniform(mat, x)
        interfaces = []

    return {
        "x": x,
        "Cdop": Cdop,
        "junction_index": junction_index,
        "Wp": Wp,
        "Wn": Wn,
        "mat_field": mat_field,
        "interfaces": interfaces,
        "h_min": h_min,
        "h_max": max(h_max_p, h_max_n),
        "h_max_p": h_max_p,
        "h_max_n": h_max_n,
        "L_D_p": L_D_p,
        "L_D_n": L_D_n,
        "p_profile": p_profile,
        "n_profile": n_profile,
    }


def build_mos_grid(mat: Material, dev, Cdop_substrate: float,
                    n_ox_points: int = 15, growth: float = 1.08,
                    bulk_spacing_debye_factor: float = 5.0,
                    interface_spacing_debye_factor: float = 0.001,
                    Cdop_gate: float = None):
    """Build the oxide+substrate mesh and per-node doping/eps/ni arrays for
    the 1D MOS capacitor. Two gate models:

      Cdop_gate=None (default): ideal metal gate - a Dirichlet contact
      sitting directly on the oxide, x=0 at the gate/oxide interface,
      oxide occupying x in [-t_ox, 0), substrate x in [0, t_si]. This is
      the original behavior, unchanged.

      Cdop_gate given: a real, depletable polysilicon gate - a THIRD region
      (dev.gate_profile / dev.t_gate) is meshed between the outer metal
      contact and the oxide, x=0 still at the oxide/substrate interface but
      now x in [-t_gate-t_ox, -t_ox) is doped silicon (not an ideal
      conductor), so it can develop its own depletion layer right at the
      oxide interface, same physics as the substrate side. Meshed with the
      same interface-refinement-near-the-oxide / geometric-growth-away
      scheme as the substrate, using the gate's OWN Debye length, mirrored
      to put the fine spacing at the gate/oxide interface instead of at x=0.

    Substrate type (p or n) is generic here: Cdop_substrate's SIGN selects
    it (negative=p, positive=n; magnitude is the reference substrate doping
    used by the closed-form comparisons) - mos_solver.py picks which
    carrier to freeze for the high-frequency C-V trick from that sign.
    Cdop_gate follows the same sign convention for the poly gate.
    `interface_spacing_debye_factor` defaults much tighter than the diode's
    equivalent (0.05): a mesh-convergence check showed the MOS-cap C-V curve
    is genuinely under-resolved at the coarser factor in strong
    accumulation - see DEVELOPMENT_LOG.md.

    Returns dict with:
      x          : node positions, cm, x=0 at the oxide/substrate interface
      Cdop       : net doping, cm^-3 (0 in oxide, signed profile sample in
                   each semiconductor region)
      eps_edge   : permittivity per mesh edge, length len(x)-1
      ni_arr     : intrinsic concentration per node (0 in oxide, mat.ni elsewhere)
      is_oxide   : boolean mask, True for oxide nodes
      oxide_index: index of the oxide/substrate interface node (first substrate node)
      gate_oxide_index: index of the LAST poly-gate node (the node immediately
                   on the gate side of the oxide) - equal to 0 for an ideal
                   metal gate (Cdop_gate=None), since then x[0] itself is
                   the gate/oxide interface.
      t_si       : substrate thickness actually used
      t_gate     : poly gate thickness actually used (0.0 for a metal gate)
      interfaces : list of core.interfaces.Interface - the oxide/substrate
                   boundary (and, for a poly gate, the gate/oxide boundary
                   too), each carrying dev.Qit_cm2/dev.Qit_gate_cm2 (0.0 by
                   default, i.e. electrically inert - today's exact
                   behavior unless a config sets it).
    """
    substrate_profile = (dev.substrate_profile if dev.substrate_profile is not None
                          else DopingProfile.flat(abs(Cdop_substrate)))
    sign_sub = 1.0 if Cdop_substrate >= 0 else -1.0

    L_D = _debye_length(mat, substrate_profile.reference_concentration())
    h_min = interface_spacing_debye_factor * L_D
    h_max = bulk_spacing_debye_factor * L_D

    t_si = dev.t_si if dev.t_si is not None else max(20 * L_D, 2.0e-4)  # >=2 um floor

    x_ox = np.linspace(-dev.t_ox, 0.0, n_ox_points)  # -t_ox .. 0, n_ox_points points, uniform
    x_si = _region_nodes(t_si, substrate_profile, mat, h_min, h_max, growth)  # 0 .. t_si

    if Cdop_gate is None:
        x = np.concatenate([x_ox[:-1], x_si])  # drop duplicate 0 from x_ox
        x = np.unique(x)

        oxide_index = int(np.searchsorted(x, 0.0))
        is_oxide = x < 0.0
        gate_oxide_index = 0
        t_gate = 0.0

        Cdop = np.where(is_oxide, 0.0, sign_sub * substrate_profile.sample(np.clip(x, 0.0, None), t_si))
        ni_arr = np.where(is_oxide, 0.0, mat.ni)
    else:
        gate_profile = (dev.gate_profile if dev.gate_profile is not None
                         else DopingProfile.flat(abs(Cdop_gate)))
        sign_gate = 1.0 if Cdop_gate >= 0 else -1.0

        L_D_gate = _debye_length(mat, gate_profile.reference_concentration())
        h_min_gate = interface_spacing_debye_factor * L_D_gate
        h_max_gate = bulk_spacing_debye_factor * L_D_gate
        t_gate = dev.t_gate if dev.t_gate is not None else max(20 * L_D_gate, 2.0e-4)

        # Same _region_nodes scheme as the substrate (fine near depth=0,
        # coarsening outward), but the gate's depth=0 is at the oxide
        # interface (x=-t_ox), so it's built outward-then-flipped-and-shifted
        # to put the fine end next to the oxide, mirroring how the diode's
        # p-side is built (see build_diode_grid).
        x_gate_depth = _region_nodes(t_gate, gate_profile, mat, h_min_gate, h_max_gate, growth)
        x_gate = -dev.t_ox - x_gate_depth[::-1]  # -(t_gate+t_ox) .. -t_ox

        x = np.concatenate([x_gate[:-1], x_ox[:-1], x_si])
        x = np.unique(x)

        oxide_index = int(np.searchsorted(x, 0.0))
        # Last node STRICTLY inside the poly (is_gate_semi = x < -t_ox below) -
        # the node exactly at x=-t_ox belongs to the oxide side (is_oxide,
        # ni=0, no carriers), so it must not be reported as "the poly
        # surface" or semiconductor_charge/the poly-depletion diagnostic
        # would read a node with no physics on it and see no VG dependence.
        gate_oxide_index = int(np.searchsorted(x, -dev.t_ox, side="left")) - 1
        is_oxide = (x >= -dev.t_ox) & (x < 0.0)
        is_gate_semi = x < -dev.t_ox

        depth_in_gate = np.clip(-dev.t_ox - x, 0.0, None)  # 0 at gate/oxide interface
        Cdop = np.where(
            is_oxide, 0.0,
            np.where(is_gate_semi,
                     sign_gate * gate_profile.sample(depth_in_gate, t_gate),
                     sign_sub * substrate_profile.sample(np.clip(x, 0.0, None), t_si)),
        )
        ni_arr = np.where(is_oxide, 0.0, mat.ni)

    x_mid = (x[:-1] + x[1:]) / 2.0
    eps_edge = np.where((x_mid >= -dev.t_ox) & (x_mid < 0.0), dev.eps_ox, mat.eps)

    interfaces = [Interface(node_index=oxide_index, material_a="SiO2", material_b="Silicon",
                             Qit_cm2=getattr(dev, "Qit_cm2", 0.0))]
    if Cdop_gate is not None:
        interfaces.append(Interface(node_index=gate_oxide_index, material_a="Silicon", material_b="SiO2"))

    return {
        "x": x,
        "Cdop": Cdop,
        "eps_edge": eps_edge,
        "ni_arr": ni_arr,
        "is_oxide": is_oxide,
        "oxide_index": oxide_index,
        "gate_oxide_index": gate_oxide_index,
        "t_si": t_si,
        "t_gate": t_gate,
        "h_min": h_min,
        "h_max": h_max,
        "substrate_profile": substrate_profile,
        "interfaces": interfaces,
    }
