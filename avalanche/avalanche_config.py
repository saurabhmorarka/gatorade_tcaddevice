"""Parses the optional `avalanche:` block of an input YAML into the kwargs
newton_solver_avalanche.py / mesh.build_diode_grid's avalanche_ii_refine
need. Kept separate from config.py (which every example, avalanche or not,
already goes through) so that a normal CMOS-flow YAML with no `avalanche:`
block at all is completely unaffected."""
import warnings

from avalanche.avalanche import AvalancheModel
from avalanche.newton_solver_avalanche import DRIVING_FORCES, DEFAULT_REF_DENSITY_CM3


def parse_avalanche_config(cfg: dict) -> dict:
    """Returns {"enabled", "ii_model", "E_crit_V_cm", "cells_per_mfp",
    "driving_force", "ref_density_cm3", "continuation"}.

    enabled: gates mesh.build_diode_grid's optional avalanche_ii_refine
        (h_min capped at the ionization mean free path 1/alpha(E_crit_V_cm)
        / cells_per_mfp) - main_avalanche.py always models the physics.
    driving_force: "hybrid" (default) | "gradqf" | "efield", with
        ref_density_cm3 the hybrid's crossover density - see
        newton_solver_avalanche.py's module docstring.
    continuation: {seed_V, J_stop_A_cm2, V_limit, ds_max} for the arc-length
        trace (newton_solver_avalanche.trace_breakdown): voltage-controlled
        seed steps out to seed_V, then trace until |J| reaches J_stop_A_cm2
        (A/cm^2) or |Va| exceeds V_limit (None = no limit); ds_max caps the
        arc-length step in the (Va/1V, ln|J|) plane.

    The old `fine_tail` block (a hand-tuned fine bias schedule right before
    breakdown) is obsolete - the arc-length trace spaces points along the
    curve itself - and is ignored with a warning if present."""
    av = cfg.get("avalanche") or {}
    if av.get("fine_tail") is not None:
        warnings.warn("avalanche.fine_tail is obsolete and ignored: main_avalanche.py now traces "
                      "breakdown with arc-length continuation (see avalanche.continuation).")
    force = av.get("driving_force", "hybrid")
    if force not in DRIVING_FORCES:
        raise ValueError(f"avalanche.driving_force must be one of {DRIVING_FORCES}, got {force!r}")
    cont = av.get("continuation") or {}
    V_limit = cont.get("V_limit")
    return {
        "enabled": bool(av.get("enabled", False)),
        "ii_model": AvalancheModel.si_von_overstraeten_de_man(),
        "E_crit_V_cm": float(av.get("E_crit_V_cm", 3.0e5)),
        "cells_per_mfp": float(av.get("cells_per_mfp", 8.0)),
        "driving_force": force,
        "ref_density_cm3": float(av.get("ref_density_cm3", DEFAULT_REF_DENSITY_CM3)),
        "continuation": {
            "seed_V": float(cont.get("seed_V", -1.0)),
            "J_stop_A_cm2": float(cont.get("J_stop_A_cm2", 1e3)),
            "V_limit": float(V_limit) if V_limit is not None else None,
            "ds_max": float(cont.get("ds_max", 1.0)),
        },
    }
