"""Loads a 2D input YAML (e.g. configs/input_diode_2d.yaml) into a
geometry2d.Domain2D plus mesh/material/output options - the 2D analog of
core/config.py. Reuses core.config._resolve_material_block for the
`material:` block verbatim (same schema, same resolution logic, no
duplication) since material resolution has nothing dimension-specific
about it.
"""
import os

import numpy as np
import yaml

from core.config import _resolve_material_block
from mesh2d.geometry2d import Contact, Domain2D, Region, TopMesa

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "configs", "input_diode_2d.yaml")

_UM_TO_CM = 1.0e-4


def load_config(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _range_cm(pair_um):
    return (float(pair_um[0]) * _UM_TO_CM, float(pair_um[1]) * _UM_TO_CM)


_REGION_MATERIALS = {}


def _region_material(block):
    """A region's optional `material:` block (same shape as the top-level
    one: name / derive_from / alloy with strain, see core.config) resolved
    to a live Material. Regions sharing a material name share one object,
    so they count as the same material."""
    if not block:
        return None
    from core.config import _resolve_material_block
    key = block.get("name") or repr(sorted(block.items()))
    if key not in _REGION_MATERIALS:
        _REGION_MATERIALS[key] = _resolve_material_block(block)
    return _REGION_MATERIALS[key]


def build_domain_from_config(cfg: dict) -> Domain2D:
    geom = cfg["geometry"]
    width_cm = float(geom["width_um"]) * _UM_TO_CM
    height_cm = float(geom["height_um"]) * _UM_TO_CM

    substrate = geom["substrate"]
    regions = [Region(
        name="substrate",
        x_range_cm=(0.0, width_cm),
        y_range_cm=(0.0, height_cm),
        doping_type=substrate["doping_type"],
        concentration_cm3=float(substrate["concentration_cm3"]),
    )]
    for r in geom.get("regions", []):
        regions.append(Region(
            name=r["name"],
            x_range_cm=_range_cm(r["x_range_um"]),
            y_range_cm=_range_cm(r["y_range_um"]),
            doping_type=r.get("doping_type", "n"),
            concentration_cm3=float(r.get("concentration_cm3", 0.0)),
            kind=r.get("kind", "semiconductor"),
            eps_r=float(r["eps_r"]) if r.get("eps_r") is not None else None,
            material=_region_material(r.get("material")),
        ))

    top_mesas = [
        TopMesa(x_range_cm=_range_cm(m["x_range_um"]), height_cm=float(m["height_um"]) * _UM_TO_CM)
        for m in geom.get("mesas", [])
    ]

    contacts = []
    for c in cfg.get("contacts", []):
        contacts.append(Contact(
            name=c["name"],
            surface=c["surface"],
            x_range_cm=_range_cm(c["x_range_um"]),
            bias_role=c["bias_role"],
        ))

    return Domain2D(width_cm=width_cm, height_cm=height_cm, regions=regions, contacts=contacts,
                     top_mesas=top_mesas)


def build_from_config(cfg: dict):
    """Returns (domain, material, mesh_opts, output_cfg, Va_list)."""
    domain = build_domain_from_config(cfg)
    material = _resolve_material_block(cfg.get("material") or {})

    mesh_cfg = cfg.get("mesh") or {}
    mesh_opts = dict(
        h_min_cm=float(mesh_cfg.get("h_min_um", 0.2)) * _UM_TO_CM,
        h_max_cm=float(mesh_cfg.get("h_max_um", 2.0)) * _UM_TO_CM,
        growth=float(mesh_cfg.get("growth", 1.3)),
    )

    vs = cfg.get("voltage_sweep") or {}
    Va_list = np.linspace(
        float(vs.get("va_start_V", -1.0)),
        float(vs.get("va_stop_V", 1.0)),
        int(vs.get("va_points", 21)),
    )

    output_cfg = cfg.get("output") or {}
    return domain, material, mesh_opts, output_cfg, Va_list
