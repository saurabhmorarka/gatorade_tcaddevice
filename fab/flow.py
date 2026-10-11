"""Run a process flow: the three-layer YAML -> a Structure, with a snapshot
after every step for the viewer's process-step slider.

YAML layers (one file):
  template: planar          # layer 1: a template ...
  params: {Lg: 25nm, ...}   #          ... and its parameters
  process: [...]            # layer 2: explicit flow (optional; overrides
                            #          the template's generated flow)
  domain: {...}             #          domain/voxel (optional override)
  device: {contacts: [...]} # layer 3 (optional override)

Numbers anywhere in a step may be expressions over the parameters
(fab/expr.py). Step keys: step (the operation), label (shown in the
viewer), and the operation's own arguments (fab/structure.py).
"""
import time

import yaml

from fab.expr import evaluate, resolve_params
from fab.structure import Grid, Structure
from fab.templates import TEMPLATES

# step arguments that are numbers/expressions
NUMERIC = {"thickness", "to_height", "depth", "rp", "sigma", "peak", "dose", "lateral_sigma", "top_height", "conc"}


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def expand(cfg):
    """Layer 1 -> {params, domain, process, device} (layer 2/3 given in
    the file override the template's)."""
    if "template" in cfg:
        if cfg["template"] not in TEMPLATES:
            raise ValueError(f"unknown template {cfg['template']!r} (have {sorted(TEMPLATES)})")
        out = TEMPLATES[cfg["template"]](cfg.get("params") or {})
    else:
        out = {"params": dict(cfg.get("params") or {}), "domain": None, "process": None, "device": {}}
    for key in ("domain", "process", "device"):
        if cfg.get(key) is not None:
            out[key] = cfg[key]
    if not out["process"] or not out["domain"]:
        raise ValueError("a flow needs a template or explicit 'domain' and 'process'")
    return out


def _mask(spec, p):
    if spec is None or isinstance(spec, str):
        return spec
    if isinstance(spec, list):
        return [_mask(s, p) for s in spec]
    out = {}
    for k, v in spec.items():
        if k == "rect":
            out[k] = [evaluate(e, p) for e in v]
        elif k == "polygon":
            out[k] = [[evaluate(a, p), evaluate(b, p)] for a, b in v]
        else:
            out[k] = _mask(v, p)
    return out


def step_args(step, p):
    """A step's keyword arguments with expressions evaluated."""
    args = {}
    for k, v in step.items():
        if k in ("step", "label"):
            continue
        if k in NUMERIC:
            args[k] = evaluate(v, p)
        elif k == "mask":
            args[k] = _mask(v, p)
        else:
            args[k] = v
    return args


def make_grid(domain, p):
    hx, hy, hz = [evaluate(v, p) for v in (p[domain["voxel"]] if isinstance(domain["voxel"], str)
                                         else domain["voxel"])]
    x0, x1 = (evaluate(v, p) for v in domain["x"])
    z0, z1 = (evaluate(v, p) for v in domain["z"])
    hb, ht = (evaluate(v, p) for v in domain["height"])
    return Grid(lo=(x0, -ht, z0), hi=(x1, -hb, z1), h=(hx, hy, hz))


def run_step(st, step, p):
    op = step["step"]
    a = step_args(step, p)
    if op == "substrate":
        st.substrate(a["material"], a["name"], a.get("top_height", 0.0))
    elif op == "pattern":
        st.pattern(a["name"], a["mask"])
    elif op == "deposit":
        st.deposit(a["material"], a["name"], a["mode"], thickness=a.get("thickness"), to_height=a.get("to_height"),
                   on=a.get("on"), mask=a.get("mask"))
    elif op == "etch":
        st.etch(a["mode"], materials=a.get("materials"), depth=a.get("depth"), stop_on=a.get("stop_on"),
                mask=a.get("mask"), thickness=a.get("thickness"))
    elif op == "planarize":
        st.planarize(a["to_height"])
    elif op == "implant":
        st.implant(a["species"], a["rp"], a["sigma"], peak=a.get("peak"), dose=a.get("dose"),
                   lateral_sigma=a.get("lateral_sigma", 0.0), blocked_by=a.get("blocked_by"), mask=a.get("mask"))
    elif op == "dope":
        st.dope(a["species"], a["conc"], regions=a.get("regions"), materials=a.get("materials"))
    else:
        raise ValueError(f"unknown process step {op!r}")


def build(cfg, on_step=None, log=print):
    """Run the flow. on_step(i, label, structure) is called after every
    step (the snapshot writer). Returns (structure, [(label, seconds)])."""
    flow = expand(cfg)
    p = resolve_params(flow["params"])
    st = Structure(make_grid(flow["domain"], p))
    st.contacts = flow["device"].get("contacts", [])
    times = []
    log(f"grid {st.grid.shape} = {st.grid.n_nodes / 1e6:.2f}M nodes, voxel {tuple(st.grid.h)} nm")
    for i, step in enumerate(flow["process"]):
        label = step.get("label", step["step"])
        t0 = time.perf_counter()
        run_step(st, step, p)
        dt = time.perf_counter() - t0
        times.append((label, dt))
        log(f"  {i:2d} {label:24s} {dt:6.2f} s")
        if on_step is not None:
            on_step(i, label, st)
    return st, times
