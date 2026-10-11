"""Virtual fab command line.

  python3 -m fab build configs/fab_planar.yaml [--out out/fab_planar] [--final-only]
  python3 -m fab expand configs/fab_planar.yaml > flow.yaml   # layer 1 -> editable layer 2/3

Then view the build step by step:  python3 -m tcadview out/fab_planar/build.pvd
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fab", description="tcad1d virtual fab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="run a process flow and write VTK snapshots")
    b.add_argument("config")
    b.add_argument("--out", help="output folder (default out/<config name>)")
    b.add_argument("--final-only", action="store_true", help="write only the final structure")
    e = sub.add_parser("expand", help="print the explicit flow (layer 2/3) a template generates")
    e.add_argument("config")
    args = ap.parse_args(argv)

    import yaml
    from fab import flow
    cfg = flow.load(args.config)
    if args.cmd == "expand":
        yaml.safe_dump(flow.expand(cfg), sys.stdout, sort_keys=False, default_flow_style=None, width=110)
        return 0
    from fab.export import SnapshotWriter
    out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.config))[0])
    w = SnapshotWriter(out, every=not args.final_only)
    t0 = time.perf_counter()
    _, times = flow.build(cfg, on_step=w)
    t_build = sum(t for _, t in times)
    path = w.finish()
    print(f"build {t_build:.1f} s (+ snapshots {time.perf_counter() - t0 - t_build:.1f} s) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
