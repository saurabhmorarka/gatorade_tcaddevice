"""python3 -m tcadview [file.pvd|file.vtu] [--screenshot out.png]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcadview", description="Viewer for tcad1d .vtu/.pvd results")
    ap.add_argument("path", nargs="?", help=".pvd sweep or .vtu state")
    ap.add_argument("--field", help="field to show first (e.g. eDensity)")
    ap.add_argument("--state", type=int, default=0, help="state index in a sweep (negative counts from the end)")
    ap.add_argument("--screenshot", metavar="PNG", help="render off screen, save a screenshot and exit")
    args = ap.parse_args(argv)
    from PySide6 import QtWidgets
    from tcadview.app import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("tcadview")
    win = MainWindow(args.path, off_screen=bool(args.screenshot))
    if args.path:
        if args.field:
            win.field_combo.setCurrentText(args.field)
        if args.state:
            win.bias_slider.setValue(args.state % len(win.session))
    if args.screenshot:
        win.plotter.screenshot(args.screenshot)
        win.close()
        return 0
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
