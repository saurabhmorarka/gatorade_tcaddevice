"""Optional compiled kernels (the Rust crate in native/, built with maturin).

    from core.native import native
    if native is not None: native.trace_field_lines(...)

`native` is None when the extension is not installed or TCAD_NATIVE=0 is
set; every caller keeps its numpy implementation as the fallback (and as the
reference the compiled kernel is tested against, testsuite/test_native.py).
Build: see README "Compiled kernels (optional)".
"""
import os

native = None
if os.environ.get("TCAD_NATIVE", "1") != "0":
    try:
        import tcad_native as native   # noqa: F401
    except ImportError:
        native = None
