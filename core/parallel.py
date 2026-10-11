"""Process-pool sizing shared by the sweep drivers."""
import os
import subprocess


def performance_cores():
    """Number of PERFORMANCE cores (Apple silicon reports its efficiency
    cores separately; an M3 Pro is 6P+6E). Measured on the M3 Pro: a sweep
    split across all 12 cores ran SLOWER (10.4 s) than 6-8 workers (7.5 s) -
    segments landing on the ~2-3x slower efficiency cores straggle and set
    the wall-clock time. Falls back to os.cpu_count() elsewhere."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                             text=True, timeout=2)
        n = int(out.stdout.strip())
        if n > 0:
            return n
    except Exception:
        pass
    return os.cpu_count() or 6
