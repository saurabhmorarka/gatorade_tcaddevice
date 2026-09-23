# gatorade_tcaddevice test suite

A small regression suite so a future change to the numerics can be checked
against known-good numbers instead of guessing whether it broke something.
Covers the project's four example configurations:

| example            | driven by                          | what it exercises                          |
|---------------------|-------------------------------------|---------------------------------------------|
| `diode`             | `input_diode.yaml` + `main.py`      | p-n junction Gummel/Newton drift-diffusion  |
| `mos_metal`         | `input_mos.yaml` + `mos_main.py`    | ideal-metal-gate MOS-cap C-V                |
| `mos_poly_single`   | `input_mos_poly.yaml` + `mos_main.py` | real poly-gate MOS-cap C-V, one doping    |
| `mos_poly_sweep`    | `input_mos_poly.yaml` + `mos_poly_sweep.py` | poly-gate C-V across 6 doping levels |

## How it works

`common.py` calls each example's own config/mesh/solver functions directly
(no plotting, no file I/O) and returns a small dict of scalar summary
metrics: physical constants (Cox, V_FB, V_T, Vbi, I0), plus C/I sampled at a
few representative sweep points. That's deliberately not a full field-by-
field dump - a handful of numbers is enough to catch the kind of bug this
project has actually hit (wrong sign, an order-of-magnitude error, a curve
that collapses to zero) without the suite being so brittle that routine
mesh/plotting changes spuriously fail it.

`golden/*.json` holds today's numbers, captured once and checked in.
`test_examples.py` reruns each example and compares against its golden
file with `rtol=1e-3` (loose enough to absorb cross-platform BLAS/LAPACK
noise, tight enough to catch a real regression).

## Running the suite

```
python3 testsuite/test_examples.py -v
```

or, equivalently:

```
python3 -m unittest discover -s testsuite -v
```

A clean run looks like:

```
test_diode ... ok
test_mos_metal ... ok
test_mos_poly_single ... ok
test_mos_poly_sweep ... ok
```

## If a test fails

1. **Assume it's a real regression first.** Go find what changed and why -
   this suite exists specifically to catch the "did I just silently break
   the physics" class of bug (this project has hit at least two of exactly
   that kind: a Dirichlet-row pivoting bug and a missing quasi-Fermi-level
   split, both caught by hand before this suite existed - it exists so the
   next one doesn't need to be caught by hand).
2. **Only if the change was intentional** (a deliberate physics/numerics
   change, a new default parameter, etc.) and you've independently verified
   the new numbers are correct - by reading the printed sanity checks in
   `mos_main.py`/`main.py`'s own output, comparing against the closed-form
   theory each driver already prints, or otherwise - regenerate the golden
   file for that example only:

   ```
   python3 testsuite/capture_golden.py mos_poly_single
   ```

   Then commit the updated `golden/*.json` in the SAME commit as the code
   change that caused it, so the diff tells the story of what changed and
   why.

Never regenerate golden files just to make a red test green without doing
that verification first - that defeats the entire point of the suite.

## Adding a fifth example

Add a `run_<name>(path=None) -> dict` function to `common.py` (metrics
only - no plotting), register it in the `EXAMPLES` dict at the bottom of
that file, then run `python3 testsuite/capture_golden.py <name>` once to
create its golden file. `test_examples.py` picks it up automatically (one
test method is generated per entry in `EXAMPLES`).
