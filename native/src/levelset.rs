//! Closest-point kernels for fab/levelset.py redistance(): Newton
//! projection onto the trilinear zero set of a level set, and the tangent
//! slide + re-projection refinement. They mirror the numpy code step for
//! step (scipy map_coordinates order=1 mode="nearest" sampling of phi and of
//! np.gradient(phi, *h)), so results agree to round-off; here each point
//! runs to completion in a tight loop, the gradient is taken on the fly at
//! the 8 cell corners (no gradient arrays), and points are split across
//! threads.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct Field<'a> {
    phi: &'a [f64],
    n: [usize; 3],
    h: [f64; 3],
}

impl<'a> Field<'a> {
    #[inline]
    fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.phi[(i * self.n[1] + j) * self.n[2] + k]
    }

    /// np.gradient(phi, *h) at node (i, j, k): central inside, one-sided
    /// (first order) on the faces.
    #[inline]
    fn grad_node(&self, idx: [usize; 3]) -> [f64; 3] {
        let mut g = [0.0; 3];
        for a in 0..3 {
            let n = self.n[a];
            let mut lo = idx;
            let mut hi = idx;
            let (den, ok) = if idx[a] == 0 {
                hi[a] = 1;
                (self.h[a], n > 1)
            } else if idx[a] == n - 1 {
                lo[a] = n - 2;
                (self.h[a], true)
            } else {
                lo[a] = idx[a] - 1;
                hi[a] = idx[a] + 1;
                (2.0 * self.h[a], true)
            };
            if ok {
                g[a] = (self.at(hi[0], hi[1], hi[2]) - self.at(lo[0], lo[1], lo[2])) / den;
            }
        }
        g
    }

    /// Lower corner and weights of the cell holding index point p, as
    /// map_coordinates(order=1, mode="nearest") does (clamp, then lerp).
    #[inline]
    fn cell(&self, p: &[f64; 3]) -> ([usize; 3], [f64; 3]) {
        let mut i0 = [0usize; 3];
        let mut t = [0.0; 3];
        for a in 0..3 {
            let n = self.n[a];
            let x = p[a].clamp(0.0, (n - 1) as f64);
            let mut i = x.floor() as usize;
            if i >= n - 1 {
                i = n.saturating_sub(2);
            }
            i0[a] = i;
            t[a] = if n > 1 { x - i as f64 } else { 0.0 };
        }
        (i0, t)
    }

    /// phi and its gradient, trilinearly interpolated at index point p.
    #[inline]
    fn sample(&self, p: &[f64; 3]) -> (f64, [f64; 3]) {
        let (i0, t) = self.cell(p);
        let mut f = 0.0;
        let mut g = [0.0; 3];
        for c in 0..8usize {
            let off = [(c >> 2) & 1, (c >> 1) & 1, c & 1];
            let mut w = 1.0;
            let mut idx = [0usize; 3];
            for a in 0..3 {
                idx[a] = (i0[a] + off[a]).min(self.n[a] - 1);
                w *= if off[a] == 1 { t[a] } else { 1.0 - t[a] };
            }
            if w == 0.0 {
                continue;
            }
            f += w * self.at(idx[0], idx[1], idx[2]);
            let gn = self.grad_node(idx);
            for a in 0..3 {
                g[a] += w * gn[a];
            }
        }
        (f, g)
    }

    fn project(&self, p: &mut [f64; 3], n_iter: usize) {
        for _ in 0..n_iter {
            let (f, g) = self.sample(p);
            let g2 = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).max(1e-30);
            for a in 0..3 {
                p[a] -= f / g2 * g[a] / self.h[a];
            }
        }
    }

    /// The voxel holding p has corners on both sides of the interface.
    fn straddles(&self, p: &[f64; 3]) -> bool {
        let mut i0 = [0usize; 3];
        for a in 0..3 {
            let hi = self.n[a] as i64 - 2;
            i0[a] = (p[a].floor() as i64).clamp(0, hi.max(0)) as usize;
        }
        let (mut neg, mut pos) = (false, false);
        for c in 0..8usize {
            let off = [(c >> 2) & 1, (c >> 1) & 1, c & 1];
            let v = self.at(i0[0] + off[0], i0[1] + off[1], i0[2] + off[2]);
            if v < 0.0 {
                neg = true;
            } else {
                pos = true;
            }
        }
        neg && pos
    }
}

fn dist(xi: &[f64; 3], p: &[f64; 3], h: &[f64; 3]) -> f64 {
    let mut s = 0.0;
    for a in 0..3 {
        let d = (xi[a] - p[a]) * h[a];
        s += d * d;
    }
    s.sqrt()
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 12)
}

/// Run `work(start, end)` over 0..n in contiguous chunks on scoped threads.
fn par_chunks<F: Fn(usize, usize) + Sync>(n: usize, work: F) {
    let nt = threads().min(n.max(1));
    let chunk = n.div_ceil(nt);
    std::thread::scope(|s| {
        for t in 0..nt {
            let (a, b) = (t * chunk, ((t + 1) * chunk).min(n));
            if a < b {
                let w = &work;
                s.spawn(move || w(a, b));
            }
        }
    });
}

fn c_order<T: numpy::Element, D: numpy::ndarray::Dimension>(a: &numpy::PyReadonlyArray<T, D>, name: &str) -> PyResult<()> {
    // as_slice() also accepts Fortran-ordered arrays (in memory order), which
    // would silently scramble the (N,3) rows: insist on C order
    if a.is_c_contiguous() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be C-contiguous (np.ascontiguousarray)")))
    }
}

fn check(phi: &PyReadonlyArray3<f64>, h: &[f64]) -> PyResult<[usize; 3]> {
    c_order(phi, "phi")?;
    let sh = phi.shape();
    if h.len() != 3 || sh.iter().any(|&n| n < 2) {
        return Err(PyValueError::new_err("phi must be 3D with >= 2 nodes per axis; h has 3 entries"));
    }
    Ok([sh[0], sh[1], sh[2]])
}

/// Newton-project index-space points p (N,3) onto phi's trilinear zero set.
#[pyfunction]
pub fn ls_project<'py>(
    py: Python<'py>,
    phi: PyReadonlyArray3<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    n_iter: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let hs = h.as_slice()?;
    let n = check(&phi, hs)?;
    let f = Field { phi: phi.as_slice()?, n, h: [hs[0], hs[1], hs[2]] };
    c_order(&p, "p")?;
    let pin = p.as_slice()?;
    let m = pin.len() / 3;
    let mut out = pin.to_vec();
    {
        let optr = out.as_mut_ptr() as usize;
        py.detach(|| {
            par_chunks(m, |a, b| {
                // disjoint chunks of `out`: each thread writes only rows a..b
                let o = unsafe { std::slice::from_raw_parts_mut((optr as *mut f64).add(3 * a), 3 * (b - a)) };
                for r in 0..(b - a) {
                    let mut q = [o[3 * r], o[3 * r + 1], o[3 * r + 2]];
                    f.project(&mut q, n_iter);
                    o[3 * r..3 * r + 3].copy_from_slice(&q);
                }
            })
        });
    }
    Ok(numpy::ndarray::Array2::from_shape_vec((m, 3), out).unwrap().into_pyarray(py))
}

/// For nodes xi (N,3, index space) whose current closest surface point is
/// p (N,3): distance to p, then `slide` rounds of tangent slide +
/// re-projection, keeping a candidate only if it is closer, on the surface
/// (|phi|/|grad| < 1e-3 h_min) and in a voxel the interface really crosses.
#[pyfunction]
pub fn ls_refine<'py>(
    py: Python<'py>,
    phi: PyReadonlyArray3<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    xi: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    slide: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let hs = h.as_slice()?;
    let n = check(&phi, hs)?;
    let hh = [hs[0], hs[1], hs[2]];
    let f = Field { phi: phi.as_slice()?, n, h: hh };
    c_order(&xi, "xi")?;
    c_order(&p, "p")?;
    let x = xi.as_slice()?;
    let p0 = p.as_slice()?;
    if x.len() != p0.len() {
        return Err(PyValueError::new_err("xi and p must have the same shape"));
    }
    let m = x.len() / 3;
    let tol = 1e-3 * hh.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut best = vec![0.0f64; m];
    {
        let bptr = best.as_mut_ptr() as usize;
        py.detach(|| {
            par_chunks(m, |a, b| {
                let bo = unsafe { std::slice::from_raw_parts_mut((bptr as *mut f64).add(a), b - a) };
                for r in a..b {
                    let xr = [x[3 * r], x[3 * r + 1], x[3 * r + 2]];
                    let mut pr = [p0[3 * r], p0[3 * r + 1], p0[3 * r + 2]];
                    let mut bd = dist(&xr, &pr, &hh);
                    for _ in 0..slide {
                        let (_, g) = f.sample(&pr);
                        let gn = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-30);
                        let nv = [g[0] / gn, g[1] / gn, g[2] / gn];
                        let v = [(xr[0] - pr[0]) * hh[0], (xr[1] - pr[1]) * hh[1], (xr[2] - pr[2]) * hh[2]];
                        let vn = v[0] * nv[0] + v[1] * nv[1] + v[2] * nv[2];
                        let mut q = [0.0; 3];
                        for a3 in 0..3 {
                            q[a3] = pr[a3] + (v[a3] - vn * nv[a3]) / hh[a3];
                        }
                        f.project(&mut q, 3);
                        let (fq, gq) = f.sample(&q);
                        let gqn = (gq[0] * gq[0] + gq[1] * gq[1] + gq[2] * gq[2]).sqrt().max(1e-30);
                        let dq = dist(&xr, &q, &hh);
                        if dq < bd && fq.abs() / gqn < tol && f.straddles(&q) {
                            bd = dq;
                            pr = q;
                        }
                    }
                    bo[r - a] = bd;
                }
            })
        });
    }
    Ok(best.into_pyarray(py))
}
