//! Compiled kernels for tcad1d. Each kernel mirrors a numpy implementation
//! step for step (the numpy one stays as the fallback and the reference), so
//! results agree to round-off. Loaded through core/native.py.
//!
//! trace_field_lines: the nonlocal-tunneling field-line tracer of
//! btbt/paths2d.py (FieldLineTracer.trace). The numpy version advances all
//! paths together, one Python-level iteration per RK2 step (~14k steps per
//! BTBT curve at ~1 ms each); here every path is traced to completion in a
//! tight loop. Point location walks the triangle adjacency from the previous
//! triangle instead of matplotlib's trapezoid map.
//!
//! ls_project / ls_refine (levelset.rs): the closest-point steps of the
//! virtual fab's re-distancing (fab/levelset.py redistance).

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod levelset;

const LAM_TOL: f64 = -1e-10; // same inside-test tolerance as TriGeom.locate
const WALK_MAX: usize = 4096;

struct Mesh<'a> {
    tris: &'a [i64],    // (M,3)
    nbr: &'a [i64],     // (M,3), neighbour across the edge opposite vertex k, -1 = boundary
    minv: &'a [f64],    // (M,2,2)
    v2: &'a [f64],      // (M,2)
    semi: &'a [bool],   // (M,)
    vt_ptr: &'a [i64],  // CSR vertex -> triangles (all triangles)
    vt_idx: &'a [i64],
}

impl<'a> Mesh<'a> {
    #[inline]
    fn bary(&self, t: usize, x: f64, y: f64) -> [f64; 3] {
        let m = &self.minv[4 * t..4 * t + 4];
        let dx = x - self.v2[2 * t];
        let dy = y - self.v2[2 * t + 1];
        let l0 = m[0] * dx + m[1] * dy;
        let l1 = m[2] * dx + m[3] * dy;
        [l0, l1, 1.0 - (l0 + l1)]
    }

    #[inline]
    fn interp(&self, f: &[f64], t: usize, x: f64, y: f64) -> f64 {
        let l = self.bary(t, x, y);
        let v = &self.tris[3 * t..3 * t + 3];
        l[0] * f[v[0] as usize] + l[1] * f[v[1] as usize] + l[2] * f[v[2] as usize]
    }

    #[inline]
    fn interp2(&self, f: &[f64], t: usize, x: f64, y: f64) -> (f64, f64) {
        let l = self.bary(t, x, y);
        let v = &self.tris[3 * t..3 * t + 3];
        let (a, b, c) = (2 * v[0] as usize, 2 * v[1] as usize, 2 * v[2] as usize);
        (
            l[0] * f[a] + l[1] * f[b] + l[2] * f[c],
            l[0] * f[a + 1] + l[1] * f[b + 1] + l[2] * f[c + 1],
        )
    }

    /// Triangle containing (x, y), walking from `guess`; -1 outside the mesh.
    fn locate(&self, x: f64, y: f64, guess: usize) -> i64 {
        let mut t = guess;
        let mut prev = usize::MAX;
        for _ in 0..WALK_MAX {
            let l = self.bary(t, x, y);
            if l[0] >= LAM_TOL && l[1] >= LAM_TOL && l[2] >= LAM_TOL {
                return t as i64;
            }
            // cross the edge opposite the most negative coordinate; if that
            // is a boundary edge try the other negative ones (non-convex
            // boundary), never stepping straight back
            let mut order = [0usize, 1, 2];
            order.sort_by(|&a, &b| l[a].partial_cmp(&l[b]).unwrap());
            let mut next = usize::MAX;
            for &k in &order {
                if l[k] >= LAM_TOL {
                    break;
                }
                let n = self.nbr[3 * t + k];
                if n >= 0 && n as usize != prev {
                    next = n as usize;
                    break;
                }
            }
            if next == usize::MAX {
                return self.ring(x, y, t);
            }
            prev = t;
            t = next;
        }
        self.scan(x, y)
    }

    /// The walk hit the mesh boundary from triangle `t`: the point is
    /// outside unless it lies in a triangle around one of t's vertices (a
    /// short step past a concave boundary corner).
    fn ring(&self, x: f64, y: f64, t: usize) -> i64 {
        for &v in &self.tris[3 * t..3 * t + 3] {
            let v = v as usize;
            for &c in &self.vt_idx[self.vt_ptr[v] as usize..self.vt_ptr[v + 1] as usize] {
                let l = self.bary(c as usize, x, y);
                if l[0] >= LAM_TOL && l[1] >= LAM_TOL && l[2] >= LAM_TOL {
                    return c;
                }
            }
        }
        -1
    }

    /// Brute-force fallback for a walk that cycled (not seen in practice).
    fn scan(&self, x: f64, y: f64) -> i64 {
        for t in 0..self.semi.len() {
            let l = self.bary(t, x, y);
            if l[0] >= LAM_TOL && l[1] >= LAM_TOL && l[2] >= LAM_TOL {
                return t as i64;
            }
        }
        -1
    }
}

/// Trace field lines; see btbt/paths2d.py FieldLineTracer.trace for the
/// method. first_tri: the start triangle per path, -1 = no path, -2 =
/// choose it here. Returns (l, end, end_tri, dpsi, s, eg_int).
#[pyfunction]
#[pyo3(signature = (points, triangles, neighbors, vt_ptr, vt_idx, minv, v2, tri_semi, g_node, level, eg_node, starts,
                    first_tri, target, l_max, sign, eg_step, ds_frac, ds_min, ds_max))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn trace_field_lines<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    triangles: PyReadonlyArray2<'py, i64>,
    neighbors: PyReadonlyArray2<'py, i64>,
    vt_ptr: PyReadonlyArray1<'py, i64>,
    vt_idx: PyReadonlyArray1<'py, i64>,
    minv: PyReadonlyArray3<'py, f64>,
    v2: PyReadonlyArray2<'py, f64>,
    tri_semi: PyReadonlyArray1<'py, bool>,
    g_node: PyReadonlyArray2<'py, f64>,
    level: PyReadonlyArray1<'py, f64>,
    eg_node: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,
    first_tri: PyReadonlyArray1<'py, i64>,
    target: PyReadonlyArray1<'py, f64>,
    l_max: PyReadonlyArray1<'py, f64>,
    sign: f64,
    eg_step: f64,
    ds_frac: f64,
    ds_min: f64,
    ds_max: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let pts = points.as_slice()?;
    let tris = triangles.as_slice()?;
    let nbr = neighbors.as_slice()?;
    let vtp = vt_ptr.as_slice()?;
    let vti = vt_idx.as_slice()?;
    let minv_s = minv.as_slice()?;
    let v2_s = v2.as_slice()?;
    let semi = tri_semi.as_slice()?;
    let gn = g_node.as_slice()?;
    let lev = level.as_slice()?;
    let egn = eg_node.as_slice()?;
    let st = starts.as_slice()?;
    let t0 = first_tri.as_slice()?;
    let tg = target.as_slice()?;
    let lm = l_max.as_slice()?;
    let m = semi.len();
    let n = lev.len();
    let k = st.len();
    if tris.len() != 3 * m || nbr.len() != 3 * m || minv_s.len() != 4 * m || v2_s.len() != 2 * m {
        return Err(PyValueError::new_err("triangle arrays disagree in length"));
    }
    if pts.len() != 2 * n || gn.len() != 2 * n || egn.len() != n {
        return Err(PyValueError::new_err("nodal arrays disagree in length"));
    }
    if t0.len() != k || tg.len() != k || lm.len() != k {
        return Err(PyValueError::new_err("per-start arrays disagree in length"));
    }
    if tris.iter().any(|&v| v < 0 || v as usize >= n)
        || st.iter().any(|&v| v < 0 || v as usize >= n)
        || t0.iter().any(|&v| v < -2 || v >= m as i64)
        || nbr.iter().any(|&v| v >= m as i64)
        || vtp.len() != n + 1
        || vtp.windows(2).any(|w| w[0] < 0 || w[1] < w[0])
        || vtp[n] as usize != vti.len()
        || vti.iter().any(|&v| v < 0 || v >= m as i64)
    {
        return Err(PyValueError::new_err("index out of range"));
    }
    let mesh = Mesh { tris, nbr, minv: minv_s, v2: v2_s, semi, vt_ptr: vtp, vt_idx: vti };

    let mut l = vec![f64::INFINITY; k];
    let mut end = vec![f64::NAN; 2 * k];
    let mut end_tri = vec![-1i64; k];
    let mut dpsi = vec![0.0f64; k];
    let mut s = vec![0.0f64; k];
    let mut eg_int = vec![0.0f64; k];

    py.detach(|| {
        let direction = |x: f64, y: f64, t: usize| -> (f64, f64, f64) {
            let (vx, vy) = mesh.interp2(gn, t, x, y);
            let (vx, vy) = (sign * vx, sign * vy);
            let mag = vx.hypot(vy);
            let d = mag.max(1e-30);
            (vx / d, vy / d, mag)
        };
        for i in 0..k {
            if t0[i] == -1 {
                continue;
            }
            let node = st[i] as usize;
            let psi0 = lev[node];
            let (mut px, mut py) = (pts[2 * node], pts[2 * node + 1]);
            let mut t = if t0[i] == -2 {
                // the start node's semiconductor triangle whose centroid lies
                // furthest along the start direction (FieldLineTracer._first_triangle)
                let (gx, gy) = (sign * gn[2 * node], sign * gn[2 * node + 1]);
                let gm = gn[2 * node].hypot(gn[2 * node + 1]).max(1e-30);
                let (dx, dy) = (gx / gm, gy / gm);
                let mut best = -1i64;
                let mut best_score = f64::NEG_INFINITY;
                for &c in &vti[vtp[node] as usize..vtp[node + 1] as usize] {
                    let c = c as usize;
                    if !semi[c] {
                        continue;
                    }
                    let v = &tris[3 * c..3 * c + 3];
                    let (a, b, e) = (v[0] as usize, v[1] as usize, v[2] as usize);
                    let cx = (pts[2 * a] + pts[2 * b] + pts[2 * e]) / 3.0 - px;
                    let cy = (pts[2 * a + 1] + pts[2 * b + 1] + pts[2 * e + 1]) / 3.0 - py;
                    let score = cx * dx + cy * dy;
                    if best < 0 || score > best_score {
                        best = c as i64;
                        best_score = score;
                    }
                }
                if best < 0 {
                    continue;
                }
                best as usize
            } else {
                t0[i] as usize
            };
            let (tgi, lmi) = (tg[i], lm[i]);
            loop {
                let (d1x, d1y, m1) = direction(px, py, t);
                let ds = (ds_frac * eg_step / m1.max(1e-30)).clamp(ds_min, ds_max);
                let (mx, my) = (px + 0.5 * ds * d1x, py + 0.5 * ds * d1y);
                let mut bad = false;
                let tm = match mesh.locate(mx, my, t) {
                    -1 => {
                        bad = true;
                        t
                    }
                    v => v as usize,
                };
                bad |= !semi[tm];
                let (d2x, d2y, _) = direction(mx, my, tm);
                let (nx, ny) = (px + ds * d2x, py + ds * d2y);
                let tn = match mesh.locate(nx, ny, tm) {
                    -1 => {
                        bad = true;
                        tm
                    }
                    v => v as usize,
                };
                bad |= !semi[tn];
                if bad {
                    break;
                }
                let dnew = sign * (mesh.interp(lev, tn, nx, ny) - psi0);
                let dold = dpsi[i];
                let done = dnew >= tgi;
                if !done && dnew < dold - 1e-12 {
                    break; // stall: potential extremum
                }
                let step_len = if done {
                    let frac = (tgi - dold) / (dnew - dold).max(1e-30);
                    l[i] = s[i] + frac * ds;
                    end[2 * i] = px + frac * (nx - px);
                    end[2 * i + 1] = py + frac * (ny - py);
                    end_tri[i] = tn as i64;
                    frac * ds
                } else {
                    ds
                };
                s[i] += step_len;
                eg_int[i] += step_len * mesh.interp(egn, tm, mx, my);
                dpsi[i] = if done { tgi } else { dnew };
                px = nx;
                py = ny;
                t = tn;
                if done || s[i] > lmi {
                    break;
                }
            }
        }
    });
    Ok((
        l.into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec((k, 2), end).unwrap().into_pyarray(py),
        end_tri.into_pyarray(py),
        dpsi.into_pyarray(py),
        s.into_pyarray(py),
        eg_int.into_pyarray(py),
    ))
}

#[pymodule]
fn tcad_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trace_field_lines, m)?)?;
    m.add_function(wrap_pyfunction!(levelset::ls_project, m)?)?;
    m.add_function(wrap_pyfunction!(levelset::ls_refine, m)?)?;
    Ok(())
}
