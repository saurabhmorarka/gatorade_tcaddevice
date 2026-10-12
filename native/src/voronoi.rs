//! Exact Voronoi cells for the octree mesher's transition nodes
//! (mesh3d/octree.py: voronoi_cell, _clip, _clip_to_box and the per-label
//! split in _dual). Mirrors the Python clipper step for step:
//!   * a convex polyhedron is a list of faces (label + ordered polygon);
//!   * a plane that cuts nothing leaves it unchanged (no duplicate face
//!     when a face already lies in the plane);
//!   * neighbours are clipped nearest first and the loop stops once the
//!     next one is farther than twice the cell's farthest vertex (security
//!     radius); `complete` is false if the candidates ran out first;
//!   * a facet below 1e-8 * |q - p|^2 is round-off (degenerate node set);
//!   * a cell overlapping leaves of several labels is clipped against each
//!     leaf box for the per-label volume and facet areas.
//! Cells are independent and split across threads.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeSet;

const TOL: f64 = 1e-9;
const FACET_REL: f64 = 1e-8;
// face labels below 0 are not neighbours: box walls and leaf-box cuts
const WALL: i64 = -1;
const CUT: i64 = -2;

type V3 = [f64; 3];

#[inline]
fn sub(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
fn norm(a: &V3) -> f64 {
    dot(a, a).sqrt()
}

#[derive(Clone)]
struct Face {
    label: i64,
    v: Vec<V3>,
}

fn area_vec(v: &[V3]) -> V3 {
    let mut s = [0.0; 3];
    let p0 = v[0];
    for k in 0..v.len() {
        let a = sub(&v[k], &p0);
        let b = sub(&v[(k + 1) % v.len()], &p0);
        let c = cross(&a, &b);
        s[0] += c[0];
        s[1] += c[1];
        s[2] += c[2];
    }
    [0.5 * s[0], 0.5 * s[1], 0.5 * s[2]]
}

fn volume(faces: &[Face]) -> f64 {
    faces.iter().map(|f| dot(&f.v[0], &area_vec(&f.v))).sum::<f64>() / 3.0
}

fn box_faces(lo: &V3, hi: &V3) -> Vec<Face> {
    let mut out = Vec::with_capacity(6);
    for a in 0..3 {
        let b = (a + 1) % 3;
        let c = (a + 2) % 3;
        for side in 0..2 {
            let mut pts = Vec::with_capacity(4);
            for (u, w) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                let mut p = [0.0; 3];
                p[a] = if side == 1 { hi[a] } else { lo[a] };
                p[b] = if u == 1 { hi[b] } else { lo[b] };
                p[c] = if w == 1 { hi[c] } else { lo[c] };
                pts.push(p);
            }
            let mut out_n = [0.0; 3];
            out_n[a] = if side == 1 { 1.0 } else { -1.0 };
            if dot(&area_vec(&pts), &out_n) < 0.0 {
                pts.reverse();
            }
            out.push(Face { label: WALL, v: pts });
        }
    }
    out
}

/// Keep n.x <= c; the new face gets `label`.
fn clip(faces: Vec<Face>, n: &V3, c: f64, label: i64) -> Vec<Face> {
    if faces.iter().all(|f| f.v.iter().all(|v| dot(n, v) - c <= TOL)) {
        return faces;
    }
    let mut out = Vec::with_capacity(faces.len() + 1);
    let mut cut: Vec<V3> = Vec::new();
    for f in faces {
        let m = f.v.len();
        let mut nv = Vec::with_capacity(m + 2);
        for k in 0..m {
            let a = f.v[k];
            let b = f.v[(k + 1) % m];
            let da = dot(n, &a) - c;
            let db = dot(n, &b) - c;
            if da <= TOL {
                nv.push(a);
                if da >= -TOL {
                    cut.push(a);
                }
            }
            if (da < -TOL && db > TOL) || (da > TOL && db < -TOL) {
                let t = da / (da - db);
                let q = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
                nv.push(q);
                cut.push(q);
            }
        }
        if nv.len() >= 3 {
            out.push(Face { label: f.label, v: nv });
        }
    }
    if cut.len() >= 3 {
        // unique points (rounded to 1e-9 as np.unique(np.round(., 9)))
        let mut uniq: Vec<V3> = Vec::new();
        let mut keys: BTreeSet<[i64; 3]> = BTreeSet::new();
        for q in &cut {
            let k = [(q[0] * 1e9).round() as i64, (q[1] * 1e9).round() as i64, (q[2] * 1e9).round() as i64];
            if keys.insert(k) {
                uniq.push(*q);
            }
        }
        if uniq.len() >= 3 {
            let m = uniq.len() as f64;
            let mut cen = [0.0; 3];
            for q in &uniq {
                for a in 0..3 {
                    cen[a] += q[a] / m;
                }
            }
            let nn = {
                let l = norm(n);
                [n[0] / l, n[1] / l, n[2] / l]
            };
            let far = uniq
                .iter()
                .max_by(|x, y| norm(&sub(x, &cen)).partial_cmp(&norm(&sub(y, &cen))).unwrap())
                .unwrap();
            let mut u = sub(far, &cen);
            let lu = norm(&u);
            u = [u[0] / lu, u[1] / lu, u[2] / lu];
            let w = cross(&nn, &u);
            let mut ang: Vec<(f64, V3)> =
                uniq.iter().map(|q| {
                    let d = sub(q, &cen);
                    (dot(&d, &w).atan2(dot(&d, &u)), *q)
                }).collect();
            ang.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            out.push(Face { label, v: ang.into_iter().map(|t| t.1).collect() });
        }
    }
    out
}

fn clip_to_box(mut faces: Vec<Face>, lo: &V3, hi: &V3) -> Vec<Face> {
    for a in 0..3 {
        let mut n = [0.0; 3];
        n[a] = 1.0;
        faces = clip(faces, &n, hi[a], CUT);
        let mut m = [0.0; 3];
        m[a] = -1.0;
        faces = clip(faces, &m, -lo[a], CUT);
    }
    faces
}

struct In<'a> {
    p: &'a [f64],
    cand_ptr: &'a [i64],
    cand_idx: &'a [i64],
    rad: &'a [f64],
    dom_hi: V3,
    h: V3,
    owner: &'a [i32],
    shape: [usize; 3],
    leaf_lo: &'a [i64],
    leaf_hi: &'a [i64],
    leaf_lab: &'a [i64],
    leaf_of_owner: &'a [i64], // owner id -> row in leaf_* arrays
}

#[derive(Default)]
struct Out {
    complete: bool,
    vol: Vec<(i64, f64)>,
    fac: Vec<(i64, i64, f64)>, // (neighbour, label, area)
}

fn pt(p: &[f64], i: usize) -> V3 {
    [p[3 * i], p[3 * i + 1], p[3 * i + 2]]
}

fn one_cell(inp: &In, r: usize, node: usize) -> Out {
    let p = pt(inp.p, node);
    let rad = inp.rad[r];
    let lo = [(p[0] - rad).max(0.0), (p[1] - rad).max(0.0), (p[2] - rad).max(0.0)];
    let hi = [(p[0] + rad).min(inp.dom_hi[0]), (p[1] + rad).min(inp.dom_hi[1]), (p[2] + rad).min(inp.dom_hi[2])];
    let mut faces = box_faces(&lo, &hi);
    let (a, b) = (inp.cand_ptr[r] as usize, inp.cand_ptr[r + 1] as usize);
    let mut cand: Vec<(f64, usize)> = inp.cand_idx[a..b]
        .iter()
        .map(|&j| {
            let q = pt(inp.p, j as usize);
            let d = sub(&q, &p);
            (dot(&d, &d), j as usize)
        })
        .filter(|t| t.1 != node)
        .collect();
    cand.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then(x.1.cmp(&y.1)));
    let mut vmax = f64::INFINITY;
    let mut complete = false;
    let nc = cand.len();
    for (k, &(d2, j)) in cand.iter().enumerate() {
        if d2.sqrt() > 2.0 * vmax * (1.0 + 1e-9) {
            complete = true;
            break;
        }
        let q = pt(inp.p, j);
        let n = sub(&q, &p);
        faces = clip(faces, &n, (dot(&q, &q) - dot(&p, &p)) / 2.0, j as i64);
        if k % 4 == 3 || k == nc - 1 {
            vmax = faces
                .iter()
                .flat_map(|f| f.v.iter())
                .map(|v| norm(&sub(v, &p)))
                .fold(0.0, f64::max);
        }
    }
    let vmax_now = faces.iter().flat_map(|f| f.v.iter()).map(|v| norm(&sub(v, &p))).fold(0.0, f64::max);
    let whole = lo.iter().all(|&x| x <= 0.0) && (0..3).all(|a| hi[a] >= inp.dom_hi[a]);
    let mut out = Out { complete: complete || 2.0 * vmax_now <= rad * (1.0 + 1e-9) || whole, ..Default::default() };
    if !out.complete {
        return out;
    }
    // leaves under the cell's bounding box
    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    for f in &faces {
        for v in &f.v {
            for a in 0..3 {
                bmin[a] = bmin[a].min(v[a]);
                bmax[a] = bmax[a].max(v[a]);
            }
        }
    }
    let mut ilo = [0usize; 3];
    let mut ihi = [0usize; 3];
    for a in 0..3 {
        let l = ((bmin[a] / inp.h[a] + 1e-9).floor() as i64).clamp(0, inp.shape[a] as i64 - 1);
        let u = ((bmax[a] / inp.h[a] - 1e-9).ceil() as i64).max(l + 1).clamp(1, inp.shape[a] as i64);
        ilo[a] = l as usize;
        ihi[a] = u as usize;
    }
    let mut leaves: BTreeSet<i64> = BTreeSet::new();
    for i in ilo[0]..ihi[0] {
        for j in ilo[1]..ihi[1] {
            for k in ilo[2]..ihi[2] {
                leaves.insert(inp.owner[(i * inp.shape[1] + j) * inp.shape[2] + k] as i64);
            }
        }
    }
    let rows: Vec<usize> = leaves.iter().map(|&o| inp.leaf_of_owner[o as usize] as usize).collect();
    let labs: BTreeSet<i64> = rows.iter().map(|&r| inp.leaf_lab[r]).collect();
    let keep = |j: i64, area: f64| -> bool {
        if j < 0 {
            return false;
        }
        let d = sub(&pt(inp.p, j as usize), &p);
        area > FACET_REL * dot(&d, &d)
    };
    if labs.len() == 1 {
        let l = *labs.iter().next().unwrap();
        out.vol.push((l, volume(&faces)));
        for f in &faces {
            let ar = norm(&area_vec(&f.v));
            if keep(f.label, ar) {
                out.fac.push((f.label, l, ar));
            }
        }
        return out;
    }
    for &row in &rows {
        let l = inp.leaf_lab[row];
        let llo = [inp.leaf_lo[3 * row] as f64 * inp.h[0], inp.leaf_lo[3 * row + 1] as f64 * inp.h[1],
                   inp.leaf_lo[3 * row + 2] as f64 * inp.h[2]];
        let lhi = [inp.leaf_hi[3 * row] as f64 * inp.h[0], inp.leaf_hi[3 * row + 1] as f64 * inp.h[1],
                   inp.leaf_hi[3 * row + 2] as f64 * inp.h[2]];
        let part = clip_to_box(faces.clone(), &llo, &lhi);
        if part.len() < 4 {
            continue;
        }
        let v = volume(&part);
        if v > 1e-12 {
            out.vol.push((l, v));
        }
        for f in &part {
            let ar = norm(&area_vec(&f.v));
            if keep(f.label, ar) {
                out.fac.push((f.label, l, ar));
            }
        }
    }
    out
}

fn c_order<T, D>(a: &numpy::PyReadonlyArray<'_, T, D>, name: &str) -> PyResult<()>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err(format!("{name} must be C-contiguous")));
    }
    Ok(())
}

/// Voronoi cells of `nodes` among all points `p` (N,3, nm). Candidates per
/// cell in CSR (cand_ptr, cand_idx) found within `rad`; the box tree as
/// owner (voxel -> leaf id), leaf boxes (voxel units) and labels.
/// Returns (complete, vol_ptr, vol_lab, vol_val, fac_ptr, fac_nbr,
/// fac_lab, fac_area), all per input cell in CSR form.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn voronoi_cells<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    nodes: PyReadonlyArray1<'py, i64>,
    cand_ptr: PyReadonlyArray1<'py, i64>,
    cand_idx: PyReadonlyArray1<'py, i64>,
    rad: PyReadonlyArray1<'py, f64>,
    dom_hi: PyReadonlyArray1<'py, f64>,
    h: PyReadonlyArray1<'py, f64>,
    owner: PyReadonlyArray3<'py, i32>,
    leaf_lo: PyReadonlyArray2<'py, i64>,
    leaf_hi: PyReadonlyArray2<'py, i64>,
    leaf_lab: PyReadonlyArray1<'py, i64>,
    leaf_of_owner: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
    c_order(&p, "p")?;
    c_order(&owner, "owner")?;
    c_order(&leaf_lo, "leaf_lo")?;
    c_order(&leaf_hi, "leaf_hi")?;
    let sh = owner.shape();
    let dh = dom_hi.as_slice()?;
    let hh = h.as_slice()?;
    let inp = In {
        p: p.as_slice()?,
        cand_ptr: cand_ptr.as_slice()?,
        cand_idx: cand_idx.as_slice()?,
        rad: rad.as_slice()?,
        dom_hi: [dh[0], dh[1], dh[2]],
        h: [hh[0], hh[1], hh[2]],
        owner: owner.as_slice()?,
        shape: [sh[0], sh[1], sh[2]],
        leaf_lo: leaf_lo.as_slice()?,
        leaf_hi: leaf_hi.as_slice()?,
        leaf_lab: leaf_lab.as_slice()?,
        leaf_of_owner: leaf_of_owner.as_slice()?,
    };
    let nodes = nodes.as_slice()?;
    let m = nodes.len();
    let results: Vec<Out> = py.detach(|| {
        let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 12);
        let chunk = m.div_ceil(nt).max(1);
        let mut parts: Vec<Vec<Out>> = Vec::new();
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..m)
                .step_by(chunk)
                .map(|a| {
                    let b = (a + chunk).min(m);
                    let inp = &inp;
                    s.spawn(move || (a..b).map(|r| one_cell(inp, r, nodes[r] as usize)).collect::<Vec<Out>>())
                })
                .collect();
            for h in hs {
                parts.push(h.join().unwrap());
            }
        });
        parts.into_iter().flatten().collect()
    });
    let mut complete = Vec::with_capacity(m);
    let (mut vp, mut vl, mut vv) = (vec![0i64], Vec::new(), Vec::new());
    let (mut fp, mut fnb, mut fl, mut fa) = (vec![0i64], Vec::new(), Vec::new(), Vec::new());
    for o in results {
        complete.push(o.complete);
        for (l, v) in o.vol {
            vl.push(l);
            vv.push(v);
        }
        vp.push(vl.len() as i64);
        for (j, l, a) in o.fac {
            fnb.push(j);
            fl.push(l);
            fa.push(a);
        }
        fp.push(fnb.len() as i64);
    }
    let items: Vec<Py<PyAny>> = vec![
        complete.into_pyarray(py).into_any().unbind(),
        vp.into_pyarray(py).into_any().unbind(),
        vl.into_pyarray(py).into_any().unbind(),
        vv.into_pyarray(py).into_any().unbind(),
        fp.into_pyarray(py).into_any().unbind(),
        fnb.into_pyarray(py).into_any().unbind(),
        fl.into_pyarray(py).into_any().unbind(),
        fa.into_pyarray(py).into_any().unbind(),
    ];
    Ok(pyo3::types::PyTuple::new(py, items)?)
}

/// All points of `p` (N,3) within `rad[r]` of point `nodes[r]` (the point
/// itself included), as CSR (ptr, idx) - what scipy's
/// cKDTree.query_ball_point gave the Voronoi clipper, without the Python
/// lists (4.4 s of a 7.5 s dual on the full planar device). A uniform
/// bucket grid with cell size `cell`; queries split across threads.
/// voronoi_cells sorts the candidates itself, so their order is free.
#[pyfunction]
pub fn ball_query<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    nodes: PyReadonlyArray1<'py, i64>,
    rad: PyReadonlyArray1<'py, f64>,
    cell: f64,
) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
    c_order(&p, "p")?;
    let pts = p.as_slice()?;
    let nodes = nodes.as_slice()?;
    let rad = rad.as_slice()?;
    if !(cell > 0.0) {
        return Err(PyValueError::new_err("cell must be > 0"));
    }
    let n = pts.len() / 3;
    let m = nodes.len();
    let (ptr, idx) = py.detach(|| {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for i in 0..n {
            for a in 0..3 {
                lo[a] = lo[a].min(pts[3 * i + a]);
                hi[a] = hi[a].max(pts[3 * i + a]);
            }
        }
        let dim: Vec<usize> = (0..3).map(|a| (((hi[a] - lo[a]) / cell).floor() as usize) + 1).collect();
        let key = |q: &[f64]| -> [usize; 3] {
            let mut k = [0usize; 3];
            for a in 0..3 {
                k[a] = (((q[a] - lo[a]) / cell).floor().max(0.0) as usize).min(dim[a] - 1);
            }
            k
        };
        let flat = |k: [usize; 3]| (k[0] * dim[1] + k[1]) * dim[2] + k[2];
        let nb = dim[0] * dim[1] * dim[2];
        let mut start = vec![0usize; nb + 1];
        for i in 0..n {
            start[flat(key(&pts[3 * i..3 * i + 3])) + 1] += 1;
        }
        for b in 0..nb {
            start[b + 1] += start[b];
        }
        let mut fill = start.clone();
        let mut members = vec![0usize; n];
        for i in 0..n {
            let b = flat(key(&pts[3 * i..3 * i + 3]));
            members[fill[b]] = i;
            fill[b] += 1;
        }
        let query = |r: usize| -> Vec<i64> {
            let i = nodes[r] as usize;
            let q = &pts[3 * i..3 * i + 3];
            let rr = rad[r];
            let r2 = rr * rr;
            let a = key(&[q[0] - rr, q[1] - rr, q[2] - rr]);
            let b = key(&[q[0] + rr, q[1] + rr, q[2] + rr]);
            let mut out = Vec::new();
            for x in a[0]..=b[0] {
                for y in a[1]..=b[1] {
                    for z in a[2]..=b[2] {
                        let c = flat([x, y, z]);
                        for &j in &members[start[c]..start[c + 1]] {
                            let d0 = pts[3 * j] - q[0];
                            let d1 = pts[3 * j + 1] - q[1];
                            let d2 = pts[3 * j + 2] - q[2];
                            if d0 * d0 + d1 * d1 + d2 * d2 <= r2 {
                                out.push(j as i64);
                            }
                        }
                    }
                }
            }
            out
        };
        let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 12);
        let chunk = m.div_ceil(nt).max(1);
        let mut parts: Vec<Vec<Vec<i64>>> = Vec::new();
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..m)
                .step_by(chunk)
                .map(|a| {
                    let b = (a + chunk).min(m);
                    let query = &query;
                    s.spawn(move || (a..b).map(query).collect::<Vec<Vec<i64>>>())
                })
                .collect();
            for h in hs {
                parts.push(h.join().unwrap());
            }
        });
        let mut ptr = Vec::with_capacity(m + 1);
        ptr.push(0i64);
        let mut idx = Vec::new();
        for v in parts.into_iter().flatten() {
            idx.extend_from_slice(&v);
            ptr.push(idx.len() as i64);
        }
        (ptr, idx)
    });
    let items: Vec<Py<PyAny>> = vec![
        ptr.into_pyarray(py).into_any().unbind(),
        idx.into_pyarray(py).into_any().unbind(),
    ];
    Ok(pyo3::types::PyTuple::new(py, items)?)
}

// silence unused-import warnings for types used only in signatures
#[allow(dead_code)]
fn _unused(_: &PyArray1<f64>) {}
