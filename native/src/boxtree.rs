//! The octree mesher's box-tree refinement (mesh3d/octree.py
//! BoxTree.refine), mirrored rule for rule and in the same queue order, so
//! the leaves are identical to the Python reference:
//!   label   - split until a leaf holds one label (at the label change
//!             closest to the middle, over all axes)
//!   size    - split the axis whose size most exceeds the target spacing
//!   balance - split if more than twice as long as a vertex neighbour
//!   face    - across a label change with a semiconductor on one side,
//!             split to match the neighbour's tangential extent

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeSet;

struct Tree<'a> {
    label: &'a [i32],
    tgt: &'a [f64], // (3, nx, ny, nz)
    semi: &'a [bool], // semi[l] for l >= 0; gas (-1) is never semi
    n: [usize; 3],
    lo: Vec<[usize; 3]>,
    hi: Vec<[usize; 3]>,
    alive: Vec<bool>,
    owner: Vec<i64>,
}

impl<'a> Tree<'a> {
    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.n[1] + j) * self.n[2] + k
    }
    #[inline]
    fn is_semi(&self, l: i32) -> bool {
        l >= 0 && (l as usize) < self.semi.len() && self.semi[l as usize]
    }

    fn add(&mut self, lo: [usize; 3], hi: [usize; 3]) -> usize {
        let id = self.lo.len();
        self.lo.push(lo);
        self.hi.push(hi);
        self.alive.push(true);
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    let x = self.idx(i, j, k);
                    self.owner[x] = id as i64;
                }
            }
        }
        id
    }

    fn split(&mut self, i: usize, axis: usize, pos: usize) -> (usize, usize) {
        let (lo, hi) = (self.lo[i], self.hi[i]);
        self.alive[i] = false;
        let mut m1 = hi;
        let mut m0 = lo;
        m1[axis] = pos;
        m0[axis] = pos;
        let a = self.add(lo, m1);
        let b = self.add(m0, hi);
        (a, b)
    }

    fn shell(&self, i: usize) -> Vec<usize> {
        let lo: Vec<usize> = (0..3).map(|a| self.lo[i][a].saturating_sub(1)).collect();
        let hi: Vec<usize> = (0..3).map(|a| (self.hi[i][a] + 1).min(self.n[a])).collect();
        let mut s = BTreeSet::new();
        for x in lo[0]..hi[0] {
            for y in lo[1]..hi[1] {
                for z in lo[2]..hi[2] {
                    let o = self.owner[self.idx(x, y, z)] as usize;
                    if o != i {
                        s.insert(o);
                    }
                }
            }
        }
        s.into_iter().collect()
    }

    fn rule_label(&self, i: usize) -> Option<(usize, usize)> {
        let (lo, hi) = (self.lo[i], self.hi[i]);
        let first = self.label[self.idx(lo[0], lo[1], lo[2])];
        let mut uniform = true;
        'outer: for x in lo[0]..hi[0] {
            for y in lo[1]..hi[1] {
                for z in lo[2]..hi[2] {
                    if self.label[self.idx(x, y, z)] != first {
                        uniform = false;
                        break 'outer;
                    }
                }
            }
        }
        if uniform {
            return None;
        }
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let mut best: Option<(f64, usize, usize)> = None;
        for a in 0..3 {
            if size[a] < 2 {
                continue;
            }
            let (b, c) = match a {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            // positions k (1..size) where slice k-1 and slice k differ anywhere
            let mut bestk: Option<(f64, usize)> = None;
            for k in 1..size[a] {
                let mut differs = false;
                'sl: for u in lo[b]..hi[b] {
                    for v in lo[c]..hi[c] {
                        let mut p0 = [0usize; 3];
                        p0[a] = lo[a] + k - 1;
                        p0[b] = u;
                        p0[c] = v;
                        let mut p1 = p0;
                        p1[a] += 1;
                        if self.label[self.idx(p0[0], p0[1], p0[2])] != self.label[self.idx(p1[0], p1[1], p1[2])] {
                            differs = true;
                            break 'sl;
                        }
                    }
                }
                if differs {
                    let d = (k as f64 - size[a] as f64 / 2.0).abs();
                    if bestk.is_none_or(|t| d < t.0) {
                        bestk = Some((d, k));
                    }
                }
            }
            if let Some((d, k)) = bestk {
                let score = d / size[a] as f64;
                if best.is_none_or(|t| score < t.0) {
                    best = Some((score, a, lo[a] + k));
                }
            }
        }
        best.map(|t| (t.1, t.2))
    }

    fn rule_size(&self, i: usize) -> Option<(usize, usize)> {
        let (lo, hi) = (self.lo[i], self.hi[i]);
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let nn = self.n[0] * self.n[1] * self.n[2];
        let mut worst: Option<(f64, usize)> = None;
        for a in 0..3 {
            if size[a] < 2 {
                continue;
            }
            let mut t = f64::INFINITY;
            for x in lo[0]..hi[0] {
                for y in lo[1]..hi[1] {
                    for z in lo[2]..hi[2] {
                        t = t.min(self.tgt[a * nn + self.idx(x, y, z)]);
                    }
                }
            }
            let s = size[a] as f64;
            if s > t + 1e-9 && worst.is_none_or(|w| s / t > w.0) {
                worst = Some((s / t, a));
            }
        }
        worst.map(|(_, a)| (a, lo[a] + size[a] / 2))
    }

    fn rule_balance(&self, i: usize, nbrs: &[usize]) -> Option<(usize, usize)> {
        let (lo, hi) = (self.lo[i], self.hi[i]);
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        for &n in nbrs {
            for a in 0..3 {
                let ns = self.hi[n][a] - self.lo[n][a];
                if size[a] > 2 * ns && size[a] >= 2 {
                    return Some((a, lo[a] + size[a] / 2));
                }
            }
        }
        None
    }

    fn has_semi_nbr(&self, i: usize) -> bool {
        let lo: Vec<usize> = (0..3).map(|a| self.lo[i][a].saturating_sub(1)).collect();
        let hi: Vec<usize> = (0..3).map(|a| (self.hi[i][a] + 1).min(self.n[a])).collect();
        for x in lo[0]..hi[0] {
            for y in lo[1]..hi[1] {
                for z in lo[2]..hi[2] {
                    if self.is_semi(self.label[self.idx(x, y, z)]) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn rule_face(&self, i: usize) -> Option<(usize, usize)> {
        let (lo, hi) = (self.lo[i], self.hi[i]);
        let my = self.label[self.idx(lo[0], lo[1], lo[2])];
        let my_semi = self.is_semi(my);
        if !my_semi && !self.has_semi_nbr(i) {
            return None;
        }
        for axis in 0..3 {
            for side in [-1i32, 1] {
                let (mut flo, mut fhi) = (lo, hi);
                if side < 0 {
                    if lo[axis] == 0 {
                        continue;
                    }
                    fhi[axis] = lo[axis];
                    flo[axis] = lo[axis] - 1;
                } else {
                    if hi[axis] == self.n[axis] {
                        continue;
                    }
                    flo[axis] = hi[axis];
                    fhi[axis] = hi[axis] + 1;
                }
                let mut owners = BTreeSet::new();
                for x in flo[0]..fhi[0] {
                    for y in flo[1]..fhi[1] {
                        for z in flo[2]..fhi[2] {
                            let l = self.label[self.idx(x, y, z)];
                            let mut d = l != my;
                            if !my_semi {
                                d &= self.is_semi(l);
                            }
                            if d {
                                owners.insert(self.owner[self.idx(x, y, z)] as usize);
                            }
                        }
                    }
                }
                for n in owners {
                    for b in 0..3 {
                        if b == axis {
                            continue;
                        }
                        if self.lo[n][b] > lo[b] {
                            return Some((b, self.lo[n][b]));
                        }
                        if self.hi[n][b] < hi[b] {
                            return Some((b, self.hi[n][b]));
                        }
                    }
                }
            }
        }
        None
    }

    fn refine(&mut self) -> usize {
        let mut queue: Vec<usize> = (0..self.lo.len()).filter(|&i| self.alive[i]).collect();
        let mut queued = vec![false; self.lo.len()];
        for &q in &queue {
            queued[q] = true;
        }
        let mut n_split = 0;
        while let Some(i) = queue.pop() {
            queued[i] = false;
            if !self.alive[i] {
                continue;
            }
            let mut s = self.rule_label(i).or_else(|| self.rule_size(i));
            let mut nbrs: Option<Vec<usize>> = None;
            if s.is_none() {
                let nb = self.shell(i);
                s = self.rule_balance(i, &nb).or_else(|| self.rule_face(i));
                nbrs = Some(nb);
            }
            let Some((axis, pos)) = s else { continue };
            let nb = nbrs.unwrap_or_else(|| self.shell(i));
            let (a, b) = self.split(i, axis, pos);
            n_split += 1;
            queued.resize(self.lo.len(), false);
            for j in [a, b].into_iter().chain(nb) {
                if self.alive[j] && !queued[j] {
                    queue.push(j);
                    queued[j] = true;
                }
            }
        }
        n_split
    }
}

/// Refine a box tree. label (nx,ny,nz) int32, tgt (3,nx,ny,nz) target
/// sizes in voxels, semi[l] per label, root boxes lo/hi (R,3) in the
/// Python tree's creation order. Returns (lo, hi) of every leaf ever
/// created (creation order), alive flags, owner (voxel -> leaf id) and the
/// number of splits - the same bookkeeping as BoxTree.
#[pyfunction]
pub fn boxtree_refine<'py>(
    py: Python<'py>,
    label: PyReadonlyArray3<'py, i32>,
    tgt: PyReadonlyArray4<'py, f64>,
    semi: PyReadonlyArray1<'py, bool>,
    root_lo: PyReadonlyArray2<'py, i64>,
    root_hi: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
    for (ok, name) in [(label.is_c_contiguous(), "label"), (tgt.is_c_contiguous(), "tgt"),
                       (root_lo.is_c_contiguous(), "root_lo"), (root_hi.is_c_contiguous(), "root_hi")] {
        if !ok {
            return Err(PyValueError::new_err(format!("{name} must be C-contiguous")));
        }
    }
    let sh = label.shape();
    let n = [sh[0], sh[1], sh[2]];
    if tgt.shape() != [3, n[0], n[1], n[2]] {
        return Err(PyValueError::new_err("tgt must be (3, nx, ny, nz)"));
    }
    let rl = root_lo.as_slice()?;
    let rh = root_hi.as_slice()?;
    let mut t = Tree {
        label: label.as_slice()?,
        tgt: tgt.as_slice()?,
        semi: semi.as_slice()?,
        n,
        lo: Vec::new(),
        hi: Vec::new(),
        alive: Vec::new(),
        owner: vec![-1; n[0] * n[1] * n[2]],
    };
    for r in 0..rl.len() / 3 {
        t.add([rl[3 * r] as usize, rl[3 * r + 1] as usize, rl[3 * r + 2] as usize],
              [rh[3 * r] as usize, rh[3 * r + 1] as usize, rh[3 * r + 2] as usize]);
    }
    let n_split = py.detach(|| t.refine());
    let flat = |v: &Vec<[usize; 3]>| v.iter().flat_map(|x| x.iter().map(|&y| y as i64)).collect::<Vec<i64>>();
    let m = t.lo.len();
    let lo = numpy::ndarray::Array2::from_shape_vec((m, 3), flat(&t.lo)).unwrap();
    let hi = numpy::ndarray::Array2::from_shape_vec((m, 3), flat(&t.hi)).unwrap();
    let owner = numpy::ndarray::Array3::from_shape_vec((n[0], n[1], n[2]), t.owner).unwrap();
    let items: Vec<Py<PyAny>> = vec![
        lo.into_pyarray(py).into_any().unbind(),
        hi.into_pyarray(py).into_any().unbind(),
        t.alive.into_pyarray(py).into_any().unbind(),
        owner.into_pyarray(py).into_any().unbind(),
        n_split.into_pyobject(py)?.into_any().unbind(),
    ];
    Ok(pyo3::types::PyTuple::new(py, items)?)
}
