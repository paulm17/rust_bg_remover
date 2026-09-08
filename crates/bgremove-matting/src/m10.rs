//! M10 closed-form alpha matting and PyMatting-compatible foreground recovery.
//!
//! The implementation deliberately keeps the three numerical stages separate:
//! Laplacian assembly, constrained sparse solve, and multilevel foreground
//! estimation.  It uses a deterministic row-oriented sparse representation;
//! no dense image-sized matrix is ever allocated.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::RembgTrimapConfig;
use anyhow::{ensure, Context, Result};
use bgremove_core::{AlphaMask, RgbImageF32, Trimap, TrimapClass};
use std::mem::size_of;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct SparseMatrix {
    nrows: usize,
    ncols: usize,
    rows: Vec<Vec<(usize, f64)>>,
    raw_assembly_entries: usize,
    assembly_peak_estimated_bytes: usize,
}

impl SparseMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Result<Self> {
        ensure!(
            nrows <= 1_000_000 && ncols <= 1_000_000,
            "sparse matrix dimensions exceed safety limit"
        );
        Ok(Self {
            nrows,
            ncols,
            rows: vec![Vec::new(); nrows],
            raw_assembly_entries: 0,
            assembly_peak_estimated_bytes: 0,
        })
    }
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn nnz(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }
    pub fn row(&self, row: usize) -> Result<&[(usize, f64)]> {
        self.rows
            .get(row)
            .map(Vec::as_slice)
            .context("sparse row out of bounds")
    }
    pub fn add(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        ensure!(
            row < self.nrows && col < self.ncols,
            "sparse entry out of bounds"
        );
        ensure!(value.is_finite(), "sparse entry is non-finite");
        if value == 0.0 {
            return Ok(());
        }
        self.rows[row].push((col, value));
        Ok(())
    }
    pub fn consolidate(&mut self) {
        for row in &mut self.rows {
            row.sort_unstable_by_key(|(col, _)| *col);
            let mut out: Vec<(usize, f64)> = Vec::with_capacity(row.len());
            for (col, value) in row.drain(..) {
                if let Some((last, sum)) = out.last_mut() {
                    if *last == col {
                        *sum += value;
                        continue;
                    }
                }
                out.push((col, value));
            }
            out.retain(|(_, value)| value.abs() > 1e-15);
            *row = out;
        }
    }
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        ensure!(x.len() == self.ncols, "sparse matvec dimension mismatch");
        let mut out = vec![0.0; self.nrows];
        for (r, row) in self.rows.iter().enumerate() {
            let mut sum = 0.0;
            for &(c, v) in row {
                sum += v * x[c];
            }
            ensure!(sum.is_finite(), "sparse matvec became non-finite");
            out[r] = sum;
        }
        Ok(out)
    }
    pub fn diagonal(&self, row: usize) -> f64 {
        self.rows[row]
            .iter()
            .find_map(|&(c, v)| (c == row).then_some(v))
            .unwrap_or(0.0)
    }
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.rows
            .iter()
            .enumerate()
            .flat_map(|(r, row)| row.iter().map(move |&(c, v)| (r, c, v)))
    }
    pub fn raw_assembly_entries(&self) -> usize {
        self.raw_assembly_entries
    }
    pub fn assembly_peak_estimated_bytes(&self) -> usize {
        self.assembly_peak_estimated_bytes
    }
}

#[derive(Clone, Debug)]
pub struct ClosedFormConfig {
    pub epsilon: f64,
    pub radius: usize,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_pixels: usize,
    pub max_nnz: usize,
    pub max_assembly_entries: usize,
    pub max_memory_bytes: usize,
    pub base_size: Option<u32>,
}

impl Default for ClosedFormConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-7,
            radius: 1,
            tolerance: 1e-7,
            max_iterations: 10_000,
            max_pixels: 1_000_000,
            max_nnz: 50_000_000,
            max_assembly_entries: 10_000_000,
            max_memory_bytes: 512 * 1024 * 1024,
            base_size: None,
        }
    }
}
impl ClosedFormConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "epsilon must be positive and finite"
        );
        ensure!(self.radius <= 8, "window radius exceeds bounded M10 limit");
        ensure!(
            self.tolerance.is_finite() && self.tolerance > 0.0,
            "tolerance must be positive and finite"
        );
        ensure!(
            self.max_iterations > 0
                && self.max_pixels > 0
                && self.max_nnz > 0
                && self.max_assembly_entries > 0
                && self.max_memory_bytes > 0,
            "solver limits must be nonzero"
        );
        if let Some(base) = self.base_size {
            ensure!(
                (1..=4096).contains(&base),
                "base size is outside safe bounds"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    Converged,
    MaxIterations,
    Breakdown,
}

#[derive(Clone, Debug)]
pub struct SolveReport {
    pub status: SolveStatus,
    pub fallback_used: bool,
    pub iterations: usize,
    pub absolute_residual: f64,
    pub relative_residual: f64,
    pub estimated_memory_bytes: usize,
    pub assembly_peak_estimated_bytes: usize,
    pub solve_peak_estimated_bytes: usize,
    pub raw_assembly_entries: usize,
    pub sparse_nnz: usize,
    pub fallback_reason: Option<String>,
    pub working_dimensions: (u32, u32),
    pub canonical_dimensions: (u32, u32),
    pub laplacian_millis: u64,
    pub solve_millis: u64,
    pub foreground_millis: u64,
    pub limits: SolverLimits,
}

#[derive(Clone, Debug)]
pub struct SolverLimits {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_pixels: usize,
    pub max_nnz: usize,
    pub max_assembly_entries: usize,
    pub max_memory_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct ClosedFormResult {
    pub alpha: AlphaMask,
    pub foreground: RgbImageF32,
    pub background: RgbImageF32,
    pub report: SolveReport,
    pub laplacian: SparseMatrix,
    pub working_image: RgbImageF32,
    pub working_coarse: AlphaMask,
    pub working_trimap: Trimap,
    pub working_alpha: AlphaMask,
    pub working_foreground: RgbImageF32,
    pub working_background: RgbImageF32,
    pub final_rgba: image::RgbaImage,
}

fn report_limits(config: &ClosedFormConfig) -> SolverLimits {
    SolverLimits {
        tolerance: config.tolerance,
        max_iterations: config.max_iterations,
        max_pixels: config.max_pixels,
        max_nnz: config.max_nnz,
        max_assembly_entries: config.max_assembly_entries,
        max_memory_bytes: config.max_memory_bytes,
    }
}

fn pixel_index(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

fn invert_spd_3x3(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    // Cholesky remains well-conditioned for the regularized covariance even
    // when epsilon / area is small enough that a determinant threshold would
    // incorrectly reject a constant window.
    let l00 = matrix[0][0].sqrt();
    ensure!(l00.is_finite() && l00 > 0.0, "singular matting covariance");
    let l10 = matrix[1][0] / l00;
    let l20 = matrix[2][0] / l00;
    let d11 = matrix[1][1] - l10 * l10;
    let l11 = d11.sqrt();
    ensure!(l11.is_finite() && l11 > 0.0, "singular matting covariance");
    let l21 = (matrix[2][1] - l20 * l10) / l11;
    let d22 = matrix[2][2] - l20 * l20 - l21 * l21;
    let l22 = d22.sqrt();
    ensure!(l22.is_finite() && l22 > 0.0, "singular matting covariance");
    let l = [[l00, 0.0, 0.0], [l10, l11, 0.0], [l20, l21, l22]];
    let mut inverse = [[0.0; 3]; 3];
    for column in 0..3 {
        let mut y = [0.0; 3];
        for row in 0..3 {
            let rhs = f64::from((row == column) as u8);
            let prior = (0..row).map(|k| l[row][k] * y[k]).sum::<f64>();
            y[row] = (rhs - prior) / l[row][row];
        }
        for row in (0..3).rev() {
            let prior = ((row + 1)..3)
                .map(|k| l[k][row] * inverse[k][column])
                .sum::<f64>();
            inverse[row][column] = (y[row] - prior) / l[row][row];
        }
    }
    ensure!(
        inverse.iter().flatten().all(|value| value.is_finite()),
        "matting covariance inverse became non-finite"
    );
    Ok(inverse)
}

/// Assemble the PyMatting `cf_laplacian` using the same interior-window and
/// covariance convention. The RGB samples are row-major and normalized.
pub fn build_closed_form_laplacian(
    image: &[[f32; 3]],
    width: u32,
    height: u32,
    epsilon: f64,
    radius: usize,
    is_known: Option<&[bool]>,
    max_pixels: usize,
    max_nnz: usize,
    max_assembly_entries: usize,
    max_memory_bytes: usize,
) -> Result<SparseMatrix> {
    ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
    let w = width as usize;
    let h = height as usize;
    let n = w.checked_mul(h).context("image size overflow")?;
    ensure!(n == image.len(), "image length does not match dimensions");
    ensure!(n <= max_pixels, "image exceeds matting pixel limit");
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "epsilon must be positive"
    );
    if let Some(k) = is_known {
        ensure!(k.len() == n, "known mask length mismatch");
    }
    ensure!(radius <= 8, "radius exceeds bounded matting limit");
    let centers_w = w.saturating_sub(2 * radius);
    let centers_h = h.saturating_sub(2 * radius);
    let side = 2usize
        .checked_mul(radius)
        .and_then(|v| v.checked_add(1))
        .context("window side overflow")?;
    let window_pairs = side
        .checked_mul(side)
        .and_then(|v| v.checked_mul(v))
        .context("window contribution overflow")?;
    let raw_contributions = centers_w
        .checked_mul(centers_h)
        .and_then(|v| v.checked_mul(window_pairs))
        .context("raw assembly size overflow")?;
    ensure!(
        raw_contributions <= max_assembly_entries,
        "raw sparse assembly exceeds configured entry limit"
    );
    let row_bytes = n
        .checked_mul(size_of::<Vec<(usize, f64)>>())
        .context("row storage size overflow")?;
    let pending_bytes = raw_contributions
        .checked_mul(size_of::<(usize, f64)>())
        .and_then(|v| v.checked_mul(2))
        .context("pending assembly size overflow")?;
    let center_bytes = side
        .checked_mul(side)
        .and_then(|v| v.checked_mul(size_of::<[f64; 3]>()))
        .context("window storage size overflow")?;
    let peak_bytes = row_bytes
        .checked_add(pending_bytes)
        .and_then(|v| v.checked_add(center_bytes))
        .and_then(|v| v.checked_add(n.checked_mul(size_of::<f64>())?))
        .context("assembly peak size overflow")?;
    ensure!(
        peak_bytes <= max_memory_bytes,
        "raw sparse assembly exceeds configured memory limit"
    );
    let mut l = SparseMatrix::new(n, n)?;
    l.raw_assembly_entries = raw_contributions;
    l.assembly_peak_estimated_bytes = peak_bytes;
    let size = side;
    let area = (size * size) as f64;
    if w > 2 * radius && h > 2 * radius {
        for cy in radius..(h - radius) {
            for cx in radius..(w - radius) {
                if let Some(known) = is_known {
                    if (cy - radius..=cy + radius)
                        .all(|y| (cx - radius..=cx + radius).all(|x| known[pixel_index(x, y, w)]))
                    {
                        continue;
                    }
                }
                let mut centered = vec![[0.0_f64; 3]; size * size];
                let mut mean = [0.0; 3];
                for dy in 0..size {
                    for dx in 0..size {
                        let p = image[pixel_index(cx + dx - radius, cy + dy - radius, w)];
                        ensure!(
                            p.iter().all(|v| v.is_finite()),
                            "image contains non-finite sample"
                        );
                        for c in 0..3 {
                            mean[c] += f64::from(p[c]);
                        }
                    }
                }
                for c in &mut mean {
                    *c /= area;
                }
                for dy in 0..size {
                    for dx in 0..size {
                        let p = image[pixel_index(cx + dx - radius, cy + dy - radius, w)];
                        let q = &mut centered[dy * size + dx];
                        for c in 0..3 {
                            q[c] = f64::from(p[c]) - mean[c];
                        }
                    }
                }
                let mut cov = [[0.0; 3]; 3];
                for q in &centered {
                    for a in 0..3 {
                        for b in 0..3 {
                            cov[a][b] += q[a] * q[b];
                        }
                    }
                }
                for a in 0..3 {
                    for b in 0..3 {
                        cov[a][b] /= area;
                    }
                }
                for c in 0..3 {
                    cov[c][c] += epsilon / area;
                }
                let inv = invert_spd_3x3(cov)?;
                for ai in 0..size {
                    for aj in 0..size {
                        let i = pixel_index(cx + aj - radius, cy + ai - radius, w); // source's x/y pair order is equivalent after flattening
                        let q = centered[ai * size + aj];
                        for bi in 0..size {
                            for bj in 0..size {
                                let j = pixel_index(cx + bj - radius, cy + bi - radius, w);
                                let r = centered[bi * size + bj];
                                let mut temp = 0.0;
                                for a in 0..3 {
                                    for b in 0..3 {
                                        temp += q[a] * inv[a][b] * r[b];
                                    }
                                }
                                let value = (if i == j { 1.0 } else { 0.0 }) - (1.0 + temp) / area;
                                l.add(i, j, value)?;
                            }
                        }
                    }
                }
            }
        }
    }
    l.consolidate();
    ensure!(
        l.nnz() <= max_nnz,
        "matting Laplacian exceeds sparse nonzero limit"
    );
    Ok(l)
}

fn cg_solve(
    a: &SparseMatrix,
    b: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Vec<f64>, SolveStatus, usize, f64, f64)> {
    ensure!(
        a.nrows == a.ncols && a.nrows == b.len(),
        "solver dimensions mismatch"
    );
    let mut x = vec![0.0; b.len()];
    let mut r = b.to_vec();
    let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let target = tolerance.max(tolerance * norm_b);
    let mut z = vec![0.0; b.len()];
    for i in 0..b.len() {
        let d = a.diagonal(i);
        z[i] = if d > 1e-15 { r[i] / d } else { r[i] };
    }
    let mut p = z.clone();
    let mut rz = r.iter().zip(&z).map(|(x, y)| x * y).sum::<f64>();
    let mut norm_r = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm_r <= target {
        return Ok((
            x,
            SolveStatus::Converged,
            0,
            norm_r,
            if norm_b > 0.0 { norm_r / norm_b } else { 0.0 },
        ));
    }
    for iteration in 0..max_iterations {
        let ap = a.matvec(&p)?;
        let denom = p.iter().zip(&ap).map(|(x, y)| x * y).sum::<f64>();
        if !denom.is_finite() || denom <= 1e-20 {
            return Ok((
                x,
                SolveStatus::Breakdown,
                iteration,
                norm_r,
                norm_r / (norm_b.max(1e-30)),
            ));
        }
        let step = rz / denom;
        for i in 0..x.len() {
            x[i] += step * p[i];
            r[i] -= step * ap[i];
        }
        norm_r = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm_r.is_finite() {
            return Ok((
                vec![0.0; b.len()],
                SolveStatus::Breakdown,
                iteration + 1,
                f64::INFINITY,
                f64::INFINITY,
            ));
        }
        if norm_r <= target {
            return Ok((
                x,
                SolveStatus::Converged,
                iteration + 1,
                norm_r,
                norm_r / (norm_b.max(1e-30)),
            ));
        }
        for i in 0..b.len() {
            let d = a.diagonal(i);
            z[i] = if d > 1e-15 { r[i] / d } else { r[i] };
        }
        let rz_new = r.iter().zip(&z).map(|(x, y)| x * y).sum::<f64>();
        if !rz_new.is_finite() {
            break;
        }
        let beta = rz_new / rz;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    Ok((
        x,
        SolveStatus::MaxIterations,
        max_iterations,
        norm_r,
        norm_r / (norm_b.max(1e-30)),
    ))
}

pub fn solve_constrained_alpha_with_coarse(
    laplacian: &SparseMatrix,
    trimap: &[f32],
    coarse: &[f32],
    width: u32,
    height: u32,
    config: &ClosedFormConfig,
) -> Result<(Vec<f32>, SolveReport)> {
    config.validate()?;
    let n = (width as usize)
        .checked_mul(height as usize)
        .context("grid size overflow")?;
    ensure!(
        n <= config.max_pixels,
        "constrained solve exceeds pixel limit"
    );
    ensure!(
        laplacian.nnz() <= config.max_nnz,
        "constrained solve exceeds sparse nonzero limit"
    );
    ensure!(
        laplacian.nrows == n && laplacian.ncols == n && trimap.len() == n && coarse.len() == n,
        "constrained system dimensions mismatch"
    );
    ensure!(
        trimap
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "trimap must be finite and normalized"
    );
    ensure!(
        coarse
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "coarse alpha must be finite and normalized"
    );
    let mut out = coarse.to_vec();
    let known_bg = trimap.iter().filter(|v| **v == 0.0).count();
    let known_fg = trimap.iter().filter(|v| **v == 1.0).count();
    let unknown_count = trimap.iter().filter(|v| **v > 0.0 && **v < 1.0).count();
    let solve_peak = n
        .checked_mul(size_of::<usize>())
        .and_then(|v| v.checked_add(n.checked_mul(size_of::<Vec<(usize, f64)>>())?))
        .and_then(|v| v.checked_add(unknown_count.checked_mul(size_of::<usize>())?))
        .and_then(|v| v.checked_add(unknown_count.checked_mul(size_of::<Vec<(usize, f64)>>())?))
        .and_then(|v| {
            v.checked_add(
                laplacian
                    .nnz()
                    .checked_mul(size_of::<(usize, f64)>())?
                    .checked_mul(2)?,
            )
        })
        .and_then(|v| v.checked_add(unknown_count.checked_mul(size_of::<f64>() * 6)?))
        .and_then(|v| v.checked_add(n.checked_mul(size_of::<[f32; 3]>() * 2)?))
        .context("solver memory estimate overflow")?;
    ensure!(
        solve_peak <= config.max_memory_bytes,
        "constrained solve exceeds configured memory limit"
    );
    let estimated_memory = laplacian.assembly_peak_estimated_bytes().max(solve_peak);
    ensure!(
        estimated_memory <= config.max_memory_bytes,
        "overall matting peak exceeds configured memory limit"
    );
    let unknown: Vec<usize> = trimap
        .iter()
        .enumerate()
        .filter_map(|(i, v)| ((*v > 0.0) && (*v < 1.0)).then_some(i))
        .collect();
    if unknown.is_empty() {
        out = trimap.to_vec();
        return Ok((
            out,
            SolveReport {
                status: SolveStatus::Converged,
                fallback_used: false,
                iterations: 0,
                absolute_residual: 0.0,
                relative_residual: 0.0,
                estimated_memory_bytes: estimated_memory,
                assembly_peak_estimated_bytes: laplacian.assembly_peak_estimated_bytes(),
                solve_peak_estimated_bytes: solve_peak,
                raw_assembly_entries: laplacian.raw_assembly_entries(),
                sparse_nnz: laplacian.nnz(),
                fallback_reason: None,
                working_dimensions: (width, height),
                canonical_dimensions: (width, height),
                laplacian_millis: 0,
                solve_millis: 0,
                foreground_millis: 0,
                limits: report_limits(config),
            },
        ));
    }
    if known_bg == 0 || known_fg == 0 || laplacian.nnz() == 0 {
        return Ok((
            out,
            SolveReport {
                status: SolveStatus::Breakdown,
                fallback_used: true,
                iterations: 0,
                absolute_residual: 0.0,
                relative_residual: 0.0,
                estimated_memory_bytes: estimated_memory,
                assembly_peak_estimated_bytes: laplacian.assembly_peak_estimated_bytes(),
                solve_peak_estimated_bytes: solve_peak,
                raw_assembly_entries: laplacian.raw_assembly_entries(),
                sparse_nnz: laplacian.nnz(),
                fallback_reason: Some(
                    if laplacian.nnz() == 0 || (known_bg == 0 && known_fg == 0) {
                        "underconstrained or singular Laplacian; returned coarse alpha".to_owned()
                    } else {
                        "trimap has only one known class; returned coarse alpha".to_owned()
                    },
                ),
                working_dimensions: (width, height),
                canonical_dimensions: (width, height),
                laplacian_millis: 0,
                solve_millis: 0,
                foreground_millis: 0,
                limits: report_limits(config),
            },
        ));
    }
    let mut map = vec![usize::MAX; n];
    for (i, &p) in unknown.iter().enumerate() {
        map[p] = i;
    }
    let mut a = SparseMatrix::new(unknown.len(), unknown.len())?;
    let mut b = vec![0.0; unknown.len()];
    for (ri, &global_i) in unknown.iter().enumerate() {
        for &(global_j, v) in laplacian.row(global_i)? {
            if map[global_j] != usize::MAX {
                a.add(ri, map[global_j], v)?;
            } else {
                b[ri] -= v * trimap[global_j] as f64;
            }
        }
    }
    a.consolidate();
    let (x, status, it, abs, rel) = cg_solve(&a, &b, config.tolerance, config.max_iterations)?;
    let failed = !matches!(status, SolveStatus::Converged) || x.iter().any(|v| !v.is_finite());
    if !failed {
        for (i, &p) in unknown.iter().enumerate() {
            out[p] = (x[i] as f32).clamp(0.0, 1.0);
        }
        for (i, value) in trimap.iter().enumerate() {
            if *value == 0.0 || *value == 1.0 {
                out[i] = *value;
            }
        }
    } else {
        out = coarse.to_vec();
    }
    let report = SolveReport {
        status,
        fallback_used: failed,
        iterations: it,
        absolute_residual: abs,
        relative_residual: rel,
        estimated_memory_bytes: estimated_memory,
        assembly_peak_estimated_bytes: laplacian.assembly_peak_estimated_bytes(),
        solve_peak_estimated_bytes: solve_peak,
        raw_assembly_entries: laplacian.raw_assembly_entries(),
        sparse_nnz: a.nnz(),
        fallback_reason: failed.then(|| {
            match status {
                SolveStatus::MaxIterations => "CG reached max_iterations; returned coarse alpha",
                SolveStatus::Breakdown => "CG breakdown; returned coarse alpha",
                _ => "solver failed; returned coarse alpha",
            }
            .to_string()
        }),
        working_dimensions: (width, height),
        canonical_dimensions: (width, height),
        laplacian_millis: 0,
        solve_millis: 0,
        foreground_millis: 0,
        limits: report_limits(config),
    };
    Ok((out, report))
}

fn resize_nearest_rgb(
    src: &[[f32; 3]],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Vec<[f32; 3]> {
    (0..dh)
        .flat_map(|y| {
            (0..dw).map(move |x| src[(y * sh / dh).min(sh - 1) * sw + (x * sw / dw).min(sw - 1)])
        })
        .collect()
}
fn resize_nearest_alpha(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    (0..dh)
        .flat_map(|y| {
            (0..dw).map(move |x| src[(y * sh / dh).min(sh - 1) * sw + (x * sw / dw).min(sw - 1)])
        })
        .collect()
}

fn resize_u8_pillow_lanczos(
    src: &[u8],
    sw: usize,
    sh: usize,
    channels: usize,
    dw: usize,
    dh: usize,
) -> Result<Vec<u8>> {
    ensure!(
        sw > 0 && sh > 0 && dw > 0 && dh > 0,
        "resize dimensions must be nonzero"
    );
    ensure!(
        channels == 1 || channels == 3 || channels == 4,
        "resize channels must be one, three, or four"
    );
    let source_len = sw
        .checked_mul(sh)
        .and_then(|v| v.checked_mul(channels))
        .context("resize source size overflow")?;
    ensure!(src.len() == source_len, "resize source length mismatch");
    const PRECISION: i64 = 1 << 22;
    fn sinc(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
        }
    }
    fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
        let scale = input as f64 / output as f64;
        let filterscale = scale.max(1.0);
        let support = 3.0 * filterscale;
        (0..output)
            .map(|xx| {
                let center = (xx as f64 + 0.5) * scale;
                let xmin = ((center - support + 0.5) as i32).max(0);
                let xmax = ((center + support + 0.5) as i32).min(input as i32);
                let count = (xmax - xmin).max(0) as usize;
                let mut weights = Vec::with_capacity(count);
                let mut sum = 0.0;
                for x in 0..count {
                    let u = (x as f64 + xmin as f64 - center + 0.5) / filterscale;
                    let weight = if (-3.0..3.0).contains(&u) {
                        sinc(u) * sinc(u / 3.0)
                    } else {
                        0.0
                    };
                    weights.push(weight);
                    sum += weight;
                }
                let fixed = weights
                    .into_iter()
                    .map(|weight| {
                        let value = if sum != 0.0 {
                            weight / sum * PRECISION as f64
                        } else {
                            0.0
                        };
                        if value < 0.0 {
                            (value - 0.5) as i64
                        } else {
                            (value + 0.5) as i64
                        }
                    })
                    .collect();
                (xmin as usize, fixed)
            })
            .collect()
    }
    fn clip(sum: i64) -> u8 {
        ((sum >> 22).clamp(0, 255)) as u8
    }
    let hcoeff = coefficients(sw, dw);
    let vcoeff = coefficients(sh, dh);
    let mut horizontal = vec![0u8; dw * sh * channels];
    for y in 0..sh {
        for (x, (start, weights)) in hcoeff.iter().enumerate() {
            for c in 0..channels {
                let mut sum = PRECISION / 2;
                for (k, weight) in weights.iter().enumerate() {
                    sum += src[(y * sw + start + k) * channels + c] as i64 * weight;
                }
                horizontal[(y * dw + x) * channels + c] = clip(sum);
            }
        }
    }
    let mut out = vec![0u8; dw * dh * channels];
    for (y, (start, weights)) in vcoeff.iter().enumerate() {
        for x in 0..dw {
            for c in 0..channels {
                let mut sum = PRECISION / 2;
                for (k, weight) in weights.iter().enumerate() {
                    sum += horizontal[((start + k) * dw + x) * channels + c] as i64 * weight;
                }
                out[(y * dw + x) * channels + c] = clip(sum);
            }
        }
    }
    Ok(out)
}

pub fn resize_lanczos_rgb(
    src: &[[f32; 3]],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Result<Vec<[f32; 3]>> {
    ensure!(
        sw > 0 && sh > 0 && dw > 0 && dh > 0,
        "resize dimensions must be nonzero"
    );
    ensure!(
        src.len() == sw.checked_mul(sh).context("resize dimensions overflow")?,
        "resize source length mismatch"
    );
    let bytes = src
        .iter()
        .flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
        .collect::<Vec<_>>();
    Ok(resize_u8_pillow_lanczos(&bytes, sw, sh, 3, dw, dh)?
        .chunks_exact(3)
        .map(|p| {
            [
                f32::from(p[0]) / 255.0,
                f32::from(p[1]) / 255.0,
                f32::from(p[2]) / 255.0,
            ]
        })
        .collect())
}

pub fn resize_lanczos_alpha(
    src: &[f32],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Result<Vec<f32>> {
    ensure!(
        sw > 0 && sh > 0 && dw > 0 && dh > 0,
        "resize dimensions must be nonzero"
    );
    ensure!(
        src.len() == sw.checked_mul(sh).context("resize dimensions overflow")?,
        "resize source length mismatch"
    );
    let bytes = src
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<_>>();
    Ok(resize_u8_pillow_lanczos(&bytes, sw, sh, 1, dw, dh)?
        .into_iter()
        .map(|v| f32::from(v) / 255.0)
        .collect())
}

/// Resize a caller-supplied trimap without inventing new classes. Nearest
/// neighbour is intentional: it is class-preserving and keeps the generic
/// RGB+coarse+trimap API independent of any background-remover profile.
fn resize_trimap_class_preserving(
    trimap: &Trimap,
    target_width: u32,
    target_height: u32,
) -> Result<Trimap> {
    ensure!(
        target_width > 0 && target_height > 0,
        "trimap dimensions must be nonzero"
    );
    let sw = trimap.width() as usize;
    let sh = trimap.height() as usize;
    let dw = target_width as usize;
    let dh = target_height as usize;
    let classes = (0..dh)
        .flat_map(|y| {
            (0..dw).map(move |x| {
                trimap.data()[(y * sh / dh).min(sh - 1) * sw + (x * sw / dw).min(sw - 1)]
            })
        })
        .collect();
    Trimap::new(target_width, target_height, classes)
}

fn coarse_mask_bytes(coarse: &[f32]) -> Vec<u8> {
    coarse
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

fn build_final_rgba(
    foreground: &RgbImageF32,
    alpha: &AlphaMask,
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Result<image::RgbaImage> {
    ensure!(
        foreground.dimensions() == alpha.dimensions(),
        "final cutout dimensions differ"
    );
    ensure!(
        foreground.width() as usize == width && foreground.height() as usize == height,
        "final cutout source dimensions differ"
    );
    let pixels = width
        .checked_mul(height)
        .context("final cutout size overflow")?;
    let mut rgb_premultiplied = vec![0u8; pixels * 3];
    let mut alpha_bytes = vec![0u8; pixels];
    for (index, (rgb, a)) in foreground.data().iter().zip(alpha.data()).enumerate() {
        let alpha_u8 = if a.is_finite() {
            (a.clamp(0.0, 1.0) * 255.0) as u8
        } else {
            0
        };
        alpha_bytes[index] = alpha_u8;
        for (channel, value) in rgb.iter().enumerate() {
            let value_u8 = if value.is_finite() {
                (value.clamp(0.0, 1.0) * 255.0) as u8
            } else {
                0
            };
            rgb_premultiplied[index * 3 + channel] =
                ((u16::from(value_u8) * u16::from(alpha_u8) + 127) / 255) as u8;
        }
    }
    let alpha_resized =
        resize_u8_pillow_lanczos(&alpha_bytes, width, height, 1, target_width, target_height)?;
    let mut resized = vec![0u8; target_width * target_height * 4];
    for channel in 0..3 {
        let source = (0..pixels)
            .map(|i| rgb_premultiplied[i * 3 + channel])
            .collect::<Vec<_>>();
        let plane =
            resize_u8_pillow_lanczos(&source, width, height, 1, target_width, target_height)?;
        for i in 0..target_width * target_height {
            let a = alpha_resized[i];
            resized[i * 4 + channel] = if a == 0 {
                0
            } else {
                ((u32::from(plane[i]) * 255) / u32::from(a)).min(255) as u8
            };
        }
    }
    for (i, a) in alpha_resized.into_iter().enumerate() {
        resized[i * 4 + 3] = a;
    }
    image::RgbaImage::from_raw(target_width as u32, target_height as u32, resized)
        .context("final cutout buffer dimensions")
}

fn python_round(value: f64) -> usize {
    let lower = value.floor();
    let fraction = value - lower;
    if fraction < 0.5 || (fraction == 0.5 && (lower as u64).is_multiple_of(2)) {
        lower as usize
    } else {
        lower as usize + 1
    }
}

/// Port of PyMatting's `_estimate_fb_ml`; unlike color division this performs
/// the same multilevel coupled foreground/background updates and clamps each
/// iterate to encoded RGB bounds.
pub fn estimate_foreground_ml(
    image: &[[f32; 3]],
    alpha: &[f32],
    width: u32,
    height: u32,
    regularization: f32,
    n_small_iterations: usize,
    n_big_iterations: usize,
    small_size: u32,
    gradient_weight: f32,
) -> Result<(RgbImageF32, RgbImageF32)> {
    ensure!(
        width > 0 && height > 0,
        "foreground dimensions must be nonzero"
    );
    let w0 = width as usize;
    let h0 = height as usize;
    let expected = w0
        .checked_mul(h0)
        .context("foreground dimensions overflow")?;
    ensure!(
        image.len() == expected && alpha.len() == expected,
        "foreground dimensions mismatch"
    );
    ensure!(
        image
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (*v >= 0.0 && *v <= 1.0))
            && alpha
                .iter()
                .all(|v| v.is_finite() && (*v >= 0.0 && *v <= 1.0)),
        "foreground inputs must be finite normalized values"
    );
    ensure!(
        regularization.is_finite()
            && regularization >= 0.0
            && gradient_weight.is_finite()
            && gradient_weight >= 0.0,
        "invalid foreground regularization"
    );
    let mut fm = [0.0f32; 3];
    let mut bm = [0.0f32; 3];
    let mut fc = 0usize;
    let mut bc = 0usize;
    for i in 0..image.len() {
        if alpha[i] > 0.9 {
            for c in 0..3 {
                fm[c] += image[i][c];
            }
            fc += 1;
        }
        if alpha[i] < 0.1 {
            for c in 0..3 {
                bm[c] += image[i][c];
            }
            bc += 1;
        }
    }
    for c in 0..3 {
        fm[c] /= (fc as f32) + 1e-5;
        bm[c] /= (bc as f32) + 1e-5;
    }
    let mut fp = vec![fm];
    let mut bp = vec![bm];
    let mut prev_w = 1usize;
    let mut prev_h = 1usize;
    let levels = (f64::from(w0.max(h0) as u32).log2().ceil() as usize).max(0);
    for level in 0..=levels {
        let f = if levels == 0 {
            0.0
        } else {
            level as f64 / levels as f64
        };
        let w = python_round((w0 as f64).powf(f)).max(1);
        let h = python_round((h0 as f64).powf(f)).max(1);
        let img = resize_nearest_rgb(image, w0, h0, w, h);
        let a = resize_nearest_alpha(alpha, w0, h0, w, h);
        let mut fg = resize_nearest_rgb(&fp, prev_w, prev_h, w, h);
        let mut bg = resize_nearest_rgb(&bp, prev_w, prev_h, w, h);
        let iterations = if w as u32 <= small_size && h as u32 <= small_size {
            n_small_iterations
        } else {
            n_big_iterations
        };
        for _ in 0..iterations {
            for y in 0..h {
                for x in 0..w {
                    let av = a[y * w + x];
                    let bv = 1.0 - av;
                    let mut a00 = av * av;
                    let a01 = av * bv;
                    let mut a11 = bv * bv;
                    let mut rhsf = [
                        av * img[y * w + x][0],
                        av * img[y * w + x][1],
                        av * img[y * w + x][2],
                    ];
                    let mut rhsb = [
                        bv * img[y * w + x][0],
                        bv * img[y * w + x][1],
                        bv * img[y * w + x][2],
                    ];
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let xx = (x as isize + dx).clamp(0, (w - 1) as isize) as usize;
                        let yy = (y as isize + dy).clamp(0, (h - 1) as isize) as usize;
                        let da = regularization + gradient_weight * (av - a[yy * w + xx]).abs();
                        a00 += da;
                        a11 += da;
                        for c in 0..3 {
                            rhsf[c] += da * fg[yy * w + xx][c];
                            rhsb[c] += da * bg[yy * w + xx][c];
                        }
                    }
                    let det = (a00 * a11 - a01 * a01).max(1e-12);
                    let q00 = a11 / det;
                    let q01 = -a01 / det;
                    let q11 = a00 / det;
                    for c in 0..3 {
                        fg[y * w + x][c] = (q00 * rhsf[c] + q01 * rhsb[c]).clamp(0.0, 1.0);
                        bg[y * w + x][c] = (q01 * rhsf[c] + q11 * rhsb[c]).clamp(0.0, 1.0);
                    }
                }
            }
        }
        fp = fg;
        bp = bg;
        prev_w = w;
        prev_h = h;
    }
    Ok((
        RgbImageF32::new(width, height, fp)?,
        RgbImageF32::new(width, height, bp)?,
    ))
}

fn refine_closed_form_unbounded(
    image: &RgbImageF32,
    coarse: &AlphaMask,
    trimap: &Trimap,
    config: &ClosedFormConfig,
) -> Result<ClosedFormResult> {
    config.validate()?;
    ensure!(
        image.dimensions() == trimap.dimensions(),
        "image and trimap dimensions differ"
    );
    ensure!(
        coarse.dimensions() == image.dimensions(),
        "coarse alpha dimensions do not match image"
    );
    ensure!(
        coarse
            .data()
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "coarse alpha must be finite and normalized"
    );
    let started = Instant::now();
    let known: Vec<bool> = trimap
        .data()
        .iter()
        .map(|c| !matches!(c, TrimapClass::Unknown))
        .collect();
    let l = build_closed_form_laplacian(
        image.data(),
        image.width(),
        image.height(),
        config.epsilon,
        config.radius,
        Some(&known),
        config.max_pixels,
        config.max_nnz,
        config.max_assembly_entries,
        config.max_memory_bytes,
    )?;
    let lap_ms = started.elapsed().as_millis() as u64;
    let t: Vec<f32> = trimap
        .data()
        .iter()
        .map(|c| match c {
            TrimapClass::Background => 0.0,
            TrimapClass::Unknown => 0.5,
            TrimapClass::Foreground => 1.0,
        })
        .collect();
    let solve_start = Instant::now();
    let (av, mut report) = solve_constrained_alpha_with_coarse(
        &l,
        &t,
        coarse.data(),
        image.width(),
        image.height(),
        config,
    )?;
    report.laplacian_millis = lap_ms;
    ensure!(
        report.estimated_memory_bytes <= config.max_memory_bytes,
        "reported peak memory exceeds configured cap"
    );
    report.solve_millis = solve_start.elapsed().as_millis() as u64;
    report.working_dimensions = image.dimensions();
    report.canonical_dimensions = image.dimensions();
    let fg_start = Instant::now();
    let (fg, bg) = estimate_foreground_ml(
        image.data(),
        &av,
        image.width(),
        image.height(),
        1e-5,
        10,
        2,
        32,
        1.0,
    )?;
    report.foreground_millis = fg_start.elapsed().as_millis() as u64;
    let alpha = AlphaMask::new(image.width(), image.height(), av)?;
    let final_rgba = build_final_rgba(
        &fg,
        &alpha,
        image.width() as usize,
        image.height() as usize,
        image.width() as usize,
        image.height() as usize,
    )?;
    Ok(ClosedFormResult {
        alpha: alpha.clone(),
        foreground: fg.clone(),
        background: bg.clone(),
        report,
        laplacian: l,
        working_image: image.clone(),
        working_coarse: coarse.clone(),
        working_trimap: trimap.clone(),
        working_alpha: alpha,
        working_foreground: fg,
        working_background: bg,
        final_rgba,
    })
}

fn refine_bounded_with_working_trimap(
    image: &RgbImageF32,
    coarse: &AlphaMask,
    working_trimap: Trimap,
    config: &ClosedFormConfig,
) -> Result<ClosedFormResult> {
    let base = config
        .base_size
        .context("bounded matting profile requires base_size")?;
    let largest = image.width().max(image.height());
    if largest <= base {
        return refine_closed_form_unbounded(image, coarse, &working_trimap, config);
    }
    let scale = f64::from(base) / f64::from(largest);
    let work_w = python_round(f64::from(image.width()) * scale).max(1) as u32;
    let work_h = python_round(f64::from(image.height()) * scale).max(1) as u32;
    let work_rgb = resize_lanczos_rgb(
        image.data(),
        image.width() as usize,
        image.height() as usize,
        work_w as usize,
        work_h as usize,
    )?;
    let work_image = RgbImageF32::new(work_w, work_h, work_rgb)?;
    let work_coarse = resize_lanczos_alpha(
        coarse.data(),
        image.width() as usize,
        image.height() as usize,
        work_w as usize,
        work_h as usize,
    )?;
    let mut work_config = config.clone();
    work_config.base_size = None;
    let mut result = refine_closed_form_unbounded(
        &work_image,
        &AlphaMask::new(work_w, work_h, work_coarse)?,
        &working_trimap,
        &work_config,
    )?;
    let alpha = resize_lanczos_alpha(
        result.alpha.data(),
        work_w as usize,
        work_h as usize,
        image.width() as usize,
        image.height() as usize,
    )?;
    let foreground = resize_lanczos_rgb(
        result.foreground.data(),
        work_w as usize,
        work_h as usize,
        image.width() as usize,
        image.height() as usize,
    )?;
    let background = resize_lanczos_rgb(
        result.background.data(),
        work_w as usize,
        work_h as usize,
        image.width() as usize,
        image.height() as usize,
    )?;
    result.alpha = AlphaMask::new(image.width(), image.height(), alpha)?;
    result.foreground = RgbImageF32::new(image.width(), image.height(), foreground)?;
    result.background = RgbImageF32::new(image.width(), image.height(), background)?;
    result.report.working_dimensions = (work_w, work_h);
    result.report.canonical_dimensions = image.dimensions();
    result.final_rgba = build_final_rgba(
        &result.working_foreground,
        &result.working_alpha,
        work_w as usize,
        work_h as usize,
        image.width() as usize,
        image.height() as usize,
    )?;
    Ok(result)
}

/// Generic closed-form refiner. The caller's trimap is always honored. When
/// bounded, its classes are resized with nearest-neighbour class preservation;
/// no background-remover thresholds or erosion assumptions are applied.
pub fn refine_closed_form_with_coarse(
    image: &RgbImageF32,
    coarse: &AlphaMask,
    trimap: &Trimap,
    config: &ClosedFormConfig,
) -> Result<ClosedFormResult> {
    config.validate()?;
    ensure!(
        image.dimensions() == trimap.dimensions(),
        "image and trimap dimensions differ"
    );
    ensure!(
        coarse.dimensions() == image.dimensions(),
        "coarse alpha dimensions do not match image"
    );
    ensure!(
        coarse
            .data()
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "coarse alpha must be finite and normalized"
    );
    if let Some(base) = config.base_size {
        if image.width().max(image.height()) <= base {
            return refine_closed_form_unbounded(image, coarse, trimap, config);
        }
        let scale = f64::from(base) / f64::from(image.width().max(image.height()));
        let work_w = python_round(f64::from(image.width()) * scale).max(1) as u32;
        let work_h = python_round(f64::from(image.height()) * scale).max(1) as u32;
        let work_trimap = resize_trimap_class_preserving(trimap, work_w, work_h)?;
        refine_bounded_with_working_trimap(image, coarse, work_trimap, config)
    } else {
        refine_closed_form_unbounded(image, coarse, trimap, config)
    }
}

/// Explicit backgroundremover profile. It resizes the canonical coarse mask,
/// derives the working trimap using the supplied rembg thresholds/erosion
/// configuration, then invokes the generic solver and foreground stages.
pub fn refine_backgroundremover_bounded(
    image: &RgbImageF32,
    coarse: &AlphaMask,
    config: &ClosedFormConfig,
    trimap_config: RembgTrimapConfig,
) -> Result<ClosedFormResult> {
    config.validate()?;
    ensure!(
        coarse.dimensions() == image.dimensions(),
        "coarse alpha dimensions do not match image"
    );
    ensure!(
        coarse
            .data()
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "coarse alpha must be finite and normalized"
    );
    let base = config
        .base_size
        .unwrap_or(image.width().max(image.height()));
    let scale = f64::from(base.min(image.width().max(image.height())))
        / f64::from(image.width().max(image.height()));
    let work_w = python_round(f64::from(image.width()) * scale).max(1) as u32;
    let work_h = python_round(f64::from(image.height()) * scale).max(1) as u32;
    let work_coarse = resize_lanczos_alpha(
        coarse.data(),
        image.width() as usize,
        image.height() as usize,
        work_w as usize,
        work_h as usize,
    )?;
    let working_trimap = crate::rembg_symmetric_trimap(
        &coarse_mask_bytes(&work_coarse),
        work_w,
        work_h,
        trimap_config,
    )?;
    if image.width().max(image.height()) <= base {
        return refine_closed_form_unbounded(image, coarse, &working_trimap, config);
    }
    refine_bounded_with_working_trimap(image, coarse, working_trimap, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_rows_are_deterministic() {
        let mut a = SparseMatrix::new(2, 2).unwrap();
        a.add(0, 1, 2.0).unwrap();
        a.add(0, 1, 3.0).unwrap();
        a.add(0, 0, 1.0).unwrap();
        a.consolidate();
        assert_eq!(a.row(0).unwrap(), &[(0, 1.0), (1, 5.0)]);
    }
    #[test]
    fn known_pixels_are_preserved() {
        let image = RgbImageF32::constant(3, 3, [0.2, 0.4, 0.7]).unwrap();
        let trimap = Trimap::new(
            3,
            3,
            vec![
                TrimapClass::Background,
                TrimapClass::Unknown,
                TrimapClass::Foreground,
                TrimapClass::Unknown,
                TrimapClass::Unknown,
                TrimapClass::Unknown,
                TrimapClass::Background,
                TrimapClass::Unknown,
                TrimapClass::Foreground,
            ],
        )
        .unwrap();
        let coarse = AlphaMask::new(
            3,
            3,
            trimap
                .data()
                .iter()
                .map(|c| match c {
                    TrimapClass::Background => 0.0,
                    TrimapClass::Unknown => 0.5,
                    TrimapClass::Foreground => 1.0,
                })
                .collect(),
        )
        .unwrap();
        let r = refine_closed_form_with_coarse(
            &image,
            &coarse,
            &trimap,
            &ClosedFormConfig {
                max_iterations: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.alpha.data()[0], 0.0);
        assert_eq!(r.alpha.data()[2], 1.0);
        assert!(r.alpha.data().iter().all(|v| v.is_finite()));
    }
    #[test]
    fn invalid_nonfinite_config_rejected() {
        let c = ClosedFormConfig {
            epsilon: f64::NAN,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }
    #[test]
    fn foreground_is_bounded_and_composite_finite() {
        let image = vec![[0.8, 0.2, 0.1]; 9];
        let alpha = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0];
        let (f, b) = estimate_foreground_ml(&image, &alpha, 3, 3, 1e-5, 2, 1, 32, 1.0).unwrap();
        assert!(f
            .data()
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (*v >= 0.0 && *v <= 1.0)));
        assert!(b
            .data()
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (*v >= 0.0 && *v <= 1.0)));
    }
    #[test]
    fn bounded_working_resolution_restores_canonical_dimensions() {
        let image = RgbImageF32::constant(16, 8, [0.3, 0.5, 0.7]).unwrap();
        let trimap = Trimap::new(
            16,
            8,
            (0..128)
                .map(|i| {
                    if i % 16 < 3 {
                        TrimapClass::Background
                    } else if i % 16 > 12 {
                        TrimapClass::Foreground
                    } else {
                        TrimapClass::Unknown
                    }
                })
                .collect(),
        )
        .unwrap();
        let coarse = (0..128)
            .map(|i| {
                if i % 16 == 0 {
                    0.2
                } else if i % 16 == 15 {
                    0.8
                } else {
                    0.37
                }
            })
            .collect::<Vec<_>>();
        let result = refine_closed_form_with_coarse(
            &image,
            &AlphaMask::new(16, 8, coarse).unwrap(),
            &trimap,
            &ClosedFormConfig {
                base_size: Some(8),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.report.working_dimensions, (8, 4));
        assert_eq!(result.report.canonical_dimensions, (16, 8));
        assert_eq!(result.alpha.dimensions(), (16, 8));
    }

    #[test]
    fn generic_bounded_path_honors_supplied_trimap_classes() {
        let image = RgbImageF32::constant(16, 8, [0.3, 0.5, 0.7]).unwrap();
        let coarse = AlphaMask::new(16, 8, vec![0.5; 128]).unwrap();
        let left_background = Trimap::new(
            16,
            8,
            (0..128)
                .map(|i| {
                    if i % 16 < 2 {
                        TrimapClass::Background
                    } else if i % 16 >= 14 {
                        TrimapClass::Foreground
                    } else {
                        TrimapClass::Unknown
                    }
                })
                .collect(),
        )
        .unwrap();
        let right_background = Trimap::new(
            16,
            8,
            (0..128)
                .map(|i| {
                    if i % 16 < 2 {
                        TrimapClass::Foreground
                    } else if i % 16 >= 14 {
                        TrimapClass::Background
                    } else {
                        TrimapClass::Unknown
                    }
                })
                .collect(),
        )
        .unwrap();
        let config = ClosedFormConfig {
            base_size: Some(8),
            max_iterations: 1,
            ..Default::default()
        };
        let first =
            refine_closed_form_with_coarse(&image, &coarse, &left_background, &config).unwrap();
        let second =
            refine_closed_form_with_coarse(&image, &coarse, &right_background, &config).unwrap();
        assert_eq!(first.report.working_dimensions, (8, 4));
        assert_eq!(second.report.working_dimensions, (8, 4));
        assert_eq!(first.working_trimap.data()[0], TrimapClass::Background);
        assert_eq!(first.working_trimap.data()[7], TrimapClass::Foreground);
        assert_eq!(second.working_trimap.data()[0], TrimapClass::Foreground);
        assert_eq!(second.working_trimap.data()[7], TrimapClass::Background);
    }

    #[test]
    fn backgroundremover_profile_uses_non_default_trimap_config() {
        let image = RgbImageF32::constant(16, 8, [0.3, 0.5, 0.7]).unwrap();
        let coarse = AlphaMask::new(
            16,
            8,
            (0..128)
                .map(|i| if i % 16 < 8 { 0.2 } else { 0.8 })
                .collect(),
        )
        .unwrap();
        let config = ClosedFormConfig {
            base_size: Some(8),
            max_iterations: 1,
            ..Default::default()
        };
        let default = refine_backgroundremover_bounded(
            &image,
            &coarse,
            &config,
            RembgTrimapConfig {
                foreground_threshold: 240,
                background_threshold: 10,
                erode_size: 3,
            },
        )
        .unwrap();
        let custom = refine_backgroundremover_bounded(
            &image,
            &coarse,
            &config,
            RembgTrimapConfig {
                foreground_threshold: 200,
                background_threshold: 100,
                erode_size: 0,
            },
        )
        .unwrap();
        assert!(default
            .working_trimap
            .data()
            .iter()
            .all(|c| *c == TrimapClass::Unknown));
        assert!(custom
            .working_trimap
            .data()
            .contains(&TrimapClass::Background));
        assert!(custom
            .working_trimap
            .data()
            .contains(&TrimapClass::Foreground));
    }
    #[test]
    fn forced_iteration_limit_falls_back_to_finite_coarse_alpha() {
        let image = RgbImageF32::constant(8, 8, [0.2, 0.4, 0.6]).unwrap();
        let trimap = Trimap::new(
            8,
            8,
            (0..64)
                .map(|i| {
                    if i % 8 == 0 {
                        TrimapClass::Background
                    } else if i % 8 == 7 {
                        TrimapClass::Foreground
                    } else {
                        TrimapClass::Unknown
                    }
                })
                .collect(),
        )
        .unwrap();
        let coarse = vec![0.37_f32; 64];
        let result = refine_closed_form_with_coarse(
            &image,
            &AlphaMask::new(8, 8, coarse.clone()).unwrap(),
            &trimap,
            &ClosedFormConfig {
                max_iterations: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result
            .alpha
            .data()
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!(result.report.fallback_used);
        assert!(matches!(
            result.report.status,
            SolveStatus::MaxIterations | SolveStatus::Breakdown
        ));
        let explicit = refine_closed_form_with_coarse(
            &image,
            &AlphaMask::new(8, 8, coarse.clone()).unwrap(),
            &trimap,
            &ClosedFormConfig {
                max_iterations: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(explicit.alpha.data(), coarse.as_slice());
    }

    #[test]
    fn assembly_preflight_rejects_tiny_nonzero_and_pixel_limits() {
        let image = vec![[0.2, 0.4, 0.6]; 16 * 16];
        assert!(build_closed_form_laplacian(
            &image,
            16,
            16,
            1e-7,
            1,
            None,
            16 * 16,
            50_000_000,
            1,
            1_000_000
        )
        .is_err());
        assert!(build_closed_form_laplacian(
            &image, 16, 16, 1e-7, 1, None, 1, 50_000_000, 50_000_000, 1_000_000
        )
        .is_err());
    }

    #[test]
    fn coarse_dimension_and_nonfinite_entries_fail_closed() {
        let image = RgbImageF32::constant(2, 2, [0.2, 0.4, 0.6]).unwrap();
        let trimap = Trimap::unknown(2, 2).unwrap();
        assert!(refine_closed_form_with_coarse(
            &image,
            &AlphaMask::new(1, 2, vec![0.5, 0.5]).unwrap(),
            &trimap,
            &ClosedFormConfig::default()
        )
        .is_err());
        assert!(AlphaMask::new(2, 2, vec![f32::NAN; 4]).is_err());
    }

    #[test]
    fn underconstrained_and_single_class_systems_return_exact_coarse() {
        let image = vec![[0.2, 0.3, 0.4]; 9];
        let coarse = vec![0.37_f32; 9];
        let all_unknown = vec![false; 9];
        let lap = build_closed_form_laplacian(
            &image,
            3,
            3,
            1e-7,
            1,
            Some(&all_unknown),
            64,
            10_000,
            10_000,
            10_000_000,
        )
        .unwrap();
        let (alpha, report) = solve_constrained_alpha_with_coarse(
            &lap,
            &coarse,
            &coarse,
            3,
            3,
            &ClosedFormConfig::default(),
        )
        .unwrap();
        assert_eq!(alpha, coarse);
        assert!(report.fallback_used);
        assert!(report.fallback_reason.unwrap().contains("underconstrained"));

        let only_background = vec![TrimapClass::Background]
            .into_iter()
            .chain(std::iter::repeat_n(TrimapClass::Unknown, 8))
            .collect::<Vec<_>>();
        let known = only_background
            .iter()
            .map(|c| !matches!(c, TrimapClass::Unknown))
            .collect::<Vec<_>>();
        let lap = build_closed_form_laplacian(
            &image,
            3,
            3,
            1e-7,
            1,
            Some(&known),
            64,
            10_000,
            10_000,
            10_000_000,
        )
        .unwrap();
        let trim = only_background
            .iter()
            .map(|c| match c {
                TrimapClass::Background => 0.0,
                _ => 0.5,
            })
            .collect::<Vec<_>>();
        let (_, report) = solve_constrained_alpha_with_coarse(
            &lap,
            &trim,
            &coarse,
            3,
            3,
            &ClosedFormConfig::default(),
        )
        .unwrap();
        assert!(report.fallback_used);
        assert!(report.fallback_reason.unwrap().contains("one known class"));
    }

    #[test]
    fn all_known_constraints_override_coarse_and_small_grid_falls_back() {
        let image = vec![[0.2, 0.3, 0.4]; 4];
        let known = vec![true; 4];
        let lap = build_closed_form_laplacian(
            &image,
            2,
            2,
            1e-7,
            1,
            Some(&known),
            16,
            10_000,
            10_000,
            10_000_000,
        )
        .unwrap();
        let trim = vec![0.0, 1.0, 0.0, 1.0];
        let coarse = vec![0.37; 4];
        let (alpha, report) = solve_constrained_alpha_with_coarse(
            &lap,
            &trim,
            &coarse,
            2,
            2,
            &ClosedFormConfig::default(),
        )
        .unwrap();
        assert_eq!(alpha, trim);
        assert!(!report.fallback_used);

        let trim = vec![0.0, 0.5, 0.5, 1.0];
        let known = vec![true, false, false, true];
        let lap = build_closed_form_laplacian(
            &image,
            2,
            2,
            1e-7,
            1,
            Some(&known),
            16,
            10_000,
            10_000,
            10_000_000,
        )
        .unwrap();
        let (alpha, report) = solve_constrained_alpha_with_coarse(
            &lap,
            &trim,
            &coarse,
            2,
            2,
            &ClosedFormConfig::default(),
        )
        .unwrap();
        assert_eq!(alpha, coarse);
        assert!(report.fallback_used);
        assert!(matches!(report.status, SolveStatus::Breakdown));
    }

    #[test]
    fn public_raw_slice_entry_points_reject_zero_dimensions() {
        assert!(resize_lanczos_alpha(&[], 0, 0, 1, 1).is_err());
        assert!(resize_lanczos_rgb(&[], 0, 0, 1, 1).is_err());
        assert!(estimate_foreground_ml(&[], &[], 0, 0, 1e-5, 1, 1, 32, 1.0).is_err());
    }

    #[test]
    fn constant_large_radius_uses_spd_regularization() {
        let image = vec![[0.3_f32, 0.3, 0.3]; 17 * 17];
        let lap = build_closed_form_laplacian(
            &image,
            17,
            17,
            1e-7,
            8,
            None,
            1000,
            1_000_000,
            1_000_000,
            512 * 1024 * 1024,
        )
        .unwrap();
        assert!(lap.raw_assembly_entries() > 0);
        assert!(lap.entries().all(|(_, _, value)| value.is_finite()));
    }
}
