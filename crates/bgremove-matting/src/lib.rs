//! Matting contracts and no-op implementations for M1.
use anyhow::{ensure, Result};
use bgremove_core::{
    AlphaMask, AlphaRefiner, CanonicalImage, MaskTransform, NoOpAlphaRefiner, NoOpMaskTransform,
    RefinedMatte, Trimap,
};

pub mod m10;
pub use m10::{
    build_closed_form_laplacian, estimate_foreground_ml, refine_backgroundremover_bounded,
    refine_closed_form_with_coarse, resize_lanczos_alpha, resize_lanczos_rgb,
    solve_constrained_alpha_with_coarse, ClosedFormConfig, ClosedFormResult, SolveReport,
    SolveStatus, SparseMatrix,
};

pub use bgremove_core::TrimapClass;

/// Explicit identity transform; morphology and trimap algorithms are deferred.
#[derive(Default)]
pub struct IdentityMaskTransform;
impl MaskTransform for IdentityMaskTransform {
    fn apply(&self, image: &CanonicalImage, alpha: AlphaMask) -> Result<AlphaMask> {
        ensure!(
            image.dimensions() == alpha.dimensions(),
            "identity transform dimension mismatch"
        );
        Ok(alpha)
    }
}

/// Explicit backgroundremover-compatible hard mask. The configured value is
/// in the source contract's uint8 range [0,255], and comparison is strict:
/// pixels equal to the threshold are background.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HardThresholdTransform {
    threshold_u8: f32,
}
impl HardThresholdTransform {
    pub fn new(threshold_u8: f32) -> Result<Self> {
        ensure!(threshold_u8.is_finite(), "hard threshold must be finite");
        ensure!(
            (0.0..=255.0).contains(&threshold_u8),
            "hard threshold must be in [0,255]"
        );
        Ok(Self { threshold_u8 })
    }
    pub fn threshold_u8(&self) -> f32 {
        self.threshold_u8
    }
}
impl MaskTransform for HardThresholdTransform {
    fn apply(&self, image: &CanonicalImage, alpha: AlphaMask) -> Result<AlphaMask> {
        ensure!(
            image.dimensions() == alpha.dimensions(),
            "hard threshold dimension mismatch"
        );
        let values = alpha
            .data()
            .iter()
            .map(|v| {
                // AlphaMask values from encoded masks are exact u8/255. Flooring
                // makes this transform deterministic for arbitrary f32 callers.
                let byte = (v.clamp(0.0, 1.0) * 255.0).floor();
                if byte > self.threshold_u8 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        AlphaMask::new(alpha.width(), alpha.height(), values)
    }
}

pub type HardThreshold = HardThresholdTransform;

/// M1 refiner marker that returns the supplied coarse alpha.
#[derive(Default)]
pub struct NoOpRefiner(NoOpAlphaRefiner);
impl AlphaRefiner for NoOpRefiner {
    fn refine(
        &mut self,
        image: &CanonicalImage,
        coarse: &AlphaMask,
        trimap: &Trimap,
    ) -> Result<RefinedMatte> {
        self.0.refine(image, coarse, trimap)
    }
}

pub type IdentityTransform = NoOpMaskTransform;

/// Radius expressed in source pixels or as a fraction of the smaller image
/// dimension. Relative radii use round-to-nearest and reject negative values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Radius {
    Absolute(u32),
    Relative(f32),
}

impl Radius {
    pub fn resolve(self, width: u32, height: u32) -> Result<u32> {
        ensure!(
            width > 0 && height > 0,
            "radius requires non-zero dimensions"
        );
        match self {
            Self::Absolute(value) => {
                ensure!(
                    value <= width.max(height),
                    "absolute radius exceeds image bounds"
                );
                Ok(value)
            }
            Self::Relative(value) => {
                ensure!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "relative radius must be finite and in [0,1]"
                );
                Ok((value * width.min(height) as f32).round() as u32)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorderValue {
    False,
    True,
    /// Reflect the nearest in-bounds samples using SciPy/skimage's
    /// half-sample symmetric convention (`-1 -> 0`, `n -> n - 1`).
    Reflect,
}

fn checked_grid(width: u32, height: u32, values: &[u8]) -> Result<(usize, usize)> {
    ensure!(
        width > 0 && height > 0,
        "morphology dimensions must be non-zero"
    );
    ensure!(
        width <= i32::MAX as u32 && height <= i32::MAX as u32,
        "morphology dimensions exceed coordinate bounds"
    );
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .ok_or_else(|| anyhow::anyhow!("morphology dimensions overflow"))?;
    ensure!(values.len() == expected, "morphology input length mismatch");
    Ok((w, h))
}

fn offsets(radius: u32, disk: bool) -> Vec<(i32, i32)> {
    let r = radius as i32;
    (-r..=r)
        .flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)))
        .filter(|(dx, dy)| {
            !disk
                || i64::from(*dx) * i64::from(*dx) + i64::from(*dy) * i64::from(*dy)
                    <= i64::from(r) * i64::from(r)
        })
        .collect()
}

fn morph(
    values: &[u8],
    width: u32,
    height: u32,
    radius: u32,
    dilate: bool,
    border: BorderValue,
    disk: bool,
) -> Result<Vec<u8>> {
    let (w, h) = checked_grid(width, height, values)?;
    ensure!(
        radius <= width.max(height),
        "morphology radius exceeds image bounds"
    );
    let offsets = offsets(radius, disk);
    let mut output = vec![0; values.len()];
    let reflect = |index: i32, limit: usize| -> usize {
        if limit == 1 {
            return 0;
        }
        let period = 2 * limit as i32;
        let mut value = index.rem_euclid(period);
        if value >= limit as i32 {
            value = period - 1 - value;
        }
        value as usize
    };
    for y in 0..h {
        for x in 0..w {
            let mut result = !dilate;
            for (dx, dy) in &offsets {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let value = match border {
                    BorderValue::Reflect => values[reflect(nx, w) + reflect(ny, h) * w] != 0,
                    BorderValue::False | BorderValue::True => {
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            matches!(border, BorderValue::True)
                        } else {
                            values[ny as usize * w + nx as usize] != 0
                        }
                    }
                };
                if dilate {
                    result |= value;
                } else {
                    result &= value;
                }
            }
            output[y * w + x] = u8::from(result);
        }
    }
    Ok(output)
}

/// Erode with the exact square-anchor convention used by SciPy for an
/// explicitly sized structure.  Even kernels are anchored at the lower-right
/// of their geometric centre (`origin=0`), so a 2x2 kernel samples offsets
/// `[-1, 0]` in each axis rather than silently becoming a 3x3 radius kernel.
pub fn binary_erosion_kernel(
    values: &[u8],
    width: u32,
    height: u32,
    kernel_size: u32,
    border: BorderValue,
) -> Result<Vec<u8>> {
    let (w, h) = checked_grid(width, height, values)?;
    ensure!(kernel_size > 0, "erosion kernel must be non-zero");
    if kernel_size > width.max(height) {
        ensure!(
            !matches!(border, BorderValue::Reflect),
            "oversized reflected erosion is unsupported"
        );
        let left = u64::from(kernel_size / 2);
        let right = u64::from((kernel_size - 1) / 2);
        let stride = w + 1;
        let mut zero_prefix = vec![0u64; (h + 1) * stride];
        for y in 0..h {
            let mut row_zeros = 0u64;
            for x in 0..w {
                if values[y * w + x] == 0 {
                    row_zeros += 1;
                }
                zero_prefix[(y + 1) * stride + x + 1] = zero_prefix[y * stride + x + 1] + row_zeros;
            }
        }
        let mut output = vec![0; values.len()];
        for y in 0..h {
            for x in 0..w {
                let x_low = (x as u64).saturating_sub(left);
                let x_high = (x as u64).saturating_add(right);
                let y_low = (y as u64).saturating_sub(left);
                let y_high = (y as u64).saturating_add(right);
                let outside = (x as u64) < left
                    || (y as u64) < left
                    || x_high >= w as u64
                    || y_high >= h as u64;
                if outside && matches!(border, BorderValue::False) {
                    continue;
                }
                let x0 = x_low.min(w as u64) as usize;
                let x1 = x_high.min((w - 1) as u64) as usize;
                let y0 = y_low.min(h as u64) as usize;
                let y1 = y_high.min((h - 1) as u64) as usize;
                let zeros = zero_prefix[(y1 + 1) * stride + x1 + 1]
                    - zero_prefix[y0 * stride + x1 + 1]
                    - zero_prefix[(y1 + 1) * stride + x0]
                    + zero_prefix[y0 * stride + x0];
                output[y * w + x] = u8::from(zeros == 0);
            }
        }
        return Ok(output);
    }
    let left = (kernel_size / 2) as i32;
    let right = ((kernel_size - 1) / 2) as i32;
    let mut output = vec![0; values.len()];
    let reflect = |index: i32, limit: usize| -> usize {
        if limit == 1 {
            return 0;
        }
        let period = 2 * limit as i32;
        let mut value = index.rem_euclid(period);
        if value >= limit as i32 {
            value = period - 1 - value;
        }
        value as usize
    };
    for y in 0..h {
        for x in 0..w {
            let mut result = true;
            'neighbour: for dy in -left..=right {
                for dx in -left..=right {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let value = match border {
                        BorderValue::Reflect => values[reflect(nx, w) + reflect(ny, h) * w] != 0,
                        BorderValue::False | BorderValue::True => {
                            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                                matches!(border, BorderValue::True)
                            } else {
                                values[ny as usize * w + nx as usize] != 0
                            }
                        }
                    };
                    if !value {
                        result = false;
                        break 'neighbour;
                    }
                }
            }
            output[y * w + x] = u8::from(result);
        }
    }
    Ok(output)
}

pub fn threshold_binary(values: &[u8], threshold: u8) -> Vec<u8> {
    values
        .iter()
        .map(|value| u8::from(*value > threshold))
        .collect()
}

pub fn binary_erosion(
    values: &[u8],
    width: u32,
    height: u32,
    radius: Radius,
    border: BorderValue,
) -> Result<Vec<u8>> {
    morph(
        values,
        width,
        height,
        radius.resolve(width, height)?,
        false,
        border,
        false,
    )
}

pub fn binary_dilation(
    values: &[u8],
    width: u32,
    height: u32,
    radius: Radius,
    border: BorderValue,
) -> Result<Vec<u8>> {
    morph(
        values,
        width,
        height,
        radius.resolve(width, height)?,
        true,
        border,
        false,
    )
}

pub fn disk_opening(values: &[u8], width: u32, height: u32, radius: Radius) -> Result<Vec<u8>> {
    let resolved = radius.resolve(width, height)?;
    let eroded = morph(
        values,
        width,
        height,
        resolved,
        false,
        BorderValue::Reflect,
        true,
    )?;
    morph(
        &eroded,
        width,
        height,
        resolved,
        true,
        BorderValue::Reflect,
        true,
    )
}

/// Gaussian convolution with scipy's default `truncate=4.0` and half-sample
/// symmetric (`reflect`) border rule. The result remains in the input's
/// numeric range; callers choose quantization explicitly.
pub fn gaussian_blur(values: &[u8], width: u32, height: u32, sigma: f64) -> Result<Vec<f64>> {
    let (w, h) = checked_grid(width, height, values)?;
    ensure!(
        sigma.is_finite() && sigma > 0.0,
        "Gaussian sigma must be positive and finite"
    );
    let radius = (4.0 * sigma + 0.5).floor() as i32;
    ensure!(radius <= 4096, "Gaussian radius is too large");
    let kernel = (-radius..=radius)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp())
        .collect::<Vec<_>>();
    let norm = kernel.iter().sum::<f64>();
    let reflect = |index: i32, limit: usize| -> usize {
        if limit == 1 {
            return 0;
        }
        let period = 2 * limit as i32;
        let mut value = index.rem_euclid(period);
        if value >= limit as i32 {
            value = period - 1 - value;
        }
        value as usize
    };
    let mut horizontal = vec![0.0; values.len()];
    for y in 0..h {
        for x in 0..w {
            horizontal[y * w + x] = (-radius..=radius)
                .zip(&kernel)
                .map(|(offset, weight)| {
                    values[y * w + reflect(x as i32 + offset, w)] as f64 * weight
                })
                .sum::<f64>()
                / norm;
        }
    }
    let mut output = vec![0.0; values.len()];
    for y in 0..h {
        for x in 0..w {
            output[y * w + x] = (-radius..=radius)
                .zip(&kernel)
                .map(|(offset, weight)| horizontal[reflect(y as i32 + offset, h) * w + x] * weight)
                .sum::<f64>()
                / norm;
        }
    }
    Ok(output)
}

fn scipy_default_erosion(
    values: &[u8],
    width: u32,
    height: u32,
    border: BorderValue,
) -> Result<Vec<u8>> {
    let (w, h) = checked_grid(width, height, values)?;
    let neighbours = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
    let mut output = vec![0; values.len()];
    for y in 0..h {
        for x in 0..w {
            let mut result = true;
            for (dx, dy) in neighbours {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let value = if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    matches!(border, BorderValue::True)
                } else {
                    values[ny as usize * w + nx as usize] != 0
                };
                result &= value;
            }
            output[y * w + x] = u8::from(result);
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetricTrimapConfig {
    pub foreground_threshold: u8,
    pub background_threshold: u8,
    pub erosion_radius: Radius,
}

/// Literal rembg `trimap_from_mask` configuration. Unlike generic radius
/// morphology, `erode_size` is the actual square structure size. `0` selects
/// SciPy's default two-dimensional connectivity-1 cross (`structure=None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RembgTrimapConfig {
    pub foreground_threshold: u8,
    pub background_threshold: u8,
    pub erode_size: u32,
}

fn rembg_erode(
    values: &[u8],
    width: u32,
    height: u32,
    size: u32,
    border: BorderValue,
) -> Result<Vec<u8>> {
    if size == 0 {
        scipy_default_erosion(values, width, height, border)
    } else {
        binary_erosion_kernel(values, width, height, size, border)
    }
}

pub fn rembg_symmetric_trimap(
    values: &[u8],
    width: u32,
    height: u32,
    config: RembgTrimapConfig,
) -> Result<Trimap> {
    ensure!(
        config.background_threshold <= config.foreground_threshold,
        "trimap thresholds are inverted"
    );
    let (w, h) = checked_grid(width, height, values)?;
    let foreground = values
        .iter()
        .map(|v| u8::from(*v > config.foreground_threshold))
        .collect::<Vec<_>>();
    let background = values
        .iter()
        .map(|v| u8::from(*v < config.background_threshold))
        .collect::<Vec<_>>();
    let foreground = rembg_erode(
        &foreground,
        width,
        height,
        config.erode_size,
        BorderValue::False,
    )?;
    let background = rembg_erode(
        &background,
        width,
        height,
        config.erode_size,
        BorderValue::True,
    )?;
    let classes = (0..w * h)
        .map(|i| {
            if foreground[i] != 0 {
                TrimapClass::Foreground
            } else if background[i] != 0 {
                TrimapClass::Background
            } else {
                TrimapClass::Unknown
            }
        })
        .collect();
    Trimap::new(width, height, classes)
}

pub fn symmetric_rembg_trimap(
    values: &[u8],
    width: u32,
    height: u32,
    config: SymmetricTrimapConfig,
) -> Result<Trimap> {
    ensure!(
        config.background_threshold <= config.foreground_threshold,
        "trimap thresholds are inverted"
    );
    let (w, h) = checked_grid(width, height, values)?;
    let foreground = values
        .iter()
        .map(|v| u8::from(*v > config.foreground_threshold))
        .collect::<Vec<_>>();
    let background = values
        .iter()
        .map(|v| u8::from(*v < config.background_threshold))
        .collect::<Vec<_>>();
    let foreground = binary_erosion(
        &foreground,
        width,
        height,
        config.erosion_radius,
        BorderValue::False,
    )?;
    let background = binary_erosion(
        &background,
        width,
        height,
        config.erosion_radius,
        BorderValue::True,
    )?;
    let classes = (0..w * h)
        .map(|i| {
            if foreground[i] != 0 {
                TrimapClass::Foreground
            } else if background[i] != 0 {
                TrimapClass::Background
            } else {
                TrimapClass::Unknown
            }
        })
        .collect();
    Trimap::new(width, height, classes)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarveKitTrimapConfig {
    pub probability_threshold: u8,
    pub dilation_radius: Radius,
    pub erosion_iterations: u32,
}

pub fn carvekit_probability_trimap(
    values: &[u8],
    width: u32,
    height: u32,
    config: CarveKitTrimapConfig,
) -> Result<Trimap> {
    let (w, h) = checked_grid(width, height, values)?;
    // This follows CarveKit's operation order exactly: `prob_filter`, the
    // CV2 generator with erosion disabled, `prob_as_unknown_area`, then
    // `post_erosion`.  Keep the intermediate byte levels here because the
    // source deliberately uses 127/200 as sentinels before its final mapping.
    let filtered = threshold_binary(values, config.probability_threshold);
    let dilated = binary_dilation(
        &filtered,
        width,
        height,
        config.dilation_radius,
        BorderValue::False,
    )?;
    let mut trimap = vec![0u8; w * h];
    for i in 0..w * h {
        let dilation_value = if dilated[i] != 0 { 127 } else { 0 };
        trimap[i] = if filtered[i] != 0 {
            255
        } else {
            dilation_value
        };
    }
    // `prob_as_unknown_area` runs after the CV2 generator and uses the
    // original probability mask, not the filtered mask.
    for (value, &probability) in trimap.iter_mut().zip(values) {
        if probability <= config.probability_threshold && probability > 0 {
            *value = 127;
        }
    }
    if config.erosion_iterations > 0 {
        let without_unknown = trimap
            .iter()
            .map(|value| u8::from(*value == 255))
            .collect::<Vec<_>>();
        let mut certain = without_unknown.clone();
        for _ in 0..config.erosion_iterations {
            certain = binary_erosion(
                &certain,
                width,
                height,
                Radius::Absolute(1),
                // OpenCV's default erosion border is the neutral value 255
                // (unlike rembg's explicit foreground=false border).
                BorderValue::True,
            )?;
        }
        for i in 0..w * h {
            if certain[i] == 0 && without_unknown[i] != 0 {
                trimap[i] = 127;
            }
        }
    }
    let classes = trimap
        .into_iter()
        .map(|value| match value {
            0 => TrimapClass::Background,
            255 => TrimapClass::Foreground,
            _ => TrimapClass::Unknown,
        })
        .collect::<Vec<_>>();
    Trimap::new(width, height, classes)
}

pub fn rembg_post_process(values: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let opened = disk_opening(values, width, height, Radius::Absolute(1))?;
    let opened = opened.iter().map(|v| v * 255).collect::<Vec<_>>();
    Ok(gaussian_blur(&opened, width, height, 2.0)?
        .into_iter()
        .map(|value| if value < 127.0 { 0 } else { 255 })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_threshold_is_strict_and_validated() {
        let image = CanonicalImage::new(3, 1, vec![[0.0; 3]; 3]).unwrap();
        let alpha = AlphaMask::new(3, 1, vec![0.0, 128.0 / 255.0, 129.0 / 255.0]).unwrap();
        let transform = HardThresholdTransform::new(128.0).unwrap();
        assert_eq!(
            transform.apply(&image, alpha).unwrap().data(),
            &[0.0, 0.0, 1.0]
        );
        assert!(HardThresholdTransform::new(f32::NAN).is_err());
        assert!(HardThresholdTransform::new(-1.0).is_err());
        assert!(HardThresholdTransform::new(256.0).is_err());
    }

    #[test]
    fn morphology_matches_hand_matrix_with_explicit_borders() {
        let input = [0, 1, 0, 1, 1, 0, 0, 1, 0];
        assert_eq!(
            binary_dilation(&input, 3, 3, Radius::Absolute(1), BorderValue::False).unwrap(),
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(
            binary_erosion(&[1; 9], 3, 3, Radius::Absolute(1), BorderValue::False).unwrap(),
            [0, 0, 0, 0, 1, 0, 0, 0, 0]
        );
        assert_eq!(threshold_binary(&[0, 10, 11, 255], 10), [0, 0, 1, 1]);
    }

    #[test]
    fn trimap_values_are_disjoint_and_radius_is_deterministic() {
        let config = SymmetricTrimapConfig {
            foreground_threshold: 200,
            background_threshold: 20,
            erosion_radius: Radius::Absolute(0),
        };
        let trimap = symmetric_rembg_trimap(&[0, 10, 30, 220, 255, 100], 3, 2, config).unwrap();
        assert_eq!(
            trimap.data(),
            &[
                TrimapClass::Background,
                TrimapClass::Background,
                TrimapClass::Unknown,
                TrimapClass::Foreground,
                TrimapClass::Foreground,
                TrimapClass::Unknown
            ]
        );
        assert_eq!(Radius::Relative(0.5).resolve(5, 3).unwrap(), 2);
        assert!(Radius::Relative(f32::NAN).resolve(5, 3).is_err());
        assert!(Radius::Relative(f32::INFINITY).resolve(5, 3).is_err());
        assert!(Radius::Relative(1.01).resolve(5, 3).is_err());
        assert!(binary_dilation(
            &[0; 15],
            5,
            3,
            Radius::Absolute(u32::MAX),
            BorderValue::False
        )
        .is_err());
        assert!(binary_dilation(&[1], 0, 1, Radius::Absolute(1), BorderValue::False).is_err());
        assert!(binary_dilation(&[], 2, 2, Radius::Absolute(1), BorderValue::False).is_err());
        assert!(binary_erosion_kernel(&[1], 1, 1, 0, BorderValue::False).is_err());
        assert_eq!(
            binary_erosion_kernel(&[1], 1, 1, 2, BorderValue::False).unwrap(),
            [0]
        );
    }

    #[test]
    fn carvekit_probability_unknowns_never_overlap_known_regions() {
        let config = CarveKitTrimapConfig {
            probability_threshold: 200,
            dilation_radius: Radius::Absolute(1),
            erosion_iterations: 0,
        };
        let trimap =
            carvekit_probability_trimap(&[0, 50, 201, 255, 0, 230, 10, 0, 0], 3, 3, config)
                .unwrap();
        assert!(trimap.data().iter().all(|value| matches!(
            value,
            TrimapClass::Background | TrimapClass::Unknown | TrimapClass::Foreground
        )));
        assert_eq!(trimap.data()[2], TrimapClass::Foreground);
        assert_eq!(trimap.data()[1], TrimapClass::Unknown);
    }

    #[test]
    fn gaussian_and_opening_are_finite_and_border_deterministic() {
        let values = [0, 0, 0, 255, 255, 0, 0, 0, 0];
        let opened = disk_opening(&values, 3, 3, Radius::Absolute(1)).unwrap();
        assert_eq!(opened, [0; 9]);
        assert_eq!(
            disk_opening(&[255; 9], 3, 3, Radius::Absolute(1)).unwrap(),
            [1; 9]
        );
        let blur = gaussian_blur(&values, 3, 3, 2.0).unwrap();
        assert!(blur
            .iter()
            .all(|value| value.is_finite() && (0.0..=255.0).contains(value)));
        assert!(gaussian_blur(&values, 3, 3, f64::NAN).is_err());
        assert!(gaussian_blur(&values, 3, 3, f64::INFINITY).is_err());
    }

    #[test]
    fn even_square_erosion_uses_scipy_origin_zero_anchor() {
        let mut values = [0u8; 25];
        for y in 2..4 {
            for x in 2..4 {
                values[y * 5 + x] = 1;
            }
        }
        let result = binary_erosion_kernel(&values, 5, 5, 2, BorderValue::False).unwrap();
        let expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn oversized_square_erosion_preserves_anchored_border_window() {
        let result = binary_erosion_kernel(&[1, 1, 1, 1, 0], 5, 1, 6, BorderValue::True).unwrap();
        assert_eq!(result, [1, 1, 0, 0, 0]);
    }

    #[test]
    fn exhaustive_small_trimaps_are_deterministic_and_disjoint() {
        for width in 1..=4u32 {
            for height in 1..=3u32 {
                let len = (width * height) as usize;
                let values = (0..len)
                    .map(|index| {
                        ((index * 73 + width as usize * 19 + height as usize * 7) % 256) as u8
                    })
                    .collect::<Vec<_>>();
                for threshold in [0, 1, 63, 127, 200, 255] {
                    let symmetric = rembg_symmetric_trimap(
                        &values,
                        width,
                        height,
                        RembgTrimapConfig {
                            foreground_threshold: threshold,
                            background_threshold: threshold.min(10),
                            erode_size: (width + height) % (width.max(height) + 1),
                        },
                    )
                    .unwrap();
                    let carvekit = carvekit_probability_trimap(
                        &values,
                        width,
                        height,
                        CarveKitTrimapConfig {
                            probability_threshold: threshold,
                            dilation_radius: Radius::Relative(0.5),
                            erosion_iterations: 0,
                        },
                    )
                    .unwrap();
                    assert!(symmetric.data().iter().all(|value| matches!(
                        value,
                        TrimapClass::Background | TrimapClass::Unknown | TrimapClass::Foreground
                    )));
                    assert!(carvekit.data().iter().all(|value| matches!(
                        value,
                        TrimapClass::Background | TrimapClass::Unknown | TrimapClass::Foreground
                    )));
                    assert_eq!(
                        symmetric.data(),
                        rembg_symmetric_trimap(
                            &values,
                            width,
                            height,
                            RembgTrimapConfig {
                                foreground_threshold: threshold,
                                background_threshold: threshold.min(10),
                                erode_size: (width + height) % (width.max(height) + 1),
                            },
                        )
                        .unwrap()
                        .data()
                    );
                    assert_eq!(
                        carvekit.data(),
                        carvekit_probability_trimap(
                            &values,
                            width,
                            height,
                            CarveKitTrimapConfig {
                                probability_threshold: threshold,
                                dilation_radius: Radius::Relative(0.5),
                                erosion_iterations: 0,
                            },
                        )
                        .unwrap()
                        .data()
                    );
                }
            }
        }
    }

    #[test]
    fn pinned_python_trimaps_match_pixel_for_pixel_and_provenance() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixture = root.join("tests/fixtures/m9/reference");
        let report: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("tests/fixtures/m9/authoritative-report.json"))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(report["authoritative_sources_executed"], true);
        let profiles = report["profiles"].as_array().unwrap();
        assert_eq!(profiles.len(), 2);
        assert_eq!(
            profiles[0]["source"]["commit"],
            "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
        );
        assert_eq!(
            profiles[1]["source"]["commit"],
            "f141a311af67fb1da64269c508a6d1f786420801"
        );
        assert_eq!(
            profiles[0]["source"]["source_tree_sha256"],
            "ea01e0cb5fbb6ff3cf62b39adf28357a55b6b94d3d9f40add8fd165b821066cf"
        );
        assert_eq!(
            profiles[1]["source"]["source_tree_sha256"],
            "22bc32aa1dc0d2d0eb65b0a4c9238204457986fe07b86120cb6b6fe17beb41fb"
        );
        assert_eq!(
            profiles[0]["source"]["license_sha256"],
            "90a3215072968fd304669c5389f04f1274a587abdd0507d99dead0f5511f8999"
        );
        assert_eq!(
            profiles[1]["source"]["license_sha256"],
            "0bf5be32b9b3623d3f001a108569e8456819a72af7a39e2aee67747527ecfd17"
        );
        assert_eq!(
            profiles[0]["source"]["source_file_sha256"]["projects/python/rembg/rembg/matting.py"],
            "e86b7e608354abd24499f64ef98edfd0e29d66b65dbc3995e46220bdcfd833e1"
        );
        assert_eq!(
            profiles[0]["source"]["source_file_sha256"]["projects/python/rembg/rembg/bg.py"],
            "2e1977b6d16d5369e2c20c5ca5d9ba2e2c0228f8beef39c5d2296df1879df62b"
        );
        for profile in profiles {
            assert_eq!(profile["source"]["tracked_source_clean"], true);
            for hash in profile["source"]["source_file_sha256"]
                .as_object()
                .unwrap()
                .values()
            {
                assert_eq!(hash.as_str().unwrap().len(), 64);
            }
            assert_eq!(
                profile["source"]["source_tree_sha256"]
                    .as_str()
                    .unwrap()
                    .len(),
                64
            );
            assert_eq!(
                profile["source"]["license_sha256"].as_str().unwrap().len(),
                64
            );
            assert_eq!(profile["output"]["sha256"].as_str().unwrap().len(), 64);
        }
        let input = image::open(fixture.join("input-mask.png"))
            .unwrap()
            .to_luma8();
        let values = input.as_raw();
        let symmetric = rembg_symmetric_trimap(
            values,
            input.width(),
            input.height(),
            RembgTrimapConfig {
                foreground_threshold: 200,
                background_threshold: 10,
                erode_size: 3,
            },
        )
        .unwrap();
        let expected = image::open(fixture.join("rembg-symmetric-trimap.png"))
            .unwrap()
            .to_luma8();
        let expected_classes = expected
            .as_raw()
            .iter()
            .map(|value| match value {
                0 => TrimapClass::Background,
                255 => TrimapClass::Foreground,
                128 => TrimapClass::Unknown,
                other => panic!("unexpected rembg trimap value {other}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(symmetric.data(), expected_classes.as_slice());

        let carvekit = carvekit_probability_trimap(
            values,
            input.width(),
            input.height(),
            CarveKitTrimapConfig {
                probability_threshold: 200,
                dilation_radius: Radius::Absolute(1),
                erosion_iterations: 1,
            },
        )
        .unwrap();
        let expected = image::open(fixture.join("carvekit-probability-trimap.png"))
            .unwrap()
            .to_luma8();
        let expected_classes = expected
            .as_raw()
            .iter()
            .map(|value| match value {
                0 => TrimapClass::Background,
                255 => TrimapClass::Foreground,
                127 => TrimapClass::Unknown,
                other => panic!("unexpected CarveKit trimap value {other}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(carvekit.data(), expected_classes.as_slice());

        let erode_input = image::open(fixture.join("erode-mask.png"))
            .unwrap()
            .to_luma8();
        for (size, name) in [
            (10, "rembg-symmetric-trimap-e10.png"),
            (0, "rembg-symmetric-trimap-e0.png"),
        ] {
            let actual = rembg_symmetric_trimap(
                erode_input.as_raw(),
                erode_input.width(),
                erode_input.height(),
                RembgTrimapConfig {
                    foreground_threshold: 200,
                    background_threshold: 10,
                    erode_size: size,
                },
            )
            .unwrap();
            let expected = image::open(fixture.join(name)).unwrap().to_luma8();
            for (class, value) in actual.data().iter().zip(expected.as_raw()) {
                let expected_class = match value {
                    0 => TrimapClass::Background,
                    128 => TrimapClass::Unknown,
                    255 => TrimapClass::Foreground,
                    other => panic!("unexpected rembg value {other}"),
                };
                assert_eq!(*class, expected_class);
            }
        }

        let post_input = image::open(fixture.join("post-process-input.png"))
            .unwrap()
            .to_luma8();
        let post_expected = image::open(fixture.join("rembg-post-process.png"))
            .unwrap()
            .to_luma8();
        assert_eq!(
            rembg_post_process(post_input.as_raw(), post_input.width(), post_input.height())
                .unwrap(),
            post_expected.as_raw().as_slice()
        );

        let gaussian: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixture.join("gaussian-values.json")).unwrap(),
        )
        .unwrap();
        let gaussian_input = gaussian["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u8)
            .collect::<Vec<_>>();
        let gaussian_expected = gaussian["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let gaussian_actual = gaussian_blur(&gaussian_input, 5, 1, 1.3).unwrap();
        assert!(gaussian_actual
            .iter()
            .zip(gaussian_expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-10));

        let oversized: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixture.join("oversized-erosion.json")).unwrap(),
        )
        .unwrap();
        let oversized_input = oversized["input"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u8)
            .collect::<Vec<_>>();
        assert_eq!(
            binary_erosion_kernel(&oversized_input, 5, 1, 6, BorderValue::True).unwrap(),
            [1, 1, 0, 0, 0]
        );
    }
}
