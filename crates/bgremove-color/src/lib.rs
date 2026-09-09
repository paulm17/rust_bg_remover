//! Straight-alpha foreground colour and deterministic compositing contracts.
use anyhow::{ensure, Context, Result};
use bgremove_core::{
    AlphaMask, CanonicalImage, ForegroundEstimator, NoOpForegroundEstimator, RefinedMatte,
    RgbImageF32,
};
use bgremove_matting::estimate_foreground_ml;

/// Working colour space for foreground recovery.  The encoded-sRGB mode is
/// the PyMatting/backgroundremover contract; linear-light is an explicit
/// ablation and never selected implicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForegroundWorkingSpace {
    #[default]
    EncodedSrgb,
    LinearLight,
}

/// Original-RGB control estimator; no decontamination is performed.
#[derive(Default)]
pub struct OriginalRgbEstimator(NoOpForegroundEstimator);
impl ForegroundEstimator for OriginalRgbEstimator {
    fn estimate(&self, image: &CanonicalImage, matte: &RefinedMatte) -> Result<RgbImageF32> {
        ensure!(
            matte.alpha().dimensions() == image.dimensions(),
            "original RGB estimator dimension mismatch"
        );
        self.0.estimate(image, matte)
    }
}

pub const M2_COLOR_POLICY: &str =
    "original-rgb; straight alpha; explicit encoded-srgb or linear-light compositing";

/// PyMatting's multilevel foreground estimator, exposed as a typed pipeline
/// candidate.  The defaults are the values used by PyMatting 1.1.15.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultilevelForegroundConfig {
    pub regularization: f32,
    pub n_small_iterations: usize,
    pub n_big_iterations: usize,
    pub small_size: u32,
    pub gradient_weight: f32,
    pub working_space: ForegroundWorkingSpace,
}

impl Default for MultilevelForegroundConfig {
    fn default() -> Self {
        Self {
            regularization: 1e-5,
            n_small_iterations: 10,
            n_big_iterations: 2,
            small_size: 32,
            gradient_weight: 1.0,
            working_space: ForegroundWorkingSpace::EncodedSrgb,
        }
    }
}

impl MultilevelForegroundConfig {
    fn validate(self) -> Result<()> {
        ensure!(
            self.regularization.is_finite() && self.regularization >= 0.0,
            "multilevel regularization must be finite and non-negative"
        );
        ensure!(
            self.gradient_weight.is_finite() && self.gradient_weight >= 0.0,
            "multilevel gradient weight must be finite and non-negative"
        );
        ensure!(
            self.small_size > 0,
            "multilevel small_size must be positive"
        );
        Ok(())
    }
}

/// Source-faithful PyMatting-style multilevel foreground/background updates.
#[derive(Clone, Copy, Debug, Default)]
pub struct MultilevelForegroundEstimator {
    pub config: MultilevelForegroundConfig,
}

impl MultilevelForegroundEstimator {
    pub fn new(config: MultilevelForegroundConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }
}

impl ForegroundEstimator for MultilevelForegroundEstimator {
    fn estimate(&self, image: &CanonicalImage, matte: &RefinedMatte) -> Result<RgbImageF32> {
        self.config.validate()?;
        ensure!(
            matte.alpha().dimensions() == image.dimensions(),
            "multilevel estimator dimension mismatch"
        );
        let working = image_in_space(image.rgb(), self.config.working_space);
        let (foreground, _) = estimate_foreground_ml(
            working.data(),
            matte.alpha().data(),
            working.width(),
            working.height(),
            self.config.regularization,
            self.config.n_small_iterations,
            self.config.n_big_iterations,
            self.config.small_size,
            self.config.gradient_weight,
        )?;
        Ok(image_from_space(&foreground, self.config.working_space))
    }
}

/// Policy for samples outside a box-blur footprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoxEdgePolicy {
    /// OpenCV `cv2.blur`'s `BORDER_DEFAULT` (`BORDER_REFLECT_101`).
    #[default]
    Reflect101,
    /// Explicit non-source extension useful for ablations and diagnostics.
    Clamp,
}

/// Parameters for the published approximate fast estimator. The two kernel
/// fields are widths (not radii), with source defaults 90/6.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FastForegroundConfig {
    pub coarse_kernel_width: u32,
    pub fine_kernel_width: u32,
    /// Number of source-compatible passes at the coarse kernel width.
    pub coarse_iterations: u32,
    /// Number of source-compatible passes at the fine kernel width.
    pub fine_iterations: u32,
    pub reference_size: u32,
    pub epsilon: f32,
    pub working_space: ForegroundWorkingSpace,
    pub edge_policy: BoxEdgePolicy,
    /// Optional reference-size scaling extension. Source parity keeps this
    /// false and uses literal 90/6 widths.
    pub scale_kernel_widths: bool,
}

impl Default for FastForegroundConfig {
    fn default() -> Self {
        Self {
            coarse_kernel_width: 90,
            fine_kernel_width: 6,
            coarse_iterations: 1,
            fine_iterations: 1,
            reference_size: 1024,
            epsilon: 1e-5,
            working_space: ForegroundWorkingSpace::EncodedSrgb,
            edge_policy: BoxEdgePolicy::Reflect101,
            scale_kernel_widths: false,
        }
    }
}

impl FastForegroundConfig {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.coarse_kernel_width > 0 && self.fine_kernel_width > 0,
            "fast kernel widths must be positive"
        );
        ensure!(
            self.coarse_iterations > 0 && self.fine_iterations > 0,
            "fast iteration counts must be positive"
        );
        ensure!(
            self.coarse_iterations <= 64 && self.fine_iterations <= 64,
            "fast iteration counts exceed the safety limit"
        );
        ensure!(
            !self.scale_kernel_widths || self.reference_size > 0,
            "fast reference_size must be positive when kernel scaling is enabled"
        );
        ensure!(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "fast epsilon must be finite and positive"
        );
        Ok(())
    }

    pub fn resolved_kernel_widths(self, width: u32, height: u32) -> Result<(u32, u32)> {
        self.validate()?;
        ensure!(
            width > 0 && height > 0,
            "fast estimator dimensions must be positive"
        );
        let scale = if self.scale_kernel_widths {
            (width.max(height) as f64 / self.reference_size as f64)
                .max(1.0 / self.reference_size as f64)
        } else {
            1.0
        };
        let resolve = |kernel_width: u32| -> Result<u32> {
            let scaled = if self.scale_kernel_widths {
                (kernel_width as f64 * scale).round()
            } else {
                kernel_width as f64
            };
            ensure!(
                scaled.is_finite() && scaled <= u32::MAX as f64,
                "fast kernel width overflow"
            );
            Ok((scaled as u32).max(1))
        };
        Ok((
            resolve(self.coarse_kernel_width)?,
            resolve(self.fine_kernel_width)?,
        ))
    }

    /// Exact PhotoRoom source configuration: literal 90/6 kernel widths, one
    /// coarse pass followed by one fine pass, OpenCV reflect-101 borders, and
    /// no reference-size scaling.
    pub fn photoroom_source() -> Self {
        Self {
            coarse_kernel_width: 90,
            fine_kernel_width: 6,
            edge_policy: BoxEdgePolicy::Reflect101,
            scale_kernel_widths: false,
            ..Self::default()
        }
    }
}

/// Approximate coarse-to-fine foreground estimator using the plan's published
/// arithmetic. Each pass performs one O(width*height) box blur for alpha and
/// each premultiplied colour field. The source returns `(F, blurred_B)`;
/// importantly, the next pass consumes that blurred background and no
/// invented symmetric background residual update is applied.
#[derive(Clone, Copy, Debug, Default)]
pub struct FastForegroundEstimator {
    pub config: FastForegroundConfig,
}

impl FastForegroundEstimator {
    pub fn new(config: FastForegroundConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }
}

impl ForegroundEstimator for FastForegroundEstimator {
    fn estimate(&self, image: &CanonicalImage, matte: &RefinedMatte) -> Result<RgbImageF32> {
        self.config.validate()?;
        ensure!(
            matte.alpha().dimensions() == image.dimensions(),
            "fast estimator dimension mismatch"
        );
        let (coarse, fine) = self
            .config
            .resolved_kernel_widths(image.width(), image.height())?;
        let working = image_in_space(image.rgb(), self.config.working_space);
        let mut foreground = working.data().to_vec();
        let mut background = working.data().to_vec();
        for (kernel_width, iterations) in [
            (coarse, self.config.coarse_iterations),
            (fine, self.config.fine_iterations),
        ] {
            for _ in 0..iterations {
                let (next_foreground, next_background) = fast_pass(
                    working.data(),
                    matte.alpha().data(),
                    &foreground,
                    &background,
                    image.width(),
                    image.height(),
                    kernel_width,
                    self.config.epsilon,
                    self.config.edge_policy,
                )?;
                foreground = next_foreground;
                background = next_background;
            }
        }
        let recovered = RgbImageF32::new(image.width(), image.height(), foreground)?;
        Ok(image_from_space(&recovered, self.config.working_space))
    }
}

/// Execute one source-compatible blur-fusion pass, retaining the blurred
/// background returned by PhotoRoom for the next pass. This is public so the
/// cross-language fixture can compare pass intermediates, not only the final
/// colour field.
pub fn fast_foreground_pass(
    image: &RgbImageF32,
    alpha: &AlphaMask,
    foreground: &RgbImageF32,
    background: &RgbImageF32,
    kernel_width: u32,
    epsilon: f32,
    edge_policy: BoxEdgePolicy,
) -> Result<(RgbImageF32, RgbImageF32)> {
    ensure!(
        image.dimensions() == alpha.dimensions(),
        "fast pass dimensions differ"
    );
    ensure!(
        foreground.dimensions() == image.dimensions(),
        "fast pass foreground dimensions differ"
    );
    ensure!(
        background.dimensions() == image.dimensions(),
        "fast pass background dimensions differ"
    );
    let (next_f, next_b) = fast_pass(
        image.data(),
        alpha.data(),
        foreground.data(),
        background.data(),
        image.width(),
        image.height(),
        kernel_width,
        epsilon,
        edge_policy,
    )?;
    Ok((
        RgbImageF32::new(image.width(), image.height(), next_f)?,
        RgbImageF32::new(image.width(), image.height(), next_b)?,
    ))
}

/// FBA-provided foreground candidate. It is deliberately strict: absence of
/// the paired FBA channel is an error, never a silent fallback to source RGB.
#[derive(Clone, Copy, Debug, Default)]
pub struct FbaForegroundEstimator;

impl ForegroundEstimator for FbaForegroundEstimator {
    fn estimate(&self, image: &CanonicalImage, matte: &RefinedMatte) -> Result<RgbImageF32> {
        ensure!(
            matte.alpha().dimensions() == image.dimensions(),
            "FBA estimator dimension mismatch"
        );
        let foreground = matte
            .foreground()
            .context("FBA foreground candidate requires RefinedMatte.foreground()")?;
        ensure!(
            foreground.dimensions() == image.dimensions(),
            "FBA foreground dimensions do not match image"
        );
        Ok(foreground.clone())
    }
}

fn image_in_space(image: &RgbImageF32, space: ForegroundWorkingSpace) -> RgbImageF32 {
    if matches!(space, ForegroundWorkingSpace::EncodedSrgb) {
        return image.clone();
    }
    RgbImageF32::new(
        image.width(),
        image.height(),
        image
            .data()
            .iter()
            .map(|pixel| std::array::from_fn(|c| srgb_to_linear(pixel[c])))
            .collect(),
    )
    .expect("finite normalized transfer output")
}

fn image_from_space(image: &RgbImageF32, space: ForegroundWorkingSpace) -> RgbImageF32 {
    if matches!(space, ForegroundWorkingSpace::EncodedSrgb) {
        return image.clone();
    }
    RgbImageF32::new(
        image.width(),
        image.height(),
        image
            .data()
            .iter()
            .map(|pixel| std::array::from_fn(|c| linear_to_srgb(pixel[c])))
            .collect(),
    )
    .expect("finite normalized transfer output")
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fast_pass(
    image: &[[f32; 3]],
    alpha: &[f32],
    foreground: &[[f32; 3]],
    background: &[[f32; 3]],
    width: u32,
    height: u32,
    kernel_width: u32,
    epsilon: f32,
    edge_policy: BoxEdgePolicy,
) -> Result<(Vec<[f32; 3]>, Vec<[f32; 3]>)> {
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "fast epsilon must be finite and positive"
    );
    let expected = checked_pixels(width, height)?;
    ensure!(
        image.len() == expected
            && foreground.len() == expected
            && background.len() == expected
            && alpha.len() == expected,
        "fast estimator input length mismatch"
    );
    let alpha_blur = box_blur(alpha, width, height, kernel_width, edge_policy)?;
    let fa: Vec<[f32; 3]> = foreground
        .iter()
        .zip(alpha)
        .map(|(f, a)| std::array::from_fn(|c| f[c] * *a))
        .collect();
    let b_weighted: Vec<[f32; 3]> = background
        .iter()
        .zip(alpha)
        .map(|(b, a)| std::array::from_fn(|c| b[c] * (1.0 - *a)))
        .collect();
    let fb_blur = box_blur_rgb(&fa, width, height, kernel_width, edge_policy)?;
    let bb_blur = box_blur_rgb(&b_weighted, width, height, kernel_width, edge_policy)?;
    let mut next_f = Vec::with_capacity(expected);
    let mut next_b = Vec::with_capacity(expected);
    for i in 0..expected {
        let a = alpha[i];
        let one_minus_a = 1.0 - a;
        let ab = alpha_blur[i];
        let one_minus_ab = 1.0 - ab;
        let f_den = ab + epsilon;
        let b_den = one_minus_ab + epsilon;
        ensure!(
            f_den.is_finite() && b_den.is_finite(),
            "fast estimator denominator overflow"
        );
        let fb: [f32; 3] = std::array::from_fn(|c| fb_blur[i][c] / f_den);
        let bb: [f32; 3] = std::array::from_fn(|c| bb_blur[i][c] / b_den);
        let nf: [f32; 3] = std::array::from_fn(|c| {
            (fb[c] + a * (image[i][c] - a * fb[c] - one_minus_a * bb[c])).clamp(0.0, 1.0)
        });
        ensure!(
            nf.iter().chain(bb.iter()).all(|value| value.is_finite()),
            "fast estimator produced non-finite colour"
        );
        next_f.push(nf);
        next_b.push(bb);
    }
    Ok((next_f, next_b))
}

fn checked_pixels(width: u32, height: u32) -> Result<usize> {
    ensure!(
        width > 0 && height > 0,
        "box blur dimensions must be positive"
    );
    (width as usize)
        .checked_mul(height as usize)
        .context("box blur dimensions overflow")
}

/// O(width*height) box blur with an OpenCV-compatible kernel width. The
/// default `Reflect101` edge policy matches `cv2.blur(..., (k,k))`, including
/// its even-kernel anchor (`k/2` samples to the left).
pub fn box_blur(
    input: &[f32],
    width: u32,
    height: u32,
    kernel_width: u32,
    edge_policy: BoxEdgePolicy,
) -> Result<Vec<f32>> {
    let expected = checked_pixels(width, height)?;
    ensure!(input.len() == expected, "box blur input length mismatch");
    ensure!(
        input.iter().all(|v| v.is_finite()),
        "box blur input is non-finite"
    );
    ensure!(kernel_width > 0, "box blur kernel width must be positive");
    let w = width as usize;
    let h = height as usize;
    let footprint = u64::from(kernel_width);
    ensure!(
        footprint <= f32::MAX as u64,
        "box blur footprint is too large"
    );
    let mut horizontal = vec![0.0f32; expected];
    for y in 0..h {
        let row = y * w;
        let blurred = blur_line(&input[row..row + w], kernel_width as usize, edge_policy)?;
        horizontal[row..row + w].copy_from_slice(&blurred);
    }
    let mut output = vec![0.0f32; expected];
    for x in 0..w {
        let column: Vec<f32> = (0..h).map(|y| horizontal[y * w + x]).collect();
        let blurred = blur_line(&column, kernel_width as usize, edge_policy)?;
        for y in 0..h {
            output[y * w + x] = blurred[y];
        }
    }
    Ok(output)
}

fn blur_line(input: &[f32], kernel_width: usize, edge_policy: BoxEdgePolicy) -> Result<Vec<f32>> {
    ensure!(!input.is_empty(), "box blur line must be non-empty");
    ensure!(kernel_width > 0, "box blur kernel width must be positive");
    let n = input.len();
    let len = kernel_width as u64;
    let anchor = kernel_width / 2;
    let mut output = vec![0.0f32; n];
    let mut prefix = vec![0.0f32; n + 1];
    for (index, value) in input.iter().enumerate() {
        prefix[index + 1] = checked_sum(prefix[index], *value)?;
    }
    let (period_prefix, period) = if matches!(edge_policy, BoxEdgePolicy::Reflect101) && n > 1 {
        let period = (n - 1)
            .checked_mul(2)
            .context("reflect-101 period overflow")?;
        let mut values = Vec::with_capacity(period);
        for index in 0..period {
            let reflected = if index < n { index } else { period - index };
            values.push(input[reflected]);
        }
        let period_plus_one = period
            .checked_add(1)
            .context("reflect-101 prefix length overflow")?;
        let mut sums = vec![0.0f32; period_plus_one];
        for (index, value) in values.iter().enumerate() {
            sums[index + 1] = checked_sum(sums[index], *value)?;
        }
        (Some(sums), period)
    } else {
        (None, 0)
    };
    for (index, output_value) in output.iter_mut().enumerate() {
        let start = index as i64 - anchor as i64;
        let sum = match edge_policy {
            BoxEdgePolicy::Clamp => clamp_range_sum(input, &prefix, start, len)?,
            BoxEdgePolicy::Reflect101 if n == 1 => checked_product(input[0], len)?,
            BoxEdgePolicy::Reflect101 => reflect101_range_sum(
                period_prefix.as_ref().expect("reflect period prefix"),
                period,
                start,
                len,
            )?,
        };
        *output_value = sum / len as f32;
    }
    Ok(output)
}

fn clamp_range_sum(input: &[f32], prefix: &[f32], start: i64, len: u64) -> Result<f32> {
    let n = input.len() as i64;
    let end = start
        .checked_add(len as i64)
        .context("box blur range endpoint overflow")?;
    let left = (-start).max(0) as u64;
    let right = (end - n).max(0) as u64;
    let inner_start = start.max(0).min(n) as usize;
    let inner_end = end.max(0).min(n) as usize;
    let mut sum = prefix[inner_end] - prefix[inner_start];
    if left > 0 {
        sum = checked_sum(sum, checked_product(input[0], left)?)?;
    }
    if right > 0 {
        sum = checked_sum(sum, checked_product(input[input.len() - 1], right)?)?;
    }
    Ok(sum)
}

fn reflect101_range_sum(prefix: &[f32], period: usize, start: i64, len: u64) -> Result<f32> {
    let period_sum = *prefix.last().expect("non-empty period");
    let cycles = len / period as u64;
    let remainder = (len % period as u64) as usize;
    let mut sum = checked_product(period_sum, cycles)?;
    let start_mod = start.rem_euclid(period as i64) as usize;
    let end = start_mod
        .checked_add(remainder)
        .context("reflect-101 range endpoint overflow")?;
    let remainder_sum = if end <= period {
        prefix[end] - prefix[start_mod]
    } else {
        (prefix[period] - prefix[start_mod]) + prefix[end - period]
    };
    sum = checked_sum(sum, remainder_sum)?;
    Ok(sum)
}

fn checked_product(value: f32, count: u64) -> Result<f32> {
    let result = value * count as f32;
    ensure!(result.is_finite(), "box blur f32 accumulation overflowed");
    Ok(result)
}

fn checked_sum(current: f32, add: f32) -> Result<f32> {
    let sum = current + add;
    ensure!(sum.is_finite(), "box blur f32 accumulation overflowed");
    Ok(sum)
}

/// Apply [`box_blur`] independently to three RGB channels in O(N).
pub fn box_blur_rgb(
    input: &[[f32; 3]],
    width: u32,
    height: u32,
    kernel_width: u32,
    edge_policy: BoxEdgePolicy,
) -> Result<Vec<[f32; 3]>> {
    let expected = checked_pixels(width, height)?;
    ensure!(
        input.len() == expected,
        "RGB box blur input length mismatch"
    );
    let channels: Vec<Vec<f32>> = (0..3)
        .map(|c| input.iter().map(|pixel| pixel[c]).collect())
        .collect();
    let blurred: Vec<Vec<f32>> = channels
        .iter()
        .map(|channel| box_blur(channel, width, height, kernel_width, edge_policy))
        .collect::<Result<_>>()?;
    Ok((0..expected)
        .map(|i| [blurred[0][i], blurred[1][i], blurred[2][i]])
        .collect())
}

/// Space in which foreground/background colours are blended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositeColorSpace {
    EncodedSrgb,
    LinearLight,
}

/// Standard IEC sRGB transfer function, encoded value to linear light.
pub fn srgb_to_linear(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
/// Standard IEC sRGB transfer function, linear light to encoded value.
pub fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Composite straight RGB over a straight RGB background using `I=A*F+(1-A)*B`.
pub fn composite_pixel(
    foreground: [f32; 3],
    background: [f32; 3],
    alpha: f32,
    space: CompositeColorSpace,
) -> [f32; 3] {
    let a = alpha.clamp(0.0, 1.0);
    std::array::from_fn(|i| match space {
        CompositeColorSpace::EncodedSrgb => a * foreground[i] + (1.0 - a) * background[i],
        CompositeColorSpace::LinearLight => linear_to_srgb(
            a * srgb_to_linear(foreground[i]) + (1.0 - a) * srgb_to_linear(background[i]),
        ),
    })
}

/// Composite an image deterministically, preserving dimensions and finite values.
pub fn composite_image(
    foreground: &RgbImageF32,
    alpha: &bgremove_core::AlphaMask,
    background: [f32; 3],
    space: CompositeColorSpace,
) -> Result<RgbImageF32> {
    ensure!(
        foreground.dimensions() == alpha.dimensions(),
        "composite dimensions differ"
    );
    let pixels = foreground
        .data()
        .iter()
        .zip(alpha.data())
        .map(|(f, a)| composite_pixel(*f, background, *a, space))
        .collect();
    RgbImageF32::new(foreground.width(), foreground.height(), pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bgremove_core::AlphaMask;
    #[test]
    fn encoded_compositing_matches_equation() {
        assert_eq!(
            composite_pixel([1.0; 3], [0.0; 3], 0.0, CompositeColorSpace::EncodedSrgb),
            [0.0; 3]
        );
        assert_eq!(
            composite_pixel([1.0; 3], [0.0; 3], 1.0, CompositeColorSpace::EncodedSrgb),
            [1.0; 3]
        );
        let p = composite_pixel([1.0; 3], [0.0; 3], 0.5, CompositeColorSpace::EncodedSrgb);
        assert!(p.iter().all(|v| (*v - 0.5).abs() < f32::EPSILON));
    }
    #[test]
    fn transfer_functions_round_trip() {
        for v in [0.0, 0.001, 0.1, 0.5, 1.0] {
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
        }
    }

    #[test]
    fn encoded_and_linear_compositing_match_reference_after_u8_quantization() {
        let ref_to_linear = |v: f64| {
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        let ref_to_srgb = |v: f64| {
            if v <= 0.0031308 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        };
        for space in [
            CompositeColorSpace::EncodedSrgb,
            CompositeColorSpace::LinearLight,
        ] {
            for alpha in [0.0, 0.25, 0.5, 1.0] {
                let f = [0.91, 0.37, 0.12];
                let b = [0.08, 0.63, 0.94];
                let actual = composite_pixel(f, b, alpha, space);
                let reference: [f32; 3] = match space {
                    CompositeColorSpace::EncodedSrgb => {
                        std::array::from_fn(|i| alpha * f[i] + (1.0 - alpha) * b[i])
                    }
                    CompositeColorSpace::LinearLight => std::array::from_fn(|i| {
                        ref_to_srgb(
                            alpha as f64 * ref_to_linear(f[i] as f64)
                                + (1.0 - alpha as f64) * ref_to_linear(b[i] as f64),
                        ) as f32
                    }),
                };
                for (a, r) in actual.into_iter().zip(reference) {
                    assert!(((a * 255.0).round() - (r * 255.0).round()).abs() <= 1.0);
                }
            }
        }
        assert!(RgbImageF32::new(1, 1, vec![[f32::NAN; 3]]).is_err());
        let image = RgbImageF32::constant(1, 1, [0.0; 3]).unwrap();
        let alpha = bgremove_core::AlphaMask::zeros(2, 1).unwrap();
        assert!(
            composite_image(&image, &alpha, [0.0; 3], CompositeColorSpace::EncodedSrgb).is_err()
        );
    }

    fn reflect101(index: isize, size: usize) -> usize {
        if size == 1 {
            return 0;
        }
        let period = 2 * (size - 1) as isize;
        let value = index.rem_euclid(period);
        if value < size as isize {
            value as usize
        } else {
            (period - value) as usize
        }
    }

    fn slow_box_blur(input: &[f32], width: usize, height: usize, kernel: usize) -> Vec<f32> {
        let mut out = vec![0.0; input.len()];
        let denominator = kernel as f32;
        let anchor = (kernel / 2) as isize;
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for dy in 0..kernel {
                    let yy = reflect101(y as isize + dy as isize - anchor, height);
                    for dx in 0..kernel {
                        let xx = reflect101(x as isize + dx as isize - anchor, width);
                        sum += input[yy * width + xx];
                    }
                }
                out[y * width + x] = sum / denominator.powi(2);
            }
        }
        out
    }

    fn slow_clamp_box_blur(input: &[f32], width: usize, height: usize, kernel: usize) -> Vec<f32> {
        let mut out = vec![0.0; input.len()];
        let denominator = kernel as f32;
        let anchor = (kernel / 2) as isize;
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for dy in 0..kernel {
                    let yy = (y as isize + dy as isize - anchor).clamp(0, height as isize - 1);
                    for dx in 0..kernel {
                        let xx = (x as isize + dx as isize - anchor).clamp(0, width as isize - 1);
                        sum += input[yy as usize * width + xx as usize];
                    }
                }
                out[y * width + x] = sum / denominator.powi(2);
            }
        }
        out
    }

    #[test]
    fn box_blur_matches_slow_reference_for_uniform_impulse_border_and_seeded_random() {
        let width = 5usize;
        let height = 4usize;
        let fixtures = [
            vec![0.25; width * height],
            {
                let mut values = vec![0.0; width * height];
                values[0] = 1.0;
                values
            },
            {
                let mut values = vec![0.0; width * height];
                values[width - 1] = 1.0;
                values
            },
            (0..width * height)
                .map(|i| ((i * 37 + 11) % 101) as f32 / 100.0)
                .collect(),
        ];
        for fixture in fixtures {
            for kernel in [1, 2, 3, 8] {
                let actual = box_blur(
                    &fixture,
                    width as u32,
                    height as u32,
                    kernel,
                    BoxEdgePolicy::Reflect101,
                )
                .unwrap();
                let expected = slow_box_blur(&fixture, width, height, kernel as usize);
                assert!(
                    actual
                        .iter()
                        .zip(expected.iter())
                        .all(|(a, e)| (a - e).abs() < 2e-6),
                    "kernel {kernel}: {actual:?} != {expected:?}"
                );
            }
        }
    }

    #[test]
    fn clamp_box_blur_matches_slow_clamp_reference() {
        let width = 5usize;
        let height = 4usize;
        let fixture: Vec<f32> = (0..width * height)
            .map(|i| ((i * 37 + 11) % 101) as f32 / 100.0)
            .collect();
        for kernel in [1, 2, 3, 8] {
            let actual = box_blur(
                &fixture,
                width as u32,
                height as u32,
                kernel,
                BoxEdgePolicy::Clamp,
            )
            .unwrap();
            let expected = slow_clamp_box_blur(&fixture, width, height, kernel as usize);
            assert!(
                actual
                    .iter()
                    .zip(expected.iter())
                    .all(|(a, e)| (a - e).abs() < 2e-6),
                "kernel {kernel}: {actual:?} != {expected:?}"
            );
        }
    }

    fn synthetic_image_and_matte() -> (CanonicalImage, RefinedMatte) {
        let width = 6;
        let height = 4;
        let foreground = [0.8, 0.2, 0.1];
        let background = [0.1, 0.3, 0.9];
        let alpha: Vec<f32> = (0..width * height)
            .map(|i| match i % 6 {
                0 => 0.0,
                1 => 0.05,
                2 => 0.25,
                3 => 0.5,
                4 => 0.95,
                _ => 1.0,
            })
            .collect();
        let pixels = alpha
            .iter()
            .map(|a| std::array::from_fn(|c| *a * foreground[c] + (1.0 - *a) * background[c]))
            .collect();
        let image = CanonicalImage::new(width, height, pixels).unwrap();
        let matte =
            RefinedMatte::new(AlphaMask::new(width, height, alpha).unwrap(), None, None).unwrap();
        (image, matte)
    }

    #[test]
    fn candidates_are_dimension_safe_deterministic_and_finite() {
        let (image, matte) = synthetic_image_and_matte();
        let original = OriginalRgbEstimator::default()
            .estimate(&image, &matte)
            .unwrap();
        assert_eq!(original, image.rgb().clone());
        let fast = FastForegroundEstimator::new(FastForegroundConfig::default()).unwrap();
        let first = fast.estimate(&image, &matte).unwrap();
        let second = fast.estimate(&image, &matte).unwrap();
        assert_eq!(first, second);
        assert!(first
            .data()
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        let ml = MultilevelForegroundEstimator::new(MultilevelForegroundConfig::default()).unwrap();
        let recovered = ml.estimate(&image, &matte).unwrap();
        assert!(recovered
            .data()
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    }

    #[test]
    fn fast_linear_ablation_is_explicit_and_differs_from_encoded_on_mixed_values() {
        let (image, matte) = synthetic_image_and_matte();
        let encoded = FastForegroundEstimator::new(FastForegroundConfig::default()).unwrap();
        let linear = FastForegroundEstimator::new(FastForegroundConfig {
            working_space: ForegroundWorkingSpace::LinearLight,
            ..Default::default()
        })
        .unwrap();
        let a = encoded.estimate(&image, &matte).unwrap();
        let b = linear.estimate(&image, &matte).unwrap();
        assert!(a.data().iter().flatten().all(|v| v.is_finite()));
        assert!(b.data().iter().flatten().all(|v| v.is_finite()));
        assert!(a
            .data()
            .iter()
            .zip(b.data())
            .flat_map(|(x, y)| x.iter().zip(y))
            .any(|(x, y)| (x - y).abs() > 1e-4));
    }

    #[test]
    fn fba_candidate_requires_model_foreground_and_never_falls_back_to_original() {
        let (image, matte) = synthetic_image_and_matte();
        assert!(FbaForegroundEstimator.estimate(&image, &matte).is_err());
        let alpha = matte.alpha().clone();
        let fba = RgbImageF32::constant(image.width(), image.height(), [0.9, 0.8, 0.7]).unwrap();
        let full = RefinedMatte::new(alpha, Some(fba.clone()), Some(image.rgb().clone())).unwrap();
        assert_eq!(FbaForegroundEstimator.estimate(&image, &full).unwrap(), fba);
    }

    #[test]
    fn invalid_fast_configuration_and_nonfinite_blur_inputs_fail_closed() {
        assert!(FastForegroundEstimator::new(FastForegroundConfig {
            epsilon: 0.0,
            ..Default::default()
        })
        .is_err());
        assert!(FastForegroundEstimator::new(FastForegroundConfig {
            coarse_kernel_width: 0,
            ..Default::default()
        })
        .is_err());
        assert!(FastForegroundEstimator::new(FastForegroundConfig {
            fine_kernel_width: 0,
            ..Default::default()
        })
        .is_err());
        assert!(FastForegroundEstimator::new(FastForegroundConfig {
            coarse_iterations: 0,
            ..Default::default()
        })
        .is_err());
        assert!(FastForegroundEstimator::new(FastForegroundConfig {
            fine_iterations: 65,
            ..Default::default()
        })
        .is_err());
        assert!(FastForegroundEstimator::new(FastForegroundConfig {
            reference_size: 0,
            scale_kernel_widths: true,
            ..Default::default()
        })
        .is_err());
        assert!(FastForegroundEstimator::new(FastForegroundConfig {
            reference_size: 0,
            scale_kernel_widths: false,
            ..Default::default()
        })
        .is_ok());
        assert!(box_blur(&[f32::NAN], 1, 1, 0, BoxEdgePolicy::Clamp).is_err());
        assert!(box_blur(&[0.0], 0, 1, 0, BoxEdgePolicy::Clamp).is_err());
        assert!(box_blur(&[], u32::MAX, u32::MAX, 1, BoxEdgePolicy::Clamp).is_err());
    }

    #[test]
    fn public_fast_pass_rejects_invalid_epsilon() {
        let (image, matte) = synthetic_image_and_matte();
        let rgb = image.rgb().clone();
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(fast_foreground_pass(
                &rgb,
                matte.alpha(),
                &rgb,
                &rgb,
                3,
                epsilon,
                BoxEdgePolicy::Reflect101,
            )
            .is_err());
        }
    }

    #[test]
    fn scaled_kernel_widths_are_reference_half_double_and_tiny_safe() {
        let config = FastForegroundConfig {
            scale_kernel_widths: true,
            reference_size: 1024,
            ..FastForegroundConfig::photoroom_source()
        };
        assert_eq!(config.resolved_kernel_widths(1024, 1024).unwrap(), (90, 6));
        assert_eq!(config.resolved_kernel_widths(512, 512).unwrap(), (45, 3));
        assert_eq!(
            config.resolved_kernel_widths(2048, 2048).unwrap(),
            (180, 12)
        );
        assert_eq!(config.resolved_kernel_widths(1, 1).unwrap(), (1, 1));
    }

    #[test]
    fn non_default_fast_iteration_counts_are_deterministic() {
        let (image, matte) = synthetic_image_and_matte();
        let config = FastForegroundConfig {
            coarse_iterations: 2,
            fine_iterations: 3,
            ..FastForegroundConfig::photoroom_source()
        };
        let estimator = FastForegroundEstimator::new(config).unwrap();
        let first = estimator.estimate(&image, &matte).unwrap();
        let second = estimator.estimate(&image, &matte).unwrap();
        assert_eq!(first, second);
        assert!(first.data().iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn m13_authority_parity() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/m13/reference");
        let read = |name: &str, count: usize| {
            let bytes = std::fs::read(root.join(name)).expect("M13 authority artifact");
            assert_eq!(bytes.len(), count * std::mem::size_of::<f32>());
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let image_flat = read("image.f32le", 6 * 4 * 3);
        let alpha = read("alpha.f32le", 6 * 4);
        let expected_f = read("foreground.f32le", 6 * 4 * 3);
        let expected_b = read("background.f32le", 6 * 4 * 3);
        let image = image_flat
            .chunks_exact(3)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect::<Vec<_>>();
        let (actual_f, actual_b) =
            estimate_foreground_ml(&image, &alpha, 6, 4, 1e-5, 10, 2, 32, 1.0).unwrap();
        let max_abs = |actual: &[[f32; 3]], expected: &[f32]| {
            actual
                .iter()
                .flat_map(|pixel| pixel.iter())
                .zip(expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(max_abs(actual_f.data(), &expected_f) <= 2e-6);
        assert!(max_abs(actual_b.data(), &expected_b) <= 2e-6);
        let (actual_f2, actual_b2) =
            estimate_foreground_ml(&image, &alpha, 6, 4, 1e-5, 10, 2, 32, 1.0).unwrap();
        assert_eq!(actual_f, actual_f2);
        assert_eq!(actual_b, actual_b2);
    }

    #[test]
    fn photoroom_fast_authority_parity_covers_even_kernels_reflect101_and_pass_state() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/m13/fast-reference");
        let read = |path: &std::path::Path, count: usize| {
            let bytes = std::fs::read(path).expect("PhotoRoom authority artifact");
            assert_eq!(bytes.len(), count * 4);
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let max_abs = |actual: &[[f32; 3]], expected: &[f32]| {
            actual
                .iter()
                .flat_map(|pixel| pixel.iter())
                .zip(expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max)
        };
        for (case, (width, height)) in [(0, (5usize, 3usize)), (1, (7, 4)), (2, (2, 6))] {
            let dir = root.join(format!("case-{case}"));
            let image_flat = read(&dir.join("image.f32le"), width * height * 3);
            let alpha_data = read(&dir.join("alpha.f32le"), width * height);
            let input = RgbImageF32::new(
                width as u32,
                height as u32,
                image_flat
                    .chunks_exact(3)
                    .map(|p| [p[0], p[1], p[2]])
                    .collect(),
            )
            .unwrap();
            let alpha = AlphaMask::new(width as u32, height as u32, alpha_data).unwrap();
            let initial = input.clone();
            let (pass1_f, pass1_b) = fast_foreground_pass(
                &input,
                &alpha,
                &initial,
                &initial,
                90,
                1e-5,
                BoxEdgePolicy::Reflect101,
            )
            .unwrap();
            let (pass2_f, pass2_b) = fast_foreground_pass(
                &input,
                &alpha,
                &pass1_f,
                &pass1_b,
                6,
                1e-5,
                BoxEdgePolicy::Reflect101,
            )
            .unwrap();
            let expected_p1f = read(&dir.join("pass1-foreground.f32le"), width * height * 3);
            let expected_p1b = read(
                &dir.join("pass1-blurred-background.f32le"),
                width * height * 3,
            );
            let expected_p2f = read(&dir.join("pass2-foreground.f32le"), width * height * 3);
            let expected_p2b = read(
                &dir.join("pass2-blurred-background.f32le"),
                width * height * 3,
            );
            assert!(
                max_abs(pass1_f.data(), &expected_p1f) <= 3e-6,
                "case {case} pass1 F"
            );
            assert!(
                max_abs(pass1_b.data(), &expected_p1b) <= 3e-6,
                "case {case} pass1 B"
            );
            assert!(
                max_abs(pass2_f.data(), &expected_p2f) <= 3e-6,
                "case {case} pass2 F"
            );
            assert!(
                max_abs(pass2_b.data(), &expected_p2b) <= 3e-6,
                "case {case} pass2 B"
            );
            let source_config = FastForegroundConfig::photoroom_source();
            assert_eq!(
                source_config
                    .resolved_kernel_widths(width as u32, height as u32)
                    .unwrap(),
                (90, 6)
            );
            let matte = RefinedMatte::new(alpha.clone(), None, None).unwrap();
            let final_result = FastForegroundEstimator::new(source_config)
                .unwrap()
                .estimate(
                    &CanonicalImage::new(
                        width as u32,
                        height as u32,
                        image_flat
                            .chunks_exact(3)
                            .map(|p| [p[0], p[1], p[2]])
                            .collect(),
                    )
                    .unwrap(),
                    &matte,
                )
                .unwrap();
            assert!(
                max_abs(final_result.data(), &expected_p2f) <= 3e-6,
                "case {case} final estimator"
            );
        }
    }

    #[test]
    fn fast_source_candidate_handles_alpha_zero_one_and_near_extremes() {
        let width = 4;
        let height = 1;
        let image = CanonicalImage::new(
            width,
            height,
            vec![
                [0.2, 0.4, 0.8],
                [0.8, 0.2, 0.1],
                [0.3, 0.6, 0.2],
                [0.9, 0.1, 0.5],
            ],
        )
        .unwrap();
        for alpha_values in [
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1e-8, 1.0 - 1e-8, 1e-6, 1.0 - 1e-6],
        ] {
            let matte = RefinedMatte::new(
                AlphaMask::new(width, height, alpha_values.clone()).unwrap(),
                None,
                None,
            )
            .unwrap();
            let result = FastForegroundEstimator::new(FastForegroundConfig::photoroom_source())
                .unwrap()
                .estimate(&image, &matte)
                .unwrap();
            assert!(result
                .data()
                .iter()
                .flatten()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
            if alpha_values.iter().all(|a| *a == 1.0) {
                assert!(result
                    .data()
                    .iter()
                    .zip(image.rgb().data())
                    .flat_map(|(a, b)| a.iter().zip(b))
                    .all(|(a, b)| (a - b).abs() < 3e-5));
            }
        }
    }
}
