//! Straight-alpha foreground colour and deterministic compositing contracts.
use anyhow::{ensure, Result};
use bgremove_core::{
    CanonicalImage, ForegroundEstimator, NoOpForegroundEstimator, RefinedMatte, RgbImageF32,
};

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
}
