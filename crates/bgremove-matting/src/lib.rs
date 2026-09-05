//! Matting contracts and no-op implementations for M1.
use anyhow::{ensure, Result};
use bgremove_core::{
    AlphaMask, AlphaRefiner, CanonicalImage, MaskTransform, NoOpAlphaRefiner, NoOpMaskTransform,
    RefinedMatte, Trimap,
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
}
