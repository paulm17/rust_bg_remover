//! Foreground colour contracts. M1 uses original RGB as a deterministic control.
use anyhow::{ensure, Result};
use bgremove_core::{
    CanonicalImage, ForegroundEstimator, NoOpForegroundEstimator, RefinedMatte, RgbImageF32,
};

/// Original-RGB control estimator; no decontamination is performed in M1.
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

pub const M1_COLOR_POLICY: &str = "original-rgb; straight alpha; no M2 compositing";
