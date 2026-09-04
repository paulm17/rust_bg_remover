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
