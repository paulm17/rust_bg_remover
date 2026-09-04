//! Shared transitional M1 input loader for both CLI surfaces.

use anyhow::{Context, Result};
use bgremove_core::CanonicalImage;
use image::{ImageDecoder, ImageReader};
use std::path::Path;

/// Decode one input and apply its EXIF orientation exactly once.
///
/// This helper intentionally only produces the canonical RGB grid. It does
/// not resize, composite, or encode output; those mechanisms start in M2.
pub fn load_canonical(path: &Path) -> Result<CanonicalImage> {
    let reader = ImageReader::open(path)
        .with_context(|| format!("open input image {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("identify input image {}", path.display()))?;
    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("decode input image {}", path.display()))?;
    let orientation = decoder
        .orientation()
        .with_context(|| format!("read EXIF orientation {}", path.display()))?;
    let mut decoded = image::DynamicImage::from_decoder(decoder)
        .with_context(|| format!("decode pixels {}", path.display()))?;
    decoded.apply_orientation(orientation);
    let rgb = decoded.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb
        .pixels()
        .map(|p| {
            [
                f32::from(p[0]) / 255.0,
                f32::from(p[1]) / 255.0,
                f32::from(p[2]) / 255.0,
            ]
        })
        .collect();
    CanonicalImage::new(width, height, pixels)
}
