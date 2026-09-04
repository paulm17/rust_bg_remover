//! Canonical image decoding and deterministic straight-alpha PNG encoding.
//! The crate-level image dependency enables only JPEG, PNG and WebP.

use anyhow::{ensure, Context, Result};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageDecoder, ImageEncoder, ImageReader};
use std::path::Path;

use crate::{AlphaMask, CanonicalImage, Foreground};

/// Decode JPEG, PNG or WebP and apply EXIF orientation exactly once.
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
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut rgb = Vec::with_capacity(width as usize * height as usize);
    let mut alpha = Vec::with_capacity(rgb.capacity());
    for pixel in rgba.pixels() {
        rgb.push([
            f32::from(pixel[0]) / 255.0,
            f32::from(pixel[1]) / 255.0,
            f32::from(pixel[2]) / 255.0,
        ]);
        alpha.push(f32::from(pixel[3]) / 255.0);
    }
    CanonicalImage::new_with_alpha(width, height, rgb, AlphaMask::new(width, height, alpha)?)
}

/// Encode straight-alpha RGBA PNG. RGB is never multiplied by alpha.
pub fn encode_straight_rgba_png(result: &Foreground) -> Result<Vec<u8>> {
    ensure!(
        result.rgb().dimensions() == result.alpha().dimensions(),
        "cutout dimensions differ"
    );
    let mut raw = Vec::with_capacity(result.rgb().len() * 4);
    for (rgb, alpha) in result.rgb().data().iter().zip(result.alpha().data()) {
        raw.extend([
            quantize_u8(rgb[0]),
            quantize_u8(rgb[1]),
            quantize_u8(rgb[2]),
            quantize_u8(*alpha),
        ]);
    }
    let mut out = Vec::new();
    PngEncoder::new(&mut out).write_image(
        &raw,
        result.rgb().width(),
        result.rgb().height(),
        ColorType::Rgba8.into(),
    )?;
    Ok(out)
}

/// Encode a finite alpha mask as an 8-bit grayscale PNG.
pub fn encode_mask_png(mask: &AlphaMask) -> Result<Vec<u8>> {
    let raw: Vec<u8> = mask.data().iter().copied().map(quantize_u8).collect();
    let mut out = Vec::new();
    PngEncoder::new(&mut out).write_image(
        &raw,
        mask.width(),
        mask.height(),
        ColorType::L8.into(),
    )?;
    Ok(out)
}

fn quantize_u8(value: f32) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
