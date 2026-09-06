//! Verified ONNX Runtime adapter. Hash and graph validation happen before a
//! session is created; model/runtime downloads are deliberately disabled.
use anyhow::{bail, ensure, Context, Result};
use bgremove_models::{
    Activation, DimensionSpec, ModelManifest, OutputNormalization, PreprocessingProfile,
    TensorElementType,
};
use ort::{session::Session, tensor::TensorElementType as OrtType, value::Tensor};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    sync::{Arc, Condvar, Mutex, OnceLock},
};

/// The byte resize used by @imgly/background-removal. It deliberately uses
/// corner-aligned coordinates (`x * src/new`) and rounds each interpolated
/// uint8 channel, including the border. This is not image-crate's half-pixel
/// convention and is kept here as a small auditable primitive.
pub fn resize_u8_bilinear_js(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    channels: usize,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>> {
    ensure!(
        src_width > 0 && src_height > 0 && dst_width > 0 && dst_height > 0,
        "resize dimensions must be positive"
    );
    ensure!(channels > 0, "resize channel count must be positive");
    ensure!(
        src.len() == src_width as usize * src_height as usize * channels,
        "resize source length mismatch"
    );
    let mut out = vec![0u8; dst_width as usize * dst_height as usize * channels];
    let sx = src_width as f64 / dst_width as f64;
    let sy = src_height as f64 / dst_height as f64;
    for y in 0..dst_height as usize {
        let fy = y as f64 * sy;
        let y1 = fy.floor() as usize;
        let y2 = (fy.ceil() as usize).min(src_height as usize - 1);
        let dy = fy - y1 as f64;
        for x in 0..dst_width as usize {
            let fx = x as f64 * sx;
            let x1 = fx.floor() as usize;
            let x2 = (fx.ceil() as usize).min(src_width as usize - 1);
            let dx = fx - x1 as f64;
            for c in 0..channels {
                let at = |yy: usize, xx: usize| {
                    src[(yy * src_width as usize + xx) * channels + c] as f64
                };
                let value = (1.0 - dx) * (1.0 - dy) * at(y1, x1)
                    + dx * (1.0 - dy) * at(y1, x2)
                    + (1.0 - dx) * dy * at(y2, x1)
                    + dx * dy * at(y2, x2);
                // JS Math.round is floor(x + 0.5) for these nonnegative values.
                out[(y * dst_width as usize + x) * channels + c] =
                    (value + 0.5).floor().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

/// M4's established image-crate Lanczos3 path. Keep this function stable:
/// M4's reports and contracts intentionally record image-crate behaviour.
pub fn resize_u8_lanczos(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    channels: usize,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>> {
    ensure!(
        channels == 1 || channels == 3,
        "Lanczos resize supports one or three channels"
    );
    ensure!(
        src.len() == src_width as usize * src_height as usize * channels,
        "Lanczos source length mismatch"
    );
    if channels == 1 {
        let image = image::GrayImage::from_raw(src_width, src_height, src.to_vec())
            .ok_or_else(|| anyhow::anyhow!("invalid grayscale resize buffer"))?;
        Ok(image::imageops::resize(
            &image,
            dst_width,
            dst_height,
            image::imageops::FilterType::Lanczos3,
        )
        .into_raw())
    } else {
        let image = image::RgbImage::from_raw(src_width, src_height, src.to_vec())
            .ok_or_else(|| anyhow::anyhow!("invalid RGB resize buffer"))?;
        Ok(image::imageops::resize(
            &image,
            dst_width,
            dst_height,
            image::imageops::FilterType::Lanczos3,
        )
        .into_raw())
    }
}

/// Exact Pillow 10.4 LANCZOS 8-bit path used by rembg's U2-Net sessions.
/// This is deliberately separate from M4's image-crate compatibility helper.
pub fn resize_u8_pillow_lanczos(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    channels: usize,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>> {
    ensure!(
        channels == 1 || channels == 3,
        "Pillow Lanczos resize supports one or three channels"
    );
    ensure!(
        src.len() == src_width as usize * src_height as usize * channels,
        "Pillow Lanczos source length mismatch"
    );
    // Compact port of Pillow 10.4's Resample.c 8-bit path: center=(x+.5)*scale,
    // support=3*max(scale,1), normalized coefficients quantized to 22
    // fractional bits, and an 8-bit clipped intermediate between passes.
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
                let mut xmin = (center - support + 0.5) as i32;
                xmin = xmin.max(0);
                let mut xmax = (center + support + 0.5) as i32;
                xmax = xmax.min(input as i32);
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
                    .collect::<Vec<_>>();
                (xmin as usize, fixed)
            })
            .collect::<Vec<_>>()
    }
    fn clip(sum: i64) -> u8 {
        ((sum >> 22).clamp(0, 255)) as u8
    }
    let hcoeff = coefficients(src_width as usize, dst_width as usize);
    let vcoeff = coefficients(src_height as usize, dst_height as usize);
    let mut horizontal = vec![0u8; dst_width as usize * src_height as usize * channels];
    for y in 0..src_height as usize {
        for (x, (start, weights)) in hcoeff.iter().enumerate() {
            for c in 0..channels {
                let mut sum = PRECISION / 2;
                for (k, weight) in weights.iter().enumerate() {
                    sum += src[(y * src_width as usize + start + k) * channels + c] as i64 * weight;
                }
                horizontal[(y * dst_width as usize + x) * channels + c] = clip(sum);
            }
        }
    }
    let mut out = vec![0u8; dst_width as usize * dst_height as usize * channels];
    for (y, (start, weights)) in vcoeff.iter().enumerate() {
        for x in 0..dst_width as usize {
            for c in 0..channels {
                let mut sum = PRECISION / 2;
                for (k, weight) in weights.iter().enumerate() {
                    sum += horizontal[((start + k) * dst_width as usize + x) * channels + c] as i64
                        * weight;
                }
                out[(y * dst_width as usize + x) * channels + c] = clip(sum);
            }
        }
    }
    Ok(out)
}

/// Pillow 10.4's 8-bit bicubic path, used by CarveKit BASNet's default
/// `Image.resize` call. Coefficients and the clipped intermediate are kept
/// integer-identical to the Lanczos compatibility path above.
pub fn resize_u8_pillow_bicubic(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    channels: usize,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>> {
    ensure!(
        channels == 1 || channels == 3,
        "Pillow bicubic resize supports one or three channels"
    );
    ensure!(
        src.len() == src_width as usize * src_height as usize * channels,
        "Pillow bicubic source length mismatch"
    );
    const PRECISION: i64 = 1 << 22;
    fn cubic(x: f64) -> f64 {
        let x = x.abs();
        if x <= 1.0 {
            1.5 * x * x * x - 2.5 * x * x + 1.0
        } else if x < 2.0 {
            -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0
        } else {
            0.0
        }
    }
    fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
        let scale = input as f64 / output as f64;
        let filterscale = scale.max(1.0);
        let support = 2.0 * filterscale;
        (0..output)
            .map(|xx| {
                let center = (xx as f64 + 0.5) * scale;
                let mut xmin = (center - support + 0.5) as i32;
                xmin = xmin.max(0);
                let mut xmax = (center + support + 0.5) as i32;
                xmax = xmax.min(input as i32);
                let count = (xmax - xmin).max(0) as usize;
                let mut weights = Vec::with_capacity(count);
                let mut sum = 0.0;
                for x in 0..count {
                    let u = (x as f64 + xmin as f64 - center + 0.5) / filterscale;
                    let weight = cubic(u);
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
                    .collect::<Vec<_>>();
                (xmin as usize, fixed)
            })
            .collect()
    }
    fn clip(sum: i64) -> u8 {
        ((sum >> 22).clamp(0, 255)) as u8
    }
    let hcoeff = coefficients(src_width as usize, dst_width as usize);
    let vcoeff = coefficients(src_height as usize, dst_height as usize);
    let mut horizontal = vec![0u8; dst_width as usize * src_height as usize * channels];
    for y in 0..src_height as usize {
        for (x, (start, weights)) in hcoeff.iter().enumerate() {
            for c in 0..channels {
                let mut sum = PRECISION / 2;
                for (k, weight) in weights.iter().enumerate() {
                    sum += src[(y * src_width as usize + start + k) * channels + c] as i64 * weight;
                }
                horizontal[(y * dst_width as usize + x) * channels + c] = clip(sum);
            }
        }
    }
    let mut out = vec![0u8; dst_width as usize * dst_height as usize * channels];
    for (y, (start, weights)) in vcoeff.iter().enumerate() {
        for x in 0..dst_width as usize {
            for c in 0..channels {
                let mut sum = PRECISION / 2;
                for (k, weight) in weights.iter().enumerate() {
                    sum += horizontal[((start + k) * dst_width as usize + x) * channels + c] as i64
                        * weight;
                }
                out[(y * dst_width as usize + x) * channels + c] = clip(sum);
            }
        }
    }
    Ok(out)
}

/// Pillow's 8-bit bilinear resampler, matching the coefficient convention
/// used by CarveKit's TRACER postprocessing (`Image.resize(..., BILINEAR)`).
pub fn resize_u8_pillow_bilinear(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    channels: usize,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>> {
    ensure!(
        channels == 1 || channels == 3,
        "Pillow bilinear resize supports one or three channels"
    );
    ensure!(
        src.len() == src_width as usize * src_height as usize * channels,
        "Pillow bilinear source length mismatch"
    );
    const PRECISION: i64 = 1 << 22;
    fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
        let scale = input as f64 / output as f64;
        let filterscale = scale.max(1.0);
        let support = filterscale;
        (0..output)
            .map(|xx| {
                let center = (xx as f64 + 0.5) * scale;
                let mut xmin = (center - support + 0.5) as i32;
                xmin = xmin.max(0);
                let mut xmax = (center + support + 0.5) as i32;
                xmax = xmax.min(input as i32);
                let count = (xmax - xmin).max(0) as usize;
                let mut weights = Vec::with_capacity(count);
                let mut sum = 0.0;
                for x in 0..count {
                    let u = (x as f64 + xmin as f64 - center + 0.5) / filterscale;
                    let weight = (1.0 - u.abs()).max(0.0);
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
    let hcoeff = coefficients(src_width as usize, dst_width as usize);
    let vcoeff = coefficients(src_height as usize, dst_height as usize);
    let mut horizontal = vec![0u8; dst_width as usize * src_height as usize * channels];
    for y in 0..src_height as usize {
        for (x, (start, weights)) in hcoeff.iter().enumerate() {
            for c in 0..channels {
                let mut sum = PRECISION / 2;
                for (k, weight) in weights.iter().enumerate() {
                    sum += src[(y * src_width as usize + start + k) * channels + c] as i64 * weight;
                }
                horizontal[(y * dst_width as usize + x) * channels + c] = clip(sum);
            }
        }
    }
    let mut out = vec![0u8; dst_width as usize * dst_height as usize * channels];
    for (y, (start, weights)) in vcoeff.iter().enumerate() {
        for x in 0..dst_width as usize {
            for c in 0..channels {
                let mut sum = PRECISION / 2;
                for (k, weight) in weights.iter().enumerate() {
                    sum += horizontal[((start + k) * dst_width as usize + x) * channels + c] as i64
                        * weight;
                }
                out[(y * dst_width as usize + x) * channels + c] = clip(sum);
            }
        }
    }
    Ok(out)
}

/// Torchvision's tensor Resize path used by TRACER after ToTensor: bilinear,
/// half-pixel coordinates, and antialiasing for downsampling. Keeping this in f32 is
/// important; converting back to u8 before resizing introduces a larger,
/// avoidable multi-code discrepancy.
fn resize_f32_torchvision_bilinear(
    src: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<[f32; 3]> {
    fn coefficients(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
        let scale = input as f32 / output as f32;
        let filter_scale = scale.max(1.0);
        let support = filter_scale;
        (0..output)
            .map(|out| {
                let center = (out as f32 + 0.5) * scale - 0.5;
                let first = (center - support).ceil() as i32;
                let last = (center + support).floor() as i32;
                let mut weights = Vec::new();
                for source in first..=last {
                    let distance = ((source as f32 - center) / filter_scale).abs();
                    let weight = (1.0 - distance).max(0.0) / filter_scale;
                    if weight > 0.0 && (0..input as i32).contains(&source) {
                        weights.push((source as usize, weight));
                    }
                }
                let sum = weights.iter().map(|(_, weight)| *weight).sum::<f32>();
                for (_, weight) in &mut weights {
                    *weight /= sum;
                }
                weights
            })
            .collect()
    }
    let x_coefficients = coefficients(src_width as usize, dst_width as usize);
    let y_coefficients = coefficients(src_height as usize, dst_height as usize);
    let mut out = vec![[0.0; 3]; dst_width as usize * dst_height as usize];
    for oy in 0..dst_height as usize {
        for ox in 0..dst_width as usize {
            let mut pixel = [0.0; 3];
            for &(y, wy) in &y_coefficients[oy] {
                for &(x, wx) in &x_coefficients[ox] {
                    for c in 0..3 {
                        pixel[c] += src[y * src_width as usize + x][c] * wy * wx;
                    }
                }
            }
            out[oy * dst_width as usize + ox] = pixel;
        }
    }
    out
}

/// Convert a canonical encoded RGB image to the exact 1024x1024 IMG.LY input
/// contract, retaining the evidence bytes used by the reference.
pub fn isnet_preprocess_rgb(
    image: &bgremove_core::CanonicalImage,
    profile: PreprocessingProfile,
) -> Result<TensorInput> {
    ensure!(
        matches!(
            profile,
            PreprocessingProfile::ImglyIsnet | PreprocessingProfile::RembgDis
        ),
        "profile {profile:?} is not an IS-Net/DIS profile"
    );
    let (w, h) = image.dimensions();
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in image.rgb().data() {
        for value in px {
            // CanonicalImage stores decoded bytes as f32 in [0,1]. Round when
            // recovering the byte so 64/255 and other decoded code values are
            // lossless despite f32 representation.
            bytes.push(((*value).clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    let resized = match profile {
        PreprocessingProfile::ImglyIsnet => resize_u8_bilinear_js(&bytes, w, h, 3, 1024, 1024)?,
        PreprocessingProfile::RembgDis => resize_u8_lanczos(&bytes, w, h, 3, 1024, 1024)?,
        PreprocessingProfile::Generic
        | PreprocessingProfile::RmbgRust
        | PreprocessingProfile::RembgBria => unreachable!(),
    };
    let rembg_max = if profile == PreprocessingProfile::RembgDis {
        resized.iter().copied().max().unwrap_or(0).max(1) as f32
    } else {
        1.0
    };
    let mut values = vec![0.0f32; 3 * 1024 * 1024];
    for i in 0..(1024 * 1024) {
        for c in 0..3 {
            let byte = resized[i * 3 + c] as f32;
            values[c * 1024 * 1024 + i] = match profile {
                PreprocessingProfile::ImglyIsnet => (byte - 128.0) / 256.0,
                PreprocessingProfile::RembgDis => byte / rembg_max - 0.5,
                PreprocessingProfile::Generic
                | PreprocessingProfile::RmbgRust
                | PreprocessingProfile::RembgBria => unreachable!(),
            };
        }
    }
    Ok(TensorInput {
        shape: vec![1, 3, 1024, 1024],
        values,
    })
}

/// Apply IMG.LY's uint8 mask conversion and the same bilinear restoration as
/// its JavaScript implementation. `raw` is a 1024x1024 direct model output.
pub fn restore_isnet_mask(
    raw: &[f32],
    source_width: u32,
    source_height: u32,
) -> Result<bgremove_core::AlphaMask> {
    ensure!(
        raw.len() == 1024 * 1024,
        "IS-Net output must contain 1024x1024 values"
    );
    ensure!(
        raw.iter().all(|v| v.is_finite()),
        "IS-Net output contains NaN/Inf"
    );
    let bytes: Vec<u8> = raw
        .iter()
        .map(|v| (v * 255.0).clamp(0.0, 255.0).floor() as u8)
        .collect();
    let restored = resize_u8_bilinear_js(&bytes, 1024, 1024, 1, source_width, source_height)?;
    bgremove_core::AlphaMask::new(
        source_width,
        source_height,
        restored.into_iter().map(|v| v as f32 / 255.0).collect(),
    )
}

pub fn restore_rembg_dis_mask(
    raw: &[f32],
    source_width: u32,
    source_height: u32,
) -> Result<bgremove_core::AlphaMask> {
    ensure!(
        raw.len() == 1024 * 1024,
        "DIS output must contain 1024x1024 values"
    );
    ensure!(
        raw.iter().all(|v| v.is_finite()),
        "DIS output contains NaN/Inf"
    );
    let (min, max) = raw
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    let bytes: Vec<u8> = if !min.is_finite() || !max.is_finite() {
        return Err(anyhow::anyhow!("DIS output contains non-finite extrema"));
    } else if (max - min).abs() <= f32::EPSILON {
        vec![0; raw.len()]
    } else {
        raw.iter()
            .map(|v| (((v - min) / (max - min)).clamp(0.0, 1.0) * 255.0).floor() as u8)
            .collect()
    };
    let restored = resize_u8_lanczos(&bytes, 1024, 1024, 1, source_width, source_height)?;
    bgremove_core::AlphaMask::new(
        source_width,
        source_height,
        restored.into_iter().map(|v| v as f32 / 255.0).collect(),
    )
}

/// Construct the M4 naive straight-alpha cutout. The RGB is intentionally
/// taken from the canonical source grid rather than from the 1024 model input.
pub fn isnet_straight_cutout(
    image: &bgremove_core::CanonicalImage,
    alpha: bgremove_core::AlphaMask,
) -> Result<bgremove_core::Foreground> {
    ensure!(
        image.dimensions() == alpha.dimensions(),
        "IS-Net cutout dimensions do not match canonical source"
    );
    bgremove_core::Foreground::new(image.rgb().clone(), alpha)
}

/// M4's model-backed segmenter. The pool owns one verified ORT session per
/// worker; this adapter only supplies the model-specific tensor and restores
/// the output to the canonical source grid.
pub struct IsnetSegmenter {
    pool: SessionPool,
    profile: PreprocessingProfile,
}

pub struct IsnetRunEvidence {
    pub tensor: TensorInput,
    pub raw_output: TensorOutput,
    pub restored: bgremove_core::AlphaMask,
}

impl IsnetSegmenter {
    pub fn new(
        manifest: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        profile: PreprocessingProfile,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        ensure!(
            manifest.algorithm_family == "isnet",
            "M4 IS-Net adapter requires algorithm_family=isnet"
        );
        ensure!(
            manifest.preprocessing_profile == profile,
            "manifest preprocessing profile {:?} does not match requested {:?}",
            manifest.preprocessing_profile,
            profile
        );
        ensure!(
            manifest.width == 1024 && manifest.height == 1024,
            "IS-Net manifest must declare 1024x1024"
        );
        ensure!(
            manifest.layout == bgremove_models::ModelLayout::Nchw,
            "IS-Net input must be NCHW"
        );
        ensure!(
            manifest.aspect == bgremove_models::AspectPolicy::Stretch,
            "M4 IS-Net requires exact square stretch geometry"
        );
        ensure!(
            manifest.channel_order == bgremove_models::ChannelOrder::Rgb,
            "M4 IS-Net requires RGB channel order"
        );
        ensure!(
            manifest.activation == Activation::None,
            "M4 IS-Net uses direct model output; activation must be none"
        );
        match profile {
            PreprocessingProfile::ImglyIsnet => {
                ensure!(
                    manifest.resize_filter == bgremove_models::ResizeFilter::Bilinear
                        && manifest.output_normalization == OutputNormalization::Clamp,
                    "IMG.LY profile requires bilinear resize and clamp output"
                );
                ensure!(
                    manifest.scale == 1.0
                        && manifest.mean == [128.0; 3]
                        && manifest.std == [256.0; 3],
                    "IMG.LY normalization contract mismatch"
                );
            }
            PreprocessingProfile::RembgDis => {
                ensure!(
                    manifest.resize_filter == bgremove_models::ResizeFilter::Lanczos3
                        && manifest.output_normalization == OutputNormalization::MinMax,
                    "rembg DIS profile requires Lanczos3 resize and min-max output"
                );
                ensure!(
                    manifest.scale == 1.0 && manifest.mean == [0.5; 3] && manifest.std == [1.0; 3],
                    "rembg DIS normalization metadata mismatch"
                );
            }
            PreprocessingProfile::Generic
            | PreprocessingProfile::RmbgRust
            | PreprocessingProfile::RembgBria => unreachable!(),
        }
        Ok(Self {
            pool: SessionPool::new(
                manifest,
                manifest_path,
                runtime,
                workers,
                requested,
                fallback_allowed,
            )?,
            profile,
        })
    }

    pub fn predict(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<bgremove_core::AlphaMask> {
        Ok(self.predict_with_evidence(image)?.restored)
    }

    pub fn predict_with_evidence(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<IsnetRunEvidence> {
        let input = isnet_preprocess_rgb(image, self.profile)?;
        let mut lease = self.pool.checkout();
        let output = lease.session_mut().run(&input.shape, &input.values)?;
        let direct_output = output.clone();
        let raw_output = match self.profile {
            PreprocessingProfile::ImglyIsnet => {
                apply_output_transform(output, Activation::None, OutputNormalization::Clamp)?
            }
            PreprocessingProfile::RembgDis => {
                apply_output_transform(output, Activation::None, OutputNormalization::None)?
            }
            PreprocessingProfile::Generic
            | PreprocessingProfile::RmbgRust
            | PreprocessingProfile::RembgBria => unreachable!(),
        };
        let raw = match raw_output.shape.as_slice() {
            [1, 1, 1024, 1024] => raw_output.values.clone(),
            [1, 1024, 1024, 1] => raw_output.values.clone(),
            [1024, 1024] => raw_output.values.clone(),
            other => bail!("IS-Net output shape {other:?} is not a single 1024x1024 mask"),
        };
        let restored = match self.profile {
            PreprocessingProfile::ImglyIsnet => {
                restore_isnet_mask(&raw, image.width(), image.height())
            }
            PreprocessingProfile::RembgDis => {
                restore_rembg_dis_mask(&raw, image.width(), image.height())
            }
            PreprocessingProfile::Generic
            | PreprocessingProfile::RmbgRust
            | PreprocessingProfile::RembgBria => unreachable!(),
        }?;
        Ok(IsnetRunEvidence {
            tensor: input,
            raw_output: direct_output,
            restored,
        })
    }

    pub fn provider(&self) -> ProviderReport {
        self.pool.provider_report()
    }
    pub fn pool_size(&self) -> usize {
        self.pool.size()
    }
}

impl bgremove_core::Segmenter for IsnetSegmenter {
    fn predict(
        &mut self,
        image: &bgremove_core::CanonicalImage,
        _prompt: Option<&bgremove_core::Prompt>,
    ) -> Result<bgremove_core::AlphaMask> {
        IsnetSegmenter::predict(self, image)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RequestedProvider {
    Cpu,
    Coreml,
    Cuda,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderReport {
    pub requested: RequestedProvider,
    pub attempted_chain: Vec<String>,
    pub active: String,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub fallback_reason: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TensorOutput {
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInput {
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInspection {
    pub manifest_id: String,
    pub model_path: String,
    pub opset: u32,
    pub input_name: String,
    pub input_type: String,
    pub input_shape: Vec<i64>,
    pub input_symbols: Vec<String>,
    pub output_name: String,
    pub output_type: String,
    pub output_shape: Vec<i64>,
    pub output_symbols: Vec<String>,
    pub provider: ProviderReport,
}

static INIT: OnceLock<Result<(), String>> = OnceLock::new();
pub fn initialize(runtime: &Path) -> Result<()> {
    INIT.get_or_init(|| {
        ort::init_from(runtime)
            .map_err(|e| e.to_string())
            .and_then(|b| {
                if b.commit() {
                    Ok(())
                } else {
                    Err("ONNX Runtime environment was not committed".into())
                }
            })
    })
    .clone()
    .map_err(anyhow::Error::msg)
}
pub fn provider_report(
    requested: RequestedProvider,
    fallback_allowed: bool,
) -> Result<ProviderReport> {
    let name = match requested {
        RequestedProvider::Cpu => "CPUExecutionProvider",
        RequestedProvider::Coreml => "CoreMLExecutionProvider",
        RequestedProvider::Cuda => "CUDAExecutionProvider",
    };
    if requested == RequestedProvider::Cpu {
        return Ok(ProviderReport {
            requested,
            attempted_chain: vec![name.into()],
            active: name.into(),
            fallback_allowed,
            fallback_used: false,
            fallback_reason: None,
        });
    }
    let available = match requested {
        RequestedProvider::Coreml => cfg!(all(feature = "coreml", target_vendor = "apple")),
        RequestedProvider::Cuda => cfg!(feature = "cuda"),
        RequestedProvider::Cpu => true,
    };
    if available {
        bail!(
            "requested provider {name} is feature-enabled; use VerifiedSession::open to perform the real provider attempt"
        );
    }
    let reason = "provider is unavailable or feature-disabled and was not attempted";
    if !fallback_allowed {
        bail!("requested provider {name} unavailable: {reason}; strict mode forbids CPU fallback")
    }
    eprintln!("warning: {name} unavailable/unconfirmed; falling back to CPU: {reason}");
    Ok(ProviderReport {
        requested,
        attempted_chain: vec!["CPUExecutionProvider".into()],
        active: "CPUExecutionProvider".into(),
        fallback_allowed,
        fallback_used: true,
        fallback_reason: Some(reason.into()),
    })
}
fn dim_matches(spec: &DimensionSpec, actual: i64) -> bool {
    match spec {
        DimensionSpec::Static(n) => u64::try_from(actual).ok() == Some(*n),
        DimensionSpec::Dynamic(_) => actual == -1 || actual > 0,
    }
}
fn provider_name(provider: RequestedProvider) -> &'static str {
    match provider {
        RequestedProvider::Cpu => "CPUExecutionProvider",
        RequestedProvider::Coreml => "CoreMLExecutionProvider",
        RequestedProvider::Cuda => "CUDAExecutionProvider",
    }
}
fn provider_feature_enabled(provider: RequestedProvider) -> bool {
    match provider {
        RequestedProvider::Cpu => true,
        RequestedProvider::Coreml => cfg!(feature = "coreml"),
        RequestedProvider::Cuda => cfg!(feature = "cuda"),
    }
}
fn cpu_session(path: &Path) -> Result<Session> {
    Ok(Session::builder()?.commit_from_file(path)?)
}
fn requested_session(path: &Path, provider: RequestedProvider) -> Result<Session> {
    let builder = Session::builder()?.with_config_entry("session.disable_cpu_ep_fallback", "1")?;
    match provider {
        RequestedProvider::Cpu => Ok(builder.commit_from_file(path)?),
        RequestedProvider::Coreml => {
            #[cfg(feature = "coreml")]
            {
                Ok(builder
                    .with_execution_providers([ort::ep::CoreML::default()
                        .build()
                        .error_on_failure()])?
                    .commit_from_file(path)?)
            }
            #[cfg(not(feature = "coreml"))]
            {
                let _ = builder;
                bail!("CoreMLExecutionProvider feature is disabled; provider was not attempted")
            }
        }
        RequestedProvider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(builder
                    .with_execution_providers([ort::ep::CUDA::default()
                        .build()
                        .error_on_failure()])?
                    .commit_from_file(path)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = builder;
                bail!("CUDAExecutionProvider feature is disabled; provider was not attempted")
            }
        }
    }
}
fn ort_type(t: TensorElementType) -> OrtType {
    match t {
        TensorElementType::F32 => OrtType::Float32,
        TensorElementType::F16 => OrtType::Float16,
        TensorElementType::I32 => OrtType::Int32,
        TensorElementType::I64 => OrtType::Int64,
    }
}
fn checked_numel(shape: &[i64], label: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, dim| {
        ensure!(
            *dim > 0,
            "{label} shape contains non-positive dimension: {shape:?}"
        );
        let dimension = usize::try_from(*dim).map_err(|_| {
            anyhow::anyhow!("{label} shape dimension does not fit usize: {shape:?}")
        })?;
        count.checked_mul(dimension).ok_or_else(|| {
            anyhow::anyhow!("{label} shape element count overflows usize: {shape:?}")
        })
    })
}
pub fn validate_output_contract(
    model_id: &str,
    output_name: &str,
    expected: &[i64],
    actual: &[i64],
    values: usize,
) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "model {model_id} output {output_name} rank mismatch: expected {}, actual {}",
        expected.len(),
        actual.len()
    );
    for (want, got) in expected.iter().zip(actual) {
        ensure!(*got > 0 && (*want < 0 || want == got), "model {model_id} output {output_name} shape mismatch: expected {expected:?}, actual {actual:?}");
    }
    let expected_values = checked_numel(actual, "output")?;
    ensure!(values == expected_values, "model {model_id} output {output_name} shape/value mismatch: shape {actual:?} implies {expected_values}, actual {values}");
    Ok(())
}
fn read_varint(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    while *pos < bytes.len() && shift < 64 {
        let b = bytes[*pos];
        *pos += 1;
        value |= ((b & 127) as u64) << shift;
        if b & 128 == 0 {
            return Some(value);
        }
        shift += 7;
    }
    None
}
fn declared_opset(bytes: &[u8]) -> Option<u32> {
    let mut p = 0;
    let mut default = None;
    while p < bytes.len() {
        let tag = read_varint(bytes, &mut p)?;
        let wire = tag & 7;
        if wire == 2 {
            let len = read_varint(bytes, &mut p)? as usize;
            let end = p.checked_add(len)?;
            if end > bytes.len() {
                return None;
            }
            if tag >> 3 == 8 {
                let mut q = p;
                let mut domain = String::new();
                let mut version = None;
                while q < end {
                    let t = read_varint(bytes, &mut q)?;
                    if t >> 3 == 1 && t & 7 == 2 {
                        let l = read_varint(bytes, &mut q)? as usize;
                        let finish = q.checked_add(l)?;
                        domain = std::str::from_utf8(bytes.get(q..finish)?).ok()?.to_owned();
                        q = finish;
                    } else if t >> 3 == 2 && t & 7 == 0 {
                        version = Some(read_varint(bytes, &mut q)? as u32);
                    } else if t & 7 == 2 {
                        let l = read_varint(bytes, &mut q)? as usize;
                        q = q.checked_add(l)?;
                        if q > end {
                            return None;
                        }
                    } else if t & 7 == 0 {
                        let _ = read_varint(bytes, &mut q)?;
                    } else if t & 7 == 1 {
                        q = q.checked_add(8)?;
                    } else if t & 7 == 5 {
                        q = q.checked_add(4)?;
                    } else {
                        return None;
                    }
                }
                let version = version?;
                if domain.is_empty() || domain == "ai.onnx" {
                    default = Some(version);
                }
            }
            p = end;
        } else if wire == 0 {
            let _ = read_varint(bytes, &mut p)?;
        } else {
            return None;
        }
    }
    default
}
fn inspect_session(
    session: &Session,
    m: &ModelManifest,
    path: &Path,
    provider: ProviderReport,
) -> Result<ModelInspection> {
    let actual_inputs = session
        .inputs()
        .iter()
        .map(|x| x.name())
        .collect::<Vec<_>>();
    let input = session
        .inputs()
        .iter()
        .find(|x| x.name() == m.input_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "model {} input name mismatch: expected {}, actual {:?}",
                m.id,
                m.input_name,
                actual_inputs
            )
        })?;
    let actual_outputs = session
        .outputs()
        .iter()
        .map(|x| x.name())
        .collect::<Vec<_>>();
    let output = session
        .outputs()
        .iter()
        .find(|x| x.name() == m.output_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "model {} output name mismatch: expected {}, actual {:?}",
                m.id,
                m.output_name,
                actual_outputs
            )
        })?;
    let (itype, ishape, isymbols) = match input.dtype() {
        ort::value::ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } => (
            *ty,
            shape.to_vec(),
            dimension_symbols.iter().cloned().collect(),
        ),
        other => bail!(
            "model {} input {} expected tensor, actual {}",
            m.id,
            m.input_name,
            other
        ),
    };
    let (otype, oshape, osymbols) = match output.dtype() {
        ort::value::ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } => (
            *ty,
            shape.to_vec(),
            dimension_symbols.iter().cloned().collect(),
        ),
        other => bail!(
            "model {} output {} expected tensor, actual {}",
            m.id,
            m.output_name,
            other
        ),
    };
    if let Some(t) = m.input_type {
        ensure!(
            itype == ort_type(t),
            "model {} input {} element type mismatch: expected {}, actual {}",
            m.id,
            m.input_name,
            ort_type(t),
            itype
        );
    }
    if let Some(t) = m.output_type {
        ensure!(
            otype == ort_type(t),
            "model {} output {} element type mismatch: expected {}, actual {}",
            m.id,
            m.output_name,
            ort_type(t),
            otype
        );
    }
    if !m.input_shape.is_empty() {
        ensure!(
            m.input_shape.len() == ishape.len(),
            "model {} input {} rank mismatch: expected {}, actual {}",
            m.id,
            m.input_name,
            m.input_shape.len(),
            ishape.len()
        );
        for (s, a) in m.input_shape.iter().zip(&ishape) {
            ensure!(
                dim_matches(s, *a),
                "model {} input {} shape mismatch: expected {:?}, actual {:?}",
                m.id,
                m.input_name,
                m.input_shape,
                ishape
            );
        }
    }
    if !m.output_shape.is_empty() {
        ensure!(
            m.output_shape.len() == oshape.len(),
            "model {} output {} rank mismatch: expected {}, actual {}",
            m.id,
            m.output_name,
            m.output_shape.len(),
            oshape.len()
        );
        for (s, a) in m.output_shape.iter().zip(&oshape) {
            ensure!(
                dim_matches(s, *a),
                "model {} output {} shape mismatch: expected {:?}, actual {:?}",
                m.id,
                m.output_name,
                m.output_shape,
                oshape
            );
        }
    }
    Ok(ModelInspection {
        manifest_id: m.id.clone(),
        model_path: path.display().to_string(),
        opset: m.opset,
        input_name: m.input_name.clone(),
        input_type: itype.to_string(),
        input_shape: ishape,
        input_symbols: isymbols,
        output_name: m.output_name.clone(),
        output_type: otype.to_string(),
        output_shape: oshape,
        output_symbols: osymbols,
        provider,
    })
}

/// Convert canonical encoded RGB into a manifest-shaped f32 tensor. Scale is
/// applied before mean/std; layout and channel order are explicit.
pub fn preprocess_rgb(
    image: &bgremove_core::CanonicalImage,
    m: &ModelManifest,
) -> Result<TensorInput> {
    let (w, h) = image.dimensions();
    let channels = 3i64;
    let mut values = Vec::with_capacity((w * h * 3) as usize);
    let mut channel_data = (0..3)
        .map(|_| Vec::with_capacity((w * h) as usize))
        .collect::<Vec<_>>();
    for px in image.rgb().data() {
        let rgb = match m.channel_order {
            bgremove_models::ChannelOrder::Rgb => *px,
            bgremove_models::ChannelOrder::Bgr => [px[2], px[1], px[0]],
        };
        for c in 0..3 {
            let v = (rgb[c] * m.scale - m.mean[c]) / m.std[c];
            ensure!(
                v.is_finite(),
                "model {} preprocessing produced NaN/Inf",
                m.id
            );
            channel_data[c].push(v);
        }
    }
    let tensor = match m.layout {
        bgremove_models::ModelLayout::Nchw => {
            values.extend(channel_data.into_iter().flatten());
            TensorInput {
                shape: vec![1, channels, h as i64, w as i64],
                values,
            }
        }
        bgremove_models::ModelLayout::Nhwc => {
            for (i, _) in channel_data[0].iter().enumerate().take((w * h) as usize) {
                for channel in channel_data.iter().take(3) {
                    values.push(channel[i]);
                }
            }
            TensorInput {
                shape: vec![1, h as i64, w as i64, channels],
                values,
            }
        }
    };
    if !m.input_shape.is_empty() {
        ensure!(
            m.input_shape.len() == tensor.shape.len(),
            "model {} preprocessing rank mismatch: expected {}, actual {}",
            m.id,
            m.input_shape.len(),
            tensor.shape.len()
        );
        for (expected, actual) in m.input_shape.iter().zip(&tensor.shape) {
            ensure!(
                dim_matches(expected, *actual),
                "model {} preprocessing shape mismatch: expected {:?}, actual {:?}",
                m.id,
                m.input_shape,
                tensor.shape
            );
        }
    }
    Ok(tensor)
}

impl TensorOutput {
    /// Map common single-channel output layouts to the core AlphaMask.
    pub fn to_alpha_mask(&self) -> Result<bgremove_core::AlphaMask> {
        ensure!(
            self.values.iter().all(|x| x.is_finite()),
            "output tensor contains NaN/Inf"
        );
        let (w, h) = match self.shape.as_slice() {
            [1, 1, h, w] => (*w, *h),
            [1, h, w, 1] => (*w, *h),
            [h, w] => (*w, *h),
            [1, h, w] => (*w, *h),
            other => bail!("output shape {:?} cannot map to AlphaMask; expected [1,1,H,W], [1,H,W,1], [H,W], or [1,H,W]", other),
        };
        let width =
            u32::try_from(w).map_err(|_| anyhow::anyhow!("output width {w} does not fit u32"))?;
        let height =
            u32::try_from(h).map_err(|_| anyhow::anyhow!("output height {h} does not fit u32"))?;
        let expected = checked_numel(&[h, w], "alpha output")?;
        ensure!(
            self.values.len() == expected,
            "output shape {:?} has {} values, expected {}",
            self.shape,
            self.values.len(),
            expected
        );
        bgremove_core::AlphaMask::new(
            width,
            height,
            self.values.iter().map(|v| v.clamp(0.0, 1.0)).collect(),
        )
    }
}

pub struct VerifiedSession {
    session: Session,
    pub inspection: ModelInspection,
    manifest: ModelManifest,
    run_count: u64,
}
impl VerifiedSession {
    pub fn open(
        m: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        let path = m
            .verify_model_hash(manifest_path)
            .context("verify model hash before session creation")?;
        let model_bytes = std::fs::read(&path)?;
        let actual_opset = declared_opset(&model_bytes)
            .ok_or_else(|| anyhow::anyhow!("model {} opset metadata is missing", m.id))?;
        ensure!(
            actual_opset == m.opset,
            "model {} opset mismatch: expected {}, actual {}",
            m.id,
            m.opset,
            actual_opset
        );
        initialize(runtime)?;
        let (session, provider) = if requested == RequestedProvider::Cpu {
            (
                cpu_session(&path)
                    .with_context(|| format!("create CPU ONNX session for {}", m.id))?,
                ProviderReport {
                    requested,
                    attempted_chain: vec![provider_name(requested).into()],
                    active: provider_name(requested).into(),
                    fallback_allowed,
                    fallback_used: false,
                    fallback_reason: None,
                },
            )
        } else {
            if !provider_feature_enabled(requested) {
                let reason = format!(
                    "{} feature is disabled; provider was not attempted",
                    provider_name(requested)
                );
                if !fallback_allowed {
                    bail!(
                        "requested provider {} unavailable: {}; strict mode forbids CPU fallback",
                        provider_name(requested),
                        reason
                    );
                }
                eprintln!("warning: {reason}; falling back to CPU");
                let session = cpu_session(&path)
                    .with_context(|| format!("create CPU fallback session for {}", m.id))?;
                (
                    session,
                    ProviderReport {
                        requested,
                        attempted_chain: vec!["CPUExecutionProvider".into()],
                        active: "CPUExecutionProvider".into(),
                        fallback_allowed,
                        fallback_used: true,
                        fallback_reason: Some(reason),
                    },
                )
            } else {
                match requested_session(&path, requested) {
                Ok(session) => (session, ProviderReport { requested, attempted_chain: vec![provider_name(requested).into()], active: provider_name(requested).into(), fallback_allowed, fallback_used: false, fallback_reason: None }),
                Err(error) if !fallback_allowed => bail!("requested provider {} registration/session creation failed: {}; strict mode forbids CPU fallback", provider_name(requested), error),
                Err(error) => {
                    let reason = format!("{} registration/session creation failed: {error}", provider_name(requested));
                    eprintln!("warning: {reason}; falling back to CPU");
                    let session = cpu_session(&path).with_context(|| format!("create CPU fallback session for {}", m.id))?;
                    (session, ProviderReport { requested, attempted_chain: vec![provider_name(requested).into(), "CPUExecutionProvider".into()], active: "CPUExecutionProvider".into(), fallback_allowed, fallback_used: true, fallback_reason: Some(reason) })
                }
            }
            }
        };
        let inspection = inspect_session(&session, m, &path, provider)?;
        if let Some(index) = m.output_index {
            ensure!(
                session.outputs().get(index).map(|x| x.name()) == Some(m.output_name.as_str()),
                "model {} output index mismatch: expected index {} to name {}, actual {:?}",
                m.id,
                index,
                m.output_name,
                session
                    .outputs()
                    .iter()
                    .map(|x| x.name())
                    .collect::<Vec<_>>()
            );
        }
        Ok(Self {
            session,
            inspection,
            manifest: m.clone(),
            run_count: 0,
        })
    }
    pub fn run(&mut self, shape: &[i64], values: &[f32]) -> Result<TensorOutput> {
        ensure!(
            shape.len() == self.inspection.input_shape.len(),
            "model {} input rank mismatch: expected {}, actual {}",
            self.manifest.id,
            self.inspection.input_shape.len(),
            shape.len()
        );
        for (actual, expected) in shape.iter().zip(&self.inspection.input_shape) {
            ensure!(
                *actual > 0 && (*expected < 0 || actual == expected),
                "model {} runtime input shape mismatch: expected {:?}, actual {:?}",
                self.manifest.id,
                self.inspection.input_shape,
                shape
            );
        }
        let expected_values = checked_numel(shape, "input")?;
        ensure!(
            values.len() == expected_values,
            "model {} input value count mismatch: expected {}, actual {}",
            self.manifest.id,
            expected_values,
            values.len()
        );
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "model {} input contains NaN/Inf",
            self.manifest.id
        );
        let input = Tensor::from_array((shape.to_vec(), values.to_vec()))?;
        let outputs = self
            .session
            .run(ort::inputs! { self.manifest.input_name.as_str() => input })?;
        self.run_count += 1;
        let out = outputs.get(&self.manifest.output_name).ok_or_else(|| {
            anyhow::anyhow!(
                "model {} declared output {} not returned",
                self.manifest.id,
                self.manifest.output_name
            )
        })?;
        let (shape, data) = out.try_extract_tensor::<f32>().with_context(|| {
            format!(
                "model {} output {} must be f32",
                self.manifest.id, self.manifest.output_name
            )
        })?;
        ensure!(
            data.iter().all(|x| x.is_finite()),
            "model {} output {} contains NaN/Inf",
            self.manifest.id,
            self.manifest.output_name
        );
        validate_output_contract(
            &self.manifest.id,
            &self.manifest.output_name,
            &self.inspection.output_shape,
            shape,
            data.len(),
        )?;
        Ok(TensorOutput {
            shape: shape.to_vec(),
            values: data.to_vec(),
        })
    }
    pub fn run_count(&self) -> u64 {
        self.run_count
    }
}
struct PoolState {
    available: Vec<VerifiedSession>,
}
pub struct SessionPool {
    state: Arc<(Mutex<PoolState>, Condvar)>,
    size: usize,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
}
pub struct SessionLease {
    state: Arc<(Mutex<PoolState>, Condvar)>,
    session: Option<VerifiedSession>,
    active: Arc<AtomicUsize>,
}
impl SessionPool {
    pub fn new(
        m: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        ensure!(workers > 0, "workers must be greater than zero");
        let mut available = Vec::with_capacity(workers);
        for _ in 0..workers {
            available.push(VerifiedSession::open(
                m,
                manifest_path,
                runtime,
                requested,
                fallback_allowed,
            )?);
        }
        Ok(Self {
            state: Arc::new((Mutex::new(PoolState { available }), Condvar::new())),
            size: workers,
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
        })
    }
    pub fn checkout(&self) -> SessionLease {
        let (lock, cv) = &*self.state;
        let mut state = lock.lock().unwrap();
        while state.available.is_empty() {
            state = cv.wait(state).unwrap();
        }
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        SessionLease {
            state: Arc::clone(&self.state),
            session: Some(state.available.pop().unwrap()),
            active: Arc::clone(&self.active),
        }
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }
    pub fn provider_report(&self) -> ProviderReport {
        let (lock, _) = &*self.state;
        let state = lock.lock().unwrap();
        state
            .available
            .first()
            .map(|s| s.inspection.provider.clone())
            .unwrap_or_else(|| ProviderReport {
                requested: RequestedProvider::Cpu,
                attempted_chain: vec![],
                active: "unknown".into(),
                fallback_allowed: false,
                fallback_used: false,
                fallback_reason: Some("all sessions are checked out".into()),
            })
    }
}
impl SessionLease {
    pub fn session_mut(&mut self) -> &mut VerifiedSession {
        self.session.as_mut().unwrap()
    }
}
impl Drop for SessionLease {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
        if let Some(s) = self.session.take() {
            let (lock, cv) = &*self.state;
            lock.lock().unwrap().available.push(s);
            cv.notify_one();
        }
    }
}

pub fn apply_output_transform(
    mut t: TensorOutput,
    activation: Activation,
    normalization: OutputNormalization,
) -> Result<TensorOutput> {
    ensure!(
        t.values.iter().all(|x| x.is_finite()),
        "output tensor contains NaN/Inf"
    );
    if activation == Activation::Sigmoid {
        for x in &mut t.values {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
    }
    match normalization {
        OutputNormalization::None => {}
        OutputNormalization::Clamp => t.values.iter_mut().for_each(|x| *x = x.clamp(0.0, 1.0)),
        OutputNormalization::MinMax => {
            let (min, max) = t
                .values
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| {
                    (a.min(*x), b.max(*x))
                });
            if max == min {
                t.values.fill(0.0);
            } else {
                for x in &mut t.values {
                    *x = ((*x - min) / (max - min)).clamp(0.0, 1.0);
                }
            }
        }
    }
    ensure!(
        t.values.iter().all(|x| x.is_finite()),
        "transformed output tensor contains NaN/Inf"
    );
    Ok(t)
}

/// Exact shared rembg U2-Net preprocessing contract.  `BaseSession.normalize`
/// first converts to RGB, resizes with Pillow LANCZOS, divides by the global
/// resized-image maximum (with an epsilon guard), applies ImageNet mean/std,
/// and finally transposes HWC to NCHW.
pub fn u2net_preprocess_rgb(
    image: &bgremove_core::CanonicalImage,
    width: u32,
    height: u32,
) -> Result<TensorInput> {
    ensure!(
        width > 0 && height > 0,
        "U2-Net input dimensions must be positive"
    );
    let (w, h) = image.dimensions();
    let mut bytes = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for px in image.rgb().data() {
        for value in px {
            bytes.push((value.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    let resized = resize_u8_pillow_lanczos(&bytes, w, h, 3, width, height)?;
    let max_value = resized.iter().copied().max().unwrap_or(0) as f32;
    let denominator = max_value.max(1e-6);
    let means = [0.485f32, 0.456, 0.406];
    let stds = [0.229f32, 0.224, 0.225];
    let plane = (width as usize) * (height as usize);
    let mut values = vec![0.0; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            let normalized = resized[i * 3 + c] as f32 / denominator;
            values[c * plane + i] = (normalized - means[c]) / stds[c];
        }
    }
    Ok(TensorInput {
        shape: vec![1, 3, height as i64, width as i64],
        values,
    })
}

/// Exact rembg BiRefNet preprocessing contract at the pinned source
/// revision.  The source image is converted to RGB, resized with Pillow's
/// LANCZOS kernel to a stretched 1024 square, divided by the global maximum
/// of the resized uint8 image (with the same epsilon guard as Python), then
/// ImageNet-normalized and transposed to NCHW.
pub fn birefnet_preprocess_rgb(image: &bgremove_core::CanonicalImage) -> Result<TensorInput> {
    let (source_w, source_h) = image.dimensions();
    ensure!(
        source_w > 0 && source_h > 0,
        "BiRefNet source dimensions must be positive"
    );
    let mut bytes = Vec::with_capacity(source_w as usize * source_h as usize * 3);
    for pixel in image.rgb().data() {
        for value in pixel {
            bytes.push((value.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    let resized = resize_u8_pillow_lanczos(&bytes, source_w, source_h, 3, 1024, 1024)?;
    // np.max(im_ary) is the maximum over all RGB channels.  Keeping this in
    // byte space avoids an unnecessary rounding change from byte/255/float.
    let denominator = (resized.iter().copied().max().unwrap_or(0) as f32).max(1.0);
    let plane = 1024usize * 1024usize;
    let mut values = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            let normalized = resized[i * 3 + c] as f32 / denominator;
            let value = (normalized - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c];
            ensure!(value.is_finite(), "BiRefNet preprocessing produced NaN/Inf");
            values[c * plane + i] = value;
        }
    }
    Ok(TensorInput {
        shape: vec![1, 3, 1024, 1024],
        values,
    })
}

/// Restore a BiRefNet probability mask using rembg's uint8 floor conversion
/// followed by Pillow LANCZOS.  The conversion is deliberately explicit so a
/// clamp profile and rembg's per-image min/max profile share identical
/// downstream geometry.
pub fn restore_birefnet_mask(
    values: &[f32],
    source_width: u32,
    source_height: u32,
) -> Result<bgremove_core::AlphaMask> {
    ensure!(
        values.len() == 1024 * 1024,
        "BiRefNet output must contain 1024x1024 values"
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "BiRefNet output contains NaN/Inf"
    );
    let bytes = values
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0) as u8)
        .collect::<Vec<_>>();
    let restored = resize_u8_pillow_lanczos(&bytes, 1024, 1024, 1, source_width, source_height)?;
    bgremove_core::AlphaMask::new(
        source_width,
        source_height,
        restored
            .into_iter()
            .map(|value| value as f32 / 255.0)
            .collect(),
    )
}

fn first_output_mask(
    output: &TensorOutput,
    width: u32,
    height: u32,
    model_id: &str,
) -> Result<Vec<f32>> {
    let expected = (width as usize) * (height as usize);
    let values = match output.shape.as_slice() {
        [1, 1, h, w] if *h == height as i64 && *w == width as i64 => &output.values,
        [1, h, w] if *h == height as i64 && *w == width as i64 => &output.values,
        [h, w] if *h == height as i64 && *w == width as i64 => &output.values,
        other => {
            bail!("model {model_id} first output shape {other:?} is not [1,1,{height},{width}]")
        }
    };
    ensure!(
        values.len() == expected,
        "model {model_id} first output value count mismatch"
    );
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "model {model_id} first output contains NaN/Inf"
    );
    Ok(values.to_vec())
}

#[cfg(feature = "bria")]
fn first_output_mask_channel0(
    output: &TensorOutput,
    width: u32,
    height: u32,
    model_id: &str,
) -> Result<Vec<f32>> {
    let expected = width as usize * height as usize;
    let values = match output.shape.as_slice() {
        [1, channels, h, w] if *channels > 0 && *h == height as i64 && *w == width as i64 => {
            ensure!(
                output.values.len() == expected * *channels as usize,
                "model {model_id} output value count mismatch"
            );
            &output.values[..expected]
        }
        _ => return first_output_mask(output, width, height, model_id),
    };
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "model {model_id} first output contains NaN/Inf"
    );
    Ok(values.to_vec())
}

/// Restore a rembg U2-Net mask: direct first-output values are safely
/// normalized, quantized to uint8 as PIL does, and then resized by Lanczos to
/// the exact canonical source dimensions.
pub fn restore_u2net_mask(
    raw: &[f32],
    source_width: u32,
    source_height: u32,
) -> Result<bgremove_core::AlphaMask> {
    restore_u2net_mask_values(raw, true, source_width, source_height)
}

fn restore_u2net_mask_values(
    raw: &[f32],
    per_image_minmax: bool,
    source_width: u32,
    source_height: u32,
) -> Result<bgremove_core::AlphaMask> {
    ensure!(!raw.is_empty(), "U2-Net output is empty");
    ensure!(
        raw.iter().all(|v| v.is_finite()),
        "U2-Net output contains NaN/Inf"
    );
    let normalized = if per_image_minmax {
        let (min, max) = raw
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(*v), hi.max(*v))
            });
        if max == min {
            vec![0u8; raw.len()]
        } else {
            raw.iter()
                .map(|v| (((v - min) / (max - min)).clamp(0.0, 1.0) * 255.0) as u8)
                .collect()
        }
    } else {
        raw.iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect()
    };
    let side = (raw.len() as f64).sqrt() as usize;
    ensure!(
        side * side == raw.len(),
        "U2-Net output is not square: {} values",
        raw.len()
    );
    let restored = resize_u8_pillow_lanczos(
        &normalized,
        side as u32,
        side as u32,
        1,
        source_width,
        source_height,
    )?;
    bgremove_core::AlphaMask::new(
        source_width,
        source_height,
        restored.into_iter().map(|v| v as f32 / 255.0).collect(),
    )
}

/// M5 evidence from the common single-output U2-Net adapter.
pub struct U2netRunEvidence {
    pub tensor: TensorInput,
    pub raw_output: TensorOutput,
    pub transformed_output: TensorOutput,
    pub restored: bgremove_core::AlphaMask,
}

/// One tested adapter serves general, light, human and Silueta U2-Net
/// checkpoints. The manifest selects only the verified model and output
/// transform; no model is downloaded here.
pub struct U2netSegmenter {
    pool: SessionPool,
    normalization: OutputNormalization,
}

impl U2netSegmenter {
    pub fn new(
        manifest: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        ensure!(
            manifest.algorithm_family == "u2net",
            "U2-Net adapter requires algorithm_family=u2net"
        );
        ensure!(
            manifest.width == 320 && manifest.height == 320,
            "U2-Net mask adapter requires 320x320"
        );
        ensure!(
            manifest.layout == bgremove_models::ModelLayout::Nchw,
            "U2-Net input must be NCHW"
        );
        ensure!(
            manifest.channel_order == bgremove_models::ChannelOrder::Rgb,
            "U2-Net input must be RGB"
        );
        ensure!(
            manifest.resize_filter == bgremove_models::ResizeFilter::Lanczos3,
            "U2-Net input resize must be Lanczos3"
        );
        ensure!(
            manifest.activation == Activation::None,
            "U2-Net output is direct; activation must be none"
        );
        ensure!(
            manifest.output_index == Some(0),
            "U2-Net must explicitly select first output (output_index=0)"
        );
        ensure!(
            manifest.input_type == Some(TensorElementType::F32)
                && manifest.output_type == Some(TensorElementType::F32),
            "U2-Net tensors must be f32"
        );
        ensure!(
            manifest.input_shape
                == vec![
                    DimensionSpec::Static(1),
                    DimensionSpec::Static(3),
                    DimensionSpec::Static(320),
                    DimensionSpec::Static(320)
                ],
            "U2-Net input shape metadata mismatch"
        );
        ensure!(
            manifest.output_shape
                == vec![
                    DimensionSpec::Static(1),
                    DimensionSpec::Static(1),
                    DimensionSpec::Static(320),
                    DimensionSpec::Static(320)
                ],
            "U2-Net output shape metadata mismatch"
        );
        ensure!(
            manifest.mean == [0.485, 0.456, 0.406] && manifest.std == [0.229, 0.224, 0.225],
            "U2-Net ImageNet normalization metadata mismatch"
        );
        ensure!(
            matches!(
                manifest.output_normalization,
                OutputNormalization::None
                    | OutputNormalization::MinMax
                    | OutputNormalization::Clamp
            ),
            "unsupported U2-Net output normalization"
        );
        Ok(Self {
            pool: SessionPool::new(
                manifest,
                manifest_path,
                runtime,
                workers,
                requested,
                fallback_allowed,
            )?,
            normalization: manifest.output_normalization,
        })
    }

    pub fn predict_with_evidence(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<U2netRunEvidence> {
        let tensor = u2net_preprocess_rgb(image, 320, 320)?;
        let mut lease = self.pool.checkout();
        let output = lease.session_mut().run(&tensor.shape, &tensor.values)?;
        let raw_output = output.clone();
        first_output_mask(&output, 320, 320, "u2net")?;
        let transformed_output =
            apply_output_transform(output, Activation::None, self.normalization)?;
        let transformed = first_output_mask(&transformed_output, 320, 320, "u2net")?;
        let restored =
            restore_u2net_mask_values(&transformed, false, image.width(), image.height())?;
        Ok(U2netRunEvidence {
            tensor,
            raw_output,
            transformed_output,
            restored,
        })
    }

    pub fn predict(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<bgremove_core::AlphaMask> {
        Ok(self.predict_with_evidence(image)?.restored)
    }
    pub fn provider(&self) -> ProviderReport {
        self.pool.provider_report()
    }
}

impl bgremove_core::Segmenter for U2netSegmenter {
    fn predict(
        &mut self,
        image: &bgremove_core::CanonicalImage,
        _prompt: Option<&bgremove_core::Prompt>,
    ) -> Result<bgremove_core::AlphaMask> {
        U2netSegmenter::predict(self, image)
    }
}

/// M7 evidence for one BiRefNet checkpoint.  The adapter is shared by all
/// registered domain variants; the manifest selects the specialist weights,
/// while runtime code never infers a variant from unavailable ground truth.
pub struct BirefnetRunEvidence {
    pub tensor: TensorInput,
    pub raw_output: TensorOutput,
    pub transformed_output: TensorOutput,
    pub restored: bgremove_core::AlphaMask,
}

/// One tested 1024-square BiRefNet engine for general, lite, portrait, DIS,
/// HRSOD, COD and massive checkpoints.
pub struct BirefnetSegmenter {
    pool: SessionPool,
    activation: Activation,
    normalization: OutputNormalization,
}

impl BirefnetSegmenter {
    pub fn new(
        manifest: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        ensure!(
            manifest.algorithm_family == "birefnet",
            "BiRefNet adapter requires algorithm_family=birefnet"
        );
        ensure!(
            manifest.width == 1024 && manifest.height == 1024,
            "BiRefNet adapter requires 1024x1024"
        );
        ensure!(
            manifest.scale == 1.0,
            "BiRefNet preprocessing scale must be exactly 1.0"
        );
        ensure!(
            matches!(
                (
                    manifest.model_variant.as_str(),
                    manifest.model_domain.as_str()
                ),
                ("general", "general")
                    | ("general-lite", "general")
                    | ("portrait", "portrait")
                    | ("dis", "dis")
                    | ("hrsod", "hrsod")
                    | ("cod", "cod")
                    | ("massive", "massive")
            ),
            "BiRefNet manifest must declare one registered specialist variant/domain pair"
        );
        ensure!(
            manifest.aspect == bgremove_models::AspectPolicy::Stretch,
            "BiRefNet adapter requires stretch geometry"
        );
        ensure!(
            manifest.resize_filter == bgremove_models::ResizeFilter::Lanczos3,
            "BiRefNet adapter requires Lanczos3 resize"
        );
        ensure!(
            manifest.layout == bgremove_models::ModelLayout::Nchw
                && manifest.channel_order == bgremove_models::ChannelOrder::Rgb,
            "BiRefNet adapter requires RGB NCHW input"
        );
        ensure!(
            manifest.activation == Activation::Sigmoid,
            "BiRefNet output must declare sigmoid activation"
        );
        ensure!(
            matches!(
                manifest.output_normalization,
                OutputNormalization::Clamp | OutputNormalization::MinMax
            ),
            "BiRefNet output must declare clamp or minmax normalization"
        );
        ensure!(
            manifest.input_type == Some(TensorElementType::F32)
                && manifest.output_type == Some(TensorElementType::F32),
            "BiRefNet tensors must be f32"
        );
        ensure!(
            manifest.output_index == Some(0),
            "BiRefNet must explicitly select output index 0"
        );
        ensure!(
            manifest.input_shape.len() == 4
                && matches!(
                    manifest.input_shape[0],
                    DimensionSpec::Static(1) | DimensionSpec::Dynamic(_)
                )
                && matches!(
                    manifest.input_shape[1],
                    DimensionSpec::Static(3) | DimensionSpec::Dynamic(_)
                )
                && matches!(
                    manifest.input_shape[2],
                    DimensionSpec::Static(1024) | DimensionSpec::Dynamic(_)
                )
                && matches!(
                    manifest.input_shape[3],
                    DimensionSpec::Static(1024) | DimensionSpec::Dynamic(_)
                ),
            "BiRefNet input shape metadata mismatch"
        );
        ensure!(
            manifest.output_shape
                == vec![
                    DimensionSpec::Static(1),
                    DimensionSpec::Static(1),
                    DimensionSpec::Static(1024),
                    DimensionSpec::Static(1024)
                ],
            "BiRefNet output shape metadata mismatch"
        );
        ensure!(
            manifest.mean == [0.485, 0.456, 0.406] && manifest.std == [0.229, 0.224, 0.225],
            "BiRefNet ImageNet normalization metadata mismatch"
        );
        Ok(Self {
            pool: SessionPool::new(
                manifest,
                manifest_path,
                runtime,
                workers,
                requested,
                fallback_allowed,
            )?,
            activation: manifest.activation,
            normalization: manifest.output_normalization,
        })
    }

    pub fn predict_with_evidence(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<BirefnetRunEvidence> {
        let tensor = birefnet_preprocess_rgb(image)?;
        let mut lease = self.pool.checkout();
        let raw_output = lease.session_mut().run(&tensor.shape, &tensor.values)?;
        let logits = first_output_mask(&raw_output, 1024, 1024, "birefnet")?;
        // apply_output_transform intentionally performs sigmoid before either
        // clamp or per-image min/max.  This is the parity-critical rembg
        // ordering; normalizing logits before sigmoid is a different model.
        let transformed_output =
            apply_output_transform(raw_output.clone(), self.activation, self.normalization)?;
        let transformed = first_output_mask(&transformed_output, 1024, 1024, "birefnet")?;
        ensure!(
            logits.iter().all(|value| value.is_finite()),
            "BiRefNet logits contain NaN/Inf"
        );
        let restored = restore_birefnet_mask(&transformed, image.width(), image.height())?;
        Ok(BirefnetRunEvidence {
            tensor,
            raw_output,
            transformed_output,
            restored,
        })
    }

    pub fn predict(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<bgremove_core::AlphaMask> {
        Ok(self.predict_with_evidence(image)?.restored)
    }

    pub fn provider(&self) -> ProviderReport {
        self.pool.provider_report()
    }
}

impl bgremove_core::Segmenter for BirefnetSegmenter {
    fn predict(
        &mut self,
        image: &bgremove_core::CanonicalImage,
        _prompt: Option<&bgremove_core::Prompt>,
    ) -> Result<bgremove_core::AlphaMask> {
        BirefnetSegmenter::predict(self, image)
    }
}

/// BRIA RMBG's two checked-in repository profiles.  The profile is part of
/// the model contract: RMBG-1.4's original Rust wrapper premultiplies RGBA
/// before its bilinear stretch, whereas rembg's RMBG-2.0 Python wrapper uses
/// Pillow RGB/LANCZOS and the global resized-image maximum.
#[cfg(feature = "bria")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmbgProfile {
    RustCrate,
    RembgPython,
}

#[cfg(feature = "bria")]
impl RmbgProfile {
    pub fn from_manifest(manifest: &ModelManifest) -> Result<Self> {
        match manifest.preprocessing_profile {
            PreprocessingProfile::RmbgRust => Ok(Self::RustCrate),
            PreprocessingProfile::RembgBria => Ok(Self::RembgPython),
            other => bail!(
                "manifest {} is not an RMBG profile ({other:?})",
                manifest.id
            ),
        }
    }
}

/// Resize an encoded RGBA image with the U8 convolution path from
/// `fast_image_resize` 3.0.4. This is a small, dependency-free port of its
/// bilinear coefficient normalization, U8 convolution rounding, and
/// `MulDiv` alpha operations. RGB is premultiplied by alpha while sampling,
/// then unpremultiplied, matching the local `rust/rmbg` source.
#[cfg(feature = "bria")]
fn resize_rmbg_rgba(
    rgb: &[[f32; 3]],
    alpha: &[f32],
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<(Vec<[f32; 3]>, Vec<f32>)> {
    ensure!(
        source_width > 0 && source_height > 0 && target_width > 0 && target_height > 0,
        "RMBG resize dimensions must be positive"
    );
    ensure!(
        rgb.len() == (source_width as usize) * (source_height as usize),
        "RMBG RGB resize length mismatch"
    );
    ensure!(
        alpha.len() == rgb.len(),
        "RMBG alpha resize length mismatch"
    );
    let rgb_u8 = rgb
        .iter()
        .map(|pixel| pixel.map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8))
        .collect::<Vec<_>>();
    let alpha_u8 = alpha
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<_>>();
    let mut source = vec![[0u8; 4]; rgb_u8.len()];
    for (index, pixel) in source.iter_mut().enumerate() {
        let alpha = alpha_u8[index];
        pixel[0] = fast_mul_div_255(rgb_u8[index][0], alpha);
        pixel[1] = fast_mul_div_255(rgb_u8[index][1], alpha);
        pixel[2] = fast_mul_div_255(rgb_u8[index][2], alpha);
        pixel[3] = alpha;
    }
    let (horizontal, horizontal_precision) = fast_bilinear_coefficients(source_width, target_width);
    let (vertical, vertical_precision) = fast_bilinear_coefficients(source_height, target_height);
    let mut vertical_pass = vec![[0u8; 4]; (source_width as usize) * (target_height as usize)];
    for oy in 0..target_height as usize {
        let (start, coefficients) = &vertical[oy];
        for x in 0..source_width as usize {
            let mut sums = [1i32 << (vertical_precision - 1); 4];
            for (tap, coefficient) in coefficients.iter().enumerate() {
                let pixel = source[(start + tap) * source_width as usize + x];
                for channel in 0..4 {
                    sums[channel] += pixel[channel] as i32 * i32::from(*coefficient);
                }
            }
            let output = &mut vertical_pass[oy * source_width as usize + x];
            for channel in 0..4 {
                output[channel] = fast_clip_u8(sums[channel], vertical_precision);
            }
        }
    }
    let mut resized = vec![[0u8; 4]; (target_width as usize) * (target_height as usize)];
    for y in 0..target_height as usize {
        for (ox, (start, coefficients)) in horizontal.iter().enumerate() {
            let mut sums = [1i32 << (horizontal_precision - 1); 4];
            for (tap, coefficient) in coefficients.iter().enumerate() {
                let pixel = vertical_pass[y * source_width as usize + start + tap];
                for channel in 0..4 {
                    sums[channel] += pixel[channel] as i32 * i32::from(*coefficient);
                }
            }
            let output = &mut resized[y * target_width as usize + ox];
            for channel in 0..4 {
                output[channel] = fast_clip_u8(sums[channel], horizontal_precision);
            }
        }
    }
    let mut out_rgb = vec![[0.0; 3]; resized.len()];
    let mut out_alpha = vec![0.0; resized.len()];
    for (index, pixel) in resized.into_iter().enumerate() {
        let alpha = pixel[3];
        out_alpha[index] = alpha as f32 / 255.0;
        out_rgb[index] = [
            fast_div_alpha(pixel[0], alpha) as f32 / 255.0,
            fast_div_alpha(pixel[1], alpha) as f32 / 255.0,
            fast_div_alpha(pixel[2], alpha) as f32 / 255.0,
        ];
    }
    Ok((out_rgb, out_alpha))
}

#[cfg(feature = "bria")]
fn fast_bilinear_coefficients(input: u32, output: u32) -> (Vec<(usize, Vec<i16>)>, u8) {
    let scale = input as f64 / output as f64;
    // fast_image_resize 3.0.4 widens the bilinear kernel for downsampling;
    // omitting this is a subtle but material mismatch on restoration.
    let filter_scale = scale.max(1.0);
    let radius = filter_scale;
    let mut raw = Vec::with_capacity(output as usize);
    for out in 0..output {
        let center = (out as f64 + 0.5) * scale;
        let minimum = (center - radius).floor().max(0.0) as u32;
        let maximum = (center + radius).ceil().min(input as f64) as u32;
        let mut coefficients = Vec::new();
        for x in minimum..maximum {
            let weight = (1.0 - ((x as f64 - (center - 0.5)) / filter_scale).abs()).max(0.0);
            coefficients.push((x, weight));
        }
        while coefficients
            .first()
            .is_some_and(|(_, weight)| *weight == 0.0)
        {
            coefficients.remove(0);
        }
        while coefficients
            .last()
            .is_some_and(|(_, weight)| *weight == 0.0)
        {
            coefficients.pop();
        }
        let sum: f64 = coefficients.iter().map(|(_, weight)| *weight).sum();
        raw.push((
            coefficients.first().map_or(0, |(x, _)| *x as usize),
            coefficients
                .into_iter()
                .map(|(x, weight)| (x, weight / sum))
                .collect::<Vec<_>>(),
        ));
    }
    let max_weight = raw
        .iter()
        .flat_map(|(_, values)| values.iter().map(|(_, weight)| *weight))
        .fold(0.0f64, f64::max);
    let mut precision = 0u8;
    for candidate in 0..22u8 {
        precision = candidate;
        let next = (max_weight * (1u32 << (candidate + 1)) as f64).round() as i32;
        if next >= (1 << 15) {
            break;
        }
    }
    let coefficients = raw
        .into_iter()
        .map(|(start, values)| {
            (
                start,
                values
                    .into_iter()
                    .map(|(_, weight)| (weight * (1u32 << precision) as f64).round() as i16)
                    .collect(),
            )
        })
        .collect();
    (coefficients, precision)
}

#[cfg(feature = "bria")]
#[inline]
fn fast_mul_div_255(a: u8, b: u8) -> u8 {
    let value = a as u32 * b as u32 + 128;
    (((value >> 8) + value) >> 8) as u8
}

#[cfg(feature = "bria")]
#[inline]
fn fast_div_alpha(value: u8, alpha: u8) -> u8 {
    let reciprocal = if alpha == 0 {
        0
    } else {
        (((255u32 * 512) / alpha as u32) + 1) >> 1
    };
    ((value as u32 * reciprocal) >> 8).min(255) as u8
}

#[cfg(feature = "bria")]
#[inline]
fn fast_clip_u8(value: i32, precision: u8) -> u8 {
    (value >> precision).clamp(0, 255) as u8
}

#[cfg(feature = "bria")]
fn resize_rmbg_mask_u8(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>> {
    ensure!(
        source.len() == source_width as usize * source_height as usize,
        "RMBG mask resize length mismatch"
    );
    let (horizontal, horizontal_precision) = fast_bilinear_coefficients(source_width, target_width);
    let (vertical, vertical_precision) = fast_bilinear_coefficients(source_height, target_height);
    let mut vertical_pass = vec![0u8; source_width as usize * target_height as usize];
    for oy in 0..target_height as usize {
        let (start, coefficients) = &vertical[oy];
        for x in 0..source_width as usize {
            let mut sum = 1i32 << (vertical_precision - 1);
            for (tap, coefficient) in coefficients.iter().enumerate() {
                sum += source[(start + tap) * source_width as usize + x] as i32
                    * i32::from(*coefficient);
            }
            vertical_pass[oy * source_width as usize + x] = fast_clip_u8(sum, vertical_precision);
        }
    }
    let mut output = vec![0u8; target_width as usize * target_height as usize];
    for y in 0..target_height as usize {
        for (ox, (start, coefficients)) in horizontal.iter().enumerate() {
            let mut sum = 1i32 << (horizontal_precision - 1);
            for (tap, coefficient) in coefficients.iter().enumerate() {
                sum += vertical_pass[y * source_width as usize + start + tap] as i32
                    * i32::from(*coefficient);
            }
            output[y * target_width as usize + ox] = fast_clip_u8(sum, horizontal_precision);
        }
    }
    Ok(output)
}

#[cfg(feature = "bria")]
fn rgb_bytes(image: &bgremove_core::CanonicalImage) -> Vec<u8> {
    image
        .rgb()
        .data()
        .iter()
        .flat_map(|px| {
            px.iter()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect()
}

/// Exact preprocessing used by the local `rust/rmbg` crate (RMBG-1.4).
/// Unlike rembg, it preserves the source alpha during the RGBA resize.
#[cfg(feature = "bria")]
pub fn rmbg_rust_preprocess_rgb(image: &bgremove_core::CanonicalImage) -> Result<TensorInput> {
    let (resized, _) = resize_rmbg_rgba(
        image.rgb().data(),
        image.source_alpha().data(),
        image.width(),
        image.height(),
        1024,
        1024,
    )?;
    let plane = 1024usize * 1024usize;
    let mut values = vec![0.0; 3 * plane];
    for (index, pixel) in resized.iter().enumerate() {
        for c in 0..3 {
            let value = pixel[c] - 0.5;
            ensure!(
                value.is_finite(),
                "RMBG Rust preprocessing produced NaN/Inf"
            );
            values[c * plane + index] = value;
        }
    }
    Ok(TensorInput {
        shape: vec![1, 3, 1024, 1024],
        values,
    })
}

/// Exact preprocessing used by rembg's pinned `BriaRmBgSession` (RMBG-2.0).
/// This deliberately does not call the BiRefNet helper so changing another
/// model family cannot silently change this profile.
#[cfg(feature = "bria")]
pub fn rembg_bria_preprocess_rgb(image: &bgremove_core::CanonicalImage) -> Result<TensorInput> {
    let resized = resize_u8_pillow_lanczos(
        &rgb_bytes(image),
        image.width(),
        image.height(),
        3,
        1024,
        1024,
    )?;
    let max_value = resized.iter().copied().max().unwrap_or(0).max(1) as f32;
    let plane = 1024usize * 1024usize;
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut values = vec![0.0; 3 * plane];
    for index in 0..plane {
        for c in 0..3 {
            let value = (resized[index * 3 + c] as f32 / max_value - mean[c]) / std[c];
            ensure!(
                value.is_finite(),
                "rembg BRIA preprocessing produced NaN/Inf"
            );
            values[c * plane + index] = value;
        }
    }
    Ok(TensorInput {
        shape: vec![1, 3, 1024, 1024],
        values,
    })
}

#[cfg(feature = "bria")]
pub fn rmbg_preprocess_rgb(
    image: &bgremove_core::CanonicalImage,
    profile: RmbgProfile,
) -> Result<TensorInput> {
    match profile {
        RmbgProfile::RustCrate => rmbg_rust_preprocess_rgb(image),
        RmbgProfile::RembgPython => rembg_bria_preprocess_rgb(image),
    }
}

/// Safely apply either the repository's per-image min/max or a calibrated
/// clamp. Constant tensors intentionally become all-zero rather than NaN.
#[cfg(feature = "bria")]
pub fn normalize_rmbg_output(
    values: &[f32],
    normalization: OutputNormalization,
) -> Result<Vec<f32>> {
    ensure!(!values.is_empty(), "RMBG output is empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "RMBG output contains NaN/Inf"
    );
    let mut out = values.to_vec();
    match normalization {
        OutputNormalization::None => {}
        OutputNormalization::Clamp => out
            .iter_mut()
            .for_each(|value| *value = value.clamp(0.0, 1.0)),
        OutputNormalization::MinMax => {
            let (min, max) = values
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), value| {
                    (lo.min(*value), hi.max(*value))
                });
            if (max - min).abs() <= f32::EPSILON {
                out.fill(0.0);
            } else {
                out.iter_mut()
                    .for_each(|value| *value = ((*value - min) / (max - min)).clamp(0.0, 1.0));
            }
        }
    }
    ensure!(
        out.iter().all(|value| value.is_finite()),
        "RMBG transformed output contains NaN/Inf"
    );
    Ok(out)
}

/// Restore a 1024² output with the profile's repository resampler. Values are
/// quantized exactly as each source wrapper does before restoring geometry.
#[cfg(feature = "bria")]
pub fn restore_rmbg_mask(
    values: &[f32],
    profile: RmbgProfile,
    normalization: OutputNormalization,
    source_width: u32,
    source_height: u32,
) -> Result<bgremove_core::AlphaMask> {
    ensure!(
        values.len() == 1024 * 1024,
        "RMBG output must contain 1024x1024 values"
    );
    let values = normalize_rmbg_output(values, normalization)?;
    let bytes = values
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0) as u8)
        .collect::<Vec<_>>();
    let restored = match profile {
        RmbgProfile::RustCrate => {
            resize_rmbg_mask_u8(&bytes, 1024, 1024, source_width, source_height)?
        }
        RmbgProfile::RembgPython => {
            resize_u8_pillow_lanczos(&bytes, 1024, 1024, 1, source_width, source_height)?
        }
    };
    bgremove_core::AlphaMask::new(
        source_width,
        source_height,
        restored
            .into_iter()
            .map(|value| value as f32 / 255.0)
            .collect(),
    )
}

#[cfg(feature = "bria")]
pub struct RmbgRunEvidence {
    pub profile: RmbgProfile,
    pub tensor: TensorInput,
    pub raw_output: TensorOutput,
    pub transformed_output: TensorOutput,
    pub restored: bgremove_core::AlphaMask,
}

/// Manifest-driven BRIA adapter. The type is behind the `bria` Cargo feature
/// so a commercial build can remove BRIA integration entirely; manifests also
/// remain fail-closed on unapproved checkpoint licences.
#[cfg(feature = "bria")]
pub struct RmbgSegmenter {
    pool: SessionPool,
    profile: RmbgProfile,
    normalization: OutputNormalization,
}

#[cfg(feature = "bria")]
impl RmbgSegmenter {
    pub fn new(
        manifest: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        ensure!(
            manifest.algorithm_family == "rmbg",
            "RMBG adapter requires algorithm_family=rmbg"
        );
        ensure!(
            manifest.width == 1024 && manifest.height == 1024,
            "RMBG adapter requires 1024x1024"
        );
        ensure!(
            manifest.layout == bgremove_models::ModelLayout::Nchw
                && manifest.aspect == bgremove_models::AspectPolicy::Stretch
                && manifest.channel_order == bgremove_models::ChannelOrder::Rgb,
            "RMBG adapter requires RGB NCHW stretch input"
        );
        ensure!(
            manifest.activation == Activation::None,
            "RMBG output is direct; activation must be none"
        );
        ensure!(
            manifest.output_index == Some(0),
            "RMBG must explicitly select output index 0"
        );
        ensure!(
            manifest.input_type == Some(TensorElementType::F32)
                && manifest.output_type == Some(TensorElementType::F32),
            "RMBG tensors must be f32"
        );
        ensure!(
            manifest.input_shape.len() == 4
                && matches!(
                    manifest.input_shape[0],
                    DimensionSpec::Static(1) | DimensionSpec::Dynamic(_)
                )
                && matches!(
                    manifest.input_shape[1],
                    DimensionSpec::Static(3) | DimensionSpec::Dynamic(_)
                )
                && matches!(
                    manifest.input_shape[2],
                    DimensionSpec::Static(1024) | DimensionSpec::Dynamic(_)
                )
                && matches!(
                    manifest.input_shape[3],
                    DimensionSpec::Static(1024) | DimensionSpec::Dynamic(_)
                ),
            "RMBG input shape metadata mismatch"
        );
        ensure!(
            manifest.output_shape.len() == 4
                && matches!(
                    manifest.output_shape[0],
                    DimensionSpec::Static(1) | DimensionSpec::Dynamic(_)
                )
                && match manifest.output_shape[1] {
                    DimensionSpec::Static(channels) => channels > 0,
                    DimensionSpec::Dynamic(_) => true,
                }
                && matches!(
                    manifest.output_shape[2],
                    DimensionSpec::Static(1024) | DimensionSpec::Dynamic(_)
                )
                && matches!(
                    manifest.output_shape[3],
                    DimensionSpec::Static(1024) | DimensionSpec::Dynamic(_)
                ),
            "RMBG output shape metadata mismatch"
        );
        let profile = RmbgProfile::from_manifest(manifest)?;
        match profile {
            RmbgProfile::RustCrate => ensure!(
                manifest.resize_filter == bgremove_models::ResizeFilter::Bilinear
                    && manifest.mean == [0.5; 3]
                    && manifest.std == [1.0; 3],
                "RMBG Rust profile metadata mismatch"
            ),
            RmbgProfile::RembgPython => ensure!(
                manifest.resize_filter == bgremove_models::ResizeFilter::Lanczos3
                    && manifest.mean == [0.485, 0.456, 0.406]
                    && manifest.std == [0.229, 0.224, 0.225],
                "rembg BRIA profile metadata mismatch"
            ),
        }
        ensure!(
            matches!(
                manifest.output_normalization,
                OutputNormalization::MinMax | OutputNormalization::Clamp
            ),
            "RMBG output must declare minmax or clamp"
        );
        Ok(Self {
            pool: SessionPool::new(
                manifest,
                manifest_path,
                runtime,
                workers,
                requested,
                fallback_allowed,
            )?,
            profile,
            normalization: manifest.output_normalization,
        })
    }

    pub fn predict_with_evidence(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<RmbgRunEvidence> {
        let tensor = rmbg_preprocess_rgb(image, self.profile)?;
        let mut lease = self.pool.checkout();
        let raw_output = lease.session_mut().run(&tensor.shape, &tensor.values)?;
        // Both pinned source profiles select output 0/channel 0 before applying
        // their profile normalization.  Normalizing the complete multi-channel
        // tensor would let unrelated channels change per-image min/max.
        let raw_channel0 = first_output_mask_channel0(&raw_output, 1024, 1024, "rmbg")?;
        let transformed = normalize_rmbg_output(&raw_channel0, self.normalization)?;
        let transformed_output = TensorOutput {
            shape: vec![1, 1, 1024, 1024],
            values: transformed.clone(),
        };
        let restored = restore_rmbg_mask(
            &transformed,
            self.profile,
            OutputNormalization::None,
            image.width(),
            image.height(),
        )?;
        Ok(RmbgRunEvidence {
            profile: self.profile,
            tensor,
            raw_output,
            transformed_output,
            restored,
        })
    }

    pub fn predict(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<bgremove_core::AlphaMask> {
        Ok(self.predict_with_evidence(image)?.restored)
    }
    pub fn provider(&self) -> ProviderReport {
        self.pool.provider_report()
    }
}

#[cfg(feature = "bria")]
impl bgremove_core::Segmenter for RmbgSegmenter {
    fn predict(
        &mut self,
        image: &bgremove_core::CanonicalImage,
        _prompt: Option<&bgremove_core::Prompt>,
    ) -> Result<bgremove_core::AlphaMask> {
        RmbgSegmenter::predict(self, image)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClothCategory {
    Upper,
    Lower,
    Full,
}
impl ClothCategory {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "upper" => Ok(Self::Upper),
            "lower" => Ok(Self::Lower),
            "full" => Ok(Self::Full),
            other => bail!("invalid cloth category {other}; expected upper, lower, or full"),
        }
    }
    pub fn class(self) -> u8 {
        match self {
            Self::Upper => 1,
            Self::Lower => 2,
            Self::Full => 3,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upper => "upper",
            Self::Lower => "lower",
            Self::Full => "full",
        }
    }
}

/// Compute the cloth class-ID map from class-major logits. The strict `>`
/// comparison deliberately gives the first class the win on exact ties, as
/// NumPy's `argmax` does.
pub fn cloth_argmax_class_map(logits: &[f32], classes: usize, pixels: usize) -> Result<Vec<u8>> {
    ensure!(classes > 0 && classes <= 256, "invalid cloth class count");
    ensure!(pixels > 0, "invalid cloth pixel count");
    ensure!(
        logits.len() == classes * pixels,
        "cloth logits length mismatch"
    );
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "cloth logits contain NaN/Inf"
    );
    Ok((0..pixels)
        .map(|pixel| {
            let mut best = 0usize;
            for class in 1..classes {
                if logits[class * pixels + pixel] > logits[best * pixels + pixel] {
                    best = class;
                }
            }
            best as u8
        })
        .collect())
}

/// Return a normalized binary alpha mask for one garment category.
pub fn cloth_category_mask(class_map: &[u8], category: ClothCategory) -> Vec<f32> {
    class_map
        .iter()
        .map(|class| f32::from(*class == category.class()))
        .collect()
}

/// Return the same garment selector in rembg's binary uint8 convention.
pub fn cloth_category_mask_u8(class_map: &[u8], category: ClothCategory) -> Vec<u8> {
    class_map
        .iter()
        .map(|class| if *class == category.class() { 255 } else { 0 })
        .collect()
}

pub struct U2netClothRunEvidence {
    pub tensor: TensorInput,
    pub raw_output: TensorOutput,
    pub class_map: Vec<u8>,
    pub restored_class_map: Vec<u8>,
}

pub struct U2netClothSegmenter {
    pool: SessionPool,
}

/// CarveKit's ImageNet preprocessing used by the three M6 segmenters.  The
/// wrappers at the pinned CarveKit revision are intentionally not treated as
/// interchangeable: BASNet stretches to 320, TRACER stretches to 640, while
/// DeepLabV3 uses a non-upscaling aspect-preserving thumbnail.  The returned
/// tensor is complete evidence (RGB, NCHW, f32) and never contains NaN/Inf.
pub fn carvekit_imagenet_preprocess_rgb(
    image: &bgremove_core::CanonicalImage,
    family: &str,
) -> Result<TensorInput> {
    let (source_w, source_h) = image.dimensions();
    let (target_w, target_h, thumbnail) = match family {
        "basnet" => (320, 320, false),
        "tracer-b7" => (640, 640, false),
        "deeplabv3" => (1024, 1024, true),
        other => bail!("unknown CarveKit M6 family {other}"),
    };
    ensure!(
        source_w > 0 && source_h > 0,
        "source dimensions must be positive"
    );
    let (resize_w, resize_h) = if thumbnail {
        // Pillow Image.thumbnail never enlarges an image and keeps the aspect
        // ratio.  Round to the nearest integer with a one-pixel lower bound.
        let scale = (target_w as f64 / source_w as f64)
            .min(target_h as f64 / source_h as f64)
            .min(1.0);
        (
            ((source_w as f64 * scale).round() as u32).max(1),
            ((source_h as f64 * scale).round() as u32).max(1),
        )
    } else {
        (target_w, target_h)
    };
    if family == "tracer-b7" {
        let resized = resize_f32_torchvision_bilinear(
            image.rgb().data(),
            source_w,
            source_h,
            resize_w,
            resize_h,
        );
        let channels = resized.len();
        let mut values = vec![0.0f32; channels * 3];
        for (i, pixel) in resized.iter().enumerate() {
            for c in 0..3 {
                let value = (pixel[c] - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c];
                ensure!(
                    value.is_finite(),
                    "CarveKit {family} tensor contains NaN/Inf"
                );
                values[c * channels + i] = value;
            }
        }
        return Ok(TensorInput {
            values,
            shape: vec![1, 3, i64::from(resize_h), i64::from(resize_w)],
        });
    }
    let mut bytes = Vec::with_capacity(source_w as usize * source_h as usize * 3);
    for px in image.rgb().data() {
        for value in px {
            bytes.push((value.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    let resized = if resize_w == source_w && resize_h == source_h {
        bytes
    } else {
        match family {
            "basnet" => {
                resize_u8_pillow_bicubic(&bytes, source_w, source_h, 3, resize_w, resize_h)?
            }
            "tracer-b7" => unreachable!("TRACER uses the f32 tensor resize path above"),
            "deeplabv3" => {
                resize_u8_pillow_bicubic(&bytes, source_w, source_h, 3, resize_w, resize_h)?
            }
            _ => unreachable!(),
        }
    };
    let channels = resize_w as usize * resize_h as usize;
    let max_value = resized.iter().copied().max().unwrap_or(0) as f32;
    let denominator = if family == "basnet" {
        max_value.max(1.0)
    } else {
        255.0
    };
    let mut values = vec![0.0f32; channels * 3];
    for i in 0..channels {
        for c in 0..3 {
            let normalized = resized[i * 3 + c] as f32 / denominator;
            let value = (normalized - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c];
            ensure!(
                value.is_finite(),
                "CarveKit {family} tensor contains NaN/Inf"
            );
            values[c * channels + i] = value;
        }
    }
    Ok(TensorInput {
        shape: vec![1, 3, resize_h as i64, resize_w as i64],
        values,
    })
}

/// Restore an M6 direct soft mask to canonical dimensions. Quantisation is
/// explicit because CarveKit wrappers convert to uint8 before restoring;
/// CatmullRom selects Pillow bicubic and Triangle selects Pillow bilinear.
pub fn restore_carvekit_soft_mask(
    raw: &[f32],
    model_width: u32,
    model_height: u32,
    source_width: u32,
    source_height: u32,
    filter: image::imageops::FilterType,
) -> Result<bgremove_core::AlphaMask> {
    ensure!(
        model_width > 0 && model_height > 0,
        "invalid model dimensions"
    );
    ensure!(
        raw.len() == model_width as usize * model_height as usize,
        "M6 mask length mismatch"
    );
    ensure!(
        raw.iter().all(|v| v.is_finite()),
        "M6 mask contains NaN/Inf"
    );
    let bytes = raw
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect::<Vec<_>>();
    let restored_bytes = match filter {
        image::imageops::FilterType::CatmullRom => resize_u8_pillow_bicubic(
            &bytes,
            model_width,
            model_height,
            1,
            source_width,
            source_height,
        )?,
        image::imageops::FilterType::Triangle => resize_u8_pillow_bilinear(
            &bytes,
            model_width,
            model_height,
            1,
            source_width,
            source_height,
        )?,
        image::imageops::FilterType::Nearest => image::imageops::resize(
            &image::GrayImage::from_raw(model_width, model_height, bytes)
                .ok_or_else(|| anyhow::anyhow!("invalid M6 mask buffer"))?,
            source_width,
            source_height,
            filter,
        )
        .into_raw(),
        other => image::imageops::resize(
            &image::GrayImage::from_raw(model_width, model_height, bytes)
                .ok_or_else(|| anyhow::anyhow!("invalid M6 mask buffer"))?,
            source_width,
            source_height,
            other,
        )
        .into_raw(),
    };
    bgremove_core::AlphaMask::new(
        source_width,
        source_height,
        restored_bytes
            .into_iter()
            .map(|v| v as f32 / 255.0)
            .collect(),
    )
}

/// Convert DeepLabV3's class-major logits into a hard foreground mask.
/// Class 0 is background and every nonzero semantic class is foreground.
/// Strict `>` ties preserve the first-class PyTorch argmax rule.
pub fn deeplab_argmax_foreground(
    logits: &[f32],
    pixels: usize,
    classes: usize,
) -> Result<Vec<f32>> {
    ensure!(
        classes >= 2,
        "M6 DeepLabV3 expects background plus foreground classes"
    );
    ensure!(
        pixels > 0 && logits.len() == classes * pixels,
        "DeepLabV3 logits shape mismatch"
    );
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "DeepLabV3 logits contain NaN/Inf"
    );
    Ok((0..pixels)
        .map(|pixel| {
            let mut best = 0usize;
            for class in 1..classes {
                if logits[class * pixels + pixel] > logits[best * pixels + pixel] {
                    best = class;
                }
            }
            if best == 0 {
                0.0
            } else {
                1.0
            }
        })
        .collect())
}

/// M6 evidence for BASNet's first output and optional per-image min/max.
pub struct BasnetRunEvidence {
    pub tensor: TensorInput,
    pub raw_output: TensorOutput,
    pub transformed_output: TensorOutput,
    pub restored: bgremove_core::AlphaMask,
}

pub struct BasnetSegmenter {
    pool: SessionPool,
    normalization: OutputNormalization,
}

impl BasnetSegmenter {
    pub fn new(
        manifest: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        validate_m6_manifest(manifest, "basnet", 320, 320)?;
        ensure!(
            manifest.output_shape
                == vec![
                    DimensionSpec::Static(1),
                    DimensionSpec::Static(1),
                    DimensionSpec::Static(320),
                    DimensionSpec::Static(320)
                ],
            "BASNet output shape metadata mismatch"
        );
        ensure!(
            manifest.aspect == bgremove_models::AspectPolicy::Stretch,
            "BASNet requires stretch geometry"
        );
        ensure!(
            manifest.output_index == Some(0),
            "BASNet must select first output"
        );
        Ok(Self {
            pool: SessionPool::new(
                manifest,
                manifest_path,
                runtime,
                workers,
                requested,
                fallback_allowed,
            )?,
            normalization: manifest.output_normalization,
        })
    }
    pub fn predict_with_evidence(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<BasnetRunEvidence> {
        let tensor = carvekit_imagenet_preprocess_rgb(image, "basnet")?;
        let mut lease = self.pool.checkout();
        let raw_output = lease.session_mut().run(&tensor.shape, &tensor.values)?;
        let _ = first_output_mask(&raw_output, 320, 320, "basnet")?;
        let transformed_output =
            apply_output_transform(raw_output.clone(), Activation::None, self.normalization)?;
        let transformed = first_output_mask(&transformed_output, 320, 320, "basnet")?;
        let restored = restore_carvekit_soft_mask(
            &transformed,
            320,
            320,
            image.width(),
            image.height(),
            image::imageops::FilterType::CatmullRom,
        )?;
        Ok(BasnetRunEvidence {
            tensor,
            raw_output,
            transformed_output,
            restored,
        })
    }
    pub fn predict(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<bgremove_core::AlphaMask> {
        Ok(self.predict_with_evidence(image)?.restored)
    }
    pub fn provider(&self) -> ProviderReport {
        self.pool.provider_report()
    }
}
impl bgremove_core::Segmenter for BasnetSegmenter {
    fn predict(
        &mut self,
        image: &bgremove_core::CanonicalImage,
        _prompt: Option<&bgremove_core::Prompt>,
    ) -> Result<bgremove_core::AlphaMask> {
        BasnetSegmenter::predict(self, image)
    }
}

pub struct TracerB7RunEvidence {
    pub tensor: TensorInput,
    pub raw_output: TensorOutput,
    pub restored: bgremove_core::AlphaMask,
}
pub struct TracerB7Segmenter {
    pool: SessionPool,
}
impl TracerB7Segmenter {
    pub fn new(
        manifest: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        validate_m6_manifest(manifest, "tracer-b7", 640, 640)?;
        ensure!(
            manifest.output_shape
                == vec![
                    DimensionSpec::Dynamic("batch".into()),
                    DimensionSpec::Dynamic("channel".into()),
                    DimensionSpec::Dynamic("height".into()),
                    DimensionSpec::Dynamic("width".into())
                ],
            "TRACER-B7 output shape metadata mismatch"
        );
        ensure!(
            manifest.aspect == bgremove_models::AspectPolicy::Stretch
                && manifest.output_normalization == OutputNormalization::None,
            "TRACER-B7 requires direct soft output"
        );
        Ok(Self {
            pool: SessionPool::new(
                manifest,
                manifest_path,
                runtime,
                workers,
                requested,
                fallback_allowed,
            )?,
        })
    }
    pub fn predict_with_evidence(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<TracerB7RunEvidence> {
        let tensor = carvekit_imagenet_preprocess_rgb(image, "tracer-b7")?;
        let mut lease = self.pool.checkout();
        let raw_output = lease.session_mut().run(&tensor.shape, &tensor.values)?;
        let raw = first_output_mask(&raw_output, 640, 640, "tracer-b7")?;
        // CarveKit casts to uint8 then uses Pillow bilinear; the helper's
        // antialiasing is active for downsampling and matches that path.
        let restored = restore_carvekit_soft_mask(
            &raw,
            640,
            640,
            image.width(),
            image.height(),
            image::imageops::FilterType::Triangle,
        )?;
        Ok(TracerB7RunEvidence {
            tensor,
            raw_output,
            restored,
        })
    }
    pub fn predict(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<bgremove_core::AlphaMask> {
        Ok(self.predict_with_evidence(image)?.restored)
    }
    pub fn provider(&self) -> ProviderReport {
        self.pool.provider_report()
    }
}
impl bgremove_core::Segmenter for TracerB7Segmenter {
    fn predict(
        &mut self,
        image: &bgremove_core::CanonicalImage,
        _prompt: Option<&bgremove_core::Prompt>,
    ) -> Result<bgremove_core::AlphaMask> {
        TracerB7Segmenter::predict(self, image)
    }
}

pub struct DeepLabV3RunEvidence {
    pub tensor: TensorInput,
    pub raw_output: TensorOutput,
    pub hard_class_map: Vec<f32>,
    pub restored: bgremove_core::AlphaMask,
}
pub struct DeepLabV3Segmenter {
    pool: SessionPool,
    class_count: usize,
}
impl DeepLabV3Segmenter {
    pub fn new(
        manifest: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        validate_m6_manifest(manifest, "deeplabv3", 1024, 1024)?;
        ensure!(
            manifest.aspect == bgremove_models::AspectPolicy::Thumbnail,
            "DeepLabV3 requires thumbnail geometry"
        );
        ensure!(
            manifest
                .class_mapping
                .as_ref()
                .is_some_and(|mapping| mapping.len() >= 2 && mapping[0] == "background"),
            "DeepLabV3 class_mapping must name background and foreground classes"
        );
        ensure!(
            manifest.output_shape
                == vec![
                    DimensionSpec::Dynamic("batch".into()),
                    DimensionSpec::Dynamic("classes".into()),
                    DimensionSpec::Dynamic("height".into()),
                    DimensionSpec::Dynamic("width".into())
                ],
            "DeepLabV3 output shape metadata must be [batch,classes,height,width]"
        );
        ensure!(
            manifest.output_normalization == OutputNormalization::None
                && manifest.activation == Activation::None,
            "DeepLabV3 logits must remain direct"
        );
        let class_count = manifest.class_mapping.as_ref().unwrap().len();
        Ok(Self {
            pool: SessionPool::new(
                manifest,
                manifest_path,
                runtime,
                workers,
                requested,
                fallback_allowed,
            )?,
            class_count,
        })
    }
    pub fn predict_with_evidence(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<DeepLabV3RunEvidence> {
        let tensor = carvekit_imagenet_preprocess_rgb(image, "deeplabv3")?;
        let mut lease = self.pool.checkout();
        let raw_output = lease.session_mut().run(&tensor.shape, &tensor.values)?;
        let (h, w, classes) = match raw_output.shape.as_slice() {
            [1, classes, h, w] if *classes >= 2 && *h > 0 && *w > 0 => {
                (*h as u32, *w as u32, *classes as usize)
            }
            other => bail!("DeepLabV3 output shape {other:?} is not [1,C,H,W] with C>=2"),
        };
        ensure!(
            self.class_count == classes,
            "DeepLabV3 runtime class count does not match the manifest mapping"
        );
        let hard_class_map =
            deeplab_argmax_foreground(&raw_output.values, (w * h) as usize, classes)?;
        // Keep the mask binary through restoration.  Any softening belongs to
        // an explicit alpha refiner, never to this segmenter.
        let restored = restore_carvekit_soft_mask(
            &hard_class_map,
            w,
            h,
            image.width(),
            image.height(),
            image::imageops::FilterType::Nearest,
        )?;
        Ok(DeepLabV3RunEvidence {
            tensor,
            raw_output,
            hard_class_map,
            restored,
        })
    }
    pub fn predict(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<bgremove_core::AlphaMask> {
        Ok(self.predict_with_evidence(image)?.restored)
    }
    pub fn provider(&self) -> ProviderReport {
        self.pool.provider_report()
    }
}
impl bgremove_core::Segmenter for DeepLabV3Segmenter {
    fn predict(
        &mut self,
        image: &bgremove_core::CanonicalImage,
        _prompt: Option<&bgremove_core::Prompt>,
    ) -> Result<bgremove_core::AlphaMask> {
        DeepLabV3Segmenter::predict(self, image)
    }
}

fn validate_m6_manifest(
    manifest: &ModelManifest,
    family: &str,
    width: u32,
    height: u32,
) -> Result<()> {
    ensure!(
        manifest.algorithm_family == family,
        "M6 adapter requires algorithm_family={family}"
    );
    ensure!(
        manifest.width == width && manifest.height == height,
        "{family} input dimensions mismatch"
    );
    ensure!(manifest.scale == 1.0, "{family} scale must be exactly 1.0");
    ensure!(
        manifest.layout == bgremove_models::ModelLayout::Nchw
            && manifest.channel_order == bgremove_models::ChannelOrder::Rgb,
        "{family} requires RGB NCHW"
    );
    let expected_filter = match family {
        "basnet" | "deeplabv3" => bgremove_models::ResizeFilter::Bicubic,
        "tracer-b7" => bgremove_models::ResizeFilter::Bilinear,
        _ => unreachable!(),
    };
    ensure!(
        manifest.resize_filter == expected_filter,
        "{family} resize filter does not match the pinned CarveKit contract"
    );
    ensure!(
        manifest.mean == [0.485, 0.456, 0.406] && manifest.std == [0.229, 0.224, 0.225],
        "{family} requires ImageNet normalization"
    );
    ensure!(
        manifest.input_type == Some(TensorElementType::F32)
            && manifest.output_type == Some(TensorElementType::F32),
        "{family} tensors must be f32"
    );
    ensure!(
        manifest.activation == Activation::None,
        "{family} output activation must be none"
    );
    match family {
        "basnet" => ensure!(
            manifest.output_index == Some(0)
                && matches!(
                    manifest.output_normalization,
                    OutputNormalization::None | OutputNormalization::MinMax
                ),
            "BASNet requires first output and none/minmax normalization"
        ),
        "deeplabv3" => {
            let labels = manifest
                .class_mapping
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("DeepLabV3 class mapping is required"))?;
            ensure!(
                labels
                    == &[
                        "background",
                        "aeroplane",
                        "bicycle",
                        "bird",
                        "boat",
                        "bottle",
                        "bus",
                        "car",
                        "cat",
                        "chair",
                        "cow",
                        "diningtable",
                        "dog",
                        "horse",
                        "motorbike",
                        "person",
                        "pottedplant",
                        "sheep",
                        "sofa",
                        "train",
                        "tvmonitor"
                    ]
                    .iter()
                    .map(|v| (*v).to_string())
                    .collect::<Vec<_>>(),
                "DeepLabV3 class mapping must be the exact 21-label VOC mapping"
            );
            ensure!(
                manifest.output_index == Some(0)
                    && manifest.output_normalization == OutputNormalization::None,
                "DeepLabV3 requires first direct output"
            );
        }
        "tracer-b7" => ensure!(
            manifest.output_index == Some(0)
                && manifest.output_normalization == OutputNormalization::None,
            "TRACER-B7 requires first direct output with no normalization"
        ),
        _ => unreachable!(),
    }
    Ok(())
}
impl U2netClothSegmenter {
    pub fn new(
        manifest: &ModelManifest,
        manifest_path: &Path,
        runtime: &Path,
        workers: usize,
        requested: RequestedProvider,
        fallback_allowed: bool,
    ) -> Result<Self> {
        ensure!(
            manifest.algorithm_family == "u2net",
            "cloth adapter requires algorithm_family=u2net"
        );
        ensure!(
            manifest.model_domain == "cloth",
            "cloth adapter requires model_domain=cloth"
        );
        ensure!(
            manifest.width == 768 && manifest.height == 768,
            "cloth model requires 768x768"
        );
        ensure!(
            manifest.layout == bgremove_models::ModelLayout::Nchw
                && manifest.channel_order == bgremove_models::ChannelOrder::Rgb,
            "cloth model requires RGB NCHW"
        );
        ensure!(
            manifest.resize_filter == bgremove_models::ResizeFilter::Lanczos3,
            "cloth input resize must be Lanczos3"
        );
        ensure!(
            manifest.output_index == Some(0),
            "cloth model must explicitly select output 0"
        );
        ensure!(
            manifest.input_shape
                == vec![
                    DimensionSpec::Dynamic("batch".into()),
                    DimensionSpec::Static(3),
                    DimensionSpec::Static(768),
                    DimensionSpec::Static(768)
                ],
            "cloth input shape metadata mismatch"
        );
        ensure!(
            manifest.output_shape
                == vec![
                    DimensionSpec::Dynamic("batch".into()),
                    DimensionSpec::Static(4),
                    DimensionSpec::Static(768),
                    DimensionSpec::Static(768)
                ],
            "cloth output shape metadata mismatch: expected background + 3 classes"
        );
        ensure!(
            manifest.input_type == Some(TensorElementType::F32)
                && manifest.output_type == Some(TensorElementType::F32),
            "cloth tensors must be f32"
        );
        ensure!(
            manifest.activation == Activation::None,
            "cloth output is direct logits; activation must be none"
        );
        ensure!(
            manifest.output_normalization == OutputNormalization::None,
            "cloth output must remain logits for argmax"
        );
        Ok(Self {
            pool: SessionPool::new(
                manifest,
                manifest_path,
                runtime,
                workers,
                requested,
                fallback_allowed,
            )?,
        })
    }
    pub fn predict_categories(
        &self,
        image: &bgremove_core::CanonicalImage,
        category: Option<ClothCategory>,
    ) -> Result<Vec<(ClothCategory, bgremove_core::AlphaMask)>> {
        let evidence = self.predict_with_evidence(image)?;
        let categories = category.into_iter().collect::<Vec<_>>();
        let categories = if categories.is_empty() {
            vec![
                ClothCategory::Upper,
                ClothCategory::Lower,
                ClothCategory::Full,
            ]
        } else {
            categories
        };
        categories
            .into_iter()
            .map(|c| {
                let data = cloth_category_mask(&evidence.restored_class_map, c);
                Ok((
                    c,
                    bgremove_core::AlphaMask::new(image.width(), image.height(), data)?,
                ))
            })
            .collect()
    }
    pub fn predict_category(
        &self,
        image: &bgremove_core::CanonicalImage,
        category: ClothCategory,
    ) -> Result<bgremove_core::AlphaMask> {
        Ok(self.predict_categories(image, Some(category))?.remove(0).1)
    }
    pub fn predict_with_evidence(
        &self,
        image: &bgremove_core::CanonicalImage,
    ) -> Result<U2netClothRunEvidence> {
        let tensor = u2net_preprocess_rgb(image, 768, 768)?;
        let mut lease = self.pool.checkout();
        let output = lease.session_mut().run(&tensor.shape, &tensor.values)?;
        ensure!(
            (output.shape == [1, 4, 768, 768] || output.shape == [-1, 4, 768, 768]),
            "cloth output shape changed: {:?}",
            output.shape
        );
        let plane = 768 * 768;
        let class_map = cloth_argmax_class_map(&output.values, 4, plane)?;
        let restored_class_map =
            resize_u8_pillow_lanczos(&class_map, 768, 768, 1, image.width(), image.height())?;
        Ok(U2netClothRunEvidence {
            tensor,
            raw_output: output,
            class_map,
            restored_class_map,
        })
    }
    pub fn provider(&self) -> ProviderReport {
        self.pool.provider_report()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    fn straight_rgba_bytes_for_test(
        image: &bgremove_core::CanonicalImage,
        alpha: bgremove_core::AlphaMask,
    ) -> Vec<u8> {
        let cutout = isnet_straight_cutout(image, alpha).unwrap();
        cutout
            .rgb()
            .data()
            .iter()
            .zip(cutout.alpha().data())
            .flat_map(|(rgb, alpha)| {
                [
                    (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
                ]
            })
            .collect()
    }
    #[test]
    fn constant_minmax_is_finite() {
        let t = apply_output_transform(
            TensorOutput {
                shape: vec![1, 1],
                values: vec![2.0, 2.0],
            },
            Activation::None,
            OutputNormalization::MinMax,
        )
        .unwrap();
        assert_eq!(t.values, vec![0.0, 0.0]);
    }

    #[test]
    fn m7_sigmoid_precedes_selected_output_normalization() {
        let logits = TensorOutput {
            shape: vec![1, 1, 1, 3],
            values: vec![-2.0, 0.0, 2.0],
        };
        let clamp = apply_output_transform(
            logits.clone(),
            Activation::Sigmoid,
            OutputNormalization::Clamp,
        )
        .unwrap();
        let expected = [
            1.0 / (1.0 + 2.0f32.exp()),
            0.5,
            1.0 / (1.0 + (-2.0f32).exp()),
        ];
        for (actual, expected) in clamp.values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let minmax =
            apply_output_transform(logits, Activation::Sigmoid, OutputNormalization::MinMax)
                .unwrap();
        assert!((minmax.values[0] - 0.0).abs() < 1e-6);
        assert!((minmax.values[1] - 0.5).abs() < 1e-6);
        assert!((minmax.values[2] - 1.0).abs() < 1e-6);
        let wrong_order = apply_output_transform(
            TensorOutput {
                shape: vec![1, 1, 1, 3],
                values: vec![-2.0, 0.0, 2.0],
            },
            Activation::None,
            OutputNormalization::MinMax,
        )
        .and_then(|tensor| {
            apply_output_transform(tensor, Activation::Sigmoid, OutputNormalization::None)
        })
        .unwrap();
        assert!((wrong_order.values[1] - 0.62245935).abs() < 1e-6);
        assert!((minmax.values[1] - wrong_order.values[1]).abs() > 0.1);
    }

    #[test]
    fn m7_preprocess_uses_global_max_and_imagenet_nchw() {
        let image = bgremove_core::CanonicalImage::new(1, 1, vec![[1.0, 0.5, 0.0]]).unwrap();
        let tensor = birefnet_preprocess_rgb(&image).unwrap();
        assert_eq!(tensor.shape, vec![1, 3, 1024, 1024]);
        let plane = 1024 * 1024;
        assert!((tensor.values[0] - (1.0 - 0.485) / 0.229).abs() < 1e-6);
        let green = 128.0 / 255.0;
        assert!((tensor.values[plane] - (green - 0.456) / 0.224).abs() < 1e-6);
        assert!((tensor.values[2 * plane] - (0.0 - 0.406) / 0.225).abs() < 1e-6);
        assert!(tensor.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn m7_restore_returns_canonical_dimensions_and_finite_values() {
        let values = vec![0.5f32; 1024 * 1024];
        let restored = restore_birefnet_mask(&values, 5, 3).unwrap();
        assert_eq!(restored.dimensions(), (5, 3));
        assert!(restored.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    #[ignore = "requires ORT_DYLIB and the externally supplied, hash-verified general/lite checkpoints"]
    fn m7_python_level2_fixture_matches_every_stage_for_general_and_lite() {
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let fixture_root = Path::new("../../tests/fixtures/m7/reference");
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture_root.join("report.json")).unwrap())
                .unwrap();
        assert_eq!(report["schema"], "m7.rembg-birefnet-python-level2.v1");
        assert_eq!(report["provenance"]["repository"], "projects/python/rembg");
        assert_eq!(
            report["provenance"]["commit"],
            "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
        );
        assert_eq!(
            report["provenance"]["dependencies"]["onnxruntime"],
            "1.23.2"
        );
        assert_eq!(report["provenance"]["dependencies"]["Pillow"], "10.4.0");
        assert_eq!(report["provenance"]["dependencies"]["numpy"], "1.26.4");
        assert_eq!(
            report["provenance"]["weight_license"]["path"],
            "models/M7_BIREFNET_WEIGHT_LICENSE.txt"
        );
        assert_eq!(
            report["provenance"]["weight_license"]["identifier"],
            "MIT (BiRefNet upstream)"
        );
        assert_eq!(
            report["provenance"]["weight_license"]["sha256"],
            "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"
        );
        assert_eq!(
            report["provenance"]["weight_license"]["upstream_repository"],
            "https://github.com/ZhengPeng7/BiRefNet"
        );
        assert_eq!(
            report["provenance"]["weight_license"]["upstream_commit"],
            "ebcc0bc8ec7fe919cec829f2dea656b3078acddc"
        );
        assert_eq!(
            report["provenance"]["weight_license"]["upstream_url"],
            "https://raw.githubusercontent.com/ZhengPeng7/BiRefNet/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/LICENSE"
        );
        let tensor_tolerance = report["tolerances"]["preprocessed_tensor_max_abs"]
            .as_f64()
            .unwrap() as f32;
        let raw_tolerance = report["tolerances"]["raw_output_max_abs"].as_f64().unwrap() as f32;
        let alpha_tolerance = report["tolerances"]["restored_alpha_max_abs"]
            .as_f64()
            .unwrap() as f32;
        let hash_file = |path: &Path| {
            let mut digest = sha2::Sha256::new();
            digest.update(std::fs::read(path).unwrap());
            digest
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        for (path, expected) in [
            (
                "../../projects/python/rembg/rembg/sessions/base.py",
                "ec4c58b33dd47ad6f03883ee375353314b19d5688fffd5c1ddb57bb21e9846a3",
            ),
            (
                "../../projects/python/rembg/rembg/sessions/birefnet_general.py",
                "e985eeb1ec72df63be6992aa3104255b225fa736d601b5bcc77cc6316c810698",
            ),
            (
                "../../models/M7_BIREFNET_WEIGHT_LICENSE.txt",
                "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb",
            ),
            (
                "../../tests/fixtures/m5/landscape-3x2.png",
                "d46290be343fb0c7c8ada13c51514738051632a7e80c9c0bd59e48210f814471",
            ),
        ] {
            assert_eq!(
                hash_file(Path::new(path)),
                expected,
                "fixture provenance hash {path}"
            );
        }
        let read_f32 = |path: &Path| {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        for (variant, manifest_path) in [
            ("general", "../../models/m7_birefnet_general.toml"),
            ("general-lite", "../../models/m7_birefnet_general_lite.toml"),
        ] {
            let dir = fixture_root.join(variant).join("landscape-3x2");
            let case = &report["cases"][variant];
            let expected_file_hashes = [
                ("decoded_rgb", "decoded-rgb.f32le"),
                ("preprocessed_tensor", "preprocessed-tensor.f32le"),
                ("raw_output", "raw-output.f32le"),
                ("restored_alpha", "restored-alpha.f32le"),
                ("final_cutout_file", "final-straight-alpha-cutout.rgba"),
            ];
            for (field, filename) in expected_file_hashes {
                let path = dir.join(filename);
                let expected = case["files"][field].as_str().unwrap();
                assert_eq!(hash_file(&path), expected, "fixture artifact hash {path:?}");
            }
            let decoded = read_f32(&dir.join("decoded-rgb.f32le"));
            let image = bgremove_core::CanonicalImage::new(
                3,
                2,
                decoded
                    .chunks_exact(3)
                    .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                    .collect(),
            )
            .unwrap();
            let manifest =
                bgremove_models::parse_toml(&std::fs::read_to_string(manifest_path).unwrap())
                    .unwrap();
            let segmenter = BirefnetSegmenter::new(
                &manifest,
                Path::new(manifest_path),
                Path::new(&runtime),
                1,
                RequestedProvider::Cpu,
                false,
            )
            .unwrap();
            let evidence = segmenter.predict_with_evidence(&image).unwrap();
            let expected_tensor = read_f32(&dir.join("preprocessed-tensor.f32le"));
            let expected_raw = read_f32(&dir.join("raw-output.f32le"));
            let expected_alpha = read_f32(&dir.join("restored-alpha.f32le"));
            assert_eq!(evidence.tensor.values.len(), expected_tensor.len());
            assert_eq!(evidence.raw_output.values.len(), expected_raw.len());
            assert_eq!(evidence.restored.data().len(), expected_alpha.len());
            let max_delta = |actual: &[f32], expected: &[f32]| {
                actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            };
            assert!(max_delta(&evidence.tensor.values, &expected_tensor) <= tensor_tolerance);
            let raw_delta = max_delta(&evidence.raw_output.values, &expected_raw);
            assert!(
                raw_delta <= raw_tolerance,
                "raw output max delta for {variant}: {raw_delta}"
            );
            assert!(max_delta(evidence.restored.data(), &expected_alpha) <= alpha_tolerance);
            let expected_cutout =
                std::fs::read(dir.join("final-straight-alpha-cutout.rgba")).unwrap();
            assert_eq!(
                straight_rgba_bytes_for_test(&image, evidence.restored),
                expected_cutout
            );
        }
    }

    #[test]
    #[ignore = "real M7 CPU ORT gate; requires all seven hash-verified external checkpoints"]
    fn m7_real_cpu_registry_shapes_and_finite_output() {
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB required");
        let image =
            bgremove_core::io::load_canonical(Path::new("../../test_images/reference/1.png"))
                .unwrap();
        for manifest_path in [
            "../../models/m7_birefnet_general.toml",
            "../../models/m7_birefnet_general_lite.toml",
            "../../models/m7_birefnet_portrait.toml",
            "../../models/m7_birefnet_dis.toml",
            "../../models/m7_birefnet_hrsod.toml",
            "../../models/m7_birefnet_cod.toml",
            "../../models/m7_birefnet_massive.toml",
        ] {
            let manifest =
                bgremove_models::parse_toml(&std::fs::read_to_string(manifest_path).unwrap())
                    .unwrap();
            let segmenter = BirefnetSegmenter::new(
                &manifest,
                Path::new(manifest_path),
                Path::new(&runtime),
                1,
                RequestedProvider::Cpu,
                false,
            )
            .unwrap();
            let evidence = segmenter.predict_with_evidence(&image).unwrap();
            assert_eq!(evidence.tensor.shape, vec![1, 3, 1024, 1024]);
            assert_eq!(evidence.raw_output.shape, vec![1, 1, 1024, 1024]);
            assert!(evidence
                .raw_output
                .values
                .iter()
                .all(|value| value.is_finite()));
            assert!(evidence
                .transformed_output
                .values
                .iter()
                .all(|value| value.is_finite()));
            assert_eq!(evidence.restored.dimensions(), image.dimensions());
            assert!(evidence
                .restored
                .data()
                .iter()
                .all(|value| value.is_finite()));
        }
    }

    #[test]
    fn m6_deeplab_argmax_is_hard_and_background_ties_win() {
        let logits = vec![
            1.0, 0.0, 0.0, // background
            1.0, 2.0, 0.0, // class 1
            0.0, 0.0, 3.0, // class 2
        ];
        let mask = deeplab_argmax_foreground(&logits, 3, 3).unwrap();
        assert_eq!(mask, vec![0.0, 1.0, 1.0]);
        assert!(mask.iter().all(|value| *value == 0.0 || *value == 1.0));
        assert!(deeplab_argmax_foreground(&[f32::NAN; 6], 3, 2).is_err());
    }

    #[test]
    fn m6_carvekit_geometry_is_family_specific_and_finite() {
        let image = bgremove_core::CanonicalImage::new(5, 3, vec![[0.0, 0.1, 0.2]; 15]).unwrap();
        let basnet = carvekit_imagenet_preprocess_rgb(&image, "basnet").unwrap();
        assert_eq!(basnet.shape, vec![1, 3, 320, 320]);
        let tracer = carvekit_imagenet_preprocess_rgb(&image, "tracer-b7").unwrap();
        assert_eq!(tracer.shape, vec![1, 3, 640, 640]);
        let deeplab = carvekit_imagenet_preprocess_rgb(&image, "deeplabv3").unwrap();
        assert_eq!(deeplab.shape, vec![1, 3, 3, 5]);
        assert!(basnet
            .values
            .iter()
            .chain(&tracer.values)
            .chain(&deeplab.values)
            .all(|v| v.is_finite()));
    }

    #[test]
    fn m6_manifest_contract_rejects_family_tampering_before_runtime() {
        let mut bas = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_basnet.toml").unwrap(),
        )
        .unwrap();
        bas.resize_filter = bgremove_models::ResizeFilter::Bilinear;
        assert!(validate_m6_manifest(&bas, "basnet", 320, 320).is_err());
        let mut deep = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_deeplabv3.toml").unwrap(),
        )
        .unwrap();
        deep.class_mapping.as_mut().unwrap().pop();
        assert!(validate_m6_manifest(&deep, "deeplabv3", 1024, 1024).is_err());
        let mut tracer = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_tracer_b7.toml").unwrap(),
        )
        .unwrap();
        tracer.output_normalization = OutputNormalization::MinMax;
        assert!(validate_m6_manifest(&tracer, "tracer-b7", 640, 640).is_err());
        tracer.output_normalization = OutputNormalization::None;
        tracer.output_index = Some(1);
        assert!(validate_m6_manifest(&tracer, "tracer-b7", 640, 640).is_err());
    }

    #[test]
    fn m6_deeplab_restore_nearest_preserves_hard_mask() {
        let mask = restore_carvekit_soft_mask(
            &[0.0, 1.0, 1.0, 0.0],
            2,
            2,
            5,
            3,
            image::imageops::FilterType::Nearest,
        )
        .unwrap();
        assert!(mask
            .data()
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0));
    }

    #[test]
    fn m6_python_level2_report_covers_three_geometries_and_two_colour_ranges() {
        let root = Path::new("../../tests/fixtures/m6");
        let report: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("python-onnx-parity.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report["schema"], "m6.carvekit-python-level2.v1");
        assert_eq!(report["verdict"], true);
        assert_eq!(
            report["provenance"]["carvekit_commit"],
            "f141a311af67fb1da64269c508a6d1f786420801"
        );
        assert_eq!(report["provenance"]["tracked_source_clean"], true);
        for wrapper in [
            "carvekit/ml/wrap/basnet.py",
            "carvekit/ml/wrap/deeplab_v3.py",
            "carvekit/ml/wrap/tracer_b7.py",
        ] {
            assert_eq!(
                report["provenance"]["source_file_sha256"][wrapper]
                    .as_str()
                    .unwrap()
                    .len(),
                64
            );
        }
        for family in ["basnet", "deeplabv3", "tracer-b7"] {
            let checkpoint = report["provenance"]["checkpoints"][family]["path"]
                .as_str()
                .unwrap();
            assert!(checkpoint.starts_with("checkpoints/"));
            assert!(!checkpoint.starts_with('/'));
            let onnx = report["provenance"]["onnx"][family]["path"]
                .as_str()
                .unwrap();
            assert!(onnx.starts_with("projects/python/image-background-remove-tool/m6-onnx/"));
            assert!(!onnx.starts_with('/'));
        }
        for key in [
            "onnx_raw_max_abs",
            "onnx_raw_mean_abs",
            "rust_tensor_mean_abs",
            "rust_raw_max_abs",
            "rust_raw_mean_abs",
            "rust_restored_max_abs",
            "rust_restored_mean_abs",
        ] {
            assert!(report["tolerances"][key]["value"]
                .as_f64()
                .unwrap()
                .is_finite());
            assert!(report["tolerances"][key]["value"].as_f64().unwrap() >= 0.0);
        }
        assert_eq!(
            report["tolerances"]["final_cutout"]["mode"],
            "rgba-byte-tolerance"
        );
        assert!(
            report["tolerances"]["final_cutout"]["max_abs"]
                .as_u64()
                .unwrap()
                <= 1
        );
        assert!(
            report["tolerances"]["final_cutout"]["mean_abs"]
                .as_f64()
                .unwrap()
                <= 0.1
        );
        let records = report["records"].as_array().unwrap();
        assert_eq!(records.len(), 18);
        let artifact_names = [
            "decoded-rgb.f32le",
            "preprocessed-tensor.f32le",
            "raw-output.f32le",
            "restored-alpha.f32le",
            "final-straight-alpha-cutout.rgba",
            "final-straight-alpha-cutout.png",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>();
        let mut expected_dirs = std::collections::BTreeSet::new();
        let hash_file = |path: &Path| {
            let mut digest = sha2::Sha256::new();
            digest.update(std::fs::read(path).unwrap());
            digest
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        for record in records {
            let family = record["family"].as_str().unwrap();
            let case = record["case"].as_str().unwrap();
            let dir = root.join("reference").join(family).join(case);
            assert!(dir.is_dir(), "missing M6 fixture directory {dir:?}");
            expected_dirs.insert(format!("{family}/{case}"));
            let artifacts = record["artifacts"].as_object().unwrap();
            assert_eq!(
                artifacts
                    .keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>(),
                artifact_names
            );
            let actual_files = std::fs::read_dir(&dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(
                actual_files, artifact_names,
                "M6 artifact inventory drift in {dir:?}"
            );
            for (name, expected_hash) in artifacts {
                let path = dir.join(name);
                assert!(path.is_file(), "missing M6 fixture artifact {path:?}");
                assert_eq!(
                    hash_file(&path),
                    expected_hash.as_str().unwrap(),
                    "stale M6 artifact {path:?}"
                );
            }
        }
        let mut actual_dirs = std::collections::BTreeSet::new();
        for family in ["basnet", "deeplabv3", "tracer-b7"] {
            for entry in std::fs::read_dir(root.join("reference").join(family)).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    actual_dirs.insert(format!("{family}/{}", entry.file_name().to_string_lossy()));
                }
            }
        }
        assert_eq!(actual_dirs, expected_dirs, "M6 fixture inventory drift");
        for family in ["basnet", "deeplabv3", "tracer-b7"] {
            let selected = records
                .iter()
                .filter(|record| record["family"] == family)
                .collect::<Vec<_>>();
            assert_eq!(selected.len(), 6);
            assert!(selected.iter().all(|record| record["verdict"] == true));
            assert_eq!(
                selected
                    .iter()
                    .map(|record| record["geometry"].to_string())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                3
            );
            assert_eq!(
                selected
                    .iter()
                    .map(|record| record["colour_range"].as_str().unwrap())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                2
            );
        }
    }

    #[test]
    #[ignore = "real M6 CPU ORT gate; run explicitly with ORT_DYLIB and external ONNX weights"]
    fn m6_real_cpu_registry_shapes_and_hard_mask() {
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB required");
        let image =
            bgremove_core::io::load_canonical(Path::new("../../test_images/reference/1.png"))
                .unwrap();
        let requested = RequestedProvider::Cpu;
        let bas_manifest = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_basnet.toml").unwrap(),
        )
        .unwrap();
        let bas = BasnetSegmenter::new(
            &bas_manifest,
            Path::new("../../models/m6_basnet.toml"),
            Path::new(&runtime),
            1,
            requested,
            false,
        )
        .unwrap();
        let bas_alpha = bas.predict(&image).unwrap();
        assert_eq!(bas_alpha.dimensions(), image.dimensions());
        let deep_manifest = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_deeplabv3.toml").unwrap(),
        )
        .unwrap();
        let deep = DeepLabV3Segmenter::new(
            &deep_manifest,
            Path::new("../../models/m6_deeplabv3.toml"),
            Path::new(&runtime),
            1,
            requested,
            false,
        )
        .unwrap();
        let deep_alpha = deep.predict(&image).unwrap();
        assert!(deep_alpha.data().iter().all(|v| *v == 0.0 || *v == 1.0));
        let tracer_manifest = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_tracer_b7.toml").unwrap(),
        )
        .unwrap();
        let tracer = TracerB7Segmenter::new(
            &tracer_manifest,
            Path::new("../../models/m6_tracer_b7.toml"),
            Path::new(&runtime),
            1,
            requested,
            false,
        )
        .unwrap();
        let tracer_alpha = tracer.predict(&image).unwrap();
        assert_eq!(tracer_alpha.dimensions(), image.dimensions());
    }

    #[test]
    #[ignore = "real M6 level-2 Rust/CarveKit parity; run explicitly with ORT_DYLIB and external ONNX weights"]
    fn m6_real_level2_rust_matches_carvekit_fixtures() {
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB required");
        let root = Path::new("../../tests/fixtures/m6");
        let parity_report: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("python-onnx-parity.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(parity_report["verdict"], true);
        let tolerance = |path: &[&str]| -> f32 {
            let mut value = &parity_report["tolerances"];
            for key in path {
                value = &value[*key];
            }
            value["value"].as_f64().unwrap() as f32
        };
        let read_f32 = |path: &Path| -> Vec<f32> {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect()
        };
        let image_for = |family: &str, case: &str| {
            let geometry = case.split('-').next().unwrap();
            let (width, height) = geometry.split_once('x').unwrap();
            let width = width.parse::<u32>().unwrap();
            let height = height.parse::<u32>().unwrap();
            let values =
                read_f32(&root.join(format!("reference/{family}/{case}/decoded-rgb.f32le")));
            bgremove_core::CanonicalImage::new(
                width,
                height,
                values
                    .chunks_exact(3)
                    .map(|px| [px[0], px[1], px[2]])
                    .collect(),
            )
            .unwrap()
        };
        let bas_manifest = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_basnet.toml").unwrap(),
        )
        .unwrap();
        let deep_manifest = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_deeplabv3.toml").unwrap(),
        )
        .unwrap();
        let tracer_manifest = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m6_tracer_b7.toml").unwrap(),
        )
        .unwrap();
        let bas = BasnetSegmenter::new(
            &bas_manifest,
            Path::new("../../models/m6_basnet.toml"),
            Path::new(&runtime),
            1,
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        let deep = DeepLabV3Segmenter::new(
            &deep_manifest,
            Path::new("../../models/m6_deeplabv3.toml"),
            Path::new(&runtime),
            1,
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        let tracer = TracerB7Segmenter::new(
            &tracer_manifest,
            Path::new("../../models/m6_tracer_b7.toml"),
            Path::new(&runtime),
            1,
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        for family in ["basnet", "deeplabv3", "tracer-b7"] {
            for geometry in ["3x2", "2x3", "1025x3"] {
                for range in ["low", "high"] {
                    let case = format!("{geometry}-{range}");
                    let image = image_for(family, &case);
                    let (tensor, raw, restored) = match family {
                        "basnet" => {
                            let e = bas.predict_with_evidence(&image).unwrap();
                            (e.tensor.values, e.raw_output.values, e.restored)
                        }
                        "deeplabv3" => {
                            let e = deep.predict_with_evidence(&image).unwrap();
                            (e.tensor.values, e.raw_output.values, e.restored)
                        }
                        "tracer-b7" => {
                            let e = tracer.predict_with_evidence(&image).unwrap();
                            (e.tensor.values, e.raw_output.values, e.restored)
                        }
                        _ => unreachable!(),
                    };
                    let dir = root.join(format!("reference/{family}/{case}"));
                    let expected_tensor = read_f32(&dir.join("preprocessed-tensor.f32le"));
                    let expected_raw = read_f32(&dir.join("raw-output.f32le"));
                    let expected_alpha = read_f32(&dir.join("restored-alpha.f32le"));
                    assert_eq!(
                        tensor.len(),
                        expected_tensor.len(),
                        "{family} {case} tensor lengths"
                    );
                    assert_eq!(raw.len(), expected_raw.len(), "{family} {case} raw lengths");
                    assert_eq!(
                        restored.data().len(),
                        expected_alpha.len(),
                        "{family} {case} alpha lengths"
                    );
                    let max_delta = |left: &[f32], right: &[f32]| {
                        left.iter()
                            .zip(right)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max)
                    };
                    let tensor_delta = max_delta(&tensor, &expected_tensor);
                    let tensor_mean = tensor
                        .iter()
                        .zip(&expected_tensor)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>()
                        / tensor.len() as f32;
                    let tensor_tolerance = tolerance(&["rust_tensor_max_abs", family]);
                    assert!(
                        tensor_delta <= tensor_tolerance,
                        "{family} {case} tensor parity max_delta={tensor_delta} tolerance={tensor_tolerance}"
                    );
                    assert!(
                        tensor_mean <= tolerance(&["rust_tensor_mean_abs"]),
                        "{family} {case} tensor parity mean_delta={tensor_mean}"
                    );
                    let raw_delta = max_delta(&raw, &expected_raw);
                    let raw_mean = raw
                        .iter()
                        .zip(&expected_raw)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>()
                        / raw.len() as f32;
                    assert!(
                        raw_delta <= tolerance(&["rust_raw_max_abs"]),
                        "{family} {case} raw parity max_delta={raw_delta}"
                    );
                    assert!(
                        raw_mean <= tolerance(&["rust_raw_mean_abs"]),
                        "{family} {case} raw parity mean_delta={raw_mean}"
                    );
                    assert!(
                        max_delta(restored.data(), &expected_alpha)
                            <= tolerance(&["rust_restored_max_abs"]),
                        "{family} {case} restore max parity"
                    );
                    let restored_mean = restored
                        .data()
                        .iter()
                        .zip(&expected_alpha)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>()
                        / restored.data().len() as f32;
                    assert!(
                        restored_mean <= tolerance(&["rust_restored_mean_abs"]),
                        "{family} {case} restore mean parity"
                    );
                    let expected_dir = root.join(format!("reference/{family}/{case}"));
                    let actual_rgba = straight_rgba_bytes_for_test(&image, restored);
                    let expected_rgba =
                        std::fs::read(expected_dir.join("final-straight-alpha-cutout.rgba"))
                            .unwrap();
                    assert_eq!(
                        actual_rgba.len(),
                        expected_rgba.len(),
                        "{family} {case} final RGBA length"
                    );
                    let mut alpha_max = 0u8;
                    let mut alpha_sum = 0usize;
                    for (index, (actual, expected)) in
                        actual_rgba.iter().zip(&expected_rgba).enumerate()
                    {
                        if index % 4 != 3 {
                            assert_eq!(
                                actual, expected,
                                "{family} {case} final RGB parity at {index}"
                            );
                        } else {
                            let delta = actual.abs_diff(*expected);
                            alpha_max = alpha_max.max(delta);
                            alpha_sum += usize::from(delta);
                        }
                    }
                    assert!(
                        alpha_max
                            <= parity_report["tolerances"]["final_cutout"]["max_abs"]
                                .as_u64()
                                .unwrap() as u8
                    );
                    assert!(
                        (alpha_sum as f64 / (actual_rgba.len() / 4) as f64)
                            <= parity_report["tolerances"]["final_cutout"]["mean_abs"]
                                .as_f64()
                                .unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn m5_u2net_preprocess_is_nchw_and_constant_safe() {
        let image =
            bgremove_core::CanonicalImage::new(2, 1, vec![[0.0, 0.25, 1.0], [1.0, 1.0, 1.0]])
                .unwrap();
        let tensor = u2net_preprocess_rgb(&image, 4, 4).unwrap();
        assert_eq!(tensor.shape, vec![1, 3, 4, 4]);
        assert!(tensor.values.iter().all(|v| v.is_finite()));
        let constant = bgremove_core::CanonicalImage::new(1, 1, vec![[0.0, 0.0, 0.0]]).unwrap();
        let tensor = u2net_preprocess_rgb(&constant, 4, 4).unwrap();
        assert!(tensor.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn m5_u2net_restore_constant_and_nonfinite_are_safe() {
        let restored = restore_u2net_mask(&[3.0; 16], 3, 2).unwrap();
        assert_eq!(restored.dimensions(), (3, 2));
        assert!(restored.data().iter().all(|v| *v == 0.0));
        assert!(restore_u2net_mask(&[f32::NAN; 16], 3, 2).is_err());
        let clamped = restore_u2net_mask_values(&[0.25; 16], false, 3, 2).unwrap();
        assert!(clamped
            .data()
            .iter()
            .all(|v| (*v - 63.0 / 255.0).abs() < f32::EPSILON));
        let near_constant = restore_u2net_mask(&[0.0, 1.0e-8, 0.0, 1.0e-8], 2, 2).unwrap();
        assert!(near_constant.data().iter().any(|v| *v > 0.0));
    }

    #[test]
    fn m5_cloth_category_policy_is_explicit() {
        assert_eq!(ClothCategory::parse("upper").unwrap().as_str(), "upper");
        assert_eq!(ClothCategory::parse("lower").unwrap().class(), 2);
        assert_eq!(ClothCategory::parse("full").unwrap().class(), 3);
        assert!(ClothCategory::parse("all").is_err());
    }

    #[test]
    fn m5_cloth_argmax_and_category_masks_cover_all_classes_and_ties() {
        // Class-major logits for five pixels: background, upper, lower, full,
        // then an upper/lower tie. Exact ties must select the first class.
        let logits = vec![
            4.0, 0.0, 0.0, 0.0, 2.0, // background
            1.0, 5.0, 0.0, 0.0, 7.0, // upper
            1.0, 0.0, 6.0, 0.0, 7.0, // lower
            1.0, 0.0, 0.0, 8.0, 1.0, // full
        ];
        let classes = cloth_argmax_class_map(&logits, 4, 5).unwrap();
        assert_eq!(classes, vec![0, 1, 2, 3, 1]);
        assert_eq!(
            cloth_category_mask(&classes, ClothCategory::Upper),
            vec![0.0, 1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            cloth_category_mask(&classes, ClothCategory::Lower),
            vec![0.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            cloth_category_mask(&classes, ClothCategory::Full),
            vec![0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            cloth_category_mask_u8(&classes, ClothCategory::Upper),
            vec![0, 255, 0, 0, 255]
        );
        assert_eq!(
            cloth_category_mask_u8(&classes, ClothCategory::Lower),
            vec![0, 0, 255, 0, 0]
        );
        assert_eq!(
            cloth_category_mask_u8(&classes, ClothCategory::Full),
            vec![0, 0, 0, 255, 0]
        );
        assert!(cloth_argmax_class_map(&logits[..19], 4, 5).is_err());
        assert!(cloth_argmax_class_map(&[f32::NAN; 4], 4, 1).is_err());
    }

    #[test]
    fn m5_python_level2_reference_files_are_tracked_and_preprocess_matches() {
        let root = Path::new("../../tests/fixtures/m5");
        for path in [
            "python-ort-reference/light/landscape-3x2/preprocessed-tensor.f32le",
            "python-ort-reference/light/landscape-3x2/raw-output.f32le",
            "python-ort-reference/light/landscape-3x2/restored-alpha.f32le",
            "python-ort-reference/light/landscape-3x2/final-straight-alpha-cutout.rgba",
        ] {
            assert!(
                root.join(path).is_file(),
                "missing tracked Python artifact {path}"
            );
        }
        for name in ["landscape-3x2", "portrait-2x3", "odd-5x3"] {
            let image =
                bgremove_core::io::load_canonical(&root.join(format!("{name}.png"))).unwrap();
            let tensor = u2net_preprocess_rgb(&image, 320, 320).unwrap();
            let expected = std::fs::read(root.join(format!("{name}.tensor.f32le"))).unwrap();
            let expected = expected
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let differences = tensor
                .values
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .collect::<Vec<_>>();
            let max_abs = differences.iter().copied().fold(0.0f32, f32::max);
            let mean_abs = differences.iter().sum::<f32>() / differences.len() as f32;
            assert!(
                max_abs <= 1e-6,
                "{name} tensor max parity failed: {max_abs}"
            );
            assert!(
                mean_abs <= 1e-7,
                "{name} tensor mean parity failed: {mean_abs}"
            );
        }
    }

    #[test]
    fn m5_python_parity_report_requires_light_and_cloth_geometry_passes() {
        let report: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("../../tests/fixtures/m5/python-ort-parity.json").unwrap(),
        )
        .unwrap();
        for domain in ["light", "cloth"] {
            for geometry in ["landscape-3x2", "portrait-2x3", "odd-5x3"] {
                assert_eq!(
                    report["models"][domain]["records"][geometry]["rust_parity"]["verdict"], "pass",
                    "tracked {domain}/{geometry} parity is not pass"
                );
            }
        }
    }

    #[test]
    #[ignore = "real CPU ORT level-2 parity gate; run explicitly with ORT_DYLIB"]
    fn m5_python_level2_reference_matches_real_rust_ort() {
        let root = Path::new("../../tests/fixtures/m5");
        let manifest_path = Path::new("../../models/m5_u2netp.toml");
        let manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let segmenter = U2netSegmenter::new(
            &manifest,
            manifest_path,
            Path::new(&runtime),
            1,
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        let image = bgremove_core::io::load_canonical(&root.join("landscape-3x2.png")).unwrap();
        let evidence = segmenter.predict_with_evidence(&image).unwrap();
        let read_f32 = |name: &str| {
            std::fs::read(
                root.join("python-ort-reference/light/landscape-3x2")
                    .join(name),
            )
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
        };
        let expected_raw = read_f32("raw-output.f32le");
        let raw_max = evidence
            .raw_output
            .values
            .iter()
            .zip(expected_raw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(raw_max <= 1e-5, "raw level-2 parity failed: {raw_max}");
        let expected_alpha = read_f32("restored-alpha.f32le");
        let alpha_max = evidence
            .restored
            .data()
            .iter()
            .zip(expected_alpha)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(alpha_max, 0.0);
        let cutout = isnet_straight_cutout(&image, evidence.restored).unwrap();
        let actual = cutout
            .rgb()
            .data()
            .iter()
            .zip(cutout.alpha().data())
            .flat_map(|(rgb, alpha)| {
                [
                    (rgb[0] * 255.0).round() as u8,
                    (rgb[1] * 255.0).round() as u8,
                    (rgb[2] * 255.0).round() as u8,
                    (alpha * 255.0).round() as u8,
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            std::fs::read(
                root.join(
                    "python-ort-reference/light/landscape-3x2/final-straight-alpha-cutout.rgba"
                )
            )
            .unwrap()
        );
    }

    #[test]
    #[ignore = "real CPU cloth ORT level-2 parity gate; run explicitly with ORT_DYLIB"]
    fn m5_python_cloth_reference_matches_real_rust_ort() {
        let root = Path::new("../../tests/fixtures/m5");
        let manifest_path = Path::new("../../models/m5_u2net_cloth.toml");
        let manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let segmenter = U2netClothSegmenter::new(
            &manifest,
            manifest_path,
            Path::new(&runtime),
            1,
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        let image = bgremove_core::io::load_canonical(&root.join("landscape-3x2.png")).unwrap();
        let evidence = segmenter.predict_with_evidence(&image).unwrap();
        let base = root.join("python-ort-cloth-reference/cloth/landscape-3x2");
        let expected_raw = std::fs::read(base.join("raw-output.f32le"))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let raw_max = evidence
            .raw_output
            .values
            .iter()
            .zip(expected_raw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            raw_max <= 1e-4,
            "cloth raw level-2 parity failed: {raw_max}"
        );
        assert_eq!(
            evidence.restored_class_map,
            std::fs::read(base.join("restored-class-map.u8")).unwrap()
        );
        for (category, mask) in segmenter.predict_categories(&image, None).unwrap() {
            let actual = mask
                .data()
                .iter()
                .map(|v| (v * 255.0).round() as u8)
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                std::fs::read(base.join(format!("{}-mask.u8", category.as_str()))).unwrap()
            );
        }
    }

    #[test]
    fn hash_failure_precedes_runtime_initialization() {
        let path = Path::new("../../models/m3_identity.toml");
        let text = std::fs::read_to_string(path).unwrap();
        let mut manifest = bgremove_models::parse_toml(&text).unwrap();
        manifest.sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        let result = VerifiedSession::open(
            &manifest,
            path,
            Path::new("/does/not/exist"),
            RequestedProvider::Cpu,
            false,
        );
        let error = match result {
            Ok(_) => panic!("hash mismatch must fail before runtime"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("hash mismatch"));
    }

    #[test]
    fn preprocessing_honours_rgb_bgr_and_layout() {
        let mut manifest = bgremove_models::parse_toml(
            &std::fs::read_to_string("../../models/m3_identity.toml").unwrap(),
        )
        .unwrap();
        let image =
            bgremove_core::CanonicalImage::new(2, 1, vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
                .unwrap();
        let nchw = preprocess_rgb(&image, &manifest).unwrap();
        assert_eq!(nchw.shape, vec![1, 3, 1, 2]);
        assert_eq!(nchw.values, vec![0.1, 0.4, 0.2, 0.5, 0.3, 0.6]);
        manifest.channel_order = bgremove_models::ChannelOrder::Bgr;
        manifest.layout = bgremove_models::ModelLayout::Nhwc;
        let nhwc = preprocess_rgb(&image, &manifest).unwrap();
        assert_eq!(nhwc.shape, vec![1, 1, 2, 3]);
        assert_eq!(nhwc.values, vec![0.3, 0.2, 0.1, 0.6, 0.5, 0.4]);
        manifest.channel_order = bgremove_models::ChannelOrder::Rgb;
        manifest.layout = bgremove_models::ModelLayout::Nchw;
        manifest.scale = 0.5;
        manifest.mean = [0.1, 0.2, 0.3];
        manifest.std = [0.5, 0.5, 0.5];
        let normalized = preprocess_rgb(&image, &manifest).unwrap();
        let expected = [-0.1, 0.2, -0.2, 0.1, -0.3, 0.0];
        assert!(normalized
            .values
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-6));
    }

    #[test]
    fn transforms_cover_sigmoid_clamp_and_nonfinite() {
        let sigmoid = apply_output_transform(
            TensorOutput {
                shape: vec![1],
                values: vec![0.0],
            },
            Activation::Sigmoid,
            OutputNormalization::None,
        )
        .unwrap();
        assert!((sigmoid.values[0] - 0.5).abs() < f32::EPSILON);
        let clamped = apply_output_transform(
            TensorOutput {
                shape: vec![2],
                values: vec![-2.0, 2.0],
            },
            Activation::None,
            OutputNormalization::Clamp,
        )
        .unwrap();
        assert_eq!(clamped.values, vec![0.0, 1.0]);
        assert!(apply_output_transform(
            TensorOutput {
                shape: vec![1],
                values: vec![f32::NAN]
            },
            Activation::None,
            OutputNormalization::None
        )
        .is_err());
    }

    #[test]
    fn alpha_mask_layouts_are_checked() {
        for shape in [
            vec![1, 1, 2, 2],
            vec![1, 2, 2, 1],
            vec![2, 2],
            vec![1, 2, 2],
        ] {
            assert!(TensorOutput {
                shape,
                values: vec![0.0; 4]
            }
            .to_alpha_mask()
            .is_ok());
        }
        assert!(TensorOutput {
            shape: vec![1, 3, 2, 2],
            values: vec![0.0; 12]
        }
        .to_alpha_mask()
        .is_err());
        assert!(TensorOutput {
            shape: vec![2, 2],
            values: vec![0.0; 3]
        }
        .to_alpha_mask()
        .is_err());
    }

    #[test]
    fn output_contract_rejects_rank_and_static_dimension_mismatch() {
        assert!(validate_output_contract("m", "mask", &[1, 1, -1, 4], &[1, 1, 2], 2).is_err());
        assert!(validate_output_contract("m", "mask", &[1, 1, -1, 4], &[1, 1, 3, 4], 12).is_ok());
        assert!(validate_output_contract("m", "mask", &[1, 1, 2, 4], &[1, 1, 3, 4], 12).is_err());
    }

    #[test]
    fn provider_reports_are_truthful_and_strict() {
        let cpu = provider_report(RequestedProvider::Cpu, false).unwrap();
        assert_eq!(cpu.attempted_chain, vec!["CPUExecutionProvider"]);
        assert!(!cpu.fallback_used);
        #[cfg(not(feature = "cuda"))]
        {
            let fallback = provider_report(RequestedProvider::Cuda, true).unwrap();
            assert_eq!(fallback.active, "CPUExecutionProvider");
            assert!(fallback.fallback_used);
            assert_eq!(fallback.attempted_chain, vec!["CPUExecutionProvider"]);
            assert!(provider_report(RequestedProvider::Cuda, false).is_err());
        }
        #[cfg(feature = "cuda")]
        assert!(provider_report(RequestedProvider::Cuda, true).is_err());
    }

    #[test]
    fn imgly_resize_matches_corner_aligned_rounding_and_borders() {
        let src = vec![0, 100, 200, 255];
        let out = resize_u8_bilinear_js(&src, 2, 2, 1, 3, 3).unwrap();
        assert_eq!(out, vec![0, 67, 100, 133, 180, 203, 200, 237, 255]);
        assert_eq!(resize_u8_bilinear_js(&src, 2, 2, 1, 1, 1).unwrap(), vec![0]);
    }

    #[test]
    fn m4_profiles_are_distinct_and_exact() {
        let image =
            bgremove_core::CanonicalImage::new(1, 1, vec![[0.0, 128.0 / 255.0, 1.0]]).unwrap();
        let a = isnet_preprocess_rgb(&image, PreprocessingProfile::ImglyIsnet).unwrap();
        let b = isnet_preprocess_rgb(&image, PreprocessingProfile::RembgDis).unwrap();
        assert_eq!(&a.values[..3], &[-0.5, -0.5, -0.5]);
        assert_eq!(&a.values[1024 * 1024..1024 * 1024 + 3], &[0.0, 0.0, 0.0]);
        assert_eq!(
            &a.values[2 * 1024 * 1024..2 * 1024 * 1024 + 3],
            &[0.49609375, 0.49609375, 0.49609375]
        );
        assert_eq!(&b.values[..3], &[-0.5, -0.5, -0.5]);
        assert_eq!(
            &b.values[1024 * 1024..1024 * 1024 + 3],
            &[0.001960814, 0.001960814, 0.001960814]
        );
        assert_eq!(
            &b.values[2 * 1024 * 1024..2 * 1024 * 1024 + 3],
            &[0.5, 0.5, 0.5]
        );
        assert!(isnet_preprocess_rgb(&image, PreprocessingProfile::Generic).is_err());
    }

    #[test]
    fn m4_tensor_hash_matches_rgba_js_reference() {
        let rgba = [
            11u8, 75, 10, 17, 64, 172, 237, 88, 117, 13, 208, 159, 170, 110, 179, 230, 223, 207,
            150, 45, 20, 48, 121, 116,
        ];
        let rgb = rgba
            .chunks_exact(4)
            .map(|p| {
                [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect();
        let image = bgremove_core::CanonicalImage::new(3, 2, rgb).unwrap();
        let tensor = isnet_preprocess_rgb(&image, PreprocessingProfile::ImglyIsnet).unwrap();
        let mut digest = sha2::Sha256::new();
        for value in tensor.values {
            digest.update(value.to_le_bytes());
        }
        let actual = digest
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            actual,
            "406348a9ead28b8ee1e72d6def88231f7779d70e658dfadd1f975f381703327f"
        );
    }

    #[test]
    #[ignore = "runtime gate; requires official ORT_DYLIB"]
    fn m4_real_fp32_cpu_parity_when_runtime_is_supplied() {
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let manifest_path = Path::new("../../models/m4_isnet_fp32.toml");
        let manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        let segmenter = IsnetSegmenter::new(
            &manifest,
            manifest_path,
            Path::new(&runtime),
            1,
            PreprocessingProfile::ImglyIsnet,
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        let rgba = [
            11u8, 75, 10, 17, 64, 172, 237, 88, 117, 13, 208, 159, 170, 110, 179, 230, 223, 207,
            150, 45, 20, 48, 121, 116,
        ];
        let image = bgremove_core::CanonicalImage::new(
            3,
            2,
            rgba.chunks_exact(4)
                .map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                    ]
                })
                .collect(),
        )
        .unwrap();
        let evidence = segmenter.predict_with_evidence(&image).unwrap();
        let hash = |values: &[f32]| {
            let mut h = sha2::Sha256::new();
            for v in values {
                h.update(v.to_le_bytes());
            }
            h.finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        assert_eq!(
            hash(&evidence.tensor.values),
            "406348a9ead28b8ee1e72d6def88231f7779d70e658dfadd1f975f381703327f"
        );
        const REFERENCE_TENSOR: &[u8] =
            include_bytes!("../../../tests/fixtures/m4/reference-landscape.tensor.f32le");
        assert_eq!(REFERENCE_TENSOR.len(), evidence.tensor.values.len() * 4);
        for (i, rust_value) in evidence.tensor.values.iter().copied().enumerate() {
            let j = i * 4;
            let js_value = f32::from_le_bytes([
                REFERENCE_TENSOR[j],
                REFERENCE_TENSOR[j + 1],
                REFERENCE_TENSOR[j + 2],
                REFERENCE_TENSOR[j + 3],
            ]);
            assert_eq!(
                rust_value.to_bits(),
                js_value.to_bits(),
                "full IMG.LY tensor parity mismatch at element {i}"
            );
        }
        eprintln!(
            "M4 Rust raw hash {} min {} max {} mean {}",
            hash(&evidence.raw_output.values),
            evidence
                .raw_output
                .values
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min),
            evidence
                .raw_output
                .values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
            evidence.raw_output.values.iter().sum::<f32>()
                / evidence.raw_output.values.len() as f32
        );
        const REFERENCE_RAW: &[u8] =
            include_bytes!("../../../tests/fixtures/m4/reference-landscape.raw.f32le");
        assert_eq!(REFERENCE_RAW.len(), evidence.raw_output.values.len() * 4);
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (i, rust_value) in evidence.raw_output.values.iter().copied().enumerate() {
            let j = i * 4;
            let js_value = f32::from_le_bytes([
                REFERENCE_RAW[j],
                REFERENCE_RAW[j + 1],
                REFERENCE_RAW[j + 2],
                REFERENCE_RAW[j + 3],
            ]);
            let delta = (rust_value - js_value).abs();
            max_abs = max_abs.max(delta);
            sum_abs += delta as f64;
        }
        let mean_abs = sum_abs / evidence.raw_output.values.len() as f64;
        assert!(
            max_abs <= 1e-6 && mean_abs <= 1e-7,
            "raw parity exceeded tolerance: max_abs={max_abs} mean_abs={mean_abs}"
        );
        if let Some(path) = std::env::var_os("M4_RUST_RAW_BIN") {
            let mut bytes = Vec::with_capacity(evidence.raw_output.values.len() * 4);
            for value in &evidence.raw_output.values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(path, bytes).unwrap();
        }
        assert_eq!(
            evidence
                .restored
                .data()
                .iter()
                .map(|v| (v * 255.0).round() as u8)
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 0, 0, 0]
        );
        assert_eq!(segmenter.provider().active, "CPUExecutionProvider");
    }

    #[test]
    #[ignore = "runtime gate; requires official ORT_DYLIB"]
    fn m4_real_fp32_geometry_parity_for_landscape_portrait_and_odd() {
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let manifest_path = Path::new("../../models/m4_isnet_fp32.toml");
        let manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        let segmenter = IsnetSegmenter::new(
            &manifest,
            manifest_path,
            Path::new(&runtime),
            1,
            PreprocessingProfile::ImglyIsnet,
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        let cases = [
            (
                3u32,
                2u32,
                11u8,
                "406348a9ead28b8ee1e72d6def88231f7779d70e658dfadd1f975f381703327f",
                vec![0, 0, 1, 0, 0, 0],
                "landscape-3x2.straight-alpha-cutout.rgba",
            ),
            (
                2,
                3,
                29,
                "1617d8c493bbda10c415987f97674a28e921929ca7fd036d7c2941a04ec3b56a",
                vec![0, 0, 0, 1, 0, 0],
                "portrait-2x3.straight-alpha-cutout.rgba",
            ),
            (
                5,
                3,
                47,
                "43e2da036968786ff8b4f0be8a35519b916da6fab7b275dedc5f4f462d078977",
                vec![0, 0, 4, 4, 5, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0],
                "odd-5x3.straight-alpha-cutout.rgba",
            ),
        ];
        for (w, h, seed, expected_tensor, expected_alpha, cutout_name) in cases {
            let rgb = (0..(w * h))
                .map(|i| {
                    let i = i as u8;
                    [
                        ((i.wrapping_mul(53)).wrapping_add(seed)) as f32 / 255.0,
                        ((i.wrapping_mul(97)).wrapping_add(64).wrapping_add(seed)) as f32 / 255.0,
                        (255u8.wrapping_sub(i.wrapping_mul(29)).wrapping_add(seed)) as f32 / 255.0,
                    ]
                })
                .collect();
            let image = bgremove_core::CanonicalImage::new(w, h, rgb).unwrap();
            let evidence = segmenter.predict_with_evidence(&image).unwrap();
            let mut digest = sha2::Sha256::new();
            for v in &evidence.tensor.values {
                digest.update(v.to_le_bytes());
            }
            let got = digest
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            assert_eq!(got, expected_tensor);
            let alpha = evidence
                .restored
                .data()
                .iter()
                .map(|v| (v * 255.0).round() as u8)
                .collect::<Vec<_>>();
            assert_eq!(alpha, expected_alpha);
            assert_eq!(evidence.restored.dimensions(), (w, h));
            let cutout = isnet_straight_cutout(&image, evidence.restored.clone()).unwrap();
            let mut cutout_bytes = Vec::with_capacity((w * h * 4) as usize);
            for (rgb, alpha) in cutout.rgb().data().iter().zip(cutout.alpha().data()) {
                cutout_bytes.extend(rgb.iter().map(|value| (value * 255.0).round() as u8));
                cutout_bytes.push((alpha * 255.0).round() as u8);
            }
            let expected = std::fs::read(format!("../../tests/fixtures/m4/{cutout_name}")).unwrap();
            assert_eq!(cutout_bytes, expected, "straight-alpha cutout mismatch");
            for (index, alpha) in alpha.iter().enumerate() {
                if *alpha == 0 {
                    let rgb = &cutout.rgb().data()[index];
                    let offset = index * 4;
                    assert_eq!(
                        &cutout_bytes[offset..offset + 3],
                        &expected[offset..offset + 3],
                        "original RGB changed at alpha-zero pixel"
                    );
                    assert_eq!(
                        rgb[0].to_bits(),
                        (expected[offset] as f32 / 255.0).to_bits()
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "runtime gate; requires official ORT_DYLIB"]
    fn m4_all_encodings_are_independently_inspected() {
        let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        for path in [
            "../../models/m4_isnet_fp32.toml",
            "../../models/m4_isnet_fp16.toml",
            "../../models/m4_isnet_quantized.toml",
        ] {
            let manifest_path = Path::new(path);
            let manifest =
                bgremove_models::parse_toml(&std::fs::read_to_string(manifest_path).unwrap())
                    .unwrap();
            let session = VerifiedSession::open(
                &manifest,
                manifest_path,
                Path::new(&runtime),
                RequestedProvider::Cpu,
                false,
            )
            .unwrap();
            assert_eq!(session.inspection.input_name, "input");
            assert_eq!(session.inspection.output_name, "output");
            assert_eq!(session.inspection.input_shape, vec![1, 3, 1024, 1024]);
            assert_eq!(session.inspection.output_shape, vec![1, 1, 1024, 1024]);
            assert_eq!(session.inspection.opset, 15);
        }
    }

    #[test]
    fn m4_restore_is_canonical_and_clamps_direct_output() {
        let raw = vec![1.2f32; 1024 * 1024];
        let mask = restore_isnet_mask(&raw, 3, 5).unwrap();
        assert_eq!(mask.dimensions(), (3, 5));
        assert!(mask.data().iter().all(|x| (*x - 1.0).abs() < f32::EPSILON));
        assert!(restore_isnet_mask(&[f32::NAN; 1024 * 1024], 3, 5).is_err());
    }

    #[test]
    fn rembg_minmax_is_safe_for_constant_and_nonfinite_outputs() {
        let constant = restore_rembg_dis_mask(&vec![4.0; 1024 * 1024], 5, 3).unwrap();
        assert!(constant.data().iter().all(|v| *v == 0.0));
        assert!(restore_rembg_dis_mask(&[f32::INFINITY; 1024 * 1024], 5, 3).is_err());
        let edge = resize_u8_lanczos(&[0, 255, 255, 0], 2, 2, 1, 3, 3).unwrap();
        assert_eq!(edge.len(), 9);
        assert_eq!(edge[0], 0);
        assert_eq!(edge[8], 0);
    }

    #[test]
    fn rembg_lanczos_matches_pinned_pillow_with_recorded_tolerance() {
        #[derive(Deserialize)]
        struct Fixture {
            source_dimensions: [u32; 2],
            source_rgb: Vec<u8>,
            resize_dimensions: [u32; 2],
            resized_rgb: Vec<u8>,
            global_max: u8,
            normalized: Vec<f64>,
            raw_mask: Vec<f64>,
            raw_min: f64,
            raw_max: f64,
            mask_u8: Vec<u8>,
            restored_u8: Vec<u8>,
            tolerances: Tolerances,
        }
        #[derive(Deserialize)]
        struct Tolerances {
            resize_u8_max_abs: f64,
            resize_u8_mean_abs: f64,
            normalized_f32_max_abs: f64,
            normalized_f32_mean_abs: f64,
            restore_u8_max_abs: f64,
            restore_u8_mean_abs: f64,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../tests/fixtures/m4/rembg-pillow-fixture.json"
        ))
        .unwrap();
        let resized = resize_u8_lanczos(
            &fixture.source_rgb,
            fixture.source_dimensions[0],
            fixture.source_dimensions[1],
            3,
            fixture.resize_dimensions[0],
            fixture.resize_dimensions[1],
        )
        .unwrap();
        let resize_errors = resized
            .iter()
            .zip(&fixture.resized_rgb)
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
            .collect::<Vec<_>>();
        let resize_max = resize_errors.iter().copied().fold(0.0, f64::max);
        let resize_mean = resize_errors.iter().sum::<f64>() / resize_errors.len() as f64;
        eprintln!("Pillow input resize compatibility: max_abs={resize_max} mean_abs={resize_mean}");
        assert!(resize_max <= fixture.tolerances.resize_u8_max_abs);
        assert!(resize_mean <= fixture.tolerances.resize_u8_mean_abs);
        assert_eq!(resized.iter().copied().max().unwrap(), fixture.global_max);

        let normalized = resized
            .iter()
            .map(|value| *value as f32 / fixture.global_max.max(1) as f32 - 0.5)
            .collect::<Vec<_>>();
        let reference_normalized = fixture
            .resized_rgb
            .iter()
            .map(|value| *value as f64 / fixture.global_max.max(1) as f64 - 0.5)
            .collect::<Vec<_>>();
        assert_eq!(normalized.len(), fixture.normalized.len());
        assert!(reference_normalized
            .iter()
            .zip(&fixture.normalized)
            .all(|(actual, expected)| (*actual - *expected).abs() <= 1e-12));
        let normalized_errors = normalized
            .iter()
            .zip(&fixture.normalized)
            .map(|(actual, expected)| (*actual as f64 - *expected).abs())
            .collect::<Vec<_>>();
        let normalized_max = normalized_errors.iter().copied().fold(0.0, f64::max);
        let normalized_mean =
            normalized_errors.iter().sum::<f64>() / normalized_errors.len() as f64;
        eprintln!(
            "Pillow rembg normalization compatibility: max_abs={normalized_max} mean_abs={normalized_mean}"
        );
        assert!(normalized_max <= fixture.tolerances.normalized_f32_max_abs);
        assert!(normalized_mean <= fixture.tolerances.normalized_f32_mean_abs);

        let restored = resize_u8_lanczos(
            &fixture.mask_u8,
            fixture.resize_dimensions[0],
            fixture.resize_dimensions[1],
            1,
            fixture.source_dimensions[0],
            fixture.source_dimensions[1],
        )
        .unwrap();
        let restore_errors = restored
            .iter()
            .zip(&fixture.restored_u8)
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
            .collect::<Vec<_>>();
        let restore_max = restore_errors.iter().copied().fold(0.0, f64::max);
        let restore_mean = restore_errors.iter().sum::<f64>() / restore_errors.len() as f64;
        eprintln!(
            "Pillow mask restore compatibility: max_abs={restore_max} mean_abs={restore_mean}"
        );
        assert!(restore_max <= fixture.tolerances.restore_u8_max_abs);
        assert!(restore_mean <= fixture.tolerances.restore_u8_mean_abs);

        let min = fixture
            .raw_mask
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = fixture
            .raw_mask
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(min, fixture.raw_min);
        assert_eq!(max, fixture.raw_max);
        let normalized_mask = fixture
            .raw_mask
            .iter()
            .map(|value| (((value - min) / (max - min)).clamp(0.0, 1.0) * 255.0).floor() as u8)
            .collect::<Vec<_>>();
        assert_eq!(normalized_mask, fixture.mask_u8);
    }

    #[test]
    fn isnet_profile_mismatch_fails_before_runtime_or_model_access() {
        let path = Path::new("../../models/m4_isnet_fp32.toml");
        let mut manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        manifest.preprocessing_profile = PreprocessingProfile::RembgDis;
        let error = match IsnetSegmenter::new(
            &manifest,
            path,
            Path::new("/missing"),
            1,
            PreprocessingProfile::ImglyIsnet,
            RequestedProvider::Cpu,
            false,
        ) {
            Ok(_) => panic!("profile mismatch must fail before runtime"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn m4_cutout_preserves_original_rgb_at_zero_alpha() {
        let image =
            bgremove_core::CanonicalImage::new(2, 1, vec![[0.9, 0.2, 0.1], [0.1, 0.8, 0.3]])
                .unwrap();
        let cutout = isnet_straight_cutout(
            &image,
            bgremove_core::AlphaMask::new(2, 1, vec![0.0, 0.5]).unwrap(),
        )
        .unwrap();
        assert_eq!(cutout.rgb().data(), image.rgb().data());
        assert_eq!(cutout.alpha().data(), &[0.0, 0.5]);
    }

    #[cfg(feature = "bria")]
    #[test]
    fn m8_profiles_are_distinct_and_finite() {
        let image = bgremove_core::CanonicalImage::new_with_alpha(
            2,
            1,
            vec![[0.0, 0.25, 1.0], [0.5, 0.75, 0.1]],
            bgremove_core::AlphaMask::new(2, 1, vec![0.0, 1.0]).unwrap(),
        )
        .unwrap();
        let rust = rmbg_rust_preprocess_rgb(&image).unwrap();
        let python = rembg_bria_preprocess_rgb(&image).unwrap();
        assert_eq!(rust.shape, vec![1, 3, 1024, 1024]);
        assert_eq!(python.shape, rust.shape);
        assert!(rust
            .values
            .iter()
            .chain(&python.values)
            .all(|v| v.is_finite()));
        assert_ne!(rust.values, python.values);
    }

    #[cfg(feature = "bria")]
    #[test]
    fn m8_constant_and_clamped_outputs_never_create_nonfinite_alpha() {
        let constant = normalize_rmbg_output(&[0.25, 0.25], OutputNormalization::MinMax).unwrap();
        assert_eq!(constant, vec![0.0, 0.0]);
        assert_eq!(
            normalize_rmbg_output(&[-1.0, 2.0], OutputNormalization::Clamp).unwrap(),
            vec![0.0, 1.0]
        );
        assert!(normalize_rmbg_output(&[f32::NAN], OutputNormalization::MinMax).is_err());
    }

    #[cfg(feature = "bria")]
    #[test]
    fn m8_manifest_normalization_profiles_diverge_on_same_raw_tensor() {
        let raw = [-1.0, 0.0, 0.25, 2.0];
        let clamp = normalize_rmbg_output(&raw, OutputNormalization::Clamp).unwrap();
        let minmax = normalize_rmbg_output(&raw, OutputNormalization::MinMax).unwrap();
        assert_eq!(clamp, [0.0, 0.0, 0.25, 1.0]);
        assert_eq!(minmax[0], 0.0);
        assert_eq!(minmax[3], 1.0);
        assert!((minmax[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((minmax[2] - (1.25 / 3.0)).abs() < 1e-6);
        assert_ne!(clamp, minmax);
        assert!(clamp
            .iter()
            .chain(minmax.iter())
            .all(|value| value.is_finite()));
    }

    #[cfg(feature = "bria")]
    #[test]
    fn m8_normalization_selects_channel_zero_before_minmax() {
        let plane = 1024 * 1024;
        let mut values = vec![0.0f32; 3 * plane];
        values[..plane].fill(0.25);
        values[1] = 0.75;
        values[plane] = -10_000.0;
        values[2 * plane] = 10_000.0;
        let output = TensorOutput {
            shape: vec![1, 3, 1024, 1024],
            values,
        };
        let channel0 = first_output_mask_channel0(&output, 1024, 1024, "m8").unwrap();
        let normalized = normalize_rmbg_output(&channel0, OutputNormalization::MinMax).unwrap();
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[1], 1.0);
    }

    #[test]
    fn m8_complete_fixture_stages_are_hash_validated() {
        let root = Path::new("../../tests/fixtures/m8/reference");
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("parity.json")).unwrap())
                .unwrap();
        let rust_report: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("../../tests/fixtures/m8/authoritative/rust-rmbg/report.json")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            rust_report["source"]["commit"],
            "8ce479cac1f2940502da1a55e19d19183f4862f7"
        );
        assert_eq!(
            rust_report["source"]["source_file_sha256"],
            "f9fc3538d1e167bc30268dae85d664fa59a97897eab65024fcb04d5eca248417"
        );
        assert_eq!(
            rust_report["source"]["instrumentation_patch_sha256"],
            "14535186936f2e72c073119ee2461c1ee394b8ffd957b1fd131bf80b100b7856"
        );
        assert_eq!(
            rust_report["model"]["license_path"],
            "models/M8_SYNTHETIC_ONNX_LICENSE.txt"
        );
        assert_eq!(
            rust_report["model"]["license_sha256"],
            "11762333d44173f00c5bbe7e7e805105f1d75ab38c93b079807e33d23136d8a6"
        );
        for profile in ["rmbg-rust", "rembg-bria"] {
            let directory = root.join(profile);
            let entry = report["profiles"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["profile"] == profile)
                .unwrap();
            for stage in [
                "decoded-rgb.f32le",
                "preprocessed-tensor.f32le",
                "raw-onnx-output.f32le",
                "restored-alpha.f32le",
                "final-straight-alpha-cutout.rgba",
            ] {
                let bytes = std::fs::read(directory.join(stage)).unwrap();
                let expected_len = match stage {
                    "decoded-rgb.f32le" => 3 * 2 * 3 * 4,
                    "preprocessed-tensor.f32le" | "raw-onnx-output.f32le" => 3 * 1024 * 1024 * 4,
                    "restored-alpha.f32le" => 3 * 2 * 4,
                    _ => 3 * 2 * 4,
                };
                assert_eq!(bytes.len(), expected_len, "fixture stage {profile}/{stage}");
                let actual = sha2::Sha256::digest(&bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>();
                assert_eq!(actual, entry["stages"][stage]);
            }
        }
    }

    #[test]
    fn m8_authoritative_rembg_stages_are_complete_and_hash_validated() {
        let root = Path::new("../../tests/fixtures/m8/authoritative");
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("report.json")).unwrap())
                .unwrap();
        assert_eq!(report["authoritative_execution"], true);
        assert_eq!(report["profile"], "rembg-bria");
        assert_eq!(
            report["source"]["commit"],
            "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
        );
        assert_eq!(
            report["source"]["source_file_sha256"],
            "e3c3747be5af3db15597796a83e73cfa5c464bb5df9c2b047c4113c4bfc3f811"
        );
        assert_eq!(
            report["source"]["base_source_file_sha256"],
            "ec4c58b33dd47ad6f03883ee375353314b19d5688fffd5c1ddb57bb21e9846a3"
        );
        assert_eq!(
            report["source"]["source_tree_sha256"],
            "a9c2584b47370c5f7f71e0049c9130a311b028150e444ed979ffafbccdd6b058"
        );
        assert_eq!(
            report["model"]["license_path"],
            "models/M3_FIXTURE_LICENSE.txt"
        );
        assert_eq!(
            report["model"]["license_sha256"],
            "cfed44a701bec837a8ae43d9e6baf69fa5b7fd88aeed383c5ad630b8f430b610"
        );
        assert_eq!(
            report["runtime"]["raw_shape"],
            serde_json::json!([1, 3, 1024, 1024])
        );
        let directory = Path::new("../../tests/fixtures/m8/reference/rembg-bria");
        for stage in [
            "decoded-rgb.f32le",
            "preprocessed-tensor.f32le",
            "raw-onnx-output.f32le",
            "restored-alpha.f32le",
            "final-straight-alpha-cutout.rgba",
        ] {
            let bytes = std::fs::read(directory.join(stage)).unwrap();
            let expected_len = match stage {
                "decoded-rgb.f32le" => 3 * 2 * 3 * 4,
                "preprocessed-tensor.f32le" | "raw-onnx-output.f32le" => 3 * 1024 * 1024 * 4,
                "restored-alpha.f32le" => 3 * 2 * 4,
                _ => 3 * 2 * 4,
            };
            assert_eq!(bytes.len(), expected_len, "authoritative stage {stage}");
            let actual = sha2::Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(actual, report["stages"][stage]);
        }
    }

    #[cfg(feature = "bria")]
    #[test]
    fn m8_authoritative_profiles_match_rust_stage_semantics() -> Result<()> {
        fn read_f32(path: &Path) -> Vec<f32> {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect()
        }
        let image = bgremove_core::CanonicalImage::new_with_alpha(
            3,
            2,
            vec![
                [0.0, 64.0 / 255.0, 1.0],
                [128.0 / 255.0, 191.0 / 255.0, 26.0 / 255.0],
                [51.0 / 255.0, 102.0 / 255.0, 204.0 / 255.0],
                [1.0, 26.0 / 255.0, 0.0],
                [77.0 / 255.0, 153.0 / 255.0, 230.0 / 255.0],
                [204.0 / 255.0, 51.0 / 255.0, 102.0 / 255.0],
            ],
            bgremove_core::AlphaMask::new(
                3,
                2,
                vec![
                    0.0,
                    64.0 / 255.0,
                    180.0 / 255.0,
                    1.0,
                    128.0 / 255.0,
                    220.0 / 255.0,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let rust_root = Path::new("../../tests/fixtures/m8/reference/rmbg-rust");
        let python_root = Path::new("../../tests/fixtures/m8/reference/rembg-bria");
        let rust_tensor = rmbg_rust_preprocess_rgb(&image).unwrap();
        let python_tensor = rembg_bria_preprocess_rgb(&image).unwrap();
        let parity: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("../../tests/fixtures/m8/reference/parity.json").unwrap(),
        )
        .unwrap();
        assert_eq!(parity["authoritative_sources_executed"], true);
        assert_eq!(
            parity["source"]["rust_rmbg_commit"],
            "8ce479cac1f2940502da1a55e19d19183f4862f7"
        );
        assert_eq!(
            parity["source"]["rembg_commit"],
            "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
        );
        assert_eq!(parity["source"]["rust_tracked_source_clean"], true);
        assert_eq!(
            parity["source"]["rust_instrumentation_patch_sha256"],
            "14535186936f2e72c073119ee2461c1ee394b8ffd957b1fd131bf80b100b7856"
        );
        let expected_rust_tensor = read_f32(&rust_root.join("preprocessed-tensor.f32le"));
        let expected_python_tensor = read_f32(&python_root.join("preprocessed-tensor.f32le"));
        assert_eq!(rust_tensor.values, expected_rust_tensor);
        assert!(python_tensor
            .values
            .iter()
            .zip(expected_python_tensor)
            .all(|(actual, expected)| (actual - expected).abs() <= 2e-5));

        let plane = 1024 * 1024;
        for (root, profile) in [
            (rust_root, RmbgProfile::RustCrate),
            (python_root, RmbgProfile::RembgPython),
        ] {
            let raw = read_f32(&root.join("raw-onnx-output.f32le"));
            assert_eq!(raw.len(), 3 * plane);
            let expected_raw = raw.clone();
            assert_eq!(read_f32(&root.join("raw-onnx-output.f32le")), expected_raw);
            let restored = restore_rmbg_mask(
                &raw[..plane],
                profile,
                OutputNormalization::MinMax,
                image.width(),
                image.height(),
            )?;
            let expected = read_f32(&root.join("restored-alpha.f32le"));
            let max_error = restored
                .data()
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                max_error, 0.0,
                "authoritative restore mismatch for {profile:?}"
            );
            assert!(restored.data().iter().all(|value| value.is_finite()));
            let cutout = isnet_straight_cutout(&image, restored)?;
            let mut actual = Vec::with_capacity(3 * 2 * 4);
            for (rgb, alpha) in cutout.rgb().data().iter().zip(cutout.alpha().data()) {
                actual.extend(rgb.iter().map(|v| (v * 255.0).round() as u8));
                actual.push((alpha * 255.0).round() as u8);
            }
            assert_eq!(
                actual,
                std::fs::read(root.join("final-straight-alpha-cutout.rgba"))?
            );
        }
        Ok(())
    }

    #[test]
    fn m8_manifest_profiles_validate_without_external_weights() {
        for path in [
            "../../models/m8_rmbg_1_4.toml",
            "../../models/m8_rmbg_2_0.toml",
        ] {
            let manifest =
                bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
            let error = manifest.verify_model_hash(Path::new(path)).unwrap_err();
            assert!(error.to_string().contains("not approved for intended use"));
            assert!(!manifest.intended_use_approved);
        }
    }

    #[cfg(feature = "bria")]
    #[test]
    #[ignore = "requires ORT_DYLIB, approved external RMBG weights, and explicit M8 licence approval"]
    fn m8_real_level2_profiles_match_authoritative_sources_when_approved() {
        let runtime_os = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let input = std::env::var_os("M8_INPUT").expect("M8_INPUT");
        let image = bgremove_core::io::load_canonical(Path::new(&input)).unwrap();
        for manifest_path in [
            "../../models/m8_rmbg_1_4.toml",
            "../../models/m8_rmbg_2_0.toml",
        ] {
            let manifest =
                bgremove_models::parse_toml(&std::fs::read_to_string(manifest_path).unwrap())
                    .unwrap();
            assert!(
                manifest.intended_use_approved,
                "M8 checkpoint licence must be explicitly approved"
            );
            let segmenter = RmbgSegmenter::new(
                &manifest,
                Path::new(manifest_path),
                Path::new(&runtime_os),
                1,
                RequestedProvider::Cpu,
                false,
            )
            .unwrap();
            let evidence = segmenter.predict_with_evidence(&image).unwrap();
            assert!(evidence
                .raw_output
                .values
                .iter()
                .all(|value| value.is_finite()));
            assert!(evidence
                .restored
                .data()
                .iter()
                .all(|value| value.is_finite()));
        }
    }

    #[cfg(feature = "bria")]
    #[test]
    #[ignore = "requires an externally supplied ORT_DYLIB; uses the permissively licensed M3 identity graph as synthetic 1024² source"]
    fn m8_synthetic_ort_gate_runs_channel_zero_and_all_stages() {
        let runtime_os = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let path = Path::new("../../models/m8_rmbg_1_4.toml");
        let mut manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        manifest.file = "../models/fixtures/m8_rmbg_identity_output.onnx".into();
        manifest.external = false;
        manifest.sha256 = "270f3af536551a7ca1a4834b987b3da9c0a5c8f55ccd30cf89a1a3eeeadd18b3".into();
        manifest.input_name = "input".into();
        manifest.output_name = "output".into();
        manifest.input_shape = vec![
            DimensionSpec::Dynamic("batch".into()),
            DimensionSpec::Dynamic("channel".into()),
            DimensionSpec::Dynamic("height".into()),
            DimensionSpec::Dynamic("width".into()),
        ];
        manifest.output_shape = vec![
            DimensionSpec::Dynamic("batch".into()),
            DimensionSpec::Dynamic("channel".into()),
            DimensionSpec::Dynamic("height".into()),
            DimensionSpec::Dynamic("width".into()),
        ];
        manifest.opset = 13;
        manifest.intended_use_approved = true;
        let image = bgremove_core::CanonicalImage::new_with_alpha(
            3,
            2,
            vec![
                [0.0, 64.0 / 255.0, 1.0],
                [128.0 / 255.0, 191.0 / 255.0, 26.0 / 255.0],
                [51.0 / 255.0, 102.0 / 255.0, 204.0 / 255.0],
                [1.0, 26.0 / 255.0, 0.0],
                [77.0 / 255.0, 153.0 / 255.0, 230.0 / 255.0],
                [204.0 / 255.0, 51.0 / 255.0, 102.0 / 255.0],
            ],
            bgremove_core::AlphaMask::new(
                3,
                2,
                vec![
                    0.0,
                    64.0 / 255.0,
                    180.0 / 255.0,
                    1.0,
                    128.0 / 255.0,
                    220.0 / 255.0,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let segmenter = RmbgSegmenter::new(
            &manifest,
            path,
            Path::new(&runtime_os),
            1,
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        let evidence = segmenter.predict_with_evidence(&image).unwrap();
        assert_eq!(evidence.tensor.values.len(), 3 * 1024 * 1024);
        assert_eq!(evidence.raw_output.values.len(), 3 * 1024 * 1024);
        assert_eq!(evidence.restored.dimensions(), image.dimensions());
        assert!(evidence
            .restored
            .data()
            .iter()
            .all(|value| value.is_finite()));
        fn f32le(path: &str) -> Vec<f32> {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        let root = Path::new("../../tests/fixtures/m8/reference/rmbg-rust");
        assert_eq!(
            evidence.tensor.values,
            f32le(root.join("preprocessed-tensor.f32le").to_str().unwrap())
        );
        assert_eq!(
            evidence.raw_output.values,
            f32le(root.join("raw-onnx-output.f32le").to_str().unwrap())
        );
        assert_eq!(
            evidence.restored.data(),
            f32le(root.join("restored-alpha.f32le").to_str().unwrap())
        );
        let cutout = isnet_straight_cutout(&image, evidence.restored).unwrap();
        let mut bytes = Vec::new();
        for (rgb, alpha) in cutout.rgb().data().iter().zip(cutout.alpha().data()) {
            bytes.extend(rgb.iter().map(|v| (v * 255.0).round() as u8));
            bytes.push((alpha * 255.0).round() as u8);
        }
        assert_eq!(
            bytes,
            std::fs::read(root.join("final-straight-alpha-cutout.rgba")).unwrap()
        );
    }

    #[test]
    fn zero_worker_pool_is_rejected_before_runtime() {
        let path = Path::new("../../models/m3_identity.toml");
        let m = bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        let result = SessionPool::new(
            &m,
            path,
            Path::new("/missing"),
            0,
            RequestedProvider::Cpu,
            false,
        );
        let error = match result {
            Ok(_) => panic!("zero workers must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("workers"));
    }

    #[test]
    #[ignore = "requires ORT_DYLIB and runs the checked-in fixture"]
    fn real_pool_reuses_sessions_and_bounds_concurrency() {
        let path = Path::new("../../models/m3_identity.toml");
        let manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        let runtime_os = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let runtime = Path::new(&runtime_os);
        let pool = Arc::new(
            SessionPool::new(&manifest, path, runtime, 2, RequestedProvider::Cpu, false).unwrap(),
        );
        {
            let mut lease = pool.checkout();
            let output = lease
                .session_mut()
                .run(&[1, 1, 1, 2], &[0.25, 0.75])
                .unwrap();
            assert_eq!(output.values, vec![0.25, 0.75]);
        }
        {
            let lease = pool.checkout();
            assert!(lease.session.as_ref().unwrap().run_count() >= 1);
        }
        let mut joins = Vec::new();
        for _ in 0..4 {
            let shared = Arc::clone(&pool);
            joins.push(std::thread::spawn(move || {
                let mut lease = shared.checkout();
                lease
                    .session_mut()
                    .run(&[1, 1, 1, 2], &[0.25, 0.75])
                    .unwrap();
                std::thread::sleep(std::time::Duration::from_millis(10));
            }));
        }
        for join in joins {
            join.join().unwrap();
        }
        assert_eq!(pool.size(), 2);
        assert!(pool.max_active() <= 2);
        assert!(pool.max_active() >= 2);
    }

    #[test]
    #[ignore = "requires ORT_DYLIB and runs the checked-in fixture"]
    fn real_fixture_runs_two_dynamic_geometries_and_selects_declared_output() {
        let path = Path::new("../../models/m3_identity.toml");
        let manifest =
            bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        let runtime_os = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
        let mut session = VerifiedSession::open(
            &manifest,
            path,
            Path::new(&runtime_os),
            RequestedProvider::Cpu,
            false,
        )
        .unwrap();
        for shape in [[1, 3, 2, 3], [1, 3, 3, 5]] {
            let count = checked_numel(&shape, "test input").unwrap();
            let values = (0..count)
                .map(|i| i as f32 / count as f32)
                .collect::<Vec<_>>();
            let output = session.run(&shape, &values).unwrap();
            assert_eq!(output.shape, shape);
            assert_eq!(output.values, values);
        }
        assert_eq!(session.run_count(), 2);
    }

    #[test]
    fn traversal_and_intended_use_are_rejected() {
        let path = Path::new("../../models/m3_identity.toml");
        let mut m = bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        m.file = "../Cargo.toml".into();
        assert!(m.verify_model_hash(path).is_err());
        let mut approved =
            bgremove_models::parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        approved.intended_use_approved = false;
        assert!(approved.verify_model_hash(path).is_err());
    }

    #[test]
    fn malformed_opset_bytes_fail_closed() {
        assert_eq!(declared_opset(&[]), None);
        assert_eq!(declared_opset(&[0x42, 0x80]), None);
        assert_eq!(declared_opset(&[0x42, 0x01, 0x80]), None);
    }

    #[test]
    fn opset_reader_prefers_default_domain_over_vendor_order() {
        // OperatorSetIdProto: domain (field 1), version (field 2).
        let vendor = [
            0x42, 0x0a, 0x0a, 0x06, b'v', b'e', b'n', b'd', b'o', b'r', 0x10, 0x07,
        ];
        let default = [0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d];
        let mut model = vendor.to_vec();
        model.extend(default);
        assert_eq!(declared_opset(&model), Some(13));
        assert_eq!(declared_opset(&vendor), None);
    }
}
