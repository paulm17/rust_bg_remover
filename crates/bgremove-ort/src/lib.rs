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

/// Deterministic Lanczos3 path used by rembg's PIL LANCZOS profile. RGB and
/// single-channel masks are supported; callers retain the explicit profile
/// name because PIL/image-crate kernels are not interchangeable silently.
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
        PreprocessingProfile::Generic => unreachable!(),
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
                PreprocessingProfile::Generic => unreachable!(),
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
            PreprocessingProfile::Generic => unreachable!(),
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
            PreprocessingProfile::Generic => unreachable!(),
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
            PreprocessingProfile::Generic => unreachable!(),
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
#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;
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
