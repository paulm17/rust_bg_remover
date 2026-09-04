//! Typed, invariant-enforcing M1 contracts for background removal.
//!
//! No format decoder, resampler, model runtime, or post-processing algorithm
//! lives here. Those mechanisms can be added behind these stable contracts.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Three-channel encoded-space RGB pixels on a declared grid.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImageF32 {
    width: u32,
    height: u32,
    data: Vec<[f32; 3]>,
}

impl RgbImageF32 {
    /// Creates an image after checking dimensions, length, and finite channels.
    pub fn new(width: u32, height: u32, data: Vec<[f32; 3]>) -> Result<Self> {
        let expected = checked_len(width, height)?;
        ensure!(
            data.len() == expected,
            "RGB length {} does not match {}x{}",
            data.len(),
            width,
            height
        );
        ensure!(
            data.iter().flatten().all(|v| v.is_finite()),
            "RGB contains NaN or infinity"
        );
        Ok(Self {
            width,
            height,
            data,
        })
    }
    pub fn constant(width: u32, height: u32, rgb: [f32; 3]) -> Result<Self> {
        Self::new(width, height, vec![rgb; checked_len(width, height)?])
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }
    pub fn into_data(self) -> Vec<[f32; 3]> {
        self.data
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// RGB pixels after EXIF orientation has been applied once by a front end.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalImage {
    rgb: RgbImageF32,
}
impl CanonicalImage {
    pub fn new(width: u32, height: u32, data: Vec<[f32; 3]>) -> Result<Self> {
        Ok(Self {
            rgb: RgbImageF32::new(width, height, data)?,
        })
    }
    pub fn from_rgb(rgb: RgbImageF32) -> Self {
        Self { rgb }
    }
    pub fn rgb(&self) -> &RgbImageF32 {
        &self.rgb
    }
    pub fn width(&self) -> u32 {
        self.rgb.width()
    }
    pub fn height(&self) -> u32 {
        self.rgb.height()
    }
    pub fn dimensions(&self) -> (u32, u32) {
        self.rgb.dimensions()
    }
}

/// A finite, normalized alpha mask on a declared grid.
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaMask {
    width: u32,
    height: u32,
    data: Vec<f32>,
}
impl AlphaMask {
    /// Values are required to be finite and in [0, 1].
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Result<Self> {
        let expected = checked_len(width, height)?;
        ensure!(
            data.len() == expected,
            "alpha length {} does not match {}x{}",
            data.len(),
            width,
            height
        );
        ensure!(
            data.iter().all(|v| v.is_finite()),
            "alpha contains NaN or infinity"
        );
        ensure!(
            data.iter().all(|v| (0.0..=1.0).contains(v)),
            "alpha must be in [0, 1]"
        );
        Ok(Self {
            width,
            height,
            data,
        })
    }
    pub fn zeros(width: u32, height: u32) -> Result<Self> {
        Self::new(width, height, vec![0.0; checked_len(width, height)?])
    }
    pub fn ones(width: u32, height: u32) -> Result<Self> {
        Self::new(width, height, vec![1.0; checked_len(width, height)?])
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Legal values in a trimap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrimapClass {
    Background,
    Unknown,
    Foreground,
}
impl TrimapClass {
    pub fn from_raw(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Background),
            1 => Ok(Self::Unknown),
            2 => Ok(Self::Foreground),
            _ => anyhow::bail!("invalid trimap class {value}; expected 0, 1, or 2"),
        }
    }
}

/// A validated trimap whose length matches its grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trimap {
    width: u32,
    height: u32,
    data: Vec<TrimapClass>,
}
impl Trimap {
    pub fn new(width: u32, height: u32, data: Vec<TrimapClass>) -> Result<Self> {
        let expected = checked_len(width, height)?;
        ensure!(
            data.len() == expected,
            "trimap length {} does not match {}x{}",
            data.len(),
            width,
            height
        );
        Ok(Self {
            width,
            height,
            data,
        })
    }
    pub fn unknown(width: u32, height: u32) -> Result<Self> {
        Self::new(
            width,
            height,
            vec![TrimapClass::Unknown; checked_len(width, height)?],
        )
    }
    pub fn from_raw(width: u32, height: u32, values: &[u8]) -> Result<Self> {
        Self::new(
            width,
            height,
            values
                .iter()
                .copied()
                .map(TrimapClass::from_raw)
                .collect::<Result<Vec<_>>>()?,
        )
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn data(&self) -> &[TrimapClass] {
        &self.data
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Refined alpha and optional, paired model-provided foreground/background.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinedMatte {
    alpha: AlphaMask,
    foreground: Option<RgbImageF32>,
    background: Option<RgbImageF32>,
}
impl RefinedMatte {
    pub fn new(
        alpha: AlphaMask,
        foreground: Option<RgbImageF32>,
        background: Option<RgbImageF32>,
    ) -> Result<Self> {
        if let Some(ref f) = foreground {
            ensure!(
                f.dimensions() == alpha.dimensions(),
                "refined foreground dimensions do not match alpha"
            );
        }
        if let Some(ref b) = background {
            ensure!(
                b.dimensions() == alpha.dimensions(),
                "refined background dimensions do not match alpha"
            );
        }
        ensure!(
            foreground.is_none() == background.is_none(),
            "refined foreground and background must be supplied together"
        );
        Ok(Self {
            alpha,
            foreground,
            background,
        })
    }
    pub fn alpha(&self) -> &AlphaMask {
        &self.alpha
    }
    pub fn foreground(&self) -> Option<&RgbImageF32> {
        self.foreground.as_ref()
    }
    pub fn background(&self) -> Option<&RgbImageF32> {
        self.background.as_ref()
    }
}

/// Final straight-alpha foreground result.
#[derive(Clone, Debug, PartialEq)]
pub struct Foreground {
    rgb: RgbImageF32,
    alpha: AlphaMask,
}
impl Foreground {
    pub fn new(rgb: RgbImageF32, alpha: AlphaMask) -> Result<Self> {
        ensure!(
            rgb.dimensions() == alpha.dimensions(),
            "foreground RGB and alpha dimensions differ"
        );
        Ok(Self { rgb, alpha })
    }
    pub fn rgb(&self) -> &RgbImageF32 {
        &self.rgb
    }
    pub fn alpha(&self) -> &AlphaMask {
        &self.alpha
    }
}

/// Prompt coordinates are canonical pixel-space coordinates.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Prompt {
    points: Vec<PromptPoint>,
    box_region: Option<PromptBox>,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PromptPoint {
    x: f32,
    y: f32,
    positive: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PromptBox {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl PromptPoint {
    pub fn new(x: f32, y: f32, positive: bool) -> Result<Self> {
        ensure!(
            x.is_finite() && y.is_finite(),
            "prompt point coordinates must be finite"
        );
        Ok(Self { x, y, positive })
    }
    pub fn coordinates(&self) -> (f32, f32) {
        (self.x, self.y)
    }
    pub fn positive(&self) -> bool {
        self.positive
    }
}
impl PromptBox {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Result<Self> {
        ensure!(
            [x0, y0, x1, y1].iter().all(|v| v.is_finite()),
            "prompt box coordinates must be finite"
        );
        ensure!(
            x0 <= x1 && y0 <= y1,
            "prompt box coordinates must be ordered"
        );
        Ok(Self { x0, y0, x1, y1 })
    }
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        (self.x0, self.y0, self.x1, self.y1)
    }
}
impl Prompt {
    pub fn new(points: Vec<PromptPoint>, box_region: Option<PromptBox>) -> Result<Self> {
        let prompt = Self { points, box_region };
        prompt.validate()?;
        Ok(prompt)
    }
    pub fn points(&self) -> &[PromptPoint] {
        &self.points
    }
    pub fn box_region(&self) -> Option<PromptBox> {
        self.box_region
    }
    pub fn validate(&self) -> Result<()> {
        for point in &self.points {
            ensure!(
                point.x.is_finite() && point.y.is_finite(),
                "prompt point coordinates must be finite"
            );
        }
        if let Some(b) = self.box_region {
            ensure!(
                [b.x0, b.y0, b.x1, b.y1].iter().all(|v| v.is_finite()),
                "prompt box coordinates must be finite"
            );
            ensure!(
                b.x0 <= b.x1 && b.y0 <= b.y1,
                "prompt box coordinates must be ordered"
            );
        }
        Ok(())
    }
    pub fn validate_for(&self, width: u32, height: u32) -> Result<()> {
        self.validate()?;
        let (w, h) = (width as f32, height as f32);
        for p in &self.points {
            ensure!(
                (0.0..=w).contains(&p.x) && (0.0..=h).contains(&p.y),
                "prompt point is outside image bounds"
            );
        }
        if let Some(b) = self.box_region {
            ensure!(
                b.x0 >= 0.0 && b.y0 >= 0.0 && b.x1 <= w && b.y1 <= h,
                "prompt box is outside image bounds"
            );
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptPointWire {
    x: f32,
    y: f32,
    positive: bool,
}
impl<'de> Deserialize<'de> for PromptPoint {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = PromptPointWire::deserialize(d)?;
        Self::new(w.x, w.y, w.positive).map_err(serde::de::Error::custom)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptBoxWire {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}
impl<'de> Deserialize<'de> for PromptBox {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = PromptBoxWire::deserialize(d)?;
        Self::new(w.x0, w.y0, w.x1, w.y1).map_err(serde::de::Error::custom)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptWire {
    points: Vec<PromptPoint>,
    box_region: Option<PromptBox>,
}
impl<'de> Deserialize<'de> for Prompt {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = PromptWire::deserialize(d)?;
        Self::new(w.points, w.box_region).map_err(serde::de::Error::custom)
    }
}

/// Interpolation policy recorded by a geometry operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResizeFilter {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
}

/// Distinct stretch, pad, crop and thumbnail policies. M2 will implement them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "kebab-case")]
pub enum GeometryPolicy {
    Stretch,
    ContainPad { pad_x: u32, pad_y: u32 },
    CoverCrop { crop_x: u32, crop_y: u32 },
    Thumbnail,
}

/// Typed source-to-model geometry metadata; no resampling occurs in M1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GeometryTransform {
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    policy: GeometryPolicy,
    filter: ResizeFilter,
}
impl GeometryTransform {
    pub fn new(
        source_width: u32,
        source_height: u32,
        target_width: u32,
        target_height: u32,
        policy: GeometryPolicy,
        filter: ResizeFilter,
    ) -> Result<Self> {
        ensure!(
            source_width > 0 && source_height > 0 && target_width > 0 && target_height > 0,
            "geometry dimensions must be positive"
        );
        Ok(Self {
            source_width,
            source_height,
            target_width,
            target_height,
            policy,
            filter,
        })
    }
    pub fn source_dimensions(&self) -> (u32, u32) {
        (self.source_width, self.source_height)
    }
    pub fn target_dimensions(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }
    pub fn policy(&self) -> &GeometryPolicy {
        &self.policy
    }
    pub fn filter(&self) -> ResizeFilter {
        self.filter
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.source_width > 0
                && self.source_height > 0
                && self.target_width > 0
                && self.target_height > 0,
            "geometry dimensions must be positive"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometryWire {
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    policy: GeometryPolicy,
    filter: ResizeFilter,
}
impl<'de> Deserialize<'de> for GeometryTransform {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = GeometryWire::deserialize(d)?;
        Self::new(
            w.source_width,
            w.source_height,
            w.target_width,
            w.target_height,
            w.policy,
            w.filter,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Foreground working-space declaration for later estimators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkingColorSpace {
    Srgb,
    Linear,
}

/// Fully resolved configuration. BTreeMap makes arbitrary parameter order stable.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PipelineConfig {
    schema: String,
    segmenter: String,
    mask_transform: String,
    alpha_refiner: String,
    foreground_estimator: String,
    geometry: GeometryTransform,
    working_color_space: WorkingColorSpace,
    output_mode: String,
    parameters: BTreeMap<String, serde_json::Value>,
}
impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            schema: "m1.pipeline.v1".into(),
            segmenter: "noop".into(),
            mask_transform: "identity".into(),
            alpha_refiner: "noop".into(),
            foreground_estimator: "original-rgb".into(),
            geometry: GeometryTransform {
                source_width: 1,
                source_height: 1,
                target_width: 1,
                target_height: 1,
                policy: GeometryPolicy::Stretch,
                filter: ResizeFilter::Nearest,
            },
            working_color_space: WorkingColorSpace::Srgb,
            output_mode: "straight-rgba".into(),
            parameters: BTreeMap::new(),
        }
    }
}
impl PipelineConfig {
    pub fn geometry(&self) -> &GeometryTransform {
        &self.geometry
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "m1.pipeline.v1",
            "unsupported pipeline config schema"
        );
        for (name, value) in [
            ("segmenter", &self.segmenter),
            ("mask_transform", &self.mask_transform),
            ("alpha_refiner", &self.alpha_refiner),
            ("foreground_estimator", &self.foreground_estimator),
            ("output_mode", &self.output_mode),
        ] {
            ensure!(
                !value.trim().is_empty(),
                "pipeline config {name} must not be empty"
            );
        }
        self.geometry.validate()
    }
    pub fn resolved_for(&self, width: u32, height: u32) -> Result<Self> {
        ensure!(
            width > 0 && height > 0,
            "resolved pipeline dimensions must be positive"
        );
        let mut c = self.clone();
        c.geometry = GeometryTransform::new(
            width,
            height,
            width,
            height,
            c.geometry.policy.clone(),
            c.geometry.filter,
        )?;
        c.validate()?;
        Ok(c)
    }
    /// Canonical JSON with stable key ordering and one LF.
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut b = serde_json::to_vec_pretty(self)?;
        b.push(b'\n');
        Ok(b)
    }
    pub fn from_canonical_json(bytes: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(bytes)?;
        config.validate()?;
        Ok(config)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineConfigWire {
    schema: String,
    segmenter: String,
    mask_transform: String,
    alpha_refiner: String,
    foreground_estimator: String,
    geometry: GeometryTransform,
    working_color_space: WorkingColorSpace,
    output_mode: String,
    parameters: BTreeMap<String, serde_json::Value>,
}
impl<'de> Deserialize<'de> for PipelineConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = PipelineConfigWire::deserialize(d)?;
        let config = Self {
            schema: w.schema,
            segmenter: w.segmenter,
            mask_transform: w.mask_transform,
            alpha_refiner: w.alpha_refiner,
            foreground_estimator: w.foreground_estimator,
            geometry: w.geometry,
            working_color_space: w.working_color_space,
            output_mode: w.output_mode,
            parameters: w.parameters,
        };
        config.validate().map_err(serde::de::Error::custom)?;
        Ok(config)
    }
}

/// Predicts a coarse alpha mask.
pub trait Segmenter {
    fn predict(&mut self, image: &CanonicalImage, prompt: Option<&Prompt>) -> Result<AlphaMask>;
}
/// Transforms a coarse alpha without changing its declared grid.
pub trait MaskTransform {
    fn apply(&self, image: &CanonicalImage, alpha: AlphaMask) -> Result<AlphaMask>;
}
/// Refines alpha and may return paired model foreground/background fields.
pub trait AlphaRefiner {
    fn refine(
        &mut self,
        image: &CanonicalImage,
        coarse: &AlphaMask,
        trimap: &Trimap,
    ) -> Result<RefinedMatte>;
}
/// Estimates straight-alpha RGB.
pub trait ForegroundEstimator {
    fn estimate(&self, image: &CanonicalImage, matte: &RefinedMatte) -> Result<RgbImageF32>;
}
/// Optional validated intermediate-stage capture for benchmark artifacts.
pub trait ArtifactSink {
    fn capture(&mut self, stage: &str, dimensions: (u32, u32), values: &[f32]) -> Result<()>;
}

/// Deterministic no-op segmenter for M1.
#[derive(Default)]
pub struct NoOpSegmenter;
impl Segmenter for NoOpSegmenter {
    fn predict(&mut self, image: &CanonicalImage, _prompt: Option<&Prompt>) -> Result<AlphaMask> {
        AlphaMask::zeros(image.width(), image.height())
    }
}
/// Identity mask transform.
#[derive(Default)]
pub struct NoOpMaskTransform;
impl MaskTransform for NoOpMaskTransform {
    fn apply(&self, image: &CanonicalImage, alpha: AlphaMask) -> Result<AlphaMask> {
        ensure!(
            alpha.dimensions() == image.dimensions(),
            "mask transform dimension mismatch"
        );
        Ok(alpha)
    }
}
/// No-op alpha refiner with an all-unknown trimap contract.
#[derive(Default)]
pub struct NoOpAlphaRefiner;
impl AlphaRefiner for NoOpAlphaRefiner {
    fn refine(
        &mut self,
        image: &CanonicalImage,
        coarse: &AlphaMask,
        trimap: &Trimap,
    ) -> Result<RefinedMatte> {
        ensure!(
            coarse.dimensions() == image.dimensions(),
            "refiner alpha dimension mismatch"
        );
        ensure!(
            trimap.dimensions() == image.dimensions(),
            "refiner trimap dimension mismatch"
        );
        RefinedMatte::new(coarse.clone(), None, None)
    }
}
/// Original RGB estimator used as the M1 control candidate.
#[derive(Default)]
pub struct NoOpForegroundEstimator;
impl ForegroundEstimator for NoOpForegroundEstimator {
    fn estimate(&self, image: &CanonicalImage, matte: &RefinedMatte) -> Result<RgbImageF32> {
        ensure!(
            matte.alpha().dimensions() == image.dimensions(),
            "colour estimator dimension mismatch"
        );
        Ok(image.rgb().clone())
    }
}

/// Validated end-to-end stage orchestration.
pub struct Pipeline {
    pub config: PipelineConfig,
    pub segmenter: Box<dyn Segmenter>,
    pub mask_transform: Box<dyn MaskTransform>,
    pub refiner: Box<dyn AlphaRefiner>,
    pub foreground: Box<dyn ForegroundEstimator>,
    pub sink: Option<Box<dyn ArtifactSink>>,
}
impl Pipeline {
    pub fn new(
        config: PipelineConfig,
        segmenter: Box<dyn Segmenter>,
        mask_transform: Box<dyn MaskTransform>,
        refiner: Box<dyn AlphaRefiner>,
        foreground: Box<dyn ForegroundEstimator>,
    ) -> Self {
        Self {
            config,
            segmenter,
            mask_transform,
            refiner,
            foreground,
            sink: None,
        }
    }
    pub fn with_sink(mut self, sink: Box<dyn ArtifactSink>) -> Self {
        self.sink = Some(sink);
        self
    }
    fn capture(&mut self, stage: &str, dimensions: (u32, u32), values: &[f32]) -> Result<()> {
        if let Some(s) = &mut self.sink {
            s.capture(stage, dimensions, values)?;
        }
        Ok(())
    }
    pub fn run(&mut self, image: &CanonicalImage, prompt: Option<&Prompt>) -> Result<Foreground> {
        self.config.validate()?;
        let expected = image.dimensions();
        if let Some(prompt) = prompt {
            prompt.validate_for(expected.0, expected.1)?;
        }
        ensure!(
            self.config.geometry.source_width == expected.0
                && self.config.geometry.source_height == expected.1,
            "pipeline geometry source dimensions do not match image"
        );
        ensure!(
            self.config.geometry.target_width == expected.0
                && self.config.geometry.target_height == expected.1,
            "M1 pipeline requires identity geometry dimensions"
        );
        let coarse = self.segmenter.predict(image, prompt)?;
        ensure!(
            coarse.dimensions() == expected,
            "segmenter returned dimensions incompatible with image"
        );
        let coarse_data = coarse.data().to_vec();
        self.capture("coarse-alpha", expected, &coarse_data)?;
        let transformed = self.mask_transform.apply(image, coarse)?;
        ensure!(
            transformed.dimensions() == expected,
            "mask transform returned dimensions incompatible with image"
        );
        let transformed_data = transformed.data().to_vec();
        self.capture("transformed-alpha", expected, &transformed_data)?;
        let trimap = Trimap::unknown(expected.0, expected.1)?;
        let trimap_values: Vec<f32> = trimap
            .data()
            .iter()
            .map(|c| match c {
                TrimapClass::Background => 0.0,
                TrimapClass::Unknown => 0.5,
                TrimapClass::Foreground => 1.0,
            })
            .collect();
        self.capture("trimap", expected, &trimap_values)?;
        let matte = self.refiner.refine(image, &transformed, &trimap)?;
        ensure!(
            matte.alpha().dimensions() == expected,
            "refiner returned dimensions incompatible with image"
        );
        let alpha_data = matte.alpha().data().to_vec();
        self.capture("refined-alpha", expected, &alpha_data)?;
        let rgb = self.foreground.estimate(image, &matte)?;
        ensure!(
            rgb.dimensions() == expected,
            "foreground estimator returned dimensions incompatible with image"
        );
        let rgb_data: Vec<f32> = rgb
            .data()
            .iter()
            .flat_map(|pixel| pixel.iter().copied())
            .collect();
        self.capture("foreground-rgb", expected, &rgb_data)?;
        Foreground::new(rgb, matte.alpha().clone())
    }
}

fn checked_len(width: u32, height: u32) -> Result<usize> {
    ensure!(width > 0 && height > 0, "dimensions must be positive");
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| anyhow::anyhow!("dimensions overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    type Captured =
        std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, (u32, u32, usize)>>>;

    struct Sink(Captured);
    impl ArtifactSink for Sink {
        fn capture(&mut self, stage: &str, dimensions: (u32, u32), values: &[f32]) -> Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(stage.to_owned(), (dimensions.0, dimensions.1, values.len()));
            Ok(())
        }
    }

    struct WrongSegmenter;
    impl Segmenter for WrongSegmenter {
        fn predict(
            &mut self,
            _image: &CanonicalImage,
            _prompt: Option<&Prompt>,
        ) -> Result<AlphaMask> {
            AlphaMask::zeros(1, 1)
        }
    }
    #[test]
    fn invariants_reject_bad_alpha_and_lengths() {
        assert!(AlphaMask::new(2, 2, vec![0.0; 3]).is_err());
        assert!(AlphaMask::new(2, 2, vec![f32::NAN; 4]).is_err());
        assert!(AlphaMask::new(2, 2, vec![f32::INFINITY; 4]).is_err());
        assert!(AlphaMask::new(2, 2, vec![1.1; 4]).is_err());
        assert!(Trimap::new(2, 2, vec![TrimapClass::Unknown; 3]).is_err());
    }
    #[test]
    fn pipeline_and_config_are_deterministic() {
        let c = PipelineConfig::default().resolved_for(2, 2).unwrap();
        let bytes = c.canonical_json().unwrap();
        assert_eq!(bytes, c.canonical_json().unwrap());
        assert_eq!(c, PipelineConfig::from_canonical_json(&bytes).unwrap());
        let image = CanonicalImage::new(2, 2, vec![[0.0; 3]; 4]).unwrap();
        let mut p = Pipeline::new(
            c,
            Box::new(NoOpSegmenter),
            Box::new(NoOpMaskTransform),
            Box::new(NoOpAlphaRefiner),
            Box::new(NoOpForegroundEstimator),
        );
        assert_eq!(p.run(&image, None).unwrap().alpha().data(), &[0.0; 4]);
    }

    #[test]
    fn pipeline_rejects_stage_dimensions_and_captures_artifacts() {
        let image = CanonicalImage::new(2, 2, vec![[0.0; 3]; 4]).unwrap();
        let config = PipelineConfig::default().resolved_for(2, 2).unwrap();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(BTreeMap::new()));
        let sink = Sink(captured.clone());
        let mut p = Pipeline::new(
            config.clone(),
            Box::new(NoOpSegmenter),
            Box::new(NoOpMaskTransform),
            Box::new(NoOpAlphaRefiner),
            Box::new(NoOpForegroundEstimator),
        )
        .with_sink(Box::new(sink));
        assert_eq!(p.run(&image, None).unwrap().rgb().dimensions(), (2, 2));
        assert_eq!(captured.lock().unwrap().len(), 5);
        assert_eq!(captured.lock().unwrap()["trimap"], (2, 2, 4));
        let mut bad = Pipeline::new(
            config,
            Box::new(WrongSegmenter),
            Box::new(NoOpMaskTransform),
            Box::new(NoOpAlphaRefiner),
            Box::new(NoOpForegroundEstimator),
        );
        assert!(bad.run(&image, None).is_err());
    }

    #[test]
    fn refined_foreground_background_must_be_compatible() {
        let alpha = AlphaMask::zeros(2, 2).unwrap();
        let rgb = RgbImageF32::constant(1, 1, [0.0; 3]).unwrap();
        assert!(RefinedMatte::new(alpha.clone(), Some(rgb.clone()), Some(rgb)).is_err());
        assert!(RefinedMatte::new(
            alpha,
            Some(RgbImageF32::constant(2, 2, [0.0; 3]).unwrap()),
            None
        )
        .is_err());
    }

    #[test]
    fn raw_trimap_parser_rejects_unknown_class_and_prompts_are_validated() {
        assert!(Trimap::from_raw(1, 1, &[3]).is_err());
        assert!(PromptPoint::new(f32::NAN, 0.0, true).is_err());
        assert!(PromptBox::new(1.0, 0.0, 0.0, 1.0).is_err());
        let prompt = Prompt::new(vec![PromptPoint::new(2.0, 1.0, true).unwrap()], None).unwrap();
        assert!(prompt.validate_for(2, 2).is_ok());
        let outside = Prompt::new(vec![PromptPoint::new(3.0, 1.0, true).unwrap()], None).unwrap();
        assert!(outside.validate_for(2, 2).is_err());
    }

    #[test]
    fn config_deserialization_rejects_unknown_fields_and_bad_geometry() {
        let good =
            serde_json::to_value(PipelineConfig::default().resolved_for(2, 2).unwrap()).unwrap();
        let mut unknown = good.clone();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("extra".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<PipelineConfig>(unknown).is_err());
        let mut bad = good;
        bad.get_mut("geometry")
            .unwrap()
            .get_mut("source_width")
            .unwrap()
            .clone_from(&serde_json::json!(0));
        assert!(serde_json::from_value::<PipelineConfig>(bad).is_err());
    }
}
