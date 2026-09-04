//! Typed, invariant-enforcing contracts and M2 image/geometry primitives for
//! background removal.
//!
//! Model runtime and later learned post-processing remain behind these stable
//! contracts; deterministic decoding, resampling, alpha and PNG IO are present
//! in M2.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod io;

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
    /// Source alpha on the canonical grid. It is always present so an opaque
    /// JPEG has an explicit alpha of one and transparent input is never lost.
    source_alpha: AlphaMask,
}
impl CanonicalImage {
    pub fn new(width: u32, height: u32, data: Vec<[f32; 3]>) -> Result<Self> {
        let alpha = AlphaMask::ones(width, height)?;
        Self::new_with_alpha(width, height, data, alpha)
    }
    pub fn new_with_alpha(
        width: u32,
        height: u32,
        data: Vec<[f32; 3]>,
        source_alpha: AlphaMask,
    ) -> Result<Self> {
        ensure!(
            source_alpha.dimensions() == (width, height),
            "source alpha dimensions do not match image"
        );
        Ok(Self {
            rgb: RgbImageF32::new(width, height, data)?,
            source_alpha,
        })
    }
    pub fn from_rgb(rgb: RgbImageF32) -> Self {
        let alpha = AlphaMask::ones(rgb.width(), rgb.height()).expect("validated RGB dimensions");
        Self {
            rgb,
            source_alpha: alpha,
        }
    }
    pub fn from_rgba(rgb: RgbImageF32, source_alpha: AlphaMask) -> Result<Self> {
        ensure!(
            rgb.dimensions() == source_alpha.dimensions(),
            "source alpha dimensions do not match image"
        );
        Ok(Self { rgb, source_alpha })
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
    pub fn source_alpha(&self) -> &AlphaMask {
        &self.source_alpha
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
    Triangle,
    Bicubic,
    Lanczos3,
}

/// Aspect modes. Integer padding/cropping extents are resolved and stored on
/// [`GeometryTransform`], never supplied as policy hints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryPolicy {
    Stretch,
    ContainPad,
    CoverCrop,
    Thumbnail,
}

impl Serialize for GeometryPolicy {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("GeometryPolicy", 1)?;
        let name = match self {
            Self::Stretch => "stretch",
            Self::ContainPad => "contain-pad",
            Self::CoverCrop => "cover-crop",
            Self::Thumbnail => "thumbnail",
        };
        s.serialize_field("policy", name)?;
        s.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometryPolicyWire {
    policy: String,
    #[serde(default)]
    pad_x: Option<u32>,
    #[serde(default)]
    pad_y: Option<u32>,
    #[serde(default)]
    crop_x: Option<u32>,
    #[serde(default)]
    crop_y: Option<u32>,
}
impl<'de> Deserialize<'de> for GeometryPolicy {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = GeometryPolicyWire::deserialize(d)?;
        let hints = [w.pad_x, w.pad_y, w.crop_x, w.crop_y];
        if hints.iter().flatten().any(|v| *v != 0) {
            return Err(serde::de::Error::custom(
                "legacy geometry placement hints must be zero; use resolved transform metadata",
            ));
        }
        match w.policy.as_str() {
            "stretch" => Ok(Self::Stretch),
            "contain-pad" => Ok(Self::ContainPad),
            "cover-crop" => Ok(Self::CoverCrop),
            "thumbnail" => Ok(Self::Thumbnail),
            _ => Err(serde::de::Error::custom("unknown geometry policy")),
        }
    }
}

/// Typed, invertible source-to-model geometry metadata.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeometryTransform {
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    policy: GeometryPolicy,
    filter: ResizeFilter,
    /// Scale and translation use pixel-center coordinates: target = source * scale + offset.
    scale_x: f64,
    scale_y: f64,
    offset_x: f64,
    offset_y: f64,
    intermediate_width: u32,
    intermediate_height: u32,
    pad_left: u32,
    pad_right: u32,
    pad_top: u32,
    pad_bottom: u32,
    crop_left: u32,
    crop_right: u32,
    crop_top: u32,
    crop_bottom: u32,
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
        let (
            scale_x,
            scale_y,
            offset_x,
            offset_y,
            intermediate_width,
            intermediate_height,
            pads,
            crops,
        ) = resolve_geometry(
            source_width,
            source_height,
            target_width,
            target_height,
            &policy,
        )?;
        let geometry = Self {
            source_width,
            source_height,
            target_width,
            target_height,
            policy,
            filter,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
            intermediate_width,
            intermediate_height,
            pad_left: pads.0,
            pad_right: pads.1,
            pad_top: pads.2,
            pad_bottom: pads.3,
            crop_left: crops.0,
            crop_right: crops.1,
            crop_top: crops.2,
            crop_bottom: crops.3,
        };
        geometry.validate()?;
        Ok(geometry)
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
    pub fn scale(&self) -> (f32, f32) {
        (self.scale_x as f32, self.scale_y as f32)
    }
    pub fn offsets(&self) -> (f32, f32) {
        (self.offset_x as f32, self.offset_y as f32)
    }
    pub fn exact_scale(&self) -> (f64, f64) {
        (self.scale_x, self.scale_y)
    }
    pub fn exact_offsets(&self) -> (f64, f64) {
        (self.offset_x, self.offset_y)
    }
    pub fn intermediate_dimensions(&self) -> (u32, u32) {
        (self.intermediate_width, self.intermediate_height)
    }
    pub fn padding(&self) -> (u32, u32, u32, u32) {
        (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
    }
    pub fn cropping(&self) -> (u32, u32, u32, u32) {
        (
            self.crop_left,
            self.crop_right,
            self.crop_top,
            self.crop_bottom,
        )
    }
    /// Map a source pixel-center coordinate to a target pixel-center coordinate.
    pub fn forward_coordinate(&self, x: f32, y: f32) -> Result<(f32, f32)> {
        ensure!(
            x.is_finite() && y.is_finite(),
            "geometry coordinates must be finite"
        );
        Ok((
            x * self.scale_x as f32 + self.offset_x as f32,
            y * self.scale_y as f32 + self.offset_y as f32,
        ))
    }
    /// Map a target pixel-center coordinate back to source coordinates.
    pub fn inverse_coordinate(&self, x: f32, y: f32) -> Result<(f32, f32)> {
        ensure!(
            x.is_finite() && y.is_finite(),
            "geometry coordinates must be finite"
        );
        Ok((
            (x - self.offset_x as f32) / self.scale_x as f32,
            (y - self.offset_y as f32) / self.scale_y as f32,
        ))
    }
    /// Resize RGB into the model grid. Contain/thumbnail padding is black.
    pub fn forward_rgb(&self, image: &RgbImageF32) -> Result<RgbImageF32> {
        ensure!(
            image.dimensions() == self.source_dimensions(),
            "RGB dimensions do not match geometry source"
        );
        let mut out = Vec::with_capacity(checked_len(self.target_width, self.target_height)?);
        for y in 0..self.target_height {
            for x in 0..self.target_width {
                let inside = x >= self.pad_left
                    && x < self.pad_left + self.intermediate_width
                    && y >= self.pad_top
                    && y < self.pad_top + self.intermediate_height;
                if !inside {
                    out.push([0.0; 3]);
                    continue;
                }
                let (sx, sy) = if matches!(
                    self.policy,
                    GeometryPolicy::ContainPad | GeometryPolicy::Thumbnail
                ) {
                    (
                        ((x - self.pad_left) as f32 + 0.5) * self.source_width as f32
                            / self.intermediate_width as f32,
                        ((y - self.pad_top) as f32 + 0.5) * self.source_height as f32
                            / self.intermediate_height as f32,
                    )
                } else {
                    self.inverse_coordinate(x as f32 + 0.5, y as f32 + 0.5)?
                };
                out.push(sample_rgb(image, sx - 0.5, sy - 0.5, self.filter, false));
            }
        }
        RgbImageF32::new(self.target_width, self.target_height, out)
    }
    /// Resize a mask into the model grid. Padded pixels are transparent.
    pub fn forward_mask(&self, mask: &AlphaMask) -> Result<AlphaMask> {
        ensure!(
            mask.dimensions() == self.source_dimensions(),
            "mask dimensions do not match geometry source"
        );
        self.resample_mask(mask, true)
    }
    /// Restore a model-grid mask to the canonical source grid.
    pub fn inverse_mask(&self, mask: &AlphaMask) -> Result<AlphaMask> {
        ensure!(
            mask.dimensions() == self.target_dimensions(),
            "mask dimensions do not match geometry target"
        );
        let mut out = Vec::with_capacity(checked_len(self.source_width, self.source_height)?);
        for y in 0..self.source_height {
            for x in 0..self.source_width {
                let (tx, ty) = self.forward_coordinate(x as f32 + 0.5, y as f32 + 0.5)?;
                if matches!(self.policy, GeometryPolicy::CoverCrop)
                    && (tx < 0.5
                        || tx > self.target_width as f32 - 0.5
                        || ty < 0.5
                        || ty > self.target_height as f32 - 0.5)
                {
                    out.push(0.0);
                } else if matches!(
                    self.policy,
                    GeometryPolicy::ContainPad | GeometryPolicy::Thumbnail
                ) {
                    let cx = tx.clamp(
                        self.pad_left as f32 + 0.5,
                        (self.pad_left + self.intermediate_width - 1) as f32 + 0.5,
                    );
                    let cy = ty.clamp(
                        self.pad_top as f32 + 0.5,
                        (self.pad_top + self.intermediate_height - 1) as f32 + 0.5,
                    );
                    out.push(sample_mask_rect(
                        mask,
                        cx - 0.5,
                        cy - 0.5,
                        self.filter,
                        (
                            self.pad_left,
                            self.pad_top,
                            self.intermediate_width,
                            self.intermediate_height,
                        ),
                    ));
                } else {
                    out.push(sample_mask(mask, tx - 0.5, ty - 0.5, self.filter, false));
                }
            }
        }
        AlphaMask::new(self.source_width, self.source_height, out)
    }
    fn resample_mask(&self, mask: &AlphaMask, forward: bool) -> Result<AlphaMask> {
        let _ = forward;
        let mut out = Vec::with_capacity(checked_len(self.target_width, self.target_height)?);
        for y in 0..self.target_height {
            for x in 0..self.target_width {
                let inside = x >= self.pad_left
                    && x < self.pad_left + self.intermediate_width
                    && y >= self.pad_top
                    && y < self.pad_top + self.intermediate_height;
                if !inside {
                    out.push(0.0);
                    continue;
                }
                let (sx, sy) = if matches!(
                    self.policy,
                    GeometryPolicy::ContainPad | GeometryPolicy::Thumbnail
                ) {
                    (
                        ((x - self.pad_left) as f32 + 0.5) * self.source_width as f32
                            / self.intermediate_width as f32,
                        ((y - self.pad_top) as f32 + 0.5) * self.source_height as f32
                            / self.intermediate_height as f32,
                    )
                } else {
                    self.inverse_coordinate(x as f32 + 0.5, y as f32 + 0.5)?
                };
                out.push(sample_mask(mask, sx - 0.5, sy - 0.5, self.filter, false));
            }
        }
        AlphaMask::new(self.target_width, self.target_height, out)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.source_width > 0
                && self.source_height > 0
                && self.target_width > 0
                && self.target_height > 0,
            "geometry dimensions must be positive"
        );
        ensure!(
            self.scale_x.is_finite()
                && self.scale_y.is_finite()
                && self.scale_x > 0.0
                && self.scale_y > 0.0,
            "geometry scales must be finite and positive"
        );
        ensure!(
            self.intermediate_width > 0 && self.intermediate_height > 0,
            "geometry intermediate dimensions must be positive"
        );
        if matches!(self.policy, GeometryPolicy::CoverCrop) {
            ensure!(
                self.pad_left == 0
                    && self.pad_right == 0
                    && self.pad_top == 0
                    && self.pad_bottom == 0,
                "crop geometry cannot contain padding metadata"
            );
        } else {
            ensure!(
                self.pad_left + self.intermediate_width + self.pad_right == self.target_width,
                "horizontal padding metadata does not cover target"
            );
            ensure!(
                self.pad_top + self.intermediate_height + self.pad_bottom == self.target_height,
                "vertical padding metadata does not cover target"
            );
        }
        if matches!(self.policy, GeometryPolicy::CoverCrop) {
            ensure!(
                self.crop_left + self.target_width + self.crop_right == self.intermediate_width,
                "horizontal crop metadata does not cover intermediate"
            );
            ensure!(
                self.crop_top + self.target_height + self.crop_bottom == self.intermediate_height,
                "vertical crop metadata does not cover intermediate"
            );
        } else {
            ensure!(
                self.crop_left == 0
                    && self.crop_right == 0
                    && self.crop_top == 0
                    && self.crop_bottom == 0,
                "non-crop geometry cannot contain crop metadata"
            );
        }
        Ok(())
    }
}

/// Convenience wrapper for forward RGB geometry.
pub fn forward_image(transform: &GeometryTransform, image: &RgbImageF32) -> Result<RgbImageF32> {
    transform.forward_rgb(image)
}

/// Convenience wrapper for forward mask geometry.
pub fn forward_mask(transform: &GeometryTransform, mask: &AlphaMask) -> Result<AlphaMask> {
    transform.forward_mask(mask)
}

/// Convenience wrapper for inverse mask restoration.
pub fn inverse_mask(transform: &GeometryTransform, mask: &AlphaMask) -> Result<AlphaMask> {
    transform.inverse_mask(mask)
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
    #[serde(default)]
    scale_x: Option<f64>,
    #[serde(default)]
    scale_y: Option<f64>,
    #[serde(default)]
    offset_x: Option<f64>,
    #[serde(default)]
    offset_y: Option<f64>,
    #[serde(default)]
    intermediate_width: Option<u32>,
    #[serde(default)]
    intermediate_height: Option<u32>,
    #[serde(default)]
    pad_left: Option<u32>,
    #[serde(default)]
    pad_right: Option<u32>,
    #[serde(default)]
    pad_top: Option<u32>,
    #[serde(default)]
    pad_bottom: Option<u32>,
    #[serde(default)]
    crop_left: Option<u32>,
    #[serde(default)]
    crop_right: Option<u32>,
    #[serde(default)]
    crop_top: Option<u32>,
    #[serde(default)]
    crop_bottom: Option<u32>,
}
impl<'de> Deserialize<'de> for GeometryTransform {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = GeometryWire::deserialize(d)?;
        let geometry = Self::new(
            w.source_width,
            w.source_height,
            w.target_width,
            w.target_height,
            w.policy,
            w.filter,
        )
        .map_err(serde::de::Error::custom)?;
        let has_metadata = w.scale_x.is_some()
            || w.scale_y.is_some()
            || w.offset_x.is_some()
            || w.offset_y.is_some()
            || w.intermediate_width.is_some()
            || w.intermediate_height.is_some()
            || w.pad_left.is_some()
            || w.pad_right.is_some()
            || w.pad_top.is_some()
            || w.pad_bottom.is_some()
            || w.crop_left.is_some()
            || w.crop_right.is_some()
            || w.crop_top.is_some()
            || w.crop_bottom.is_some();
        if has_metadata {
            let scale_x = w
                .scale_x
                .ok_or_else(|| serde::de::Error::custom("geometry scale_x missing"))?;
            let actual = [
                geometry.scale_x,
                geometry.scale_y,
                geometry.offset_x,
                geometry.offset_y,
            ];
            let supplied = [
                scale_x,
                w.scale_y
                    .ok_or_else(|| serde::de::Error::custom("geometry scale_y missing"))?,
                w.offset_x
                    .ok_or_else(|| serde::de::Error::custom("geometry offset_x missing"))?,
                w.offset_y
                    .ok_or_else(|| serde::de::Error::custom("geometry offset_y missing"))?,
            ];
            for (a, b) in actual.into_iter().zip(supplied) {
                if (a - b).abs() > 1e-12 {
                    return Err(serde::de::Error::custom(
                        "geometry transform metadata does not match dimensions/policy",
                    ));
                }
            }
            let ints = (w.intermediate_width, w.intermediate_height);
            if ints
                != (
                    Some(geometry.intermediate_width),
                    Some(geometry.intermediate_height),
                )
            {
                return Err(serde::de::Error::custom(
                    "geometry intermediate dimensions mismatch",
                ));
            }
            let supplied_meta = [
                w.pad_left,
                w.pad_right,
                w.pad_top,
                w.pad_bottom,
                w.crop_left,
                w.crop_right,
                w.crop_top,
                w.crop_bottom,
            ];
            let actual_meta = [
                geometry.pad_left,
                geometry.pad_right,
                geometry.pad_top,
                geometry.pad_bottom,
                geometry.crop_left,
                geometry.crop_right,
                geometry.crop_top,
                geometry.crop_bottom,
            ];
            if supplied_meta.iter().any(Option::is_none)
                || supplied_meta
                    .iter()
                    .zip(actual_meta)
                    .any(|(a, b)| a.unwrap() != b)
            {
                return Err(serde::de::Error::custom(
                    "geometry placement metadata mismatch",
                ));
            }
        }
        Ok(geometry)
    }
}

type GeometryResolved = (
    f64,
    f64,
    f64,
    f64,
    u32,
    u32,
    (u32, u32, u32, u32),
    (u32, u32, u32, u32),
);

fn resolve_geometry(
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    policy: &GeometryPolicy,
) -> Result<GeometryResolved> {
    let sw = source_width as f64;
    let sh = source_height as f64;
    let tw = target_width as f64;
    let th = target_height as f64;
    match policy {
        GeometryPolicy::Stretch => Ok((
            tw / sw,
            th / sh,
            0.0,
            0.0,
            target_width,
            target_height,
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )),
        GeometryPolicy::ContainPad | GeometryPolicy::Thumbnail => {
            let scale = if matches!(policy, GeometryPolicy::Thumbnail) {
                (tw / sw).min(th / sh).min(1.0)
            } else {
                (tw / sw).min(th / sh)
            };
            ensure!(
                scale.is_finite() && scale > 0.0,
                "contain geometry scale is invalid"
            );
            let scaled_w = (sw * scale).round().max(1.0);
            let scaled_h = (sh * scale).round().max(1.0);
            let sx = scaled_w / sw;
            let sy = scaled_h / sh;
            let excess_x = target_width - scaled_w as u32;
            let excess_y = target_height - scaled_h as u32;
            let left = excess_x / 2;
            let top = excess_y / 2;
            Ok((
                sx,
                sy,
                left as f64,
                top as f64,
                scaled_w as u32,
                scaled_h as u32,
                (left, excess_x - left, top, excess_y - top),
                (0, 0, 0, 0),
            ))
        }
        GeometryPolicy::CoverCrop => {
            let scale = (tw / sw).max(th / sh);
            ensure!(
                scale.is_finite() && scale > 0.0,
                "cover geometry scale is invalid"
            );
            let scaled_w = (sw * scale).round().max(1.0);
            let scaled_h = (sh * scale).round().max(1.0);
            let sx = scaled_w / sw;
            let sy = scaled_h / sh;
            let excess_x = scaled_w as u32 - target_width;
            let excess_y = scaled_h as u32 - target_height;
            let left = excess_x / 2;
            let top = excess_y / 2;
            Ok((
                sx,
                sy,
                -(left as f64),
                -(top as f64),
                scaled_w as u32,
                scaled_h as u32,
                (0, 0, 0, 0),
                (left, excess_x - left, top, excess_y - top),
            ))
        }
    }
}

fn clamp_index(value: i32, len: u32) -> Option<usize> {
    if value < 0 || value >= len as i32 {
        None
    } else {
        Some(value as usize)
    }
}

fn sample_rgb(
    image: &RgbImageF32,
    x: f32,
    y: f32,
    filter: ResizeFilter,
    zero_outside: bool,
) -> [f32; 3] {
    sample_kernel(
        x,
        y,
        image.width(),
        image.height(),
        filter,
        zero_outside,
        |ix, iy| image.data()[iy * image.width() as usize + ix],
    )
}

fn sample_mask(mask: &AlphaMask, x: f32, y: f32, filter: ResizeFilter, zero_outside: bool) -> f32 {
    sample_kernel(
        x,
        y,
        mask.width(),
        mask.height(),
        filter,
        zero_outside,
        |ix, iy| [mask.data()[iy * mask.width() as usize + ix]; 3],
    )[0]
}

fn sample_mask_rect(
    mask: &AlphaMask,
    x: f32,
    y: f32,
    filter: ResizeFilter,
    rect: (u32, u32, u32, u32),
) -> f32 {
    let (left, top, width, height) = rect;
    sample_kernel(
        x - left as f32,
        y - top as f32,
        width,
        height,
        filter,
        false,
        |ix, iy| [mask.data()[(iy + top as usize) * mask.width() as usize + ix + left as usize]; 3],
    )[0]
}

fn sample_kernel<const N: usize, F: Fn(usize, usize) -> [f32; N]>(
    x: f32,
    y: f32,
    width: u32,
    height: u32,
    filter: ResizeFilter,
    zero_outside: bool,
    get: F,
) -> [f32; N] {
    if !x.is_finite() || !y.is_finite() {
        return [0.0; N];
    }
    if matches!(filter, ResizeFilter::Nearest) {
        let ix = x.round() as i32;
        let iy = y.round() as i32;
        return match (clamp_index(ix, width), clamp_index(iy, height)) {
            (Some(ix), Some(iy)) => get(ix, iy),
            _ if zero_outside => [0.0; N],
            _ => get(
                ix.clamp(0, width as i32 - 1) as usize,
                iy.clamp(0, height as i32 - 1) as usize,
            ),
        };
    }
    let radius = match filter {
        ResizeFilter::Bicubic => 2,
        ResizeFilter::Lanczos3 => 3,
        _ => 1,
    };
    let mut out = [0.0; N];
    let mut sum = 0.0;
    let x0 = x.floor() as i32 - radius + 1;
    let y0 = y.floor() as i32 - radius + 1;
    for iy in y0..=y.floor() as i32 + radius {
        let wy = kernel_weight(y - iy as f32, filter);
        if wy == 0.0 {
            continue;
        }
        for ix in x0..=x.floor() as i32 + radius {
            let wx = kernel_weight(x - ix as f32, filter);
            let w = wx * wy;
            if w == 0.0 {
                continue;
            }
            if let (Some(cx), Some(cy)) = (clamp_index(ix, width), clamp_index(iy, height)) {
                let p = get(cx, cy);
                for c in 0..N {
                    out[c] += p[c] * w;
                }
                sum += w;
            } else if !zero_outside {
                let p = get(
                    ix.clamp(0, width as i32 - 1) as usize,
                    iy.clamp(0, height as i32 - 1) as usize,
                );
                for c in 0..N {
                    out[c] += p[c] * w;
                }
                sum += w;
            }
        }
    }
    if sum > 0.0 {
        for value in &mut out {
            *value = (*value / sum).clamp(0.0, 1.0);
        }
    }
    out
}

fn kernel_weight(distance: f32, filter: ResizeFilter) -> f32 {
    let d = distance.abs();
    match filter {
        ResizeFilter::Bilinear | ResizeFilter::Triangle => (1.0 - d).max(0.0),
        ResizeFilter::Bicubic => {
            let a = -0.5;
            if d < 1.0 {
                (a + 2.0) * d.powi(3) - (a + 3.0) * d.powi(2) + 1.0
            } else if d < 2.0 {
                a * d.powi(3) - 5.0 * a * d.powi(2) + 8.0 * a * d - 4.0 * a
            } else {
                0.0
            }
        }
        ResizeFilter::Lanczos3 => {
            if d < 3.0 {
                sinc(d) * sinc(d / 3.0)
            } else {
                0.0
            }
        }
        ResizeFilter::Nearest => unreachable!(),
    }
}

fn sinc(x: f32) -> f32 {
    if x.abs() < f32::EPSILON {
        1.0
    } else {
        let p = std::f32::consts::PI * x;
        p.sin() / p
    }
}

/// Foreground working-space declaration for later estimators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkingColorSpace {
    Srgb,
    Linear,
}

/// How predicted alpha interacts with alpha already present in the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TransparentInputPolicy {
    MultiplyPredicted,
    ReplaceSourceAlpha,
}
impl TransparentInputPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MultiplyPredicted => "multiply-predicted",
            Self::ReplaceSourceAlpha => "replace-source-alpha",
        }
    }
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
    transparent_input_policy: TransparentInputPolicy,
}
impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            schema: "m2.pipeline.v1".into(),
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
                scale_x: 1.0,
                scale_y: 1.0,
                offset_x: 0.0,
                offset_y: 0.0,
                intermediate_width: 1,
                intermediate_height: 1,
                pad_left: 0,
                pad_right: 0,
                pad_top: 0,
                pad_bottom: 0,
                crop_left: 0,
                crop_right: 0,
                crop_top: 0,
                crop_bottom: 0,
            },
            working_color_space: WorkingColorSpace::Srgb,
            output_mode: "straight-rgba".into(),
            parameters: BTreeMap::new(),
            transparent_input_policy: TransparentInputPolicy::MultiplyPredicted,
        }
    }
}
impl PipelineConfig {
    pub fn geometry(&self) -> &GeometryTransform {
        &self.geometry
    }
    pub fn transparent_input_policy(&self) -> TransparentInputPolicy {
        self.transparent_input_policy
    }
    pub fn with_transparent_input_policy(mut self, policy: TransparentInputPolicy) -> Self {
        self.transparent_input_policy = policy;
        self
    }
    /// M2 resolved configuration with an explicit transparent-input policy.
    pub fn canonical_json_m2(&self) -> Result<Vec<u8>> {
        self.canonical_json()
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "m1.pipeline.v1" || self.schema == "m2.pipeline.v1",
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
            c.geometry.policy,
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
    #[serde(default = "default_transparent_policy")]
    transparent_input_policy: TransparentInputPolicy,
}
fn default_transparent_policy() -> TransparentInputPolicy {
    TransparentInputPolicy::MultiplyPredicted
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
            transparent_input_policy: w.transparent_input_policy,
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
            self.config.geometry.source_dimensions() == expected,
            "pipeline geometry source dimensions do not match image"
        );
        let model_rgb = self.config.geometry.forward_rgb(image.rgb())?;
        let model_image = CanonicalImage::from_rgb(model_rgb);
        let coarse = self.segmenter.predict(&model_image, prompt)?;
        ensure!(
            coarse.dimensions() == self.config.geometry.target_dimensions(),
            "segmenter returned dimensions incompatible with model grid"
        );
        let coarse_data = coarse.data().to_vec();
        self.capture("coarse-alpha", coarse.dimensions(), &coarse_data)?;
        let transformed = self.mask_transform.apply(&model_image, coarse)?;
        ensure!(
            transformed.dimensions() == self.config.geometry.target_dimensions(),
            "mask transform returned dimensions incompatible with model grid"
        );
        let transformed_data = transformed.data().to_vec();
        self.capture(
            "transformed-alpha",
            transformed.dimensions(),
            &transformed_data,
        )?;
        let model_dims = self.config.geometry.target_dimensions();
        let trimap = Trimap::unknown(model_dims.0, model_dims.1)?;
        let trimap_values: Vec<f32> = trimap
            .data()
            .iter()
            .map(|c| match c {
                TrimapClass::Background => 0.0,
                TrimapClass::Unknown => 0.5,
                TrimapClass::Foreground => 1.0,
            })
            .collect();
        self.capture("trimap", model_dims, &trimap_values)?;
        let matte = self.refiner.refine(&model_image, &transformed, &trimap)?;
        ensure!(
            matte.alpha().dimensions() == model_dims,
            "refiner returned dimensions incompatible with model grid"
        );
        let alpha_data = matte.alpha().data().to_vec();
        self.capture("refined-alpha", model_dims, &alpha_data)?;
        let restored = self.config.geometry.inverse_mask(matte.alpha())?;
        let alpha = match self.config.transparent_input_policy {
            TransparentInputPolicy::MultiplyPredicted => AlphaMask::new(
                expected.0,
                expected.1,
                restored
                    .data()
                    .iter()
                    .zip(image.source_alpha().data())
                    .map(|(p, s)| (p * s).clamp(0.0, 1.0))
                    .collect(),
            )?,
            TransparentInputPolicy::ReplaceSourceAlpha => restored,
        };
        let source_matte = RefinedMatte::new(alpha, None, None)?;
        let rgb = self.foreground.estimate(image, &source_matte)?;
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
        Foreground::new(rgb, source_matte.alpha().clone())
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

    #[test]
    fn geometry_pixel_centres_and_padding_are_invertible() {
        let g = GeometryTransform::new(
            4,
            2,
            4,
            4,
            GeometryPolicy::ContainPad,
            ResizeFilter::Nearest,
        )
        .unwrap();
        assert_eq!(g.offsets(), (0.0, 1.0));
        let source = AlphaMask::ones(4, 2).unwrap();
        let model = g.forward_mask(&source).unwrap();
        assert_eq!(&model.data()[..4], &[0.0; 4]);
        assert_eq!(&model.data()[12..16], &[0.0; 4]);
        let restored = g.inverse_mask(&model).unwrap();
        assert_eq!(restored.data(), source.data());
        let (tx, ty) = g.forward_coordinate(1.5, 0.5).unwrap();
        let (sx, sy) = g.inverse_coordinate(tx, ty).unwrap();
        assert!((sx - 1.5).abs() < 1e-6 && (sy - 0.5).abs() < 1e-6);
    }

    #[test]
    fn all_resampling_filters_are_finite_and_dimension_safe() {
        let image = RgbImageF32::new(
            3,
            2,
            vec![
                [0.0, 0.0, 0.0],
                [0.2, 0.2, 0.2],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.7, 0.7, 0.7],
                [0.1, 0.1, 0.1],
            ],
        )
        .unwrap();
        let mut resized_images = Vec::new();
        for filter in [
            ResizeFilter::Nearest,
            ResizeFilter::Bilinear,
            ResizeFilter::Triangle,
            ResizeFilter::Bicubic,
            ResizeFilter::Lanczos3,
        ] {
            let g = GeometryTransform::new(3, 2, 7, 5, GeometryPolicy::Stretch, filter).unwrap();
            let resized = g.forward_rgb(&image).unwrap();
            assert_eq!(resized.dimensions(), (7, 5));
            assert!(resized
                .data()
                .iter()
                .flatten()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
            resized_images.push(resized);
        }
        assert!(
            (resized_images[1].data()[7][0] - resized_images[2].data()[7][0]).abs() < 1e-7,
            "triangle must be an explicit bilinear/tent path"
        );
        assert!(
            resized_images[0]
                .data()
                .iter()
                .all(|p| [0.0, 0.1, 0.2, 0.7, 1.0]
                    .iter()
                    .any(|v| (p[0] - v).abs() < 1e-7)),
            "nearest is not exact"
        );
        assert!(
            resized_images[3..]
                .iter()
                .zip(resized_images[1..].iter())
                .any(|(a, b)| a
                    .data()
                    .iter()
                    .zip(b.data())
                    .any(|(x, y)| (x[0] - y[0]).abs() > 1e-5)),
            "higher-order kernels collapsed to bilinear"
        );
    }

    #[test]
    fn geometry_policy_matrix_covers_aspect_and_crop_boundaries() {
        for (sw, sh, tw, th) in [(3, 3, 5, 5), (2, 4, 6, 3), (4, 2, 3, 6)] {
            for policy in [
                GeometryPolicy::Stretch,
                GeometryPolicy::ContainPad,
                GeometryPolicy::Thumbnail,
                GeometryPolicy::CoverCrop,
            ] {
                let g =
                    GeometryTransform::new(sw, sh, tw, th, policy, ResizeFilter::Nearest).unwrap();
                let round = serde_json::to_vec(&g).unwrap();
                let decoded: GeometryTransform = serde_json::from_slice(&round).unwrap();
                assert_eq!(g, decoded);
                assert_eq!(
                    g.forward_mask(&AlphaMask::zeros(sw, sh).unwrap())
                        .unwrap()
                        .dimensions(),
                    (tw, th)
                );
            }
        }
        let cover =
            GeometryTransform::new(3, 2, 2, 2, GeometryPolicy::CoverCrop, ResizeFilter::Nearest)
                .unwrap();
        let restored = cover.inverse_mask(&AlphaMask::ones(2, 2).unwrap()).unwrap();
        assert_eq!(restored.dimensions(), (3, 2));
        assert_eq!(
            restored.data()[2],
            0.0,
            "discarded cover region must remain outside/zero"
        );
        let thumbnail =
            GeometryTransform::new(2, 2, 5, 5, GeometryPolicy::Thumbnail, ResizeFilter::Nearest)
                .unwrap();
        assert_eq!(
            thumbnail.intermediate_dimensions(),
            (2, 2),
            "thumbnail may not upscale"
        );
    }

    #[test]
    fn odd_raster_excess_is_integer_and_padding_is_exact() {
        let horizontal = GeometryTransform::new(
            1,
            2,
            3,
            3,
            GeometryPolicy::ContainPad,
            ResizeFilter::Nearest,
        )
        .unwrap();
        assert_eq!(horizontal.intermediate_dimensions(), (2, 3));
        assert_eq!(horizontal.padding(), (0, 1, 0, 0));
        let h = horizontal
            .forward_mask(&AlphaMask::ones(1, 2).unwrap())
            .unwrap();
        assert_eq!(h.data(), &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let vertical = GeometryTransform::new(
            2,
            1,
            3,
            3,
            GeometryPolicy::ContainPad,
            ResizeFilter::Nearest,
        )
        .unwrap();
        assert_eq!(vertical.intermediate_dimensions(), (3, 2));
        assert_eq!(vertical.padding(), (0, 0, 0, 1));
        let v = vertical
            .forward_mask(&AlphaMask::ones(2, 1).unwrap())
            .unwrap();
        assert_eq!(v.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let impulse = GeometryTransform::new(
            1,
            1,
            5,
            3,
            GeometryPolicy::ContainPad,
            ResizeFilter::Bilinear,
        )
        .unwrap();
        let rgb = impulse
            .forward_rgb(&RgbImageF32::constant(1, 1, [1.0, 0.5, 0.25]).unwrap())
            .unwrap();
        assert_eq!(rgb.data()[0], [0.0; 3]);
        assert_eq!(rgb.data()[2], [1.0, 0.5, 0.25]);
        assert_eq!(rgb.data()[4], [0.0; 3]);
    }

    #[test]
    fn geometry_metadata_tampering_is_rejected() {
        let g = GeometryTransform::new(
            2,
            1,
            3,
            3,
            GeometryPolicy::ContainPad,
            ResizeFilter::Bilinear,
        )
        .unwrap();
        let value = serde_json::to_value(&g).unwrap();
        for field in ["scale_x", "offset_x", "intermediate_width", "pad_right"] {
            let mut tampered = value.clone();
            let entry = tampered.get_mut(field).unwrap();
            *entry = if field == "intermediate_width" || field == "pad_right" {
                serde_json::json!(99)
            } else {
                serde_json::json!(99.0)
            };
            assert!(
                serde_json::from_value::<GeometryTransform>(tampered).is_err(),
                "tampering {field} accepted"
            );
        }
        let mut legacy = value;
        legacy["policy"]["pad_x"] = serde_json::json!(2);
        assert!(serde_json::from_value::<GeometryTransform>(legacy).is_err());
    }

    #[test]
    fn filters_match_independent_slow_reference_samples() {
        let source = RgbImageF32::new(3, 1, vec![[0.0; 3], [0.25; 3], [1.0; 3]]).unwrap();
        let coords = [-0.2_f64, 0.4, 1.0, 1.6, 2.2];
        let bilinear = |x: f64| {
            let x = x.clamp(0.0, 2.0);
            let i = x.floor() as usize;
            let j = (i + 1).min(2);
            let t = x - i as f64;
            (1.0 - t) * [0.0, 0.25, 1.0][i] + t * [0.0, 0.25, 1.0][j]
        };
        let cubic_weight = |d: f64| {
            let d = d.abs();
            if d < 1.0 {
                1.5 * d.powi(3) - 2.5 * d.powi(2) + 1.0
            } else if d < 2.0 {
                -0.5 * d.powi(3) + 2.5 * d.powi(2) - 4.0 * d + 2.0
            } else {
                0.0
            }
        };
        let lanczos_weight = |d: f64| {
            let d = d.abs();
            if d >= 3.0 {
                0.0
            } else if d == 0.0 {
                1.0
            } else {
                let s = |x: f64| {
                    if x == 0.0 {
                        1.0
                    } else {
                        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                    }
                };
                s(d) * s(d / 3.0)
            }
        };
        let slow = |x: f64, weight: &dyn Fn(f64) -> f64| {
            let mut sum = 0.0;
            let mut out = 0.0;
            for i in -3..=5 {
                let w = weight(x - i as f64);
                let p = [0.0, 0.25, 1.0][i.clamp(0, 2) as usize];
                out += w * p;
                sum += w;
            }
            if sum == 0.0 {
                0.0
            } else {
                (out / sum).clamp(0.0, 1.0)
            }
        };
        for (filter, expected) in [
            (ResizeFilter::Bilinear, coords.map(bilinear)),
            (ResizeFilter::Triangle, coords.map(bilinear)),
            (
                ResizeFilter::Bicubic,
                coords.map(|x| slow(x, &cubic_weight)),
            ),
            (
                ResizeFilter::Lanczos3,
                coords.map(|x| slow(x, &lanczos_weight)),
            ),
        ] {
            let g = GeometryTransform::new(3, 1, 5, 1, GeometryPolicy::Stretch, filter).unwrap();
            let actual: Vec<f64> = g
                .forward_rgb(&source)
                .unwrap()
                .data()
                .iter()
                .map(|p| p[0] as f64)
                .collect();
            for (a, e) in actual.into_iter().zip(expected) {
                assert!((a - e).abs() <= 1e-5, "{filter:?}: {a} != {e}");
            }
        }
        let g = GeometryTransform::new(3, 1, 5, 1, GeometryPolicy::Stretch, ResizeFilter::Nearest)
            .unwrap();
        let actual: Vec<f32> = g
            .forward_rgb(&source)
            .unwrap()
            .data()
            .iter()
            .map(|p| p[0])
            .collect();
        assert_eq!(actual, vec![0.0, 0.0, 0.25, 1.0, 1.0]);
    }

    #[test]
    fn transparent_policy_is_applied_at_pipeline_output() {
        struct OnesSegmenter;
        impl Segmenter for OnesSegmenter {
            fn predict(
                &mut self,
                image: &CanonicalImage,
                _prompt: Option<&Prompt>,
            ) -> Result<AlphaMask> {
                AlphaMask::ones(image.width(), image.height())
            }
        }
        let image = CanonicalImage::new_with_alpha(
            1,
            1,
            vec![[0.25, 0.5, 0.75]],
            AlphaMask::new(1, 1, vec![0.5]).unwrap(),
        )
        .unwrap();
        let mut multiplied = Pipeline::new(
            PipelineConfig::default().resolved_for(1, 1).unwrap(),
            Box::new(OnesSegmenter),
            Box::new(NoOpMaskTransform),
            Box::new(NoOpAlphaRefiner),
            Box::new(NoOpForegroundEstimator),
        );
        assert_eq!(multiplied.run(&image, None).unwrap().alpha().data(), &[0.5]);
        let config = PipelineConfig::default()
            .with_transparent_input_policy(TransparentInputPolicy::ReplaceSourceAlpha)
            .resolved_for(1, 1)
            .unwrap();
        let mut replaced = Pipeline::new(
            config,
            Box::new(OnesSegmenter),
            Box::new(NoOpMaskTransform),
            Box::new(NoOpAlphaRefiner),
            Box::new(NoOpForegroundEstimator),
        );
        assert_eq!(replaced.run(&image, None).unwrap().alpha().data(), &[1.0]);
        let m2_bytes = replaced.config.canonical_json_m2().unwrap();
        let decoded = PipelineConfig::from_canonical_json(&m2_bytes).unwrap();
        assert_eq!(
            decoded.transparent_input_policy(),
            TransparentInputPolicy::ReplaceSourceAlpha
        );
    }
}
