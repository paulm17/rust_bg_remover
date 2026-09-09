//! Prompted Segment Anything (SAM) encoder/decoder contracts.
//!
//! This module follows the pinned `rembg` SAM adapter at commit
//! `030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709`.  The canonical image grid is
//! the only prompt coordinate space exposed at this boundary. Conversion to
//! the fixed 684x1024 SAM encoder canvas happens exactly once in
//! [`SamTransform`].
//! Keeping that transform explicit prevents a generic pipeline from applying
//! a second resize to already-transformed prompts.

use anyhow::{bail, ensure, Context, Result};
use bgremove_core::{AlphaMask, CanonicalImage, Prompt};
use bgremove_models::ModelManifest;

use crate::{RequestedProvider, TensorInput, TensorOutput, VerifiedSession};
use std::path::Path;

pub const SAM_TARGET_LENGTH: u32 = 1024;
pub const SAM_ENCODER_HEIGHT: u32 = 684;
pub const SAM_ENCODER_WIDTH: u32 = 1024;
pub const SAM_MASK_SIZE: u32 = 256;
const MAX_IMAGE_PIXELS: u64 = 64 * 1024 * 1024;
const MAX_DECODER_VALUES: usize = 64 * 1024 * 1024;

/// The exact model names accepted by the rembg release allowlist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamVariant {
    VitB,
    VitL,
    VitH,
}

impl SamVariant {
    pub const ALL: [Self; 3] = [Self::VitB, Self::VitL, Self::VitH];

    pub const fn id(self) -> &'static str {
        match self {
            Self::VitB => "sam_vit_b_01ec64",
            Self::VitL => "sam_vit_l_0b3195",
            Self::VitH => "sam_vit_h_4b8939",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamEncoding {
    Fp32,
    Quantized,
}

impl SamEncoding {
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Fp32 => "",
            Self::Quantized => ".quant",
        }
    }
}

/// A finite 256x256 low-resolution prior mask.  SAM's `has_mask_input` flag
/// is derived from `Option<SamPriorMask>`, never from whether a caller happens
/// to pass an all-zero buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct SamPriorMask {
    values: Vec<f32>,
}

impl SamPriorMask {
    pub fn new(values: Vec<f32>) -> Result<Self> {
        ensure!(
            values.len() == (SAM_MASK_SIZE * SAM_MASK_SIZE) as usize,
            "SAM prior mask must contain exactly 256x256 values"
        );
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "SAM prior mask contains NaN/Inf"
        );
        Ok(Self { values })
    }

    pub fn zeros() -> Self {
        Self {
            values: vec![0.0; (SAM_MASK_SIZE * SAM_MASK_SIZE) as usize],
        }
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// A prompt plus optional iterative SAM decoder state.
#[derive(Clone, Debug, PartialEq)]
pub struct SamPromptRequest {
    prompt: Prompt,
    prior: Option<SamPriorMask>,
    assistance_mode: SamAssistanceMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamAssistanceMode {
    Manual,
    CentreDefault,
    AutoDerived,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamRecordEligibility {
    AssistedNonAutomatic,
    AutomaticDerived,
}

impl SamAssistanceMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::CentreDefault => "centre-default-assisted",
            Self::AutoDerived => "auto-derived-assisted",
        }
    }
}

impl SamPromptRequest {
    pub fn new(prompt: Prompt) -> Result<Self> {
        Self::with_prior(prompt, None)
    }

    pub fn with_prior(prompt: Prompt, prior: Option<SamPriorMask>) -> Result<Self> {
        Self::with_mode(prompt, prior, SamAssistanceMode::Manual)
    }

    pub fn with_mode(
        prompt: Prompt,
        prior: Option<SamPriorMask>,
        assistance_mode: SamAssistanceMode,
    ) -> Result<Self> {
        ensure!(
            !prompt.points().is_empty() || prompt.box_region().is_some(),
            "SAM prompt must contain at least one point or an ordered box"
        );
        prompt.validate()?;
        Ok(Self {
            prompt,
            prior,
            assistance_mode,
        })
    }

    pub fn prompt(&self) -> &Prompt {
        &self.prompt
    }

    pub fn prior(&self) -> Option<&SamPriorMask> {
        self.prior.as_ref()
    }

    pub fn assistance_mode(&self) -> SamAssistanceMode {
        self.assistance_mode
    }

    /// Classify a prompt for benchmark/leaderboard serialization. Automatic
    /// eligibility is opt-in even for a coarse-component-derived prompt, so a
    /// manual or centre-default prompt can never leak into an automatic record.
    pub fn record_eligibility(
        &self,
        pipeline_declares_auto_derived: bool,
    ) -> Result<SamRecordEligibility> {
        match self.assistance_mode {
            SamAssistanceMode::Manual | SamAssistanceMode::CentreDefault => {
                Ok(SamRecordEligibility::AssistedNonAutomatic)
            }
            SamAssistanceMode::AutoDerived => {
                ensure!(
                    pipeline_declares_auto_derived,
                    "auto-derived SAM prompt requires an explicit pipeline declaration"
                );
                Ok(SamRecordEligibility::AutomaticDerived)
            }
        }
    }

    pub fn centre_default(width: u32, height: u32) -> Result<Self> {
        ensure!(
            width > 0 && height > 0,
            "centre-default dimensions must be positive"
        );
        let prompt = Prompt::new(
            vec![bgremove_core::PromptPoint::new(
                (width / 2) as f32,
                (height / 2) as f32,
                true,
            )?],
            None,
        )?;
        Self::with_mode(prompt, None, SamAssistanceMode::CentreDefault)
    }

    pub fn auto_derived(prompt: Prompt, prior: Option<SamPriorMask>) -> Result<Self> {
        Self::with_mode(prompt, prior, SamAssistanceMode::AutoDerived)
    }
}

/// Geometry shared by encoder preprocessing, prompt encoding and restoration.
/// `scale` is the rembg transform matrix's diagonal and is deliberately the
/// only canonical-to-model coordinate conversion in this adapter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamTransform {
    original_width: u32,
    original_height: u32,
    resized_width: u32,
    resized_height: u32,
    scale: f32,
    scale_exact: f64,
}

impl SamTransform {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        ensure!(
            width > 0 && height > 0,
            "SAM image dimensions must be positive"
        );
        ensure!(
            (width as u64) * (height as u64) <= MAX_IMAGE_PIXELS,
            "SAM image exceeds the bounded allocation limit"
        );
        // rembg's SamSession uses a fixed (height,width) encoder canvas of
        // (684,1024), then applies one diagonal affine scale:
        // min(1024/original_width, 684/original_height).
        let scale_exact = (SAM_ENCODER_WIDTH as f64 / width as f64)
            .min(SAM_ENCODER_HEIGHT as f64 / height as f64);
        let scale = scale_exact as f32;
        let resized_width = ((width as f64 * scale_exact) + 0.5).floor() as u32;
        let resized_height = ((height as f64 * scale_exact) + 0.5).floor() as u32;
        ensure!(
            resized_width > 0 && resized_height > 0,
            "SAM resize rounded to zero"
        );
        Ok(Self {
            original_width: width,
            original_height: height,
            resized_width,
            resized_height,
            scale,
            scale_exact,
        })
    }

    pub fn original_size(&self) -> (u32, u32) {
        (self.original_width, self.original_height)
    }

    pub fn resized_size(&self) -> (u32, u32) {
        (self.resized_width, self.resized_height)
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn scale_exact(&self) -> f64 {
        self.scale_exact
    }

    /// Convert a canonical pixel coordinate to the decoder's point space.
    /// This is equivalent to rembg's `apply_coords` followed by its affine
    /// matrix, simplified algebraically to one multiplication by `scale`.
    pub fn canonical_to_decoder(&self, x: f32, y: f32) -> Result<[f32; 2]> {
        ensure!(
            x.is_finite() && y.is_finite(),
            "SAM prompt coordinate is non-finite"
        );
        ensure!(
            x >= 0.0
                && x <= self.original_width as f32
                && y >= 0.0
                && y <= self.original_height as f32,
            "SAM prompt coordinate is outside the canonical image bounds"
        );
        Ok([
            (x as f64 * self.scale_exact) as f32,
            (y as f64 * self.scale_exact) as f32,
        ])
    }

    pub fn decoder_to_canonical(&self, x: f32, y: f32) -> Result<[f32; 2]> {
        ensure!(
            x.is_finite() && y.is_finite(),
            "SAM decoder coordinate is non-finite"
        );
        let result = [x as f64 / self.scale_exact, y as f64 / self.scale_exact];
        ensure!(
            result[0] >= -(f32::EPSILON as f64)
                && result[0] <= self.original_width as f64 + f32::EPSILON as f64
                && result[1] >= -(f32::EPSILON as f64)
                && result[1] <= self.original_height as f64 + f32::EPSILON as f64,
            "SAM decoder coordinate is outside the canonical image bounds"
        );
        Ok([result[0] as f32, result[1] as f32])
    }

    pub fn round_trip(&self, x: f32, y: f32) -> Result<[f32; 2]> {
        let p = self.canonical_to_decoder(x, y)?;
        self.decoder_to_canonical(p[0], p[1])
    }
}

/// Encoder input evidence.  The source adapter supplies an HWC float32 RGB
/// tensor in 0..255, with a fixed 684x1024 canvas and zero outside the scaled
/// source image.
pub fn sam_encoder_preprocess(image: &CanonicalImage) -> Result<(SamTransform, TensorInput)> {
    let transform = SamTransform::new(image.width(), image.height())?;
    let source = image
        .rgb()
        .data()
        .iter()
        .flat_map(|pixel| {
            pixel
                .iter()
                .map(|value| (*value * 255.0).round().clamp(0.0, 255.0) as f64)
        })
        .collect::<Vec<f64>>();
    ensure!(
        source.iter().all(|value| value.is_finite()),
        "SAM source RGB is non-finite"
    );
    let values = resize_affine_zero(
        &source,
        image.width(),
        image.height(),
        3,
        SAM_ENCODER_WIDTH,
        SAM_ENCODER_HEIGHT,
        transform.scale_exact(),
    )?;
    Ok((
        transform,
        TensorInput {
            shape: vec![SAM_ENCODER_HEIGHT as i64, SAM_ENCODER_WIDTH as i64, 3],
            values,
        },
    ))
}

/// Typed decoder inputs, including the exact padding point and -1 label.
#[derive(Clone, Debug, PartialEq)]
pub struct SamDecoderInputs {
    pub point_coords: TensorInput,
    pub point_labels: TensorInput,
    pub mask_input: TensorInput,
    pub has_mask_input: TensorInput,
    pub orig_im_size: TensorInput,
}

pub fn sam_encode_prompts(
    transform: &SamTransform,
    request: &SamPromptRequest,
) -> Result<SamDecoderInputs> {
    request
        .prompt
        .validate_for(transform.original_width, transform.original_height)?;
    let mut coords = Vec::new();
    let mut labels = Vec::new();
    for point in request.prompt.points() {
        let (x, y) = point.coordinates();
        let encoded = transform.canonical_to_decoder(x, y)?;
        coords.extend(encoded);
        labels.push(if point.positive() { 1.0 } else { 0.0 });
    }
    if let Some(region) = request.prompt.box_region() {
        let (x0, y0, x1, y1) = region.bounds();
        let a = transform.canonical_to_decoder(x0, y0)?;
        let b = transform.canonical_to_decoder(x1, y1)?;
        coords.extend([a[0], a[1], b[0], b[1]]);
        labels.extend([2.0, 3.0]);
    }
    ensure!(!labels.is_empty(), "SAM prompt produced no decoder labels");
    coords.extend([0.0, 0.0]);
    labels.push(-1.0);
    let prior = request
        .prior
        .as_ref()
        .map_or_else(SamPriorMask::zeros, Clone::clone);
    let has_prior = if request.prior.is_some() { 1.0 } else { 0.0 };
    Ok(SamDecoderInputs {
        point_coords: TensorInput {
            shape: vec![1, labels.len() as i64, 2],
            values: coords,
        },
        point_labels: TensorInput {
            shape: vec![1, labels.len() as i64],
            values: labels,
        },
        mask_input: TensorInput {
            shape: vec![1, 1, SAM_MASK_SIZE as i64, SAM_MASK_SIZE as i64],
            values: prior.values().to_vec(),
        },
        has_mask_input: TensorInput {
            shape: vec![1],
            values: vec![has_prior],
        },
        orig_im_size: TensorInput {
            shape: vec![2],
            values: vec![SAM_ENCODER_HEIGHT as f32, SAM_ENCODER_WIDTH as f32],
        },
    })
}

/// A candidate is retained verbatim instead of collapsing SAM's three output
/// hypotheses into an unconditional union.
#[derive(Clone, Debug, PartialEq)]
pub struct SamCandidate {
    pub index: usize,
    /// Decoder mask logits on the decoder output grid, before restoration.
    pub raw_logits: Vec<f32>,
    /// Decoder mask logits restored to the canonical image grid.
    pub restored_logits: Vec<f32>,
    pub restored_mask: AlphaMask,
    pub quality_score: f32,
    pub low_resolution_logits: Vec<f32>,
}

impl SamCandidate {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.quality_score.is_finite(),
            "SAM quality score is NaN/Inf"
        );
        ensure!(
            self.raw_logits.iter().all(|value| value.is_finite())
                && self.restored_logits.iter().all(|value| value.is_finite())
                && self
                    .low_resolution_logits
                    .iter()
                    .all(|value| value.is_finite()),
            "SAM candidate contains NaN/Inf logits"
        );
        ensure!(
            self.restored_logits.len() == self.restored_mask.len(),
            "SAM restored logits and mask dimensions differ"
        );
        ensure!(
            self.low_resolution_logits.len() == (SAM_MASK_SIZE * SAM_MASK_SIZE) as usize,
            "SAM candidate low-resolution logits must be 256x256"
        );
        Ok(())
    }

    pub fn prior_mask(&self) -> Result<SamPriorMask> {
        self.validate()?;
        SamPriorMask::new(self.low_resolution_logits.clone())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamSelectionPolicy {
    #[default]
    HighestQuality,
    /// This is only the source-compatible rembg behaviour, never the default
    /// quality policy.  It unions candidates after restoration in index order.
    SourceCompatibleUnion,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamSelection {
    pub policy: SamSelectionPolicy,
    pub selected: AlphaMask,
    pub selected_index: Option<usize>,
}

pub fn select_sam_candidates(
    candidates: &[SamCandidate],
    policy: SamSelectionPolicy,
) -> Result<SamSelection> {
    ensure!(!candidates.is_empty(), "SAM decoder returned no candidates");
    let mut indices = std::collections::BTreeSet::new();
    let dimensions = candidates[0].restored_mask.dimensions();
    for candidate in candidates {
        candidate.validate()?;
        ensure!(
            candidate.restored_mask.dimensions() == dimensions,
            "SAM candidate dimensions differ"
        );
        ensure!(
            indices.insert(candidate.index),
            "SAM candidate indices must be unique"
        );
    }
    match policy {
        SamSelectionPolicy::HighestQuality => {
            let selected = candidates
                .iter()
                .enumerate()
                .max_by(|(left_pos, left), (right_pos, right)| {
                    left.quality_score
                        .total_cmp(&right.quality_score)
                        .then_with(|| right.index.cmp(&left.index))
                        .then_with(|| right_pos.cmp(left_pos))
                })
                .expect("non-empty candidate list");
            Ok(SamSelection {
                policy,
                selected: selected.1.restored_mask.clone(),
                selected_index: Some(selected.1.index),
            })
        }
        SamSelectionPolicy::SourceCompatibleUnion => {
            let dimensions = candidates[0].restored_mask.dimensions();
            let mut values = vec![0.0f32; candidates[0].restored_mask.len()];
            for candidate in candidates {
                ensure!(
                    candidate.restored_mask.dimensions() == dimensions,
                    "SAM candidate dimensions differ"
                );
                for (target, source) in values.iter_mut().zip(candidate.restored_mask.data()) {
                    *target = (*target).max(*source);
                }
            }
            Ok(SamSelection {
                policy,
                selected: AlphaMask::new(dimensions.0, dimensions.1, values)?,
                selected_index: None,
            })
        }
    }
}

/// Provenance carried by the optional coarse-component auto-prompt experiment.
/// It is intentionally not interchangeable with a human prompt in reports or
/// leaderboard records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoPromptProvenance {
    pub mode: &'static str,
    pub source: &'static str,
    pub component_label: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutoPromptComponent {
    pub prompt: SamPromptRequest,
    pub provenance: AutoPromptProvenance,
    pub area: usize,
}

/// Derive one deterministic point+box prompt per connected foreground
/// component of a coarse alpha mask.  Components use 4-connectivity, labels
/// are row-major discovery order, and the point is the component pixel with
/// minimum squared distance to the integer centroid (then row-major tie).
pub fn derive_auto_prompts(coarse: &AlphaMask) -> Result<Vec<AutoPromptComponent>> {
    let (width, height) = coarse.dimensions();
    let mut visited = vec![false; coarse.len()];
    let mut components = Vec::new();
    let mut label = 0usize;
    for start in 0..coarse.len() {
        if visited[start] || coarse.data()[start] <= 0.5 {
            continue;
        }
        visited[start] = true;
        let mut queue = std::collections::VecDeque::from([start]);
        let mut pixels = Vec::new();
        while let Some(index) = queue.pop_front() {
            pixels.push(index);
            let x = index % width as usize;
            let y = index / width as usize;
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx >= width as usize || ny >= height as usize {
                    continue;
                }
                let neighbor = ny * width as usize + nx;
                if !visited[neighbor] && coarse.data()[neighbor] > 0.5 {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        let mut min_x = width as usize;
        let mut min_y = height as usize;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let (sum_x, sum_y) = pixels.iter().fold((0usize, 0usize), |(sx, sy), index| {
            let x = index % width as usize;
            let y = index / width as usize;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            (sx + x, sy + y)
        });
        let centroid_x = sum_x as f32 / pixels.len() as f32;
        let centroid_y = sum_y as f32 / pixels.len() as f32;
        let point = pixels
            .iter()
            .copied()
            .min_by(|left, right| {
                let l = *left % width as usize;
                let ly = *left / width as usize;
                let r = *right % width as usize;
                let ry = *right / width as usize;
                let ld = (l as f32 - centroid_x).powi(2) + (ly as f32 - centroid_y).powi(2);
                let rd = (r as f32 - centroid_x).powi(2) + (ry as f32 - centroid_y).powi(2);
                ld.total_cmp(&rd).then_with(|| left.cmp(right))
            })
            .expect("component is non-empty");
        let px = point % width as usize;
        let py = point / width as usize;
        let prompt = Prompt::new(
            vec![bgremove_core::PromptPoint::new(px as f32, py as f32, true)?],
            Some(bgremove_core::PromptBox::new(
                min_x as f32,
                min_y as f32,
                max_x as f32,
                max_y as f32,
            )?),
        )?;
        components.push(AutoPromptComponent {
            prompt: SamPromptRequest::auto_derived(prompt, None)?,
            provenance: AutoPromptProvenance {
                mode: "auto-prompt",
                source: "coarse-connected-components",
                component_label: label,
            },
            area: pixels.len(),
        });
        label += 1;
    }
    Ok(components)
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamRunEvidence {
    pub assistance_mode: SamAssistanceMode,
    pub transform: SamTransform,
    pub encoder_input: TensorInput,
    pub embedding: TensorOutput,
    pub decoder_inputs: SamDecoderInputs,
    pub candidates: Vec<SamCandidate>,
    pub selection: SamSelection,
}

/// Real two-session ORT SAM adapter.  The caller supplies separately pinned
/// encoder and decoder manifests; no model download or implicit fallback is
/// performed here.
pub struct SamSegmenter {
    encoder: VerifiedSession,
    decoder: VerifiedSession,
    selection_policy: SamSelectionPolicy,
}

pub fn require_explicit_sam_prompt(prompt: Option<&Prompt>) -> Result<&Prompt> {
    prompt.ok_or_else(|| {
        anyhow::anyhow!(
            "SAM Segmenter requires an explicit prompt; centre defaults are assisted-only"
        )
    })
}

impl SamSegmenter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        encoder_manifest: &ModelManifest,
        encoder_manifest_path: &Path,
        decoder_manifest: &ModelManifest,
        decoder_manifest_path: &Path,
        runtime: &Path,
        variant: SamVariant,
        encoding: SamEncoding,
        requested: RequestedProvider,
        fallback_allowed: bool,
        selection_policy: SamSelectionPolicy,
    ) -> Result<Self> {
        validate_manifest(encoder_manifest, variant, encoding, true)?;
        validate_manifest(decoder_manifest, variant, encoding, false)?;
        ensure!(
            encoder_manifest.model_variant == decoder_manifest.model_variant,
            "SAM encoder/decoder variants differ"
        );
        let encoder = VerifiedSession::open(
            encoder_manifest,
            encoder_manifest_path,
            runtime,
            requested,
            fallback_allowed,
        )
        .context("open verified SAM encoder session")?;
        let decoder = VerifiedSession::open(
            decoder_manifest,
            decoder_manifest_path,
            runtime,
            requested,
            fallback_allowed,
        )
        .context("open verified SAM decoder session")?;
        Ok(Self {
            encoder,
            decoder,
            selection_policy,
        })
    }

    pub fn predict_with_evidence(
        &mut self,
        image: &CanonicalImage,
        request: &SamPromptRequest,
    ) -> Result<SamRunEvidence> {
        let (transform, encoder_input) = sam_encoder_preprocess(image)?;
        let embedding = self
            .encoder
            .run(&encoder_input.shape, &encoder_input.values)?;
        let decoder_inputs = sam_encode_prompts(&transform, request)?;
        let inputs = [
            (
                "image_embeddings",
                embedding.shape.as_slice(),
                embedding.values.as_slice(),
            ),
            (
                "point_coords",
                decoder_inputs.point_coords.shape.as_slice(),
                decoder_inputs.point_coords.values.as_slice(),
            ),
            (
                "point_labels",
                decoder_inputs.point_labels.shape.as_slice(),
                decoder_inputs.point_labels.values.as_slice(),
            ),
            (
                "mask_input",
                decoder_inputs.mask_input.shape.as_slice(),
                decoder_inputs.mask_input.values.as_slice(),
            ),
            (
                "has_mask_input",
                decoder_inputs.has_mask_input.shape.as_slice(),
                decoder_inputs.has_mask_input.values.as_slice(),
            ),
            (
                "orig_im_size",
                decoder_inputs.orig_im_size.shape.as_slice(),
                decoder_inputs.orig_im_size.values.as_slice(),
            ),
        ];
        // The pinned rembg graph has six total decoder inputs: the primary
        // embedding plus five auxiliary prompt/state inputs. Manifests are
        // validated at construction, and names are kept
        // explicit here so an input cannot silently move to another slot.
        let outputs = self
            .decoder
            .run_named(&inputs, &["masks", "iou_predictions", "low_res_masks"])?;
        ensure!(
            outputs.len() == 3,
            "SAM decoder must return masks, quality and low-res outputs"
        );
        let candidates = decode_sam_outputs(&outputs[0], &outputs[1], &outputs[2], &transform)?;
        let selection = select_sam_candidates(&candidates, self.selection_policy)?;
        Ok(SamRunEvidence {
            assistance_mode: request.assistance_mode(),
            transform,
            encoder_input,
            embedding,
            decoder_inputs,
            candidates,
            selection,
        })
    }

    pub fn predict_prompted(
        &mut self,
        image: &CanonicalImage,
        request: &SamPromptRequest,
    ) -> Result<AlphaMask> {
        Ok(self
            .predict_with_evidence(image, request)?
            .selection
            .selected)
    }

    pub fn selection_policy(&self) -> SamSelectionPolicy {
        self.selection_policy
    }
}

// Intentionally no `Segmenter` implementation: `Pipeline::run` may transform
// the image to a model grid while forwarding the original-grid prompt. Until
// the core pipeline exposes a coordinate-aware prompt transform, the standalone
// `predict_prompted` API is the only safe integration surface.

fn validate_manifest(
    manifest: &ModelManifest,
    variant: SamVariant,
    encoding: SamEncoding,
    encoder: bool,
) -> Result<()> {
    ensure!(
        manifest.algorithm_family == "sam",
        "SAM manifest must use algorithm_family=sam"
    );
    ensure!(
        manifest.model_variant == variant.id(),
        "SAM manifest variant mismatch"
    );
    ensure!(
        manifest.model_encoding
            == match encoding {
                SamEncoding::Fp32 => bgremove_models::ModelEncoding::Fp32,
                SamEncoding::Quantized => bgremove_models::ModelEncoding::Quantized,
            },
        "SAM manifest encoding mismatch"
    );
    if encoder {
        ensure!(
            manifest.input_name == "input_image" || manifest.input_name == "image",
            "SAM encoder input must be input_image or image"
        );
        ensure!(
            manifest.output_name == "image_embeddings",
            "SAM encoder output must be image_embeddings"
        );
        ensure!(
            manifest.input_shape
                == vec![
                    bgremove_models::DimensionSpec::Static(SAM_ENCODER_HEIGHT as u64),
                    bgremove_models::DimensionSpec::Static(SAM_ENCODER_WIDTH as u64),
                    bgremove_models::DimensionSpec::Static(3),
                ],
            "SAM encoder shape metadata mismatch"
        );
    } else {
        ensure!(
            manifest.input_name == "image_embeddings",
            "SAM decoder primary input must be image_embeddings"
        );
        ensure!(
            manifest.auxiliary_input_names
                == vec![
                    "point_coords",
                    "point_labels",
                    "mask_input",
                    "has_mask_input",
                    "orig_im_size"
                ],
            "SAM decoder input names mismatch"
        );
        ensure!(
            manifest.output_name == "masks",
            "SAM decoder output must be masks"
        );
        ensure!(
            manifest.output_shape
                == vec![
                    bgremove_models::DimensionSpec::Static(1),
                    bgremove_models::DimensionSpec::Static(3),
                    bgremove_models::DimensionSpec::Dynamic("mask_height".into()),
                    bgremove_models::DimensionSpec::Dynamic("mask_width".into()),
                ],
            "SAM decoder masks must use dynamic orig_im_size dimensions"
        );
    }
    manifest.validate()
}

pub fn decode_sam_outputs(
    masks: &TensorOutput,
    quality: &TensorOutput,
    low_res: &TensorOutput,
    transform: &SamTransform,
) -> Result<Vec<SamCandidate>> {
    let (count_i64, height_i64, width_i64) = match masks.shape.as_slice() {
        [batch, n, h, w] => {
            ensure!(*batch == 1, "SAM masks batch dimension must be one");
            (*n, *h, *w)
        }
        other => bail!("SAM masks output shape {other:?} must be [1,N,H,W]"),
    };
    ensure!(
        count_i64 > 0 && height_i64 > 0 && width_i64 > 0,
        "SAM masks dimensions must be positive"
    );
    let count = usize::try_from(count_i64).context("SAM candidate count does not fit usize")?;
    let height = u32::try_from(height_i64).context("SAM mask height does not fit u32")?;
    let width = u32::try_from(width_i64).context("SAM mask width does not fit u32")?;
    let plane = checked_decoder_product(&[height_i64, width_i64], "SAM mask plane")?;
    let mask_values = checked_decoder_product(&[count_i64, height_i64, width_i64], "SAM masks")?;
    ensure!(
        mask_values <= MAX_DECODER_VALUES,
        "SAM mask output exceeds allocation limit"
    );
    ensure!(count > 0, "SAM decoder returned zero candidates");
    let quality_values = flatten_quality(quality)?;
    ensure!(
        quality_values.len() == count,
        "SAM candidate/quality count mismatch"
    );
    let low_shape = low_res.shape.as_slice();
    ensure!(
        low_shape.len() == 4 && low_shape[0] == 1,
        "SAM low-resolution output shape mismatch"
    );
    let low_count = low_shape[1];
    ensure!(
        low_count > 0
            && low_shape[2] == SAM_MASK_SIZE as i64
            && low_shape[3] == SAM_MASK_SIZE as i64
            && low_count == count_i64,
        "SAM low-resolution output shape mismatch"
    );
    let low_values = checked_decoder_product(&low_shape[1..], "SAM low-resolution masks")?;
    ensure!(
        low_values <= MAX_DECODER_VALUES,
        "SAM low-resolution output exceeds allocation limit"
    );
    ensure!(
        low_res.values.len() == low_values,
        "SAM low-resolution shape/value mismatch"
    );
    ensure!(
        low_res.values.iter().all(|value| value.is_finite()),
        "SAM low-resolution logits contain NaN/Inf"
    );
    ensure!(
        masks.values.len() == mask_values,
        "SAM masks shape/value mismatch"
    );
    let mut result = Vec::with_capacity(count);
    for (index, quality_score) in quality_values.iter().copied().enumerate().take(count) {
        let logits = masks.values[index * plane..(index + 1) * plane].to_vec();
        ensure!(
            logits.iter().all(|value| value.is_finite()),
            "SAM mask logits contain NaN/Inf"
        );
        let restored_logits = restore_sam_mask(&logits, width, height, transform)?;
        let restored_mask = AlphaMask::new(
            transform.original_width,
            transform.original_height,
            restored_logits
                .iter()
                .map(|value| if *value > 0.0 { 1.0 } else { 0.0 })
                .collect(),
        )?;
        let low_plane = (SAM_MASK_SIZE * SAM_MASK_SIZE) as usize;
        result.push(SamCandidate {
            index,
            raw_logits: logits,
            restored_logits,
            restored_mask,
            quality_score,
            low_resolution_logits: low_res.values[index * low_plane..(index + 1) * low_plane]
                .to_vec(),
        });
    }
    Ok(result)
}

fn flatten_quality(output: &TensorOutput) -> Result<Vec<f32>> {
    ensure!(
        output.values.iter().all(|value| value.is_finite()),
        "SAM quality output contains NaN/Inf"
    );
    let n = match output.shape.as_slice() {
        [batch, n] => {
            ensure!(*batch == 1, "SAM quality batch dimension must be one");
            *n
        }
        [n] => *n,
        other => bail!("SAM quality output shape {other:?} must be [1,N] or [N]"),
    };
    ensure!(n > 0, "SAM quality candidate count must be positive");
    let n = usize::try_from(n).context("SAM quality count does not fit usize")?;
    ensure!(
        n <= MAX_DECODER_VALUES,
        "SAM quality output exceeds allocation limit"
    );
    ensure!(output.values.len() == n, "SAM quality shape/value mismatch");
    Ok(output.values.clone())
}

fn checked_decoder_product(shape: &[i64], label: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, dimension| {
        ensure!(
            *dimension > 0,
            "{label} shape contains non-positive dimension"
        );
        let dimension = usize::try_from(*dimension)
            .with_context(|| format!("{label} dimension does not fit usize"))?;
        let total = total
            .checked_mul(dimension)
            .ok_or_else(|| anyhow::anyhow!("{label} element count overflows usize"))?;
        ensure!(
            total <= MAX_DECODER_VALUES,
            "{label} exceeds allocation limit"
        );
        Ok(total)
    })
}

fn restore_sam_mask(
    mask: &[f32],
    width: u32,
    height: u32,
    transform: &SamTransform,
) -> Result<Vec<f32>> {
    ensure!(
        mask.len() == width as usize * height as usize,
        "SAM mask dimensions/value mismatch"
    );
    // rembg computes inv_transform_matrix and passes it to warp_affine;
    // warp_affine inverts that matrix internally. The destination sample is
    // therefore at decoder coordinate (x*scale, y*scale), not x/scale.
    let decoder_scale = transform.scale_exact;
    let mut output =
        vec![0.0; transform.original_width as usize * transform.original_height as usize];
    for y in 0..transform.original_height as usize {
        let source_y = y as f64 * decoder_scale;
        for x in 0..transform.original_width as usize {
            let source_x = x as f64 * decoder_scale;
            output[y * transform.original_width as usize + x] =
                sample_bilinear_zero(mask, width, height, source_x, source_y);
        }
    }
    ensure!(
        output.iter().all(|value| value.is_finite()),
        "SAM restored mask contains NaN/Inf"
    );
    Ok(output)
}

fn sample_bilinear_zero(values: &[f32], width: u32, height: u32, x: f64, y: f64) -> f32 {
    if x < 0.0 || y < 0.0 || x > width as f64 - 1.0 || y > height as f64 - 1.0 {
        return 0.0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width as usize - 1);
    let y1 = (y0 + 1).min(height as usize - 1);
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    let at = |xx: usize, yy: usize| values[yy * width as usize + xx];
    ((1.0 - dx) * (1.0 - dy) * at(x0, y0) as f64
        + dx * (1.0 - dy) * at(x1, y0) as f64
        + (1.0 - dx) * dy * at(x0, y1) as f64
        + dx * dy * at(x1, y1) as f64) as f32
}

fn resize_affine_zero(
    src: &[f64],
    src_width: u32,
    src_height: u32,
    channels: usize,
    dst_width: u32,
    dst_height: u32,
    scale: f64,
) -> Result<Vec<f32>> {
    ensure!(
        src.len() == src_width as usize * src_height as usize * channels,
        "SAM source tensor length mismatch"
    );
    let mut out = vec![0.0; dst_width as usize * dst_height as usize * channels];
    // scipy's affine helper obtains these coordinates by multiplying with the
    // inverse matrix (rather than dividing each coordinate).  Retaining that
    // operation order matters at the uint8 truncation boundary.
    let inverse_scale = 1.0 / scale;
    for y in 0..dst_height as usize {
        let sy = y as f64 * inverse_scale;
        for x in 0..dst_width as usize {
            let sx = x as f64 * inverse_scale;
            for c in 0..channels {
                out[(y * dst_width as usize + x) * channels + c] =
                    sample_channel_zero(src, src_width, src_height, channels, sx, sy, c)
                        .floor()
                        .clamp(0.0, 255.0) as f32;
            }
        }
    }
    Ok(out)
}

fn sample_channel_zero(
    values: &[f64],
    width: u32,
    height: u32,
    channels: usize,
    x: f64,
    y: f64,
    channel: usize,
) -> f64 {
    if x < 0.0 || y < 0.0 || x > width as f64 - 1.0 || y > height as f64 - 1.0 {
        return 0.0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width as usize - 1);
    let y1 = (y0 + 1).min(height as usize - 1);
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    let at = |xx: usize, yy: usize| values[(yy * width as usize + xx) * channels + channel];
    (1.0 - dy) * ((1.0 - dx) * at(x0, y0) + dx * at(x1, y0))
        + dy * ((1.0 - dx) * at(x0, y1) + dx * at(x1, y1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bgremove_core::{PromptBox, PromptPoint};

    fn prompt() -> SamPromptRequest {
        SamPromptRequest::new(
            Prompt::new(
                vec![
                    PromptPoint::new(5.0, 2.0, true).unwrap(),
                    PromptPoint::new(8.0, 3.0, false).unwrap(),
                ],
                Some(PromptBox::new(1.0, 1.0, 9.0, 5.0).unwrap()),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn rembg_geometry_round_trips_all_aspect_categories() {
        for (w, h) in [(2, 3), (3, 2), (4, 4)] {
            let transform = SamTransform::new(w, h).unwrap();
            for (x, y) in [
                (0.0, 0.0),
                (w as f32, h as f32),
                (w as f32 / 2.0, h as f32 / 2.0),
            ] {
                let round = transform.round_trip(x, y).unwrap();
                assert!((round[0] - x).abs() < 1e-5 && (round[1] - y).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn rembg_scale_matches_fixed_canvas_numeric_contract() {
        let portrait = SamTransform::new(500, 1000).unwrap();
        assert!((portrait.scale() - 0.684).abs() < 1e-6);
        assert_eq!(portrait.resized_size(), (342, 684));
        assert_eq!(
            portrait.canonical_to_decoder(500.0, 1000.0).unwrap(),
            [342.0, 684.0]
        );
        let square = SamTransform::new(1000, 1000).unwrap();
        assert!((square.scale() - 0.684).abs() < 1e-6);
        assert_eq!(square.resized_size(), (684, 684));
        let landscape = SamTransform::new(1600, 800).unwrap();
        assert!((landscape.scale() - 0.64).abs() < 1e-6);
        assert_eq!(landscape.resized_size(), (1024, 512));
    }

    #[test]
    fn restoration_samples_decoder_at_encoder_scale_for_all_aspects() {
        for (width, height) in [(5, 3), (3, 5), (4, 4)] {
            let transform = SamTransform::new(width, height).unwrap();
            let decoder_width = 1024u32;
            let decoder_height = 684u32;
            let mask = (0..decoder_height as usize)
                .flat_map(|y| (0..decoder_width as usize).map(move |x| x as f32 + y as f32 * 0.25))
                .collect::<Vec<_>>();
            let restored =
                restore_sam_mask(&mask, decoder_width, decoder_height, &transform).unwrap();
            for (x, y) in [
                (0u32, 0u32),
                (width - 1, height - 1),
                (width / 2, height / 2),
            ] {
                let source_x = x as f64 * transform.scale_exact();
                let source_y = y as f64 * transform.scale_exact();
                let expected =
                    sample_bilinear_zero(&mask, decoder_width, decoder_height, source_x, source_y);
                assert!(
                    (restored[y as usize * width as usize + x as usize] - expected).abs() < 1e-5
                );
            }
        }
    }

    #[test]
    fn prompt_encoding_preserves_positive_negative_box_order_and_padding() {
        let transform = SamTransform::new(10, 6).unwrap();
        let encoded = sam_encode_prompts(&transform, &prompt()).unwrap();
        assert_eq!(encoded.point_labels.values, vec![1.0, 0.0, 2.0, 3.0, -1.0]);
        assert_eq!(encoded.point_coords.values.len(), 10);
        assert_eq!(&encoded.point_coords.values[8..], &[0.0, 0.0]);
        assert_eq!(encoded.has_mask_input.values, vec![0.0]);
    }

    #[test]
    fn prior_shape_and_finite_checks_are_strict() {
        assert!(SamPriorMask::new(vec![0.0; 255 * 256]).is_err());
        assert!(SamPriorMask::new(vec![f32::NAN; 256 * 256]).is_err());
        let prior = SamPriorMask::new(vec![0.25; 256 * 256]).unwrap();
        let encoded = sam_encode_prompts(
            &SamTransform::new(10, 6).unwrap(),
            &SamPromptRequest::with_prior(prompt().prompt.clone(), Some(prior)).unwrap(),
        )
        .unwrap();
        assert_eq!(encoded.has_mask_input.values, vec![1.0]);
    }

    #[test]
    fn quality_selection_is_not_an_unconditional_union_and_ties_are_stable() {
        assert_eq!(
            SamSelectionPolicy::default(),
            SamSelectionPolicy::HighestQuality
        );
        let alpha_a = AlphaMask::new(2, 1, vec![1.0, 0.0]).unwrap();
        let alpha_b = AlphaMask::new(2, 1, vec![0.0, 1.0]).unwrap();
        let make = |index, score, mask| SamCandidate {
            index,
            raw_logits: vec![0.0; 2],
            restored_logits: vec![0.0; 2],
            restored_mask: mask,
            quality_score: score,
            low_resolution_logits: vec![0.0; 256 * 256],
        };
        let candidates = vec![make(0, 0.5, alpha_a.clone()), make(1, 0.5, alpha_b.clone())];
        assert_eq!(
            candidates[0].prior_mask().unwrap().values().len(),
            256 * 256
        );
        let selected =
            select_sam_candidates(&candidates, SamSelectionPolicy::HighestQuality).unwrap();
        assert_eq!(selected.selected_index, Some(0));
        assert_eq!(selected.selected, alpha_a);
        let union =
            select_sam_candidates(&candidates, SamSelectionPolicy::SourceCompatibleUnion).unwrap();
        assert_eq!(union.selected.data(), &[1.0, 1.0]);
        let mut duplicate = candidates.clone();
        duplicate[1].index = 0;
        assert!(select_sam_candidates(&duplicate, SamSelectionPolicy::HighestQuality).is_err());
        let mut nonfinite = candidates.clone();
        nonfinite[0].quality_score = f32::NAN;
        assert!(select_sam_candidates(&nonfinite, SamSelectionPolicy::HighestQuality).is_err());
    }

    #[test]
    fn auto_prompt_components_are_deterministic_and_handle_zero_full_masks() {
        let zero = AlphaMask::zeros(3, 2).unwrap();
        assert!(derive_auto_prompts(&zero).unwrap().is_empty());
        let full = AlphaMask::ones(3, 2).unwrap();
        let prompts = derive_auto_prompts(&full).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].area, 6);
        assert_eq!(prompts[0].provenance.mode, "auto-prompt");
        assert_eq!(
            prompts[0].prompt.prompt().box_region().unwrap().bounds(),
            (0.0, 0.0, 2.0, 1.0)
        );
        let disconnected = AlphaMask::new(3, 2, vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0]).unwrap();
        let parts = derive_auto_prompts(&disconnected).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].provenance.component_label, 0);
        assert_eq!(parts[1].provenance.component_label, 1);
    }

    #[test]
    fn decoder_candidate_shape_and_finiteness_checks_fail_closed() {
        let transform = SamTransform::new(2, 2).unwrap();
        let masks = TensorOutput {
            shape: vec![1, 2, 1, 1],
            values: vec![0.0, 1.0],
        };
        let quality = TensorOutput {
            shape: vec![1, 3],
            values: vec![0.1, 0.2, 0.3],
        };
        let low = TensorOutput {
            shape: vec![1, 2, 256, 256],
            values: vec![0.0; 2 * 256 * 256],
        };
        assert!(decode_sam_outputs(&masks, &quality, &low, &transform).is_err());
        let bad_quality = TensorOutput {
            shape: vec![1, 2],
            values: vec![f32::NAN, 0.2],
        };
        assert!(decode_sam_outputs(&masks, &bad_quality, &low, &transform).is_err());
        for bad_shape in [vec![1, -1, 1, 1], vec![1, 1, 0, 1], vec![1, 1, 1, -1]] {
            let hostile = TensorOutput {
                shape: bad_shape,
                values: vec![],
            };
            assert!(decode_sam_outputs(&hostile, &quality, &low, &transform).is_err());
        }
        let bad_low = TensorOutput {
            shape: vec![1, 2, 256, 256],
            values: vec![0.0; 2 * 256 * 256 - 1],
        };
        assert!(decode_sam_outputs(
            &masks,
            &TensorOutput {
                shape: vec![1, 2],
                values: vec![0.1, 0.2]
            },
            &bad_low,
            &transform
        )
        .is_err());
        assert!(decode_sam_outputs(
            &masks,
            &TensorOutput {
                shape: vec![1, -1],
                values: vec![]
            },
            &low,
            &transform
        )
        .is_err());
    }

    #[test]
    fn assistance_modes_are_explicit_and_none_is_not_a_default() {
        let manual = prompt();
        assert_eq!(manual.assistance_mode(), SamAssistanceMode::Manual);
        let centre = SamPromptRequest::centre_default(5, 3).unwrap();
        assert_eq!(centre.assistance_mode(), SamAssistanceMode::CentreDefault);
        let auto = SamPromptRequest::auto_derived(manual.prompt().clone(), None).unwrap();
        assert_eq!(auto.assistance_mode(), SamAssistanceMode::AutoDerived);
        assert_eq!(
            manual.record_eligibility(false).unwrap(),
            SamRecordEligibility::AssistedNonAutomatic
        );
        assert_eq!(
            centre.record_eligibility(false).unwrap(),
            SamRecordEligibility::AssistedNonAutomatic
        );
        assert!(auto.record_eligibility(false).is_err());
        assert_eq!(
            auto.record_eligibility(true).unwrap(),
            SamRecordEligibility::AutomaticDerived
        );
        assert!(require_explicit_sam_prompt(None).is_err());
        assert!(require_explicit_sam_prompt(Some(manual.prompt())).is_ok());
        assert!(SamPromptRequest::new(Prompt::new(vec![], None).unwrap()).is_err());
    }

    #[test]
    fn generic_pipeline_geometry_pairing_fails_closed() {
        // Pipeline::run may pass a resized model image with an original-grid
        // prompt. The standalone SAM encoder contract rejects that pairing;
        // there is deliberately no generic Segmenter implementation.
        let prompt = Prompt::new(
            vec![bgremove_core::PromptPoint::new(499.0, 999.0, true).unwrap()],
            None,
        )
        .unwrap();
        let request = SamPromptRequest::new(prompt).unwrap();
        assert!(sam_encode_prompts(&SamTransform::new(256, 256).unwrap(), &request).is_err());
    }

    #[test]
    fn pinned_python_fixture_is_consumed_for_prompt_and_mask_stages() {
        fn read_f32(path: &std::path::Path) -> Vec<f32> {
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(bytes.len() % 4, 0, "unaligned fixture {:?}", path);
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        }
        fn read_bits(path: &std::path::Path, bit_count: usize) -> Vec<u8> {
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(bytes.len(), bit_count.div_ceil(8));
            (0..bit_count)
                .map(|index| (bytes[index / 8] >> (index % 8)) & 1)
                .collect()
        }
        fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
            assert_eq!(actual.len(), expected.len(), "{label} length");
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, e)| (a - e).abs() <= tolerance),
                "{label} mismatch"
            );
        }
        fn digest_f32(values: &[f32]) -> String {
            use sha2::Digest;
            let mut hasher = sha2::Sha256::new();
            for value in values {
                hasher.update(value.to_le_bytes());
            }
            hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        }
        let root = std::path::Path::new("../../tests/fixtures/m14/reference");
        let authority: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("report.json")).unwrap()).unwrap();
        let sample_indices = [
            0usize, 1, 2, 17, 101, 997, 4096, 65535, 123456, 250000, 500000, 700000, 900000,
            1200000, 1500000, 2000000,
        ];
        for (name, width, height) in [
            ("portrait", 500u32, 1000u32),
            ("landscape", 1600, 800),
            ("square", 1000, 1000),
        ] {
            let image_data = (0..(width as usize * height as usize))
                .map(|index| {
                    let byte = index * 3;
                    [
                        ((u8::try_from(byte % 256)
                            .unwrap()
                            .wrapping_mul(37)
                            .wrapping_add(17)
                            % 251) as f32)
                            / 255.0,
                        ((u8::try_from((byte + 1) % 256)
                            .unwrap()
                            .wrapping_mul(37)
                            .wrapping_add(17)
                            % 251) as f32)
                            / 255.0,
                        ((u8::try_from((byte + 2) % 256)
                            .unwrap()
                            .wrapping_mul(37)
                            .wrapping_add(17)
                            % 251) as f32)
                            / 255.0,
                    ]
                })
                .collect::<Vec<_>>();
            let image = CanonicalImage::new(width, height, image_data).unwrap();
            let (transform, encoder) = sam_encoder_preprocess(&image).unwrap();
            let expected_encoder_digest = authority["cases"][name]["encoder_input_sha256"]
                .as_str()
                .unwrap();
            assert_eq!(
                digest_f32(&encoder.values),
                expected_encoder_digest,
                "full encoder digest"
            );
            let encoder_indices = sample_indices
                .iter()
                .copied()
                .filter(|index| *index < encoder.values.len())
                .collect::<Vec<_>>();
            let expected_encoder =
                read_f32(&root.join(format!("{name}-encoder_input.samples.f32le")));
            let actual_encoder = encoder_indices
                .iter()
                .map(|index| encoder.values[*index])
                .collect::<Vec<_>>();
            assert_close(&actual_encoder, &expected_encoder, 0.0, "encoder samples");
            for prompt_id in 0..2 {
                let prefix = format!("{name}-case-{prompt_id}");
                let prompt = if prompt_id == 0 {
                    Prompt::new(
                        vec![
                            bgremove_core::PromptPoint::new(0.0, 0.0, true).unwrap(),
                            bgremove_core::PromptPoint::new(
                                width as f32 - 1.0,
                                height as f32 - 1.0,
                                false,
                            )
                            .unwrap(),
                        ],
                        Some(
                            bgremove_core::PromptBox::new(
                                1.0,
                                2.0,
                                width as f32 - 2.0,
                                height as f32 - 3.0,
                            )
                            .unwrap(),
                        ),
                    )
                    .unwrap()
                } else {
                    Prompt::new(
                        vec![bgremove_core::PromptPoint::new(
                            width as f32 / 2.0,
                            height as f32 / 2.0,
                            true,
                        )
                        .unwrap()],
                        None,
                    )
                    .unwrap()
                };
                let request = SamPromptRequest::new(prompt).unwrap();
                let encoded = sam_encode_prompts(&transform, &request).unwrap();
                let expected_coords =
                    read_f32(&root.join(format!("{prefix}-encoded_coords.f32le")));
                let expected_labels = read_f32(&root.join(format!("{prefix}-labels.f32le")));
                assert_close(
                    &encoded.point_coords.values,
                    &expected_coords,
                    1e-3,
                    "prompt coordinates",
                );
                assert_eq!(encoded.point_labels.values, expected_labels);
                let expected_no_prior =
                    read_f32(&root.join(format!("{prefix}-mask_input_no_prior.samples.f32le")));
                let expected_prior =
                    read_f32(&root.join(format!("{prefix}-mask_input_prior.samples.f32le")));
                let prior_sample_indices = sample_indices
                    .iter()
                    .copied()
                    .filter(|i| *i < encoded.mask_input.values.len())
                    .collect::<Vec<_>>();
                assert_close(
                    &prior_sample_indices
                        .iter()
                        .map(|i| encoded.mask_input.values[*i])
                        .collect::<Vec<_>>(),
                    &expected_no_prior,
                    0.0,
                    "no-prior input",
                );
                let prior_request = SamPromptRequest::with_prior(
                    request.prompt().clone(),
                    Some(SamPriorMask::new(vec![0.25; 256 * 256]).unwrap()),
                )
                .unwrap();
                let prior_encoded = sam_encode_prompts(&transform, &prior_request).unwrap();
                assert_close(
                    &prior_sample_indices
                        .iter()
                        .map(|i| prior_encoded.mask_input.values[*i])
                        .collect::<Vec<_>>(),
                    &expected_prior,
                    0.0,
                    "prior input",
                );
                assert_eq!(encoded.has_mask_input.values, vec![0.0]);
                assert_eq!(prior_encoded.has_mask_input.values, vec![1.0]);
                let plane = 684usize * 1024;
                let raw = (0..(3 * plane))
                    .map(|index| {
                        let candidate = index / plane;
                        let local = index % plane;
                        match candidate {
                            0 => -1.0 + 2.0 * local as f32 / (plane - 1) as f32,
                            1 => 1.0 - 2.0 * local as f32 / (plane - 1) as f32,
                            _ => (local as f32 / 37.0).sin(),
                        }
                    })
                    .collect::<Vec<_>>();
                let quality = vec![0.5, 0.75, 0.75];
                let low = (0..3 * 256 * 256)
                    .map(|index| match index / (256 * 256) {
                        0 => 0.0,
                        1 => 0.25,
                        _ => 1.0,
                    })
                    .collect::<Vec<_>>();
                let embedding_samples =
                    read_f32(&root.join(format!("{prefix}-embedding.samples.f32le")));
                let embedding_indices = sample_indices
                    .iter()
                    .copied()
                    .filter(|i| *i < 256 * 64 * 64)
                    .collect::<Vec<_>>();
                let actual_embedding = embedding_indices
                    .iter()
                    .map(|i| *i as f32 / 1000.0)
                    .collect::<Vec<_>>();
                assert_close(
                    &actual_embedding,
                    &embedding_samples,
                    0.0,
                    "embedding samples",
                );
                let candidates = decode_sam_outputs(
                    &TensorOutput {
                        shape: vec![1, 3, 684, 1024],
                        values: raw,
                    },
                    &TensorOutput {
                        shape: vec![1, 3],
                        values: quality.clone(),
                    },
                    &TensorOutput {
                        shape: vec![1, 3, 256, 256],
                        values: low,
                    },
                    &transform,
                )
                .unwrap();
                assert_eq!(candidates.len(), 3);
                let raw_samples = read_f32(&root.join(format!("{prefix}-raw_masks.samples.f32le")));
                let actual_raw = candidates
                    .iter()
                    .flat_map(|candidate| candidate.raw_logits.iter().copied())
                    .collect::<Vec<_>>();
                let raw_indices = sample_indices
                    .iter()
                    .copied()
                    .filter(|i| *i < actual_raw.len())
                    .collect::<Vec<_>>();
                assert_close(
                    &raw_indices
                        .iter()
                        .map(|i| actual_raw[*i])
                        .collect::<Vec<_>>(),
                    &raw_samples,
                    2e-6,
                    "returned raw logits",
                );
                let expected_quality = read_f32(&root.join(format!("{prefix}-quality.f32le")));
                assert_eq!(expected_quality.len(), 3);
                assert_eq!(
                    candidates
                        .iter()
                        .map(|candidate| candidate.quality_score)
                        .collect::<Vec<_>>(),
                    expected_quality
                );
                let low_samples = read_f32(&root.join(format!("{prefix}-low_res.samples.f32le")));
                let actual_low = candidates
                    .iter()
                    .flat_map(|candidate| candidate.low_resolution_logits.iter().copied())
                    .collect::<Vec<_>>();
                assert_close(
                    &sample_indices
                        .iter()
                        .copied()
                        .filter(|i| *i < actual_low.len())
                        .map(|i| actual_low[i])
                        .collect::<Vec<_>>(),
                    &low_samples,
                    0.0,
                    "low-res logits",
                );
                let restored_samples =
                    read_f32(&root.join(format!("{prefix}-restored_masks.samples.f32le")));
                let actual_restored = candidates
                    .iter()
                    .flat_map(|candidate| candidate.restored_logits.iter().copied())
                    .collect::<Vec<_>>();
                let expected_indices = sample_indices
                    .iter()
                    .copied()
                    .filter(|i| *i < actual_restored.len())
                    .collect::<Vec<_>>();
                assert_close(
                    &expected_indices
                        .iter()
                        .map(|i| actual_restored[*i])
                        .collect::<Vec<_>>(),
                    &restored_samples,
                    2e-4,
                    "restored logits",
                );
                let selection =
                    select_sam_candidates(&candidates, SamSelectionPolicy::HighestQuality).unwrap();
                assert_eq!(selection.selected_index, Some(1));
                let source_selected = read_bits(
                    &root.join(format!("{prefix}-selected_mask.bits")),
                    width as usize * height as usize,
                );
                let source_union = read_bits(
                    &root.join(format!("{prefix}-source_union.bits")),
                    width as usize * height as usize,
                );
                let actual_selected = selection
                    .selected
                    .data()
                    .iter()
                    .map(|value| u8::from(*value > 0.0))
                    .collect::<Vec<_>>();
                assert_eq!(actual_selected.len(), source_selected.len());
                assert_eq!(actual_selected, source_selected);
                let union =
                    select_sam_candidates(&candidates, SamSelectionPolicy::SourceCompatibleUnion)
                        .unwrap();
                let actual_union = union
                    .selected
                    .data()
                    .iter()
                    .map(|value| u8::from(*value > 0.0))
                    .collect::<Vec<_>>();
                assert_eq!(actual_union.len(), source_union.len());
                assert_eq!(actual_union, source_union);
            }
        }
    }
}
