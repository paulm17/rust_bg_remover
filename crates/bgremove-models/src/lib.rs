//! Declarative model metadata. Weight loading and hash enforcement begin in M3.

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::path::{Path, PathBuf};

macro_rules! string_enum {
    ($name:ident { $($variant:ident => $value:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(rename_all = "kebab-case")]
        pub enum $name { $(#[serde(rename = $value)] $variant),+ }
    };
}

string_enum!(ModelLayout { Nchw => "nchw", Nhwc => "nhwc" });
string_enum!(AspectPolicy { Stretch => "stretch", Identity => "identity", Dynamic => "dynamic", Thumbnail => "thumbnail" });
string_enum!(ResizeFilter { Nearest => "nearest", Bilinear => "bilinear", Triangle => "triangle", Bicubic => "bicubic", Lanczos3 => "lanczos3" });
string_enum!(ChannelOrder { Rgb => "rgb", Bgr => "bgr" });
string_enum!(Activation { None => "none", Sigmoid => "sigmoid" });
string_enum!(OutputNormalization { None => "none", MinMax => "minmax", Clamp => "clamp" });
string_enum!(TensorElementType { F32 => "f32", F16 => "f16", I32 => "i32", I64 => "i64" });
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelEncoding {
    #[serde(rename = "fp32")]
    Fp32,
    #[serde(rename = "fp16")]
    Fp16,
    #[serde(rename = "quantized")]
    Quantized,
    #[serde(rename = "other")]
    Other,
}
string_enum!(PreprocessingProfile { ImglyIsnet => "imgly-isnet", RembgDis => "rembg-dis", Generic => "generic" });
string_enum!(ProfileOutputNormalization { Clamp => "clamp", SafeMinMax => "safe-per-image-minmax" });

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreprocessingProfileManifest {
    pub id: String,
    pub profile: PreprocessingProfile,
    pub layout: ModelLayout,
    pub resize_filter: ResizeFilter,
    pub channel_order: ChannelOrder,
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub output_normalization: ProfileOutputNormalization,
    pub restore_filter: ResizeFilter,
}

impl PreprocessingProfileManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.trim().is_empty(),
            "preprocessing profile id is empty"
        );
        ensure!(
            self.layout == ModelLayout::Nchw,
            "M4 profiles require NCHW layout"
        );
        ensure!(
            self.channel_order == ChannelOrder::Rgb,
            "M4 profiles require RGB channel order"
        );
        ensure!(
            self.scale.is_finite() && self.std.iter().all(|v| v.is_finite() && *v > 0.0),
            "profile normalization metadata is invalid"
        );
        ensure!(
            self.mean.iter().all(|v| v.is_finite()),
            "profile mean contains NaN or infinity"
        );
        match self.profile {
            PreprocessingProfile::ImglyIsnet => ensure!(
                self.resize_filter == ResizeFilter::Bilinear
                    && self.restore_filter == ResizeFilter::Bilinear
                    && self.scale == 1.0
                    && self.mean == [128.0; 3]
                    && self.std == [256.0; 3]
                    && self.output_normalization == ProfileOutputNormalization::Clamp,
                "IMG.LY profile contract mismatch"
            ),
            PreprocessingProfile::RembgDis => ensure!(
                self.resize_filter == ResizeFilter::Lanczos3
                    && self.restore_filter == ResizeFilter::Lanczos3
                    && self.scale == 1.0
                    && self.mean == [0.5; 3]
                    && self.std == [1.0; 3]
                    && self.output_normalization == ProfileOutputNormalization::SafeMinMax,
                "rembg DIS profile contract mismatch"
            ),
            PreprocessingProfile::Generic => bail!("generic is not an M4 profile"),
        }
        Ok(())
    }
}

pub fn parse_profile_toml(input: &str) -> Result<PreprocessingProfileManifest> {
    let profile: PreprocessingProfileManifest = toml::from_str(input)?;
    profile.validate()?;
    Ok(profile)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DimensionSpec {
    Static(u64),
    Dynamic(String),
}
impl DimensionSpec {
    fn validate(&self, field: &str) -> Result<()> {
        match self {
            Self::Static(value) => ensure!(*value > 0, "{field} contains static zero dimension"),
            Self::Dynamic(symbol) => ensure!(
                !symbol.trim().is_empty(),
                "{field} contains an empty dynamic dimension symbol"
            ),
        }
        Ok(())
    }
}

/// A tracked model manifest containing only declarative metadata in M1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub id: String,
    pub family: String,
    /// Algorithm family (for example `u2net`) is deliberately distinct from
    /// the checkpoint's variant, domain and deployment encoding.
    #[serde(default = "default_model_variant")]
    pub model_variant: String,
    #[serde(default = "default_model_domain")]
    pub model_domain: String,
    pub file: String,
    pub sha256: String,
    pub input_name: String,
    pub output_name: String,
    pub layout: ModelLayout,
    pub width: u32,
    pub height: u32,
    pub aspect: AspectPolicy,
    pub resize_filter: ResizeFilter,
    pub channel_order: ChannelOrder,
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub activation: Activation,
    pub output_normalization: OutputNormalization,
    pub source_url: String,
    pub source_commit: String,
    pub model_version: String,
    pub license_identifier: String,
    pub license_file: String,
    pub license_sha256: String,
    pub intended_use_approved: bool,
    pub opset: u32,
    #[serde(default)]
    pub input_shape: Vec<DimensionSpec>,
    #[serde(default)]
    pub output_shape: Vec<DimensionSpec>,
    #[serde(default)]
    pub input_type: Option<TensorElementType>,
    #[serde(default)]
    pub output_type: Option<TensorElementType>,
    #[serde(default)]
    pub output_index: Option<usize>,
    /// Semantic labels for multi-class outputs.  M6 DeepLabV3 is deliberately
    /// explicit about class 0/1 rather than inferring foreground polarity.
    #[serde(default)]
    pub class_mapping: Option<Vec<String>>,
    /// Algorithm family is intentionally separate from deployment encoding.
    #[serde(default = "default_algorithm_family")]
    pub algorithm_family: String,
    #[serde(default = "default_model_encoding")]
    pub model_encoding: ModelEncoding,
    #[serde(default = "default_preprocessing_profile")]
    pub preprocessing_profile: PreprocessingProfile,
    /// An external file is never fetched. It is permitted only to make the
    /// checked-in reference-tree weights auditable without committing them.
    #[serde(default)]
    pub external: bool,
}

fn default_algorithm_family() -> String {
    "unspecified".into()
}
fn default_model_variant() -> String {
    "unspecified".into()
}
fn default_model_domain() -> String {
    "unspecified".into()
}
fn default_model_encoding() -> ModelEncoding {
    ModelEncoding::Other
}
fn default_preprocessing_profile() -> PreprocessingProfile {
    PreprocessingProfile::Generic
}

impl ModelManifest {
    fn external_allowed(&self, base: &Path, canonical: &Path, kind: &str) -> Result<()> {
        let base = base
            .canonicalize()
            .context("canonicalize manifest directory")?;
        let workspace = base
            .parent()
            .ok_or_else(|| anyhow::anyhow!("manifest has no workspace parent"))?;
        let root = match self.algorithm_family.as_str() {
            "u2net" => workspace.join("projects/python/rembg"),
            "birefnet" => workspace.join("projects/python/rembg"),
            "basnet" | "deeplabv3" | "tracer-b7" => {
                workspace.join("projects/python/image-background-remove-tool")
            }
            _ => workspace.join("projects/javascript/background-removal-js"),
        };
        let allowed = match kind {
            "license" if self.algorithm_family == "birefnet" => workspace.to_path_buf(),
            "model"
                if matches!(
                    self.algorithm_family.as_str(),
                    "u2net" | "birefnet" | "basnet" | "deeplabv3" | "tracer-b7"
                ) =>
            {
                root.clone()
            }
            "model" => root.join("bundle/models"),
            "license" => root,
            _ => unreachable!(),
        }
        .canonicalize()
        .with_context(|| format!("canonicalize pinned IMG.LY external {kind} root"))?;
        ensure!(
            canonical.starts_with(&allowed),
            "model {} external {kind} is outside the pinned IMG.LY reference root",
            self.id
        );
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "model id is empty");
        ensure!(!self.family.trim().is_empty(), "model family is empty");
        ensure!(
            !self.algorithm_family.trim().is_empty(),
            "model algorithm family is empty"
        );
        ensure!(
            !self.model_variant.trim().is_empty(),
            "model variant is empty"
        );
        ensure!(
            !self.model_domain.trim().is_empty(),
            "model domain is empty"
        );
        ensure!(
            (self.width > 0 && self.height > 0)
                || self
                    .input_shape
                    .iter()
                    .any(|d| matches!(d, DimensionSpec::Dynamic(_))),
            "model dimensions must be positive unless dynamic input dimensions are declared"
        );
        ensure!(self.file.trim() != "", "model file is empty");
        ensure!(
            !self.source_url.trim().is_empty(),
            "model source URL is empty"
        );
        ensure!(
            !self.source_commit.trim().is_empty(),
            "model source commit is empty"
        );
        for (name, value) in [
            ("input_name", self.input_name.as_str()),
            ("output_name", self.output_name.as_str()),
        ] {
            ensure!(!value.trim().is_empty(), "model {name} is empty");
        }
        for (name, value) in [
            ("sha256", self.sha256.as_str()),
            ("license_sha256", self.license_sha256.as_str()),
        ] {
            ensure!(
                value.len() == 64
                    && value
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                "model {name} must be 64 lowercase hexadecimal characters"
            );
        }
        ensure!(
            !self.model_version.trim().is_empty(),
            "model version is empty"
        );
        ensure!(
            !self.license_identifier.trim().is_empty(),
            "license identifier is empty"
        );
        ensure!(
            !self.license_file.trim().is_empty(),
            "license file is empty"
        );
        ensure!(
            self.mean
                .iter()
                .chain(self.std.iter())
                .all(|v| v.is_finite()),
            "model normalization metadata contains NaN or infinity"
        );
        ensure!(self.scale.is_finite(), "model scale is NaN or infinity");
        ensure!(
            self.std.iter().all(|v| *v > 0.0),
            "model std must be positive"
        );
        if let Some(index) = self.output_index {
            ensure!(index < 1024, "model output index is unreasonably large");
        }
        for dimension in &self.input_shape {
            dimension.validate("input_shape")?;
        }
        for dimension in &self.output_shape {
            dimension.validate("output_shape")?;
        }
        Ok(())
    }

    pub fn resolve_model_path(&self, manifest_path: &Path) -> Result<PathBuf> {
        self.validate()?;
        let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let path = base.join(&self.file);
        let canonical_base = base.canonicalize()?;
        let canonical = path.canonicalize()?;
        ensure!(self.external || canonical.starts_with(&canonical_base),
            "model {} resolves outside manifest directory; set external=true only for a supplied, hashed reference-tree file",
            self.id);
        if self.external {
            self.external_allowed(base, &canonical, "model")?;
        }
        Ok(canonical)
    }

    pub fn verify_model_hash(&self, manifest_path: &Path) -> Result<PathBuf> {
        let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let license = base.join(&self.license_file).canonicalize()?;
        let canonical_base = base.canonicalize()?;
        ensure!(self.external || license.starts_with(&canonical_base),
            "model {} license resolves outside manifest directory; set external=true only for supplied metadata",
            self.id);
        if self.external {
            self.external_allowed(base, &license, "license")?;
        }
        let license_bytes = std::fs::read(&license)?;
        let license_digest = sha2::Sha256::digest(&license_bytes);
        let license_actual = license_digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        ensure!(
            license_actual == self.license_sha256,
            "model {} license hash mismatch: expected {}, actual {}",
            self.id,
            self.license_sha256,
            license_actual
        );
        let path = self.resolve_model_path(manifest_path)?;
        let bytes = std::fs::read(&path)?;
        let digest = sha2::Sha256::digest(&bytes);
        let actual = digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        ensure!(
            actual == self.sha256,
            "model {} hash mismatch: expected {}, actual {}",
            self.id,
            self.sha256,
            actual
        );
        ensure!(
            self.intended_use_approved,
            "model {} is not approved for intended use",
            self.id
        );
        Ok(path)
    }
}

impl std::fmt::Display for ModelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Nchw => "nchw",
                Self::Nhwc => "nhwc",
            }
        )
    }
}
impl std::fmt::Display for AspectPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Stretch => "stretch",
                Self::Identity => "identity",
                Self::Dynamic => "dynamic",
                Self::Thumbnail => "thumbnail",
            }
        )
    }
}

/// Parse and validate the tracked TOML manifest representation.
pub fn parse_toml(input: &str) -> Result<ModelManifest> {
    let manifest: ModelManifest = toml::from_str(input)?;
    manifest.validate()?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tracked_manifest_parses_and_unknown_fields_fail() {
        let text = include_str!("../../../models/noop.toml");
        let manifest = parse_toml(text).unwrap();
        assert_eq!(manifest.family, "noop");
        assert!(parse_toml(&format!("{text}\nextra = true\n")).is_err());
    }
    #[test]
    fn malformed_numeric_metadata_fails_closed() {
        let text = include_str!("../../../models/noop.toml").replace("scale = 1.0", "scale = inf");
        assert!(parse_toml(&text).is_err());
    }

    #[test]
    fn m3_fixture_manifest_verifies_model_and_license_hashes() {
        let path = std::path::Path::new("../../models/m3_identity.toml");
        let manifest = parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap();
        let model = manifest.verify_model_hash(path).unwrap();
        assert!(model.ends_with("models/fixtures/m3_identity.onnx"));
        assert!(manifest
            .input_shape
            .iter()
            .all(|d| matches!(d, DimensionSpec::Dynamic(_))));
    }

    #[test]
    fn dimensions_reject_zero_static_and_empty_dynamic_symbols() {
        let text = include_str!("../../../models/m3_identity.toml");
        assert!(parse_toml(&text.replace(
            "input_shape = [\"batch\", \"channel\", \"height\", \"width\"]",
            "input_shape = [0, \"channel\", \"height\", \"width\"]"
        ))
        .is_err());
        assert!(parse_toml(&text.replace(
            "input_shape = [\"batch\", \"channel\", \"height\", \"width\"]",
            "input_shape = [\"\", \"channel\", \"height\", \"width\"]"
        ))
        .is_err());
    }

    #[test]
    fn m4_manifests_pin_distinct_encodings_and_profiles() {
        let paths = [
            "../../models/m4_isnet_fp32.toml",
            "../../models/m4_isnet_fp16.toml",
            "../../models/m4_isnet_quantized.toml",
        ];
        let manifests = paths
            .iter()
            .map(|path| parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(manifests.iter().all(|m| m.algorithm_family == "isnet"));
        assert_eq!(
            manifests
                .iter()
                .map(|m| m.model_encoding)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert!(manifests
            .iter()
            .all(|m| m.preprocessing_profile == PreprocessingProfile::ImglyIsnet));
        assert!(manifests
            .iter()
            .zip(paths)
            .all(|(m, path)| m.verify_model_hash(std::path::Path::new(path)).is_ok()));
        let rembg_path = "../../models/m4_isnet_fp32_rembg.toml";
        let rembg = parse_toml(&std::fs::read_to_string(rembg_path).unwrap()).unwrap();
        assert_eq!(rembg.preprocessing_profile, PreprocessingProfile::RembgDis);
        assert!(rembg
            .verify_model_hash(std::path::Path::new(rembg_path))
            .is_ok());
    }

    #[test]
    fn rembg_profile_registry_is_typed_and_tamper_resistant() {
        let text = std::fs::read_to_string("../../models/m4_rembg_dis_profile.toml").unwrap();
        let profile = parse_profile_toml(&text).unwrap();
        assert_eq!(profile.profile, PreprocessingProfile::RembgDis);
        assert!(parse_profile_toml(&text.replace(
            "resize_filter = \"lanczos3\"",
            "resize_filter = \"bilinear\""
        ))
        .is_err());
        assert!(parse_profile_toml(&text.replace(
            "output_normalization = \"safe-per-image-minmax\"",
            "output_normalization = \"clamp\""
        ))
        .is_err());
    }

    #[test]
    fn m5_u2net_registry_separates_variant_domain_and_encoding() {
        let paths = [
            "../../models/m5_u2net.toml",
            "../../models/m5_u2netp.toml",
            "../../models/m5_u2net_human.toml",
            "../../models/m5_silueta.toml",
            "../../models/m5_u2net_cloth.toml",
        ];
        let manifests = paths
            .iter()
            .map(|p| parse_toml(&std::fs::read_to_string(p).unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(manifests.iter().all(|m| m.algorithm_family == "u2net"));
        assert_eq!(manifests[0].model_variant, "general");
        assert_eq!(manifests[2].model_domain, "human");
        assert_eq!(manifests[4].model_domain, "cloth");
        assert!(manifests
            .iter()
            .all(|m| m.model_encoding == ModelEncoding::Fp32));
        assert!(manifests.iter().all(|m| m.output_index == Some(0)));
    }

    #[test]
    fn m6_carvekit_registry_pins_three_real_adapters_and_no_hair_checkpoint() {
        let paths = [
            "../../models/m6_basnet.toml",
            "../../models/m6_deeplabv3.toml",
            "../../models/m6_tracer_b7.toml",
        ];
        let manifests = paths
            .iter()
            .map(|path| parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            manifests
                .iter()
                .map(|manifest| manifest.algorithm_family.as_str())
                .collect::<std::collections::BTreeSet<_>>(),
            ["basnet", "deeplabv3", "tracer-b7"]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        );
        assert_eq!(
            manifests[0].output_normalization,
            OutputNormalization::MinMax
        );
        assert_eq!(manifests[1].aspect, AspectPolicy::Thumbnail);
        assert_eq!(manifests[1].class_mapping.as_ref().unwrap().len(), 21);
        assert_eq!(manifests[2].output_normalization, OutputNormalization::None);
        assert!(manifests.iter().all(|manifest| manifest.external));
        assert!(manifests.iter().all(|manifest| manifest.sha256.len() == 64));
        assert!(paths.iter().all(|path| std::path::Path::new(path).exists()));
    }

    #[test]
    fn m7_birefnet_registry_separates_specialist_variants_and_pins_sources() {
        let paths = [
            "../../models/m7_birefnet_general.toml",
            "../../models/m7_birefnet_general_lite.toml",
            "../../models/m7_birefnet_portrait.toml",
            "../../models/m7_birefnet_dis.toml",
            "../../models/m7_birefnet_hrsod.toml",
            "../../models/m7_birefnet_cod.toml",
            "../../models/m7_birefnet_massive.toml",
        ];
        let manifests = paths
            .iter()
            .map(|path| parse_toml(&std::fs::read_to_string(path).unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(manifests.iter().all(|manifest| {
            manifest.algorithm_family == "birefnet"
                && manifest.width == 1024
                && manifest.height == 1024
                && manifest.activation == Activation::Sigmoid
                && manifest.output_normalization == OutputNormalization::MinMax
                && manifest.source_commit == "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
                && manifest.license_identifier == "MIT (BiRefNet upstream)"
                && manifest.external
        }));
        assert_eq!(
            manifests
                .iter()
                .map(|manifest| manifest.model_variant.as_str())
                .collect::<std::collections::BTreeSet<_>>(),
            [
                "general",
                "general-lite",
                "portrait",
                "dis",
                "hrsod",
                "cod",
                "massive"
            ]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
        );
        assert!(manifests.iter().all(|manifest| manifest.sha256.len() == 64));
    }
}
