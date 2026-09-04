//! Declarative model metadata. Weight loading and hash enforcement begin in M3.

use anyhow::{ensure, Result};
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
string_enum!(AspectPolicy { Stretch => "stretch", Identity => "identity", Dynamic => "dynamic" });
string_enum!(ResizeFilter { Nearest => "nearest", Bilinear => "bilinear", Triangle => "triangle", Bicubic => "bicubic", Lanczos3 => "lanczos3" });
string_enum!(ChannelOrder { Rgb => "rgb", Bgr => "bgr" });
string_enum!(Activation { None => "none", Sigmoid => "sigmoid" });
string_enum!(OutputNormalization { None => "none", MinMax => "minmax", Clamp => "clamp" });
string_enum!(TensorElementType { F32 => "f32", F16 => "f16", I32 => "i32", I64 => "i64" });

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
}

impl ModelManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "model id is empty");
        ensure!(!self.family.trim().is_empty(), "model family is empty");
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
        ensure!(
            canonical.starts_with(&canonical_base),
            "model {} resolves outside manifest directory",
            self.id
        );
        Ok(canonical)
    }

    pub fn verify_model_hash(&self, manifest_path: &Path) -> Result<PathBuf> {
        let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let license = base.join(&self.license_file).canonicalize()?;
        let canonical_base = base.canonicalize()?;
        ensure!(
            license.starts_with(&canonical_base),
            "model {} license resolves outside manifest directory",
            self.id
        );
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
}
