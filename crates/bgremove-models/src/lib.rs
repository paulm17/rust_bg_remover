//! Declarative model metadata. Weight loading and hash enforcement begin in M3.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

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
    pub layout: String,
    pub width: u32,
    pub height: u32,
    pub aspect: String,
    pub resize_filter: String,
    pub channel_order: String,
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub activation: String,
    pub output_normalization: String,
    pub source_url: String,
    pub source_commit: String,
    pub model_version: String,
    pub license_identifier: String,
    pub license_file: String,
    pub license_sha256: String,
    pub intended_use_approved: bool,
    pub opset: u32,
}

impl ModelManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "model id is empty");
        ensure!(!self.family.trim().is_empty(), "model family is empty");
        ensure!(
            self.width > 0 && self.height > 0,
            "model dimensions must be positive"
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
            ("layout", self.layout.as_str()),
            ("aspect", self.aspect.as_str()),
            ("resize_filter", self.resize_filter.as_str()),
            ("channel_order", self.channel_order.as_str()),
            ("activation", self.activation.as_str()),
            ("output_normalization", self.output_normalization.as_str()),
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
        Ok(())
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
}
