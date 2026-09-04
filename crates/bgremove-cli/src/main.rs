//! User-facing M2 decode, geometry, alpha and straight-alpha PNG commands.

use anyhow::{Context, Result};
use bgremove_color::OriginalRgbEstimator;
use bgremove_core::{NoOpSegmenter, Pipeline, PipelineConfig, TransparentInputPolicy};
use bgremove_matting::IdentityMaskTransform;
use bgremove_models::parse_toml;
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use bgremove_core::io::{encode_mask_png, encode_straight_rgba_png, load_canonical};

#[derive(Debug, Parser)]
#[command(
    name = "bgremove",
    version,
    about = "Typed background-removal pipeline"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Execute the deterministic no-op pipeline and emit a canonical cutout.
    Run {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "runs/m2-run")]
        output: PathBuf,
        #[arg(long, default_value = "multiply-predicted", value_parser = parse_policy)]
        transparent_input_policy: TransparentInputPolicy,
    },
    /// Emit a canonical-dimension mask PNG.
    Mask {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "runs/m2-mask")]
        output: PathBuf,
        #[arg(long, default_value = "multiply-predicted", value_parser = parse_policy)]
        transparent_input_policy: TransparentInputPolicy,
    },
    /// Inspect a declarative model manifest without loading weights.
    InspectModel {
        #[arg(long, default_value = "models/noop.toml")]
        manifest: PathBuf,
    },
}

#[derive(Serialize)]
struct Artifact {
    schema: &'static str,
    width: u32,
    height: u32,
    alpha_min: f32,
    alpha_max: f32,
    alpha_sum: f32,
    source_alpha_present: bool,
    transparent_input_policy: &'static str,
    geometry_scale: (f32, f32),
    geometry_offsets: (f32, f32),
    note: &'static str,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run {
            input,
            output,
            transparent_input_policy,
        } => execute(&input, &output, "run", transparent_input_policy),
        Command::Mask {
            input,
            output,
            transparent_input_policy,
        } => execute(&input, &output, "mask", transparent_input_policy),
        Command::InspectModel { manifest } => inspect(&manifest),
    }
}

fn parse_policy(value: &str) -> std::result::Result<TransparentInputPolicy, String> {
    match value {
        "multiply-predicted" => Ok(TransparentInputPolicy::MultiplyPredicted),
        "replace-source-alpha" => Ok(TransparentInputPolicy::ReplaceSourceAlpha),
        _ => Err("expected multiply-predicted or replace-source-alpha".into()),
    }
}

fn execute(
    input: &Path,
    output: &Path,
    command: &str,
    policy: TransparentInputPolicy,
) -> Result<()> {
    let image = load_canonical(input)?;
    let config = PipelineConfig::default()
        .with_transparent_input_policy(policy)
        .resolved_for(image.width(), image.height())?;
    let mut pipeline = Pipeline::new(
        config.clone(),
        Box::new(NoOpSegmenter),
        Box::new(IdentityMaskTransform),
        Box::new(bgremove_matting::NoOpRefiner::default()),
        Box::new(OriginalRgbEstimator::default()),
    );
    let result = pipeline.run(&image, None)?;
    fs::create_dir_all(output)
        .with_context(|| format!("create output directory {}", output.display()))?;
    fs::write(
        output.join("resolved-config.json"),
        config.canonical_json_m2()?,
    )
    .with_context(|| format!("write resolved config in {}", output.display()))?;
    let alpha = result.alpha().data();
    if command == "run" {
        fs::write(
            output.join("cutout.png"),
            encode_straight_rgba_png(&result)?,
        )
        .with_context(|| format!("write cutout in {}", output.display()))?;
    } else {
        fs::write(output.join("mask.png"), encode_mask_png(result.alpha())?)
            .with_context(|| format!("write mask in {}", output.display()))?;
    }
    let artifact = Artifact {
        schema: "m2.artifact.v1",
        width: result.alpha().width(),
        height: result.alpha().height(),
        alpha_min: alpha.iter().copied().fold(f32::INFINITY, f32::min),
        alpha_max: alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        alpha_sum: alpha.iter().sum(),
        source_alpha_present: image.source_alpha().data().iter().any(|a| *a < 1.0),
        transparent_input_policy: policy.as_str(),
        geometry_scale: config.geometry().scale(),
        geometry_offsets: config.geometry().offsets(),
        note: "M2 no-op pipeline; straight-alpha PNG and canonical geometry",
    };
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    fs::write(output.join(format!("{command}.json")), bytes)
        .with_context(|| format!("write {command} artifact in {}", output.display()))?;
    println!("wrote {}", output.display());
    Ok(())
}

fn inspect(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| {
        format!(
            "read model manifest {}; supply --manifest to a tracked manifest",
            path.display()
        )
    })?;
    let manifest = parse_toml(&text)
        .with_context(|| format!("parse TOML model manifest {}", path.display()))?;
    println!(
        "model={} family={} size={}x{} runtime=deferred-until-M3",
        manifest.id, manifest.family, manifest.width, manifest.height
    );
    Ok(())
}
