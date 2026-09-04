//! User-facing M1 skeleton commands. Decode is limited to obtaining the
//! canonical RGB grid; actual geometry and encoded output begin in M2.

use anyhow::{Context, Result};
use bgremove_color::OriginalRgbEstimator;
use bgremove_core::{NoOpSegmenter, Pipeline, PipelineConfig};
use bgremove_matting::IdentityMaskTransform;
use bgremove_models::parse_toml;
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use bgremove_cli::load_canonical;

#[derive(Debug, Parser)]
#[command(
    name = "bgremove",
    version,
    about = "Typed M1 background-removal pipeline"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Execute the validated no-op M1 pipeline for one image.
    Run {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "runs/m1-run")]
        output: PathBuf,
    },
    /// Emit the deterministic M1 alpha artifact for one image.
    Mask {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "runs/m1-mask")]
        output: PathBuf,
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
    note: &'static str,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run { input, output } => execute(&input, &output, "run"),
        Command::Mask { input, output } => execute(&input, &output, "mask"),
        Command::InspectModel { manifest } => inspect(&manifest),
    }
}

fn execute(input: &Path, output: &Path, command: &str) -> Result<()> {
    let image = load_canonical(input)?;
    let config = PipelineConfig::default().resolved_for(image.width(), image.height())?;
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
        config.canonical_json()?,
    )
    .with_context(|| format!("write resolved config in {}", output.display()))?;
    let alpha = result.alpha().data();
    let artifact = Artifact {
        schema: "m1.artifact.v1",
        width: result.alpha().width(),
        height: result.alpha().height(),
        alpha_min: alpha.iter().copied().fold(f32::INFINITY, f32::min),
        alpha_max: alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        alpha_sum: alpha.iter().sum(),
        note: "M1 no-op pipeline; M2 decode/resampling/encoding is not implemented",
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
