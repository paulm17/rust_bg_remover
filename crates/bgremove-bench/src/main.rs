// M0 corpus validator and M2 deterministic benchmark implementation.

use anyhow::{anyhow, bail, ensure, Context, Result};
use bgremove_color::OriginalRgbEstimator;
use bgremove_core::io::{encode_mask_png, encode_straight_rgba_png, load_canonical};
use bgremove_core::{NoOpSegmenter, Pipeline, PipelineConfig, TransparentInputPolicy};
use bgremove_matting::IdentityMaskTransform;
use clap::{Parser, Subcommand};
use image::{GenericImageView, ImageDecoder, ImageReader};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

fn straight_rgba_bytes(result: &bgremove_core::Foreground) -> Vec<u8> {
    result
        .rgb()
        .data()
        .iter()
        .zip(result.alpha().data())
        .flat_map(|(rgb, alpha)| {
            let mut pixel = [0u8; 4];
            for channel in 0..3 {
                pixel[channel] = if rgb[channel].is_finite() {
                    (rgb[channel].clamp(0.0, 1.0) * 255.0).round() as u8
                } else {
                    0
                };
            }
            pixel[3] = if alpha.is_finite() {
                (alpha.clamp(0.0, 1.0) * 255.0).round() as u8
            } else {
                0
            };
            pixel
        })
        .collect()
}

const SCHEMA_VERSION: &str = "m0.corpus.v1";
const BASELINE_REPORT_VERSION: &str = "m0.baseline.v1";

#[derive(Debug, Parser)]
#[command(
    name = "bgremove-bench",
    version,
    about = "M0 regression and M2 benchmark harness"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate every manifest record, image, hash, split and leakage rule.
    Validate {
        #[arg(long, default_value = "corpus/manifest.jsonl")]
        manifest: PathBuf,
    },
    /// Generate a deterministic all-zero/all-one alpha baseline report.
    Baseline {
        #[arg(long, default_value = "corpus/manifest.jsonl")]
        manifest: PathBuf,
        #[arg(long, default_value = "runs/m0-baseline/report.json")]
        output: PathBuf,
    },
    /// Run validation and then write the deterministic baseline report.
    Check {
        #[arg(long, default_value = "corpus/manifest.jsonl")]
        manifest: PathBuf,
        #[arg(long, default_value = "runs/m0-baseline/report.json")]
        output: PathBuf,
    },
    /// Execute the deterministic M2 no-op pipeline and write PNG/run artifacts.
    Run {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "runs/m2-bench")]
        output: PathBuf,
        #[arg(long, default_value = "multiply-predicted", value_parser = parse_policy)]
        transparent_input_policy: TransparentInputPolicy,
    },
    /// Compare two deterministic artifact files byte-for-byte.
    Compare {
        #[arg(long)]
        left: PathBuf,
        #[arg(long)]
        right: PathBuf,
    },
    /// Run the checked-in M3 CPU fixture twice and write a deterministic report.
    M3Smoke {
        #[arg(long, default_value = "models/m3_identity.toml")]
        manifest: PathBuf,
        #[arg(long, default_value = "runs/m3-ort")]
        output: PathBuf,
        #[arg(long, default_value_t = 2)]
        workers: usize,
    },
    /// Run the deterministic M4 tensor/profile smoke report. No checkpoint
    /// is downloaded; runtime parity is enabled only with ORT_DYLIB.
    M4Smoke {
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long, default_value = "models/m4_isnet_fp32.toml")]
        manifest: PathBuf,
        #[arg(long, default_value = "imgly-isnet")]
        profile: String,
        #[arg(long, default_value = "cpu")]
        provider: String,
        #[arg(long, default_value = "runs/m4-isnet")]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Run the deterministic M5 U2-Net family report. Runtime inference is
    /// performed only when an explicit input and ORT_DYLIB are supplied.
    M5Smoke {
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long, default_value = "models/m5_u2net.toml")]
        manifest: PathBuf,
        #[arg(long)]
        category: Option<String>,
        #[arg(long, default_value = "cpu")]
        provider: String,
        #[arg(long, default_value = "runs/m5-u2net")]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Run the deterministic M6 CarveKit segmenter registry and raw-alpha
    /// tournament. Runtime inference requires an explicit input and ORT_DYLIB.
    M6Smoke {
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long, requires = "input")]
        reference: Option<PathBuf>,
        #[arg(long, default_value = "runs/m6-carvekit")]
        output: PathBuf,
        #[arg(long, default_value = "cpu")]
        provider: String,
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Split {
    Tune,
    Validation,
    Blind,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tune => "tune",
            Self::Validation => "validation",
            Self::Blind => "blind",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum SubjectPolicy {
    PrimarySubject,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum ShadowPolicy {
    PreserveTargetEffects,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestRecord {
    schema_version: String,
    id: String,
    input: String,
    reference: String,
    input_sha256: String,
    reference_sha256: String,
    input_decoded_sha256: String,
    reference_decoded_sha256: String,
    width: u32,
    height: u32,
    input_width: u32,
    input_height: u32,
    reference_width: u32,
    reference_height: u32,
    input_orientation: String,
    reference_orientation: String,
    input_alpha_present: bool,
    reference_alpha_present: bool,
    reference_png_bit_depth: Option<u8>,
    reference_png_color_type: Option<u8>,
    reference_png_color_metadata: String,
    reference_alpha_levels: u16,
    tags: Vec<String>,
    split: Split,
    subject_policy: SubjectPolicy,
    shadow_policy: ShadowPolicy,
    prompt: Option<serde_json::Value>,
    reference_created_at: Option<String>,
    reference_tool: Option<String>,
    reference_tool_version: Option<String>,
    duplicate_group: String,
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArenaRecord {
    id: String,
    input: String,
    target: String,
    challenge: String,
    split: Split,
    duplicate_group: String,
    fractional_alpha_reported: f64,
}

#[derive(Debug, Serialize)]
struct ValidationSummary {
    schema_version: &'static str,
    manifest: String,
    records: usize,
    split_counts: BTreeMap<String, usize>,
    coverage: Coverage,
    blind_excluded_from_tuning: bool,
    valid: bool,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Coverage {
    observed_tags: Vec<String>,
    missing_taxonomy_tags: Vec<String>,
    limitation: String,
}

#[derive(Debug, Serialize)]
struct BaselineReport {
    report_version: &'static str,
    schema_version: &'static str,
    manifest: String,
    metric_definition: MetricDefinition,
    tuning_policy: TuningPolicy,
    images: Vec<BaselineImage>,
    aggregate: Aggregate,
    coverage: Coverage,
    gate: GateStatus,
}

#[derive(Debug, Serialize)]
struct MetricDefinition {
    alpha_mae: String,
    soft_iou: String,
    alpha_levels: String,
    comparison: String,
}
#[derive(Debug, Serialize)]
struct TuningPolicy {
    blind_is_evaluation_only: bool,
    sweep_inputs: Vec<String>,
    statement: String,
}

#[derive(Debug, Serialize)]
struct GateStatus {
    status: &'static str,
    every_item_valid: bool,
    blind_not_used_by_sweeps: bool,
    coverage_limitations_declared: bool,
}

#[derive(Debug, Serialize)]
struct BaselineImage {
    id: String,
    split: Split,
    width: u32,
    height: u32,
    reference_fractional_alpha: f64,
    zero: BaselineCandidate,
    one: BaselineCandidate,
}

#[derive(Debug, Serialize)]
struct BaselineCandidate {
    alpha_mae: f64,
    soft_iou: f64,
    agreement_alpha_only: f64,
    score_status: &'static str,
}

#[derive(Debug, Serialize)]
struct Aggregate {
    by_candidate: BTreeMap<String, AggregateCandidate>,
    by_split: BTreeMap<String, BTreeMap<String, AggregateCandidate>>,
}
#[derive(Debug, Serialize)]
struct AggregateCandidate {
    image_count: usize,
    mean_alpha_mae: f64,
    mean_soft_iou: f64,
    mean_agreement_alpha_only: f64,
}

#[derive(Debug)]
struct ImageInfo {
    width: u32,
    height: u32,
    orientation: String,
    alpha_present: bool,
    png_bit_depth: Option<u8>,
    png_color_type: Option<u8>,
    png_color_metadata: String,
    alpha_levels: u16,
    rgba: image::RgbaImage,
}

#[derive(Debug)]
struct LoadedRecord {
    record: ManifestRecord,
    _input: ImageInfo,
    reference: ImageInfo,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Validate { manifest } => {
            let summary = validate_manifest(&manifest)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            if !summary.valid {
                bail!(
                    "M0 validation failed with {} error(s)",
                    summary.errors.len()
                );
            }
        }
        Command::Baseline { manifest, output } => {
            let records = load_validated(&manifest)?;
            write_baseline(&manifest, &output, &records)?;
            println!("wrote {}", output.display());
        }
        Command::Check { manifest, output } => {
            let summary = validate_manifest(&manifest)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            if !summary.valid {
                bail!(
                    "M0 validation failed with {} error(s)",
                    summary.errors.len()
                );
            }
            let records = load_validated(&manifest)?;
            write_baseline(&manifest, &output, &records)?;
            println!("wrote {}", output.display());
        }
        Command::Run {
            input,
            output,
            transparent_input_policy,
        } => write_m2_run(&input, &output, transparent_input_policy)?,
        Command::Compare { left, right } => compare_m2_runs(&left, &right)?,
        Command::M3Smoke {
            manifest,
            output,
            workers,
        } => write_m3_smoke(&manifest, &output, workers)?,
        Command::M4Smoke {
            input,
            manifest,
            profile,
            provider,
            output,
            workers,
        } => write_m4_smoke(
            &output,
            input.as_deref(),
            &manifest,
            &profile,
            &provider,
            workers,
        )?,
        Command::M5Smoke {
            input,
            manifest,
            category,
            provider,
            output,
            workers,
        } => write_m5_smoke(
            &output,
            input.as_deref(),
            &manifest,
            category.as_deref(),
            &provider,
            workers,
        )?,
        Command::M6Smoke {
            input,
            reference,
            output,
            provider,
            workers,
        } => write_m6_smoke(
            &output,
            input.as_deref(),
            reference.as_deref(),
            &provider,
            workers,
        )?,
    }
    Ok(())
}

fn write_m4_smoke(
    output: &Path,
    input: Option<&Path>,
    manifest_path: &Path,
    profile_name: &str,
    provider_name: &str,
    workers: usize,
) -> Result<()> {
    use bgremove_models::PreprocessingProfile;
    let profile_text = match profile_name {
        "imgly-isnet" => None,
        "rembg-dis" => Some(fs::read_to_string("models/m4_rembg_dis_profile.toml")?),
        other => bail!("unknown M4 profile {other}; expected imgly-isnet or rembg-dis"),
    };
    if let Some(text) = profile_text {
        let profile_manifest = bgremove_models::parse_profile_toml(&text)?;
        ensure!(
            profile_manifest.profile == PreprocessingProfile::RembgDis,
            "profile registry mismatch"
        );
    }
    let effective_manifest =
        if profile_name == "rembg-dis" && manifest_path == Path::new("models/m4_isnet_fp32.toml") {
            Path::new("models/m4_isnet_fp32_rembg.toml")
        } else {
            manifest_path
        };
    let manifest_text = fs::read_to_string(effective_manifest)?;
    let manifest = bgremove_models::parse_toml(&manifest_text)?;
    let profile = match profile_name {
        "imgly-isnet" => PreprocessingProfile::ImglyIsnet,
        "rembg-dis" => PreprocessingProfile::RembgDis,
        other => bail!("unknown M4 profile {other}; expected imgly-isnet or rembg-dis"),
    };
    let requested = match provider_name {
        "cpu" => bgremove_ort::RequestedProvider::Cpu,
        "coreml" => bgremove_ort::RequestedProvider::Coreml,
        "cuda" => bgremove_ort::RequestedProvider::Cuda,
        other => bail!("unknown provider {other}; expected cpu, coreml, or cuda"),
    };
    let rgb = bgremove_core::RgbImageF32::new(
        3,
        2,
        vec![
            [0.0, 0.25, 1.0],
            [1.0, 0.5, 0.0],
            [0.2, 0.4, 0.6],
            [0.9, 0.1, 0.3],
            [0.7, 0.8, 0.05],
            [1.0, 1.0, 1.0],
        ],
    )?;
    let image = bgremove_core::CanonicalImage::from_rgb(rgb);
    let imgly = bgremove_ort::isnet_preprocess_rgb(&image, PreprocessingProfile::ImglyIsnet)?;
    let rembg = bgremove_ort::isnet_preprocess_rgb(&image, PreprocessingProfile::RembgDis)?;
    let hash_f32 = |values: &[f32]| {
        let mut h = Sha256::new();
        for value in values {
            h.update(value.to_le_bytes());
        }
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let hash_bytes = |bytes: &[u8]| {
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let manifests = [
        ("fp32", "models/m4_isnet_fp32.toml"),
        ("fp16", "models/m4_isnet_fp16.toml"),
        ("quantized", "models/m4_isnet_quantized.toml"),
    ]
    .into_iter()
    .map(|(encoding, path)| {
        let text = fs::read(path)?;
        let manifest = bgremove_models::parse_toml(std::str::from_utf8(&text)?)?;
        Ok::<_, anyhow::Error>(serde_json::json!({
            "encoding": encoding,
            "id": manifest.id,
            "algorithm_family": manifest.algorithm_family,
            "model_sha256": manifest.sha256,
            "manifest_sha256": hash_bytes(&text),
            "available": manifest.verify_model_hash(Path::new(path)).is_ok()
        }))
    })
    .collect::<Result<Vec<_>>>()?;
    let reference = serde_json::from_str::<serde_json::Value>(&fs::read_to_string(
        "tests/fixtures/m4/parity.json",
    )?)?;
    let rembg_pillow = serde_json::from_str::<serde_json::Value>(&fs::read_to_string(
        "tests/fixtures/m4/rembg-pillow-fixture.json",
    )?)?;
    let reference_raw_comparison = reference
        .get("raw_comparison")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let reference_verdict = reference
        .get("verdict")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let mut sample = Vec::new();
    for &i in &[0usize, 1, 1023, 1024, 1024 * 1024 - 1] {
        sample.push(imgly.values[i]);
        sample.push(imgly.values[1024 * 1024 + i]);
        sample.push(imgly.values[2 * 1024 * 1024 + i]);
    }
    fs::create_dir_all(output)?;
    let runtime_run = if let Some(input_path) = input {
        let runtime = std::env::var_os("ORT_DYLIB").ok_or_else(|| anyhow!("ORT_DYLIB must point to an installed full ONNX Runtime; runtime downloads are disabled"))?;
        let image = load_canonical(input_path)?;
        let segmenter = bgremove_ort::IsnetSegmenter::new(
            &manifest,
            effective_manifest,
            Path::new(&runtime),
            workers,
            profile,
            requested,
            false,
        )?;
        let evidence = segmenter.predict_with_evidence(&image)?;
        let cutout = bgremove_ort::isnet_straight_cutout(&image, evidence.restored.clone())?;
        fs::write(
            output.join("mask.png"),
            encode_mask_png(&evidence.restored)?,
        )?;
        fs::write(
            output.join("cutout.png"),
            encode_straight_rgba_png(&cutout)?,
        )?;
        let input_hash = hash_f32(&evidence.tensor.values);
        let raw_hash = hash_f32(&evidence.raw_output.values);
        serde_json::json!({"input":input_path.display().to_string(),"dimensions":[image.width(),image.height()],"tensor_sha256":input_hash,"tensor_samples":evidence.tensor.values.iter().step_by(1024*1024).take(3).copied().collect::<Vec<_>>(),"raw_output_sha256":raw_hash,"raw_min":evidence.raw_output.values.iter().copied().fold(f32::INFINITY,f32::min),"raw_max":evidence.raw_output.values.iter().copied().fold(f32::NEG_INFINITY,f32::max),"raw_mean":evidence.raw_output.values.iter().sum::<f32>()/evidence.raw_output.values.len() as f32,"restored_mask_sha256":hash_bytes(&encode_mask_png(&evidence.restored)?),"cutout_sha256":hash_bytes(&encode_straight_rgba_png(&cutout)?),"provider":segmenter.provider(),"status":"pass"})
    } else {
        serde_json::json!({"status":"fixture-only"})
    };
    let report = serde_json::json!({
        "schema": "m4.isnet-smoke.v1",
        "algorithm_family": "isnet",
        "encodings": ["fp32", "fp16", "quantized"],
        "models": manifests,
        "preprocessing_profiles": {
            "imgly-isnet": {"formula":"(u8-128)/256", "layout":"nchw", "resize":"js-corner-aligned-bilinear-u8"},
            "rembg-dis": {"formula":"u8/max(resized_u8).max(1e-6)-0.5", "layout":"nchw", "resize":"deterministic-lanczos3", "output":"safe-per-image-minmax"}
        },
        "unit_fixture": {"dimensions":[3,2], "imgly_tensor_sha256":hash_f32(&imgly.values), "rembg_tensor_sha256":hash_f32(&rembg.values), "imgly_samples":sample},
        "reference": reference,
        "rembg_pillow": rembg_pillow,
        "parity": {"input_tensor_max_abs_tolerance": 0.0, "raw_float_max_abs_tolerance": 0.000001, "raw_float_mean_abs_tolerance": 0.0000001, "restored_u8_max_abs_tolerance": 0, "raw_comparison": reference_raw_comparison, "verdict": reference_verdict},
        "manifest": {"id":manifest.id,"algorithm_family":manifest.algorithm_family,"model_encoding":manifest.model_encoding,"sha256":manifest.sha256,"profile":manifest.preprocessing_profile},
        "provider": runtime_run.get("provider").cloned().unwrap_or(serde_json::json!({"requested":provider_name,"active":"not-run"})),
        "runtime_run": runtime_run,
        "deterministic": true
    });
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(output.join("report.json"), bytes)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn write_m5_smoke(
    output: &Path,
    input: Option<&Path>,
    manifest_path: &Path,
    category: Option<&str>,
    provider_name: &str,
    workers: usize,
) -> Result<()> {
    let manifest = bgremove_models::parse_toml(&fs::read_to_string(manifest_path)?)?;
    ensure!(
        manifest.algorithm_family == "u2net",
        "M5 smoke requires a U2-Net manifest"
    );
    ensure!(
        category.is_none() || manifest.model_domain == "cloth",
        "cloth category is valid only with the cloth domain manifest"
    );
    if manifest.model_domain == "cloth" {
        ensure!(
            category.is_some(),
            "cloth smoke requires explicit upper, lower, or full category"
        );
    }
    let requested = match provider_name {
        "cpu" => bgremove_ort::RequestedProvider::Cpu,
        "coreml" => bgremove_ort::RequestedProvider::Coreml,
        "cuda" => bgremove_ort::RequestedProvider::Cuda,
        other => bail!("unknown provider {other}; expected cpu, coreml, or cuda"),
    };
    if let Some(value) = category {
        ensure!(
            ["upper", "lower", "full"].contains(&value),
            "invalid cloth category {value}"
        );
    }
    let hash_bytes = |bytes: &[u8]| {
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let hash_f32 = |values: &[f32]| {
        let mut h = Sha256::new();
        for value in values {
            h.update(value.to_le_bytes());
        }
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let family_paths = [
        "models/m5_u2net.toml",
        "models/m5_u2netp.toml",
        "models/m5_u2net_human.toml",
        "models/m5_silueta.toml",
        "models/m5_u2net_cloth.toml",
    ];
    let models = family_paths.into_iter().map(|path| {
        let text = fs::read_to_string(path)?;
        let m = bgremove_models::parse_toml(&text)?;
        Ok::<_, anyhow::Error>(serde_json::json!({
            "id":m.id, "algorithm_family":m.algorithm_family, "model_variant":m.model_variant,
            "model_domain":m.model_domain, "model_encoding":m.model_encoding,
            "input_name":m.input_name, "output_name":m.output_name, "input_shape":m.input_shape,
            "output_shape":m.output_shape, "input_type":m.input_type, "output_type":m.output_type,
            "opset":m.opset, "model_sha256":m.sha256, "manifest_sha256":hash_bytes(text.as_bytes()),
            "available":m.verify_model_hash(Path::new(path)).is_ok()
        }))
    }).collect::<Result<Vec<_>>>()?;
    let rgb = bgremove_core::RgbImageF32::new(
        3,
        2,
        vec![
            [0.0, 0.25, 1.0],
            [1.0, 0.5, 0.0],
            [0.2, 0.4, 0.6],
            [0.9, 0.1, 0.3],
            [0.7, 0.8, 0.05],
            [1.0, 1.0, 1.0],
        ],
    )?;
    let fixture = bgremove_core::CanonicalImage::from_rgb(rgb);
    let tensor = bgremove_ort::u2net_preprocess_rgb(&fixture, manifest.width, manifest.height)?;
    let decoded = fixture
        .rgb()
        .data()
        .iter()
        .flat_map(|px| px.iter().copied())
        .collect::<Vec<_>>();
    let mut runtime_run = serde_json::json!({
        "status":"fixture-only", "decoded_rgb_sha256":null,
        "preprocessed_tensor_sha256":null, "raw_output_sha256":null,
        "restored_alpha_sha256":null, "restored_class_map_sha256":null,
        "final_straight_alpha_cutout_sha256":null,
        "note":"Set ORT_DYLIB and provide an external verified checkpoint for real inference; inference never downloads models"
    });
    if let Some(input_path) = input {
        let runtime = std::env::var_os("ORT_DYLIB")
            .ok_or_else(|| anyhow!("ORT_DYLIB is required for runtime M5 smoke"))?;
        let image = load_canonical(input_path)?;
        if manifest.model_domain == "cloth" {
            let segmenter = bgremove_ort::U2netClothSegmenter::new(
                &manifest,
                manifest_path,
                Path::new(&runtime),
                workers,
                requested,
                false,
            )?;
            let evidence = segmenter.predict_with_evidence(&image)?;
            let cloth_category = bgremove_ort::ClothCategory::parse(category.unwrap())?;
            let categories = segmenter.predict_categories(&image, None)?;
            let mask = categories
                .iter()
                .find(|(candidate, _)| *candidate == cloth_category)
                .map(|(_, mask)| mask.clone())
                .ok_or_else(|| anyhow!("requested cloth category missing"))?;
            let cutout = bgremove_ort::isnet_straight_cutout(&image, mask)?;
            let cutout_rgba = straight_rgba_bytes(&cutout);
            let cutout_png = encode_straight_rgba_png(&cutout)?;
            if let Some(dir) = std::env::var_os("M5_ARTIFACT_DIR") {
                let dir = PathBuf::from(dir);
                fs::create_dir_all(&dir)?;
                let f32_bytes = |values: &[f32]| {
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                };
                fs::write(
                    dir.join("preprocessed-tensor.f32le"),
                    f32_bytes(&evidence.tensor.values),
                )?;
                fs::write(
                    dir.join("raw-output.f32le"),
                    f32_bytes(&evidence.raw_output.values),
                )?;
                fs::write(
                    dir.join("restored-class-map.u8"),
                    &evidence.restored_class_map,
                )?;
                for (candidate, _candidate_mask) in &categories {
                    let bytes = bgremove_ort::cloth_category_mask_u8(
                        &evidence.restored_class_map,
                        *candidate,
                    );
                    fs::write(dir.join(format!("{}-mask.u8", candidate.as_str())), bytes)?;
                }
                fs::write(dir.join("final-straight-alpha-cutout.rgba"), &cutout_rgba)?;
                fs::write(dir.join("final-straight-alpha-cutout.png"), &cutout_png)?;
            }
            runtime_run = serde_json::json!({"status":"pass", "input":input_path.display().to_string(), "category":cloth_category.as_str(), "dimensions":[image.width(),image.height()], "decoded_rgb_sha256":hash_f32(&image.rgb().data().iter().flat_map(|px| px.iter().copied()).collect::<Vec<_>>()), "tensor_sha256":hash_f32(&evidence.tensor.values), "raw_output_sha256":hash_f32(&evidence.raw_output.values), "restored_class_map_sha256":hash_bytes(&evidence.restored_class_map), "final_straight_alpha_cutout_rgba_sha256":hash_bytes(&cutout_rgba), "final_straight_alpha_cutout_png_sha256":hash_bytes(&cutout_png), "provider":segmenter.provider()});
        } else {
            let segmenter = bgremove_ort::U2netSegmenter::new(
                &manifest,
                manifest_path,
                Path::new(&runtime),
                workers,
                requested,
                false,
            )?;
            let evidence = segmenter.predict_with_evidence(&image)?;
            let cutout = bgremove_ort::isnet_straight_cutout(&image, evidence.restored.clone())?;
            let cutout_rgba = straight_rgba_bytes(&cutout);
            let cutout_png = encode_straight_rgba_png(&cutout)?;
            if let Some(dir) = std::env::var_os("M5_ARTIFACT_DIR") {
                let dir = PathBuf::from(dir);
                fs::create_dir_all(&dir)?;
                let rgb = image
                    .rgb()
                    .data()
                    .iter()
                    .flat_map(|px| px.iter().copied())
                    .collect::<Vec<_>>();
                let f32_bytes = |values: &[f32]| {
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                };
                fs::write(dir.join("decoded-rgb.f32le"), f32_bytes(&rgb))?;
                fs::write(
                    dir.join("preprocessed-tensor.f32le"),
                    f32_bytes(&evidence.tensor.values),
                )?;
                fs::write(
                    dir.join("raw-output.f32le"),
                    f32_bytes(&evidence.raw_output.values),
                )?;
                fs::write(
                    dir.join("restored-alpha.f32le"),
                    f32_bytes(evidence.restored.data()),
                )?;
                fs::write(dir.join("final-straight-alpha-cutout.rgba"), &cutout_rgba)?;
                fs::write(
                    dir.join("final-straight-alpha-cutout.png"),
                    encode_straight_rgba_png(&cutout)?,
                )?;
            }
            runtime_run = serde_json::json!({"status":"pass", "input":input_path.display().to_string(), "dimensions":[image.width(),image.height()], "decoded_rgb_sha256":hash_f32(&image.rgb().data().iter().flat_map(|px| px.iter().copied()).collect::<Vec<_>>()), "tensor_sha256":hash_f32(&evidence.tensor.values), "raw_output_sha256":hash_f32(&evidence.raw_output.values), "restored_alpha_sha256":hash_f32(evidence.restored.data()), "final_straight_alpha_cutout_rgba_sha256":hash_bytes(&cutout_rgba), "final_straight_alpha_cutout_png_sha256":hash_bytes(&cutout_png), "provider":segmenter.provider()});
        }
    }
    let domain_runs = if let Some(input_path) = input {
        let image = load_canonical(input_path)?;
        [
            ("general", "models/m5_u2net.toml"),
            ("light", "models/m5_u2netp.toml"),
            ("human", "models/m5_u2net_human.toml"),
            ("silueta", "models/m5_silueta.toml"),
        ]
        .into_iter()
        .map(|(domain, path)| {
            let selected = bgremove_models::parse_toml(&fs::read_to_string(path)?)?;
            let segmenter = bgremove_ort::U2netSegmenter::new(
                &selected,
                Path::new(path),
                Path::new(
                    &std::env::var_os("ORT_DYLIB")
                        .ok_or_else(|| anyhow!("ORT_DYLIB is required for domain smoke"))?,
                ),
                workers,
                requested,
                false,
            )?;
            let evidence = segmenter.predict_with_evidence(&image)?;
            let cutout = bgremove_ort::isnet_straight_cutout(&image, evidence.restored.clone())?;
            let cutout_rgba = straight_rgba_bytes(&cutout);
            Ok::<_, anyhow::Error>(serde_json::json!({
                "domain": domain,
                "variant": domain,
                "manifest": path,
                "model_variant": selected.model_variant,
                "model_domain": selected.model_domain,
                "model_encoding": selected.model_encoding,
                "algorithm_family": selected.algorithm_family,
                "model_sha256": selected.sha256,
                "status": "pass",
                "dimensions": [image.width(), image.height()],
                "tensor_sha256": hash_f32(&evidence.tensor.values),
                "raw_output_sha256": hash_f32(&evidence.raw_output.values),
                "restored_alpha_sha256": hash_f32(evidence.restored.data()),
                "final_straight_alpha_cutout_rgba_sha256": hash_bytes(&cutout_rgba),
                "provider": segmenter.provider(),
            }))
        })
        .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let python_ort_parity = serde_json::from_str::<serde_json::Value>(&fs::read_to_string(
        "tests/fixtures/m5/python-ort-parity.json",
    )?)?;
    fs::create_dir_all(output)?;
    let report = serde_json::json!({
        "schema":"m5.u2net-smoke.v1", "algorithm_family":"u2net",
        "shared_adapter":"U2netSegmenter (general/light/human/Silueta)",
        "cloth_adapter":"U2netClothSegmenter (768 square, output 0 argmax class map)",
        "models":models, "selected_manifest":manifest.id, "selected_variant":manifest.model_variant,
        "selected_domain":manifest.model_domain, "selected_encoding":manifest.model_encoding,
        "preprocessing":{"channel_order":"rgb","layout":"nchw","resize":"lanczos3","size":[manifest.width,manifest.height],"normalization":"global resized-image max with epsilon 1e-6, ImageNet mean/std"},
        "output_contract":{"general":"first output index 0, direct mask, safe per-image min/max, uint8 Lanczos3 restore","cloth":"[1,4,768,768] logits, argmax to background/upper/lower/full, resize class IDs then equality"},
        "hard_threshold":{"comparison":"strict greater-than","range_u8":[0,255]},
        "tolerances":{"input_tensor_max_abs":1e-6,"input_tensor_mean_abs":1e-7,"single_mask_raw_output_max_abs":1e-5,"cloth_raw_logits_max_abs":1e-4,"cloth_raw_logits_mean_abs":3e-6,"restored_mask_u8_max_abs":0},
        "cloth_policy":{"excluded_from_general_scores":true,"explicit_category_required":true,"requested_category":category},
        "unit_fixture":{"decoded_rgb_sha256":hash_f32(&decoded),"preprocessed_tensor_sha256":hash_f32(&tensor.values),"preprocessed_tensor_shape":tensor.shape},
        "provider":runtime_run.get("provider").cloned().unwrap_or(serde_json::json!({"requested":provider_name,"active":"not-run"})),
        "python_ort_parity":python_ort_parity,
        "domain_runs":domain_runs, "runtime_run":runtime_run, "deterministic":true
    });
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(output.join("report.json"), bytes)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn write_m6_smoke(
    output: &Path,
    input: Option<&Path>,
    reference_path: Option<&Path>,
    provider_name: &str,
    workers: usize,
) -> Result<()> {
    let requested = match provider_name {
        "cpu" => bgremove_ort::RequestedProvider::Cpu,
        "coreml" => bgremove_ort::RequestedProvider::Coreml,
        "cuda" => bgremove_ort::RequestedProvider::Cuda,
        other => bail!("unknown provider {other}; expected cpu, coreml, or cuda"),
    };
    let manifests = [
        ("basnet", "models/m6_basnet.toml"),
        ("deeplabv3", "models/m6_deeplabv3.toml"),
        ("tracer-b7", "models/m6_tracer_b7.toml"),
    ];
    let hash_bytes = |bytes: &[u8]| {
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let hash_f32 = |values: &[f32]| {
        let mut h = Sha256::new();
        for value in values {
            h.update(value.to_le_bytes());
        }
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let registry = manifests.iter().map(|(family, path)| {
        let text = fs::read_to_string(path)?;
        let manifest = bgremove_models::parse_toml(&text)?;
        let available = manifest.verify_model_hash(Path::new(path)).is_ok();
        Ok::<_, anyhow::Error>(serde_json::json!({
            "family": family,
            "manifest": path,
            "id": manifest.id,
            "algorithm_family": manifest.algorithm_family,
            "model_variant": manifest.model_variant,
            "model_domain": manifest.model_domain,
            "model_encoding": manifest.model_encoding,
            "geometry": {"aspect": manifest.aspect, "input": [manifest.width, manifest.height], "resize_filter": manifest.resize_filter},
            "input_name": manifest.input_name,
            "output_name": manifest.output_name,
            "output_index": manifest.output_index,
            "output_normalization": manifest.output_normalization,
            "class_mapping": manifest.class_mapping,
            "model_sha256": manifest.sha256,
            "manifest_sha256": hash_bytes(text.as_bytes()),
            "available": available,
            "hair_checkpoint_registered": false,
        }))
    }).collect::<Result<Vec<_>>>()?;
    let mut runs = Vec::new();
    let mut status = "fixture-only";
    if let Some(input_path) = input {
        let reference_path = reference_path.ok_or_else(|| {
            anyhow!("M6 runtime tournament requires --reference paired alpha input")
        })?;
        let runtime = std::env::var_os("ORT_DYLIB")
            .ok_or_else(|| anyhow!("ORT_DYLIB is required for runtime M6 smoke"))?;
        let image = load_canonical(input_path)?;
        let reference = ImageReader::open(reference_path)?.decode()?.to_rgba8();
        ensure!(
            reference.dimensions() == image.dimensions(),
            "reference alpha dimensions must match the original input"
        );
        let reference_alpha = reference
            .pixels()
            .map(|pixel| f32::from(pixel.0[3]) / 255.0)
            .collect::<Vec<_>>();
        let input_file_sha256 = hash_bytes(&fs::read(input_path)?);
        let reference_file_sha256 = hash_bytes(&fs::read(reference_path)?);
        let decoded_rgb_sha256 = hash_f32(
            &image
                .rgb()
                .data()
                .iter()
                .flat_map(|pixel| pixel.iter().copied())
                .collect::<Vec<_>>(),
        );
        let reference_alpha_sha256 = hash_f32(&reference_alpha);
        let runtime_path = Path::new(&runtime);
        let run_one = |family: &str, manifest_path: &str| -> Result<serde_json::Value> {
            let manifest_text = fs::read_to_string(manifest_path)?;
            let m = bgremove_models::parse_toml(&manifest_text)?;
            let (tensor, raw, alpha, provider) = match family {
                "basnet" => {
                    let s = bgremove_ort::BasnetSegmenter::new(
                        &m,
                        Path::new(manifest_path),
                        runtime_path,
                        workers,
                        requested,
                        false,
                    )?;
                    let e = s.predict_with_evidence(&image)?;
                    (e.tensor, e.raw_output, e.restored, s.provider())
                }
                "deeplabv3" => {
                    let s = bgremove_ort::DeepLabV3Segmenter::new(
                        &m,
                        Path::new(manifest_path),
                        runtime_path,
                        workers,
                        requested,
                        false,
                    )?;
                    let e = s.predict_with_evidence(&image)?;
                    (e.tensor, e.raw_output, e.restored, s.provider())
                }
                "tracer-b7" => {
                    let s = bgremove_ort::TracerB7Segmenter::new(
                        &m,
                        Path::new(manifest_path),
                        runtime_path,
                        workers,
                        requested,
                        false,
                    )?;
                    let e = s.predict_with_evidence(&image)?;
                    (e.tensor, e.raw_output, e.restored, s.provider())
                }
                "u2net" => {
                    let s = bgremove_ort::U2netSegmenter::new(
                        &m,
                        Path::new(manifest_path),
                        runtime_path,
                        workers,
                        requested,
                        false,
                    )?;
                    let e = s.predict_with_evidence(&image)?;
                    (e.tensor, e.raw_output, e.restored, s.provider())
                }
                "isnet" => {
                    let s = bgremove_ort::IsnetSegmenter::new(
                        &m,
                        Path::new(manifest_path),
                        runtime_path,
                        workers,
                        bgremove_models::PreprocessingProfile::ImglyIsnet,
                        requested,
                        false,
                    )?;
                    let e = s.predict_with_evidence(&image)?;
                    (e.tensor, e.raw_output, e.restored, s.provider())
                }
                _ => bail!("unknown M6 family {family}"),
            };
            ensure!(
                alpha.data().len() == reference_alpha.len(),
                "{family} alpha/reference lengths differ"
            );
            let (alpha_mae_sum, intersection, union) = alpha
                .data()
                .iter()
                .zip(&reference_alpha)
                .map(|(candidate, target)| {
                    (
                        (candidate - target).abs(),
                        candidate.min(*target),
                        candidate.max(*target),
                    )
                })
                .fold((0.0f64, 0.0f64, 0.0f64), |acc, item| {
                    (
                        acc.0 + f64::from(item.0),
                        acc.1 + f64::from(item.1),
                        acc.2 + f64::from(item.2),
                    )
                });
            let alpha_mae = alpha_mae_sum / alpha.data().len() as f64;
            let soft_iou = intersection / union.max(1e-12);
            let cutout = bgremove_ort::isnet_straight_cutout(&image, alpha.clone())?;
            let cutout_png = encode_straight_rgba_png(&cutout)?;
            Ok(serde_json::json!({
                "family": family, "status": "pass", "dimensions": [image.width(), image.height()],
                "model": {"id": m.id, "variant": m.model_variant, "encoding": m.model_encoding, "algorithm_family": m.algorithm_family, "manifest_sha256": hash_bytes(manifest_text.as_bytes()), "model_sha256": m.sha256},
                "input": {"path": input_path.display().to_string(), "file_sha256": input_file_sha256, "decoded_rgb_sha256": decoded_rgb_sha256},
                "reference": {"path": reference_path.display().to_string(), "file_sha256": reference_file_sha256, "decoded_alpha_sha256": reference_alpha_sha256, "alpha_channel": "RGBA channel 3"},
                "preprocessed_tensor_sha256": hash_f32(&tensor.values), "preprocessed_tensor_shape": tensor.shape,
                "raw_output_sha256": hash_f32(&raw.values), "raw_output_shape": raw.shape,
                "restored_alpha_sha256": hash_f32(alpha.data()), "final_straight_alpha_cutout_png_sha256": hash_bytes(&cutout_png),
                "metrics": {"alpha_mae": alpha_mae, "soft_iou": soft_iou},
                "provider": provider,
            }))
        };
        for (family, path) in manifests {
            runs.push(run_one(family, path)?);
        }
        // M6's tournament is deliberately raw-alpha and uses the already
        // accepted M4/M5 adapters for the two control candidates.
        runs.push(run_one("u2net", "models/m5_u2net.toml")?);
        runs.push(run_one("isnet", "models/m4_isnet_fp32.toml")?);
        status = "pass";
    }
    fs::create_dir_all(output)?;
    let report = serde_json::json!({
        "schema": "m6.carvekit-raw-alpha-tournament.v1",
        "status": status,
        "source": {"repository": "projects/python/image-background-remove-tool", "commit": "f141a311af67fb1da64269c508a6d1f786420801"},
        "models": registry,
        "tournament": {"candidates": ["u2net", "isnet", "basnet", "deeplabv3", "tracer-b7"], "shared_downstream": "raw alpha, canonical dimensions, straight-alpha original RGB, no cleanup/refiner", "identical_downstream": true, "reference": "explicit --reference RGBA alpha paired with explicit --input RGB", "metrics": {"alpha_mae": "mean(abs(candidate_alpha - reference_alpha)) over canonical pixels; lower is better; units=alpha in [0,1]", "soft_iou": "sum(min(candidate_alpha, reference_alpha)) / sum(max(candidate_alpha, reference_alpha)); higher is better; units=unitless"}},
        "gates": {"deep_lab_hard_argmax": true, "deep_lab_softening_before_refiner": false, "tracer_hair_registered": false, "nan_inf_rejected": true, "constant_minmax_safe": true},
        "runs": runs,
        "deterministic": true,
    });
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(output.join("report.json"), bytes)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn write_m3_smoke(manifest_path: &Path, output: &Path, workers: usize) -> Result<()> {
    let manifest = bgremove_models::parse_toml(&fs::read_to_string(manifest_path)?)?;
    let runtime = std::env::var_os("ORT_DYLIB").ok_or_else(|| anyhow!("ORT_DYLIB must point to an installed ONNX Runtime dylib; runtime downloads are disabled"))?;
    let pool = std::sync::Arc::new(bgremove_ort::SessionPool::new(
        &manifest,
        manifest_path,
        Path::new(&runtime),
        workers,
        bgremove_ort::RequestedProvider::Cpu,
        false,
    )?);
    let run_once = |shape: &[i64]| -> Result<bgremove_ort::TensorOutput> {
        let n = shape.iter().product::<i64>() as usize;
        let values = (0..n)
            .map(|i| i as f32 / (n.saturating_sub(1).max(1) as f32))
            .collect::<Vec<_>>();
        let mut lease = pool.checkout();
        let output = lease.session_mut().run(shape, &values)?;
        ensure!(output.shape == shape, "M3 fixture changed output shape");
        ensure!(
            output.values == values,
            "M3 fixture arithmetic output changed"
        );
        Ok(output)
    };
    let first = run_once(&[1, 3, 2, 3])?;
    let second = run_once(&[1, 3, 3, 5])?;
    let mut joins: Vec<std::thread::JoinHandle<Result<bgremove_ort::TensorOutput>>> = Vec::new();
    for _ in 0..4 {
        let shared = std::sync::Arc::clone(&pool);
        joins.push(std::thread::spawn(move || {
            let mut lease = shared.checkout();
            let values = vec![0.25_f32; 6];
            let output = lease.session_mut().run(&[1, 3, 1, 2], &values)?;
            std::thread::sleep(std::time::Duration::from_millis(5));
            Ok(output)
        }));
    }
    for join in joins {
        join.join()
            .map_err(|_| anyhow!("M3 pool worker panicked"))??;
    }
    ensure!(
        pool.max_active() <= workers,
        "session pool exceeded worker bound"
    );
    let inspection = {
        let mut lease = pool.checkout();
        let mut inspection = lease.session_mut().inspection.clone();
        inspection.model_path = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&manifest.file)
            .display()
            .to_string();
        inspection
    };
    let provider = inspection.provider.clone();
    let hash = |v: &[f32]| {
        let mut h = Sha256::new();
        for x in v {
            h.update(x.to_le_bytes());
        }
        let digest = h.finalize();
        digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    fs::create_dir_all(output)?;
    let report = serde_json::json!({"schema":"m3.ort-smoke.v1","manifest":manifest_path.display().to_string(),"model_sha256":manifest.sha256,"inspection":inspection,"provider":provider,"workers":workers,"max_concurrent_sessions":workers,"observed_max_concurrency":pool.max_active(),"runs":[{"shape":first.shape,"values":first.values,"values_sha256":hash(&first.values)},{"shape":second.shape,"values":second.values,"values_sha256":hash(&second.values)}],"deterministic":true,"runtime_linkage":"external ORT_DYLIB; never downloaded by inference"});
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(output.join("report.json"), bytes)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn parse_policy(value: &str) -> std::result::Result<TransparentInputPolicy, String> {
    match value {
        "multiply-predicted" => Ok(TransparentInputPolicy::MultiplyPredicted),
        "replace-source-alpha" => Ok(TransparentInputPolicy::ReplaceSourceAlpha),
        _ => Err("expected multiply-predicted or replace-source-alpha".into()),
    }
}

fn write_m2_run(input: &Path, output: &Path, policy: TransparentInputPolicy) -> Result<()> {
    let image = load_canonical(input)?;
    let (width, height) = image.dimensions();
    let config = PipelineConfig::default()
        .with_transparent_input_policy(policy)
        .resolved_for(width, height)?;
    let mut pipeline = Pipeline::new(
        config.clone(),
        Box::new(NoOpSegmenter),
        Box::new(IdentityMaskTransform),
        Box::new(bgremove_matting::NoOpRefiner::default()),
        Box::new(OriginalRgbEstimator::default()),
    );
    let result = pipeline.run(&image, None)?;
    fs::create_dir_all(output)?;
    fs::write(
        output.join("resolved-config.json"),
        config.canonical_json_m2()?,
    )?;
    fs::write(
        output.join("cutout.png"),
        encode_straight_rgba_png(&result)?,
    )?;
    fs::write(output.join("mask.png"), encode_mask_png(result.alpha())?)?;
    let artifact = serde_json::json!({ "schema": "m2.bench-artifact.v1", "width": width, "height": height, "alpha_sum": result.alpha().data().iter().sum::<f32>(), "source_alpha_present": image.source_alpha().data().iter().any(|a| *a < 1.0), "transparent_input_policy": policy.as_str(), "png_outputs": ["cutout.png", "mask.png"], "note": "M2 deterministic no-op benchmark artifact" });
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    fs::write(output.join("run.json"), bytes)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn compare_m2_runs(left: &Path, right: &Path) -> Result<()> {
    let a = fs::read(left).with_context(|| format!("read left artifact {}", left.display()))?;
    let b = fs::read(right).with_context(|| format!("read right artifact {}", right.display()))?;
    let equal = a == b;
    println!(
        "equal={} left_bytes={} right_bytes={}",
        equal,
        a.len(),
        b.len()
    );
    if !equal {
        bail!("M2 artifact bytes differ")
    }
    Ok(())
}

fn read_manifest(path: &Path) -> Result<Vec<ManifestRecord>> {
    let file = File::open(path).with_context(|| format!("open manifest {}", path.display()))?;
    let mut records = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("read manifest line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parse manifest JSONL record on line {}", line_no + 1))?,
        );
    }
    Ok(records)
}

fn validate_manifest(path: &Path) -> Result<ValidationSummary> {
    let records = read_manifest(path)?;
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    let mut errors = Vec::new();
    let mut ids = BTreeSet::new();
    let mut splits = BTreeMap::<String, usize>::new();
    let mut input_hashes = HashMap::<String, String>::new();
    let mut reference_hashes = HashMap::<String, String>::new();
    let mut input_decoded_hashes = HashMap::<String, String>::new();
    let mut reference_decoded_hashes = HashMap::<String, String>::new();
    let mut paths = HashMap::<String, String>::new();
    let mut observed_tags = BTreeSet::new();
    for record in &records {
        if !ids.insert(record.id.clone()) {
            errors.push(format!("duplicate id {}", record.id));
        }
        *splits.entry(record.split.as_str().to_owned()).or_default() += 1;
        observed_tags.extend(record.tags.iter().cloned());
        validate_record(root, record, &mut errors);
        for (kind, raw_path) in [("input", &record.input), ("reference", &record.reference)] {
            if let Ok(path) = checked_path(root, raw_path) {
                let key = path.display().to_string();
                if let Some(previous) = paths.insert(key, format!("{}:{kind}", record.id)) {
                    errors.push(format!(
                        "path is reused by {previous} and {}:{kind}; possible leakage",
                        record.id
                    ));
                }
            }
        }
        check_hash_collision(
            &mut input_hashes,
            &record.input_sha256,
            &record.id,
            "input",
            &mut errors,
        );
        check_hash_collision(
            &mut reference_hashes,
            &record.reference_sha256,
            &record.id,
            "reference",
            &mut errors,
        );
        check_hash_collision(
            &mut input_decoded_hashes,
            &record.input_decoded_sha256,
            &record.id,
            "input decoded",
            &mut errors,
        );
        check_hash_collision(
            &mut reference_decoded_hashes,
            &record.reference_decoded_sha256,
            &record.id,
            "reference decoded",
            &mut errors,
        );
    }
    for required in ["tune", "validation", "blind"] {
        if !splits.contains_key(required) {
            errors.push(format!("required split {required} has no records"));
        }
    }
    if records.is_empty() {
        errors.push("manifest has no records".to_owned());
    }
    validate_duplicate_group_splits(&records, &mut errors);
    if let Some(arena_path) = root
        .parent()
        .map(|parent| parent.join("test_images/arena.jsonl"))
    {
        if arena_path.is_file() {
            validate_arena_consistency(root, &arena_path, &records, &mut errors);
        }
    }
    let coverage = coverage_report(&observed_tags);
    let valid = errors.is_empty();
    Ok(ValidationSummary {
        schema_version: SCHEMA_VERSION,
        manifest: path.display().to_string(),
        records: records.len(),
        split_counts: splits,
        coverage,
        blind_excluded_from_tuning: true,
        valid,
        errors,
    })
}

fn validate_duplicate_group_splits(records: &[ManifestRecord], errors: &mut Vec<String>) {
    let mut groups = HashMap::<&str, Split>::new();
    for record in records {
        if let Some(previous) = groups.insert(&record.duplicate_group, record.split) {
            if previous != record.split {
                errors.push(format!(
                    "duplicate_group {} crosses split boundaries",
                    record.duplicate_group
                ));
            }
        }
    }
}

fn validate_record(root: &Path, record: &ManifestRecord, errors: &mut Vec<String>) {
    validate_record_invariants(record, errors);
    let input_path = match checked_path(root, &record.input) {
        Ok(path) => path,
        Err(error) => {
            errors.push(format!("{} input path: {error:#}", record.id));
            return;
        }
    };
    let reference_path = match checked_path(root, &record.reference) {
        Ok(path) => path,
        Err(error) => {
            errors.push(format!("{} reference path: {error:#}", record.id));
            return;
        }
    };
    let input = match inspect_image(&input_path) {
        Ok(info) => Some(info),
        Err(error) => {
            errors.push(format!("{} input: {error:#}", record.id));
            None
        }
    };
    let reference = match inspect_image(&reference_path) {
        Ok(info) => Some(info),
        Err(error) => {
            errors.push(format!("{} reference: {error:#}", record.id));
            None
        }
    };
    if let (Some(input_info), Some(reference_info)) = (input.as_ref(), reference.as_ref()) {
        validate_pair_dimensions(record, input_info, reference_info, errors);
    }
    if let Some(info) = input.as_ref() {
        compare_info(
            record,
            "input",
            info,
            record.input_width,
            record.input_height,
            record.input_alpha_present,
            &record.input_decoded_sha256,
            errors,
        );
        if !hash_file(&input_path).is_ok_and(|hash| hash == record.input_sha256) {
            errors.push(format!("{} input file SHA-256 mismatch", record.id));
        }
        if !record
            .input_orientation
            .eq_ignore_ascii_case(&info.orientation)
        {
            errors.push(format!("{} input orientation mismatch", record.id));
        }
    }
    if let Some(info) = reference.as_ref() {
        compare_info(
            record,
            "reference",
            info,
            record.reference_width,
            record.reference_height,
            record.reference_alpha_present,
            &record.reference_decoded_sha256,
            errors,
        );
        if !hash_file(&reference_path).is_ok_and(|hash| hash == record.reference_sha256) {
            errors.push(format!("{} reference file SHA-256 mismatch", record.id));
        }
        if !record
            .reference_orientation
            .eq_ignore_ascii_case(&info.orientation)
        {
            errors.push(format!("{} reference orientation mismatch", record.id));
        }
        if info.width != record.width || info.height != record.height {
            errors.push(format!(
                "{} canonical dimensions do not match width/height",
                record.id
            ));
        }
        if !info.alpha_present {
            errors.push(format!(
                "{} PhotoRoom reference has no alpha channel",
                record.id
            ));
        }
        if info.png_bit_depth != Some(8) || info.png_color_type != Some(6) {
            errors.push(format!("{} PhotoRoom reference must be 8-bit RGBA PNG (got bit depth {:?}, color type {:?})", record.id, info.png_bit_depth, info.png_color_type));
        }
        if info.png_color_metadata == "unsupported" {
            errors.push(format!(
                "{} PhotoRoom reference has unsupported colour metadata",
                record.id
            ));
        }
        if record.reference_png_bit_depth != info.png_bit_depth {
            errors.push(format!(
                "{} reference_png_bit_depth metadata mismatch",
                record.id
            ));
        }
        if record.reference_png_color_type != info.png_color_type {
            errors.push(format!(
                "{} reference_png_color_type metadata mismatch",
                record.id
            ));
        }
        if record.reference_png_color_metadata != info.png_color_metadata {
            errors.push(format!(
                "{} reference_png_color_metadata mismatch",
                record.id
            ));
        }
        if info.alpha_levels != record.reference_alpha_levels {
            errors.push(format!(
                "{} reference_alpha_levels metadata mismatch",
                record.id
            ));
        }
        if info.alpha_levels != 256 {
            errors.push(format!(
                "{} PhotoRoom target must expose all 256 8-bit alpha levels (got {})",
                record.id, info.alpha_levels
            ));
        }
    }
    if record.tags.is_empty() {
        errors.push(format!("{} has no taxonomy tags", record.id));
    }
    if record.prompt.is_some() {
        errors.push(format!(
            "{} prompt must be null for automatic arena records",
            record.id
        ));
    }
    if record.subject_policy != SubjectPolicy::PrimarySubject {
        errors.push(format!("{} has an unsupported subject policy", record.id));
    }
    if record.shadow_policy != ShadowPolicy::PreserveTargetEffects {
        errors.push(format!("{} has an unsupported shadow policy", record.id));
    }
}

fn validate_record_invariants(record: &ManifestRecord, errors: &mut Vec<String>) {
    if record.schema_version != SCHEMA_VERSION {
        errors.push(format!(
            "{} schema_version must be {SCHEMA_VERSION}",
            record.id
        ));
    }
    if record.id.trim().is_empty() {
        errors.push("record id must not be empty or whitespace".to_owned());
    }
    if record.duplicate_group.trim().is_empty() {
        errors.push(format!(
            "{} duplicate_group must not be empty or whitespace",
            record.id
        ));
    }
    if record.tags.is_empty() {
        errors.push(format!("{} has no taxonomy tags", record.id));
    }
    let mut tags = BTreeSet::new();
    for tag in &record.tags {
        if tag.trim().is_empty() {
            errors.push(format!("{} has an empty taxonomy tag", record.id));
        }
        if !tags.insert(tag) {
            errors.push(format!("{} has duplicate taxonomy tag {tag:?}", record.id));
        }
    }
    for (name, hash) in [
        ("input_sha256", &record.input_sha256),
        ("reference_sha256", &record.reference_sha256),
        ("input_decoded_sha256", &record.input_decoded_sha256),
        ("reference_decoded_sha256", &record.reference_decoded_sha256),
    ] {
        if !is_sha256_hex(hash) {
            errors.push(format!(
                "{} {name} must be exactly 64 lowercase hexadecimal characters",
                record.id
            ));
        }
    }
    for (name, value) in [
        ("width", record.width),
        ("height", record.height),
        ("input_width", record.input_width),
        ("input_height", record.input_height),
        ("reference_width", record.reference_width),
        ("reference_height", record.reference_height),
    ] {
        if value == 0 {
            errors.push(format!("{} {name} must be positive", record.id));
        }
    }
    if let Some(date) = &record.reference_created_at {
        if !is_iso_date_shape(date) {
            errors.push(format!(
                "{} reference_created_at is not a valid ISO-8601 date/time shape",
                record.id
            ));
        }
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_iso_date_shape(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes[..4]
            .iter()
            .chain(&bytes[5..7])
            .chain(&bytes[8..10])
            .all(u8::is_ascii_digit)
    {
        return false;
    }
    let month = value[5..7].parse::<u8>().unwrap_or(0);
    let day = value[8..10].parse::<u8>().unwrap_or(0);
    if month == 0 || month > 12 || day == 0 || day > 31 {
        return false;
    }
    bytes.len() == 10 || (bytes[10] == b'T' && bytes.len() > 11)
}

fn validate_pair_dimensions(
    record: &ManifestRecord,
    input: &ImageInfo,
    reference: &ImageInfo,
    errors: &mut Vec<String>,
) {
    if input.width != reference.width || input.height != reference.height {
        errors.push(format!(
            "{} input/reference canonical dimensions differ ({}x{} vs {}x{})",
            record.id, input.width, input.height, reference.width, reference.height
        ));
    }
    if input.width != record.width || input.height != record.height {
        errors.push(format!(
            "{} input canonical dimensions do not match manifest width/height",
            record.id
        ));
    }
    if reference.width != record.width || reference.height != record.height {
        errors.push(format!(
            "{} reference canonical dimensions do not match manifest width/height",
            record.id
        ));
    }
}

fn read_arena(path: &Path) -> Result<Vec<ArenaRecord>> {
    let file = File::open(path).with_context(|| format!("open arena {}", path.display()))?;
    let mut records = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("read arena line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parse arena JSONL record on line {}", line_no + 1))?,
        );
    }
    Ok(records)
}

fn validate_arena_consistency(
    root: &Path,
    arena_path: &Path,
    manifest: &[ManifestRecord],
    errors: &mut Vec<String>,
) {
    let arena = match read_arena(arena_path) {
        Ok(records) => records,
        Err(error) => {
            errors.push(format!("arena: {error:#}"));
            return;
        }
    };
    if arena.len() != manifest.len() {
        errors.push(format!(
            "arena/manifest record count differs ({} vs {})",
            arena.len(),
            manifest.len()
        ));
    }
    let manifest_by_id: HashMap<&str, &ManifestRecord> = manifest
        .iter()
        .map(|record| (record.id.as_str(), record))
        .collect();
    let arena_root = arena_path.parent().unwrap_or_else(|| Path::new("."));
    for item in &arena {
        let Some(record) = manifest_by_id.get(item.id.as_str()) else {
            errors.push(format!("arena id {} is missing from manifest", item.id));
            continue;
        };
        if let (Ok(arena_input), Ok(manifest_input), Ok(arena_target), Ok(manifest_target)) = (
            canonical_under(arena_root, &item.input),
            checked_path(root, &record.input),
            canonical_under(arena_root, &item.target),
            checked_path(root, &record.reference),
        ) {
            if arena_input != manifest_input {
                errors.push(format!(
                    "arena {} input path differs from manifest",
                    item.id
                ));
            }
            if arena_target != manifest_target {
                errors.push(format!(
                    "arena {} target path differs from manifest",
                    item.id
                ));
            }
        } else {
            errors.push(format!(
                "arena {} contains an invalid input/target path",
                item.id
            ));
        }
        if item.split != record.split {
            errors.push(format!("arena {} split differs from manifest", item.id));
        }
        if item.duplicate_group != record.duplicate_group {
            errors.push(format!(
                "arena {} duplicate_group differs from manifest",
                item.id
            ));
        }
        if item.challenge.trim().is_empty() {
            errors.push(format!("arena {} challenge is empty", item.id));
        }
        let target_path = match checked_path(root, &record.reference) {
            Ok(path) => path,
            Err(_) => continue,
        };
        let target = match inspect_image(&target_path) {
            Ok(info) => info,
            Err(error) => {
                errors.push(format!("arena {} target: {error:#}", item.id));
                continue;
            }
        };
        if target.width != record.width || target.height != record.height {
            errors.push(format!("arena {} dimensions differ from manifest", item.id));
        }
        let fractional = target
            .rgba
            .pixels()
            .filter(|pixel| pixel[3] > 0 && pixel[3] < 255)
            .count() as f64
            / f64::from(target.width * target.height);
        if (fractional - item.fractional_alpha_reported).abs() > 0.000002 {
            errors.push(format!(
                "arena {} fractional-alpha ratio differs (declared {}, measured {})",
                item.id, item.fractional_alpha_reported, fractional
            ));
        }
    }
    let arena_ids: BTreeSet<&str> = arena.iter().map(|item| item.id.as_str()).collect();
    for record in manifest {
        if !arena_ids.contains(record.id.as_str()) {
            errors.push(format!("manifest id {} is missing from arena", record.id));
        }
    }
}

fn canonical_under(root: &Path, raw: &str) -> Result<PathBuf> {
    if raw.trim().is_empty() || Path::new(raw).is_absolute() {
        bail!("arena path must be relative and non-empty");
    }
    let path = fs::canonicalize(root.join(raw))?;
    let workspace = fs::canonicalize(root)?
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("arena root has no parent"))?;
    if !path.starts_with(&workspace) || !path.is_file() {
        bail!("invalid arena path {raw}");
    }
    Ok(path)
}

#[allow(clippy::too_many_arguments)]
fn compare_info(
    record: &ManifestRecord,
    kind: &str,
    info: &ImageInfo,
    width: u32,
    height: u32,
    alpha: bool,
    decoded_hash: &str,
    errors: &mut Vec<String>,
) {
    if info.width != width || info.height != height {
        errors.push(format!("{} {kind} dimensions mismatch", record.id));
    }
    if info.alpha_present != alpha {
        errors.push(format!("{} {kind} alpha presence mismatch", record.id));
    }
    if sha256_hex(info.rgba.as_raw()) != decoded_hash {
        errors.push(format!(
            "{} {kind} decoded canonical pixel SHA-256 mismatch",
            record.id
        ));
    }
}

fn check_hash_collision(
    map: &mut HashMap<String, String>,
    hash: &str,
    id: &str,
    kind: &str,
    errors: &mut Vec<String>,
) {
    if let Some(previous) = map.insert(hash.to_owned(), id.to_owned()) {
        if previous != id {
            errors.push(format!(
                "{kind} hash is shared by {previous} and {id}; possible leakage"
            ));
        }
    }
}

fn load_validated(manifest: &Path) -> Result<Vec<LoadedRecord>> {
    let summary = validate_manifest(manifest)?;
    if !summary.valid {
        bail!(
            "cannot continue: M0 manifest validation failed: {}",
            summary.errors.join("; ")
        );
    }
    let records = read_manifest(manifest)?;
    let root = manifest.parent().unwrap_or_else(|| Path::new("."));
    records
        .into_iter()
        .map(|record| {
            let input = inspect_image(&checked_path(root, &record.input)?)?;
            let reference = inspect_image(&checked_path(root, &record.reference)?)?;
            Ok(LoadedRecord {
                record,
                _input: input,
                reference,
            })
        })
        .collect()
}

fn inspect_image(path: &Path) -> Result<ImageInfo> {
    let mut reader = ImageReader::open(path).with_context(|| format!("open {}", path.display()))?;
    reader = reader
        .with_guessed_format()
        .with_context(|| format!("guess format for {}", path.display()))?;
    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("decode header {}", path.display()))?;
    let (raw_width, raw_height) = decoder.dimensions();
    let orientation = decoder
        .orientation()
        .with_context(|| format!("read orientation {}", path.display()))?;
    let orientation_name = format!("{orientation:?}");
    let raw_color = decoder.original_color_type();
    drop(decoder);
    let mut reader =
        ImageReader::open(path).with_context(|| format!("reopen {}", path.display()))?;
    reader = reader
        .with_guessed_format()
        .with_context(|| format!("guess format for {}", path.display()))?;
    let mut image = reader
        .decode()
        .with_context(|| format!("decode {}", path.display()))?;
    image.apply_orientation(orientation);
    let (width, height) = image.dimensions();
    let alpha_present = has_alpha(raw_color);
    let rgba = image.to_rgba8();
    let mut levels = [false; 256];
    for pixel in rgba.pixels() {
        levels[usize::from(pixel[3])] = true;
    }
    let alpha_levels = levels.into_iter().filter(|present| *present).count() as u16;
    let (png_bit_depth, png_color_type, png_color_metadata) = if path
        .extension()
        .is_some_and(|x| x.eq_ignore_ascii_case("png"))
    {
        parse_png_metadata(&fs::read(path)?)?
    } else {
        (None, None, "not-png".to_owned())
    };
    if raw_width == 0 || raw_height == 0 || width == 0 || height == 0 {
        bail!("{} has zero dimensions", path.display());
    }
    Ok(ImageInfo {
        width,
        height,
        orientation: orientation_name,
        alpha_present,
        png_bit_depth,
        png_color_type,
        png_color_metadata,
        alpha_levels,
        rgba,
    })
}

fn checked_path(root: &Path, raw: &str) -> Result<PathBuf> {
    let relative = Path::new(raw);
    if raw.trim().is_empty() {
        bail!("path is empty");
    }
    if relative.is_absolute() {
        bail!("absolute paths are not allowed");
    }
    let path = fs::canonicalize(root.join(relative)).with_context(|| format!("resolve {raw}"))?;
    let workspace = fs::canonicalize(root)?
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("manifest root has no parent"))?;
    if !path.starts_with(&workspace) {
        bail!("path escapes the workspace: {raw}");
    }
    if !path.is_file() {
        bail!("path is not a regular file: {raw}");
    }
    Ok(path)
}

fn has_alpha(color: image::ExtendedColorType) -> bool {
    matches!(
        color,
        image::ExtendedColorType::La8
            | image::ExtendedColorType::La16
            | image::ExtendedColorType::Rgba8
            | image::ExtendedColorType::Rgba16
            | image::ExtendedColorType::Rgba32F
    )
}

fn parse_png_metadata(bytes: &[u8]) -> Result<(Option<u8>, Option<u8>, String)> {
    const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if !bytes.starts_with(SIGNATURE) {
        bail!("invalid PNG signature");
    }
    let mut offset = 8usize;
    let mut bit_depth = None;
    let mut color_type = None;
    let mut srgb = false;
    let mut icc = false;
    let mut gamma = false;
    while offset + 12 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into()?) as usize;
        let end = offset
            .checked_add(12)
            .and_then(|x| x.checked_add(length))
            .ok_or_else(|| anyhow!("PNG chunk overflow"))?;
        if end > bytes.len() {
            bail!("truncated PNG chunk");
        }
        let kind = &bytes[offset + 4..offset + 8];
        let data = &bytes[offset + 8..offset + 8 + length];
        if kind == b"IHDR" && data.len() >= 13 {
            bit_depth = Some(data[8]);
            color_type = Some(data[9]);
        } else if kind == b"sRGB" {
            srgb = true;
        } else if kind == b"iCCP" {
            icc = true;
        } else if kind == b"gAMA" {
            gamma = true;
        }
        offset = end;
        if kind == b"IEND" {
            break;
        }
    }
    let colour = if srgb {
        "sRGB"
    } else if icc {
        "ICC"
    } else if gamma {
        "gamma-tagged (sRGB-compatible input convention)"
    } else {
        "unprofiled (PNG sRGB default)"
    };
    Ok((bit_depth, color_type, colour.to_owned()))
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(sha256_hex(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    ))
}
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn coverage_report(tags: &BTreeSet<String>) -> Coverage {
    const TAXONOMY: &[&str] = &[
        "portrait",
        "hair",
        "fur",
        "rigid-product",
        "food",
        "vehicle",
        "foliage",
        "holes",
        "thin-structures",
        "low-contrast",
        "reflections",
        "translucency",
        "glass",
        "shadows",
        "multiple-subjects",
        "edge-touching",
        "small-subject",
        "very-high-resolution",
        "very-low-resolution",
    ];
    let missing_taxonomy_tags = TAXONOMY
        .iter()
        .filter(|tag| !tags.contains(**tag))
        .map(|tag| (*tag).to_owned())
        .collect();
    Coverage { observed_tags: tags.iter().cloned().collect(), missing_taxonomy_tags, limitation: "The six supplied arena pairs are frozen for source-faithful comparison, but are not statistically adequate coverage of the broader taxonomy. Missing categories remain declared limitations, not malformed-corpus failures.".to_owned() }
}

fn write_baseline(manifest: &Path, output: &Path, records: &[LoadedRecord]) -> Result<()> {
    let mut images = Vec::with_capacity(records.len());
    for loaded in records {
        let reference = &loaded.reference.rgba;
        let alpha: Vec<f64> = reference
            .pixels()
            .map(|p| f64::from(p[3]) / 255.0)
            .collect();
        let fractional =
            alpha.iter().filter(|a| **a > 0.0 && **a < 1.0).count() as f64 / alpha.len() as f64;
        images.push(BaselineImage {
            id: loaded.record.id.clone(),
            split: loaded.record.split,
            width: loaded.reference.width,
            height: loaded.reference.height,
            reference_fractional_alpha: fractional,
            zero: candidate_metrics(&alpha, 0.0),
            one: candidate_metrics(&alpha, 1.0),
        });
    }
    images.sort_by(|a, b| a.id.cmp(&b.id));
    let aggregate = aggregate_metrics(&images);
    let tags: BTreeSet<_> = records
        .iter()
        .flat_map(|r| r.record.tags.iter().cloned())
        .collect();
    let report = BaselineReport {
        report_version: BASELINE_REPORT_VERSION, schema_version: SCHEMA_VERSION, manifest: manifest.display().to_string(),
        metric_definition: MetricDefinition {
            alpha_mae: "mean(abs(candidate_alpha - reference_alpha)) over canonical pixels".to_owned(),
            soft_iou: "sum(min(candidate_alpha, reference_alpha)) / sum(max(candidate_alpha, reference_alpha)); 1.0 when both sums are zero".to_owned(),
            alpha_levels: "Reference alpha is decoded as 8-bit RGBA and normalized to [0,1].".to_owned(),
            comparison: "M0 baseline is alpha-only and intentionally does not claim the Section 6 composite agreement score; full candidates begin in later milestones.".to_owned(),
        },
        tuning_policy: TuningPolicy {
            blind_is_evaluation_only: true, sweep_inputs: vec!["tune".to_owned(), "validation".to_owned()],
            statement: "All-zero and all-one controls are fixed reports, not tunable models. No sweep, threshold, or configuration decision may read blind records.".to_owned(),
        },
        images,
        aggregate,
        coverage: coverage_report(&tags),
        gate: GateStatus {
            status: "pass",
            every_item_valid: true,
            blind_not_used_by_sweeps: true,
            coverage_limitations_declared: true,
        },
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(output).with_context(|| format!("create {}", output.display()))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn candidate_metrics(reference: &[f64], value: f64) -> BaselineCandidate {
    let mae = reference
        .iter()
        .map(|alpha| (value - alpha).abs())
        .sum::<f64>()
        / reference.len() as f64;
    let intersection = reference.iter().map(|alpha| value.min(*alpha)).sum::<f64>();
    let union = reference.iter().map(|alpha| value.max(*alpha)).sum::<f64>();
    let soft_iou = if union == 0.0 {
        1.0
    } else {
        intersection / union
    };
    BaselineCandidate {
        alpha_mae: mae,
        soft_iou,
        agreement_alpha_only: 1.0 - mae,
        score_status: "control-only",
    }
}

fn aggregate_metrics(images: &[BaselineImage]) -> Aggregate {
    let mut by_split = BTreeMap::new();
    for split in [Split::Tune, Split::Validation, Split::Blind] {
        let subset: Vec<_> = images.iter().filter(|image| image.split == split).collect();
        let mut candidates = BTreeMap::new();
        candidates.insert(
            "all-zero".to_owned(),
            average_candidate(subset.iter().map(|image| &image.zero).collect()),
        );
        candidates.insert(
            "all-one".to_owned(),
            average_candidate(subset.iter().map(|image| &image.one).collect()),
        );
        by_split.insert(split.as_str().to_owned(), candidates);
    }
    let mut by_candidate = BTreeMap::new();
    by_candidate.insert(
        "all-zero".to_owned(),
        average_candidate(images.iter().map(|image| &image.zero).collect()),
    );
    by_candidate.insert(
        "all-one".to_owned(),
        average_candidate(images.iter().map(|image| &image.one).collect()),
    );
    Aggregate {
        by_candidate,
        by_split,
    }
}

fn average_candidate(candidates: Vec<&BaselineCandidate>) -> AggregateCandidate {
    let count = candidates.len();
    if count == 0 {
        return AggregateCandidate {
            image_count: 0,
            mean_alpha_mae: 0.0,
            mean_soft_iou: 0.0,
            mean_agreement_alpha_only: 0.0,
        };
    }
    AggregateCandidate {
        image_count: count,
        mean_alpha_mae: candidates
            .iter()
            .map(|candidate| candidate.alpha_mae)
            .sum::<f64>()
            / count as f64,
        mean_soft_iou: candidates
            .iter()
            .map(|candidate| candidate.soft_iou)
            .sum::<f64>()
            / count as f64,
        mean_agreement_alpha_only: candidates
            .iter()
            .map(|candidate| candidate.agreement_alpha_only)
            .sum::<f64>()
            / count as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    struct BaselineFixture {
        reference_alpha: Vec<f64>,
        all_zero_alpha_mae: f64,
        all_one_alpha_mae: f64,
    }

    #[test]
    fn png_metadata_rejects_bad_signature() {
        assert!(parse_png_metadata(b"not-png").is_err());
    }

    #[test]
    fn constant_baselines_are_deterministic_and_bounded() {
        let fixture: BaselineFixture =
            serde_json::from_str(include_str!("../../../tests/fixtures/alpha-baseline.json"))
                .unwrap();
        let zero = candidate_metrics(&fixture.reference_alpha, 0.0);
        let one = candidate_metrics(&fixture.reference_alpha, 1.0);
        assert!((zero.alpha_mae - fixture.all_zero_alpha_mae).abs() < f64::EPSILON);
        assert!((one.alpha_mae - fixture.all_one_alpha_mae).abs() < f64::EPSILON);
        assert!((0.0..=1.0).contains(&zero.soft_iou));
        assert!((0.0..=1.0).contains(&one.soft_iou));
    }

    #[test]
    fn duplicate_group_split_is_checked_by_validator() {
        let first = test_record("one", Split::Tune, "same");
        let second = test_record("two", Split::Blind, "same");
        let mut errors = Vec::new();
        validate_duplicate_group_splits(&[first, second], &mut errors);
        assert!(errors
            .iter()
            .any(|error| error.contains("crosses split boundaries")));
    }

    #[test]
    fn record_invariants_reject_bad_hash_schema_and_empty_tags() {
        let mut record = test_record(" ", Split::Tune, "group");
        record.schema_version = "wrong".to_owned();
        record.input_sha256 = "not-a-hash".to_owned();
        record.tags = vec![String::new(), "x".to_owned(), "x".to_owned()];
        let mut errors = Vec::new();
        validate_record_invariants(&record, &mut errors);
        assert!(errors.iter().any(|error| error.contains("schema_version")));
        assert!(errors
            .iter()
            .any(|error| error.contains("64 lowercase hexadecimal")));
        assert!(errors
            .iter()
            .any(|error| error.contains("empty taxonomy tag")));
        assert!(errors
            .iter()
            .any(|error| error.contains("duplicate taxonomy tag")));
    }

    #[test]
    fn serde_rejects_unknown_manifest_fields() {
        let mut value = serde_json::to_value(test_record("one", Split::Tune, "group")).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ManifestRecord>(value).is_err());
    }

    #[test]
    fn m6_accepted_report_has_five_truthful_quantitative_runs() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let report: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("runs/m6-carvekit/report.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report["status"], "pass");
        assert_eq!(report["tournament"]["identical_downstream"], true);
        let runs = report["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 5);
        let ids = runs
            .iter()
            .map(|run| run["model"]["id"].as_str().unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), 5);
        assert!(runs
            .iter()
            .any(|run| run["family"] == "u2net" && run["model"]["variant"] == "general"));
        for run in runs {
            assert_eq!(run["status"], "pass");
            assert_eq!(run["provider"]["active"], "CPUExecutionProvider");
            assert_eq!(run["provider"]["fallback_used"], false);
            for metric in ["alpha_mae", "soft_iou"] {
                let value = run["metrics"][metric].as_f64().unwrap();
                assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            }
            assert_eq!(run["input"]["path"], "test_images/reference/1.png");
            assert_eq!(run["reference"]["path"], "test_images/photoroom/1.png");
            assert_eq!(
                run["input"]["decoded_rgb_sha256"],
                runs[0]["input"]["decoded_rgb_sha256"]
            );
            assert_eq!(
                run["reference"]["decoded_alpha_sha256"],
                runs[0]["reference"]["decoded_alpha_sha256"]
            );
            assert_eq!(run["model"]["manifest_sha256"].as_str().unwrap().len(), 64);
            assert_eq!(run["model"]["model_sha256"].as_str().unwrap().len(), 64);
        }
        let image = load_canonical(&root.join("test_images/reference/1.png")).unwrap();
        let mut digest = Sha256::new();
        for pixel in image.rgb().data() {
            for value in pixel {
                digest.update(value.to_le_bytes());
            }
        }
        let input_hash = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(runs[0]["input"]["decoded_rgb_sha256"], input_hash);
        let reference = ImageReader::open(root.join("test_images/photoroom/1.png"))
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        let mut digest = Sha256::new();
        for pixel in reference.pixels() {
            digest.update((f32::from(pixel.0[3]) / 255.0).to_le_bytes());
        }
        let alpha_hash = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(runs[0]["reference"]["decoded_alpha_sha256"], alpha_hash);
    }

    #[test]
    fn pair_dimension_invariant_rejects_mismatch() {
        let record = test_record("one", Split::Tune, "group");
        let input = test_info(10, 20);
        let reference = test_info(11, 20);
        let mut errors = Vec::new();
        validate_pair_dimensions(&record, &input, &reference, &mut errors);
        assert!(errors
            .iter()
            .any(|error| error.contains("dimensions differ")));
    }

    fn test_record(id: &str, split: Split, duplicate_group: &str) -> ManifestRecord {
        ManifestRecord {
            schema_version: SCHEMA_VERSION.to_owned(),
            id: id.to_owned(),
            input: "input.png".to_owned(),
            reference: "reference.png".to_owned(),
            input_sha256: "a".repeat(64),
            reference_sha256: "b".repeat(64),
            input_decoded_sha256: "c".repeat(64),
            reference_decoded_sha256: "d".repeat(64),
            width: 10,
            height: 20,
            input_width: 10,
            input_height: 20,
            reference_width: 10,
            reference_height: 20,
            input_orientation: "NoTransforms".to_owned(),
            reference_orientation: "NoTransforms".to_owned(),
            input_alpha_present: false,
            reference_alpha_present: true,
            reference_png_bit_depth: Some(8),
            reference_png_color_type: Some(6),
            reference_png_color_metadata: "sRGB".to_owned(),
            reference_alpha_levels: 256,
            tags: vec!["portrait".to_owned()],
            split,
            subject_policy: SubjectPolicy::PrimarySubject,
            shadow_policy: ShadowPolicy::PreserveTargetEffects,
            prompt: None,
            reference_created_at: None,
            reference_tool: None,
            reference_tool_version: None,
            duplicate_group: duplicate_group.to_owned(),
            notes: None,
        }
    }

    fn test_info(width: u32, height: u32) -> ImageInfo {
        ImageInfo {
            width,
            height,
            orientation: "NoTransforms".to_owned(),
            alpha_present: true,
            png_bit_depth: Some(8),
            png_color_type: Some(6),
            png_color_metadata: "sRGB".to_owned(),
            alpha_levels: 256,
            rgba: image::RgbaImage::new(width, height),
        }
    }
}
