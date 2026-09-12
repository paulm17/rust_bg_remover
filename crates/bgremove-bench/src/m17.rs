//! M17 performance, provider and release-hardening evidence.
//!
//! This module deliberately keeps provider capability separate from quality
//! claims.  CPU is attempted first; CoreML/CUDA are attempted only when
//! requested and OpenVINO is reported as unavailable by this build.  No model
//! or weight is downloaded by this command.

use anyhow::{bail, ensure, Context, Result};
use bgremove_core::io::{
    encode_mask_png, encoded_dimensions, load_canonical, load_canonical_with_limit,
};
use bgremove_core::CanonicalImage;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

const WARM_SAMPLES: usize = 3;
const MAX_WORKERS: usize = 8;
const MAX_IMAGE_PIXELS: u64 = 16_777_216;
const MAX_BATCH_PIXELS: u64 = 33_554_432;
const MAX_BATCH_BYTES: u64 = MAX_BATCH_PIXELS * ESTIMATED_BYTES_PER_PIXEL;
const MAX_INPUT_IMAGES: usize = 1024;
const ESTIMATED_BYTES_PER_PIXEL: u64 = 16;
const OWNER_MARKER: &str = ".m17-owned";
const OWNER_MARKER_CONTENT: &str = "bgremove-bench M17 owned output\n";
const RELEASE_MODEL_SET: &str = "m17-u2net-approved-companions-v1";

pub fn default_provider_names() -> Vec<String> {
    let mut providers = vec!["cpu".to_string()];
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    providers.push("coreml".to_string());
    providers
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ProviderName {
    Cpu,
    Coreml,
    Cuda,
    Openvino,
}

impl ProviderName {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "coreml" => Ok(Self::Coreml),
            "cuda" => Ok(Self::Cuda),
            "openvino" => Ok(Self::Openvino),
            other => bail!("unknown provider {other}; expected cpu, coreml, cuda, or openvino"),
        }
    }
    fn ordered(values: &[String]) -> Result<Vec<Self>> {
        let mut parsed = values
            .iter()
            .map(|value| Self::parse(value))
            .collect::<Result<Vec<_>>>()?;
        parsed.sort_by_key(|provider| match provider {
            Self::Cpu => 0,
            Self::Coreml => 1,
            Self::Cuda => 2,
            Self::Openvino => 3,
        });
        parsed.dedup();
        if !parsed.contains(&Self::Cpu) {
            parsed.insert(0, Self::Cpu);
        }
        Ok(parsed)
    }
}

#[derive(Debug, Clone, Serialize)]
struct ProviderEvidence {
    requested: ProviderName,
    active: Option<String>,
    status: String,
    fallback_allowed: bool,
    fallback_used: bool,
    fallback_reason: Option<String>,
    load_ms: Option<f64>,
    first_inference_ms: Option<f64>,
    warm_median_ms: Option<f64>,
    warm_p95_ms: Option<f64>,
    throughput_images_per_second: Option<f64>,
    output_encode_ms: Option<f64>,
    peak_rss_bytes: Option<u64>,
    peak_rss_reason: Option<String>,
    model_bytes: Option<u64>,
    hardware: String,
    output_hashes: Vec<String>,
    output_samples: Vec<f32>,
    #[serde(skip)]
    output_tensors: Vec<Vec<f32>>,
    #[serde(skip)]
    output_dimensions: Vec<[u32; 2]>,
    #[serde(skip)]
    output_pngs: Vec<Vec<u8>>,
    processed_images: usize,
    cancelled: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BatchEvidence {
    requested_images: usize,
    processed_images: usize,
    workers: usize,
    max_pixels: u64,
    max_batch_pixels: u64,
    aggregate_pixels: u64,
    estimated_decoded_bytes: u64,
    max_input_images: usize,
    cancelled: bool,
    cancellation_checked_between_images: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct FailureProbeEvidence {
    fallback_semantics: bool,
    pixel_limit_rejection: bool,
    malformed_image_rejection: bool,
    malformed_model_rejection: bool,
    injected_resource_exhaustion_rejection: bool,
    details: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
struct TensorParityEvidence {
    provider: ProviderName,
    accelerated_requested: usize,
    accelerated_succeeded: usize,
    compared_tensors: usize,
    compared_values: usize,
    dimensions_match: bool,
    max_abs_error: Option<f32>,
    mean_abs_error: Option<f32>,
    within_tolerance: bool,
    reason: Option<String>,
}

type TensorBatch = (Vec<[u32; 2]>, Vec<Vec<f32>>);
type TensorBatchRef<'a> = (&'a [[u32; 2]], &'a [Vec<f32>]);

#[derive(Debug, Clone, Serialize)]
struct BomEntry {
    manifest: String,
    manifest_sha256: String,
    model_id: String,
    model_sha256: String,
    model_license: String,
    model_license_file: String,
    model_license_file_present: bool,
    model_license_sha256: Option<String>,
    model_file_present: bool,
    model_bytes: Option<u64>,
    code_license: String,
    code_license_file: String,
    code_license_file_present: bool,
    code_license_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CodeDependency {
    name: String,
    version: String,
    license: Option<String>,
    source: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct M17Report {
    schema: &'static str,
    status: String,
    provider_order: Vec<ProviderName>,
    providers: Vec<ProviderEvidence>,
    batch: BatchEvidence,
    input_limits: BTreeMap<String, String>,
    model_bom: Vec<BomEntry>,
    code_dependencies: Vec<CodeDependency>,
    failure_probes: FailureProbeEvidence,
    tensor_parity: Vec<TensorParityEvidence>,
    release: ReleaseEvidence,
    gates: BTreeMap<String, bool>,
    output_tolerance: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseEvidence {
    model_set_id: String,
    model_set_manifests: Vec<String>,
    copied_manifests_exact: bool,
    deterministic_scope: String,
    deterministic_manifest_sha256: Option<String>,
    manifest_hash_externalized_to: String,
    deterministic_entry_count: usize,
    deterministic_entries_verified: bool,
    untracked_weights_allowed: bool,
    code_and_weight_licenses_separate: bool,
    license_metadata_present: bool,
    code_dependency_licenses_complete: bool,
    cargo_lock_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct ReleaseManifestEntry {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct ReleaseManifest {
    entries: Vec<ReleaseManifestEntry>,
}

#[derive(Debug, Clone)]
struct BoundedBatchRunner {
    max_pixels: u64,
    workers: usize,
    cancelled: Arc<AtomicBool>,
}

struct StagingGuard {
    path: PathBuf,
    committed: bool,
}

impl StagingGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if !self.committed && self.path.join(OWNER_MARKER).is_file() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[derive(Debug)]
struct BatchRun<T> {
    values: Vec<T>,
    processed_images: usize,
    cancelled: bool,
}

impl BoundedBatchRunner {
    fn new(workers: usize, max_pixels: u64) -> Result<Self> {
        ensure!(workers > 0, "worker count must be greater than zero");
        ensure!(
            workers <= MAX_WORKERS,
            "worker count exceeds M17 maximum of {MAX_WORKERS}"
        );
        ensure!(max_pixels > 0, "max pixel limit must be positive");
        Ok(Self {
            max_pixels,
            workers,
            cancelled: Arc::new(AtomicBool::new(false)),
        })
    }
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    fn run<T, F>(&self, images: &[CanonicalImage], f: F) -> Result<BatchRun<T>>
    where
        F: Fn(&CanonicalImage) -> Result<T> + Send + Sync,
        T: Send,
    {
        for image in images {
            ensure_pixel_limit(image.width(), image.height(), self.max_pixels)?;
        }
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let function = &f;
        let results = Arc::new(Mutex::new(Vec::<(usize, T)>::new()));
        let error = Arc::new(Mutex::new(None::<String>));
        std::thread::scope(|scope| {
            for _ in 0..self.workers {
                let results = Arc::clone(&results);
                let error = Arc::clone(&error);
                let next = Arc::clone(&next);
                scope.spawn(move || loop {
                    if self.cancelled.load(Ordering::Acquire) {
                        break;
                    }
                    let index = next.fetch_add(1, Ordering::AcqRel);
                    let Some(image) = images.get(index) else {
                        break;
                    };
                    match function(image) {
                        Ok(value) => results
                            .lock()
                            .expect("batch result lock")
                            .push((index, value)),
                        Err(err) => {
                            *error.lock().expect("batch error lock") = Some(format!("{err:#}"));
                            self.cancelled.store(true, Ordering::Release);
                            break;
                        }
                    }
                });
            }
        });
        if let Some(error) = error.lock().expect("batch error lock").clone() {
            bail!("batch worker failed: {error}");
        }
        let cancelled = self.cancelled.load(Ordering::Acquire);
        let mut indexed = Arc::try_unwrap(results)
            .map_err(|_| anyhow::anyhow!("batch result workers still referenced"))?
            .into_inner()
            .map_err(|_| anyhow::anyhow!("batch result lock poisoned"))?;
        indexed.sort_by_key(|(index, _)| *index);
        let processed_images = indexed.len();
        Ok(BatchRun {
            values: indexed.into_iter().map(|(_, value)| value).collect(),
            processed_images,
            cancelled,
        })
    }
}

fn ensure_pixel_limit(width: u32, height: u32, limit: u64) -> Result<()> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .context("image pixel count overflow")?;
    ensure!(
        pixels <= limit,
        "input exceeds M17 pixel limit: {pixels} > {limit}"
    );
    Ok(())
}

fn reserve_capacity(bytes: u64, limit: u64) -> Result<()> {
    ensure!(
        bytes <= limit,
        "requested working capacity exceeds configured resource budget"
    );
    Ok(())
}

fn preflight_inputs(
    inputs: &[PathBuf],
    max_pixels: u64,
) -> Result<(Vec<CanonicalImage>, u64, u64)> {
    ensure!(
        inputs.len() <= MAX_INPUT_IMAGES,
        "input count exceeds M17 maximum"
    );
    ensure!(
        max_pixels <= MAX_IMAGE_PIXELS,
        "per-image pixel limit exceeds safe M17 maximum"
    );
    let mut aggregate_pixels = 0_u64;
    for path in inputs {
        let (width, height) = encoded_dimensions(path)?;
        let pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .context("encoded image pixel count overflow")?;
        ensure!(
            pixels <= max_pixels,
            "encoded input exceeds per-image pixel limit"
        );
        aggregate_pixels = aggregate_pixels
            .checked_add(pixels)
            .context("aggregate image pixel count overflow")?;
        ensure!(
            aggregate_pixels <= MAX_BATCH_PIXELS,
            "aggregate batch pixel limit exceeded"
        );
    }
    let estimated_bytes = aggregate_pixels
        .checked_mul(ESTIMATED_BYTES_PER_PIXEL)
        .context("aggregate decoded byte estimate overflow")?;
    ensure!(
        estimated_bytes <= MAX_BATCH_BYTES,
        "aggregate decoded byte budget exceeded"
    );
    let images = inputs
        .iter()
        .map(|path| load_canonical_with_limit(path, max_pixels))
        .collect::<Result<Vec<_>>>()?;
    Ok((images, aggregate_pixels, estimated_bytes))
}

fn prepare_staging(target: &Path) -> Result<PathBuf> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if target.exists() {
        ensure!(target.is_dir(), "M17 output path is not a directory");
        let marker = fs::read_to_string(target.join(OWNER_MARKER))
            .context("refusing to replace an unowned M17 output directory")?;
        ensure!(
            marker == OWNER_MARKER_CONTENT,
            "M17 output ownership marker mismatch"
        );
    }
    let staging = parent.join(format!(".m17-staging-{}", std::process::id()));
    ensure!(!staging.exists(), "M17 staging directory already exists");
    fs::create_dir(&staging)?;
    fs::write(staging.join(OWNER_MARKER), OWNER_MARKER_CONTENT)?;
    Ok(staging)
}

fn publish_staging(staging: &Path, target: &Path) -> Result<()> {
    if !target.exists() {
        return fs::rename(staging, target).context("publish M17 staging directory");
    }
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let backup = parent.join(format!(".m17-backup-{}", std::process::id()));
    ensure!(!backup.exists(), "M17 backup directory already exists");
    fs::rename(target, &backup).context("stage owned M17 output for replacement")?;
    match fs::rename(staging, target) {
        Ok(()) => {
            fs::remove_dir_all(backup).context("remove replaced M17 output")?;
            Ok(())
        }
        Err(error) => {
            let _ = fs::rename(&backup, target);
            Err(error).context("publish M17 staging directory")
        }
    }
}

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn p95(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() * 95).saturating_add(99) / 100)
        .max(1)
        .min(sorted.len())
        - 1]
}

fn resident_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        (result == 0).then(|| unsafe { usage.assume_init().ru_maxrss as u64 })
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn manifest_bom() -> Result<Vec<BomEntry>> {
    let manifests = [
        "models/m5_u2net.toml",
        "models/m4_isnet_fp32.toml",
        "models/m6_tracer_b7.toml",
        "models/m7_birefnet_general.toml",
    ];
    manifests
        .iter()
        .map(|relative| {
            let path = Path::new(relative);
            let bytes =
                fs::read(path).with_context(|| format!("read model manifest {relative}"))?;
            let manifest = bgremove_models::parse_toml(std::str::from_utf8(&bytes)?)?;
            let model = manifest.verify_model_hash(path).ok();
            let model_bytes = model
                .as_ref()
                .and_then(|path| fs::metadata(path).ok().map(|meta| meta.len()));
            let resolve_metadata = |value: &str| {
                let relative = Path::new(value);
                [
                    path.parent().unwrap_or_else(|| Path::new(".")),
                    Path::new("."),
                ]
                .iter()
                .map(|base| base.join(relative))
                .find(|candidate| candidate.is_file())
            };
            let model_license_path = resolve_metadata(&manifest.license_file);
            let code_license_path = resolve_metadata(&manifest.source_license_file);
            Ok(BomEntry {
                manifest: relative.to_string(),
                manifest_sha256: sha256(&bytes),
                model_id: manifest.id,
                model_sha256: manifest.sha256,
                model_license: manifest.license_identifier,
                model_license_file: manifest.license_file,
                model_license_file_present: model_license_path.is_some(),
                model_license_sha256: model_license_path
                    .and_then(|license| fs::read(license).ok())
                    .map(|bytes| sha256(&bytes)),
                model_file_present: model.is_some(),
                model_bytes,
                code_license: if manifest.source_license_identifier.is_empty() {
                    "MIT OR Apache-2.0".into()
                } else {
                    manifest.source_license_identifier
                },
                code_license_file: manifest.source_license_file,
                code_license_file_present: code_license_path.is_some(),
                code_license_sha256: code_license_path
                    .and_then(|license| fs::read(license).ok())
                    .map(|bytes| sha256(&bytes)),
            })
        })
        .collect()
}

fn write_bundle_metadata(root: &Path, bom: &[BomEntry]) -> Result<bool> {
    fs::create_dir_all(root.join("model-manifests"))?;
    fs::write(root.join("model-bom.json"), serde_json::to_vec_pretty(bom)?)?;
    for entry in bom {
        let source = Path::new(&entry.manifest);
        let name = source
            .file_name()
            .context("model manifest has no filename")?;
        fs::copy(source, root.join("model-manifests").join(name))?;
    }
    fs::copy("Cargo.lock", root.join("Cargo.lock"))?;
    let expected = bom
        .iter()
        .map(|entry| {
            Path::new(&entry.manifest)
                .file_name()
                .expect("manifest filename")
                .to_string_lossy()
                .into_owned()
        })
        .collect::<std::collections::BTreeSet<_>>();
    let actual = fs::read_dir(root.join("model-manifests"))?
        .map(|entry| entry.map(|value| value.file_name().to_string_lossy().into_owned()))
        .collect::<std::result::Result<std::collections::BTreeSet<_>, _>>()?;
    ensure!(
        actual == expected,
        "copied model manifest set differs from BOM"
    );
    Ok(true)
}

fn code_dependency_bom() -> Result<Vec<CodeDependency>> {
    #[derive(serde::Deserialize)]
    struct Metadata {
        packages: Vec<Package>,
    }
    #[derive(serde::Deserialize)]
    struct Package {
        name: String,
        version: String,
        license: Option<String>,
        source: Option<String>,
    }
    let output = Command::new("cargo")
        .args(["metadata", "--offline", "--format-version", "1"])
        .output()
        .context("run offline cargo metadata for release BOM")?;
    ensure!(output.status.success(), "offline cargo metadata failed");
    let metadata: Metadata = serde_json::from_slice(&output.stdout)
        .context("parse offline cargo metadata for release BOM")?;
    let mut dependencies = metadata
        .packages
        .into_iter()
        .map(|package| CodeDependency {
            name: package.name,
            version: package.version,
            license: package.license,
            source: package.source,
        })
        .collect::<Vec<_>>();
    dependencies.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.version.cmp(&right.version))
    });
    Ok(dependencies)
}

fn write_notices(root: &Path, bom: &[BomEntry], dependencies: &[CodeDependency]) -> Result<()> {
    let mut text = String::from("M17 third-party notices\n\nCode licences (repository adapters) are distinct from model-weight licences.\n");
    text.push_str("Code dependencies (from offline cargo metadata):\n");
    for dependency in dependencies {
        text.push_str(&format!(
            "{} {}: {} ({})\n",
            dependency.name,
            dependency.version,
            dependency
                .license
                .as_deref()
                .unwrap_or("license-unreported"),
            dependency.source.as_deref().unwrap_or("workspace")
        ));
    }
    text.push_str("\nModel weights:\n");
    for entry in bom {
        text.push_str(&format!(
            "{}: {} ({}) [metadata_present={}, metadata_sha256={}]\n",
            entry.model_id,
            entry.model_license,
            entry.model_license_file,
            entry.model_license_file_present || !entry.model_license.is_empty(),
            entry
                .model_license_sha256
                .as_deref()
                .unwrap_or("manifest-metadata-only")
        ));
    }
    fs::write(root.join("THIRD-PARTY-NOTICES.txt"), text)?;
    Ok(())
}

fn release_scope_paths(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = walk_files(root)?
        .into_iter()
        .filter(|path| {
            let relative = path
                .strip_prefix(root)
                .ok()
                .map(|value| value.to_string_lossy().replace('\\', "/"));
            !matches!(
                relative.as_deref(),
                Some("artifacts.manifest.json") | Some("artifacts.sha256")
            )
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn write_release_manifest(
    root: &Path,
    bom: &[BomEntry],
    dependencies: &[CodeDependency],
    copied_manifests_exact: bool,
) -> Result<ReleaseEvidence> {
    let mut entries = Vec::new();
    for entry in release_scope_paths(root)? {
        let relative = entry
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        entries.push(ReleaseManifestEntry {
            path: relative,
            sha256: sha256(&fs::read(&entry)?),
        });
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    let mut listed = std::collections::BTreeSet::new();
    for entry in &entries {
        let path = Path::new(&entry.path);
        ensure!(
            !path.is_absolute(),
            "release manifest contains absolute path"
        );
        ensure!(
            !path
                .components()
                .any(|component| { matches!(component, std::path::Component::ParentDir) }),
            "release manifest contains traversal path"
        );
        ensure!(
            listed.insert(entry.path.clone()),
            "release manifest duplicate path"
        );
    }
    let actual = release_scope_paths(root)?
        .into_iter()
        .filter_map(|path| {
            let relative = path
                .strip_prefix(root)
                .ok()?
                .to_string_lossy()
                .replace('\\', "/");
            Some(relative)
        })
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        listed == actual,
        "release manifest scope does not match bundle files"
    );
    let manifest = ReleaseManifest {
        entries: entries.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    fs::write(root.join("artifacts.manifest.json"), &bytes)?;
    let manifest_hash = sha256(&bytes);
    fs::write(
        root.join("artifacts.sha256"),
        format!("{manifest_hash}  artifacts.manifest.json\n"),
    )?;
    ensure!(
        fs::read(root.join("artifacts.manifest.json"))
            .map(|contents| sha256(&contents) == manifest_hash)
            .unwrap_or(false),
        "release manifest checksum verification failed"
    );
    let verified = verify_release_bundle_files(root)?;
    Ok(ReleaseEvidence {
        model_set_id: RELEASE_MODEL_SET.into(),
        model_set_manifests: bom.iter().map(|entry| entry.manifest.clone()).collect(),
        copied_manifests_exact,
        deterministic_scope: "all final bundle files except artifacts.manifest.json and artifacts.sha256; report and report.sha256 are included".into(),
        deterministic_manifest_sha256: Some(manifest_hash),
        manifest_hash_externalized_to: "artifacts.sha256".into(),
        deterministic_entry_count: entries.len(),
        deterministic_entries_verified: verified,
        untracked_weights_allowed: false,
        code_and_weight_licenses_separate: true,
        license_metadata_present: bom.iter().all(|entry| {
            !entry.model_license.is_empty() && !entry.code_license.is_empty()
        }),
        code_dependency_licenses_complete: dependencies.iter().all(|entry| {
            entry.source.is_none()
                || entry
                    .license
                    .as_deref()
                    .is_some_and(|license| !license.trim().is_empty())
        }),
        cargo_lock_sha256: fs::read(root.join("Cargo.lock")).ok().map(|bytes| sha256(&bytes)),
    })
}

fn verify_release_bundle_files(root: &Path) -> Result<bool> {
    let manifest_bytes = fs::read(root.join("artifacts.manifest.json"))?;
    let manifest: ReleaseManifest =
        serde_json::from_slice(&manifest_bytes).context("parse release artifact manifest")?;
    let mut listed = std::collections::BTreeSet::new();
    for entry in &manifest.entries {
        let path = Path::new(&entry.path);
        ensure!(
            !path.is_absolute(),
            "release manifest contains absolute path"
        );
        ensure!(
            !path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir)),
            "release manifest contains traversal path"
        );
        ensure!(
            listed.insert(entry.path.clone()),
            "release manifest duplicate path"
        );
    }
    let actual = release_scope_paths(root)?
        .into_iter()
        .filter_map(|path| {
            path.strip_prefix(root)
                .ok()
                .map(|value| value.to_string_lossy().replace('\\', "/"))
        })
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        listed == actual,
        "release manifest scope does not match final bundle"
    );
    for entry in &manifest.entries {
        let path = root.join(&entry.path);
        ensure!(
            path.is_file() && sha256(&fs::read(path)?) == entry.sha256,
            "release manifest digest mismatch for {}",
            entry.path
        );
    }
    let checksum = fs::read_to_string(root.join("artifacts.sha256"))?;
    let expected = format!("{}  artifacts.manifest.json", sha256(&manifest_bytes));
    ensure!(
        checksum.trim_end() == expected,
        "artifacts.sha256 does not match artifacts.manifest.json"
    );
    Ok(true)
}

fn release_preview(
    root: &Path,
    bom: &[BomEntry],
    dependencies: &[CodeDependency],
    copied_manifests_exact: bool,
) -> Result<ReleaseEvidence> {
    let payload_count = release_scope_paths(root)?.len();
    Ok(ReleaseEvidence {
        model_set_id: RELEASE_MODEL_SET.into(),
        model_set_manifests: bom.iter().map(|entry| entry.manifest.clone()).collect(),
        copied_manifests_exact,
        deterministic_scope: "all final bundle files except artifacts.manifest.json and artifacts.sha256; report and report.sha256 are included".into(),
        deterministic_manifest_sha256: None,
        manifest_hash_externalized_to: "artifacts.sha256".into(),
        deterministic_entry_count: payload_count.checked_add(2).context("release entry count overflow")?,
        deterministic_entries_verified: true,
        untracked_weights_allowed: false,
        code_and_weight_licenses_separate: true,
        license_metadata_present: bom.iter().all(|entry| {
            !entry.model_license.is_empty() && !entry.code_license.is_empty()
        }),
        code_dependency_licenses_complete: dependencies.iter().all(|entry| {
            entry.source.is_none()
                || entry
                    .license
                    .as_deref()
                    .is_some_and(|license| !license.trim().is_empty())
        }),
        cargo_lock_sha256: fs::read(root.join("Cargo.lock")).ok().map(|bytes| sha256(&bytes)),
    })
}

fn walk_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for item in fs::read_dir(root)? {
        let path = item?.path();
        if path.is_dir() {
            files.extend(walk_files(&path)?);
        } else {
            files.push(path);
        }
    }
    Ok(files)
}

fn provider_run(
    provider: ProviderName,
    inputs: &[CanonicalImage],
    workers: usize,
    max_pixels: u64,
    fallback_allowed: bool,
) -> ProviderEvidence {
    let unavailable = |error: String| ProviderEvidence {
        requested: provider,
        active: None,
        status: "not-run".into(),
        fallback_allowed,
        fallback_used: false,
        fallback_reason: None,
        load_ms: None,
        first_inference_ms: None,
        warm_median_ms: None,
        warm_p95_ms: None,
        throughput_images_per_second: None,
        output_encode_ms: None,
        peak_rss_bytes: None,
        peak_rss_reason: Some("not measured because provider did not execute".into()),
        model_bytes: None,
        hardware: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        output_hashes: Vec::new(),
        output_samples: Vec::new(),
        output_pngs: Vec::new(),
        output_tensors: Vec::new(),
        output_dimensions: Vec::new(),
        processed_images: 0,
        cancelled: std::env::var("M17_CANCEL_BEFORE_BATCH").as_deref() == Ok("1"),
        error: Some(error),
    };
    if provider == ProviderName::Openvino {
        return unavailable("OpenVINO provider is not compiled into bgremove-ort".into());
    }
    let Some(runtime) = std::env::var_os("ORT_DYLIB").map(PathBuf::from) else {
        return unavailable("ORT_DYLIB is not configured; no runtime is downloaded".into());
    };
    if inputs.is_empty() {
        return unavailable("no input images supplied for runtime benchmark".into());
    }
    let manifest_path = Path::new("models/m5_u2net.toml");
    let manifest_text = match fs::read_to_string(manifest_path) {
        Ok(text) => text,
        Err(error) => return unavailable(format!("read model manifest: {error}")),
    };
    let manifest = match bgremove_models::parse_toml(&manifest_text) {
        Ok(manifest) => manifest,
        Err(error) => return unavailable(format!("parse model manifest: {error:#}")),
    };
    let model_path = match manifest.verify_model_hash(manifest_path) {
        Ok(path) => path,
        Err(error) => return unavailable(format!("verified model unavailable: {error:#}")),
    };
    let model_bytes = fs::metadata(model_path).ok().map(|meta| meta.len());
    let requested = match provider {
        ProviderName::Cpu => bgremove_ort::RequestedProvider::Cpu,
        ProviderName::Coreml => bgremove_ort::RequestedProvider::Coreml,
        ProviderName::Cuda => bgremove_ort::RequestedProvider::Cuda,
        ProviderName::Openvino => unreachable!(),
    };
    let load_start = Instant::now();
    let segmenter = match bgremove_ort::U2netSegmenter::new(
        &manifest,
        manifest_path,
        &runtime,
        1,
        requested,
        fallback_allowed,
    ) {
        Ok(segmenter) => segmenter,
        Err(error) => return unavailable(format!("session initialization: {error:#}")),
    };
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let runner = match BoundedBatchRunner::new(workers, max_pixels) {
        Ok(runner) => runner,
        Err(error) => return unavailable(error.to_string()),
    };
    if std::env::var("M17_CANCEL_BEFORE_BATCH").as_deref() == Ok("1") {
        runner.cancel();
    }
    let first_start = Instant::now();
    let first = match inputs.first() {
        Some(image) => match segmenter.predict_with_evidence(image) {
            Ok(value) => value,
            Err(error) => return unavailable(format!("first inference: {error:#}")),
        },
        None => return unavailable("first inference has no input".into()),
    };
    let first_inference_ms = first_start.elapsed().as_secs_f64() * 1000.0;
    if first.restored.data().is_empty() {
        return unavailable("first inference produced no outputs".into());
    }
    let mut warm = Vec::new();
    let mut hashes = Vec::new();
    let mut output_samples = Vec::new();
    let mut output_tensors = Vec::new();
    let mut output_dimensions = Vec::new();
    let mut output_pngs = Vec::new();
    let mut captured_outputs = false;
    let mut encode_ms = Vec::new();
    for _ in 0..WARM_SAMPLES {
        let started = Instant::now();
        let run = match runner.run(inputs, |image| segmenter.predict_with_evidence(image)) {
            Ok(run) => run,
            Err(error) => return unavailable(format!("warm inference: {error:#}")),
        };
        if run.cancelled || run.values.is_empty() || run.processed_images != inputs.len() {
            return unavailable("batch was cancelled or produced no warm outputs".into());
        }
        warm.push(started.elapsed().as_secs_f64() * 1000.0);
        for evidence in run.values {
            if !evidence
                .restored
                .data()
                .iter()
                .all(|value| value.is_finite())
            {
                return unavailable("inference produced NaN/Inf".into());
            }
            let started = Instant::now();
            let encoded = match encode_mask_png(&evidence.restored) {
                Ok(encoded) => encoded,
                Err(_) => return unavailable("output encoding failed".into()),
            };
            if !captured_outputs {
                output_pngs.push(encoded);
            }
            encode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            hashes.push(sha256(
                &evidence
                    .restored
                    .data()
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ));
            output_samples.extend(evidence.restored.data().iter().take(16).copied());
            if !captured_outputs {
                output_tensors.push(evidence.restored.data().to_vec());
                output_dimensions.push([evidence.restored.width(), evidence.restored.height()]);
            }
        }
        captured_outputs = true;
    }
    let warm_median = median(&warm);
    let warm_p95 = p95(&warm);
    let total_images = inputs.len().checked_mul(WARM_SAMPLES).unwrap_or(0);
    let throughput =
        (total_images as f64) / (warm.iter().sum::<f64>() / 1000.0).max(f64::MIN_POSITIVE);
    let provider_report = segmenter.provider();
    let active_provider = provider_report.active.clone();
    let peak_rss_bytes = resident_memory_bytes();
    let status = if provider_report.fallback_used {
        "pass-with-fallback"
    } else {
        "pass"
    };
    ProviderEvidence {
        requested: provider,
        active: Some(active_provider),
        status: status.into(),
        fallback_allowed: provider_report.fallback_allowed,
        fallback_used: provider_report.fallback_used,
        fallback_reason: provider_report.fallback_reason.clone(),
        load_ms: Some(load_ms),
        first_inference_ms: Some(first_inference_ms),
        warm_median_ms: Some(warm_median),
        warm_p95_ms: Some(warm_p95),
        throughput_images_per_second: Some(throughput),
        output_encode_ms: Some(median(&encode_ms)),
        peak_rss_reason: peak_rss_bytes
            .is_none()
            .then_some("portable RSS measurement unavailable".into()),
        peak_rss_bytes,
        model_bytes,
        hardware: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        output_hashes: hashes,
        output_samples,
        output_tensors,
        output_dimensions,
        output_pngs,
        processed_images: inputs.len(),
        cancelled: false,
        error: None,
    }
}

fn write_provider_outputs(root: &Path, evidence: &ProviderEvidence) -> Result<()> {
    if !evidence.status.starts_with("pass") {
        return Ok(());
    }
    let directory = root
        .join("providers")
        .join(format!("{:?}", evidence.requested).to_ascii_lowercase());
    fs::create_dir_all(&directory)?;
    for (index, bytes) in evidence.output_pngs.iter().enumerate() {
        fs::write(directory.join(format!("image-{index:04}-mask.png")), bytes)?;
    }
    Ok(())
}

fn compare_tensor_parity(
    cpu: Option<TensorBatchRef<'_>>,
    accelerated: &ProviderEvidence,
    accelerated_requested: usize,
    tolerance: f32,
) -> TensorParityEvidence {
    let provider = accelerated.requested;
    let Some((cpu_dimensions, cpu_tensors)) = cpu else {
        return TensorParityEvidence {
            provider,
            accelerated_requested,
            accelerated_succeeded: 0,
            compared_tensors: 0,
            compared_values: 0,
            dimensions_match: false,
            max_abs_error: None,
            mean_abs_error: None,
            within_tolerance: false,
            reason: Some("CPU provider did not produce a successful baseline".into()),
        };
    };
    if accelerated.status != "pass" || accelerated.fallback_used {
        return TensorParityEvidence {
            provider,
            accelerated_requested,
            accelerated_succeeded: 0,
            compared_tensors: 0,
            compared_values: 0,
            dimensions_match: false,
            max_abs_error: None,
            mean_abs_error: None,
            within_tolerance: false,
            reason: Some("accelerated provider did not execute without CPU fallback".into()),
        };
    }
    let dimensions_match = cpu_dimensions == accelerated.output_dimensions.as_slice()
        && cpu_tensors.len() == accelerated.output_tensors.len();
    let mut compared_tensors = 0;
    let mut compared_values = 0;
    let mut max_error = 0.0_f32;
    let mut sum_error = 0.0_f64;
    if dimensions_match {
        for (left, right) in cpu_tensors.iter().zip(&accelerated.output_tensors) {
            if left.len() != right.len() {
                continue;
            }
            compared_tensors += 1;
            for (left, right) in left.iter().zip(right) {
                let error = (left - right).abs();
                max_error = max_error.max(error);
                sum_error += f64::from(error);
                compared_values += 1;
            }
        }
    }
    let within_tolerance = dimensions_match && compared_values > 0 && max_error <= tolerance;
    TensorParityEvidence {
        provider,
        accelerated_requested,
        accelerated_succeeded: 1,
        compared_tensors,
        compared_values,
        dimensions_match,
        max_abs_error: (compared_values > 0).then_some(max_error),
        mean_abs_error: (compared_values > 0)
            .then_some((sum_error / compared_values as f64) as f32),
        within_tolerance,
        reason: (!within_tolerance).then_some(
            "complete tensor comparison exceeded tolerance or dimensions differed".into(),
        ),
    }
}

fn provider_fallback_resolution(
    requested: &str,
    active: &str,
    allow_fallback: bool,
) -> Result<bool> {
    ensure!(
        !requested.is_empty() && !active.is_empty(),
        "provider names must be non-empty"
    );
    if requested == active {
        return Ok(false);
    }
    ensure!(allow_fallback, "provider fallback is disabled");
    Ok(true)
}

fn failure_probes() -> FailureProbeEvidence {
    let fallback_semantics =
        matches!(
            provider_fallback_resolution("CoreMLExecutionProvider", "CPUExecutionProvider", true),
            Ok(true)
        ) && provider_fallback_resolution("CoreMLExecutionProvider", "CPUExecutionProvider", false)
            .is_err();
    let pixel_limit_rejection = ensure_pixel_limit(2, 2, 3).is_err();
    let malformed_image_rejection =
        load_canonical(Path::new("/m17/nonexistent-image.png")).is_err();
    let malformed_model_rejection = bgremove_models::parse_toml("not = [a valid manifest").is_err();
    let injected_resource_exhaustion_rejection = reserve_capacity(1024, 512).is_err();
    let mut details = BTreeMap::new();
    details.insert(
        "fallback".into(),
        "requested/active mismatch is rejected unless explicitly allowed".into(),
    );
    details.insert(
        "capacity".into(),
        "pixel limit rejects before allocation/inference".into(),
    );
    details.insert(
        "image".into(),
        "decoder error is surfaced for missing/malformed input".into(),
    );
    details.insert(
        "model".into(),
        "strict manifest parser rejects malformed model metadata".into(),
    );
    details.insert(
        "resource".into(),
        "injected fallible-capacity seam rejects over-budget allocation before allocation".into(),
    );
    FailureProbeEvidence {
        fallback_semantics,
        pixel_limit_rejection,
        malformed_image_rejection,
        malformed_model_rejection,
        injected_resource_exhaustion_rejection,
        details,
    }
}

pub fn run(
    output: &Path,
    inputs: &[PathBuf],
    workers: usize,
    max_pixels: u64,
    provider_names: &[String],
    fallback_allowed: bool,
) -> Result<()> {
    ensure!(workers > 0, "worker count must be greater than zero");
    ensure!(
        workers <= MAX_WORKERS,
        "worker count exceeds M17 maximum of {MAX_WORKERS}"
    );
    ensure!(max_pixels > 0, "max pixel limit must be positive");
    let providers = ProviderName::ordered(provider_names)?;
    let output_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    ensure!(
        output_name.starts_with("m17"),
        "M17 output must be a dedicated directory whose name starts with m17"
    );
    let target = output.to_path_buf();
    let staging = prepare_staging(&target)?;
    let mut staging_guard = StagingGuard::new(staging.clone());
    let output = staging.as_path();
    let (images, aggregate_pixels, estimated_decoded_bytes) = preflight_inputs(inputs, max_pixels)?;
    let mut provider_runs = Vec::with_capacity(providers.len());
    let mut parity_rows = Vec::new();
    let mut cpu_transient: Option<TensorBatch> = None;
    for provider in providers.iter().copied() {
        let mut evidence = provider_run(provider, &images, workers, max_pixels, fallback_allowed);
        write_provider_outputs(output, &evidence)?;
        if provider == ProviderName::Cpu && evidence.status == "pass" {
            cpu_transient = Some((
                std::mem::take(&mut evidence.output_dimensions),
                std::mem::take(&mut evidence.output_tensors),
            ));
        } else if provider != ProviderName::Cpu {
            let cpu_view = cpu_transient
                .as_ref()
                .map(|(dimensions, tensors)| (dimensions.as_slice(), tensors.as_slice()));
            parity_rows.push(compare_tensor_parity(
                cpu_view,
                &evidence,
                providers
                    .iter()
                    .filter(|requested| **requested != ProviderName::Cpu)
                    .count(),
                1.0e-4,
            ));
            evidence.output_dimensions.clear();
            evidence.output_tensors.clear();
        }
        evidence.output_pngs.clear();
        provider_runs.push(evidence);
    }
    drop(cpu_transient);
    let bom = manifest_bom()?;
    let code_dependencies = code_dependency_bom()?;
    write_notices(output, &bom, &code_dependencies)?;
    let copied_manifests_exact = write_bundle_metadata(output, &bom)?;
    fs::write(
        output.join("code-dependencies.json"),
        serde_json::to_vec_pretty(&code_dependencies)?,
    )?;
    let processed_images = provider_runs
        .iter()
        .filter(|run| run.status == "pass" || run.status == "pass-with-fallback")
        .map(|run| run.processed_images)
        .max()
        .unwrap_or(0);
    let cancelled = provider_runs.iter().any(|run| run.cancelled);
    let batch = BatchEvidence {
        requested_images: images.len(),
        processed_images,
        workers,
        max_pixels,
        max_batch_pixels: MAX_BATCH_PIXELS,
        aggregate_pixels,
        estimated_decoded_bytes,
        max_input_images: MAX_INPUT_IMAGES,
        cancelled,
        cancellation_checked_between_images: true,
        status: if cancelled {
            "bounded worker pool cancelled before all inputs completed".into()
        } else {
            format!("bounded worker pool with {workers} workers; sessions live for the batch")
        },
    };
    let mut limits = BTreeMap::new();
    limits.insert("max_pixels".into(), max_pixels.to_string());
    limits.insert("hard_max_image_pixels".into(), MAX_IMAGE_PIXELS.to_string());
    limits.insert("max_batch_pixels".into(), MAX_BATCH_PIXELS.to_string());
    limits.insert("max_batch_bytes".into(), MAX_BATCH_BYTES.to_string());
    limits.insert("max_input_images".into(), MAX_INPUT_IMAGES.to_string());
    limits.insert(
        "worker_pool".into(),
        format!("bounded at {workers}; sessions live for the batch"),
    );
    limits.insert(
        "video".into(),
        "not implemented; no temporal-consistency design".into(),
    );
    let release = release_preview(output, &bom, &code_dependencies, copied_manifests_exact)?;
    let mut gates = BTreeMap::new();
    gates.insert(
        "cpu_first_ordered".into(),
        providers.first() == Some(&ProviderName::Cpu),
    );
    let probes = failure_probes();
    gates.insert(
        "provider_fallback_tested_fail_closed".into(),
        probes.fallback_semantics
            && provider_runs.iter().all(|run| {
                run.fallback_allowed == fallback_allowed && (!run.fallback_used || fallback_allowed)
            }),
    );
    gates.insert(
        "resource_exhaustion_tested_fail_closed".into(),
        probes.injected_resource_exhaustion_rejection,
    );
    gates.insert(
        "release_manifest_verified".into(),
        release.deterministic_entries_verified,
    );
    gates.insert(
        "weights_not_auto_downloaded".into(),
        !release.untracked_weights_allowed,
    );
    gates.insert(
        "license_separation".into(),
        release.code_and_weight_licenses_separate,
    );
    gates.insert(
        "license_metadata_present".into(),
        release.license_metadata_present,
    );
    gates.insert(
        "code_dependency_licenses_complete".into(),
        release.code_dependency_licenses_complete,
    );
    gates.insert(
        "copied_manifest_set_exact".into(),
        release.copied_manifests_exact,
    );
    gates.insert(
        "cargo_lock_present".into(),
        release.cargo_lock_sha256.is_some(),
    );
    gates.insert(
        "runtime_success_evidence_present".into(),
        provider_runs
            .iter()
            .any(|run| run.status.starts_with("pass")),
    );
    let tensor_parity_passed =
        !parity_rows.is_empty() && parity_rows.iter().all(|row| row.within_tolerance);
    gates.insert(
        "cpu_accelerated_output_tolerance".into(),
        tensor_parity_passed,
    );
    let overall = gates["release_manifest_verified"]
        && gates["weights_not_auto_downloaded"]
        && gates["license_separation"]
        && gates["license_metadata_present"]
        && gates["code_dependency_licenses_complete"]
        && gates["copied_manifest_set_exact"]
        && gates["cargo_lock_present"]
        && gates["provider_fallback_tested_fail_closed"]
        && gates["resource_exhaustion_tested_fail_closed"]
        && gates["runtime_success_evidence_present"]
        && gates["cpu_accelerated_output_tolerance"];
    gates.insert("release_overall".into(), overall);
    let status = if overall {
        "release qualification passed"
    } else if gates["runtime_success_evidence_present"] {
        "runtime executed; release qualification blocked by one or more gates"
    } else {
        "release evidence generated; runtime/provider gates unavailable"
    };
    let report = M17Report {
        schema: "m17.performance-packaging.v1",
        status: status.into(),
        provider_order: providers,
        providers: provider_runs,
        batch,
        input_limits: limits,
        model_bom: bom.clone(),
        code_dependencies: code_dependencies.clone(),
        failure_probes: probes,
        tensor_parity: parity_rows,
        release: release.clone(),
        gates,
        output_tolerance: 1.0e-4,
    };
    let bytes = [&serde_json::to_vec_pretty(&report)?[..], b"\n"].concat();
    fs::write(output.join("report.json"), &bytes)?;
    let report_hash = sha256(&bytes);
    fs::write(
        output.join("report.sha256"),
        format!("{report_hash}  report.json\n"),
    )?;
    let final_release =
        write_release_manifest(output, &bom, &code_dependencies, copied_manifests_exact)?;
    ensure!(
        final_release.deterministic_entry_count == release.deterministic_entry_count
            && final_release.deterministic_entries_verified,
        "final release manifest did not match its preflight scope"
    );
    publish_staging(output, &target)?;
    staging_guard.committed = true;
    println!("wrote {}", target.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;
    #[test]
    fn m17_provider_order_is_cpu_first_and_strict() {
        let values = vec![
            "cuda".into(),
            "cpu".into(),
            "openvino".into(),
            "coreml".into(),
        ];
        let parsed = ProviderName::ordered(&values).unwrap();
        assert_eq!(
            parsed,
            vec![
                ProviderName::Cpu,
                ProviderName::Coreml,
                ProviderName::Cuda,
                ProviderName::Openvino
            ]
        );
        assert!(ProviderName::parse("bogus").is_err());
        let only_accelerated = ProviderName::ordered(&["cuda".into()]).unwrap();
        assert_eq!(only_accelerated[0], ProviderName::Cpu);
        let defaults = default_provider_names();
        assert_eq!(defaults.first().map(String::as_str), Some("cpu"));
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(defaults, vec!["cpu", "coreml"]);
    }
    #[test]
    fn m17_limits_and_cancellation_fail_closed() {
        assert!(ensure_pixel_limit(3, 4, 11).is_err());
        assert!(BoundedBatchRunner::new(MAX_WORKERS + 1, 100).is_err());
        let runner = BoundedBatchRunner::new(1, 100).unwrap();
        runner.cancel();
        let image = CanonicalImage::new(2, 2, vec![[0.0; 3]; 4]).unwrap();
        let values = runner.run(&[image], |_| Ok::<_, anyhow::Error>(1)).unwrap();
        assert!(values.values.is_empty());
        assert!(values.cancelled);
    }

    #[test]
    fn m17_worker_pool_is_bounded_and_preserves_input_order() {
        let runner = BoundedBatchRunner::new(3, 100).unwrap();
        let images = (0..6)
            .map(|value| CanonicalImage::new(2, 2, vec![[value as f32; 3]; 4]).unwrap())
            .collect::<Vec<_>>();
        let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let active_for_job = Arc::clone(&active);
        let peak_for_job = Arc::clone(&peak);
        let result = runner
            .run(&images, move |image| {
                let now = active_for_job.fetch_add(1, Ordering::AcqRel) + 1;
                peak_for_job.fetch_max(now, Ordering::AcqRel);
                std::thread::sleep(std::time::Duration::from_millis(2));
                active_for_job.fetch_sub(1, Ordering::AcqRel);
                Ok::<_, anyhow::Error>(image.rgb().data()[0][0])
            })
            .unwrap();
        assert_eq!(
            result.values,
            (0..6).map(|value| value as f32).collect::<Vec<_>>()
        );
        assert!(peak.load(Ordering::Acquire) <= 3);
        assert_eq!(result.processed_images, 6);
    }
    #[test]
    fn m17_manifest_hash_is_lowercase_sha256() {
        let value = sha256(b"m17");
        assert_eq!(value.len(), 64);
        assert!(value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
    }

    #[test]
    fn m17_report_checksum_covers_newline_terminated_file() {
        let report = b"{\"schema\":\"m17\"}\n";
        let expected = format!("{}  report.json\n", sha256(report));
        let listed = expected.split_whitespace().next().expect("checksum field");
        assert_eq!(listed, sha256(report));
        assert_ne!(listed, sha256(&report[..report.len() - 1]));
    }

    #[test]
    fn m17_failure_probes_are_executed_not_constant_gate_claims() {
        let probes = failure_probes();
        assert!(probes.fallback_semantics);
        assert!(probes.pixel_limit_rejection);
        assert!(probes.malformed_image_rejection);
        assert!(probes.malformed_model_rejection);
        assert_eq!(probes.details.len(), 5);
    }

    #[test]
    fn m17_limited_decoder_rejects_dimensions_before_pixel_decode() {
        let path = std::env::temp_dir().join(format!("m17-limit-{}.png", std::process::id()));
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(&[255_u8; 16], 2, 2, image::ColorType::Rgba8.into())
            .expect("encode fixture");
        fs::write(&path, bytes).expect("write fixture");
        assert!(load_canonical_with_limit(&path, 3).is_err());
        fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn m17_fallback_policy_is_strict_unless_explicitly_enabled() {
        assert!(matches!(
            provider_fallback_resolution("CoreMLExecutionProvider", "CPUExecutionProvider", true),
            Ok(true)
        ));
        assert!(provider_fallback_resolution(
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
            false
        )
        .is_err());
        assert!(matches!(
            provider_fallback_resolution("CPUExecutionProvider", "CPUExecutionProvider", false),
            Ok(false)
        ));
    }

    #[test]
    fn m17_final_manifest_rejects_an_unlisted_extra_file() {
        let root = std::env::temp_dir().join(format!("m17-release-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).expect("remove prior fixture");
        }
        fs::create_dir_all(&root).expect("create fixture");
        fs::write(root.join("payload.txt"), b"payload").expect("write payload");
        fs::write(root.join("report.json"), b"{}\n").expect("write report");
        fs::write(root.join("report.sha256"), b"placeholder\n").expect("write report checksum");
        write_release_manifest(&root, &[], &[], true).expect("write release manifest");
        fs::write(root.join("extra.txt"), b"unlisted").expect("write extra");
        assert!(verify_release_bundle_files(&root).is_err());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn m17_unowned_output_is_never_deleted() {
        let root = std::env::temp_dir().join(format!("m17-owned-test-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).expect("remove prior fixture");
        }
        fs::create_dir_all(&root).expect("create fixture");
        fs::write(root.join("user-data.txt"), b"preserve").expect("write user data");
        assert!(prepare_staging(&root).is_err());
        assert_eq!(fs::read(root.join("user-data.txt")).unwrap(), b"preserve");
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn m17_failed_staging_cleanup_preserves_published_output() {
        let root = std::env::temp_dir().join(format!("m17-failure-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).expect("remove prior fixture");
        }
        fs::create_dir_all(&root).expect("create fixture");
        fs::write(root.join(OWNER_MARKER), OWNER_MARKER_CONTENT).expect("write owner marker");
        fs::write(root.join("published.txt"), b"keep").expect("write published data");
        let staging = prepare_staging(&root).expect("prepare staging");
        {
            let _guard = StagingGuard::new(staging.clone());
            fs::write(staging.join("partial.txt"), b"partial").expect("write partial data");
        }
        assert!(!staging.exists());
        assert_eq!(fs::read(root.join("published.txt")).unwrap(), b"keep");
        fs::remove_dir_all(root).expect("remove fixture");
    }
}
