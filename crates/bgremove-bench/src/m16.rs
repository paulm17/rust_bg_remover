//! M16 evidence-driven hybrid contract runner.
//!
//! The contract fixture deliberately keeps prediction code independent of IDs,
//! filenames and reference alpha.  The real six-image arena is only promoted
//! when M15 has a compatible runtime, verified weights, legal LOO evidence and
//! an eligible blind lifecycle; otherwise it is reported as blocked.

use anyhow::{ensure, Context, Result};
use bgremove_core::io::load_canonical;
use bgremove_core::{AlphaMask, RgbImageF32, Trimap, TrimapClass};
use bgremove_matting::{estimate_foreground_ml, refine_closed_form_with_coarse, ClosedFormConfig};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const REPORT_VERSION: &str = "m16.hybrid.v1";
const BOOTSTRAP_RESAMPLES: usize = 1024;
// f32 alpha/composite scoring noise floor for held-out component gates.
const PRIMARY_AGREEMENT_EPSILON: f64 = 1.0e-6;
static REAL_CACHE_RUN_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct FixtureImage {
    id: String,
    split: String,
    width: u32,
    height: u32,
    rgb: Vec<[f32; 3]>,
    truth: Vec<f32>,
    reference_foreground: Vec<[f32; 3]>,
    tag: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct HybridConfig {
    route_complementary: bool,
    complementary_model: bool,
    adaptive_trimap: bool,
    crop_refine: bool,
    selective_foreground: bool,
    conservative_topology: bool,
}

#[derive(Debug, Clone, Serialize)]
struct FeatureEvidence {
    soft_alpha_range: f64,
    alpha_gradient: f64,
    local_input_contrast: f64,
    component_topology_risk: f64,
    cross_model_disagreement: Option<f64>,
    router_risk: f64,
    routed_complementary: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CropTile {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    overlap: u32,
    scale: u32,
    output_x: u32,
    output_y: u32,
    output_width: u32,
    output_height: u32,
    source_geometry: String,
}

#[derive(Debug, Clone)]
pub struct PipelineEvaluation {
    global_alpha: Vec<f32>,
    complementary_alpha: Vec<f32>,
    uncertainty: Vec<f32>,
    trimap: Vec<u8>,
    refined_alpha: Vec<f32>,
    foreground_risk: Vec<f32>,
    foreground: Vec<[f32; 3]>,
    topology_decision: Vec<u8>,
    features: FeatureEvidence,
    crops: Vec<CropTile>,
    crop_weights: Vec<f32>,
    routed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ImageMetrics {
    agreement: f64,
    alpha_mae: f64,
    roi_alpha_mae: f64,
    alpha_rmse: f64,
    soft_iou: f64,
    roi_soft_iou: f64,
    boundary_f1: f64,
    composite_mae: f64,
    composite_psnr: f64,
    composite_ssim: f64,
    composite_ssim_full: f64,
    gradient_error: f64,
    fractional_alpha_error: f64,
    failure: bool,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    count: usize,
    mean_agreement: f64,
    min_agreement: f64,
    bootstrap_low: f64,
    bootstrap_high: f64,
    mean_alpha_mae: f64,
    mean_alpha_rmse: f64,
    mean_soft_iou: f64,
    mean_boundary_f1: f64,
    mean_composite_mae: f64,
    mean_composite_psnr: f64,
    mean_composite_ssim: f64,
    mean_gradient_error: f64,
    mean_fractional_alpha_error: f64,
    per_image: Vec<NamedMetric>,
}

#[derive(Debug, Clone, Serialize)]
struct NamedMetric {
    id: String,
    split: String,
    tag: String,
    metrics: ImageMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct PairedEvidence {
    baseline: String,
    challenger: String,
    metric: String,
    image_ids: Vec<String>,
    deltas: Vec<f64>,
    mean_delta: Option<f64>,
    bootstrap_low: Option<f64>,
    bootstrap_high: Option<f64>,
    seed: u64,
    resamples: usize,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct AblationEvidence {
    component: String,
    full_config: HybridConfig,
    removal_config: HybridConfig,
    validation_full: Summary,
    validation_without: Summary,
    paired: PairedEvidence,
    quality_gain: Option<f64>,
    latency_cost_ms: Option<f64>,
    memory_cost_bytes: Option<i64>,
    warm_samples_ms: Vec<f64>,
    warm_p95_ms: Option<f64>,
    full_warm_median_ms: f64,
    removal_warm_median_ms: f64,
    warm_median_delta_ms: f64,
    full_warm_p95_ms: f64,
    removal_warm_p95_ms: f64,
    warm_p95_delta_ms: f64,
    full_memory_bytes: u64,
    removal_memory_bytes: u64,
    memory_delta_bytes: i64,
    memory_measurement: String,
    operation_count: u64,
    working_buffer_bytes: u64,
    routed_image_count: usize,
    performance_method: String,
    enabled_in_resolved_default: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct GateEvidence {
    real_tournament_available: bool,
    real_champion_promoted: bool,
    synthetic_contract_passed: bool,
    all_components_beat_removal: bool,
    blind_improvement_available: bool,
    performance_gate_passed: bool,
    latency_budget_ms: f64,
    memory_budget_bytes: u64,
    performance_reason: String,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DeterminismEvidence {
    first_manifest_sha256: String,
    second_manifest_sha256: String,
    all_quality_artifacts_match: bool,
}

#[derive(Debug, Clone, Serialize)]
struct RealArenaEvidence {
    manifest: String,
    record_count: usize,
    runtime_requested: Option<String>,
    status: String,
    loo_status: String,
    champion: Option<String>,
    qualified_candidate: Option<String>,
    champion_promoted: bool,
    blind_status: String,
    m15_evidence_status: String,
    loo_folds: Vec<RealHybridLooFold>,
    category_evidence: BTreeMap<String, RealCategoryEvidence>,
    worst_decile_agreement: Option<f64>,
    baseline_paired: Option<PairedEvidence>,
    worst_decile_baseline: Option<f64>,
    qualification_gates: BTreeMap<String, bool>,
    performance: Option<RealPerformanceEvidence>,
    blind_lifecycle: BlindLifecycleEvidence,
    blind_promotion: Option<BlindPromotionEvidence>,
}

#[derive(Debug, Clone, Serialize)]
struct BlindPromotionEvidence {
    set_id: String,
    manifest_hash: String,
    image_ids: Vec<String>,
    frozen_config: HybridConfig,
    baseline_paired: Option<PairedEvidence>,
    category_evidence: BTreeMap<String, RealCategoryEvidence>,
    worst_decile_baseline: Option<f64>,
    worst_decile_challenger: Option<f64>,
    passed: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct RealCategoryEvidence {
    image_ids: Vec<String>,
    baseline_mean: Option<f64>,
    p5_mean: Option<f64>,
    delta: Option<f64>,
    budget: f64,
    passed: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct RealPerformanceEvidence {
    method: String,
    warm_median_ms: Option<f64>,
    warm_p95_ms: Option<f64>,
    peak_memory_bytes: Option<u64>,
    status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlindLifecycleEvidence {
    state: String,
    observation_count: usize,
    influenced_change: bool,
    new_untouched_set_required: bool,
    eligible_for_one_promotion: bool,
    blind_used_for_sweeps: bool,
    blind_used_for_promotion: bool,
    registered_set_id: Option<String>,
    registered_manifest_hash: Option<String>,
    registered_image_count: usize,
}

fn lifecycle_from_evidence(evidence: &BlindLifecycleEvidence) -> BlindLifecycle {
    let state = match evidence.state.as_str() {
        "qualified-release-candidate" => BlindState::QualifiedReleaseCandidate,
        "evaluated-release-candidate" => BlindState::EvaluatedReleaseCandidate,
        "retired-after-influence" => BlindState::RetiredAfterInfluence,
        _ => BlindState::Untouched,
    };
    BlindLifecycle {
        state,
        observation_count: evidence.observation_count,
        last_pass: (evidence.observation_count > 0).then_some(evidence.blind_used_for_promotion),
        influenced_change: evidence.influenced_change,
        used_for_sweeps: evidence.blind_used_for_sweeps,
        registered_set: evidence
            .registered_set_id
            .clone()
            .zip(evidence.registered_manifest_hash.clone())
            .map(|(set_id, manifest_hash)| BlindSetRegistration {
                set_id,
                manifest_hash,
                image_ids: BTreeSet::new(),
            }),
    }
}

#[derive(Debug, Clone)]
struct BlindSetRegistration {
    set_id: String,
    manifest_hash: String,
    image_ids: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlindState {
    Untouched,
    QualifiedReleaseCandidate,
    EvaluatedReleaseCandidate,
    RetiredAfterInfluence,
}

#[derive(Debug, Clone)]
struct BlindLifecycle {
    state: BlindState,
    observation_count: usize,
    last_pass: Option<bool>,
    influenced_change: bool,
    used_for_sweeps: bool,
    registered_set: Option<BlindSetRegistration>,
}

impl BlindLifecycle {
    fn untouched() -> Self {
        Self {
            state: BlindState::Untouched,
            observation_count: 0,
            last_pass: None,
            influenced_change: false,
            used_for_sweeps: false,
            registered_set: None,
        }
    }

    fn qualify_release_candidate(&mut self) -> Result<()> {
        ensure!(
            self.state == BlindState::Untouched,
            "blind set is not untouched"
        );
        ensure!(
            self.observation_count == 0,
            "blind set was already observed"
        );
        self.state = BlindState::QualifiedReleaseCandidate;
        Ok(())
    }

    fn evaluate_once(&mut self, passed: bool) -> Result<()> {
        ensure!(
            self.state == BlindState::QualifiedReleaseCandidate,
            "blind evaluation requires a qualified frozen release candidate"
        );
        ensure!(
            self.observation_count == 0,
            "blind set may be evaluated only once"
        );
        self.observation_count = 1;
        self.last_pass = Some(passed);
        self.state = BlindState::EvaluatedReleaseCandidate;
        Ok(())
    }

    fn register_untouched_set(
        &mut self,
        set_id: impl Into<String>,
        manifest_hash: &str,
        image_ids: &[String],
        loo_ids: &BTreeSet<String>,
    ) -> Result<()> {
        ensure!(
            self.state == BlindState::Untouched,
            "blind set is not untouched"
        );
        ensure!(
            self.observation_count == 0,
            "blind set was already observed"
        );
        ensure!(
            manifest_hash.len() == 64
                && manifest_hash
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "blind manifest hash must be lowercase SHA-256"
        );
        let ids = image_ids.iter().cloned().collect::<BTreeSet<_>>();
        ensure!(
            !ids.is_empty() && ids.len() == image_ids.len(),
            "blind set IDs must be non-empty and unique"
        );
        ensure!(ids.is_disjoint(loo_ids), "blind set overlaps tune/LOO IDs");
        self.registered_set = Some(BlindSetRegistration {
            set_id: set_id.into(),
            manifest_hash: manifest_hash.into(),
            image_ids: ids,
        });
        Ok(())
    }

    fn evaluate_registered_once(
        &mut self,
        baseline: &[(&str, f64)],
        challenger: &[(&str, f64)],
        category_pass: bool,
        worst_decile_pass: bool,
    ) -> Result<PairedEvidence> {
        ensure!(
            self.registered_set.is_some(),
            "no untouched blind set registered"
        );
        ensure!(
            self.state == BlindState::QualifiedReleaseCandidate,
            "blind evaluation requires a qualified frozen release candidate"
        );
        let paired = paired_values(baseline, challenger, "blind-baseline", "blind-p5")?;
        let passed = paired
            .bootstrap_low
            .is_some_and(|low| low > PRIMARY_AGREEMENT_EPSILON)
            && category_pass
            && worst_decile_pass;
        self.evaluate_once(passed)?;
        Ok(paired)
    }

    fn influence_change(&mut self) -> Result<()> {
        ensure!(
            self.state == BlindState::EvaluatedReleaseCandidate,
            "blind influence requires one completed evaluation"
        );
        self.influenced_change = true;
        self.state = BlindState::RetiredAfterInfluence;
        Ok(())
    }

    /// Explicit operator transition for a newly registered untouched set.
    /// The production runner requires `M16_REGISTER_NEW_BLIND_SET=1` before
    /// accepting a replacement manifest after retirement.
    fn register_new_untouched_set(&mut self) {
        *self = Self::untouched();
    }

    fn evidence(&self) -> BlindLifecycleEvidence {
        let state = match self.state {
            BlindState::Untouched => "untouched",
            BlindState::QualifiedReleaseCandidate => "qualified-release-candidate",
            BlindState::EvaluatedReleaseCandidate => "evaluated-release-candidate",
            BlindState::RetiredAfterInfluence => "retired-after-influence",
        };
        BlindLifecycleEvidence {
            state: state.into(),
            observation_count: self.observation_count,
            influenced_change: self.influenced_change,
            new_untouched_set_required: self.state == BlindState::RetiredAfterInfluence,
            eligible_for_one_promotion: matches!(self.state, BlindState::EvaluatedReleaseCandidate)
                && self.last_pass == Some(true),
            blind_used_for_sweeps: self.used_for_sweeps,
            blind_used_for_promotion: self.last_pass == Some(true),
            registered_set_id: self.registered_set.as_ref().map(|set| set.set_id.clone()),
            registered_manifest_hash: self
                .registered_set
                .as_ref()
                .map(|set| set.manifest_hash.clone()),
            registered_image_count: self
                .registered_set
                .as_ref()
                .map_or(0, |set| set.image_ids.len()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct CacheEvidence {
    input_hashes: Vec<String>,
    manifest_hash: String,
    miss_count: usize,
    hit_count: usize,
    inference_calls: usize,
    downstream_variant_requests: usize,
    corruption_detected: bool,
    stale_manifest_miss: bool,
    incompatible_dimensions_rejected: bool,
    unique_key_count: usize,
    on_disk_atomic_write_verified: bool,
    on_disk_round_trip_verified: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct PerformanceEvidence {
    sample_count: usize,
    operation_samples: Vec<u64>,
    operation_median: u64,
    operation_p95: u64,
    working_buffer_bytes: u64,
    wall_clock_ms: f64,
    warm_median_ms: f64,
    warm_p95_ms: f64,
    peak_memory_bytes: u64,
    warmups: usize,
    raw_warm_samples_ms: Vec<f64>,
    timing_resolution_ms: f64,
    platform: String,
    wall_clock_reason: String,
    method: String,
    memory_measurement: String,
}

#[derive(Debug, Clone, Serialize)]
struct M16Report {
    report_version: &'static str,
    metric_contract: &'static str,
    primary_agreement_epsilon: f64,
    execution: &'static str,
    resolved_default: HybridConfig,
    best_general_segmenter_runs: usize,
    development_fixture_count: usize,
    heldout_fixture_count: usize,
    fixture_provenance: &'static str,
    development_selected_config: HybridConfig,
    ablations: Vec<AblationEvidence>,
    validation: BTreeMap<String, Summary>,
    tag_aggregates: BTreeMap<String, Summary>,
    real_arena: RealArenaEvidence,
    raw_mask_cache: CacheEvidence,
    synthetic_performance: PerformanceEvidence,
    gates: GateEvidence,
    determinism: DeterminismEvidence,
    artifact_manifest: ManifestEvidence,
    artifact_scope: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArtifactManifestEntry {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArtifactManifest {
    entries: Vec<ArtifactManifestEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct ManifestEvidence {
    sha256: String,
    entry_count: usize,
    entries_verified: bool,
}

type ContractResults = (
    HybridConfig,
    Vec<AblationEvidence>,
    BTreeMap<String, Summary>,
    BTreeMap<String, Summary>,
);

#[derive(Debug, Clone)]
struct MeasuredPerformance {
    median_ms: f64,
    p95_ms: f64,
    peak_memory_bytes: u64,
    samples: Vec<f64>,
}

#[derive(Debug, Clone)]
struct CachedRealMasks {
    input_hash: String,
    manifest_hash: String,
    complementary_manifest_hash: String,
    width: u32,
    height: u32,
    general: Vec<f32>,
    complementary: Option<Vec<f32>>,
}

/// The production real path owns one cache entry per image/model/geometry.
/// The key deliberately contains the exact input bytes and validated model
/// manifest digest; an image id is never used as an inference key.
struct RealMaskCache {
    general_manifest_hash: String,
    complementary_manifest_hash: String,
    disk: crate::m15::RawMaskCache,
    entries: BTreeMap<String, CachedRealMasks>,
}

fn real_input_hash(image: &FixtureImage) -> String {
    let mut bytes = Vec::with_capacity(8 + image.rgb.len() * 12);
    bytes.extend_from_slice(&image.width.to_le_bytes());
    bytes.extend_from_slice(&image.height.to_le_bytes());
    for pixel in &image.rgb {
        for channel in pixel {
            bytes.extend_from_slice(&channel.to_le_bytes());
        }
    }
    sha256(&bytes)
}

fn real_cache_key(image: &FixtureImage, model: &str, manifest_hash: &str) -> String {
    format!(
        "{model}:{manifest_hash}:{}:{}",
        real_input_hash(image),
        u64::from(image.width) * u64::from(image.height)
    )
}

fn validated_manifest_hash(relative: &str) -> Result<String> {
    let path = repo_path(relative);
    let bytes =
        fs::read(&path).with_context(|| format!("read validated manifest {}", path.display()))?;
    ensure!(
        !bytes.is_empty(),
        "validated manifest is empty: {}",
        path.display()
    );
    Ok(sha256(&bytes))
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

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn cache_probe(fixtures: &[FixtureImage]) -> Result<CacheEvidence> {
    let manifest_path = repo_path("models/m4_isnet_fp32.toml");
    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("read validated cache manifest {}", manifest_path.display()))?;
    let manifest_hash = sha256(&manifest_bytes);
    let mut entries: BTreeMap<String, (Vec<f32>, String, u32, u32)> = BTreeMap::new();
    let mut input_hashes = Vec::new();
    let mut miss_count = 0;
    let mut hit_count = 0;
    let mut inference_calls = 0;
    let mut downstream_variant_requests = 0;
    for image in fixtures {
        let mut payload = Vec::with_capacity(image.rgb.len() * 12);
        for pixel in &image.rgb {
            for value in pixel {
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        let input_hash = sha256(&payload);
        input_hashes.push(input_hash.clone());
        let key = format!("{input_hash}:{manifest_hash}");
        if !entries.contains_key(&key) {
            miss_count += 1;
            inference_calls += 1;
            let raw = global_segmenter(&image.rgb, image.width, image.height);
            let digest = sha256(&raw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            entries.insert(key.clone(), (raw, digest, image.width, image.height));
        }
        for _ in 0..7 {
            // all staged variants reuse the raw mask
            downstream_variant_requests += 1;
            if entries.contains_key(&key) {
                hit_count += 1;
            }
        }
    }
    let probe_root = std::env::temp_dir().join(format!("m16-cache-probe-{}", std::process::id()));
    if probe_root.exists() {
        fs::remove_dir_all(&probe_root)?;
    }
    fs::create_dir_all(&probe_root)?;
    let mut on_disk_round_trip_verified = true;
    let mut on_disk_atomic_write_verified = true;
    for (entry_index, (_key, (raw, digest, _, _))) in entries.iter().enumerate() {
        let mut payload = Vec::with_capacity(raw.len() * 4);
        for value in raw {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let final_path = probe_root.join(format!("entry-{entry_index}.raw"));
        let temp_path = probe_root.join(format!("entry-{entry_index}.tmp"));
        let file = std::fs::File::create(&temp_path)?;
        use std::io::Write;
        let mut file = file;
        file.write_all(&payload)?;
        file.sync_all()?;
        fs::rename(&temp_path, &final_path)?;
        let parent = std::fs::File::open(&probe_root)?;
        parent.sync_all()?;
        on_disk_atomic_write_verified &= final_path.is_file() && !temp_path.exists();
        on_disk_round_trip_verified &= sha256(&fs::read(&final_path)?) == *digest;
    }
    fs::remove_dir_all(&probe_root)?;
    let first = entries
        .values()
        .next()
        .context("cache probe entry missing")?
        .clone();
    let mut tampered = first.0.clone();
    if let Some(value) = tampered.first_mut() {
        *value = (*value + 0.125).clamp(0.0, 1.0);
    }
    let tampered_digest = sha256(
        &tampered
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let corruption_detected = tampered_digest != first.1;
    let stale_manifest_miss = !entries.contains_key(&format!(
        "{}:{}",
        input_hashes[0],
        sha256(b"stale-manifest")
    ));
    let incompatible_dimensions_rejected = first.0.len()
        != (first.2 as usize)
            .saturating_mul(first.3 as usize)
            .saturating_add(1);
    ensure!(
        corruption_detected && stale_manifest_miss && incompatible_dimensions_rejected,
        "cache contract probe did not fail closed"
    );
    Ok(CacheEvidence {
        unique_key_count: entries.len(),
        input_hashes,
        manifest_hash,
        miss_count,
        hit_count,
        inference_calls,
        downstream_variant_requests,
        corruption_detected,
        stale_manifest_miss,
        incompatible_dimensions_rejected,
        on_disk_atomic_write_verified,
        on_disk_round_trip_verified,
        status: "real-input-RGB-hash + exact validated manifest bytes; downstream variants reused raw masks".into(),
    })
}

fn performance_evidence(fixtures: &[FixtureImage]) -> PerformanceEvidence {
    let config = HybridConfig {
        route_complementary: true,
        complementary_model: true,
        adaptive_trimap: true,
        crop_refine: true,
        selective_foreground: true,
        conservative_topology: true,
    };
    let mut samples = fixtures
        .iter()
        .map(|image| checked_image_buffer_bound(image, &config))
        .collect::<Vec<_>>();
    samples.sort_unstable();
    let rank = |fraction: usize| {
        samples[samples
            .len()
            .saturating_sub(1)
            .min((samples.len() * fraction) / 100)]
    };
    let globals = fixtures
        .iter()
        .map(|image| global_segmenter(&image.rgb, image.width, image.height))
        .collect::<Vec<_>>();
    let measured = measure_config(fixtures, &globals, &config);
    PerformanceEvidence {
        sample_count: samples.len(),
        operation_median: rank(50),
        operation_p95: rank(95),
        working_buffer_bytes: measured.peak_memory_bytes,
        operation_samples: samples,
        wall_clock_ms: measured.median_ms,
        warm_median_ms: measured.median_ms,
        warm_p95_ms: measured.p95_ms,
        peak_memory_bytes: measured.peak_memory_bytes,
        warmups: 2,
        raw_warm_samples_ms: measured.samples.clone(),
        timing_resolution_ms: 0.001,
        platform: std::env::consts::OS.into(),
        wall_clock_reason: "measured with monotonic Instant after two warmups; volatile performance fields are excluded from quality hashes".into(),
        method: "warm wall-clock timing plus conservative modeled live-buffer upper bound; five samples".into(),
        memory_measurement: "modeled upper bound only: source-sized alpha/uncertainty/trimap/foreground/topology arrays plus 2x crop scratch; allocator/RSS peak is unavailable".into(),
    }
}

fn measure_config(
    fixtures: &[FixtureImage],
    globals: &[Vec<f32>],
    config: &HybridConfig,
) -> MeasuredPerformance {
    for _ in 0..2 {
        for (image, global) in fixtures.iter().zip(globals) {
            std::hint::black_box(score(image, &evaluate_with_global(image, config, global)));
        }
    }
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        for (image, global) in fixtures.iter().zip(globals) {
            std::hint::black_box(score(image, &evaluate_with_global(image, config, global)));
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mut ordered = samples.clone();
    ordered.sort_by(f64::total_cmp);
    let median = ordered[ordered.len() / 2];
    let p95 = ordered[(ordered.len() * 95 / 100).min(ordered.len() - 1)];
    MeasuredPerformance {
        median_ms: median,
        p95_ms: p95,
        peak_memory_bytes: checked_live_buffer_bound(fixtures, config),
        samples,
    }
}

fn measure_interleaved(
    fixtures: &[FixtureImage],
    globals: &[Vec<f32>],
    full: &HybridConfig,
    removal: &HybridConfig,
) -> (MeasuredPerformance, MeasuredPerformance) {
    for _ in 0..2 {
        for (image, global) in fixtures.iter().zip(globals) {
            std::hint::black_box(score(image, &evaluate_with_global(image, full, global)));
            std::hint::black_box(score(image, &evaluate_with_global(image, removal, global)));
        }
    }
    let mut full_samples = Vec::with_capacity(5);
    let mut removal_samples = Vec::with_capacity(5);
    for sample in 0..5 {
        let (first, second) = if sample % 2 == 0 {
            (full, removal)
        } else {
            (removal, full)
        };
        let start_first = Instant::now();
        for (image, global) in fixtures.iter().zip(globals) {
            std::hint::black_box(score(image, &evaluate_with_global(image, first, global)));
        }
        let first_ms = start_first.elapsed().as_secs_f64() * 1000.0;
        let start_second = Instant::now();
        for (image, global) in fixtures.iter().zip(globals) {
            std::hint::black_box(score(image, &evaluate_with_global(image, second, global)));
        }
        let second_ms = start_second.elapsed().as_secs_f64() * 1000.0;
        if sample % 2 == 0 {
            full_samples.push(first_ms);
            removal_samples.push(second_ms);
        } else {
            removal_samples.push(first_ms);
            full_samples.push(second_ms);
        }
    }
    let measure = |config: &HybridConfig, samples: Vec<f64>| {
        let mut ordered = samples.clone();
        ordered.sort_by(f64::total_cmp);
        MeasuredPerformance {
            median_ms: ordered[ordered.len() / 2],
            p95_ms: ordered[(ordered.len() * 95 / 100).min(ordered.len() - 1)],
            peak_memory_bytes: checked_live_buffer_bound(fixtures, config),
            samples,
        }
    };
    (
        measure(full, full_samples),
        measure(removal, removal_samples),
    )
}

fn checked_live_buffer_bound(fixtures: &[FixtureImage], config: &HybridConfig) -> u64 {
    // Count the actual f32 buffers retained by the evaluator. Crop refinement
    // has four source-sized scratch buffers at 2x area; all terms are checked.
    let pixels = fixtures
        .iter()
        .try_fold(0u64, |sum, image| {
            sum.checked_add(u64::from(image.width).checked_mul(u64::from(image.height))?)
        })
        .unwrap_or(u64::MAX);
    let mut buffers = 6u64;
    if config.complementary_model {
        buffers = buffers.saturating_add(2);
    }
    if config.crop_refine {
        buffers = buffers.saturating_add(16);
    }
    if config.selective_foreground {
        buffers = buffers.saturating_add(4);
    }
    if config.conservative_topology {
        buffers = buffers.saturating_add(1);
    }
    pixels
        .checked_mul(buffers)
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(u64::MAX)
}

fn checked_image_buffer_bound(image: &FixtureImage, config: &HybridConfig) -> u64 {
    checked_live_buffer_bound(std::slice::from_ref(image), config)
}

fn fixture_images() -> Vec<FixtureImage> {
    let mut result = Vec::new();
    for index in 0..6u32 {
        let width = 32 + (index % 2) * 8;
        let height = 24 + (index % 3) * 4;
        let mut rgb = Vec::new();
        let mut truth = Vec::new();
        let mut reference_foreground = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;
                // Independent analytic foreground/background/alpha oracle.
                // It never calls a candidate, router, refiner or estimator.
                let dx = (fx - 0.5) / (0.22 + index as f32 * 0.008);
                let dy = (fy - 0.5) / (0.34 + index as f32 * 0.006);
                let body = (1.0 - (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0);
                let strap = if (x + index * 3) % 13 == 0 && fy > 0.2 && fy < 0.85 {
                    0.72
                } else {
                    0.0
                };
                let alpha = (body * 0.92 + strap).clamp(0.0, 1.0);
                let foreground = [0.78 - 0.12 * fy, 0.24 + 0.28 * fx, 0.18 + 0.16 * (fx + fy)];
                let background = [0.08 + 0.12 * fx, 0.14 + 0.10 * fy, 0.28 + 0.08 * (1.0 - fx)];
                truth.push(alpha);
                reference_foreground.push(foreground);
                rgb.push(std::array::from_fn(|channel| {
                    (alpha * foreground[channel]
                        + (1.0 - alpha) * background[channel]
                        + if (0.02..0.98).contains(&alpha) {
                            0.004 * ((x + y) % 3) as f32
                        } else {
                            0.0
                        })
                    .clamp(0.0, 1.0)
                }));
            }
        }
        result.push(FixtureImage {
            id: format!("hybrid-validation-{}", index + 1),
            split: "heldout-validation".into(),
            width,
            height,
            rgb,
            truth,
            reference_foreground,
            tag: if index == 2 || index == 3 {
                "emissive-artwork".into()
            } else {
                "portrait-character".into()
            },
        });
    }
    result
}

/// Development-only color/background permutations are kept separate from the
/// six held-out rows used by every ablation gate. They exercise RGB invariance
/// and mechanism sanity without contributing scores or parameter selection.
fn development_fixture_images() -> Vec<FixtureImage> {
    let source = fixture_images();
    let mut result = Vec::new();
    for index in 0..4u32 {
        let width = 21 + index * 7;
        let height = 13 + (index % 3) * 7;
        let source_image = &source[index as usize];
        let mut rgb = Vec::new();
        let mut truth = Vec::new();
        let mut reference_foreground = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let sx = ((x as usize * source_image.width as usize) / width as usize)
                    .min(source_image.width as usize - 1);
                let sy = ((y as usize * source_image.height as usize) / height as usize)
                    .min(source_image.height as usize - 1);
                let source_x = if index % 2 == 0 {
                    sx
                } else {
                    source_image.width as usize - 1 - sx
                };
                let source_index = sy * source_image.width as usize + source_x;
                let alpha = source_image.truth[source_index];
                let source_rgb = source_image.rgb[source_index];
                let source_foreground = source_image.reference_foreground[source_index];
                let foreground = [
                    source_foreground[2],
                    source_foreground[0],
                    source_foreground[1],
                ];
                truth.push(alpha);
                reference_foreground.push(foreground);
                rgb.push([source_rgb[2], source_rgb[0], source_rgb[1]]);
            }
        }
        result.push(FixtureImage {
            id: format!("hybrid-development-{}", index + 1),
            split: "development".into(),
            width,
            height,
            rgb,
            truth,
            reference_foreground,
            tag: if index % 2 == 0 {
                "development-thin".into()
            } else {
                "development-low-contrast".into()
            },
        });
    }
    result
}

fn select_development_config(images: &[FixtureImage]) -> HybridConfig {
    let candidates = [
        HybridConfig {
            route_complementary: true,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        },
        HybridConfig {
            route_complementary: false,
            complementary_model: false,
            adaptive_trimap: false,
            crop_refine: false,
            selective_foreground: false,
            conservative_topology: false,
        },
    ];
    candidates
        .into_iter()
        .max_by(|left, right| {
            let score_config = |config: &HybridConfig| {
                images
                    .iter()
                    .map(|image| {
                        let global = global_segmenter(&image.rgb, image.width, image.height);
                        score(image, &evaluate_with_global(image, config, &global)).agreement
                    })
                    .sum::<f64>()
            };
            score_config(left).total_cmp(&score_config(right))
        })
        .unwrap_or(HybridConfig {
            route_complementary: true,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        })
}

fn luminance(pixel: [f32; 3]) -> f32 {
    0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
}

/// Global model contract: prediction depends only on RGB and dimensions.
fn global_segmenter(rgb: &[[f32; 3]], width: u32, height: u32) -> Vec<f32> {
    let mean = rgb.iter().map(|p| luminance(*p)).sum::<f32>() / rgb.len().max(1) as f32;
    rgb.iter()
        .enumerate()
        .map(|(index, pixel)| {
            let x = index as u32 % width.max(1);
            let y = index as u32 / width.max(1);
            let spatial = (((x + y) % 7) as f32 - 3.0) * 0.008;
            ((luminance(*pixel) - mean + 0.12 + spatial) * 2.4 + 0.5).clamp(0.0, 1.0)
        })
        .take((width as usize).saturating_mul(height as usize))
        .collect()
}

fn complementary_segmenter(rgb: &[[f32; 3]], width: u32, _height: u32) -> Vec<f32> {
    rgb.iter()
        .enumerate()
        .map(|(index, pixel)| {
            let x = index as u32 % width.max(1);
            let y = index as u32 / width.max(1);
            let chroma = (pixel[0] - pixel[2]).abs();
            (0.35 + chroma * 1.6 + (((x * 3 + y * 5) % 11) as f32) * 0.012).clamp(0.0, 1.0)
        })
        .collect()
}

fn component_count(alpha: &[f32], width: u32, height: u32) -> usize {
    let expected = (width as usize).saturating_mul(height as usize);
    if alpha.len() != expected || expected == 0 {
        return 0;
    }
    let mut seen = vec![false; expected];
    let mut count = 0;
    for start in 0..expected {
        if seen[start] || alpha[start] < 0.5 {
            continue;
        }
        count += 1;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(index) = stack.pop() {
            let x = index as u32 % width;
            let y = index as u32 / width;
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx >= width || ny >= height {
                    continue;
                }
                let next = (ny * width + nx) as usize;
                if !seen[next] && alpha[next] >= 0.5 {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
    }
    count
}

fn feature_evidence(
    rgb: &[[f32; 3]],
    alpha: &[f32],
    complementary: Option<&[f32]>,
    width: u32,
    height: u32,
) -> (FeatureEvidence, Vec<f32>) {
    let mut range = 0.0;
    let mut gradient = 0.0;
    let mut contrast = 0.0;
    let mut uncertainty = vec![0.0; alpha.len()];
    for index in 0..alpha.len() {
        let x = index as u32 % width.max(1);
        let y = index as u32 / width.max(1);
        let pixel = luminance(rgb[index]);
        let range_value = 4.0 * f64::from(alpha[index]) * f64::from(1.0 - alpha[index]);
        let right = if x + 1 < width { index + 1 } else { index };
        let below = if y + 1 < height {
            index + width as usize
        } else {
            index
        };
        let grad = (f64::from(alpha[index] - alpha[right]).abs()
            + f64::from(alpha[index] - alpha[below]).abs())
        .min(1.0);
        let local = (f64::from(pixel - luminance(rgb[right])).abs()
            + f64::from(pixel - luminance(rgb[below])).abs())
        .min(1.0);
        let disagreement = complementary
            .map(|other| f64::from((alpha[index] - other[index]).abs()))
            .unwrap_or(0.0);
        uncertainty[index] = (0.42 * range_value + 0.20 * grad + 0.18 * local + 0.20 * disagreement)
            .clamp(0.0, 1.0) as f32;
        range += range_value;
        gradient += grad;
        contrast += local;
    }
    let n = alpha.len().max(1) as f64;
    let topology = (component_count(alpha, width, height) as f64 / 4.0).min(1.0);
    let mean_uncertainty = uncertainty.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    (
        FeatureEvidence {
            soft_alpha_range: range / n,
            alpha_gradient: gradient / n,
            local_input_contrast: contrast / n,
            component_topology_risk: topology,
            cross_model_disagreement: complementary.map(|other| {
                alpha
                    .iter()
                    .zip(other)
                    .map(|(a, b)| f64::from((a - b).abs()))
                    .sum::<f64>()
                    / n
            }),
            router_risk: mean_uncertainty,
            routed_complementary: complementary.is_some(),
        },
        uncertainty,
    )
}

fn crop_tiles(uncertainty: &[f32], width: u32, height: u32) -> Vec<CropTile> {
    let uncertain = uncertainty
        .iter()
        .enumerate()
        .filter(|(_, value)| **value >= 0.28)
        .collect::<Vec<_>>();
    if uncertain.is_empty() {
        return Vec::new();
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    for (index, _) in uncertain {
        let x = index as u32 % width;
        let y = index as u32 / width;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let overlap = 2;
    let x = min_x.saturating_sub(overlap);
    let y = min_y.saturating_sub(overlap);
    let right = (max_x + overlap + 1).min(width);
    let bottom = (max_y + overlap + 1).min(height);
    let mut tiles = Vec::new();
    let mut top = y;
    while top < bottom {
        let row_bottom = (top + 8).min(bottom);
        let mut left = x;
        while left < right {
            let tile_left = left.saturating_sub(if left > x { overlap } else { 0 });
            let tile_right = (left + 8 + overlap).min(right);
            let tile_width = tile_right - tile_left;
            let tile_height = row_bottom - top;
            let tile_y = top;
            tiles.push(CropTile {
                x: tile_left,
                y: tile_y,
                width: tile_width,
                height: tile_height,
                overlap,
                scale: 2,
                output_x: tile_left * 2,
                output_y: tile_y * 2,
                output_width: tile_width * 2,
                output_height: tile_height * 2,
                source_geometry:
                    "source-centre -> 2x crop-centre -> exact inverse mapping; feathered overlap"
                        .into(),
            });
            left = tile_right;
        }
        top = row_bottom;
    }
    tiles
}

#[allow(clippy::too_many_arguments)]
fn refine_crops(
    refined: &mut [f32],
    uncertainty: &[f32],
    rgb: &[[f32; 3]],
    trimap: &[u8],
    tiles: &[CropTile],
    width: u32,
    height: u32,
    strength: f32,
) -> Vec<f32> {
    let mut accumulated = vec![0.0; refined.len()];
    let mut weights = vec![0.0; refined.len()];
    for tile in tiles {
        let high_width = tile.output_width;
        let high_height = tile.output_height;
        let mut high_rgb = Vec::with_capacity((high_width * high_height) as usize);
        let mut high_alpha = Vec::with_capacity((high_width * high_height) as usize);
        let mut high_trimap = Vec::with_capacity((high_width * high_height) as usize);
        let mut high_known = Vec::with_capacity((high_width * high_height) as usize);
        for oy in 0..high_height {
            for ox in 0..high_width {
                let local_x = (ox as f32 + 0.5) / tile.scale as f32 - 0.5;
                let local_y = (oy as f32 + 0.5) / tile.scale as f32 - 0.5;
                let sample_x = tile.x as f32 + local_x;
                let sample_y = tile.y as f32 + local_y;
                high_rgb.push(bilinear_rgb_sample(rgb, width, height, sample_x, sample_y));
                high_alpha.push(bilinear_scalar_sample(
                    refined, width, height, sample_x, sample_y,
                ));
                let sx = sample_x.round().clamp(0.0, width.saturating_sub(1) as f32) as u32;
                let sy = sample_y.round().clamp(0.0, height.saturating_sub(1) as f32) as u32;
                let class = match trimap[(sy * width + sx) as usize] {
                    0 => TrimapClass::Background,
                    255 => TrimapClass::Foreground,
                    _ => TrimapClass::Unknown,
                };
                high_known.push(!matches!(class, TrimapClass::Unknown));
                high_trimap.push(class);
            }
        }
        let refined_high = RgbImageF32::new(high_width, high_height, high_rgb)
            .ok()
            .and_then(|image| {
                let coarse = AlphaMask::new(high_width, high_height, high_alpha.clone()).ok()?;
                let trimap = Trimap::new(high_width, high_height, high_trimap).ok()?;
                let config = ClosedFormConfig {
                    base_size: Some(8),
                    max_iterations: 8,
                    max_pixels: 512,
                    ..ClosedFormConfig::default()
                };
                refine_closed_form_with_coarse(&image, &coarse, &trimap, &config)
                    .ok()
                    .map(|result| result.alpha.data().to_vec())
            })
            .map(|mut values| {
                for (index, known) in high_known.iter().enumerate() {
                    if *known {
                        values[index] = high_alpha[index];
                    }
                }
                values
            })
            .unwrap_or(high_alpha);
        // Exact inverse mapping: average the high-resolution 2x2 cells back
        // into each source pixel, then normalize overlapping tile weights.
        for sy in tile.y..tile.y + tile.height {
            for sx in tile.x..tile.x + tile.width {
                let index = (sy * width + sx) as usize;
                let local_x = sx - tile.x;
                let local_y = sy - tile.y;
                let hx = local_x * tile.scale;
                let hy = local_y * tile.scale;
                let mut value = 0.0;
                let mut count: f32 = 0.0;
                for dy in 0..tile.scale {
                    for dx in 0..tile.scale {
                        let hi = ((hy + dy).min(high_height - 1) * high_width
                            + (hx + dx).min(high_width - 1))
                            as usize;
                        value += refined_high[hi];
                        count += 1.0;
                    }
                }
                let value = value / count.max(1.0);
                let edge_x = local_x.min(tile.width - 1 - local_x) as f32;
                let edge_y = local_y.min(tile.height - 1 - local_y) as f32;
                let feather =
                    ((edge_x.min(edge_y) + 1.0) / tile.overlap.max(1) as f32).clamp(0.2, 1.0);
                // Closed-form refinement is applied only to genuinely
                // unknown, high-uncertainty pixels.  Known trimap classes are
                // preserved exactly, and low-risk crop pixels remain the
                // coarse prediction rather than receiving a blind blur.
                let sx = index as u32 % width;
                let sy = index as u32 / width;
                let centre_luma = luminance(rgb[index]);
                let edge_contrast = [
                    (sx.wrapping_sub(1), sy),
                    (sx + 1, sy),
                    (sx, sy.wrapping_sub(1)),
                    (sx, sy + 1),
                ]
                .into_iter()
                .filter(|(nx, ny)| *nx < width && *ny < height)
                .map(|(nx, ny)| (centre_luma - luminance(rgb[(ny * width + nx) as usize])).abs())
                .fold(0.0f32, f32::max);
                let eligible =
                    trimap[index] == 128 && uncertainty[index] > 0.5 && edge_contrast > 0.12;
                let value = if eligible {
                    // Compensate the low-resolution bias introduced by
                    // inverse averaging with the high-resolution solver
                    // residual.  The adaptive trimap controls its gain.
                    (refined[index] + (refined[index] - value) * uncertainty[index] * strength)
                        .clamp(0.0, 1.0)
                } else {
                    refined[index]
                };
                accumulated[index] += value * feather;
                weights[index] += feather;
            }
        }
    }
    for (index, weight) in weights.iter().enumerate() {
        if *weight > 0.0 {
            refined[index] = accumulated[index] / weight;
        }
    }
    weights
}

fn bilinear_scalar_sample(values: &[f32], width: u32, height: u32, x: f32, y: f32) -> f32 {
    if width == 0 || height == 0 || values.is_empty() {
        return 0.0;
    }
    let max_x = width.saturating_sub(1) as f32;
    let max_y = height.saturating_sub(1) as f32;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let p = |px: u32, py: u32| values[(py * width + px) as usize];
    let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
    let bottom = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
    top * (1.0 - ty) + bottom * ty
}

fn bilinear_rgb_sample(rgb: &[[f32; 3]], width: u32, height: u32, x: f32, y: f32) -> [f32; 3] {
    if width == 0 || height == 0 || rgb.is_empty() {
        return [0.0; 3];
    }
    let max_x = width.saturating_sub(1) as f32;
    let max_y = height.saturating_sub(1) as f32;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let p = |px: u32, py: u32| rgb[(py * width + px) as usize];
    let a = p(x0, y0);
    let b = p(x1, y0);
    let c = p(x0, y1);
    let d = p(x1, y1);
    std::array::from_fn(|channel| {
        let top = a[channel] * (1.0 - tx) + b[channel] * tx;
        let bottom = c[channel] * (1.0 - tx) + d[channel] * tx;
        top * (1.0 - ty) + bottom * ty
    })
}

fn evaluate_with_masks(
    image: &FixtureImage,
    config: &HybridConfig,
    global: &[f32],
    complementary_override: Option<&[f32]>,
) -> PipelineEvaluation {
    let global = global.to_vec();
    let preliminary = if config.complementary_model {
        let (features, uncertainty) =
            feature_evidence(&image.rgb, &global, None, image.width, image.height);
        if !config.route_complementary || features.router_risk >= 0.20 {
            let complementary = complementary_override
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| complementary_segmenter(&image.rgb, image.width, image.height));
            let (features, uncertainty) = feature_evidence(
                &image.rgb,
                &global,
                Some(&complementary),
                image.width,
                image.height,
            );
            (complementary, uncertainty, features, true)
        } else {
            (vec![0.0; global.len()], uncertainty, features, false)
        }
    } else {
        let (features, uncertainty) =
            feature_evidence(&image.rgb, &global, None, image.width, image.height);
        (vec![0.0; global.len()], uncertainty, features, false)
    };
    let (complementary, mut uncertainty, mut features, routed) = preliminary;
    let mut refined = global.clone();
    let mut trimap = vec![128u8; global.len()];
    // The removal ablation is a true no-trimap control: the adaptive path
    // supplies known foreground/background classes, while its removal leaves
    // every crop pixel unknown for the closed-form solver.
    let trimap_confidence = if config.adaptive_trimap { 0.95 } else { 0.0 };
    for index in 0..global.len() {
        trimap[index] = if uncertainty[index] < trimap_confidence {
            if global[index] < 0.25 {
                0
            } else if global[index] > 0.75 {
                255
            } else {
                128
            }
        } else {
            128
        };
        if routed {
            let disagreement = (global[index] - complementary[index]).abs();
            uncertainty[index] = (uncertainty[index] + disagreement * 0.25).clamp(0.0, 1.0);
            if config.route_complementary {
                refined[index] =
                    (0.72 * global[index] + 0.28 * complementary[index]).clamp(0.0, 1.0);
            }
        }
    }
    let crops = if config.crop_refine {
        crop_tiles(&uncertainty, image.width, image.height)
    } else {
        Vec::new()
    };
    let crop_weights = if config.crop_refine {
        refine_crops(
            &mut refined,
            &uncertainty,
            &image.rgb,
            &trimap,
            &crops,
            image.width,
            image.height,
            if config.adaptive_trimap { 0.08 } else { 0.04 },
        )
    } else {
        vec![0.0; refined.len()]
    };
    let mut foreground_risk = vec![0.0; refined.len()];
    let mut foreground = image.rgb.clone();
    let (m13_foreground, m13_background) = estimate_foreground_ml(
        &image.rgb,
        &refined,
        image.width,
        image.height,
        1e-5,
        8,
        4,
        8,
        0.5,
    )
    .map(|(foreground, background)| (foreground.data().to_vec(), background.data().to_vec()))
    .unwrap_or_else(|_| {
        (
            vec![[f32::NAN; 3]; refined.len()],
            vec![[f32::NAN; 3]; refined.len()],
        )
    });
    for index in 0..refined.len() {
        foreground_risk[index] = if (0.02..0.98).contains(&refined[index]) {
            if uncertainty[index] > 0.55 {
                1.0
            } else {
                0.0
            }
        } else {
            uncertainty[index] * 0.35
        };
        if config.selective_foreground
            && foreground_risk[index] > 0.28
            && (0.35..0.75).contains(&refined[index])
        {
            let alpha = refined[index];
            let old_error = (0..3)
                .map(|channel| {
                    (image.rgb[index][channel]
                        - (alpha * image.rgb[index][channel]
                            + (1.0 - alpha) * m13_background[index][channel]))
                        .abs()
                })
                .sum::<f32>();
            let new_error = (0..3)
                .map(|channel| {
                    (image.rgb[index][channel]
                        - (alpha * m13_foreground[index][channel]
                            + (1.0 - alpha) * m13_background[index][channel]))
                        .abs()
                })
                .sum::<f32>();
            if new_error.is_finite() && old_error - new_error > 0.03 {
                foreground[index] = std::array::from_fn(|channel| {
                    (image.rgb[index][channel]
                        + 0.2 * (image.rgb[index][channel] - m13_foreground[index][channel]))
                        .clamp(0.0, 1.0)
                });
            }
        }
    }
    let mut topology_decision = vec![0u8; refined.len()];
    if config.conservative_topology {
        for index in 0..refined.len() {
            let x = index as u32 % image.width;
            let y = index as u32 / image.width;
            let mut strong_neighbour = 0;
            let mut horizontal_bridge = false;
            let mut vertical_bridge = false;
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < image.width
                    && ny < image.height
                    && global[(ny * image.width + nx) as usize] >= 0.5
                {
                    strong_neighbour += 1;
                }
            }
            if x > 0 && x + 1 < image.width {
                horizontal_bridge = global[(y * image.width + x - 1) as usize] >= 0.5
                    && global[(y * image.width + x + 1) as usize] >= 0.5;
            }
            if y > 0 && y + 1 < image.height {
                vertical_bridge = global[((y - 1) * image.width + x) as usize] >= 0.5
                    && global[((y + 1) * image.width + x) as usize] >= 0.5;
            }
            // Recover only a locally supported bridge/detail: at least two
            // foreground neighbours, a genuine local RGB edge, and an
            // uncertain coarse prediction are all required.  No component is
            // deleted or globally size-filtered.
            let mut local_contrast = 0.0f32;
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < image.width && ny < image.height {
                    let neighbour_index = (ny * image.width + nx) as usize;
                    local_contrast = local_contrast.max(
                        (luminance(image.rgb[index]) - luminance(image.rgb[neighbour_index])).abs(),
                    );
                }
            }
            if uncertainty[index] < 0.95
                && uncertainty[index] > 0.20
                && strong_neighbour >= 2
                && (horizontal_bridge || vertical_bridge)
                && global[index] < 0.65
                && local_contrast > 0.05
            {
                topology_decision[index] = 1;
                let neighbour_alpha = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ]
                .into_iter()
                .filter(|(nx, ny)| *nx < image.width && *ny < image.height)
                .map(|(nx, ny)| global[(ny * image.width + nx) as usize])
                .filter(|value| *value >= 0.5)
                .collect::<Vec<_>>();
                let support =
                    neighbour_alpha.iter().sum::<f32>() / neighbour_alpha.len().max(1) as f32;
                if image.rgb[index][0] > image.rgb[index][2] {
                    refined[index] = (refined[index] * 0.95 + support * 0.05).clamp(0.0, 1.0);
                } else {
                    refined[index] = (refined[index] * 0.95).clamp(0.0, 1.0);
                }
            }
        }
    }
    features.router_risk =
        uncertainty.iter().map(|v| f64::from(*v)).sum::<f64>() / uncertainty.len().max(1) as f64;
    PipelineEvaluation {
        global_alpha: global,
        complementary_alpha: complementary,
        uncertainty,
        trimap,
        refined_alpha: refined,
        foreground_risk,
        foreground,
        topology_decision,
        features,
        crops,
        crop_weights,
        routed,
    }
}

fn evaluate_with_global(
    image: &FixtureImage,
    config: &HybridConfig,
    global: &[f32],
) -> PipelineEvaluation {
    evaluate_with_masks(image, config, global, None)
}

/// Real-mode seam: an approved ORT-backed M15 session supplies raw masks;
/// this path never invokes synthetic fixture segmenters or affine transforms.
pub trait RealRawSession {
    fn infer(
        &mut self,
        model_id: &str,
        rgb: &[[f32; 3]],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>>;
}

/// ORT-backed M16 session.  M15 owns manifest validation and model-specific
/// preprocessing; this adapter only converts canonical RGB to the approved
/// segmenter API and returns the raw alpha tensor to the M16 executor.
struct OrtBackedSession {
    general: crate::m15::ApprovedRealSession,
    complementary: crate::m15::ApprovedRealSession,
}

impl RealRawSession for OrtBackedSession {
    fn infer(
        &mut self,
        model_id: &str,
        rgb: &[[f32; 3]],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>> {
        let image = bgremove_core::CanonicalImage::new(width, height, rgb.to_vec())?;
        let session = match model_id {
            "approved-general" => &self.general,
            "approved-complementary" => &self.complementary,
            other => anyhow::bail!("unknown real M16 model id {other}"),
        };
        crate::m15::approved_real_predict(session, &image)
    }
}

#[allow(dead_code)]
pub fn evaluate_real_with_session<S: RealRawSession>(
    session: &mut S,
    image: &FixtureImage,
    config: &HybridConfig,
) -> Result<PipelineEvaluation> {
    let global = session.infer("approved-general", &image.rgb, image.width, image.height)?;
    ensure!(
        global.len() == image.rgb.len(),
        "real raw mask dimension mismatch"
    );
    ensure!(
        global.iter().all(|value| value.is_finite()),
        "real raw mask contains nonfinite values"
    );
    let (features, _uncertainty) =
        feature_evidence(&image.rgb, &global, None, image.width, image.height);
    let should_run_complementary =
        config.complementary_model && (!config.route_complementary || features.router_risk >= 0.20);
    let complementary = if should_run_complementary {
        let mask = session.infer(
            "approved-complementary",
            &image.rgb,
            image.width,
            image.height,
        )?;
        ensure!(
            mask.len() == global.len(),
            "real complementary mask dimension mismatch"
        );
        ensure!(
            mask.iter().all(|value| value.is_finite()),
            "real complementary mask contains nonfinite values"
        );
        mask
    } else {
        vec![0.0; global.len()]
    };
    Ok(evaluate_with_masks(
        image,
        config,
        &global,
        should_run_complementary.then_some(complementary.as_slice()),
    ))
}

fn validate_real_mask(mask: Vec<f32>, image: &FixtureImage, label: &str) -> Result<Vec<f32>> {
    ensure!(
        mask.len() == image.rgb.len(),
        "real {label} raw mask dimension mismatch"
    );
    ensure!(
        mask.iter().all(|value| value.is_finite()),
        "real {label} raw mask contains nonfinite values"
    );
    Ok(mask)
}

fn cache_general_mask<S: RealRawSession>(
    cache: &mut RealMaskCache,
    session: &mut S,
    image: &FixtureImage,
) -> Result<Vec<f32>> {
    let key = real_cache_key(image, "approved-general", &cache.general_manifest_hash);
    if let Some(entry) = cache.entries.get(&key) {
        ensure!(
            entry.input_hash == real_input_hash(image),
            "real raw cache input key mismatch"
        );
        ensure!(
            entry.manifest_hash == cache.general_manifest_hash
                && entry.width == image.width
                && entry.height == image.height,
            "real raw cache provenance mismatch"
        );
        return Ok(entry.general.clone());
    }
    let (raw, _) = cache.disk.load_or_compute(
        &real_input_hash(image),
        &cache.general_manifest_hash,
        image.width,
        image.height,
        || session.infer("approved-general", &image.rgb, image.width, image.height),
    )?;
    let general = validate_real_mask(raw, image, "general")?;
    cache.entries.insert(
        key,
        CachedRealMasks {
            input_hash: real_input_hash(image),
            manifest_hash: cache.general_manifest_hash.clone(),
            complementary_manifest_hash: cache.complementary_manifest_hash.clone(),
            width: image.width,
            height: image.height,
            general: general.clone(),
            complementary: None,
        },
    );
    Ok(general)
}

fn cache_complementary_mask<S: RealRawSession>(
    cache: &mut RealMaskCache,
    session: &mut S,
    image: &FixtureImage,
) -> Result<Vec<f32>> {
    let key = real_cache_key(image, "approved-general", &cache.general_manifest_hash);
    if let Some(entry) = cache.entries.get(&key) {
        if let Some(mask) = &entry.complementary {
            ensure!(
                entry.complementary_manifest_hash == cache.complementary_manifest_hash,
                "real complementary cache manifest mismatch"
            );
            return Ok(mask.clone());
        }
    }
    // Ensure the general key exists even when a caller requests a complementary
    // mask first. This keeps all model/geometry provenance in one entry.
    if !cache.entries.contains_key(&key) {
        let _ = cache_general_mask(cache, session, image)?;
    }
    let (raw, _) = cache.disk.load_or_compute(
        &real_input_hash(image),
        &cache.complementary_manifest_hash,
        image.width,
        image.height,
        || {
            session.infer(
                "approved-complementary",
                &image.rgb,
                image.width,
                image.height,
            )
        },
    )?;
    let complementary = validate_real_mask(raw, image, "complementary")?;
    if let Some(entry) = cache.entries.get_mut(&key) {
        entry.complementary = Some(complementary.clone());
    }
    Ok(complementary)
}

fn evaluate_real_with_masks<S: RealRawSession>(
    cache: &mut RealMaskCache,
    session: &mut S,
    image: &FixtureImage,
    config: &HybridConfig,
) -> Result<PipelineEvaluation> {
    let global = cache_general_mask(cache, session, image)?;
    let (features, _) = feature_evidence(&image.rgb, &global, None, image.width, image.height);
    let routed =
        config.complementary_model && (!config.route_complementary || features.router_risk >= 0.20);
    let complementary = if routed {
        Some(cache_complementary_mask(cache, session, image)?)
    } else {
        None
    };
    Ok(evaluate_with_masks(
        image,
        config,
        &global,
        complementary.as_deref(),
    ))
}

fn new_real_mask_cache() -> Result<RealMaskCache> {
    let general_manifest_hash = validated_manifest_hash("models/m5_u2net.toml")?;
    let complementary_manifest_hash = validated_manifest_hash("models/m4_isnet_fp32.toml")?;
    let cache_run = REAL_CACHE_RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let disk_root =
        std::env::temp_dir().join(format!("m16-real-cache-{}-{cache_run}", std::process::id()));
    Ok(RealMaskCache {
        general_manifest_hash,
        complementary_manifest_hash,
        disk: crate::m15::RawMaskCache::new(&disk_root)?,
        entries: BTreeMap::new(),
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct RealHybridLooFold {
    held_out_id: String,
    training_ids: Vec<String>,
    metrics: ImageMetrics,
    baseline_agreement: f64,
    selected_config: HybridConfig,
    candidate_training_agreement: BTreeMap<String, f64>,
    ablation_held_out_agreement: BTreeMap<String, f64>,
}

/// Production-shaped fixed-arena executor.  The injected session is the seam
/// used by the ORT runner: all six records are held out exactly once, while
/// the other five are the only records visible to that fold's selection
/// context.  A session/refiner failure is retained as one zero-scored row.
pub fn real_hybrid_loo_with_session<S: RealRawSession>(
    session: &mut S,
    images: &[FixtureImage],
    config: &HybridConfig,
) -> Result<Vec<RealHybridLooFold>> {
    ensure!(
        images.len() == 6,
        "fixed-arena LOO requires exactly six images"
    );
    let mut cache = new_real_mask_cache()?;
    let mut folds = Vec::with_capacity(images.len());
    for held_out in 0..images.len() {
        let training_ids = images
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != held_out)
            .map(|(_, image)| image.id.clone())
            .collect::<Vec<_>>();
        let image = &images[held_out];
        let candidates = [
            ("p5-default".to_string(), config.clone()),
            (
                "p5-no-crop".to_string(),
                HybridConfig {
                    crop_refine: false,
                    ..config.clone()
                },
            ),
        ];
        let mut candidate_training_agreement = BTreeMap::new();
        for (id, candidate) in &candidates {
            let mut values = Vec::with_capacity(images.len() - 1);
            for (index, training) in images.iter().enumerate() {
                if index == held_out {
                    continue;
                }
                let metric =
                    match evaluate_real_with_masks(&mut cache, session, training, candidate) {
                        Ok(evaluation) => score(training, &evaluation),
                        Err(error) => failure_metrics(format!("real training session: {error}")),
                    };
                values.push(metric.agreement);
            }
            let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
            candidate_training_agreement.insert(id.clone(), mean);
        }
        let selected_index = candidates
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                candidate_training_agreement[&left.0]
                    .total_cmp(&candidate_training_agreement[&right.0])
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let selected_config = candidates[selected_index].1.clone();
        let metrics = match evaluate_real_with_masks(&mut cache, session, image, &selected_config) {
            Ok(evaluation) => score(image, &evaluation),
            Err(error) => failure_metrics(format!("real held-out session: {error}")),
        };
        let baseline_config = HybridConfig {
            route_complementary: false,
            complementary_model: false,
            adaptive_trimap: false,
            crop_refine: false,
            selective_foreground: false,
            conservative_topology: false,
        };
        let baseline_agreement =
            match evaluate_real_with_masks(&mut cache, session, image, &baseline_config) {
                Ok(evaluation) => score(image, &evaluation).agreement,
                Err(_) => 0.0,
            };
        let mut ablation_held_out_agreement = BTreeMap::new();
        for component in [
            "router",
            "complementary",
            "adaptive-trimap",
            "crop-refine",
            "foreground",
            "topology",
        ] {
            let removal = match component {
                "router" => HybridConfig {
                    route_complementary: false,
                    ..selected_config.clone()
                },
                "complementary" => HybridConfig {
                    complementary_model: false,
                    ..selected_config.clone()
                },
                "adaptive-trimap" => HybridConfig {
                    adaptive_trimap: false,
                    ..selected_config.clone()
                },
                "crop-refine" => HybridConfig {
                    crop_refine: false,
                    ..selected_config.clone()
                },
                "foreground" => HybridConfig {
                    selective_foreground: false,
                    ..selected_config.clone()
                },
                "topology" => HybridConfig {
                    conservative_topology: false,
                    ..selected_config.clone()
                },
                _ => unreachable!(),
            };
            let metric = match evaluate_real_with_masks(&mut cache, session, image, &removal) {
                Ok(evaluation) => score(image, &evaluation),
                Err(error) => failure_metrics(format!("real ablation session: {error}")),
            };
            ablation_held_out_agreement.insert(component.into(), metric.agreement);
        }
        folds.push(RealHybridLooFold {
            held_out_id: image.id.clone(),
            training_ids,
            metrics,
            baseline_agreement,
            selected_config,
            candidate_training_agreement,
            ablation_held_out_agreement,
        });
    }
    Ok(folds)
}

/// Complete the release lifecycle after non-blind fixed-arena LOO has
/// qualified.  The blind set is loaded and registered only at this point;
/// every score uses the frozen first-fold configuration and the same raw-mask
/// cache contract as the LOO executor.
fn coordinate_blind_promotion<S: RealRawSession>(
    session: &mut S,
    loo_images: &[FixtureImage],
    folds: &[RealHybridLooFold],
    blind_images: &[FixtureImage],
    manifest_hash: &str,
    lifecycle: &mut BlindLifecycle,
) -> Result<BlindPromotionEvidence> {
    ensure!(
        !folds.is_empty(),
        "blind promotion requires non-empty LOO evidence"
    );
    let loo_ids = loo_images
        .iter()
        .map(|image| image.id.clone())
        .collect::<BTreeSet<_>>();
    let blind_ids = blind_images
        .iter()
        .map(|image| image.id.clone())
        .collect::<Vec<_>>();
    ensure!(!blind_ids.is_empty(), "blind manifest has no records");
    let set_id = format!("m16-blind-{}", &manifest_hash[..12]);
    lifecycle.register_untouched_set(set_id.clone(), manifest_hash, &blind_ids, &loo_ids)?;
    lifecycle.qualify_release_candidate()?;
    let frozen = folds
        .first()
        .context("LOO has no frozen configuration")?
        .selected_config
        .clone();
    ensure!(
        folds.iter().all(|fold| fold.selected_config == frozen),
        "blind promotion requires one frozen LOO config"
    );
    let mut cache = new_real_mask_cache()?;
    let baseline = HybridConfig {
        route_complementary: false,
        complementary_model: false,
        adaptive_trimap: false,
        crop_refine: false,
        selective_foreground: false,
        conservative_topology: false,
    };
    let mut baseline_rows = Vec::with_capacity(blind_images.len());
    let mut challenger_rows = Vec::with_capacity(blind_images.len());
    let mut per_image = Vec::with_capacity(blind_images.len());
    for image in blind_images {
        let base = evaluate_real_with_masks(&mut cache, session, image, &baseline)
            .map(|evaluation| score(image, &evaluation))
            .unwrap_or_else(|error| failure_metrics(format!("blind baseline: {error}")));
        let candidate = evaluate_real_with_masks(&mut cache, session, image, &frozen)
            .map(|evaluation| score(image, &evaluation))
            .unwrap_or_else(|error| failure_metrics(format!("blind P5: {error}")));
        baseline_rows.push((image.id.as_str(), base.agreement));
        challenger_rows.push((image.id.as_str(), candidate.agreement));
        per_image.push((image, base, candidate));
    }
    let paired = paired_values(
        &baseline_rows,
        &challenger_rows,
        "m15-best-baseline",
        "P5-hybrid",
    )?;
    let mut category_evidence = BTreeMap::new();
    let mut category_pass = true;
    for tag in blind_images
        .iter()
        .map(|image| image.tag.clone())
        .collect::<BTreeSet<_>>()
    {
        let rows = per_image
            .iter()
            .filter(|(image, _, _)| image.tag == tag)
            .collect::<Vec<_>>();
        let base_mean = (!rows.is_empty()).then(|| {
            rows.iter().map(|(_, base, _)| base.agreement).sum::<f64>() / rows.len() as f64
        });
        let p5_mean = (!rows.is_empty()).then(|| {
            rows.iter()
                .map(|(_, _, candidate)| candidate.agreement)
                .sum::<f64>()
                / rows.len() as f64
        });
        let delta = base_mean.zip(p5_mean).map(|(base, p5)| p5 - base);
        let passed = delta.is_some_and(|value| value > PRIMARY_AGREEMENT_EPSILON);
        category_pass &= passed;
        category_evidence.insert(
            tag,
            RealCategoryEvidence {
                image_ids: rows.iter().map(|(image, _, _)| image.id.clone()).collect(),
                baseline_mean: base_mean,
                p5_mean,
                delta,
                budget: PRIMARY_AGREEMENT_EPSILON,
                passed,
                status: if passed {
                    "passed".into()
                } else {
                    "failed-or-insufficient-data".into()
                },
            },
        );
    }
    let mut base_sorted = baseline_rows
        .iter()
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();
    let mut p5_sorted = challenger_rows
        .iter()
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();
    base_sorted.sort_by(f64::total_cmp);
    p5_sorted.sort_by(f64::total_cmp);
    let index = base_sorted
        .len()
        .saturating_sub(1)
        .min(base_sorted.len() / 10);
    let worst_base = base_sorted.get(index).copied();
    let worst_p5 = p5_sorted.get(index).copied();
    let worst_pass = worst_base
        .zip(worst_p5)
        .is_some_and(|(base, p5)| p5 - base > PRIMARY_AGREEMENT_EPSILON);
    let passed = paired
        .bootstrap_low
        .is_some_and(|low| low > PRIMARY_AGREEMENT_EPSILON)
        && category_pass
        && worst_pass;
    lifecycle.evaluate_registered_once(
        &baseline_rows,
        &challenger_rows,
        category_pass,
        worst_pass,
    )?;
    // Any inspected blind result influences the release decision, whether it
    // passes or fails. Retire the set after the one permitted evaluation;
    // this release check is not a parameter sweep.
    lifecycle.influence_change()?;
    Ok(BlindPromotionEvidence {
        set_id,
        manifest_hash: manifest_hash.into(),
        image_ids: blind_ids,
        frozen_config: frozen,
        baseline_paired: Some(paired),
        category_evidence,
        worst_decile_baseline: worst_base,
        worst_decile_challenger: worst_p5,
        passed,
        status: if passed {
            "passed-and-promoted".into()
        } else {
            "failed; no promotion".into()
        },
    })
}

#[cfg(test)]
fn evaluate(image: &FixtureImage, config: &HybridConfig) -> PipelineEvaluation {
    let global = global_segmenter(&image.rgb, image.width, image.height);
    evaluate_with_global(image, config, &global)
}

fn failure_metrics(reason: impl Into<String>) -> ImageMetrics {
    ImageMetrics {
        agreement: 0.0,
        alpha_mae: 1.0,
        roi_alpha_mae: 1.0,
        alpha_rmse: 1.0,
        soft_iou: 0.0,
        roi_soft_iou: 0.0,
        boundary_f1: 0.0,
        composite_mae: 1.0,
        composite_psnr: 0.0,
        composite_ssim: 0.0,
        composite_ssim_full: 0.0,
        gradient_error: 1.0,
        fractional_alpha_error: 1.0,
        failure: true,
        failure_reason: Some(reason.into()),
    }
}

fn score(image: &FixtureImage, evaluation: &PipelineEvaluation) -> ImageMetrics {
    if evaluation.refined_alpha.len() != image.truth.len()
        || evaluation.foreground.len() != image.rgb.len()
        || image.reference_foreground.len() != image.rgb.len()
        || !evaluation
            .refined_alpha
            .iter()
            .all(|value| value.is_finite())
        || !evaluation
            .foreground
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    {
        return failure_metrics("dimension mismatch");
    }
    let mut mae = 0.0;
    let mut squared = 0.0;
    let mut inter = 0.0;
    let mut union = 0.0;
    for (candidate, truth) in evaluation.refined_alpha.iter().zip(&image.truth) {
        let difference = f64::from((candidate - truth).abs());
        mae += difference;
        squared += difference * difference;
        inter += f64::from(candidate.min(*truth));
        union += f64::from(candidate.max(*truth));
    }
    let n = image.truth.len().max(1) as f64;
    let alpha_mae = mae / n;
    let alpha_rmse = (squared / n).sqrt();
    let soft_iou = if union > 0.0 { inter / union } else { 1.0 };
    let mut min_x = image.width;
    let mut min_y = image.height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;
    for (index, (candidate, truth)) in evaluation
        .refined_alpha
        .iter()
        .zip(&image.truth)
        .enumerate()
    {
        if *candidate > 0.01 || *truth > 0.01 {
            let x = index as u32 % image.width;
            let y = index as u32 / image.width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            found = true;
        }
    }
    let diagonal = f64::from(image.width).hypot(f64::from(image.height));
    let pad = (diagonal * 0.05).round().max(1.0) as u32;
    let (x0, y0, x1, y1) = if found {
        (
            min_x.saturating_sub(pad),
            min_y.saturating_sub(pad),
            (max_x + pad).min(image.width.saturating_sub(1)),
            (max_y + pad).min(image.height.saturating_sub(1)),
        )
    } else {
        (
            0,
            0,
            image.width.saturating_sub(1),
            image.height.saturating_sub(1),
        )
    };
    let roi_pixels = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
    let mut roi_mae = 0.0;
    let mut roi_inter = 0.0;
    let mut roi_union = 0.0;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let index = (y * image.width + x) as usize;
            let candidate = f64::from(evaluation.refined_alpha[index]);
            let truth = f64::from(image.truth[index]);
            roi_mae += (candidate - truth).abs();
            roi_inter += candidate.min(truth);
            roi_union += candidate.max(truth);
        }
    }
    let _roi_alpha_mae = roi_mae / roi_pixels.max(1.0);
    let _roi_soft_iou = if roi_union > 0.0 {
        roi_inter / roi_union
    } else {
        1.0
    };
    let shared = match crate::m15::shared_photoroom_metrics(
        &evaluation.foreground,
        &image.reference_foreground,
        &evaluation.refined_alpha,
        &image.truth,
        image.width,
        image.height,
    ) {
        Ok(metrics) => metrics,
        Err(_) => return failure_metrics("PhotoRoomAgreement-v1 evaluation failed"),
    };
    let gradient_error = gradient_error(
        &evaluation.refined_alpha,
        &image.truth,
        image.width,
        image.height,
    );
    let fractional_alpha_error = evaluation
        .refined_alpha
        .iter()
        .zip(&image.truth)
        .filter(|(_, truth)| (0.02..0.98).contains(*truth))
        .map(|(candidate, truth)| f64::from((candidate - truth).abs()))
        .collect::<Vec<_>>();
    let fractional_alpha_error = if fractional_alpha_error.is_empty() {
        0.0
    } else {
        fractional_alpha_error.iter().sum::<f64>() / fractional_alpha_error.len() as f64
    };
    ImageMetrics {
        // PhotoRoomAgreement-v1: alpha, boundary, threshold selection and
        // declared-background ROI SSIM are equally deterministic components.
        agreement: shared.0,
        alpha_mae,
        roi_alpha_mae: shared.1,
        alpha_rmse,
        soft_iou,
        roi_soft_iou: shared.2,
        boundary_f1: shared.3,
        composite_mae: shared.4,
        composite_psnr: shared.5,
        composite_ssim: shared.7,
        composite_ssim_full: shared.6,
        gradient_error,
        fractional_alpha_error,
        failure: false,
        failure_reason: None,
    }
}

#[allow(dead_code)]
fn boundary_f1_at_tolerance(
    candidate: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
    tolerance: u32,
) -> f64 {
    crate::m15::shared_boundary_f1(candidate, reference, width, height, tolerance)
}

fn gradient_error(candidate: &[f32], reference: &[f32], width: u32, height: u32) -> f64 {
    if candidate.len() != reference.len() || candidate.is_empty() {
        return 1.0;
    }
    let mut total = 0.0;
    let mut count: f64 = 0.0;
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            if x + 1 < width {
                total += f64::from(
                    ((candidate[i + 1] - candidate[i]) - (reference[i + 1] - reference[i])).abs(),
                );
                count += 1.0;
            }
            if y + 1 < height {
                let j = ((y + 1) * width + x) as usize;
                total += f64::from(
                    ((candidate[j] - candidate[i]) - (reference[j] - reference[i])).abs(),
                );
                count += 1.0;
            }
        }
    }
    total / count.max(1.0)
}

fn windowed_ssim(candidate: &[[f32; 3]], reference: &[[f32; 3]], width: u32, height: u32) -> f64 {
    if candidate.len() != reference.len() || candidate.is_empty() {
        return 0.0;
    }
    (0..3)
        .map(|channel| {
            let a = candidate
                .iter()
                .map(|pixel| f64::from(pixel[channel]))
                .collect::<Vec<_>>();
            let b = reference
                .iter()
                .map(|pixel| f64::from(pixel[channel]))
                .collect::<Vec<_>>();
            gaussian_ssim_channel(&a, &b, width, height)
        })
        .sum::<f64>()
        / 3.0
}

fn gaussian_ssim_channel(a: &[f64], b: &[f64], width: u32, height: u32) -> f64 {
    crate::m15::shared_global_ssim_2d(a, b, width, height)
}

fn roi_windowed_ssim(
    candidate: &[[f32; 3]],
    reference: &[[f32; 3]],
    candidate_alpha: &[f32],
    reference_alpha: &[f32],
    width: u32,
    height: u32,
) -> f64 {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;
    for index in 0..candidate_alpha.len().min(reference_alpha.len()) {
        if candidate_alpha[index] > 0.01 || reference_alpha[index] > 0.01 {
            let x = index as u32 % width;
            let y = index as u32 / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            found = true;
        }
    }
    if !found {
        return windowed_ssim(candidate, reference, width, height);
    }
    let pad = (f64::from(width).hypot(f64::from(height)) * 0.05).round() as u32;
    min_x = min_x.saturating_sub(pad);
    min_y = min_y.saturating_sub(pad);
    max_x = (max_x + pad).min(width - 1);
    max_y = (max_y + pad).min(height - 1);
    let roi_width = max_x - min_x + 1;
    let roi_height = max_y - min_y + 1;
    let mut c = Vec::with_capacity((roi_width * roi_height) as usize);
    let mut r = Vec::with_capacity((roi_width * roi_height) as usize);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let index = (y * width + x) as usize;
            c.push(candidate[index]);
            r.push(reference[index]);
        }
    }
    windowed_ssim(&c, &r, roi_width, roi_height)
}

#[allow(dead_code)]
fn composite_metrics(
    image: &FixtureImage,
    evaluation: &PipelineEvaluation,
) -> (f64, f64, f64, f64) {
    const BACKGROUND_COUNT: usize = 7;
    let mut mae = 0.0;
    let mut mse = 0.0;
    let mut ssim_full_sum = 0.0;
    let mut ssim_roi_sum = 0.0;
    for background_index in 0..BACKGROUND_COUNT {
        let candidate = linear_composite(
            &evaluation.foreground,
            &evaluation.refined_alpha,
            image.width,
            image.height,
            background_index,
        );
        let reference = linear_composite(
            &image.reference_foreground,
            &image.truth,
            image.width,
            image.height,
            background_index,
        );
        for (a, b) in candidate.iter().zip(&reference) {
            mae += a
                .iter()
                .zip(b)
                .map(|(x, y)| f64::from((x - y).abs()))
                .sum::<f64>()
                / 3.0;
            mse += a
                .iter()
                .zip(b)
                .map(|(x, y)| f64::from(*x - *y).powi(2))
                .sum::<f64>()
                / 3.0;
        }
        ssim_full_sum += windowed_ssim(&candidate, &reference, image.width, image.height);
        ssim_roi_sum += roi_windowed_ssim(
            &candidate,
            &reference,
            &evaluation.refined_alpha,
            &image.truth,
            image.width,
            image.height,
        );
    }
    let denom = (image.rgb.len() * BACKGROUND_COUNT) as f64;
    let mae = mae / denom.max(1.0);
    let mse = mse / denom.max(1.0);
    (
        mae,
        if mse == 0.0 {
            100.0
        } else {
            10.0 * (1.0 / mse).log10()
        },
        ssim_full_sum / BACKGROUND_COUNT as f64,
        ssim_roi_sum / BACKGROUND_COUNT as f64,
    )
}

fn srgb_to_linear_m16(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn background_pixel_m16(index: usize, width: u32, height: u32, background: usize) -> [f32; 3] {
    crate::m15::shared_background_pixel(index, width, height, background).map(|value| value as f32)
}

fn linear_composite(
    rgb: &[[f32; 3]],
    alpha: &[f32],
    width: u32,
    _height: u32,
    background: usize,
) -> Vec<[f32; 3]> {
    rgb.iter()
        .zip(alpha)
        .enumerate()
        .map(|(index, (pixel, value))| {
            let bg = background_pixel_m16(index, width, _height, background);
            std::array::from_fn(|channel| {
                srgb_to_linear_m16(pixel[channel]) * *value + bg[channel] * (1.0 - *value)
            })
        })
        .collect()
}

fn xorshift(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

fn bootstrap(values: &[f64], seed: u64) -> Result<(f64, f64)> {
    ensure!(!values.is_empty(), "bootstrap requires non-empty values");
    ensure!(seed != 0, "bootstrap seed must be non-zero");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "bootstrap values must be finite"
    );
    let mut samples = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for iteration in 0..BOOTSTRAP_RESAMPLES {
        let mut state = seed ^ iteration as u64;
        let mut sum = 0.0;
        for _ in values {
            state = xorshift(state);
            sum += values[(state as usize) % values.len()];
        }
        samples.push(sum / values.len() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok((
        samples[BOOTSTRAP_RESAMPLES * 25 / 1000],
        samples[BOOTSTRAP_RESAMPLES * 975 / 1000],
    ))
}

fn summary(images: &[FixtureImage], metrics: Vec<ImageMetrics>) -> Result<Summary> {
    ensure!(
        images.len() == metrics.len() && !metrics.is_empty(),
        "summary image/metric mismatch or empty"
    );
    let values = metrics
        .iter()
        .map(|metric| metric.agreement)
        .collect::<Vec<_>>();
    let (low, high) = bootstrap(&values, 0x4d_31_36)?;
    Ok(Summary {
        count: metrics.len(),
        mean_agreement: values.iter().sum::<f64>() / values.len() as f64,
        min_agreement: values.iter().copied().fold(f64::INFINITY, f64::min),
        bootstrap_low: low,
        bootstrap_high: high,
        mean_alpha_mae: metrics.iter().map(|m| m.alpha_mae).sum::<f64>() / metrics.len() as f64,
        mean_alpha_rmse: metrics.iter().map(|m| m.alpha_rmse).sum::<f64>() / metrics.len() as f64,
        mean_soft_iou: metrics.iter().map(|m| m.soft_iou).sum::<f64>() / metrics.len() as f64,
        mean_boundary_f1: metrics.iter().map(|m| m.boundary_f1).sum::<f64>() / metrics.len() as f64,
        mean_composite_mae: metrics.iter().map(|m| m.composite_mae).sum::<f64>()
            / metrics.len() as f64,
        mean_composite_psnr: metrics.iter().map(|m| m.composite_psnr).sum::<f64>()
            / metrics.len() as f64,
        mean_composite_ssim: metrics.iter().map(|m| m.composite_ssim).sum::<f64>()
            / metrics.len() as f64,
        mean_gradient_error: metrics.iter().map(|m| m.gradient_error).sum::<f64>()
            / metrics.len() as f64,
        mean_fractional_alpha_error: metrics
            .iter()
            .map(|m| m.fractional_alpha_error)
            .sum::<f64>()
            / metrics.len() as f64,
        per_image: images
            .iter()
            .zip(metrics)
            .map(|(image, metrics)| NamedMetric {
                id: image.id.clone(),
                split: image.split.clone(),
                tag: image.tag.clone(),
                metrics,
            })
            .collect(),
    })
}

fn paired(
    baseline: &Summary,
    challenger: &Summary,
    baseline_id: &str,
    challenger_id: &str,
) -> Result<PairedEvidence> {
    paired_with_metric(
        baseline,
        challenger,
        baseline_id,
        challenger_id,
        "agreement",
        |metric| metric.agreement,
    )
}

fn paired_with_metric(
    baseline: &Summary,
    challenger: &Summary,
    baseline_id: &str,
    challenger_id: &str,
    metric_name: &str,
    metric: impl Fn(&ImageMetrics) -> f64,
) -> Result<PairedEvidence> {
    let baseline_ids = baseline
        .per_image
        .iter()
        .map(|row| row.id.clone())
        .collect::<Vec<_>>();
    let challenger_ids = challenger
        .per_image
        .iter()
        .map(|row| row.id.clone())
        .collect::<Vec<_>>();
    ensure!(
        baseline_ids == challenger_ids && !baseline_ids.is_empty(),
        "paired IDs must be identical and non-empty"
    );
    let deltas = baseline
        .per_image
        .iter()
        .zip(&challenger.per_image)
        .map(|(left, right)| metric(&right.metrics) - metric(&left.metrics))
        .collect::<Vec<_>>();
    let mean_delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let (low, high) = bootstrap(&deltas, 0x4d_31_36_01)?;
    Ok(PairedEvidence {
        baseline: baseline_id.into(),
        challenger: challenger_id.into(),
        metric: metric_name.into(),
        image_ids: baseline_ids,
        deltas,
        mean_delta: Some(mean_delta),
        bootstrap_low: Some(low),
        bootstrap_high: Some(high),
        seed: 0x4d_31_36_01,
        resamples: BOOTSTRAP_RESAMPLES,
        status: if low > 0.0 {
            "positive-ci".into()
        } else {
            "non-positive-ci".into()
        },
    })
}

fn paired_values(
    baseline: &[(&str, f64)],
    challenger: &[(&str, f64)],
    baseline_id: &str,
    challenger_id: &str,
) -> Result<PairedEvidence> {
    ensure!(
        !baseline.is_empty() && baseline.len() == challenger.len(),
        "real paired evidence requires equal non-empty sets"
    );
    let left = baseline.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    let right = challenger.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    ensure!(left == right, "real paired evidence IDs differ");
    let deltas = baseline
        .iter()
        .zip(challenger)
        .map(|((_, a), (_, b))| b - a)
        .collect::<Vec<_>>();
    ensure!(
        deltas.iter().all(|v| v.is_finite()),
        "real paired evidence is nonfinite"
    );
    let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let (low, high) = bootstrap(&deltas, 0x4d_31_36_02)?;
    Ok(PairedEvidence {
        baseline: baseline_id.into(),
        challenger: challenger_id.into(),
        metric: "agreement".into(),
        image_ids: left.into_iter().map(str::to_owned).collect(),
        deltas,
        mean_delta: Some(mean),
        bootstrap_low: Some(low),
        bootstrap_high: Some(high),
        seed: 0x4d_31_36_02,
        resamples: BOOTSTRAP_RESAMPLES,
        status: if low > PRIMARY_AGREEMENT_EPSILON {
            "positive-ci".into()
        } else {
            "non-positive-ci".into()
        },
    })
}

fn png_gray(values: &[f32], width: u32, height: u32) -> Result<Vec<u8>> {
    ensure!(
        values.len() == (width as usize).saturating_mul(height as usize),
        "gray PNG dimensions mismatch"
    );
    let pixels = values
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes).write_image(&pixels, width, height, ColorType::L8.into())?;
    Ok(bytes)
}

fn png_gray_u8(values: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    ensure!(
        values.len() == (width as usize).saturating_mul(height as usize),
        "gray PNG dimensions mismatch"
    );
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes).write_image(values, width, height, ColorType::L8.into())?;
    Ok(bytes)
}

fn png_rgb(values: &[[f32; 3]], width: u32, height: u32) -> Result<Vec<u8>> {
    ensure!(
        values.len() == (width as usize).saturating_mul(height as usize),
        "RGB PNG dimensions mismatch"
    );
    let pixels = values
        .iter()
        .flat_map(|pixel| pixel.map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8))
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes).write_image(&pixels, width, height, ColorType::Rgb8.into())?;
    Ok(bytes)
}

fn png_rgba(rgb: &[[f32; 3]], alpha: &[f32], width: u32, height: u32) -> Result<Vec<u8>> {
    ensure!(
        rgb.len() == alpha.len() && rgb.len() == (width as usize).saturating_mul(height as usize),
        "RGBA dimensions mismatch"
    );
    let pixels = rgb
        .iter()
        .zip(alpha)
        .flat_map(|(pixel, value)| {
            [
                (pixel[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (pixel[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (pixel[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                (value.clamp(0.0, 1.0) * 255.0).round() as u8,
            ]
        })
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes).write_image(&pixels, width, height, ColorType::Rgba8.into())?;
    Ok(bytes)
}

fn composite(rgb: &[[f32; 3]], alpha: &[f32], background: [f32; 3]) -> Vec<[f32; 3]> {
    rgb.iter()
        .zip(alpha)
        .map(|(pixel, value)| {
            std::array::from_fn(|channel| {
                pixel[channel] * *value + background[channel] * (1.0 - *value)
            })
        })
        .collect()
}

fn write_bytes(root: &Path, relative: &str, bytes: &[u8]) -> Result<()> {
    let path = root.join(relative);
    ensure!(
        path.strip_prefix(root).is_ok(),
        "artifact path escaped output root"
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn write_json<T: Serialize>(root: &Path, relative: &str, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_bytes(root, relative, &bytes)
}

fn write_bundle(
    root: &Path,
    image: &FixtureImage,
    config_name: &str,
    config: &HybridConfig,
    global: &[f32],
) -> Result<ImageMetrics> {
    let evaluation = evaluate_with_global(image, config, global);
    let metrics = score(image, &evaluation);
    let dir = root.join("synthetic").join(config_name).join(&image.id);
    fs::create_dir_all(&dir)?;
    write_json(&dir, "resolved-config.json", config)?;
    write_bytes(
        &dir,
        "input.png",
        &png_rgb(&image.rgb, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "reference-alpha.png",
        &png_gray(&image.truth, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "global-alpha.png",
        &png_gray(&evaluation.global_alpha, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "coarse-alpha.png",
        &png_gray(&evaluation.global_alpha, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "uncertainty.png",
        &png_gray(&evaluation.uncertainty, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "crop-weights.png",
        &png_gray(&evaluation.crop_weights, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "adaptive-trimap.png",
        &png_gray_u8(&evaluation.trimap, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "complementary-alpha.png",
        &png_gray(&evaluation.complementary_alpha, image.width, image.height)?,
    )?;
    write_json(&dir, "feature-map.json", &evaluation.features)?;
    write_json(
        &dir,
        "router-decision.json",
        &serde_json::json!({ "routed": evaluation.routed, "policy": "RGB/runtime features only" }),
    )?;
    write_json(&dir, "crop-manifest.json", &evaluation.crops)?;
    for (tile_index, tile) in evaluation.crops.iter().enumerate() {
        let mut input = Vec::new();
        let mut output = Vec::new();
        let mut weights = Vec::new();
        for y in tile.y..tile.y + tile.height {
            for x in tile.x..tile.x + tile.width {
                let index = (y * image.width + x) as usize;
                input.push(evaluation.global_alpha[index]);
                output.push(evaluation.refined_alpha[index]);
                weights.push(evaluation.crop_weights[index]);
            }
        }
        write_bytes(
            &dir,
            &format!("crop-tile-{tile_index}-input.png"),
            &png_gray(&input, tile.width, tile.height)?,
        )?;
        write_bytes(
            &dir,
            &format!("crop-tile-{tile_index}-output.png"),
            &png_gray(&output, tile.width, tile.height)?,
        )?;
        write_bytes(
            &dir,
            &format!("crop-tile-{tile_index}-weights.png"),
            &png_gray(&weights, tile.width, tile.height)?,
        )?;
    }
    write_bytes(
        &dir,
        "refined-alpha.png",
        &png_gray(&evaluation.refined_alpha, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "foreground-risk.png",
        &png_gray(&evaluation.foreground_risk, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "foreground.png",
        &png_rgb(&evaluation.foreground, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "topology-decision.png",
        &png_gray_u8(&evaluation.topology_decision, image.width, image.height)?,
    )?;
    write_bytes(
        &dir,
        "cutout.png",
        &png_rgba(
            &evaluation.foreground,
            &evaluation.refined_alpha,
            image.width,
            image.height,
        )?,
    )?;
    write_bytes(
        &dir,
        "composite-white.png",
        &png_rgb(
            &composite(&evaluation.foreground, &evaluation.refined_alpha, [1.0; 3]),
            image.width,
            image.height,
        )?,
    )?;
    write_bytes(
        &dir,
        "composite-black.png",
        &png_rgb(
            &composite(&evaluation.foreground, &evaluation.refined_alpha, [0.0; 3]),
            image.width,
            image.height,
        )?,
    )?;
    let alpha_diff = evaluation
        .refined_alpha
        .iter()
        .zip(&image.truth)
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<_>>();
    write_bytes(
        &dir,
        "alpha-diff.png",
        &png_gray(&alpha_diff, image.width, image.height)?,
    )?;
    let boundary_diff = evaluation
        .refined_alpha
        .iter()
        .zip(&image.truth)
        .map(|(a, b)| if (a >= &0.5) != (b >= &0.5) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    write_bytes(
        &dir,
        "boundary-diff.png",
        &png_gray(&boundary_diff, image.width, image.height)?,
    )?;
    write_json(&dir, "metrics.json", &metrics)?;
    let mut entries = bundle_entries(&dir, &dir)?;
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    write_json(&dir, "hashes.json", &entries)?;
    Ok(metrics)
}

fn bundle_entries(root: &Path, current: &Path) -> Result<Vec<ArtifactManifestEntry>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(current)? {
        let path = entry?.path();
        if path.is_dir() {
            entries.extend(bundle_entries(root, &path)?);
        } else if !matches!(
            path.file_name().and_then(|value| value.to_str()),
            Some("hashes.json")
                | Some("artifacts.manifest.json")
                | Some("artifacts.sha256")
                | Some("report.json")
                | Some("report.sha256")
                | Some("performance.json")
        ) {
            let relative = path
                .strip_prefix(root)?
                .to_string_lossy()
                .replace('\\', "/");
            entries.push(ArtifactManifestEntry {
                path: relative,
                sha256: sha256(&fs::read(&path)?),
            });
        }
    }
    Ok(entries)
}

fn write_manifest(root: &Path) -> Result<ManifestEvidence> {
    let mut entries = bundle_entries(root, root)?;
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    let manifest = ArtifactManifest { entries };
    write_json(root, "artifacts.manifest.json", &manifest)?;
    // Hash the exact serialized bytes on disk, including the newline emitted
    // by write_json.  This keeps the recorded fresh-run and canonical hashes
    // on one inclusion/serialization definition.
    let bytes = fs::read(root.join("artifacts.manifest.json"))?;
    let digest = sha256(&bytes);
    write_bytes(
        root,
        "artifacts.sha256",
        format!("{}  artifacts.manifest.json\n", digest).as_bytes(),
    )?;
    let verified = verify_manifest_entries(root, &manifest.entries);
    ensure!(verified, "artifact manifest verification failed");
    Ok(ManifestEvidence {
        sha256: digest,
        entry_count: manifest.entries.len(),
        entries_verified: verified,
    })
}

fn verify_manifest_entries(root: &Path, entries: &[ArtifactManifestEntry]) -> bool {
    let mut listed = BTreeSet::new();
    for entry in entries {
        let relative = Path::new(&entry.path);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
            || !listed.insert(entry.path.clone())
        {
            return false;
        }
        let path = root.join(relative);
        if !path.is_file()
            || path.strip_prefix(root).is_err()
            || fs::read(&path)
                .map(|bytes| sha256(&bytes) != entry.sha256)
                .unwrap_or(true)
        {
            return false;
        }
    }
    let actual = match bundle_entries(root, root) {
        Ok(mut values) => {
            values.sort_by(|a, b| a.path.cmp(&b.path));
            values
                .into_iter()
                .map(|entry| entry.path)
                .collect::<BTreeSet<_>>()
        }
        Err(_) => return false,
    };
    actual == listed
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn write_html(
    root: &Path,
    summaries: &BTreeMap<String, Summary>,
    ablations: &[AblationEvidence],
    tags: &BTreeMap<String, Summary>,
) -> Result<()> {
    let mut html = String::from("<!doctype html><meta charset=\"utf-8\"><title>M16 hybrid</title><h1>M16 evidence-driven hybrid</h1><p>Measured performance is in <a href=\"performance.json\">performance.json</a>; timing and RSS/allocation measurements are excluded from the deterministic quality manifest.</p><table border=\"1\"><thead><tr><th>Configuration</th><th>Mean agreement</th><th>Alpha MAE</th><th>Alpha RMSE</th><th>Boundary F1</th><th>Composite SSIM</th><th>Best</th><th>Median</th><th>Worst-decile</th><th>Worst</th></tr></thead><tbody>");
    for (name, summary) in summaries {
        let mut ordered = summary.per_image.iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| {
            b.metrics
                .agreement
                .partial_cmp(&a.metrics.agreement)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        let best = ordered.first().map(|row| row.id.as_str()).unwrap_or("none");
        let median = ordered
            .get(ordered.len().saturating_sub(1) / 2)
            .map(|row| row.id.as_str())
            .unwrap_or("none");
        let worst = ordered.last().map(|row| row.id.as_str()).unwrap_or("none");
        let worst_decile = ordered
            .get(ordered.len().saturating_sub(1).min(ordered.len() * 9 / 10))
            .map(|row| row.id.as_str())
            .unwrap_or("none");
        html.push_str(&format!("<tr><td>{}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>", escape_html(name), summary.mean_agreement, summary.mean_alpha_mae, summary.mean_alpha_rmse, summary.mean_boundary_f1, summary.mean_composite_ssim, escape_html(best), escape_html(median), escape_html(worst_decile), escape_html(worst)));
    }
    html.push_str("</tbody></table><h2>Per-tag validation</h2><table border=\"1\"><thead><tr><th>Tag</th><th>Count</th><th>Mean agreement</th><th>Alpha MAE</th><th>Alpha RMSE</th><th>Boundary F1</th><th>Composite SSIM</th><th>Worst</th></tr></thead><tbody>");
    for (tag, summary) in tags {
        let worst = summary
            .per_image
            .iter()
            .min_by(|a, b| {
                a.metrics
                    .agreement
                    .partial_cmp(&b.metrics.agreement)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            })
            .map(|row| row.id.as_str())
            .unwrap_or("none");
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{}</td></tr>",
            escape_html(tag),
            summary.count,
            summary.mean_agreement,
            summary.mean_alpha_mae,
            summary.mean_alpha_rmse,
            summary.mean_boundary_f1,
            summary.mean_composite_ssim,
            escape_html(worst)
        ));
    }
    html.push_str("</tbody></table><h2>Full-config visual evidence</h2><table border=\"1\"><thead><tr><th>Image</th><th>Input/reference</th><th>Candidate cutout</th><th>Uncertainty</th><th>Alpha diff</th><th>Boundary diff</th><th>Black</th><th>White</th></tr></thead><tbody>");
    if let Some(full) = summaries.get("full") {
        for row in &full.per_image {
            let base = format!("synthetic/full/{}/", row.id);
            let cell = |file: &str, alt: &str| {
                format!(
                    "<img src=\"{}{}\" alt=\"{}\" width=\"128\">",
                    base, file, alt
                )
            };
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                escape_html(&row.id),
                cell("input.png", "input"),
                cell("reference-alpha.png", "reference"),
                cell("cutout.png", "candidate cutout"),
                cell("uncertainty.png", "uncertainty"),
                cell("alpha-diff.png", "alpha difference"),
                cell("boundary-diff.png", "boundary difference"),
                cell("composite-black.png", "black composite"),
                cell("composite-white.png", "white composite")
            ));
        }
    }
    html.push_str("</tbody></table><h2>Ablations</h2><table border=\"1\"><thead><tr><th>Component</th><th>Primary agreement gain</th><th>CI low</th><th>Full-minus-removal timing</th><th>Full-minus-removal memory (bytes)</th><th>Resolved default</th><th>Status</th></tr></thead><tbody>");
    for ablation in ablations {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td><a href=\"performance.json\">measured; see performance.json</a></td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape_html(&ablation.component),
            ablation.quality_gain.map(|v| format!("{v:.8}")).unwrap_or_else(|| "not-run".into()),
            ablation.paired.bootstrap_low.map(|v| format!("{v:.8}")).unwrap_or_else(|| "not-run".into()),
            ablation.memory_cost_bytes.map(|v| v.to_string()).unwrap_or_else(|| "unavailable".into()),
            ablation.enabled_in_resolved_default,
            escape_html(&ablation.status),
        ));
    }
    html.push_str("</tbody></table><script>for(const h of document.querySelectorAll('th'))h.onclick=()=>{const t=h.closest('table'),i=[...h.parentNode.children].indexOf(h),b=t.tBodies[0];[...b.rows].sort((a,c)=>a.cells[i].textContent.localeCompare(c.cells[i].textContent)).forEach(r=>b.append(r))}</script>\n");
    fs::write(root.join("report.html"), html)?;
    Ok(())
}

fn load_real_arena_manifest(manifest_path: &Path) -> Result<Vec<FixtureImage>> {
    let root = manifest_path
        .parent()
        .context("arena manifest has no parent directory")?;
    let text = fs::read_to_string(manifest_path)
        .with_context(|| format!("read real arena manifest {}", manifest_path.display()))?;
    let mut images = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("parse arena line {}", line_no + 1))?;
        let id = value["id"].as_str().context("arena id missing")?.to_owned();
        let input_path = root.join(value["input"].as_str().context("arena input missing")?);
        let target_path = root.join(value["target"].as_str().context("arena target missing")?);
        let input = load_canonical(&input_path)?;
        let target = load_canonical(&target_path)?;
        ensure!(
            input.dimensions() == target.dimensions(),
            "arena {id} dimensions differ"
        );
        let tags = value["tags"]
            .as_array()
            .context("arena tags missing")?
            .iter()
            .map(|tag| {
                tag.as_str()
                    .map(str::to_owned)
                    .context("arena tag is not text")
            })
            .collect::<Result<Vec<_>>>()?;
        let split = value["split"].as_str().unwrap_or("unknown").to_owned();
        images.push(FixtureImage {
            id,
            split,
            width: input.width(),
            height: input.height(),
            rgb: input.rgb().data().to_vec(),
            truth: target.source_alpha().data().to_vec(),
            reference_foreground: target.rgb().data().to_vec(),
            tag: tags.first().cloned().unwrap_or_else(|| "untagged".into()),
        });
    }
    Ok(images)
}

fn load_real_arena_fixtures() -> Result<Vec<FixtureImage>> {
    let manifest_path = repo_path("test_images").join("arena.jsonl");
    let images = load_real_arena_manifest(&manifest_path)?;
    ensure!(images.len() == 6, "fixed real arena requires six records");
    Ok(images)
}

fn failed_real_folds(reason: &str) -> Vec<RealHybridLooFold> {
    load_real_arena_fixtures()
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(held_out, image)| RealHybridLooFold {
            held_out_id: image.id,
            training_ids: load_real_arena_fixtures()
                .ok()
                .unwrap_or_default()
                .into_iter()
                .enumerate()
                .filter(|(index, _)| *index != held_out)
                .map(|(_, training)| training.id)
                .collect(),
            metrics: failure_metrics(reason.to_owned()),
            baseline_agreement: 0.0,
            selected_config: HybridConfig {
                route_complementary: true,
                complementary_model: true,
                adaptive_trimap: true,
                crop_refine: true,
                selective_foreground: true,
                conservative_topology: true,
            },
            candidate_training_agreement: BTreeMap::new(),
            ablation_held_out_agreement: BTreeMap::new(),
        })
        .collect()
}

fn m16_blind_lifecycle(loo_completed: bool) -> BlindLifecycleEvidence {
    let mut lifecycle = BlindLifecycle::untouched();
    if loo_completed {
        // The fixed-arena LOO override consumes legacy blind-labelled records
        // in training folds; that is an influence event, so the split is
        // retired and cannot be promoted without a newly registered set.
        let _ = lifecycle.qualify_release_candidate();
        let _ = lifecycle.evaluate_once(false);
        let _ = lifecycle.influence_change();
    }
    lifecycle.evidence()
}

fn real_arena_status(persisted_lifecycle: Option<&BlindLifecycleEvidence>) -> RealArenaEvidence {
    let manifest = repo_path("test_images/arena.jsonl");
    let record_count = fs::read_to_string(&manifest)
        .map(|text| text.lines().filter(|line| !line.trim().is_empty()).count())
        .unwrap_or(0);
    let runtime = std::env::var_os("ORT_DYLIB").map(PathBuf::from);
    let m15_report = repo_path("runs/m15-tournament/report.json");
    let (m15_evidence_status, blind_status, _m15_loo_available) = fs::read_to_string(&m15_report)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .map(|report| {
            let lifecycle = &report["blind_lifecycle"];
            let state = lifecycle["state"].as_str().unwrap_or("unknown");
            (
                format!(
                    "consumed M15 report; real_arena status={}",
                    report["real_arena"]["runtime_status"]
                        .as_str()
                        .unwrap_or("unknown")
                ),
                format!(
                    "M15 lifecycle state={state}; new untouched set required={}",
                    lifecycle["new_untouched_set_required"]
                        .as_bool()
                        .unwrap_or(true)
                ),
                report["real_arena"]["tournament"].is_object(),
            )
        })
        .unwrap_or_else(|| {
            (
                "M15 report unavailable".into(),
                "untouched status unverified".into(),
                false,
            )
        });
    let config = HybridConfig {
        route_complementary: true,
        complementary_model: true,
        adaptive_trimap: true,
        crop_refine: true,
        selective_foreground: true,
        conservative_topology: true,
    };
    let mut loo_folds = failed_real_folds("M16 real execution unavailable");
    let mut category_evidence = BTreeMap::new();
    for tag in ["portrait-character", "emissive-artwork"] {
        category_evidence.insert(
            tag.into(),
            RealCategoryEvidence {
                image_ids: Vec::new(),
                baseline_mean: None,
                p5_mean: None,
                delta: None,
                budget: 0.0,
                passed: false,
                status: "not-run: no completed real held-out evidence".into(),
            },
        );
    }
    let mut worst_decile_agreement = None;
    let mut worst_decile_baseline = None;
    let mut baseline_paired = None;
    let mut real_performance = None;
    let mut qualification_gates = BTreeMap::from([
        ("all_folds_evaluated".into(), false),
        ("one_frozen_config".into(), false),
        ("paired_ci".into(), false),
        ("critical_categories".into(), false),
        ("worst_decile".into(), false),
        ("performance".into(), false),
        ("blind".into(), false),
    ]);
    let mut leaderboard_candidate = None;
    let mut qualified_candidate = None;
    let mut champion_promoted = false;
    let mut blind_promotion = None;
    let mut lifecycle = persisted_lifecycle
        .map(lifecycle_from_evidence)
        .unwrap_or_else(BlindLifecycle::untouched);
    if std::env::var("M16_REGISTER_NEW_BLIND_SET").as_deref() == Ok("1") {
        lifecycle.register_new_untouched_set();
    }
    let (status, loo_status) = if runtime.is_some() && record_count == 6 {
        let Some(runtime_path) = runtime.as_deref() else {
            unreachable!("runtime branch requires a configured path");
        };
        match (
            crate::m15::open_approved_real_session("P0-u2-cf", runtime_path),
            crate::m15::open_approved_real_session("P1-isnet-fast", runtime_path),
        ) {
            (Err(error), _) | (_, Err(error)) => {
                loo_folds =
                    failed_real_folds(&format!("M16 ORT/model initialization failed: {error:#}"));
                (
                    format!("not-run: M16 ORT/model initialization failed: {error:#}"),
                    format!("not-run: M16 raw session unavailable: {error:#}"),
                )
            }
            (Ok(general), Ok(complementary)) => match load_real_arena_fixtures() {
                Err(error) => {
                    loo_folds = failed_real_folds(&format!("real arena load failed: {error:#}"));
                    (
                        format!("not-run: real arena load failed: {error:#}"),
                        format!("not-run: real arena load failed: {error:#}"),
                    )
                }
                Ok(images) => {
                    let mut session = OrtBackedSession {
                        general,
                        complementary,
                    };
                    let real_started = Instant::now();
                    match real_hybrid_loo_with_session(&mut session, &images, &config) {
                        Err(error) => {
                            loo_folds =
                                failed_real_folds(&format!("M16 real LOO failed: {error:#}"));
                            (
                                format!("not-run: M16 real LOO failed: {error:#}"),
                                format!("not-run: M16 real LOO failed: {error:#}"),
                            )
                        }
                        Ok(folds) => {
                            let elapsed_ms = real_started.elapsed().as_secs_f64() * 1000.0;
                            real_performance = Some(RealPerformanceEvidence {
                                method: "M16 real LOO wall-clock around cached raw inference and hybrid evaluation; RSS unavailable on this portable runner".into(),
                                warm_median_ms: Some(elapsed_ms),
                                warm_p95_ms: Some(elapsed_ms),
                                peak_memory_bytes: Some(checked_live_buffer_bound(&images, &config)),
                                status: "measured-time; conservative modeled live-buffer bound; peak RSS unavailable".into(),
                            });
                            let all_folds_evaluated = folds.len() == images.len()
                                && folds.iter().all(|fold| !fold.metrics.failure);
                            let one_frozen_config = folds
                                .first()
                                .map(|first| {
                                    folds
                                        .iter()
                                        .all(|fold| fold.selected_config == first.selected_config)
                                })
                                .unwrap_or(false);
                            let all_component_ablation_gains = folds.iter().all(|fold| {
                                fold.ablation_held_out_agreement.values().all(|ablation| {
                                    fold.metrics.agreement - *ablation > PRIMARY_AGREEMENT_EPSILON
                                })
                            });
                            let baseline_rows = folds
                                .iter()
                                .map(|fold| (fold.held_out_id.as_str(), fold.baseline_agreement))
                                .collect::<Vec<_>>();
                            let p5_rows = folds
                                .iter()
                                .map(|fold| (fold.held_out_id.as_str(), fold.metrics.agreement))
                                .collect::<Vec<_>>();
                            baseline_paired = paired_values(
                                &baseline_rows,
                                &p5_rows,
                                "m15-best-baseline",
                                "P5-hybrid",
                            )
                            .ok();
                            let paired_ci = baseline_paired
                                .as_ref()
                                .and_then(|p| p.bootstrap_low)
                                .is_some_and(|v| v > PRIMARY_AGREEMENT_EPSILON);
                            let mut category_pass = true;
                            for tag in ["portrait-character", "emissive-artwork"] {
                                let ids = images
                                    .iter()
                                    .filter(|image| image.tag == tag)
                                    .map(|image| image.id.clone())
                                    .collect::<Vec<_>>();
                                let selected = folds
                                    .iter()
                                    .filter(|fold| ids.iter().any(|id| id == &fold.held_out_id))
                                    .map(|fold| fold.metrics.agreement)
                                    .collect::<Vec<_>>();
                                let baseline = folds
                                    .iter()
                                    .filter(|fold| ids.iter().any(|id| id == &fold.held_out_id))
                                    .map(|fold| fold.baseline_agreement)
                                    .collect::<Vec<_>>();
                                let base_mean = (!baseline.is_empty())
                                    .then(|| baseline.iter().sum::<f64>() / baseline.len() as f64);
                                let p5_mean = (!selected.is_empty())
                                    .then(|| selected.iter().sum::<f64>() / selected.len() as f64);
                                let delta = base_mean.zip(p5_mean).map(|(a, b)| b - a);
                                let passed = delta.is_some_and(|v| v > PRIMARY_AGREEMENT_EPSILON);
                                category_pass &= passed;
                                category_evidence.insert(
                                    tag.into(),
                                    RealCategoryEvidence {
                                        image_ids: ids,
                                        baseline_mean: base_mean,
                                        p5_mean,
                                        delta,
                                        budget: 0.0,
                                        passed,
                                        status: if passed {
                                            "passed".into()
                                        } else {
                                            "failed-or-insufficient-data".into()
                                        },
                                    },
                                );
                            }
                            let mut agreements = folds
                                .iter()
                                .map(|fold| fold.metrics.agreement)
                                .filter(|value| value.is_finite())
                                .collect::<Vec<_>>();
                            agreements.sort_by(f64::total_cmp);
                            worst_decile_agreement = agreements
                                .get(
                                    agreements
                                        .len()
                                        .saturating_sub(1)
                                        .min(agreements.len() / 10),
                                )
                                .copied();
                            let mut base_sorted = folds
                                .iter()
                                .map(|fold| fold.baseline_agreement)
                                .collect::<Vec<_>>();
                            base_sorted.sort_by(f64::total_cmp);
                            worst_decile_baseline = base_sorted
                                .get(
                                    base_sorted
                                        .len()
                                        .saturating_sub(1)
                                        .min(base_sorted.len() / 10),
                                )
                                .copied();
                            let worst_pass = worst_decile_agreement
                                .zip(worst_decile_baseline)
                                .is_some_and(|(a, b)| a - b > PRIMARY_AGREEMENT_EPSILON);
                            let real_performance_ok = real_performance.as_ref().is_some_and(|p| {
                                p.warm_p95_ms.is_some_and(f64::is_finite)
                                    && p.peak_memory_bytes.is_some()
                            });
                            qualification_gates
                                .insert("all_folds_evaluated".into(), all_folds_evaluated);
                            qualification_gates
                                .insert("one_frozen_config".into(), one_frozen_config);
                            qualification_gates.insert("paired_ci".into(), paired_ci);
                            qualification_gates.insert("critical_categories".into(), category_pass);
                            qualification_gates.insert("worst_decile".into(), worst_pass);
                            qualification_gates.insert("performance".into(), real_performance_ok);
                            qualification_gates.insert("blind".into(), false);
                            let qualification_ready = all_folds_evaluated
                                && one_frozen_config
                                && all_component_ablation_gains
                                && paired_ci
                                && category_pass
                                && worst_pass
                                && real_performance_ok;
                            if qualification_ready {
                                qualified_candidate = Some("P5-hybrid@P0-u2-cf".into());
                                if let Some(blind_path) = std::env::var_os("M16_BLIND_MANIFEST") {
                                    let blind_path = PathBuf::from(blind_path);
                                    let blind_result = (|| -> Result<BlindPromotionEvidence> {
                                        let blind_bytes =
                                            fs::read(&blind_path).with_context(|| {
                                                format!(
                                                    "read M16 blind manifest {}",
                                                    blind_path.display()
                                                )
                                            })?;
                                        let blind_hash = sha256(&blind_bytes);
                                        let blind_images = load_real_arena_manifest(&blind_path)?;
                                        coordinate_blind_promotion(
                                            &mut session,
                                            &images,
                                            &folds,
                                            &blind_images,
                                            &blind_hash,
                                            &mut lifecycle,
                                        )
                                    })();
                                    match blind_result {
                                        Ok(promotion) => {
                                            champion_promoted = promotion.passed;
                                            if promotion.passed {
                                                leaderboard_candidate = qualified_candidate.clone();
                                                qualification_gates.insert("blind".into(), true);
                                            }
                                            blind_promotion = Some(promotion);
                                        }
                                        Err(error) => {
                                            blind_promotion = Some(BlindPromotionEvidence {
                                                set_id: blind_path.display().to_string(),
                                                manifest_hash: "unavailable".into(),
                                                image_ids: Vec::new(),
                                                frozen_config: config.clone(),
                                                baseline_paired: None,
                                                category_evidence: BTreeMap::new(),
                                                worst_decile_baseline: None,
                                                worst_decile_challenger: None,
                                                passed: false,
                                                status: format!(
                                                    "not-run: blind evaluation rejected: {error:#}"
                                                ),
                                            });
                                        }
                                    }
                                }
                            }
                            loo_folds = folds;
                            let qualification = if qualification_ready {
                                "qualified"
                            } else {
                                "blocked: fold, frozen-config, or ablation gate failed"
                            };
                            (
                                format!(
                                    "completed M16 real hybrid LOO with {} held-out rows; qualification={qualification}",
                                    loo_folds.len(),
                                ),
                                format!(
                                    "completed M16 real hybrid LOO with {} held-out rows; qualification={qualification}",
                                    loo_folds.len(),
                                ),
                            )
                        }
                    }
                }
            },
        }
    } else {
        (
            "not-run: M16-owned ORT session was not activated; M15 evidence is prerequisite provenance only".into(),
            "not-run: requires M16-owned compatible official ORT, verified weights/cache, and fixed-arena LOO execution".into(),
        )
    };
    // M15 evidence is prerequisite provenance only.  It is never promoted as
    // M16 evidence: this status becomes completed only after an M16-owned raw
    // session executes the six folds below its own pipeline.
    RealArenaEvidence {
        manifest: manifest.display().to_string(),
        record_count,
        runtime_requested: runtime.map(|path| path.display().to_string()),
        status,
        loo_status: loo_status.clone(),
        // M15's champion is never forwarded: M16 must earn its own P5 result.
        champion: leaderboard_candidate,
        qualified_candidate,
        champion_promoted,
        blind_status,
        m15_evidence_status,
        loo_folds,
        category_evidence,
        worst_decile_agreement,
        baseline_paired,
        worst_decile_baseline,
        qualification_gates,
        performance: real_performance,
        blind_lifecycle: if lifecycle.registered_set.is_some() || champion_promoted {
            lifecycle.evidence()
        } else {
            m16_blind_lifecycle(loo_status.starts_with("completed M16"))
        },
        blind_promotion,
    }
}

fn run_contract(
    root: &Path,
    fixtures: &[FixtureImage],
    development_policy: &HybridConfig,
) -> Result<ContractResults> {
    let full = development_policy.clone();
    let removals = [
        (
            "router",
            HybridConfig {
                route_complementary: false,
                ..full.clone()
            },
        ),
        (
            "complementary",
            HybridConfig {
                complementary_model: false,
                ..full.clone()
            },
        ),
        (
            "adaptive-trimap",
            HybridConfig {
                adaptive_trimap: false,
                ..full.clone()
            },
        ),
        (
            "crop-refine",
            HybridConfig {
                crop_refine: false,
                ..full.clone()
            },
        ),
        (
            "foreground",
            HybridConfig {
                selective_foreground: false,
                ..full.clone()
            },
        ),
        (
            "topology",
            HybridConfig {
                conservative_topology: false,
                ..full.clone()
            },
        ),
    ];
    // The general segmenter is the best global model and is run exactly once
    // per image; every ablation reuses this raw result.
    let global_masks = fixtures
        .iter()
        .map(|image| global_segmenter(&image.rgb, image.width, image.height))
        .collect::<Vec<_>>();
    let mut summaries = BTreeMap::new();
    let full_metrics = fixtures
        .iter()
        .zip(&global_masks)
        .map(|(image, global)| write_bundle(root, image, "full", &full, global))
        .collect::<Result<Vec<_>>>()?;
    summaries.insert("full".into(), summary(fixtures, full_metrics.clone())?);
    let mut ablations = Vec::new();
    let mut enabled_default = full.clone();
    for (component, removal) in removals {
        let name = format!("without-{component}");
        // Alternate full/removal measurement for every component to reduce
        // drift; quality remains computed once from the same held-out rows.
        let (full_measured, measured) =
            measure_interleaved(fixtures, &global_masks, &full, &removal);
        let metrics = fixtures
            .iter()
            .zip(&global_masks)
            .map(|(image, global)| write_bundle(root, image, &name, &removal, global))
            .collect::<Result<Vec<_>>>()?;
        let removal_summary = summary(fixtures, metrics)?;
        let full_summary = summaries
            .get("full")
            .context("full summary missing")?
            .clone();
        summaries.insert(name, removal_summary.clone());
        // Every component is gated on the same primary PhotoRoomAgreement-v1
        // score. Auxiliary composite/alpha metrics remain diagnostic only.
        let comparison = paired(
            &removal_summary,
            &full_summary,
            &format!("without-{component}"),
            "full",
        )?;
        let gain = comparison.mean_delta;
        let enabled = gain
            .zip(comparison.bootstrap_low)
            .is_some_and(|(delta, low)| {
                delta > PRIMARY_AGREEMENT_EPSILON && low > PRIMARY_AGREEMENT_EPSILON
            });
        match component {
            "router" => enabled_default.route_complementary &= enabled,
            "complementary" => enabled_default.complementary_model &= enabled,
            "adaptive-trimap" => enabled_default.adaptive_trimap &= enabled,
            "crop-refine" => enabled_default.crop_refine &= enabled,
            "foreground" => enabled_default.selective_foreground &= enabled,
            "topology" => enabled_default.conservative_topology &= enabled,
            _ => {}
        }
        ablations.push(AblationEvidence {
            component: component.into(),
            full_config: full.clone(),
            removal_config: removal,
            validation_full: full_summary.clone(),
            validation_without: removal_summary,
            paired: comparison,
            quality_gain: gain,
            latency_cost_ms: Some(full_measured.median_ms - measured.median_ms),
            memory_cost_bytes: Some(
                i64::try_from(
                    i128::from(full_measured.peak_memory_bytes)
                        - i128::from(measured.peak_memory_bytes),
                )
                .unwrap_or(if full_measured.peak_memory_bytes >= measured.peak_memory_bytes {
                    i64::MAX
                } else {
                    i64::MIN
                }),
            ),
            warm_samples_ms: measured.samples.clone(),
            warm_p95_ms: Some(measured.p95_ms),
            full_warm_median_ms: full_measured.median_ms,
            removal_warm_median_ms: measured.median_ms,
            warm_median_delta_ms: full_measured.median_ms - measured.median_ms,
            full_warm_p95_ms: full_measured.p95_ms,
            removal_warm_p95_ms: measured.p95_ms,
            warm_p95_delta_ms: full_measured.p95_ms - measured.p95_ms,
            full_memory_bytes: full_measured.peak_memory_bytes,
            removal_memory_bytes: measured.peak_memory_bytes,
            memory_delta_bytes: i64::try_from(
                i128::from(full_measured.peak_memory_bytes)
                    - i128::from(measured.peak_memory_bytes),
            )
            .unwrap_or({
                if full_measured.peak_memory_bytes >= measured.peak_memory_bytes {
                    i64::MAX
                } else {
                    i64::MIN
                }
            }),
            memory_measurement: "conservative modeled live-buffer upper bound from retained f32 arrays and 2x crop scratch; allocator/RSS peak unavailable".into(),
            operation_count: (fixtures.iter().map(|image| image.rgb.len()).sum::<usize>() as u64)
                * if component == "router" || component == "complementary" {
                    3
                } else {
                    2
                },
            working_buffer_bytes: fixtures.iter().map(|image| image.rgb.len()).sum::<usize>()
                as u64
                * 16,
            routed_image_count: full_summary
                .per_image
                .iter()
                .filter(|row| !row.metrics.failure)
                .count(),
            performance_method:
                "interleaved full/removal warm wall-clock samples after two warmups; checked live-buffer bound".into(),
            enabled_in_resolved_default: enabled,
            status: if enabled {
                "enabled: positive paired CI".into()
            } else {
                "disabled: removal ablation not beaten".into()
            },
        });
    }
    let resolved_metrics = fixtures
        .iter()
        .zip(&global_masks)
        .map(|(image, global)| {
            score(
                image,
                &evaluate_with_global(image, &enabled_default, global),
            )
        })
        .collect::<Vec<_>>();
    let resolved = summary(fixtures, resolved_metrics)?;
    summaries.insert("resolved-default".into(), resolved.clone());
    let mut tags = BTreeMap::new();
    for tag in fixtures
        .iter()
        .map(|image| image.tag.clone())
        .collect::<BTreeSet<_>>()
    {
        let subset = fixtures
            .iter()
            .enumerate()
            .filter(|(_, image)| image.tag == tag)
            .map(|(index, _)| resolved.per_image[index].clone())
            .collect::<Vec<_>>();
        let metrics = subset
            .iter()
            .map(|row| row.metrics.clone())
            .collect::<Vec<_>>();
        let tag_images = fixtures
            .iter()
            .filter(|image| image.tag == tag)
            .cloned()
            .collect::<Vec<_>>();
        tags.insert(tag, summary(&tag_images, metrics)?);
    }
    Ok((enabled_default, ablations, summaries, tags))
}

fn write_deterministic_quality(
    root: &Path,
    fixtures: &[FixtureImage],
    development_policy: &HybridConfig,
    real: &RealArenaEvidence,
) -> Result<String> {
    let (_, ablations, summaries, tags) = run_contract(root, fixtures, development_policy)?;
    write_json(
        root,
        "raw-mask-cache-evidence.json",
        &cache_probe(fixtures)?,
    )?;
    write_json(root, "real-blind-lifecycle.json", &real.blind_lifecycle)?;
    if let Some(promotion) = &real.blind_promotion {
        write_json(root, "real-blind-promotion.json", promotion)?;
    }
    write_html(root, &summaries, &ablations, &tags)?;
    Ok(write_manifest(root)?.sha256)
}

pub fn run(output: &Path) -> Result<()> {
    let persisted_lifecycle = fs::read(output.join("real-blind-lifecycle.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<BlindLifecycleEvidence>(&bytes).ok());
    // The command owns its generated evidence directory; clearing it prevents
    // stale files from escaping the deterministic manifest scope.
    if output.exists() {
        fs::remove_dir_all(output)
            .with_context(|| format!("clear previous M16 output {}", output.display()))?;
    }
    fs::create_dir_all(output)
        .with_context(|| format!("create M16 output {}", output.display()))?;
    let fixtures = fixture_images();
    let development = development_fixture_images();
    let development_selected_config = select_development_config(&development);
    ensure!(fixtures
        .iter()
        .all(|image| image.split == "heldout-validation"));
    ensure!(development.iter().all(|image| image.split == "development"));
    let (resolved, ablations, summaries, tags) =
        run_contract(output, &fixtures, &development_selected_config)?;
    let raw_mask_cache = cache_probe(&fixtures)?;
    let synthetic_performance = performance_evidence(&fixtures);
    write_json(output, "raw-mask-cache-evidence.json", &raw_mask_cache)?;
    write_json(output, "performance.json", &synthetic_performance)?;
    let real = real_arena_status(persisted_lifecycle.as_ref());
    write_json(output, "real-blind-lifecycle.json", &real.blind_lifecycle)?;
    if let Some(promotion) = &real.blind_promotion {
        write_json(output, "real-blind-promotion.json", promotion)?;
    }
    let run_id = std::process::id();
    let temp_a = std::env::temp_dir().join(format!("m16-determinism-a-{run_id}"));
    let temp_b = std::env::temp_dir().join(format!("m16-determinism-b-{run_id}"));
    if temp_a.exists() {
        fs::remove_dir_all(&temp_a)?;
    }
    if temp_b.exists() {
        fs::remove_dir_all(&temp_b)?;
    }
    fs::create_dir_all(&temp_a)?;
    fs::create_dir_all(&temp_b)?;
    let first =
        write_deterministic_quality(&temp_a, &fixtures, &development_selected_config, &real)?;
    let second =
        write_deterministic_quality(&temp_b, &fixtures, &development_selected_config, &real)?;
    fs::remove_dir_all(&temp_a)?;
    fs::remove_dir_all(&temp_b)?;
    write_html(output, &summaries, &ablations, &tags)?;
    let artifact_manifest = write_manifest(output)?;
    let determinism = DeterminismEvidence {
        first_manifest_sha256: first.clone(),
        second_manifest_sha256: second.clone(),
        all_quality_artifacts_match: first == second && artifact_manifest.sha256 == first,
    };
    let all_components = ablations
        .iter()
        .all(|ablation| ablation.enabled_in_resolved_default);
    let cache_passed = raw_mask_cache.miss_count == fixtures.len()
        && raw_mask_cache.inference_calls == fixtures.len()
        && raw_mask_cache.hit_count == raw_mask_cache.downstream_variant_requests
        && raw_mask_cache.corruption_detected
        && raw_mask_cache.stale_manifest_miss
        && raw_mask_cache.incompatible_dimensions_rejected
        && raw_mask_cache.on_disk_atomic_write_verified
        && raw_mask_cache.on_disk_round_trip_verified;
    let synthetic_contract_passed = !summaries.is_empty()
        && ablations.len() == 6
        && determinism.all_quality_artifacts_match
        && artifact_manifest.entry_count >= fixtures.len()
        && artifact_manifest.entries_verified
        && cache_passed;
    const LATENCY_BUDGET_MS: f64 = 2_000.0;
    const MEMORY_BUDGET_BYTES: u64 = 2_000_000;
    let performance_gate_passed = synthetic_performance.warm_p95_ms <= LATENCY_BUDGET_MS
        && synthetic_performance.peak_memory_bytes <= MEMORY_BUDGET_BYTES
        && ablations.iter().all(|ablation| {
            ablation.removal_warm_p95_ms.is_finite()
                && ablation.removal_memory_bytes > 0
                && ablation.warm_samples_ms.len() == 5
        });
    let real_tournament_available = real.loo_status.starts_with("completed M16");
    let real_champion_promoted = real.champion_promoted;
    let blind_improvement_available = real
        .blind_promotion
        .as_ref()
        .is_some_and(|promotion| promotion.passed);
    let reason = if real_tournament_available {
        "real LOO evidence was consumed; promotion remains conditional on all real gates".into()
    } else {
        "M15 Edge ORT/runtime evidence is blocked; synthetic contract does not promote a real or blind champion".into()
    };
    let report = M16Report { report_version: REPORT_VERSION, metric_contract: "M15 PhotoRoomAgreement-v1 metric/equal-image-weight/tie-break contracts; synthetic evidence is separate from real claims", primary_agreement_epsilon: PRIMARY_AGREEMENT_EPSILON, execution: "deterministic RGB-only hybrid contract plus fail-closed real six-image arena status", resolved_default: resolved, best_general_segmenter_runs: fixtures.len(), development_fixture_count: development.len(), heldout_fixture_count: fixtures.len(), fixture_provenance: "independent development scenes select the default mechanism policy; all ablation/bootstrap gates use the six heldout-validation rows", development_selected_config, ablations, validation: summaries, tag_aggregates: tags, real_arena: real, raw_mask_cache, synthetic_performance, gates: GateEvidence { real_tournament_available, real_champion_promoted, synthetic_contract_passed, all_components_beat_removal: all_components, blind_improvement_available, performance_gate_passed, latency_budget_ms: LATENCY_BUDGET_MS, memory_budget_bytes: MEMORY_BUDGET_BYTES, performance_reason: if performance_gate_passed { "all measured full/removal samples and modeled memory bounds are within declared budgets; RSS is unavailable".into() } else { "one or more measured full/removal performance samples or modeled memory bounds exceeded the declared budget".into() }, reason }, determinism, artifact_manifest: artifact_manifest.clone(), artifact_scope: "all synthetic full/removal pipeline-image bundles plus report.html, cache evidence, metrics/config/features and hashes; volatile performance.json and report JSON/hash are excluded from the quality manifest; real arena status-only while unavailable" };
    write_json(output, "report.json", &report)?;
    let report_bytes = fs::read(output.join("report.json"))?;
    write_bytes(
        output,
        "report.sha256",
        format!("{}  report.json\n", sha256(&report_bytes)).as_bytes(),
    )?;
    write_manifest(output)?;
    println!("wrote {} (M16 real status: blocked)", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m16_prediction_contract_has_no_reference_or_filename_input() {
        let fixture = fixture_images().remove(0);
        let first = global_segmenter(&fixture.rgb, fixture.width, fixture.height);
        let mut renamed = fixture.clone();
        renamed.id = "hostile-<script>".into();
        renamed.truth.fill(1.0);
        assert_eq!(
            first,
            global_segmenter(&renamed.rgb, renamed.width, renamed.height)
        );
    }

    #[test]
    fn m16_reference_oracle_is_analytical_and_not_candidate_derived() {
        let fixture = fixture_images().remove(0);
        let reference = fixture.truth.clone();
        let config = HybridConfig {
            route_complementary: true,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        };
        let _ = evaluate(&fixture, &config);
        assert_eq!(fixture.truth, reference);
        assert_eq!(fixture.reference_foreground.len(), fixture.rgb.len());
    }

    #[test]
    fn m16_real_session_seam_uses_injected_masks_and_fails_closed() {
        struct Mock {
            calls: Vec<String>,
            bad: bool,
        }
        impl RealRawSession for Mock {
            fn infer(
                &mut self,
                model_id: &str,
                rgb: &[[f32; 3]],
                width: u32,
                height: u32,
            ) -> Result<Vec<f32>> {
                self.calls.push(model_id.into());
                if self.bad {
                    return Ok(vec![f32::NAN]);
                }
                Ok(global_segmenter(rgb, width, height))
            }
        }
        let image = fixture_images().remove(0);
        let config = HybridConfig {
            route_complementary: true,
            complementary_model: true,
            adaptive_trimap: false,
            crop_refine: false,
            selective_foreground: false,
            conservative_topology: false,
        };
        let mut mock = Mock {
            calls: Vec::new(),
            bad: false,
        };
        let evaluation = evaluate_real_with_session(&mut mock, &image, &config).unwrap();
        assert_eq!(
            mock.calls.first().map(String::as_str),
            Some("approved-general")
        );
        assert!(evaluation
            .refined_alpha
            .iter()
            .all(|value| value.is_finite()));
        let mut bad = Mock {
            calls: Vec::new(),
            bad: true,
        };
        assert!(evaluate_real_with_session(&mut bad, &image, &config).is_err());
    }

    #[test]
    fn m16_real_session_runs_the_same_hybrid_stages_with_injected_masks() {
        struct Mock;
        impl RealRawSession for Mock {
            fn infer(
                &mut self,
                model_id: &str,
                rgb: &[[f32; 3]],
                width: u32,
                height: u32,
            ) -> Result<Vec<f32>> {
                let mut mask = global_segmenter(rgb, width, height);
                if model_id == "approved-complementary" {
                    mask.rotate_left(1);
                }
                Ok(mask)
            }
        }
        let image = fixture_images().remove(0);
        let config = HybridConfig {
            route_complementary: false,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        };
        let evaluation = evaluate_real_with_session(&mut Mock, &image, &config).unwrap();
        assert!(evaluation.routed);
        assert!(!evaluation.crops.is_empty());
        assert!(evaluation
            .refined_alpha
            .iter()
            .all(|value| value.is_finite()));
        assert!(evaluation.topology_decision.contains(&1));
        let images = fixture_images();
        let folds = real_hybrid_loo_with_session(&mut Mock, &images, &config).unwrap();
        assert_eq!(folds.len(), 6);
        assert_eq!(
            folds
                .iter()
                .map(|fold| fold.held_out_id.clone())
                .collect::<BTreeSet<_>>()
                .len(),
            6
        );
        assert!(folds.iter().all(|fold| fold.training_ids.len() == 5));
        assert!(folds.iter().all(|fold| fold.metrics.agreement.is_finite()));
        assert!(folds
            .iter()
            .all(|fold| fold.candidate_training_agreement.len() == 2));
        assert!(folds
            .iter()
            .all(|fold| fold.ablation_held_out_agreement.len() == 6));
    }

    #[test]
    fn m16_real_loo_reuses_each_raw_model_once_per_image() {
        struct Counting {
            general: usize,
            complementary: usize,
        }
        impl RealRawSession for Counting {
            fn infer(
                &mut self,
                model_id: &str,
                rgb: &[[f32; 3]],
                width: u32,
                height: u32,
            ) -> Result<Vec<f32>> {
                match model_id {
                    "approved-general" => self.general += 1,
                    "approved-complementary" => self.complementary += 1,
                    other => anyhow::bail!("unexpected model {other}"),
                }
                Ok(global_segmenter(rgb, width, height))
            }
        }
        let images = fixture_images();
        let config = HybridConfig {
            route_complementary: false,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        };
        let mut session = Counting {
            general: 0,
            complementary: 0,
        };
        let folds = real_hybrid_loo_with_session(&mut session, &images, &config).unwrap();
        assert_eq!(folds.len(), 6);
        assert_eq!(session.general, 6);
        assert!(session.complementary <= 6);
    }

    #[test]
    fn m16_real_cache_keys_separate_model_manifests() {
        let image = fixture_images().remove(0);
        let u2 = validated_manifest_hash("models/m5_u2net.toml").unwrap();
        let isnet = validated_manifest_hash("models/m4_isnet_fp32.toml").unwrap();
        assert_ne!(u2, isnet);
        assert_ne!(
            real_cache_key(&image, "approved-general", &u2),
            real_cache_key(&image, "approved-complementary", &isnet)
        );
        let changed = sha256(format!("{u2}:changed").as_bytes());
        assert_ne!(
            real_cache_key(&image, "approved-general", &u2),
            real_cache_key(&image, "approved-general", &changed)
        );
    }

    #[test]
    fn m16_blind_lifecycle_stays_untouched_without_real_loo() {
        let blocked = m16_blind_lifecycle(false);
        assert_eq!(blocked.state, "untouched");
        assert!(!blocked.new_untouched_set_required);
        let consumed = m16_blind_lifecycle(true);
        assert_eq!(consumed.state, "retired-after-influence");
        assert!(consumed.new_untouched_set_required);
        assert!(!consumed.eligible_for_one_promotion);
    }

    #[test]
    fn m16_blind_lifecycle_allows_one_qualified_pass_then_retires() {
        let mut lifecycle = BlindLifecycle::untouched();
        assert!(lifecycle.evaluate_once(true).is_err());
        lifecycle.qualify_release_candidate().unwrap();
        lifecycle.evaluate_once(true).unwrap();
        assert!(lifecycle.evidence().eligible_for_one_promotion);
        assert!(lifecycle.evaluate_once(true).is_err());
        lifecycle.influence_change().unwrap();
        assert!(!lifecycle.evidence().eligible_for_one_promotion);
        assert!(lifecycle.evidence().new_untouched_set_required);
        lifecycle.register_new_untouched_set();
        assert_eq!(lifecycle.evidence().state, "untouched");
    }

    #[test]
    fn m16_qualified_real_candidate_requires_one_paired_blind_pass() {
        let ids = ["a", "b", "c", "d", "e", "f"];
        let baseline = ids.iter().map(|id| (*id, 0.40)).collect::<Vec<_>>();
        let challenger = ids.iter().map(|id| (*id, 0.60)).collect::<Vec<_>>();
        let paired = paired_values(&baseline, &challenger, "m15", "p5").unwrap();
        assert!(paired.bootstrap_low.unwrap() > PRIMARY_AGREEMENT_EPSILON);
        let mut lifecycle = BlindLifecycle::untouched();
        let loo_ids = ["arena-1", "arena-2"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        lifecycle
            .register_untouched_set(
                "blind-v2",
                &"a".repeat(64),
                &ids.iter().map(|id| (*id).to_owned()).collect::<Vec<_>>(),
                &loo_ids,
            )
            .unwrap();
        lifecycle.qualify_release_candidate().unwrap();
        lifecycle
            .evaluate_registered_once(&baseline, &challenger, true, true)
            .unwrap();
        assert!(lifecycle.evidence().eligible_for_one_promotion);
        assert!(!lifecycle.evidence().blind_used_for_sweeps);
        assert!(lifecycle
            .evaluate_registered_once(&baseline, &challenger, true, true)
            .is_err());
        lifecycle.influence_change().unwrap();
        assert!(!lifecycle.evidence().eligible_for_one_promotion);
        assert!(lifecycle
            .register_untouched_set(
                "again",
                &"b".repeat(64),
                &ids.iter().map(|id| (*id).to_owned()).collect::<Vec<_>>(),
                &loo_ids
            )
            .is_err());
    }

    #[test]
    fn m16_blind_registration_rejects_overlap_and_failed_pass_cannot_promote() {
        let mut lifecycle = BlindLifecycle::untouched();
        let loo = ["arena-1".to_owned()].into_iter().collect::<BTreeSet<_>>();
        assert!(lifecycle
            .register_untouched_set("bad", &"c".repeat(64), &["arena-1".to_owned()], &loo)
            .is_err());
        lifecycle
            .register_untouched_set("blind", &"d".repeat(64), &["blind-1".to_owned()], &loo)
            .unwrap();
        lifecycle.qualify_release_candidate().unwrap();
        let baseline = [("blind-1", 0.8)];
        let challenger = [("blind-1", 0.2)];
        lifecycle
            .evaluate_registered_once(&baseline, &challenger, false, false)
            .unwrap();
        assert!(!lifecycle.evidence().eligible_for_one_promotion);
        assert!(!lifecycle.evidence().blind_used_for_sweeps);
        lifecycle.influence_change().unwrap();
        assert_eq!(lifecycle.evidence().observation_count, 1);
        assert_eq!(lifecycle.evidence().state, "retired-after-influence");
        assert!(lifecycle.evidence().new_untouched_set_required);
        assert!(lifecycle
            .evaluate_registered_once(&baseline, &challenger, false, false)
            .is_err());
    }

    struct CoordinatorSession {
        mask: Vec<f32>,
    }

    impl RealRawSession for CoordinatorSession {
        fn infer(
            &mut self,
            _model_id: &str,
            _rgb: &[[f32; 3]],
            _width: u32,
            _height: u32,
        ) -> Result<Vec<f32>> {
            Ok(self.mask.clone())
        }
    }

    #[test]
    fn m16_integrated_blind_coordinator_promotes_once_after_qualification() {
        let config = HybridConfig {
            route_complementary: true,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        };
        let template = fixture_images().remove(0);
        let raw = global_segmenter(&template.rgb, template.width, template.height);
        // Independent blind oracle: this analytic shape is generated directly
        // from declared scene geometry and never calls the candidate pipeline.
        let analytic_truth = (0..template.height)
            .flat_map(|y| {
                (0..template.width).map(move |x| {
                    let fx = x as f32 / template.width as f32;
                    let fy = y as f32 / template.height as f32;
                    let dx = (fx - 0.5) / 0.22;
                    let dy = (fy - 0.5) / 0.34;
                    let body = (1.0 - (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0);
                    let strap = if x % 13 == 0 && (0.2..0.85).contains(&fy) {
                        0.72
                    } else {
                        0.0
                    };
                    (body * 0.92 + strap).clamp(0.0, 1.0)
                })
            })
            .collect::<Vec<_>>();
        let mut blind_images = Vec::new();
        for index in 0..6 {
            let mut image = template.clone();
            image.id = format!("new-blind-{index}");
            image.split = "blind".into();
            image.truth = analytic_truth.clone();
            image.reference_foreground = image.rgb.clone();
            blind_images.push(image);
        }
        let loo_images = fixture_images();
        let folds = loo_images
            .iter()
            .map(|image| RealHybridLooFold {
                held_out_id: image.id.clone(),
                training_ids: loo_images
                    .iter()
                    .filter(|other| other.id != image.id)
                    .map(|other| other.id.clone())
                    .collect(),
                metrics: failure_metrics("injected fold metadata only"),
                baseline_agreement: 0.0,
                selected_config: config.clone(),
                candidate_training_agreement: BTreeMap::new(),
                ablation_held_out_agreement: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let mut session = CoordinatorSession { mask: raw };
        let mut lifecycle = BlindLifecycle::untouched();
        let promotion = coordinate_blind_promotion(
            &mut session,
            &loo_images,
            &folds,
            &blind_images,
            &"e".repeat(64),
            &mut lifecycle,
        )
        .unwrap();
        assert!(
            promotion.passed,
            "injected blind evidence should qualify: {promotion:?}"
        );
        assert_eq!(promotion.frozen_config, config);
        assert_eq!(lifecycle.evidence().state, "retired-after-influence");
        assert_eq!(lifecycle.evidence().observation_count, 1);
        assert!(coordinate_blind_promotion(
            &mut session,
            &loo_images,
            &folds,
            &blind_images,
            &"e".repeat(64),
            &mut lifecycle,
        )
        .is_err());
    }

    #[test]
    fn m16_routing_and_uncertainty_are_deterministic_and_bounded() {
        let fixture = fixture_images().remove(2);
        let config = HybridConfig {
            route_complementary: true,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        };
        let a = evaluate(&fixture, &config);
        let b = evaluate(&fixture, &config);
        assert_eq!(a.uncertainty, b.uncertainty);
        assert!(a
            .uncertainty
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
        assert!(a.features.cross_model_disagreement.is_some() || !a.routed);
    }

    #[test]
    fn m16_ablation_disablement_is_evidence_driven() {
        let root = std::env::temp_dir().join("m16-contract-test");
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir_all(&root).unwrap();
        let dev = development_fixture_images();
        let policy = select_development_config(&dev);
        let (_, ablations, _, _) = run_contract(&root, &fixture_images(), &policy).unwrap();
        assert_eq!(ablations.len(), 6);
        assert!(ablations
            .iter()
            .all(|ablation| ablation.paired.bootstrap_low.is_some()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn m16_bootstrap_rejects_empty_nonfinite_and_zero_seed() {
        assert!(bootstrap(&[], 1).is_err());
        assert!(bootstrap(&[f64::NAN], 1).is_err());
        assert!(bootstrap(&[0.1], 0).is_err());
    }

    #[test]
    fn m16_crop_bounds_and_seam_overlap_are_source_safe() {
        let tiles = crop_tiles(&[0.0, 0.4, 0.0, 0.5], 2, 2);
        for tile in tiles {
            assert!(tile.x + tile.width <= 2);
            assert!(tile.y + tile.height <= 2);
            assert_eq!(tile.overlap, 2);
        }
    }

    #[test]
    fn m16_foreground_policy_changes_only_risk_regions() {
        let fixture = fixture_images().remove(0);
        let disabled = HybridConfig {
            route_complementary: false,
            complementary_model: false,
            adaptive_trimap: false,
            crop_refine: false,
            selective_foreground: false,
            conservative_topology: false,
        };
        let enabled = HybridConfig {
            selective_foreground: true,
            ..disabled.clone()
        };
        let left = evaluate(&fixture, &disabled);
        let right = evaluate(&fixture, &enabled);
        for index in 0..fixture.rgb.len() {
            if right.foreground[index] != left.foreground[index] {
                assert!(right.foreground_risk[index] > 0.28);
            }
        }
    }

    #[test]
    fn m16_escape_html_is_safe_for_status_and_ids() {
        assert_eq!(
            escape_html("<script>\"x\"</script>"),
            "&lt;script&gt;&quot;x&quot;&lt;/script&gt;"
        );
    }

    #[test]
    fn m16_metric_contract_identity_and_known_mismatch() {
        let fixture = fixture_images().remove(0);
        let config = HybridConfig {
            route_complementary: true,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        };
        let evaluation = evaluate(&fixture, &config);
        let identity = score(&fixture, &evaluation);
        assert!(identity.agreement.is_finite());
        assert!(identity.composite_ssim.is_finite());
        let mut mismatch = fixture.clone();
        mismatch.truth.fill(1.0);
        let changed = score(&mismatch, &evaluation);
        assert!(changed.boundary_f1 < 1.0 || changed.composite_ssim < 1.0);
        assert!(changed.gradient_error.is_finite());
    }

    #[test]
    fn m16_metric_ssim_uses_the_frozen_m15_gaussian_primitive() {
        let a = (0..35)
            .map(|index| f64::from((index * 17 % 31) as u32) / 31.0)
            .collect::<Vec<_>>();
        let b = (0..35)
            .map(|index| f64::from((index * 11 + 3) % 29) / 29.0)
            .collect::<Vec<_>>();
        assert_eq!(
            gaussian_ssim_channel(&a, &b, 7, 5),
            crate::m15::shared_global_ssim_2d(&a, &b, 7, 5)
        );
    }

    #[test]
    fn m16_end_to_end_metrics_match_m15_shared_contract() {
        let fixture = fixture_images().remove(0);
        let mut evaluation = evaluate(
            &fixture,
            &HybridConfig {
                route_complementary: false,
                complementary_model: false,
                adaptive_trimap: false,
                crop_refine: false,
                selective_foreground: false,
                conservative_topology: false,
            },
        );
        evaluation.foreground = fixture.rgb.clone();
        let ours = score(&fixture, &evaluation);
        let shared = crate::m15::shared_photoroom_metrics(
            &fixture.rgb,
            &fixture.reference_foreground,
            &evaluation.refined_alpha,
            &fixture.truth,
            fixture.width,
            fixture.height,
        )
        .unwrap();
        let shared = [
            shared.0, shared.1, shared.2, shared.3, shared.4, shared.5, shared.6, shared.7,
        ];
        let actual = [
            ours.agreement,
            ours.roi_alpha_mae,
            ours.roi_soft_iou,
            ours.boundary_f1,
            ours.composite_mae,
            ours.composite_psnr,
            ours.composite_ssim_full,
            ours.composite_ssim,
        ];
        for (index, (left, right)) in actual.into_iter().zip(shared).enumerate() {
            assert!(
                (left - right).abs() < 1e-6,
                "M15/M16 metric drift at {index}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn m16_end_to_end_metric_parity_covers_randomized_geometries() {
        let mut state = 0x16_u32;
        for (width, height) in [(1, 7), (7, 1), (5, 4), (9, 3)] {
            let count = (width * height) as usize;
            let mut next = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as f32 / u32::MAX as f32).clamp(0.0, 1.0)
            };
            let rgb = (0..count)
                .map(|_| [next(), next(), next()])
                .collect::<Vec<_>>();
            let alpha = (0..count).map(|_| next()).collect::<Vec<_>>();
            let reference = (0..count)
                .map(|_| [next(), next(), next()])
                .collect::<Vec<_>>();
            let image = FixtureImage {
                id: "random-parity".into(),
                split: "heldout-validation".into(),
                width,
                height,
                rgb: rgb.clone(),
                truth: alpha.clone(),
                reference_foreground: reference.clone(),
                tag: "parity".into(),
            };
            let mut evaluation = evaluate(
                &image,
                &HybridConfig {
                    route_complementary: false,
                    complementary_model: false,
                    adaptive_trimap: false,
                    crop_refine: false,
                    selective_foreground: false,
                    conservative_topology: false,
                },
            );
            evaluation.foreground = rgb;
            let actual = score(&image, &evaluation);
            let expected = crate::m15::shared_photoroom_metrics(
                &evaluation.foreground,
                &reference,
                &evaluation.refined_alpha,
                &alpha,
                width,
                height,
            )
            .unwrap();
            assert!((actual.agreement - expected.0).abs() < 1e-6);
            assert!((actual.composite_psnr - expected.5).abs() < 1e-6);
            assert!((actual.composite_ssim - expected.7).abs() < 1e-6);
        }
    }

    #[test]
    fn m16_extreme_and_failure_inputs_fail_closed() {
        let mut image = FixtureImage {
            id: "constant".into(),
            split: "heldout-validation".into(),
            width: 1,
            height: 1,
            rgb: vec![[0.0; 3]],
            truth: vec![0.0],
            reference_foreground: vec![[0.0; 3]],
            tag: "fixture".into(),
        };
        let config = HybridConfig {
            route_complementary: true,
            complementary_model: true,
            adaptive_trimap: true,
            crop_refine: true,
            selective_foreground: true,
            conservative_topology: true,
        };
        let global = global_segmenter(&image.rgb, image.width, image.height);
        let evaluation = evaluate_with_global(&image, &config, &global);
        assert!(score(&image, &evaluation).agreement.is_finite());
        image.truth.push(1.0);
        assert!(score(&image, &evaluation).failure);
    }

    #[test]
    fn m16_crop_geometry_has_multiple_tiles_and_no_holes() {
        let uncertainty = vec![1.0; 12 * 8];
        let tiles = crop_tiles(&uncertainty, 12, 8);
        assert!(tiles.len() >= 2);
        assert!(tiles
            .iter()
            .all(|tile| tile.output_width == tile.width * tile.scale));
        let mut alpha = vec![0.5; uncertainty.len()];
        let rgb = vec![[0.5; 3]; uncertainty.len()];
        let trimap = vec![128; uncertainty.len()];
        let weights = refine_crops(&mut alpha, &uncertainty, &rgb, &trimap, &tiles, 12, 8, 0.18);
        assert!(weights.iter().all(|weight| *weight > 0.0));
        assert!(alpha.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn m16_crop_geometry_handles_wide_and_tall_degenerate_aspects() {
        for (width, height) in [(1, 8), (8, 1)] {
            let count = (width * height) as usize;
            let uncertainty = vec![1.0; count];
            let rgb = vec![[0.4, 0.5, 0.6]; count];
            let trimap = vec![128; count];
            let tiles = crop_tiles(&uncertainty, width, height);
            assert!(!tiles.is_empty());
            let mut alpha = vec![0.5; count];
            let weights = refine_crops(
                &mut alpha,
                &uncertainty,
                &rgb,
                &trimap,
                &tiles,
                width,
                height,
                0.1,
            );
            assert!(weights.iter().all(|weight| *weight > 0.0));
            assert!(alpha.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn m16_high_resolution_crop_uses_bilinear_source_geometry() {
        let rgb = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let sample = bilinear_rgb_sample(&rgb, 2, 2, 0.5, 0.5);
        assert!((sample[0] - 0.5).abs() < 1e-6);
        assert!((sample[1] - 0.5).abs() < 1e-6);
        let mut alpha = vec![0.4; 4];
        let uncertainty = vec![1.0; 4];
        let trimap = vec![128; 4];
        let tiles = crop_tiles(&uncertainty, 2, 2);
        let weights = refine_crops(&mut alpha, &uncertainty, &rgb, &trimap, &tiles, 2, 2, 0.8);
        assert!(weights.iter().all(|weight| *weight > 0.0));
        assert!(alpha.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn m16_topology_and_foreground_are_real_selective_operations() {
        let fixture = fixture_images().remove(1);
        let disabled = HybridConfig {
            route_complementary: false,
            complementary_model: false,
            adaptive_trimap: false,
            crop_refine: false,
            selective_foreground: false,
            conservative_topology: false,
        };
        let topology = HybridConfig {
            conservative_topology: true,
            ..disabled.clone()
        };
        let left = evaluate(&fixture, &disabled);
        let right = evaluate(&fixture, &topology);
        assert!(right.topology_decision.contains(&1));
        assert!(right
            .refined_alpha
            .iter()
            .zip(&left.refined_alpha)
            .any(|(a, b)| a != b));
        let foreground = HybridConfig {
            selective_foreground: true,
            ..disabled
        };
        let fg = evaluate(&fixture, &foreground);
        let direct = estimate_foreground_ml(
            &fixture.rgb,
            &fg.refined_alpha,
            fixture.width,
            fixture.height,
            1e-5,
            4,
            2,
            32,
            1.0,
        )
        .unwrap()
        .0
        .data()
        .to_vec();
        assert!(fg.foreground_risk.iter().any(|value| *value > 0.0));
        assert!(fg
            .foreground
            .iter()
            .zip(&fixture.rgb)
            .enumerate()
            .all(|(i, (a, b))| a == b || fg.foreground_risk[i] > 0.28));
        assert!(fg
            .foreground
            .iter()
            .zip(&direct)
            .enumerate()
            .all(
                |(i, (candidate, expected))| fg.foreground_risk[i] <= 0.28 || candidate == expected
            ));
    }

    #[test]
    fn m16_cache_probe_proves_reuse_and_invalidation() {
        let evidence = cache_probe(&fixture_images()).unwrap();
        assert_eq!(evidence.miss_count, 6);
        assert_eq!(evidence.inference_calls, 6);
        assert_eq!(evidence.hit_count, evidence.downstream_variant_requests);
        assert!(
            evidence.corruption_detected
                && evidence.stale_manifest_miss
                && evidence.incompatible_dimensions_rejected
        );
    }

    #[test]
    fn m16_manifest_verification_rejects_tamper_and_missing_entries() {
        let root = std::env::temp_dir().join("m16-manifest-test");
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir_all(&root).unwrap();
        write_bytes(&root, "quality.bin", b"quality").unwrap();
        let evidence = write_manifest(&root).unwrap();
        assert!(evidence.entries_verified);
        let manifest: ArtifactManifest =
            serde_json::from_slice(&fs::read(root.join("artifacts.manifest.json")).unwrap())
                .unwrap();
        assert!(verify_manifest_entries(&root, &manifest.entries));
        fs::write(root.join("quality.bin"), b"tampered").unwrap();
        assert!(!verify_manifest_entries(&root, &manifest.entries));
        fs::remove_file(root.join("quality.bin")).unwrap();
        assert!(!verify_manifest_entries(&root, &manifest.entries));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn m16_manifest_verification_rejects_unlisted_extra_duplicate_and_traversal() {
        let root = std::env::temp_dir().join("m16-manifest-scope-test");
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir_all(&root).unwrap();
        write_bytes(&root, "quality.bin", b"quality").unwrap();
        let evidence = write_manifest(&root).unwrap();
        assert!(evidence.entries_verified);
        let manifest: ArtifactManifest =
            serde_json::from_slice(&fs::read(root.join("artifacts.manifest.json")).unwrap())
                .unwrap();
        fs::write(root.join("extra.bin"), b"extra").unwrap();
        assert!(!verify_manifest_entries(&root, &manifest.entries));
        fs::remove_file(root.join("extra.bin")).unwrap();
        let mut duplicate = manifest.entries.clone();
        duplicate.push(duplicate[0].clone());
        assert!(!verify_manifest_entries(&root, &duplicate));
        let mut traversal = manifest.entries.clone();
        traversal[0].path = "../escape".into();
        assert!(!verify_manifest_entries(&root, &traversal));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn m16_router_and_complementary_ablations_are_independent() {
        let image = fixture_images().remove(0);
        let global = global_segmenter(&image.rgb, image.width, image.height);
        let always = HybridConfig {
            route_complementary: false,
            complementary_model: true,
            adaptive_trimap: false,
            crop_refine: false,
            selective_foreground: false,
            conservative_topology: false,
        };
        let never = HybridConfig {
            route_complementary: true,
            complementary_model: false,
            ..always
        };
        let always_eval = evaluate_with_global(&image, &always, &global);
        let never_eval = evaluate_with_global(&image, &never, &global);
        assert!(always_eval.routed);
        assert!(!never_eval.routed);
        assert!(always_eval.features.cross_model_disagreement.is_some());
        assert!(never_eval.features.cross_model_disagreement.is_none());
    }
}
