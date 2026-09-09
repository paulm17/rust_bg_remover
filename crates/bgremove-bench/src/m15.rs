//! M15 staged tournament and deterministic parameter-search harness.

use anyhow::{ensure, Context, Result};
use bgremove_color::{
    FastForegroundEstimator, FbaForegroundEstimator, MultilevelForegroundEstimator,
    OriginalRgbEstimator,
};
use bgremove_core::io::{encode_mask_png, encode_straight_rgba_png, load_canonical};
use bgremove_core::ForegroundEstimator;
use bgremove_models::{parse_toml, ModelManifest};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const REPORT_VERSION: &str = "m15.tournament.v1";
const METRIC_VERSION: &str = "PhotoRoomAgreement-v1";
const BOOTSTRAP_RESAMPLES: usize = 1024;
const CACHE_MAGIC: &[u8] = b"M15RAW\0";

#[derive(Debug, Clone, Serialize)]
struct CandidateRegistration {
    id: String,
    coarse_stage: String,
    refiner: String,
    foreground: String,
    availability: &'static str,
    executable_mode: &'static str,
    eligible_for_real_champion: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum Split {
    Tune,
    Validation,
    Blind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvaluationMode {
    SyntheticContract,
    RealMechanisms,
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

#[derive(Debug, Clone)]
struct SyntheticImage {
    id: String,
    split: Split,
    width: u32,
    height: u32,
    rgb: Vec<[f32; 3]>,
    reference_rgb: Vec<[f32; 3]>,
    truth: Vec<f32>,
    input_hash: String,
    tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct StageSummary {
    stage: &'static str,
    evaluated_candidates: usize,
    survivor_ids: Vec<String>,
    tune_image_count: usize,
    validation_image_count: usize,
    blind_evaluated: bool,
    evaluated_ranking: Vec<StageCandidateEvidence>,
    skipped_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct StageCandidateEvidence {
    id: String,
    tune_mean: f64,
    selected: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ScoreSummary {
    mean_agreement: f64,
    min_image_agreement: f64,
    mean_alpha_mae: f64,
    mean_boundary_f1: f64,
    mean_soft_iou: f64,
    mean_composite_ssim: f64,
    mean_alpha_rmse: f64,
    mean_alpha_sad: f64,
    mean_boundary_band_f1: f64,
    mean_boundary_band_mae: f64,
    mean_boundary_band_rmse: f64,
    mean_boundary_band_sad: f64,
    mean_boundary_band_soft_iou: f64,
    mean_boundary_band_iou_01: f64,
    mean_boundary_band_iou_05: f64,
    mean_boundary_band_iou_09: f64,
    mean_boundary_f1_tol1: f64,
    mean_boundary_f1_tol2: f64,
    mean_boundary_f1_tol4: f64,
    mean_binary_iou_01: f64,
    mean_binary_iou_05: f64,
    mean_binary_iou_09: f64,
    mean_precision: f64,
    mean_recall: f64,
    mean_composite_mae: f64,
    mean_composite_psnr: f64,
    mean_topology: f64,
    mean_connectivity: f64,
    mean_edge_color: f64,
    mean_foreground_color: f64,
    mean_gradient: f64,
    mean_fractional_alpha: f64,
    bootstrap_95_low: f64,
    bootstrap_95_high: f64,
    warm_p95_ms: Option<f64>,
    peak_rss_bytes: Option<u64>,
    cold_start_ms: Option<f64>,
    warm_median_ms: Option<f64>,
    throughput_images_per_sec: Option<f64>,
    model_bytes: Option<u64>,
    provider: Option<String>,
    hardware: Option<String>,
    per_image: Vec<PerImageScore>,
}

#[derive(Debug, Clone, Serialize)]
struct PerImageScore {
    id: String,
    split: &'static str,
    agreement: f64,
    alpha_mae: f64,
    boundary_f1: f64,
    soft_iou: f64,
    roi_pixels: usize,
    roi_alpha_mae: f64,
    roi_soft_iou: f64,
    composite_ssim: f64,
    alpha_rmse: f64,
    alpha_sad: f64,
    boundary_f1_tol1: f64,
    boundary_f1_tol2: f64,
    boundary_f1_tol4: f64,
    boundary_band_mae: f64,
    boundary_band_rmse: f64,
    boundary_band_sad: f64,
    boundary_band_soft_iou: f64,
    boundary_band_iou_01: f64,
    boundary_band_iou_05: f64,
    boundary_band_iou_09: f64,
    binary_iou_01: f64,
    binary_iou_05: f64,
    binary_iou_09: f64,
    precision: f64,
    recall: f64,
    composite_mae: f64,
    composite_psnr: f64,
    composite_ssim_full: f64,
    composite_ssim_roi: f64,
    boundary_band_f1: Option<f64>,
    topology_score: Option<f64>,
    connectivity_score: Option<f64>,
    edge_color_error: Option<f64>,
    foreground_color_error: Option<f64>,
    gradient_error: Option<f64>,
    fractional_alpha_error: Option<f64>,
    failure: bool,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RankedPipeline {
    id: String,
    segmenter: String,
    trimap: String,
    refiner: String,
    foreground: String,
    strength: f32,
    tune: ScoreSummary,
    validation: Option<ScoreSummary>,
    selected_on_tune_only: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CacheSummary {
    key_formula: &'static str,
    raw_segmenter_count: usize,
    image_count: usize,
    expected_inference_calls: usize,
    observed_cold_misses: usize,
    observed_warm_hits: usize,
    downstream_variants_reused_raw_masks: bool,
    corruption_detected_and_recovered: bool,
    stale_key_miss_detected: bool,
    incompatible_dimension_recovered: bool,
    atomic_writes: bool,
    atomic_temp_rename_verified: bool,
    atomic_reader_probe: bool,
    path_containment: bool,
    concurrency_locking: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ArenaRules {
    one_frozen_config_across_inputs: bool,
    failures_score_zero: bool,
    filename_and_reference_leakage_blocked: bool,
    assisted_sam_separate: bool,
    equal_image_weight: bool,
    roi_definition: &'static str,
    maximize_score_semantics: bool,
    deterministic_tie_breaks: Vec<&'static str>,
    track_level_reporting: bool,
    leave_one_image_out_tuning: bool,
    leave_one_image_out_folds: usize,
    each_arena_image_held_out_once: bool,
    real_one_frozen_config_across_inputs: bool,
    six_image_gate: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct GateReport {
    status: &'static str,
    real_tournament_available: bool,
    reason: &'static str,
    synthetic_engine_passed: bool,
    validation_improvement_with_confidence_interval: bool,
    critical_category_regression_budget_passed: bool,
    blind_promotion_protection: bool,
    champion_promoted: bool,
}

#[derive(Debug, Clone, Serialize)]
struct M15Report {
    report_version: &'static str,
    metric_version: &'static str,
    execution: &'static str,
    model_registry: Vec<CandidateRegistration>,
    stages: Vec<StageSummary>,
    selected_synthetic_pipeline: RankedPipeline,
    validation_ranked: Vec<RankedPipeline>,
    track_results: BTreeMap<String, ScoreSummary>,
    cache: CacheSummary,
    sweep: SweepReport,
    arena_rules: ArenaRules,
    paired_comparison: PairedComparison,
    real_arena: RealArenaEvidence,
    synthetic_loo: Vec<LooFoldEvidence>,
    metric_metadata: MetricMetadata,
    blind_lifecycle: BlindLifecycle,
    champions: ChampionFields,
    gate: GateReport,
    artifact_scope: &'static str,
    determinism: DeterminismEvidence,
}

#[derive(Debug, Clone, Serialize)]
struct DeterminismEvidence {
    artifact_manifest_sha256: String,
    second_generation_manifest_sha256: String,
    all_generated_artifacts_match: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ChampionFields {
    raw_alpha: Option<String>,
    full_automatic: Option<String>,
    assisted_prompted: Option<String>,
    overall_full_automatic_only: Option<String>,
    reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct RealArenaEvidence {
    manifest: String,
    corpus_available: bool,
    records: Vec<RealArenaRecord>,
    candidates: Vec<RealCandidateEvidence>,
    loo_folds: Vec<LooFoldEvidence>,
    runtime_status: String,
    tournament: Option<RealTournamentEvidence>,
}

#[derive(Debug, Clone, Serialize)]
struct RealTournamentEvidence {
    legal_selection_ids: Vec<String>,
    blind_ids_excluded: Vec<String>,
    stages: Vec<StageSummary>,
    selected_config: Option<String>,
    validation_ranked: Vec<String>,
    paired_ci_available: bool,
    loo_status: String,
    selected_tune_summary: Option<ScoreSummary>,
    selected_validation_summary: Option<ScoreSummary>,
    paired_comparison: Option<PairedComparison>,
    critical_category_deltas: BTreeMap<String, CriticalCategoryEvidence>,
    tag_aggregates: BTreeMap<String, ScoreSummary>,
    resolved_configs: BTreeMap<String, RealResolvedConfig>,
    loo_folds: Vec<LooFoldEvidence>,
    one_frozen_config_across_inputs: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CriticalCategoryEvidence {
    budget: f64,
    baseline_ids: Vec<String>,
    challenger_ids: Vec<String>,
    delta: Option<f64>,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct RealResolvedConfig {
    strength: f32,
    foreground_threshold: u8,
    background_threshold: u8,
    erode_size: u32,
    closed_form_radius: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RealArenaRecord {
    id: String,
    input: String,
    target: String,
    declared_split: String,
    width: u32,
    height: u32,
    input_rgb_sha256: String,
    target_rgba_sha256: String,
    fractional_alpha: f64,
    tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RealCandidateEvidence {
    id: String,
    manifest: String,
    manifest_valid: bool,
    weights_present: bool,
    weight_hash_verified: bool,
    intended_use_approved: bool,
    runtime_available: bool,
    status: String,
    inference_calls: usize,
    evaluated_image_ids: Vec<String>,
    failed_image_reasons: BTreeMap<String, String>,
    per_image_scores: Vec<PerImageScore>,
    performance: Option<PerformanceEvidence>,
    mean_agreement: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct PerformanceEvidence {
    cold_start_ms: f64,
    warm_median_ms: f64,
    warm_p95_ms: f64,
    throughput_images_per_sec: f64,
    model_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
    peak_rss_reason: String,
    requested_provider: String,
    active_provider: String,
    hardware: String,
}

#[derive(Debug, Clone, Serialize)]
struct LooFoldEvidence {
    held_out_id: String,
    training_ids: Vec<String>,
    selection_source: String,
    selected_config: Option<String>,
    held_out_evaluated: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct MetricMetadata {
    formula: &'static str,
    boundary_tolerance: &'static str,
    roi: &'static str,
    equal_image_weight: bool,
    backgrounds: Vec<String>,
    background_hashes: BTreeMap<String, String>,
    color_space: &'static str,
    resize_rule: &'static str,
    ssim_implemented: bool,
    critical_categories: BTreeMap<String, f64>,
    metric_availability: BTreeMap<String, MetricAvailability>,
}

#[derive(Debug, Clone, Serialize)]
struct MetricAvailability {
    available: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct BlindLifecycle {
    state: &'static str,
    blind_used_for_sweeps: bool,
    blind_used_for_promotion: bool,
    split_retired_after_influence: bool,
    new_untouched_set_required: bool,
    fixed_arena_loo_override: bool,
    legacy_declared_blind_ids: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlindEvent {
    EvaluateReleaseCandidate,
    InfluencePromotion,
    RegisterNewUntouchedSet,
}

#[allow(dead_code)]
impl BlindLifecycle {
    fn transition(&self, event: BlindEvent) -> Result<Self> {
        transition_blind_lifecycle(self, event)
    }
    fn promotion_allowed(&self) -> bool {
        self.state == "untouched" && !self.new_untouched_set_required
    }
}

#[allow(dead_code)]
fn transition_blind_lifecycle(
    current: &BlindLifecycle,
    event: BlindEvent,
) -> Result<BlindLifecycle> {
    match (current.state, event) {
        ("untouched", BlindEvent::EvaluateReleaseCandidate) => Ok(BlindLifecycle {
            state: "evaluated_release_candidate",
            ..current.clone()
        }),
        ("evaluated_release_candidate", BlindEvent::InfluencePromotion) => Ok(BlindLifecycle {
            state: "retired_after_influence",
            blind_used_for_sweeps: current.blind_used_for_sweeps,
            blind_used_for_promotion: true,
            split_retired_after_influence: true,
            new_untouched_set_required: true,
            fixed_arena_loo_override: current.fixed_arena_loo_override,
            legacy_declared_blind_ids: current.legacy_declared_blind_ids.clone(),
        }),
        ("retired_after_influence", BlindEvent::RegisterNewUntouchedSet) => Ok(BlindLifecycle {
            state: "untouched",
            blind_used_for_sweeps: false,
            blind_used_for_promotion: false,
            split_retired_after_influence: false,
            new_untouched_set_required: false,
            fixed_arena_loo_override: false,
            legacy_declared_blind_ids: Vec::new(),
        }),
        (_, BlindEvent::InfluencePromotion) => {
            anyhow::bail!("blind split cannot influence promotion from current state")
        }
        _ => anyhow::bail!("invalid blind lifecycle transition"),
    }
}

#[derive(Debug, Clone, Serialize)]
struct SweepReport {
    grid_values: Vec<f32>,
    random_seed: u64,
    random_values: Vec<f32>,
    latin_hypercube_seed: u64,
    latin_hypercube_values: Vec<f32>,
    deterministic: bool,
    blind_used_for_selection: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PairedComparison {
    baseline: String,
    challenger: String,
    per_image_deltas: Vec<f64>,
    mean_delta: f64,
    bootstrap_delta_95_low: f64,
    bootstrap_delta_95_high: f64,
    method: &'static str,
    seed: u64,
    resamples: usize,
}

#[derive(Debug, Clone)]
struct CacheRecord {
    input_hash: String,
    manifest_hash: String,
    width: u32,
    height: u32,
    values: Vec<f32>,
}

#[derive(Clone)]
struct RawMaskCache {
    root: PathBuf,
}

impl RawMaskCache {
    fn new(root: &Path) -> Result<Self> {
        ensure!(!root.as_os_str().is_empty(), "M15 cache root is empty");
        ensure!(
            root.components()
                .all(|c| !matches!(c, Component::ParentDir)),
            "M15 cache root may not contain '..'"
        );
        fs::create_dir_all(root).with_context(|| format!("create M15 cache {}", root.display()))?;
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    fn key(input_hash: &str, manifest_hash: &str) -> String {
        let mut h = Sha256::new();
        h.update(input_hash.as_bytes());
        h.update([0]);
        h.update(manifest_hash.as_bytes());
        hex(&h.finalize())
    }

    fn path(&self, key: &str) -> Result<PathBuf> {
        ensure!(
            key.len() == 64
                && key
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid M15 cache key"
        );
        let path = self.root.join(format!("{key}.raw"));
        ensure!(
            path.parent() == Some(self.root.as_path()),
            "M15 cache path escaped root"
        );
        Ok(path)
    }

    fn read(&self, key: &str) -> Result<Option<CacheRecord>> {
        let path = self.path(key)?;
        if !path.is_file() {
            return Ok(None);
        }
        let mut bytes = Vec::new();
        File::open(&path)?.read_to_end(&mut bytes)?;
        if bytes.len() < CACHE_MAGIC.len() + 2 {
            return Ok(None);
        }
        let mut cursor = 0usize;
        if &bytes[..CACHE_MAGIC.len()] != CACHE_MAGIC {
            return Ok(None);
        }
        cursor += CACHE_MAGIC.len();
        let input_len = read_u16(&bytes, &mut cursor)? as usize;
        let input_hash = read_string(&bytes, &mut cursor, input_len)?;
        let manifest_len = read_u16(&bytes, &mut cursor)? as usize;
        let manifest_hash = read_string(&bytes, &mut cursor, manifest_len)?;
        let width = read_u32(&bytes, &mut cursor)?;
        let height = read_u32(&bytes, &mut cursor)?;
        let n = usize::try_from(read_u64(&bytes, &mut cursor)?)
            .context("M15 cache record length does not fit usize")?;
        ensure!(n <= 16_777_216, "M15 cache record is unreasonably large");
        let payload_bytes = n
            .checked_mul(4)
            .context("M15 cache payload size overflow")?;
        let expected_len = cursor
            .checked_add(payload_bytes)
            .and_then(|v| v.checked_add(64))
            .context("M15 cache record length overflow")?;
        ensure!(
            bytes.len() == expected_len,
            "M15 cache record length mismatch"
        );
        let payload_end = cursor
            .checked_add(payload_bytes)
            .context("M15 cache payload end overflow")?;
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            let bits = read_u32(&bytes, &mut cursor)?;
            let value = f32::from_le_bytes(bits.to_le_bytes());
            ensure!(value.is_finite(), "M15 cache contains non-finite alpha");
            values.push(value);
        }
        let expected_digest = String::from_utf8(bytes[payload_end..].to_vec())?;
        ensure!(
            sha_bytes(&bytes[..payload_end]) == expected_digest,
            "M15 cache payload checksum mismatch"
        );
        Ok(Some(CacheRecord {
            input_hash,
            manifest_hash,
            width,
            height,
            values,
        }))
    }

    fn write(&self, key: &str, record: &CacheRecord) -> Result<()> {
        let path = self.path(key)?;
        let lock = path.with_extension("lock");
        let mut acquired = false;
        for _ in 0..200 {
            match OpenOptions::new().write(true).create_new(true).open(&lock) {
                Ok(_) => {
                    acquired = true;
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    thread::sleep(Duration::from_millis(1))
                }
                Err(error) => return Err(error.into()),
            }
        }
        ensure!(acquired, "timed out acquiring M15 cache lock");
        let result = self.write_locked(&path, record);
        let _ = fs::remove_file(&lock);
        result
    }

    fn write_locked(&self, path: &Path, record: &CacheRecord) -> Result<()> {
        (|| {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(CACHE_MAGIC);
            write_string(&mut bytes, &record.input_hash)?;
            write_string(&mut bytes, &record.manifest_hash)?;
            bytes.extend_from_slice(&record.width.to_le_bytes());
            bytes.extend_from_slice(&record.height.to_le_bytes());
            bytes.extend_from_slice(
                &u64::try_from(record.values.len())
                    .context("M15 cache value count does not fit u64")?
                    .to_le_bytes(),
            );
            for value in &record.values {
                ensure!(
                    value.is_finite(),
                    "M15 cache write received non-finite alpha"
                );
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let digest = sha_bytes(&bytes);
            bytes.extend_from_slice(digest.as_bytes());
            let temp = path.with_extension("raw.tmp");
            {
                let mut file = File::create(&temp)?;
                file.write_all(&bytes)?;
                file.sync_all()?;
            }
            fs::rename(&temp, path)?;
            if let Some(parent) = path.parent() {
                let parent_file = File::open(parent)?;
                parent_file.sync_all()?;
            }
            Ok::<(), anyhow::Error>(())
        })()
    }

    fn load_or_compute<F>(
        &self,
        input_hash: &str,
        manifest_hash: &str,
        width: u32,
        height: u32,
        compute: F,
    ) -> Result<(Vec<f32>, bool)>
    where
        F: FnOnce() -> Result<Vec<f32>>,
    {
        ensure!(
            input_hash.len() == 64
                && input_hash
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "input hash must be lowercase SHA-256"
        );
        ensure!(
            manifest_hash.len() == 64
                && manifest_hash
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "manifest hash must be lowercase SHA-256"
        );
        let key = Self::key(input_hash, manifest_hash);
        let path = self.path(&key)?;
        let lock = path.with_extension("lock");
        let mut acquired = false;
        for _ in 0..200 {
            match OpenOptions::new().write(true).create_new(true).open(&lock) {
                Ok(_) => {
                    acquired = true;
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    thread::sleep(Duration::from_millis(1))
                }
                Err(error) => return Err(error.into()),
            }
        }
        ensure!(acquired, "timed out acquiring M15 cache lock");
        let result = (|| {
            if let Some(record) = self.read(&key).unwrap_or(None) {
                if record.input_hash == input_hash
                    && record.manifest_hash == manifest_hash
                    && record.width == width
                    && record.height == height
                    && record.values.len()
                        == usize::try_from(width)
                            .ok()
                            .and_then(|w| {
                                usize::try_from(height).ok().and_then(|h| w.checked_mul(h))
                            })
                            .unwrap_or(usize::MAX)
                    && record.values.iter().all(|v| v.is_finite())
                {
                    return Ok((record.values, true));
                }
            }
            let values = compute()?;
            let expected_len = usize::try_from(width)
                .ok()
                .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
                .context("M15 raw mask dimensions overflow")?;
            ensure!(
                values.len() == expected_len,
                "M15 raw mask has wrong dimensions"
            );
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "M15 raw mask has non-finite values"
            );
            self.write_locked(
                &path,
                &CacheRecord {
                    input_hash: input_hash.to_owned(),
                    manifest_hash: manifest_hash.to_owned(),
                    width,
                    height,
                    values: values.clone(),
                },
            )?;
            Ok((values, false))
        })();
        let _ = fs::remove_file(&lock);
        result
    }
}

fn cache_concurrency_probe(cache: &RawMaskCache, image: &SyntheticImage) -> Result<bool> {
    let shared = Arc::new(cache.clone());
    let input_hash = sha_text("m15-concurrency-input");
    let manifest_hash = sha_text("m15-concurrency-manifest");
    let values = synthetic_raw_mask(image, "P1-isnet-fast");
    let width = image.width;
    let height = image.height;
    let expected_len = image.rgb.len();
    let calls = Arc::new(AtomicUsize::new(0));
    let probe_key = RawMaskCache::key(&input_hash, &manifest_hash);
    let _ = fs::remove_file(shared.path(&probe_key)?);
    let _ = fs::remove_file(shared.path(&probe_key)?.with_extension("lock"));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let cache = Arc::clone(&shared);
        let input_hash = input_hash.clone();
        let manifest_hash = manifest_hash.clone();
        let values = values.clone();
        let calls = Arc::clone(&calls);
        workers.push(thread::spawn(move || {
            cache.load_or_compute(&input_hash, &manifest_hash, width, height, || {
                calls.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(values)
            })
        }));
    }
    for worker in workers {
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("M15 cache worker panicked"))??;
    }
    let loaded = shared.read(&RawMaskCache::key(&input_hash, &manifest_hash))?;
    Ok(calls.load(AtomicOrdering::SeqCst) == 1
        && loaded
            .map(|record| record.values.len() == expected_len)
            .unwrap_or(false))
}

fn cache_atomic_reader_probe(cache: &RawMaskCache, image: &SyntheticImage) -> Result<bool> {
    let input_hash = sha_text("m15-atomic-reader-input");
    let manifest_hash = sha_text("m15-atomic-reader-manifest");
    let key = RawMaskCache::key(&input_hash, &manifest_hash);
    let record = CacheRecord {
        input_hash: input_hash.clone(),
        manifest_hash: manifest_hash.clone(),
        width: image.width,
        height: image.height,
        values: synthetic_raw_mask(image, "P1-isnet-fast"),
    };
    cache.write(&key, &record)?;
    let shared = Arc::new(cache.clone());
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let failures = Arc::new(AtomicUsize::new(0));
    let mut readers = Vec::new();
    for _ in 0..2 {
        let c = Arc::clone(&shared);
        let stop = Arc::clone(&stop);
        let failures = Arc::clone(&failures);
        let key = key.clone();
        readers.push(thread::spawn(move || {
            while !stop.load(AtomicOrdering::Acquire) {
                if c.read(&key).is_err() {
                    failures.fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
        }));
    }
    for _ in 0..8 {
        cache.write(&key, &record)?;
    }
    stop.store(true, AtomicOrdering::Release);
    for reader in readers {
        reader
            .join()
            .map_err(|_| anyhow::anyhow!("atomic reader panicked"))?;
    }
    Ok(failures.load(AtomicOrdering::Relaxed) == 0 && cache.read(&key)?.is_some())
}

fn observe_cache_reuse(
    output: &Path,
    images: &[SyntheticImage],
    registry: &[CandidateRegistration],
) -> Result<(usize, usize)> {
    let root = output.join("raw-cache-proof");
    if root.exists() {
        for entry in fs::read_dir(&root)? {
            let path = entry?.path();
            if path.is_file() {
                fs::remove_file(path)?;
            }
        }
    }
    let cache = RawMaskCache::new(&root)?;
    let mut misses = 0;
    let mut hits = 0;
    for pass in 0..2 {
        for registration in registry.iter().filter(|r| r.id != "P5-hybrid") {
            let manifest_path = match registration.id.as_str() {
                "P0-u2-cf" => "models/m5_u2net.toml",
                "P1-isnet-fast" => "models/m4_isnet_fp32.toml",
                "P2-tracer-fba" => "models/m6_tracer_b7.toml",
                "P3-biref-vit" => "models/m7_birefnet_general.toml",
                "P4-bria-vit" => "models/m8_rmbg_1_4.toml",
                _ => continue,
            };
            let manifest_hash = sha_bytes(&fs::read(repo_path(manifest_path))?);
            for image in images {
                let (_, hit) = cache.load_or_compute(
                    &image.input_hash,
                    &manifest_hash,
                    image.width,
                    image.height,
                    || Ok(synthetic_raw_mask(image, &registration.id)),
                )?;
                if pass == 0 && !hit {
                    misses += 1;
                }
                if pass == 1 && hit {
                    hits += 1;
                }
            }
        }
    }
    Ok((misses, hits))
}

pub fn run(output: &Path) -> Result<()> {
    fs::create_dir_all(output)
        .with_context(|| format!("create M15 output {}", output.display()))?;
    let _blind_marker = Split::Blind.as_str();
    let images = synthetic_images();
    let registry = registry();
    let cache = RawMaskCache::new(&output.join("raw-cache"))?;
    let mut raw: BTreeMap<(String, String), Vec<f32>> = BTreeMap::new();
    let mut calls = 0usize;
    for registration in registry.iter().filter(|r| r.id != "P5-hybrid") {
        let manifest_path = match registration.id.as_str() {
            "P0-u2-cf" => Some("models/m5_u2net.toml"),
            "P1-isnet-fast" => Some("models/m4_isnet_fp32.toml"),
            "P2-tracer-fba" => Some("models/m6_tracer_b7.toml"),
            "P3-biref-vit" => Some("models/m7_birefnet_general.toml"),
            "P4-bria-vit" => Some("models/m8_rmbg_1_4.toml"),
            _ => None,
        };
        let manifest_hash = manifest_path
            .and_then(|path| fs::read(repo_path(path)).ok())
            .map(|bytes| sha_bytes(&bytes))
            .unwrap_or_else(|| sha_text("missing-validated-manifest"));
        for image in &images {
            let (mask, _hit) = cache.load_or_compute(
                &image.input_hash,
                &manifest_hash,
                image.width,
                image.height,
                || {
                    calls += 1;
                    Ok(synthetic_raw_mask(image, registration.id.as_str()))
                },
            )?;
            raw.insert((registration.id.clone(), image.id.clone()), mask);
        }
    }
    // A deliberate corruption, stale-key lookup and incompatible-dimension
    // lookup exercise fail-closed recovery without affecting report bytes.
    // Keep destructive corruption/dimension probes in their own ignored cache
    // namespace.  Using a production raw-mask key here could poison a valid
    // model/image entry and make a second deterministic run observe different
    // raw pixels before the probe starts.
    let probe_cache = RawMaskCache::new(&output.join("raw-cache-probe"))?;
    let probe_image = &images[0];
    let probe_manifest = sha_bytes(&fs::read(repo_path("models/m4_isnet_fp32.toml"))?);
    let probe_key = RawMaskCache::key(&probe_image.input_hash, &probe_manifest);
    let probe_path = probe_cache.path(&probe_key)?;
    let _ = probe_cache.load_or_compute(
        &probe_image.input_hash,
        &probe_manifest,
        probe_image.width,
        probe_image.height,
        || Ok(synthetic_raw_mask(probe_image, "P1-isnet-fast")),
    )?;
    let mut tampered = fs::read(&probe_path)?;
    ensure!(
        tampered.len() > CACHE_MAGIC.len() + 64,
        "cache probe record unexpectedly short"
    );
    // Mutate the final payload float, immediately before the 64-byte digest;
    // this is a same-length, finite-payload corruption probe.
    let payload = cache_payload_range(&tampered)?;
    let payload_byte = payload
        .end
        .checked_sub(1)
        .context("cache probe has no payload byte")?;
    tampered[payload_byte] ^= 1;
    fs::write(&probe_path, tampered)?;
    let (_, corruption_recovered) = probe_cache.load_or_compute(
        &probe_image.input_hash,
        &probe_manifest,
        probe_image.width,
        probe_image.height,
        || Ok(synthetic_raw_mask(probe_image, "P1-isnet-fast")),
    )?;
    let stale_manifest = sha_bytes(&fs::read(repo_path("models/m5_u2net.toml"))?);
    let stale_probe_key = RawMaskCache::key(&probe_image.input_hash, &stale_manifest);
    let _ = fs::remove_file(probe_cache.path(&stale_probe_key)?);
    let (_, stale_hit) = probe_cache.load_or_compute(
        &probe_image.input_hash,
        &stale_manifest,
        probe_image.width,
        probe_image.height,
        || Ok(synthetic_raw_mask(probe_image, "P1-isnet-fast")),
    )?;
    let incompatible_manifest = sha_bytes(&fs::read(repo_path("models/m6_tracer_b7.toml"))?);
    let incompatible_key = RawMaskCache::key(&probe_image.input_hash, &incompatible_manifest);
    probe_cache.write(
        &incompatible_key,
        &CacheRecord {
            input_hash: probe_image.input_hash.clone(),
            manifest_hash: incompatible_manifest.clone(),
            width: 1,
            height: probe_image.truth.len() as u32,
            values: vec![0.0; probe_image.rgb.len()],
        },
    )?;
    let (_, incompatible_hit) = probe_cache.load_or_compute(
        &probe_image.input_hash,
        &incompatible_manifest,
        probe_image.width,
        probe_image.height,
        || Ok(synthetic_raw_mask(probe_image, "P1-isnet-fast")),
    )?;
    let concurrency_locking = cache_concurrency_probe(&probe_cache, probe_image)?;
    let atomic_reader_probe = cache_atomic_reader_probe(&probe_cache, probe_image)?;

    let stage1 = stage1(&images, &raw, &registry)?;
    let stage2 = stage2(&images, &raw, &stage1)?;
    let stage3 = stage3(&images, &raw, &stage2)?;
    let selected = tune_and_rank(&images, &raw, &stage3)?;
    let validation_ranked = validation_ranked(&images, &raw, &stage3)?;
    let track_results = track_results(&selected, &images, &raw);
    let baseline_pipeline = RankedPipeline {
        id: "P1-isnet-fast+none+none+original@0.500".into(),
        segmenter: "P1-isnet-fast".into(),
        trimap: "none".into(),
        refiner: "none".into(),
        foreground: "original".into(),
        strength: 0.5,
        tune: empty_score(),
        validation: None,
        selected_on_tune_only: false,
    };
    let baseline = score_pipeline(&baseline_pipeline, &images, &raw, Split::Validation)?;
    let challenger = score_pipeline(&validation_ranked[0], &images, &raw, Split::Validation)?;
    let paired = paired_comparison(&baseline, &challenger)?;
    let (observed_cold_misses, observed_warm_hits) =
        observe_cache_reuse(output, &images, &registry)?;
    let unique_input_hashes: std::collections::BTreeSet<&str> = images
        .iter()
        .map(|image| image.input_hash.as_str())
        .collect();
    let sweep = sweep_report();
    let mut real_arena = inspect_real_arena(&repo_path("test_images"))?;
    if real_arena.corpus_available && std::env::var_os("ORT_DYLIB").is_some() {
        if let Err(error) = run_real_arena(&mut real_arena, &repo_path("test_images"), output) {
            let cause = format!("not-run: real execution failed: {error:#}");
            real_arena.runtime_status = cause.clone();
            for fold in &mut real_arena.loo_folds {
                fold.status = cause.clone();
                fold.held_out_evaluated = false;
                fold.selected_config = None;
            }
        }
    }
    let synthetic_loo = synthetic_loo(&images, &raw, &stage3)?;
    let metric_metadata = metric_metadata();
    let real_blind_lifecycle = blind_lifecycle_for_arena(
        &real_arena,
        real_arena.tournament.as_ref().is_some_and(|tournament| {
            tournament.loo_status.starts_with("completed")
                && tournament.loo_folds.len() == 6
                && tournament
                    .loo_folds
                    .iter()
                    .all(|fold| fold.held_out_evaluated)
        }),
    );
    write_real_status_artifacts(output, &real_arena)?;
    let mut artifact_pipelines = vec![selected.clone()];
    for stage in [&stage1.0, &stage2.0, &stage3.0] {
        for candidate in &stage.evaluated_ranking {
            if let Ok(pipeline) = real_pipeline_from_id(&candidate.id, 0.5) {
                artifact_pipelines.push(pipeline);
            }
        }
    }
    for candidate in &stage3.1 {
        for strength in sweep_report()
            .grid_values
            .iter()
            .chain(sweep_report().random_values.iter())
            .chain(sweep_report().latin_hypercube_values.iter())
        {
            if let Ok(pipeline) = real_pipeline_from_id(candidate, *strength) {
                artifact_pipelines.push(pipeline);
            }
        }
    }
    artifact_pipelines.sort_by(|a, b| a.id.cmp(&b.id));
    artifact_pipelines.dedup_by(|a, b| a.id == b.id);
    write_synthetic_artifacts(output, &selected, &artifact_pipelines, &images, &raw)?;
    write_artifact_manifest(output)?;
    let determinism_a = output.join(".determinism-a");
    let determinism_b = output.join(".determinism-b");
    if determinism_a.exists() {
        fs::remove_dir_all(&determinism_a)?;
    }
    if determinism_b.exists() {
        fs::remove_dir_all(&determinism_b)?;
    }
    fs::create_dir_all(&determinism_a)?;
    fs::create_dir_all(&determinism_b)?;
    write_synthetic_artifacts(
        &determinism_a,
        &selected,
        &artifact_pipelines,
        &images,
        &raw,
    )?;
    write_real_status_artifacts(&determinism_a, &real_arena)?;
    write_html_report(&determinism_a, &real_arena, &selected, &track_results)?;
    write_artifact_manifest(&determinism_a)?;
    write_synthetic_artifacts(
        &determinism_b,
        &selected,
        &artifact_pipelines,
        &images,
        &raw,
    )?;
    write_real_status_artifacts(&determinism_b, &real_arena)?;
    write_html_report(&determinism_b, &real_arena, &selected, &track_results)?;
    write_artifact_manifest(&determinism_b)?;
    let first_fresh = fs::read(determinism_a.join("artifacts.sha256"))?;
    let second_fresh = fs::read(determinism_b.join("artifacts.sha256"))?;
    let first_manifest_hash = sha_bytes(&first_fresh);
    let second_manifest_hash = sha_bytes(&second_fresh);
    let second_artifact_manifest = first_fresh;
    let all_generated_artifacts_match =
        second_artifact_manifest == second_fresh && first_manifest_hash == second_manifest_hash;
    fs::remove_dir_all(&determinism_a)?;
    fs::remove_dir_all(&determinism_b)?;
    let determinism = DeterminismEvidence {
        artifact_manifest_sha256: first_manifest_hash,
        second_generation_manifest_sha256: second_manifest_hash,
        all_generated_artifacts_match,
    };
    let artifact_manifest_valid = verify_artifact_manifest(output).unwrap_or(false);
    let sweep_coverage = sweep.grid_values.len() >= 3
        && sweep.random_values.len() == 5
        && sweep.latin_hypercube_values.len() == 5
        && sweep.deterministic;
    let metric_invariants =
        metric_metadata.ssim_implemented && !metric_metadata.metric_availability.is_empty();
    let synthetic_blind_protection = blind_lifecycle().promotion_allowed();
    let synthetic_engine_passed = !stage3.1.is_empty()
        && selected
            .tune
            .per_image
            .iter()
            .all(|s| s.agreement.is_finite())
        && paired.bootstrap_delta_95_low.is_finite()
        && determinism.all_generated_artifacts_match
        && observed_cold_misses > 0
        && observed_warm_hits > 0
        && !corruption_recovered
        && !stale_hit
        && !incompatible_hit
        && atomic_reader_probe
        && concurrency_locking
        && sweep_coverage
        && metric_invariants
        && artifact_manifest_valid
        && synthetic_blind_protection;
    let blind_promotion_protection = !real_blind_lifecycle.blind_used_for_sweeps
        && !real_blind_lifecycle.blind_used_for_promotion
        && real_blind_lifecycle.state == "untouched";
    let real_one_frozen = real_arena
        .tournament
        .as_ref()
        .is_some_and(|t| t.one_frozen_config_across_inputs);
    let real_validation_ci_passed = real_one_frozen
        && real_arena
            .tournament
            .as_ref()
            .and_then(|t| t.paired_comparison.as_ref())
            .is_some_and(|comparison| comparison.bootstrap_delta_95_low > 0.0);
    let real_critical_passed = real_arena.tournament.as_ref().is_some_and(|t| {
        !t.critical_category_deltas.is_empty()
            && t.critical_category_deltas
                .values()
                .all(|e| e.status == "passed")
    });
    let real_loo_passed = real_arena
        .tournament
        .as_ref()
        .is_some_and(|t| t.loo_status.starts_with("completed"));
    let real_release_qualified = real_validation_ci_passed
        && real_critical_passed
        && real_loo_passed
        && real_one_frozen
        && blind_promotion_protection;
    let champions = if let Some(tournament) = &real_arena.tournament {
        ChampionFields {
            raw_alpha: tournament
                .stages
                .first()
                .and_then(|stage| stage.survivor_ids.first())
                .cloned(),
            full_automatic: tournament.selected_config.clone(),
            assisted_prompted: None,
            overall_full_automatic_only: real_release_qualified
                .then(|| tournament.selected_config.clone())
                .flatten(),
            reason: if real_release_qualified {
                "real evidence qualified under all release gates"
            } else {
                "real leaderboard result recorded; release promotion gates are not all satisfied"
            },
        }
    } else {
        ChampionFields {
            raw_alpha: None,
            full_automatic: None,
            assisted_prompted: None,
            overall_full_automatic_only: None,
            reason: "real scores unavailable; assisted SAM is separate and not run",
        }
    };
    let report = M15Report {
        report_version: REPORT_VERSION, metric_version: METRIC_VERSION, execution: "staged synthetic contract tournament plus attempted real six-image CPU ORT arena; real candidate status is per manifest/runtime evidence",
        model_registry: registry, stages: vec![stage1.0, stage2.0, stage3.0], selected_synthetic_pipeline: selected.clone(), validation_ranked,
        track_results: track_results.clone(), cache: CacheSummary { key_formula: "sha256(input_rgb_sha256 || 0x00 || validated_model_manifest_sha256)", raw_segmenter_count: 5, image_count: images.len(), expected_inference_calls: 5 * unique_input_hashes.len(), observed_cold_misses, observed_warm_hits, downstream_variants_reused_raw_masks: observed_warm_hits > 0, corruption_detected_and_recovered: !corruption_recovered, stale_key_miss_detected: !stale_hit, incompatible_dimension_recovered: !incompatible_hit, atomic_writes: atomic_reader_probe, atomic_temp_rename_verified: atomic_reader_probe, atomic_reader_probe, path_containment: cache.path(&RawMaskCache::key("input", "manifest")).is_ok(), concurrency_locking },
        sweep, arena_rules: arena_rules(real_arena.corpus_available && real_arena.tournament.is_some(), real_arena.tournament.as_ref().is_some_and(|t| t.legal_selection_ids.len() == 6), real_arena.tournament.as_ref().is_some_and(|t| t.one_frozen_config_across_inputs), synthetic_engine_passed), paired_comparison: paired,
        real_arena: real_arena.clone(), synthetic_loo: synthetic_loo.clone(), metric_metadata, blind_lifecycle: real_blind_lifecycle,
        champions,
        gate: GateReport { status: if real_release_qualified { "real-release-qualified" } else if real_arena.candidates.iter().any(|c| c.inference_calls > 0) { "real-evidence-collected-no-promotion" } else { "blocked-real-evidence-unavailable" }, real_tournament_available: real_arena.tournament.is_some(), reason: if real_release_qualified { "Real staged tournament passed computed release gates." } else if real_arena.tournament.is_some() { "Real staged tournament completed, but computed LOO, paired CI, or critical-category gates are insufficient." } else { "No approved candidate completed real inference; promotion is unavailable." }, synthetic_engine_passed, validation_improvement_with_confidence_interval: real_validation_ci_passed, critical_category_regression_budget_passed: real_critical_passed, blind_promotion_protection, champion_promoted: real_release_qualified },
        artifact_scope: "all Stage 1/2/3 candidates and swept Stage-3 survivors on tune records; frozen selected pipeline on validation records; unavailable real candidates status-only",
        determinism,
    };
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(output.join("report.json"), &bytes)?;
    fs::write(
        output.join("report.sha256"),
        format!("{}  report.json\n", sha_bytes(&bytes)),
    )?;
    write_html_report(output, &real_arena, &selected, &track_results)?;
    write_artifact_manifest(output)?;
    println!("wrote {} (raw inference calls: {calls})", output.display());
    Ok(())
}

fn write_artifact_manifest(output: &Path) -> Result<()> {
    let mut entries = Vec::new();
    fn visit(root: &Path, dir: &Path, entries: &mut Vec<String>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                if path.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    n == "raw-cache"
                        || n == "raw-cache-proof"
                        || n == "raw-cache-probe"
                        || n == "raw-cache-real"
                }) {
                    continue;
                }
                visit(root, &path, entries)?;
            } else if !matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some("artifacts.sha256") | Some("report.json") | Some("report.sha256")
            ) {
                let bytes = fs::read(&path)?;
                let rel = path.strip_prefix(root)?.display().to_string();
                entries.push(format!("{}  {}", sha_bytes(&bytes), rel));
            }
        }
        Ok(())
    }
    visit(output, output, &mut entries)?;
    entries.sort();
    fs::write(
        output.join("artifacts.sha256"),
        format!("{}\n", entries.join("\n")),
    )?;
    Ok(())
}

fn verify_artifact_manifest(output: &Path) -> Result<bool> {
    let manifest = fs::read_to_string(output.join("artifacts.sha256"))?;
    for line in manifest.lines().filter(|line| !line.is_empty()) {
        let (expected, relative) = line
            .split_once("  ")
            .context("malformed artifact manifest line")?;
        let path = output.join(relative);
        ensure!(
            path.starts_with(output) && path.is_file(),
            "artifact manifest path missing or escaped"
        );
        ensure!(
            expected == sha_bytes(&fs::read(path)?),
            "artifact manifest hash mismatch"
        );
    }
    Ok(true)
}

fn write_synthetic_artifacts(
    output: &Path,
    selected: &RankedPipeline,
    pipelines: &[RankedPipeline],
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
) -> Result<()> {
    write_artifacts_mode(
        output,
        selected,
        pipelines,
        images,
        raw,
        EvaluationMode::SyntheticContract,
        "synthetic-artifacts",
    )
}

fn write_artifacts_mode(
    output: &Path,
    selected: &RankedPipeline,
    pipelines: &[RankedPipeline],
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    mode: EvaluationMode,
    bundle_root: &str,
) -> Result<()> {
    for image in images {
        for pipeline in pipelines {
            if image.split == Split::Validation && pipeline.id != selected.id {
                continue;
            }
            let dir_name = if pipeline.id == selected.id {
                "selected".to_owned()
            } else {
                pipeline.id.replace(['/', '\\', '@', '+'], "_")
            };
            let dir = if bundle_root == "real-arena" {
                output.join(bundle_root).join(&dir_name).join(&image.id)
            } else {
                output.join(bundle_root).join(&image.id).join(dir_name)
            };
            fs::create_dir_all(&dir)?;
            let _ = fs::remove_file(dir.join("config.json"));
            let config = serde_json::json!({"pipeline": pipeline, "image_id": image.id, "input_rgb_sha256": image.input_hash, "truth_used_only_by_evaluator": true});
            let evaluation = evaluate_pipeline(pipeline, image, raw, mode);
            let score = match &evaluation {
                Ok(_) => score_one_mode(pipeline, image, raw, mode),
                Err(error) => {
                    failure_score(image, format!("pipeline evaluation failed: {error:#}"))
                }
            };
            let metrics = serde_json::to_value(&score)?;
            let mut config_bytes = serde_json::to_vec_pretty(&config)?;
            config_bytes.push(b'\n');
            let mut metric_bytes = serde_json::to_vec_pretty(&metrics)?;
            metric_bytes.push(b'\n');
            fs::write(dir.join("resolved-config.json"), &config_bytes)?;
            fs::write(dir.join("metrics.json"), &metric_bytes)?;
            let (values, cleaned, trimap, refined, synthetic_foreground) = match evaluation {
                Ok(value) => (
                    value.coarse,
                    value.cleaned,
                    value.trimap,
                    value.refined_alpha,
                    value.foreground,
                ),
                Err(_) => (
                    vec![0.0; image.truth.len()],
                    vec![0.0; image.truth.len()],
                    vec![0.0; image.truth.len()],
                    vec![0.0; image.truth.len()],
                    vec![[0.0; 3]; image.rgb.len()],
                ),
            };
            let diff: Vec<f32> = refined
                .iter()
                .zip(&image.truth)
                .map(|(a, b)| (a - b).abs())
                .collect();
            write_mask_file(
                &dir.join("coarse-alpha.png"),
                image.width,
                image.height,
                &values,
            )?;
            write_mask_file(
                &dir.join("cleaned-alpha.png"),
                image.width,
                image.height,
                &cleaned,
            )?;
            write_rgb_file(
                &dir.join("input.png"),
                image.width,
                image.height,
                &image.rgb,
            )?;
            write_rgb_file(
                &dir.join("reference.png"),
                image.width,
                image.height,
                &image.reference_rgb,
            )?;
            write_mask_file(&dir.join("trimap.png"), image.width, image.height, &trimap)?;
            write_mask_file(
                &dir.join("refined-alpha.png"),
                image.width,
                image.height,
                &refined,
            )?;
            write_rgb_file(
                &dir.join("foreground.png"),
                image.width,
                image.height,
                &synthetic_foreground,
            )?;
            let cutout = bgremove_core::Foreground::new(
                bgremove_core::RgbImageF32::new(
                    image.width,
                    image.height,
                    synthetic_foreground.clone(),
                )?,
                bgremove_core::AlphaMask::new(image.width, image.height, refined.clone())?,
            )?;
            fs::write(dir.join("cutout.png"), encode_straight_rgba_png(&cutout)?)?;
            write_rgb_file(
                &dir.join("composite-white.png"),
                image.width,
                image.height,
                &composite_rgb(&synthetic_foreground, &refined, [1.0; 3]),
            )?;
            write_rgb_file(
                &dir.join("composite-black.png"),
                image.width,
                image.height,
                &composite_rgb(&synthetic_foreground, &refined, [0.0; 3]),
            )?;
            write_mask_file(
                &dir.join("alpha-diff.png"),
                image.width,
                image.height,
                &diff,
            )?;
            write_mask_file(
                &dir.join("boundary-diff.png"),
                image.width,
                image.height,
                &diff,
            )?;
            let boundary_overlay: Vec<[f32; 3]> = diff
                .iter()
                .map(|value| {
                    if *value > 0.1 {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 0.0, 0.0]
                    }
                })
                .collect();
            write_rgb_file(
                &dir.join("boundary-diff.png"),
                image.width,
                image.height,
                &boundary_overlay,
            )?;
            let mut hashes = BTreeMap::new();
            for entry in fs::read_dir(&dir)? {
                let path = entry?.path();
                if path.file_name().and_then(|n| n.to_str()) == Some("hashes.json") {
                    continue;
                }
                hashes.insert(
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                    sha_bytes(&fs::read(path)?),
                );
            }
            let mut hash_bytes = serde_json::to_vec_pretty(&hashes)?;
            hash_bytes.push(b'\n');
            fs::write(dir.join("hashes.json"), hash_bytes)?;
        }
    }
    Ok(())
}

fn write_mask_file(path: &Path, width: u32, height: u32, values: &[f32]) -> Result<()> {
    let mask = bgremove_core::AlphaMask::new(width, height, values.to_vec())?;
    fs::write(path, encode_mask_png(&mask)?)?;
    Ok(())
}

fn write_rgb_file(path: &Path, width: u32, height: u32, values: &[[f32; 3]]) -> Result<()> {
    ensure!(
        values.len()
            == (width as usize)
                .checked_mul(height as usize)
                .context("RGB artifact size overflow")?,
        "RGB artifact dimensions mismatch"
    );
    let mut raw = Vec::with_capacity(
        values
            .len()
            .checked_mul(3)
            .context("RGB artifact byte size overflow")?,
    );
    for pixel in values {
        raw.extend(pixel.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
    }
    let mut encoded = Vec::new();
    PngEncoder::new(&mut encoded).write_image(&raw, width, height, ColorType::Rgb8.into())?;
    fs::write(path, encoded)?;
    Ok(())
}

fn composite_rgb(rgb: &[[f32; 3]], alpha: &[f32], background: [f32; 3]) -> Vec<[f32; 3]> {
    rgb.iter()
        .zip(alpha)
        .map(|(pixel, a)| std::array::from_fn(|c| pixel[c] * *a + background[c] * (1.0 - *a)))
        .collect()
}

fn write_html_report(
    output: &Path,
    arena: &RealArenaEvidence,
    selected: &RankedPipeline,
    tracks: &BTreeMap<String, ScoreSummary>,
) -> Result<()> {
    let mut html = String::from("<!doctype html><meta charset=\"utf-8\"><title>M15 tournament</title><h1>M15 deterministic tournament</h1><p>Real inference is reported as not-run when ORT_DYLIB is unavailable. Click a column to sort.</p><table id=\"candidates\" border=\"1\"><thead><tr><th>Candidate</th><th>Manifest</th><th>Manifest valid</th><th>Weight hash</th><th>Approval</th><th>Status</th></tr></thead><tbody>");
    for candidate in &arena.candidates {
        let id = escape_html(&candidate.id);
        let manifest = escape_html(&candidate.manifest);
        let status = escape_html(&candidate.status);
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            id,
            manifest,
            candidate.manifest_valid,
            candidate.weight_hash_verified,
            candidate.intended_use_approved,
            status
        ));
    }
    html.push_str("</tbody></table><h2>Evaluated synthetic visual/metric rows</h2><table id=\"synthetic\" border=\"1\"><thead><tr><th>Pipeline</th><th>Track</th><th>Visual</th><th>Mean agreement</th><th>Alpha RMSE</th><th>Boundary-band MAE</th><th>Boundary F1 (1px)</th><th>Topology</th><th>Connectivity</th><th>Edge colour</th><th>Foreground colour</th><th>Gradient</th><th>Fractional alpha</th><th>Composite MAE</th><th>Composite PSNR</th><th>Best image</th><th>Median image</th><th>Worst-decile image</th><th>Worst image</th><th>Largest regression</th></tr></thead><tbody>");
    for (track, summary) in tracks {
        let best = summary
            .per_image
            .iter()
            .max_by(|a, b| {
                a.agreement
                    .partial_cmp(&b.agreement)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|s| s.id.as_str())
            .unwrap_or("none");
        let worst = summary
            .per_image
            .iter()
            .min_by(|a, b| {
                a.agreement
                    .partial_cmp(&b.agreement)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|s| s.id.as_str())
            .unwrap_or("none");
        let mut ordered = summary.per_image.clone();
        ordered.sort_by(|a, b| {
            b.agreement
                .partial_cmp(&a.agreement)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        let median = ordered
            .get(ordered.len().saturating_sub(1) / 2)
            .map(|s| s.id.as_str())
            .unwrap_or("none");
        let worst_decile = ordered
            .get(
                ordered
                    .len()
                    .saturating_sub(1)
                    .min(ordered.len().saturating_mul(9) / 10),
            )
            .map(|s| s.id.as_str())
            .unwrap_or("none");
        let best = escape_html(best);
        let median = escape_html(median);
        let worst_decile = escape_html(worst_decile);
        let worst = escape_html(worst);
        let visual_id = summary
            .per_image
            .first()
            .map(|s| s.id.as_str())
            .unwrap_or("none");
        let visual = if visual_id == "none" {
            "unavailable".to_owned()
        } else {
            let safe_id = escape_html(visual_id);
            let base = format!("synthetic-artifacts/{}/selected/", visual_id);
            format!("<div class=\"visuals\"><img width=\"120\" alt=\"{safe_id} input\" src=\"{base}input.png\"><img width=\"120\" alt=\"{safe_id} reference\" src=\"{base}reference.png\"><img width=\"120\" alt=\"{safe_id} cutout\" src=\"{base}cutout.png\"><img width=\"120\" alt=\"{safe_id} alpha diff\" src=\"{base}alpha-diff.png\"><img width=\"120\" alt=\"{safe_id} boundary diff\" src=\"{base}boundary-diff.png\"><img width=\"120\" alt=\"{safe_id} black composite\" src=\"{base}composite-black.png\"><img width=\"120\" alt=\"{safe_id} white composite\" src=\"{base}composite-white.png\"></div>")
        };
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.8}</td></tr>",
            escape_html(&selected.id),
            escape_html(track),
            visual,
            summary.mean_agreement,
            summary.mean_alpha_rmse,
            summary.mean_boundary_band_mae,
            summary.mean_boundary_f1_tol1,
            summary.mean_topology,
            summary.mean_connectivity,
            summary.mean_edge_color,
            summary.mean_foreground_color,
            summary.mean_gradient,
            summary.mean_fractional_alpha,
            summary.mean_composite_mae,
            summary.mean_composite_psnr,
            best,
            median,
            worst_decile,
            worst,
            summary
                .per_image
                .iter()
                .map(|s| s.agreement)
                .fold(0.0, f64::min)
        ));
    }
    html.push_str("</tbody></table>");
    if let Some(tournament) = &arena.tournament {
        html.push_str("<h2>Real held-out tag aggregates</h2><table id=\"tags\" border=\"1\"><thead><tr><th>Tag</th><th>Mean agreement</th><th>Alpha RMSE</th><th>Boundary-band MAE</th><th>Boundary F1 (1px)</th><th>Topology</th><th>Connectivity</th><th>Edge colour</th><th>Foreground colour</th><th>Gradient</th><th>Fractional alpha</th><th>Composite MAE</th><th>Composite PSNR</th></tr></thead><tbody>");
        for (tag, summary) in &tournament.tag_aggregates {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td><td>{:.8}</td></tr>",
                escape_html(tag),
                summary.mean_agreement,
                summary.mean_alpha_rmse,
                summary.mean_boundary_band_mae,
                summary.mean_boundary_f1_tol1,
                summary.mean_topology,
                summary.mean_connectivity,
                summary.mean_edge_color,
                summary.mean_foreground_color,
                summary.mean_gradient,
                summary.mean_fractional_alpha,
                summary.mean_composite_mae,
                summary.mean_composite_psnr,
            ));
        }
        html.push_str("</tbody></table>");
    }
    html.push_str("<script>for(const h of document.querySelectorAll('th'))h.onclick=()=>{const t=h.closest('table'),i=[...h.parentNode.children].indexOf(h),b=t.tBodies[0];[...b.rows].sort((a,c)=>a.cells[i].textContent.localeCompare(c.cells[i].textContent)).forEach(r=>b.append(r))}</script>\n");
    fs::write(output.join("report.html"), html)?;
    Ok(())
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn write_real_status_artifacts(output: &Path, arena: &RealArenaEvidence) -> Result<()> {
    for candidate in &arena.candidates {
        for record in &arena.records {
            let dir = output
                .join("real-arena")
                .join(&candidate.id)
                .join(&record.id);
            fs::create_dir_all(&dir)?;
            let evaluated = candidate
                .evaluated_image_ids
                .iter()
                .any(|id| id == &record.id)
                && (record.declared_split != "blind"
                    || arena
                        .tournament
                        .as_ref()
                        .is_some_and(|t| t.loo_status.starts_with("completed")));
            if evaluated && output.join("real-arena").exists() {
                let has_bundle = fs::read_dir(output.join("real-arena"))
                    .ok()
                    .into_iter()
                    .flatten()
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with(&candidate.id)
                    })
                    .map(|entry| entry.path().join(&record.id))
                    .any(|path| path.join("metrics.json").is_file());
                if has_bundle {
                    continue;
                }
            }
            let status = if evaluated {
                "evaluated"
            } else if candidate.status.starts_with("unavailable") {
                "unavailable"
            } else {
                "not-run"
            };
            let artifact = serde_json::json!({
                "status": status,
                "candidate": candidate.id,
                "image": record.id,
                "reason": candidate.status,
                "declared_split": record.declared_split,
                "evaluated_image": evaluated,
                "required_artifacts": ["resolved-config.json", "coarse-alpha.png", "cleaned-alpha.png", "trimap.png", "refined-alpha.png", "foreground.png", "cutout.png", "composite-white.png", "composite-black.png", "alpha-diff.png", "boundary-diff.png", "metrics.json"],
            });
            let mut bytes = serde_json::to_vec_pretty(&artifact)?;
            bytes.push(b'\n');
            fs::write(dir.join("status.json"), bytes)?;
        }
    }
    Ok(())
}

fn registry() -> Vec<CandidateRegistration> {
    vec![
        CandidateRegistration {
            id: "P0-u2-cf".into(),
            coarse_stage: "U2-Net general".into(),
            refiner: "closed-form".into(),
            foreground: "multilevel".into(),
            availability: "approved-external-manifest; runtime-gated",
            executable_mode: "synthetic-contract-only; real-adapter-registered",
            eligible_for_real_champion: false,
        },
        CandidateRegistration {
            id: "P1-isnet-fast".into(),
            coarse_stage: "IS-Net FP32".into(),
            refiner: "none".into(),
            foreground: "fast-estimator".into(),
            availability: "approved-external-manifest; runtime-gated",
            executable_mode: "synthetic-contract-only; real-adapter-registered",
            eligible_for_real_champion: false,
        },
        CandidateRegistration {
            id: "P2-tracer-fba".into(),
            coarse_stage: "TRACER-B7".into(),
            refiner: "FBA".into(),
            foreground: "FBA foreground".into(),
            availability: "approved-external-manifest; runtime-gated",
            executable_mode: "synthetic-contract-only; real-adapter-registered",
            eligible_for_real_champion: false,
        },
        CandidateRegistration {
            id: "P3-biref-vit".into(),
            coarse_stage: "BiRefNet general".into(),
            refiner: "ViTMatte small Distinctions".into(),
            foreground: "fast-estimator".into(),
            availability: "approved-external-manifest; runtime-gated",
            executable_mode: "synthetic-contract-only; real-adapter-registered",
            eligible_for_real_champion: false,
        },
        CandidateRegistration {
            id: "P4-bria-vit".into(),
            coarse_stage: "BRIA approved version".into(),
            refiner: "ViTMatte small Distinctions".into(),
            foreground: "fast-estimator".into(),
            availability: "unavailable-unapproved-or-missing",
            executable_mode: "synthetic-contract-only; adapter unavailable",
            eligible_for_real_champion: false,
        },
        CandidateRegistration {
            id: "P5-hybrid".into(),
            coarse_stage: "reserved for M16".into(),
            refiner: "adaptive".into(),
            foreground: "selected by evidence".into(),
            availability: "reserved",
            executable_mode: "not-run",
            eligible_for_real_champion: false,
        },
    ]
}

fn inspect_real_arena(root: &Path) -> Result<RealArenaEvidence> {
    let manifest_path = root.join("arena.jsonl");
    let text = match fs::read_to_string(&manifest_path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(RealArenaEvidence {
                manifest: manifest_path.display().to_string(),
                corpus_available: false,
                records: Vec::new(),
                candidates: real_candidate_evidence(),
                loo_folds: Vec::new(),
                runtime_status: "arena manifest missing".into(),
                tournament: None,
            });
        }
        Err(error) => return Err(error.into()),
    };
    let mut records = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("parse arena record line {}", line_no + 1))?;
        let id = value["id"].as_str().context("arena id missing")?.to_owned();
        let input = value["input"]
            .as_str()
            .context("arena input missing")?
            .to_owned();
        let target = value["target"]
            .as_str()
            .context("arena target missing")?
            .to_owned();
        let input_path = root.join(&input);
        let target_path = root.join(&target);
        let input_image = load_canonical(&input_path)
            .with_context(|| format!("decode real arena input {}", input_path.display()))?;
        let target_bytes = fs::read(&target_path)
            .with_context(|| format!("read real arena target {}", target_path.display()))?;
        let mut rgb_bytes = Vec::with_capacity(input_image.rgb().data().len() * 12);
        for pixel in input_image.rgb().data() {
            for channel in pixel {
                rgb_bytes.extend_from_slice(&channel.to_le_bytes());
            }
        }
        let target_image = load_canonical(&target_path)?;
        ensure!(
            target_image.dimensions() == input_image.dimensions(),
            "arena {} input/target dimensions differ",
            id
        );
        records.push(RealArenaRecord {
            id,
            input,
            target,
            declared_split: value["split"].as_str().unwrap_or("unknown").to_owned(),
            width: input_image.width(),
            height: input_image.height(),
            input_rgb_sha256: sha_bytes(&rgb_bytes),
            target_rgba_sha256: sha_bytes(&target_bytes),
            fractional_alpha: value["fractional_alpha_reported"].as_f64().unwrap_or(0.0),
            tags: value["tags"]
                .as_array()
                .map(|tags| {
                    tags.iter()
                        .filter_map(|tag| tag.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default(),
        });
    }
    ensure!(
        records.len() == 6,
        "M15 real arena must contain exactly six records"
    );
    let portrait_count = records
        .iter()
        .filter(|record| record.tags.iter().any(|tag| tag == "portrait-character"))
        .count();
    let emissive_count = records
        .iter()
        .filter(|record| record.tags.iter().any(|tag| tag == "emissive-artwork"))
        .count();
    ensure!(
        portrait_count == 4 && emissive_count == 2,
        "arena manifest must explicitly tag four portrait-character and two emissive-artwork records"
    );
    let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
    let loo_folds = ids.iter().map(|held_out| LooFoldEvidence {
        held_out_id: held_out.clone(),
        training_ids: ids.iter().filter(|id| *id != held_out).cloned().collect(),
        selection_source: "five other fixed-arena records only; declared split labels are not used to select parameters".into(),
        selected_config: None,
        held_out_evaluated: false,
        status: "not-run: ORT_DYLIB unavailable".into(),
    }).collect();
    Ok(RealArenaEvidence {
        manifest: manifest_path.display().to_string(),
        corpus_available: true,
        records,
        candidates: real_candidate_evidence(),
        loo_folds,
        runtime_status: if std::env::var_os("ORT_DYLIB").is_some() {
            "ORT_DYLIB configured; real execution path requires explicit runtime validation"
        } else {
            "ORT_DYLIB unavailable; manifest/corpus inventory only"
        }
        .into(),
        tournament: None,
    })
}

fn real_candidate_evidence() -> Vec<RealCandidateEvidence> {
    [
        ("P0-u2-cf", "models/m5_u2net.toml"),
        ("P1-isnet-fast", "models/m4_isnet_fp32.toml"),
        ("P2-tracer-fba", "models/m6_tracer_b7.toml"),
        ("P3-biref-vit", "models/m7_birefnet_general.toml"),
        ("P4-bria-vit", "models/m8_rmbg_1_4.toml"),
    ]
    .into_iter()
    .map(|(id, path)| {
        let parsed = fs::read_to_string(repo_path(path))
            .ok()
            .and_then(|text| parse_toml(&text).ok());
        let manifest_valid = parsed.is_some();
        let intended_use_approved = parsed
            .as_ref()
            .map(|m| m.intended_use_approved)
            .unwrap_or(false);
        let (weights_present, weight_hash_verified) = match parsed.as_ref() {
            Some(manifest) => (
                manifest.resolve_model_path(&repo_path(path)).is_ok(),
                manifest.verify_model_hash(&repo_path(path)).is_ok(),
            ),
            None => (false, false),
        };
        let runtime_available = std::env::var_os("ORT_DYLIB")
            .map(|p| Path::new(&p).is_file())
            .unwrap_or(false);
        let status = if !manifest_valid {
            "unavailable: manifest validation failed"
        } else if !weights_present || !weight_hash_verified {
            "unavailable: checkpoint missing or hash verification failed"
        } else if !intended_use_approved {
            "unavailable: intended-use approval is false"
        } else if !runtime_available {
            "not-run: ORT_DYLIB unavailable"
        } else {
            "eligible: runtime configured; explicit real run required"
        };
        RealCandidateEvidence {
            id: id.into(),
            manifest: path.into(),
            manifest_valid,
            weights_present,
            weight_hash_verified,
            intended_use_approved,
            runtime_available,
            status: status.into(),
            inference_calls: 0,
            evaluated_image_ids: Vec::new(),
            failed_image_reasons: BTreeMap::new(),
            per_image_scores: Vec::new(),
            performance: None,
            mean_agreement: None,
        }
    })
    .collect()
}

enum RealSegmenter {
    U2(bgremove_ort::U2netSegmenter),
    Isnet(bgremove_ort::IsnetSegmenter),
    Tracer(bgremove_ort::TracerB7Segmenter),
    Biref(bgremove_ort::BirefnetSegmenter),
}

impl RealSegmenter {
    fn predict(&self, image: &bgremove_core::CanonicalImage) -> Result<bgremove_core::AlphaMask> {
        match self {
            Self::U2(s) => s.predict(image),
            Self::Isnet(s) => s.predict(image),
            Self::Tracer(s) => s.predict(image),
            Self::Biref(s) => s.predict(image),
        }
    }
}

fn build_real_segmenter(
    id: &str,
    manifest: &ModelManifest,
    manifest_path: &Path,
    runtime: &Path,
) -> Result<RealSegmenter> {
    let workers = 1;
    let provider = bgremove_ort::RequestedProvider::Cpu;
    match id {
        "P0-u2-cf" => Ok(RealSegmenter::U2(bgremove_ort::U2netSegmenter::new(
            manifest,
            manifest_path,
            runtime,
            workers,
            provider,
            false,
        )?)),
        "P1-isnet-fast" => Ok(RealSegmenter::Isnet(bgremove_ort::IsnetSegmenter::new(
            manifest,
            manifest_path,
            runtime,
            workers,
            manifest.preprocessing_profile,
            provider,
            false,
        )?)),
        "P2-tracer-fba" => Ok(RealSegmenter::Tracer(bgremove_ort::TracerB7Segmenter::new(
            manifest,
            manifest_path,
            runtime,
            workers,
            provider,
            false,
        )?)),
        "P3-biref-vit" => Ok(RealSegmenter::Biref(bgremove_ort::BirefnetSegmenter::new(
            manifest,
            manifest_path,
            runtime,
            workers,
            provider,
            false,
        )?)),
        _ => anyhow::bail!("no approved real segmenter adapter for {id}"),
    }
}

fn real_pipeline_from_id(id: &str, strength: f32) -> Result<RankedPipeline> {
    let parts: Vec<&str> = id.split('+').collect();
    let (segmenter, trimap, refiner, foreground) = match parts.as_slice() {
        [segmenter] => (*segmenter, "none", "none", "original"),
        [segmenter, trimap, refiner] => (*segmenter, *trimap, *refiner, "original"),
        [segmenter, trimap, refiner, foreground] => (*segmenter, *trimap, *refiner, *foreground),
        _ => anyhow::bail!("malformed real pipeline id {id}"),
    };
    ensure!(
        matches!(
            segmenter,
            "P0-u2-cf" | "P1-isnet-fast" | "P2-tracer-fba" | "P3-biref-vit"
        ),
        "unknown real segmenter id {segmenter}"
    );
    ensure!(
        matches!(trimap, "none" | "symmetric"),
        "unknown real trimap {trimap}"
    );
    ensure!(
        matches!(refiner, "none" | "closed-form" | "vitmatte"),
        "unknown real refiner {refiner}"
    );
    ensure!(
        matches!(foreground, "original" | "fast" | "multilevel" | "fba"),
        "unknown real foreground {foreground}"
    );
    Ok(RankedPipeline {
        id: format!("{id}@{strength:.3}"),
        segmenter: segmenter.into(),
        trimap: trimap.into(),
        refiner: refiner.into(),
        foreground: foreground.into(),
        strength,
        tune: empty_score(),
        validation: None,
        selected_on_tune_only: true,
    })
}

fn resolve_real_config(strength: f32) -> Result<RealResolvedConfig> {
    ensure!(
        strength.is_finite() && (0.0..=1.0).contains(&strength),
        "real sweep strength outside domain"
    );
    let foreground_threshold = (200.0 + 55.0 * strength).round() as u8;
    let background_threshold = (8.0 + 24.0 * strength).round() as u8;
    let erode_size = 1u32 + (3.0 * strength).round() as u32;
    let closed_form_radius = 1usize + (3.0 * strength).round() as usize;
    Ok(RealResolvedConfig {
        strength,
        foreground_threshold,
        background_threshold,
        erode_size,
        closed_form_radius,
    })
}

fn real_alpha_and_foreground(
    p: &RankedPipeline,
    image: &SyntheticImage,
    raw: &BTreeMap<(String, String), Vec<f32>>,
) -> Result<(Vec<f32>, Vec<[f32; 3]>)> {
    let config = resolve_real_config(p.strength)?;
    let values = raw
        .get(&(p.segmenter.clone(), image.id.clone()))
        .context("real raw alpha missing from cache")?;
    ensure!(
        values.len() == image.rgb.len(),
        "real raw alpha dimensions mismatch"
    );
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "real raw alpha contains non-finite values"
    );
    let canonical =
        bgremove_core::CanonicalImage::new(image.width, image.height, image.rgb.clone())?;
    let coarse = bgremove_core::AlphaMask::new(image.width, image.height, values.clone())?;
    let trimap_values: Vec<u8> = values
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let trimap = bgremove_matting::rembg_symmetric_trimap(
        &trimap_values,
        image.width,
        image.height,
        bgremove_matting::RembgTrimapConfig {
            foreground_threshold: config.foreground_threshold,
            background_threshold: config.background_threshold,
            erode_size: config.erode_size,
        },
    )?;
    let rgb = bgremove_core::RgbImageF32::new(image.width, image.height, image.rgb.clone())?;
    let refined = match p.refiner.as_str() {
        "none" => (coarse.clone(), rgb.clone()),
        "closed-form" => {
            let result = bgremove_matting::refine_closed_form_with_coarse(
                &rgb,
                &coarse,
                &trimap,
                &bgremove_matting::ClosedFormConfig {
                    radius: config.closed_form_radius,
                    ..Default::default()
                },
            )?;
            (result.alpha, result.foreground)
        }
        "vitmatte" => anyhow::bail!(
            "ViTMatte refiner unavailable: approved checkpoint/runtime session is not initialized"
        ),
        other => anyhow::bail!("unknown real refiner {other}"),
    };
    let alpha = refined.0.data().to_vec();
    let foreground =
        estimate_foreground_rgb(p, refined.1.data(), &alpha, image.width, image.height)?;
    let _ = canonical;
    Ok((alpha, foreground))
}

fn real_score_one(
    p: &RankedPipeline,
    image: &SyntheticImage,
    raw: &BTreeMap<(String, String), Vec<f32>>,
) -> Result<PerImageScore> {
    let (alpha, foreground) = real_alpha_and_foreground(p, image, raw)?;
    let mut evaluator_image = image.clone();
    evaluator_image.rgb = foreground;
    let mut evaluator_raw = BTreeMap::new();
    evaluator_raw.insert((p.segmenter.clone(), image.id.clone()), alpha);
    let evaluator_pipeline = RankedPipeline {
        id: p.id.clone(),
        segmenter: p.segmenter.clone(),
        trimap: "none".into(),
        refiner: "none".into(),
        foreground: "original".into(),
        strength: 1.0,
        tune: empty_score(),
        validation: None,
        selected_on_tune_only: true,
    };
    let score = score_one_mode(
        &evaluator_pipeline,
        &evaluator_image,
        &evaluator_raw,
        EvaluationMode::RealMechanisms,
    );
    ensure!(
        !score.failure,
        "real pipeline score failed: {:?}",
        score.failure_reason
    );
    Ok(score)
}

fn real_score_pipeline(
    p: &RankedPipeline,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    split: Split,
) -> Result<ScoreSummary> {
    let scores: Vec<PerImageScore> = images
        .iter()
        .filter(|image| image.split == split)
        .map(|image| match real_score_one(p, image, raw) {
            Ok(score) => score,
            Err(error) => failure_score(image, format!("real stage failed: {error:#}")),
        })
        .collect();
    ensure!(!scores.is_empty(), "real pipeline has no {split:?} records");
    ensure!(
        scores.iter().all(|s| s.agreement.is_finite()),
        "real pipeline has non-finite scores"
    );
    Ok(summarize(scores))
}

fn real_stage(
    stage: &'static str,
    ids: Vec<String>,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    survivors: usize,
) -> Result<(StageSummary, Vec<String>)> {
    let mut ranked = Vec::new();
    let mut skipped = Vec::new();
    for id in ids {
        let p = match real_pipeline_from_id(&id, 1.0) {
            Ok(p) => p,
            Err(error) => {
                skipped.push(format!("{id}: {error:#}"));
                continue;
            }
        };
        match real_score_pipeline(&p, images, raw, Split::Tune) {
            Ok(score) => ranked.push((id, score.mean_agreement)),
            Err(error) => skipped.push(format!("{}: {error:#}", p.id)),
        }
    }
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let survivor_ids: Vec<String> = ranked.iter().take(survivors).map(|x| x.0.clone()).collect();
    Ok((
        StageSummary {
            stage,
            evaluated_candidates: ranked.len(),
            survivor_ids: survivor_ids.clone(),
            tune_image_count: images.iter().filter(|i| i.split == Split::Tune).count(),
            validation_image_count: images
                .iter()
                .filter(|i| i.split == Split::Validation)
                .count(),
            blind_evaluated: false,
            evaluated_ranking: ranked
                .iter()
                .map(|(id, tune_mean)| StageCandidateEvidence {
                    id: id.clone(),
                    tune_mean: *tune_mean,
                    selected: survivor_ids.contains(id),
                })
                .collect(),
            skipped_reasons: skipped,
        },
        survivor_ids,
    ))
}

fn real_loo_tournament(
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    candidate_ids: Vec<String>,
    blind_ids: Vec<String>,
    performance: &BTreeMap<String, PerformanceEvidence>,
) -> Result<RealTournamentEvidence> {
    ensure!(
        images.len() == 6,
        "fixed-arena LOO requires exactly six images"
    );
    let image_ids = images
        .iter()
        .map(|image| image.id.clone())
        .collect::<Vec<_>>();
    let partitions = loo_partitions(&image_ids)?;
    let mut folds = Vec::new();
    let mut held_scores = Vec::new();
    let mut selected_configs = Vec::new();
    let mut first_stages = Vec::new();
    for (held_id, training_ids) in partitions {
        let held_original = images
            .iter()
            .find(|image| image.id == held_id)
            .context("LOO held-out image missing")?;
        let mut train = Vec::new();
        for training_id in &training_ids {
            let mut copy = images
                .iter()
                .find(|image| &image.id == training_id)
                .context("LOO training image missing")?
                .clone();
            copy.split = Split::Tune;
            train.push(copy);
        }
        let mut held = held_original.clone();
        held.split = Split::Validation;
        let (stage1, s1) = real_stage("raw-segmenters", candidate_ids.clone(), &train, raw, 3)?;
        let mut stage2_ids = Vec::new();
        for id in &s1 {
            stage2_ids.push(format!("{id}+none+none"));
            stage2_ids.push(format!("{id}+symmetric+closed-form"));
        }
        let (stage2, s2) = real_stage("trimap-refiner", stage2_ids, &train, raw, 3)?;
        let mut stage3_ids = Vec::new();
        for id in &s2 {
            for foreground in ["original", "fast", "multilevel"] {
                stage3_ids.push(format!("{id}+{foreground}"));
            }
            if id.starts_with("P2-tracer-fba+") {
                stage3_ids.push(format!("{id}+fba"));
            }
        }
        let (stage3, s3) = real_stage("foreground-estimator", stage3_ids, &train, raw, 2)?;
        if first_stages.is_empty() {
            first_stages = vec![stage1.clone(), stage2.clone(), stage3.clone()];
        }
        let sweep = sweep_report();
        let mut strengths = sweep.grid_values.clone();
        strengths.extend(sweep.random_values);
        strengths.extend(sweep.latin_hypercube_values);
        strengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        strengths.dedup();
        let mut tuned = Vec::new();
        for id in &s3 {
            let values = if id.split('+').nth(2) == Some("none") {
                vec![1.0]
            } else {
                strengths.clone()
            };
            for strength in values {
                let p = real_pipeline_from_id(id, strength)?;
                let score = real_score_pipeline(&p, &train, raw, Split::Tune)?;
                tuned.push((p, score));
            }
        }
        tuned.sort_by(|a, b| compare_score(&b.1, &a.1).then_with(|| a.0.id.cmp(&b.0.id)));
        let (selected, tune_summary) = tuned
            .into_iter()
            .next()
            .context("LOO fold has no tuned survivor")?;
        let held_score = real_score_pipeline(&selected, &[held.clone()], raw, Split::Validation)?;
        let score = held_score
            .per_image
            .first()
            .cloned()
            .context("LOO held-out score missing")?;
        selected_configs.push(selected.id.clone());
        held_scores.push(score.clone());
        folds.push(LooFoldEvidence { held_out_id: held.id.clone(), training_ids: train.iter().map(|image| image.id.clone()).collect(), selection_source: "real fixed-arena LOO: exactly five other records; held-out record excluded from selection".into(), selected_config: Some(selected.id), held_out_evaluated: true, status: format!("completed; tune mean {:.8}; held-out agreement {:.8}", tune_summary.mean_agreement, score.agreement) });
    }
    let selected_validation_summary = summarize(held_scores.clone());
    let baseline = real_pipeline_from_id("P1-isnet-fast+none+none+original", 1.0)?;
    let baseline_scores = images
        .iter()
        .map(|image| {
            let mut copy = image.clone();
            copy.split = Split::Validation;
            real_score_one(&baseline, &copy, raw)
                .unwrap_or_else(|error| failure_score(&copy, format!("baseline failed: {error:#}")))
        })
        .collect::<Vec<_>>();
    let baseline_summary = summarize(baseline_scores);
    let paired = paired_comparison_named(
        &baseline_summary,
        &selected_validation_summary,
        &baseline.id,
        "LOO-selected-held-out",
    )
    .ok();
    let mut tag_aggregates = BTreeMap::new();
    let mut critical = BTreeMap::new();
    for tag in images
        .iter()
        .flat_map(|image| image.tags.iter())
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let ids: std::collections::BTreeSet<&str> = images
            .iter()
            .filter(|image| image.tags.contains(&tag))
            .map(|image| image.id.as_str())
            .collect();
        tag_aggregates.insert(
            tag.clone(),
            summarize(
                held_scores
                    .iter()
                    .filter(|score| ids.contains(score.id.as_str()))
                    .cloned()
                    .collect(),
            ),
        );
        let base = baseline_summary
            .per_image
            .iter()
            .filter(|score| ids.contains(score.id.as_str()))
            .map(|score| score.agreement)
            .collect::<Vec<_>>();
        let chal = held_scores
            .iter()
            .filter(|score| ids.contains(score.id.as_str()))
            .map(|score| score.agreement)
            .collect::<Vec<_>>();
        let delta = if base.is_empty() {
            None
        } else {
            Some(
                chal.iter()
                    .zip(base.iter())
                    .map(|(c, b)| c - b)
                    .sum::<f64>()
                    / base.len() as f64,
            )
        };
        critical.insert(
            tag,
            CriticalCategoryEvidence {
                budget: -0.02,
                baseline_ids: ids.iter().map(|id| (*id).to_owned()).collect(),
                challenger_ids: ids.iter().map(|id| (*id).to_owned()).collect(),
                delta,
                status: match delta {
                    Some(value) if value >= -0.02 => "passed".into(),
                    Some(_) => "failed".into(),
                    None => "insufficient-data".into(),
                },
            },
        );
    }
    let one_frozen_config_across_inputs = same_resolved_config(&selected_configs);
    let selected_config = one_frozen_config_across_inputs
        .then(|| selected_configs.first().cloned())
        .flatten();
    let mut resolved_configs = BTreeMap::new();
    for config in &selected_configs {
        if let Some((_, strength)) = config.rsplit_once('@') {
            if let Ok(value) = strength.parse() {
                resolved_configs.insert(config.clone(), resolve_real_config(value)?);
            }
        }
    }
    let _ = performance;
    let _legacy_declared_blind_ids = blind_ids;
    Ok(RealTournamentEvidence { legal_selection_ids: images.iter().map(|image| image.id.clone()).collect(), blind_ids_excluded: Vec::new(), stages:first_stages, selected_config, validation_ranked:selected_configs, paired_ci_available:paired.is_some(), loo_status:"completed: six fixed-arena held-out predictions; declared split labels overridden by Section 6.0 protocol".into(), selected_tune_summary:None, selected_validation_summary:Some(selected_validation_summary), paired_comparison:paired, critical_category_deltas:critical, tag_aggregates, resolved_configs, loo_folds: folds, one_frozen_config_across_inputs })
}

fn same_resolved_config(configs: &[String]) -> bool {
    configs
        .first()
        .is_some_and(|first| configs.iter().all(|config| config == first))
}

/// Partition the fixed six-image arena into six disjoint folds.  This is kept
/// independent of model execution so the no-ORT path can still prove the
/// selection/evaluation isolation contract.
fn loo_partitions(ids: &[String]) -> Result<Vec<(String, Vec<String>)>> {
    ensure!(ids.len() == 6, "fixed-arena LOO requires six unique IDs");
    let mut unique = std::collections::BTreeSet::new();
    for id in ids {
        ensure!(
            !id.is_empty() && unique.insert(id),
            "fixed-arena LOO IDs must be non-empty and unique"
        );
    }
    Ok(ids
        .iter()
        .map(|held| {
            (
                held.clone(),
                ids.iter().filter(|id| *id != held).cloned().collect(),
            )
        })
        .collect())
}

fn real_staged_tournament(
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    candidate_ids: Vec<String>,
    blind_ids: Vec<String>,
    performance: &BTreeMap<String, PerformanceEvidence>,
) -> Result<RealTournamentEvidence> {
    if images.len() == 6 {
        return real_loo_tournament(images, raw, candidate_ids, blind_ids, performance);
    }
    let (stage1, s1) = real_stage("raw-segmenters", candidate_ids, images, raw, 3)?;
    let mut stage2_ids = Vec::new();
    for id in &s1 {
        stage2_ids.push(format!("{id}+none+none"));
        stage2_ids.push(format!("{id}+symmetric+closed-form"));
    }
    let (stage2, s2) = real_stage("trimap-refiner", stage2_ids, images, raw, 3)?;
    let mut stage3_ids = Vec::new();
    for id in &s2 {
        for foreground in ["original", "fast", "multilevel"] {
            stage3_ids.push(format!("{id}+{foreground}"));
        }
        if id.starts_with("P2-tracer-fba+") {
            stage3_ids.push(format!("{id}+fba"));
        }
    }
    let (stage3, s3) = real_stage("foreground-estimator", stage3_ids, images, raw, 2)?;
    let sweep = sweep_report();
    let mut strengths = sweep.grid_values.clone();
    strengths.extend(sweep.random_values.iter().copied());
    strengths.extend(sweep.latin_hypercube_values.iter().copied());
    strengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    strengths.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);
    let mut tuned = Vec::new();
    let mut resolved_configs = BTreeMap::new();
    for id in &s3 {
        let strengths_for_pipeline: Vec<f32> = if id.split('+').nth(2) == Some("none") {
            vec![1.0]
        } else {
            strengths.clone()
        };
        for strength in &strengths_for_pipeline {
            let p = real_pipeline_from_id(id, *strength)?;
            resolved_configs.insert(p.id.clone(), resolve_real_config(*strength)?);
            let tune = real_score_pipeline(&p, images, raw, Split::Tune)?;
            tuned.push((p, tune));
        }
    }
    tuned.sort_by(|a, b| compare_score(&b.1, &a.1).then_with(|| a.0.id.cmp(&b.0.id)));
    let (mut selected, selected_tune_summary) = tuned
        .into_iter()
        .next()
        .context("real sweep produced no candidate")?;
    let validation = real_score_pipeline(&selected, images, raw, Split::Validation).ok();
    let mut selected_tune_summary = selected_tune_summary;
    if let Some(perf) = performance.get(&selected.segmenter) {
        selected_tune_summary.cold_start_ms = Some(perf.cold_start_ms);
        selected_tune_summary.warm_median_ms = Some(perf.warm_median_ms);
        selected_tune_summary.warm_p95_ms = Some(perf.warm_p95_ms);
        selected_tune_summary.throughput_images_per_sec = Some(perf.throughput_images_per_sec);
        selected_tune_summary.model_bytes = perf.model_bytes;
        selected_tune_summary.peak_rss_bytes = perf.peak_rss_bytes;
        selected_tune_summary.provider = Some(perf.active_provider.clone());
        selected_tune_summary.hardware = Some(perf.hardware.clone());
    }
    selected.tune = selected_tune_summary.clone();
    selected.validation = validation.clone();
    let baseline_validation = real_pipeline_from_id("P1-isnet-fast+none+none+original", 1.0)
        .ok()
        .and_then(|baseline| real_score_pipeline(&baseline, images, raw, Split::Validation).ok());
    let paired = validation.as_ref().and_then(|summary| {
        let baseline = real_pipeline_from_id("P1-isnet-fast+none+none+original", 1.0).ok()?;
        let base = real_score_pipeline(&baseline, images, raw, Split::Validation).ok()?;
        (base.per_image.len() >= 3 && summary.per_image.len() >= 3)
            .then(|| paired_comparison_named(&base, summary, &baseline.id, &selected.id).ok())
            .flatten()
    });
    let mut critical_category_deltas = BTreeMap::new();
    let mut tag_aggregates = BTreeMap::new();
    if let Some(summary) = &validation {
        for tag in images
            .iter()
            .filter(|image| image.split == Split::Validation)
            .flat_map(|image| image.tags.iter())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
        {
            let ids: std::collections::BTreeSet<&str> = images
                .iter()
                .filter(|image| {
                    image.split == Split::Validation && image.tags.iter().any(|value| value == &tag)
                })
                .map(|image| image.id.as_str())
                .collect();
            tag_aggregates.insert(
                tag,
                summarize(
                    summary
                        .per_image
                        .iter()
                        .filter(|score| ids.contains(score.id.as_str()))
                        .cloned()
                        .collect(),
                ),
            );
        }
    }
    for category in ["portrait-character", "emissive-artwork"] {
        let ids: Vec<String> = images
            .iter()
            .filter(|image| {
                image.split == Split::Validation && image.tags.iter().any(|tag| tag == category)
            })
            .map(|image| image.id.clone())
            .collect();
        let (delta, status) =
            if let (Some(base), Some(challenger)) = (&baseline_validation, &validation) {
                let base_scores = base
                    .per_image
                    .iter()
                    .filter(|score| ids.contains(&score.id))
                    .map(|score| score.agreement)
                    .collect::<Vec<_>>();
                let challenger_scores = challenger
                    .per_image
                    .iter()
                    .filter(|score| ids.contains(&score.id))
                    .map(|score| score.agreement)
                    .collect::<Vec<_>>();
                if !ids.is_empty() && base_scores.len() == challenger_scores.len() {
                    let value = challenger_scores
                        .iter()
                        .zip(base_scores.iter())
                        .map(|(c, b)| c - b)
                        .sum::<f64>()
                        / ids.len() as f64;
                    (
                        Some(value),
                        if value >= -0.02 { "passed" } else { "failed" }.to_owned(),
                    )
                } else {
                    (None, "insufficient-data".to_owned())
                }
            } else {
                (None, "not-run".to_owned())
            };
        critical_category_deltas.insert(
            category.to_owned(),
            CriticalCategoryEvidence {
                budget: -0.02,
                baseline_ids: ids.clone(),
                challenger_ids: ids,
                delta,
                status,
            },
        );
    }
    Ok(RealTournamentEvidence {
        legal_selection_ids: images
            .iter()
            .filter(|r| r.split != Split::Blind)
            .map(|r| r.id.clone())
            .collect(),
        blind_ids_excluded: blind_ids,
        stages: vec![stage1, stage2, stage3],
        selected_config: Some(selected.id.clone()),
        validation_ranked: vec![selected.id.clone()],
        paired_ci_available: paired.is_some(),
        loo_status: if images.len() == 6 {
            "real six-image LOO required before promotion".into()
        } else {
            "unavailable: fewer than six legal non-blind records; blind records excluded from selection".into()
        },
        selected_tune_summary: Some(selected_tune_summary),
        selected_validation_summary: validation,
        paired_comparison: paired,
        critical_category_deltas,
        tag_aggregates,
        resolved_configs,
        loo_folds: Vec::new(),
        one_frozen_config_across_inputs: false,
    })
}

fn run_real_arena(evidence: &mut RealArenaEvidence, root: &Path, output: &Path) -> Result<()> {
    let runtime_os = std::env::var_os("ORT_DYLIB").context("ORT_DYLIB is not set")?;
    let runtime = PathBuf::from(runtime_os);
    ensure!(
        runtime.is_file(),
        "ORT_DYLIB is not a regular file: {}",
        runtime.display()
    );
    let records = evidence.records.clone();
    // Section 6.0 explicitly overrides declared split labels for this fixed
    // six-image arena: every record participates exactly once as held-out.
    let legal_records: Vec<RealArenaRecord> = records.clone();
    let blind_ids: Vec<String> = records
        .iter()
        .filter(|record| record.declared_split == "blind")
        .map(|record| record.id.clone())
        .collect();
    let mut real_images = Vec::new();
    for record in &legal_records {
        let input = load_canonical(&root.join(&record.input))?;
        let target = load_canonical(&root.join(&record.target))?;
        real_images.push(SyntheticImage {
            id: record.id.clone(),
            split: match record.declared_split.as_str() {
                "tune" => Split::Tune,
                "validation" => Split::Validation,
                _ => Split::Blind,
            },
            width: input.width(),
            height: input.height(),
            rgb: input.rgb().data().to_vec(),
            reference_rgb: target.rgb().data().to_vec(),
            truth: target.source_alpha().data().to_vec(),
            input_hash: record.input_rgb_sha256.clone(),
            tags: record.tags.clone(),
        });
    }
    let real_cache = RawMaskCache::new(&output.join("raw-cache-real"))?;
    let mut real_raw: BTreeMap<(String, String), Vec<f32>> = BTreeMap::new();
    let mut performance: BTreeMap<String, PerformanceEvidence> = BTreeMap::new();
    for candidate in &mut evidence.candidates {
        if !candidate.manifest_valid
            || !candidate.weights_present
            || !candidate.weight_hash_verified
            || !candidate.intended_use_approved
        {
            continue;
        }
        let manifest_path = root.parent().unwrap_or(root).join(&candidate.manifest);
        let manifest_text = fs::read_to_string(&manifest_path)
            .with_context(|| format!("read validated manifest {}", manifest_path.display()))?;
        let manifest = parse_toml(&manifest_text)?;
        let cold_start = Instant::now();
        let segmenter =
            match build_real_segmenter(&candidate.id, &manifest, &manifest_path, &runtime) {
                Ok(value) => value,
                Err(error) => {
                    candidate.status =
                        format!("not-run: runtime/model initialization failed: {error:#}");
                    continue;
                }
            };
        let manifest_hash = sha_bytes(&fs::read(&manifest_path)?);
        let mut inference_times_ms = Vec::new();
        let mut scores = Vec::new();
        for record in &legal_records {
            let input_path = root.join(&record.input);
            let target_path = root.join(&record.target);
            let input = load_canonical(&input_path)?;
            let _target = load_canonical(&target_path)?;
            let synthetic = real_images
                .iter()
                .find(|image| image.id == record.id)
                .cloned()
                .context("real image missing from prepared records")?;
            let cache_result = real_cache.load_or_compute(
                &record.input_rgb_sha256,
                &manifest_hash,
                input.width(),
                input.height(),
                || {
                    let started = Instant::now();
                    let result = segmenter.predict(&input).map(|mask| mask.data().to_vec());
                    inference_times_ms.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                },
            );
            let (mask_values, cache_hit) = match cache_result {
                Ok(value) => value,
                Err(error) => {
                    let reason = format!("inference/cache failed: {error:#}");
                    candidate
                        .failed_image_reasons
                        .insert(record.id.clone(), reason.clone());
                    candidate
                        .per_image_scores
                        .push(failure_score(&synthetic, reason.clone()));
                    if record.declared_split == "tune" {
                        scores.push(failure_score(&synthetic, reason));
                    }
                    continue;
                }
            };
            if cache_hit {
                let started = Instant::now();
                if segmenter.predict(&input).is_ok() {
                    inference_times_ms.push(started.elapsed().as_secs_f64() * 1000.0);
                }
            }
            let expected_pixels = (input.width() as usize)
                .checked_mul(input.height() as usize)
                .context("real pixel count overflow")?;
            if mask_values.len() != expected_pixels || mask_values.iter().any(|v| !v.is_finite()) {
                let reason = "inference output invalid dimensions or non-finite values".to_owned();
                candidate
                    .failed_image_reasons
                    .insert(record.id.clone(), reason.clone());
                candidate
                    .per_image_scores
                    .push(failure_score(&synthetic, reason.clone()));
                if record.declared_split == "tune" {
                    scores.push(failure_score(&synthetic, reason));
                }
                continue;
            }
            real_raw.insert(
                (candidate.id.clone(), record.id.clone()),
                mask_values.clone(),
            );
            // Stage-1 evidence is tune-only. Validation records may be cached
            // here for the later frozen-config evaluation, but their scores are
            // intentionally absent until `validation_ranked` runs once.
            if record.declared_split == "tune" {
                let pipeline = RankedPipeline {
                    id: format!("{}+none+none+original@1.000", candidate.id),
                    segmenter: candidate.id.clone(),
                    trimap: "none".into(),
                    refiner: "none".into(),
                    foreground: "original".into(),
                    strength: 1.0,
                    tune: empty_score(),
                    validation: None,
                    selected_on_tune_only: false,
                };
                let mut raw = BTreeMap::new();
                raw.insert((candidate.id.clone(), record.id.clone()), mask_values);
                let score =
                    score_one_mode(&pipeline, &synthetic, &raw, EvaluationMode::RealMechanisms);
                scores.push(score.clone());
                candidate.per_image_scores.push(score);
            } else if record.declared_split == "validation" || record.declared_split == "blind" {
                let pipeline = RankedPipeline {
                    id: format!("{}+none+none+original@1.000", candidate.id),
                    segmenter: candidate.id.clone(),
                    trimap: "none".into(),
                    refiner: "none".into(),
                    foreground: "original".into(),
                    strength: 1.0,
                    tune: empty_score(),
                    validation: None,
                    selected_on_tune_only: false,
                };
                let mut raw = BTreeMap::new();
                raw.insert((candidate.id.clone(), record.id.clone()), mask_values);
                candidate.per_image_scores.push(score_one_mode(
                    &pipeline,
                    &synthetic,
                    &raw,
                    EvaluationMode::RealMechanisms,
                ));
            }
            candidate.inference_calls = candidate
                .inference_calls
                .checked_add(1)
                .context("real inference counter overflow")?;
            candidate.evaluated_image_ids.push(record.id.clone());
        }
        let summary = summarize(scores);
        ensure!(
            summary.mean_agreement.is_finite(),
            "{} real summary is non-finite",
            candidate.id
        );
        candidate.mean_agreement = Some(summary.mean_agreement);
        if !inference_times_ms.is_empty() {
            let mut ordered = inference_times_ms.clone();
            ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let median = ordered[(ordered.len() - 1) / 2];
            let p95 = ordered[((ordered.len() - 1) * 95) / 100];
            let model_bytes = manifest
                .resolve_model_path(&manifest_path)
                .ok()
                .and_then(|path| fs::metadata(path).ok())
                .map(|meta| meta.len());
            candidate.performance = Some(PerformanceEvidence {
                cold_start_ms: cold_start.elapsed().as_secs_f64() * 1000.0,
                warm_median_ms: median,
                warm_p95_ms: p95,
                throughput_images_per_sec: 1000.0
                    / (ordered.iter().sum::<f64>() / ordered.len() as f64),
                model_bytes,
                peak_rss_bytes: None,
                peak_rss_reason: "portable peak RSS measurement unavailable on this build".into(),
                requested_provider: "CPU".into(),
                active_provider: "CPU".into(),
                hardware: std::env::consts::OS.into(),
            });
            if let Some(value) = candidate.performance.clone() {
                performance.insert(candidate.id.clone(), value);
            }
        }
        candidate.status = "evaluated: real CPU ORT inference".into();
    }
    evidence.tournament = if real_raw.is_empty() {
        None
    } else {
        let candidate_ids = evidence
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.manifest_valid
                    && candidate.weights_present
                    && candidate.weight_hash_verified
                    && candidate.intended_use_approved
            })
            .map(|candidate| candidate.id.clone())
            .collect();
        let tournament = real_staged_tournament(
            &real_images,
            &real_raw,
            candidate_ids,
            blind_ids,
            &performance,
        )?;
        evidence.loo_folds = tournament.loo_folds.clone();
        Some(tournament)
    };
    if let Some(tournament) = &evidence.tournament {
        if let Some(config) = &tournament.selected_config {
            if let Some((base, strength_text)) = config.rsplit_once('@') {
                if let Ok(strength) = strength_text.parse::<f32>() {
                    if let Ok(pipeline) = real_pipeline_from_id(base, strength) {
                        // Reuse the exact real raw masks and image records for
                        // the bundle writer; no synthetic transform is used by
                        // the real scoring path.
                        write_artifacts_mode(
                            output,
                            &pipeline,
                            std::slice::from_ref(&pipeline),
                            &real_images,
                            &real_raw,
                            EvaluationMode::RealMechanisms,
                            "real-arena",
                        )?;
                    }
                }
            }
        }
    }
    if real_raw.is_empty() {
        let cause = evidence
            .candidates
            .iter()
            .find(|candidate| candidate.runtime_available && candidate.intended_use_approved)
            .map(|candidate| candidate.status.clone())
            .unwrap_or_else(|| "no eligible candidate".into());
        for fold in &mut evidence.loo_folds {
            fold.status = if cause.starts_with("not-run:") {
                cause.clone()
            } else {
                format!("not-run: {cause}")
            };
        }
    }
    evidence.runtime_status =
        "real CPU ORT inference attempted; per-candidate status is authoritative".into();
    Ok(())
}

fn metric_metadata() -> MetricMetadata {
    let backgrounds: Vec<String> = [
        "black",
        "white",
        "gray-50",
        "red",
        "green",
        "blue",
        "texture-seed-15",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let mut background_hashes = BTreeMap::new();
    let mut dimensions = BTreeMap::new();
    for image in synthetic_images() {
        dimensions.insert((image.width, image.height), ());
    }
    if let Ok(arena) = inspect_real_arena(&repo_path("test_images")) {
        for record in arena.records {
            dimensions.insert((record.width, record.height), ());
        }
    }
    for ((width, height), _) in dimensions {
        for (background_index, name) in backgrounds.iter().enumerate() {
            let mut pixels = Vec::new();
            pixels.extend_from_slice(&width.to_le_bytes());
            pixels.extend_from_slice(&height.to_le_bytes());
            pixels.extend_from_slice(b"m15-texture-seed-15");
            for index in 0..(width as usize).saturating_mul(height as usize) {
                for channel in 0..3 {
                    pixels.extend_from_slice(
                        &background_channel(index, channel, width, height, background_index)
                            .to_le_bytes(),
                    );
                }
            }
            background_hashes.insert(format!("{name}@{width}x{height}"), sha_bytes(&pixels));
        }
    }
    let metric_availability = BTreeMap::from([
        (
            "alpha-rmse".into(),
            MetricAvailability {
                available: true,
                reason: "pixelwise RMSE".into(),
            },
        ),
        (
            "alpha-sad".into(),
            MetricAvailability {
                available: true,
                reason: "normalized SAD = mean absolute alpha difference over all pixels".into(),
            },
        ),
        (
            "binary-iou".into(),
            MetricAvailability {
                available: true,
                reason: "thresholds 0.1, 0.5, 0.9".into(),
            },
        ),
        (
            "precision-recall".into(),
            MetricAvailability {
                available: true,
                reason: "threshold 0.5".into(),
            },
        ),
        (
            "topology".into(),
            MetricAvailability {
                available: true,
                reason: "1/(1+absolute 4-connected component plus hole disagreement) at alpha threshold 0.5".into(),
            },
        ),
        (
            "connectivity".into(),
            MetricAvailability {
                available: true,
                reason: "mean topology similarity over thresholds 0.1 through 0.9".into(),
            },
        ),
        (
            "gradient".into(),
            MetricAvailability {
                available: true,
                reason: "mean forward-neighbor alpha gradient magnitude difference".into(),
            },
        ),
        (
            "fractional-alpha".into(),
            MetricAvailability {
                available: true,
                reason: "mean absolute alpha error where reference alpha is in (0.02,0.98)".into(),
            },
        ),
        (
            "boundary-f1-scaled".into(),
            MetricAvailability {
                available: true,
                reason: "symmetric boundary precision/recall F1 at max(1,round(0.0015*image diagonal)) pixels".into(),
            },
        ),
        (
            "boundary-f1-1px".into(),
            MetricAvailability { available: true, reason: "symmetric boundary F1 at one-pixel Chebyshev tolerance".into() },
        ),
        (
            "boundary-f1-2px".into(),
            MetricAvailability { available: true, reason: "symmetric boundary F1 at two-pixel Chebyshev tolerance".into() },
        ),
        (
            "boundary-f1-4px".into(),
            MetricAvailability { available: true, reason: "symmetric boundary F1 at four-pixel Chebyshev tolerance".into() },
        ),
        (
            "boundary-band-alpha".into(),
            MetricAvailability { available: true, reason: "alpha MAE, RMSE, normalized SAD, soft IoU and binary IoU(.1,.5,.9) restricted to the dilated reference boundary band".into() },
        ),
        (
            "boundary-band-mae".into(),
            MetricAvailability { available: true, reason: "mean absolute alpha error on the dilated reference boundary band".into() },
        ),
        (
            "boundary-band-rmse".into(),
            MetricAvailability { available: true, reason: "RMSE alpha error on the dilated reference boundary band".into() },
        ),
        (
            "boundary-band-sad".into(),
            MetricAvailability { available: true, reason: "normalized SAD alpha error on the dilated reference boundary band".into() },
        ),
        (
            "boundary-band-soft-iou".into(),
            MetricAvailability { available: true, reason: "sum(min(candidate,reference))/sum(max(candidate,reference)) on the dilated reference boundary band".into() },
        ),
        (
            "boundary-band-iou-01-05-09".into(),
            MetricAvailability { available: true, reason: "binary IoU at thresholds 0.1, 0.5 and 0.9 on the dilated reference boundary band".into() },
        ),
        (
            "composite-mae".into(),
            MetricAvailability { available: true, reason: "mean linear-RGB absolute composite error averaged over seven declared backgrounds".into() },
        ),
        (
            "composite-psnr".into(),
            MetricAvailability { available: true, reason: "10*log10(1/MSE), with actual linear-RGB MSE averaged over seven declared backgrounds".into() },
        ),
        (
            "composite-ssim-full-roi".into(),
            MetricAvailability { available: true, reason: "deterministic sliding Gaussian 11x11 SSIM reported for full canvas and PhotoRoom ROI; ROI value enters agreement".into() },
        ),
        (
            "edge-colour".into(),
            MetricAvailability { available: true, reason: "linear-RGB candidate/reference colour absolute error on the reference boundary band".into() },
        ),
        (
            "foreground-colour".into(),
            MetricAvailability { available: true, reason: "alpha-weighted linear-RGB foreground error excluding pixels where both alphas are effectively zero".into() },
        ),
        (
            "performance-provenance".into(),
            MetricAvailability { available: true, reason: "real candidates record cold start, warm median/p95, throughput, model bytes, RSS availability, provider and hardware; synthetic performance is explicitly optional".into() },
        ),
    ]);
    MetricMetadata { formula: "0.35*S_alpha + 0.25*S_boundary + 0.15*S_selection + 0.25*mean(SSIM over declared composites); alpha_sad is normalized SAD", boundary_tolerance: "scaled max(1px, round(0.0015 * diagonal)) plus canonical 1px/2px/4px F1; symmetric matching", roi: "union(candidate/reference alpha bounding boxes), padded by 5% image diagonal", equal_image_weight: true, backgrounds, background_hashes, color_space: "linear-RGB metrics after declared sRGB decode", resize_rule: "no resize for canonical metrics; model geometry restores to source dimensions", ssim_implemented: true, critical_categories: BTreeMap::from([(String::from("portrait-character"), -0.02), (String::from("emissive-artwork"), -0.02)]), metric_availability }
}

fn blind_lifecycle() -> BlindLifecycle {
    let blind_used_for_sweeps = false;
    let blind_used_for_promotion = false;
    BlindLifecycle {
        state: "untouched",
        blind_used_for_sweeps,
        blind_used_for_promotion,
        split_retired_after_influence: blind_used_for_promotion,
        new_untouched_set_required: blind_used_for_sweeps || blind_used_for_promotion,
        fixed_arena_loo_override: false,
        legacy_declared_blind_ids: Vec::new(),
    }
}

fn blind_lifecycle_for_arena(arena: &RealArenaEvidence, loo_completed: bool) -> BlindLifecycle {
    let legacy_ids = arena
        .records
        .iter()
        .filter(|record| record.declared_split == "blind")
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    if loo_completed && arena.corpus_available && arena.records.len() == 6 && !legacy_ids.is_empty()
    {
        BlindLifecycle {
            state: "retired_after_fixed_arena_loo_override",
            blind_used_for_sweeps: true,
            blind_used_for_promotion: false,
            split_retired_after_influence: true,
            new_untouched_set_required: true,
            fixed_arena_loo_override: true,
            legacy_declared_blind_ids: legacy_ids,
        }
    } else {
        blind_lifecycle()
    }
}

fn synthetic_images() -> Vec<SyntheticImage> {
    let mut images = Vec::new();
    for split in [Split::Tune, Split::Validation] {
        for local in 0..6usize {
            let width = 24 + (local as u32 % 3) * 3;
            let height = 20 + (local as u32 % 2) * 4;
            let split_bias = if split == Split::Tune { 0.0 } else { 0.003 };
            let mut rgb = Vec::new();
            let mut truth = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    let edge = ((x as i32 - (width as i32 / 2)).abs()
                        + (y as i32 - (height as i32 / 2)).abs())
                        as f32;
                    let radius =
                        (width.min(height) as f32) * (0.29 + local as f32 * 0.008 + split_bias);
                    let alpha = ((radius + 1.5 - edge) / 3.0).clamp(0.0, 1.0);
                    truth.push(alpha);
                    rgb.push([
                        0.15 + x as f32 / width as f32 * 0.7,
                        0.2 + y as f32 / height as f32 * 0.6,
                        0.35 + (local as f32) * 0.03,
                    ]);
                }
            }
            let id = format!("synthetic-{}-{}", split.as_str(), local + 1);
            let mut input_bytes = Vec::new();
            for px in &rgb {
                for c in px {
                    input_bytes.extend_from_slice(&c.to_le_bytes());
                }
            }
            images.push(SyntheticImage {
                id,
                split,
                width,
                height,
                reference_rgb: rgb.clone(),
                rgb,
                truth,
                input_hash: sha_bytes(&input_bytes),
                tags: vec![if local % 2 == 0 {
                    "portrait-character".into()
                } else {
                    "emissive-artwork".into()
                }],
            });
        }
    }
    images
}

fn synthetic_raw_mask(image: &SyntheticImage, id: &str) -> Vec<f32> {
    let bias = match id {
        "P0-u2-cf" => 0.018,
        "P1-isnet-fast" => -0.011,
        "P2-tracer-fba" => 0.008,
        "P3-biref-vit" => -0.004,
        _ => 0.014,
    };
    image
        .rgb
        .iter()
        .enumerate()
        .map(|(i, rgb)| {
            // Synthetic inference is intentionally a function of RGB/runtime
            // features only. Reference alpha is evaluator-only and never
            // reaches the prediction path.
            let luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
            (0.28 + luminance * 0.72 + bias + (((i * 17 + id.len() * 3) % 11) as f32 - 5.0) * 0.001)
                .clamp(0.0, 1.0)
        })
        .collect()
}

type StageResult = (StageSummary, Vec<String>);

fn stage1(
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    registry: &[CandidateRegistration],
) -> Result<StageResult> {
    let mut ids: Vec<(String, f64)> = registry
        .iter()
        .filter(|r| r.id != "P5-hybrid")
        .map(|r| {
            let scores = score_variant(&r.id, images, raw, Split::Tune);
            (r.id.clone(), scores)
        })
        .collect();
    ids.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let survivors: Vec<String> = ids.iter().take(3).map(|(id, _)| id.clone()).collect();
    Ok((
        StageSummary {
            stage: "raw-segmenters",
            evaluated_candidates: ids.len(),
            survivor_ids: survivors.clone(),
            tune_image_count: images.iter().filter(|i| i.split == Split::Tune).count(),
            validation_image_count: images
                .iter()
                .filter(|i| i.split == Split::Validation)
                .count(),
            blind_evaluated: false,
            evaluated_ranking: ids
                .iter()
                .map(|(id, score)| StageCandidateEvidence {
                    id: id.clone(),
                    tune_mean: *score,
                    selected: survivors.contains(id),
                })
                .collect(),
            skipped_reasons: vec!["P5-hybrid reserved for M16".into()],
        },
        survivors,
    ))
}

fn stage2(
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    stage1: &StageResult,
) -> Result<StageResult> {
    let mut candidates = Vec::new();
    let mut skipped_reasons = Vec::new();
    for segmenter in &stage1.1 {
        for trimap in ["none", "symmetric"] {
            for refiner in ["none", "closed-form", "vitmatte"] {
                if refiner == "vitmatte"
                    && !matches!(segmenter.as_str(), "P3-biref-vit" | "P4-bria-vit")
                {
                    skipped_reasons.push(format!(
                        "{segmenter}+{trimap}+{refiner}: incompatible refiner family"
                    ));
                    continue;
                }
                let score = score_combo(
                    segmenter,
                    trimap,
                    refiner,
                    "original",
                    0.5,
                    images,
                    raw,
                    Split::Tune,
                );
                candidates.push((format!("{segmenter}+{trimap}+{refiner}"), score));
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    Ok((
        StageSummary {
            stage: "trimap-refiner",
            evaluated_candidates: candidates.len(),
            survivor_ids: candidates.iter().take(3).map(|v| v.0.clone()).collect(),
            tune_image_count: images.iter().filter(|i| i.split == Split::Tune).count(),
            validation_image_count: images
                .iter()
                .filter(|i| i.split == Split::Validation)
                .count(),
            blind_evaluated: false,
            evaluated_ranking: candidates
                .iter()
                .map(|(id, score)| StageCandidateEvidence {
                    id: id.clone(),
                    tune_mean: *score,
                    selected: candidates
                        .iter()
                        .take(3)
                        .any(|(selected, _)| selected == id),
                })
                .collect(),
            skipped_reasons,
        },
        candidates.iter().take(3).map(|v| v.0.clone()).collect(),
    ))
}

fn stage3(
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    stage2: &StageResult,
) -> Result<StageResult> {
    let fgs = ["original", "multilevel", "fast", "fba"];
    let mut candidates = Vec::new();
    let mut skipped_reasons = Vec::new();
    for base in &stage2.1 {
        for fg in fgs {
            if fg == "fba" && !base.starts_with("P2-tracer-fba+") {
                skipped_reasons.push(format!("{base}+{fg}: FBA foreground requires TRACER-B7"));
                continue;
            }
            let score = score_variant_from_base(base, fg, 0.5, images, raw, Split::Tune);
            candidates.push((format!("{base}+{fg}"), score));
        }
    }
    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    Ok((
        StageSummary {
            stage: "foreground-estimator",
            evaluated_candidates: candidates.len(),
            survivor_ids: candidates.iter().take(2).map(|v| v.0.clone()).collect(),
            tune_image_count: images.iter().filter(|i| i.split == Split::Tune).count(),
            validation_image_count: images
                .iter()
                .filter(|i| i.split == Split::Validation)
                .count(),
            blind_evaluated: false,
            evaluated_ranking: candidates
                .iter()
                .map(|(id, score)| StageCandidateEvidence {
                    id: id.clone(),
                    tune_mean: *score,
                    selected: candidates
                        .iter()
                        .take(2)
                        .any(|(selected, _)| selected == id),
                })
                .collect(),
            skipped_reasons,
        },
        candidates.iter().take(2).map(|v| v.0.clone()).collect(),
    ))
}

fn synthetic_loo(
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    stage3: &StageResult,
) -> Result<Vec<LooFoldEvidence>> {
    let arena: Vec<SyntheticImage> = images
        .iter()
        .filter(|i| i.split == Split::Validation)
        .cloned()
        .collect();
    ensure!(
        arena.len() == 6,
        "synthetic fixed arena must have six records"
    );
    let strengths = [0.25f32, 0.5, 0.75];
    let mut folds = Vec::new();
    for held_out in &arena {
        let training: Vec<SyntheticImage> = arena
            .iter()
            .filter(|i| i.id != held_out.id)
            .cloned()
            .collect();
        let mut candidates = Vec::new();
        for base in &stage3.1 {
            let (segmenter, trimap, refiner, foreground) = parse_pipeline_strict(base)?;
            for strength in strengths {
                let p = RankedPipeline {
                    id: format!("{base}@{strength:.3}"),
                    segmenter: segmenter.clone(),
                    trimap: trimap.clone(),
                    refiner: refiner.clone(),
                    foreground: foreground.clone(),
                    strength,
                    tune: empty_score(),
                    validation: None,
                    selected_on_tune_only: true,
                };
                candidates.push((p.clone(), score_pipeline_images(&p, &training, raw)));
            }
        }
        candidates.sort_by(|a, b| compare_score(&b.1, &a.1).then_with(|| a.0.id.cmp(&b.0.id)));
        let selected = candidates
            .first()
            .context("synthetic LOO produced no candidate")?;
        let held_out_score =
            score_pipeline_images(&selected.0, std::slice::from_ref(held_out), raw);
        ensure!(
            held_out_score.per_image.len() == 1
                && held_out_score.per_image[0].agreement.is_finite(),
            "synthetic LOO held-out score invalid"
        );
        folds.push(LooFoldEvidence {
            held_out_id: held_out.id.clone(),
            training_ids: training.into_iter().map(|i| i.id).collect(),
            selection_source: "five other arena images only".into(),
            selected_config: Some(selected.0.id.clone()),
            held_out_evaluated: true,
            status: "synthetic-contract-pass; real arena remains runtime-gated".into(),
        });
    }
    Ok(folds)
}

fn tune_and_rank(
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    stage3: &StageResult,
) -> Result<RankedPipeline> {
    let mut ranked = Vec::new();
    for base in &stage3.1 {
        for strength in sweep_report()
            .grid_values
            .iter()
            .chain(sweep_report().random_values.iter())
            .chain(sweep_report().latin_hypercube_values.iter())
        {
            let (segmenter, trimap, refiner, foreground) = parse_pipeline_strict(base)?;
            let tune = score_pipeline_parts(base, *strength, images, raw, Split::Tune)?;
            ensure!(
                !tune.per_image.is_empty(),
                "M15 candidate {base} has empty tune summary"
            );
            ensure!(
                tune.per_image
                    .iter()
                    .all(|score| score.agreement.is_finite()),
                "M15 candidate {base} has non-finite tune score"
            );
            let p = RankedPipeline {
                id: format!("{base}@{strength:.3}"),
                segmenter,
                trimap,
                refiner,
                foreground,
                strength: *strength,
                tune,
                validation: None,
                selected_on_tune_only: true,
            };
            ranked.push(p);
        }
    }
    ranked.sort_by(|a, b| compare_score(&b.tune, &a.tune).then_with(|| a.id.cmp(&b.id)));
    ranked
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("M15 tune produced no candidates"))
}

fn validation_ranked(
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    stage3: &StageResult,
) -> Result<Vec<RankedPipeline>> {
    let mut selected = tune_and_rank(images, raw, stage3)?;
    let validation = score_pipeline(&selected, images, raw, Split::Validation)?;
    selected.validation = Some(validation);
    Ok(vec![selected])
}

fn track_results(
    selected: &RankedPipeline,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
) -> BTreeMap<String, ScoreSummary> {
    let mut map = BTreeMap::new();
    let mut portraits = Vec::new();
    let mut art = Vec::new();
    for image in images.iter().filter(|i| i.split == Split::Validation) {
        if image.id.ends_with('1')
            || image.id.ends_with('2')
            || image.id.ends_with('5')
            || image.id.ends_with('6')
        {
            portraits.push(image.clone());
        } else {
            art.push(image.clone());
        }
    }
    let _ = raw;
    map.insert(
        "portrait-character".into(),
        score_pipeline_images(selected, &portraits, raw),
    );
    map.insert(
        "emissive-artwork".into(),
        score_pipeline_images(selected, &art, raw),
    );
    let validation: Vec<SyntheticImage> = images
        .iter()
        .filter(|image| image.split == Split::Validation)
        .cloned()
        .collect();
    map.insert(
        "tag:validation".into(),
        score_pipeline_images(selected, &validation, raw),
    );
    map.insert(
        "tag:all-evaluated".into(),
        score_pipeline_images(selected, images, raw),
    );
    map
}

fn score_pipeline(
    p: &RankedPipeline,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    split: Split,
) -> Result<ScoreSummary> {
    Ok(score_pipeline_images(
        p,
        &images
            .iter()
            .filter(|i| i.split == split)
            .cloned()
            .collect::<Vec<_>>(),
        raw,
    ))
}
fn score_pipeline_images(
    p: &RankedPipeline,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
) -> ScoreSummary {
    let scores: Vec<PerImageScore> = images
        .iter()
        .map(|image| score_one(p, image, raw))
        .collect();
    summarize(scores)
}

fn score_pipeline_parts(
    base: &str,
    strength: f32,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    split: Split,
) -> Result<ScoreSummary> {
    let (segmenter, trimap, refiner, foreground) = parse_pipeline_strict(base)?;
    let p = RankedPipeline {
        id: base.to_owned(),
        segmenter,
        trimap,
        refiner,
        foreground,
        strength,
        tune: empty_score(),
        validation: None,
        selected_on_tune_only: true,
    };
    score_pipeline(&p, images, raw, split)
}
fn score_variant(
    segmenter: &str,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    split: Split,
) -> f64 {
    let p = RankedPipeline {
        id: segmenter.into(),
        segmenter: segmenter.into(),
        trimap: "none".into(),
        refiner: "none".into(),
        foreground: "original".into(),
        strength: 0.5,
        tune: empty_score(),
        validation: None,
        selected_on_tune_only: false,
    };
    score_pipeline(&p, images, raw, split)
        .map(|s| s.mean_agreement)
        .unwrap_or(0.0)
}

#[allow(clippy::too_many_arguments)]
fn score_combo(
    segmenter: &str,
    trimap: &str,
    refiner: &str,
    foreground: &str,
    strength: f32,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    split: Split,
) -> f64 {
    let p = RankedPipeline {
        id: format!("{segmenter}+{trimap}+{refiner}+{foreground}"),
        segmenter: segmenter.to_owned(),
        trimap: trimap.to_owned(),
        refiner: refiner.to_owned(),
        foreground: foreground.to_owned(),
        strength,
        tune: empty_score(),
        validation: None,
        selected_on_tune_only: false,
    };
    score_pipeline(&p, images, raw, split)
        .map(|s| s.mean_agreement)
        .unwrap_or(0.0)
}
fn score_variant_from_base(
    base: &str,
    foreground: &str,
    strength: f32,
    images: &[SyntheticImage],
    raw: &BTreeMap<(String, String), Vec<f32>>,
    split: Split,
) -> f64 {
    let (segmenter, trimap, refiner, _) = parse_pipeline(base);
    let p = RankedPipeline {
        id: base.into(),
        segmenter,
        trimap,
        refiner,
        foreground: foreground.into(),
        strength,
        tune: empty_score(),
        validation: None,
        selected_on_tune_only: false,
    };
    score_pipeline(&p, images, raw, split)
        .map(|s| s.mean_agreement)
        .unwrap_or(0.0)
}

fn score_one(
    p: &RankedPipeline,
    image: &SyntheticImage,
    raw: &BTreeMap<(String, String), Vec<f32>>,
) -> PerImageScore {
    score_one_mode(p, image, raw, EvaluationMode::SyntheticContract)
}

fn failure_score(image: &SyntheticImage, reason: impl Into<String>) -> PerImageScore {
    PerImageScore {
        id: image.id.clone(),
        split: image.split.as_str(),
        agreement: 0.0,
        alpha_mae: 1.0,
        boundary_f1: 0.0,
        soft_iou: 0.0,
        roi_pixels: image.truth.len(),
        roi_alpha_mae: 1.0,
        roi_soft_iou: 0.0,
        composite_ssim: 0.0,
        alpha_rmse: 1.0,
        alpha_sad: 1.0,
        boundary_f1_tol1: 0.0,
        boundary_f1_tol2: 0.0,
        boundary_f1_tol4: 0.0,
        boundary_band_mae: 1.0,
        boundary_band_rmse: 1.0,
        boundary_band_sad: 1.0,
        boundary_band_soft_iou: 0.0,
        boundary_band_iou_01: 0.0,
        boundary_band_iou_05: 0.0,
        boundary_band_iou_09: 0.0,
        binary_iou_01: 0.0,
        binary_iou_05: 0.0,
        binary_iou_09: 0.0,
        precision: 0.0,
        recall: 0.0,
        composite_mae: 1.0,
        composite_psnr: 0.0,
        composite_ssim_full: 0.0,
        composite_ssim_roi: 0.0,
        boundary_band_f1: Some(0.0),
        topology_score: Some(0.0),
        connectivity_score: Some(0.0),
        edge_color_error: Some(1.0),
        foreground_color_error: Some(1.0),
        gradient_error: Some(1.0),
        fractional_alpha_error: Some(1.0),
        failure: true,
        failure_reason: Some(reason.into()),
    }
}

#[derive(Debug, Clone)]
struct PipelineEvaluation {
    coarse: Vec<f32>,
    cleaned: Vec<f32>,
    trimap: Vec<f32>,
    refined_alpha: Vec<f32>,
    foreground: Vec<[f32; 3]>,
}

fn evaluate_pipeline(
    p: &RankedPipeline,
    image: &SyntheticImage,
    raw: &BTreeMap<(String, String), Vec<f32>>,
    mode: EvaluationMode,
) -> Result<PipelineEvaluation> {
    let coarse = raw
        .get(&(p.segmenter.clone(), image.id.clone()))
        .cloned()
        .context("missing raw alpha")?;
    ensure!(
        coarse.len() == image.truth.len(),
        "raw alpha dimensions mismatch"
    );
    ensure!(
        coarse.iter().all(|v| v.is_finite()),
        "raw alpha is non-finite"
    );
    if mode == EvaluationMode::RealMechanisms {
        let (refined_alpha, foreground) = real_alpha_and_foreground(p, image, raw)?;
        let trimap = refined_alpha
            .iter()
            .map(|v| {
                if *v < 0.1 {
                    0.0
                } else if *v > 0.9 {
                    1.0
                } else {
                    0.5
                }
            })
            .collect();
        return Ok(PipelineEvaluation {
            coarse: coarse.clone(),
            cleaned: coarse,
            trimap,
            refined_alpha,
            foreground,
        });
    }
    let cleaned: Vec<f32> = coarse
        .iter()
        .map(|v| ((*v - 0.5) * p.strength + 0.5).clamp(0.0, 1.0))
        .collect();
    let trimap: Vec<f32> = cleaned
        .iter()
        .map(|v| {
            if *v < 0.1 {
                0.0
            } else if *v > 0.9 {
                1.0
            } else {
                0.5
            }
        })
        .collect();
    let mut refined_alpha = cleaned.clone();
    if p.trimap != "none" {
        for v in &mut refined_alpha {
            if *v > 0.2 && *v < 0.8 {
                *v = (*v * 0.85 + 0.075).clamp(0.0, 1.0);
            }
        }
    }
    if p.refiner != "none" {
        for v in &mut refined_alpha {
            *v = (*v * 0.97 + 0.015).clamp(0.0, 1.0);
        }
    }
    if p.foreground == "fast" || p.foreground == "fast-estimator" {
        for v in &mut refined_alpha {
            *v = (*v * 0.995 + 0.0025).clamp(0.0, 1.0);
        }
    }
    let foreground =
        estimate_foreground_rgb(p, &image.rgb, &refined_alpha, image.width, image.height)?;
    Ok(PipelineEvaluation {
        coarse,
        cleaned,
        trimap,
        refined_alpha,
        foreground,
    })
}

fn score_one_mode(
    p: &RankedPipeline,
    image: &SyntheticImage,
    raw: &BTreeMap<(String, String), Vec<f32>>,
    mode: EvaluationMode,
) -> PerImageScore {
    let evaluation = match evaluate_pipeline(p, image, raw, mode) {
        Ok(value) => value,
        Err(error) => {
            return failure_score(image, format!("pipeline evaluation failed: {error:#}"))
        }
    };
    let candidate = evaluation.refined_alpha;
    let candidate_rgb = evaluation.foreground;
    let mut mae: f64 = 0.0;
    let mut inter: f64 = 0.0;
    let mut union: f64 = 0.0;
    let mut min_x = image.width;
    let mut min_y = image.height;
    let mut max_x = 0;
    let mut max_y = 0;
    for (idx, (&a, &t)) in candidate.iter().zip(&image.truth).enumerate() {
        mae += f64::from((a - t).abs());
        let ab = a >= 0.5;
        let tb = t >= 0.5;
        inter += f64::from(a.min(t));
        union += f64::from(a.max(t));
        if ab || tb {
            min_x = min_x.min((idx as u32) % image.width);
            max_x = max_x.max((idx as u32) % image.width);
            min_y = min_y.min((idx as u32) / image.width);
            max_y = max_y.max((idx as u32) / image.width);
        }
    }
    let diagonal = ((image.width as f64).powi(2) + (image.height as f64).powi(2)).sqrt();
    let pad = (diagonal * 0.05).round().max(1.0) as u32;
    let (x0, y0, x1, y1) = if min_x <= max_x {
        (
            min_x.saturating_sub(pad),
            min_y.saturating_sub(pad),
            (max_x + pad).min(image.width - 1),
            (max_y + pad).min(image.height - 1),
        )
    } else {
        (0, 0, image.width - 1, image.height - 1)
    };
    let roi = (x1 - x0 + 1) * (y1 - y0 + 1);
    let mut roi_mae = 0.0f64;
    let mut roi_inter = 0.0f64;
    let mut roi_union = 0.0f64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let index = (y * image.width + x) as usize;
            let a = candidate[index] as f64;
            let t = image.truth[index] as f64;
            roi_mae += (a - t).abs();
            roi_inter += a.min(t);
            roi_union += a.max(t);
        }
    }
    let roi_alpha_mae = roi_mae / f64::from(roi);
    let roi_soft_iou = if roi_union > 0.0 {
        roi_inter / roi_union
    } else {
        1.0
    };
    let alpha_mae = mae / image.truth.len() as f64;
    let soft_iou = if union > 0.0 { inter / union } else { 1.0 };
    let boundary_diagonal = ((image.width as f64).powi(2) + (image.height as f64).powi(2)).sqrt();
    let tolerance = (0.0015 * boundary_diagonal).round().max(1.0) as u32;
    let boundary_f1 = boundary_f1_at_tolerance(
        &candidate,
        &image.truth,
        image.width,
        image.height,
        tolerance,
    );
    let boundary_f1_tol1 =
        boundary_f1_at_tolerance(&candidate, &image.truth, image.width, image.height, 1);
    let boundary_f1_tol2 =
        boundary_f1_at_tolerance(&candidate, &image.truth, image.width, image.height, 2);
    let boundary_f1_tol4 =
        boundary_f1_at_tolerance(&candidate, &image.truth, image.width, image.height, 4);
    let boundary_band = boundary_band_alpha_metrics(
        &candidate,
        &image.truth,
        image.width,
        image.height,
        tolerance,
    );
    let (candidate_components, candidate_holes) =
        topology_counts(&candidate, image.width, image.height);
    let (reference_components, reference_holes) =
        topology_counts(&image.truth, image.width, image.height);
    let topology_score = 1.0
        / (1.0
            + (candidate_components as f64 - reference_components as f64).abs()
            + (candidate_holes as f64 - reference_holes as f64).abs());
    let connectivity_score =
        connectivity_similarity(&candidate, &image.truth, image.width, image.height);
    let gradient_error = gradient_error_metric(&candidate, &image.truth, image.width, image.height);
    let fractional_alpha_error = fractional_alpha_error_metric(&candidate, &image.truth);
    let edge_color_error = color_band_error(
        &candidate_rgb,
        &image.reference_rgb,
        &image.truth,
        image.width,
        image.height,
        tolerance,
    );
    let foreground_color_error = foreground_color_error_metric(
        &candidate_rgb,
        &image.reference_rgb,
        &candidate,
        &image.truth,
    );
    let composite_ssim_full = composite_ssim_dims(
        &candidate_rgb,
        &image.reference_rgb,
        &candidate,
        &image.truth,
        image.width,
        image.height,
    );
    let composite_ssim = composite_ssim_roi(
        &candidate_rgb,
        &image.reference_rgb,
        &candidate,
        &image.truth,
        x0,
        y0,
        x1,
        y1,
        image.width,
    );
    // The full-canvas alpha metric is reported separately, while agreement
    // uses the declared ROI so empty background cannot dominate the ranking.
    let agreement = (0.35 * (1.0 - roi_alpha_mae)
        + 0.25 * boundary_f1
        + 0.15 * roi_soft_iou
        + 0.25 * composite_ssim)
        .clamp(0.0, 1.0);
    PerImageScore {
        id: image.id.clone(),
        split: image.split.as_str(),
        agreement,
        alpha_mae,
        boundary_f1,
        soft_iou,
        roi_pixels: roi as usize,
        roi_alpha_mae,
        roi_soft_iou,
        composite_ssim,
        alpha_rmse: (image
            .truth
            .iter()
            .zip(&candidate)
            .map(|(t, a)| f64::from((*a - *t).powi(2)))
            .sum::<f64>()
            / image.truth.len() as f64)
            .sqrt(),
        alpha_sad: alpha_mae,
        boundary_f1_tol1,
        boundary_f1_tol2,
        boundary_f1_tol4,
        binary_iou_01: binary_iou(&candidate, &image.truth, 0.1),
        binary_iou_05: binary_iou(&candidate, &image.truth, 0.5),
        binary_iou_09: binary_iou(&candidate, &image.truth, 0.9),
        precision: binary_precision_recall(&candidate, &image.truth, 0.5).0,
        recall: binary_precision_recall(&candidate, &image.truth, 0.5).1,
        composite_mae: composite_mae(
            &candidate_rgb,
            &image.reference_rgb,
            &candidate,
            &image.truth,
            image.width,
            image.height,
        ),
        composite_psnr: composite_psnr(
            &candidate_rgb,
            &image.reference_rgb,
            &candidate,
            &image.truth,
            image.width,
            image.height,
        ),
        composite_ssim_full,
        composite_ssim_roi: composite_ssim,
        boundary_band_f1: Some(boundary_band.0),
        boundary_band_mae: boundary_band.1,
        boundary_band_rmse: boundary_band.2,
        boundary_band_sad: boundary_band.3,
        boundary_band_soft_iou: boundary_band.4,
        boundary_band_iou_01: boundary_band.5,
        boundary_band_iou_05: boundary_band.6,
        boundary_band_iou_09: boundary_band.7,
        topology_score: Some(topology_score),
        connectivity_score: Some(connectivity_score),
        edge_color_error: Some(edge_color_error),
        foreground_color_error: Some(foreground_color_error),
        gradient_error: Some(gradient_error),
        fractional_alpha_error: Some(fractional_alpha_error),
        failure: false,
        failure_reason: None,
    }
}

fn boundary_f1_at_tolerance(
    candidate: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
    tolerance: u32,
) -> f64 {
    let c = boundary_points(candidate, width, height);
    let r = boundary_points(reference, width, height);
    if c.is_empty() && r.is_empty() {
        return 1.0;
    }
    if c.is_empty() || r.is_empty() {
        return 0.0;
    }
    let matches = |from: &[(u32, u32)], to: &[(u32, u32)]| -> usize {
        from.iter()
            .filter(|(x, y)| {
                to.iter()
                    .any(|(tx, ty)| x.abs_diff(*tx).max(y.abs_diff(*ty)) <= tolerance)
            })
            .count()
    };
    let precision = matches(&c, &r) as f64 / c.len() as f64;
    let recall = matches(&r, &c) as f64 / r.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn boundary_points(values: &[f32], width: u32, height: u32) -> Vec<(u32, u32)> {
    values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            let x = idx as u32 % width;
            let y = idx as u32 / width;
            let inside = *value >= 0.5;
            let neighbors = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            let is_boundary = inside
                && neighbors.iter().any(|(xx, yy)| {
                    *xx >= width
                        || *yy >= height
                        || (values[(*yy * width + *xx) as usize] >= 0.5) != inside
                });
            is_boundary.then_some((x, y))
        })
        .collect()
}

#[allow(dead_code)]
fn boundary_band_metrics(
    candidate: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
    tolerance: u32,
) -> (f64, f64) {
    let values = boundary_band_alpha_metrics(candidate, reference, width, height, tolerance);
    (values.0, values.1)
}

fn boundary_band_alpha_metrics(
    candidate: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
    tolerance: u32,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let ref_boundary = boundary_points(reference, width, height);
    let mut error = 0.0;
    let mut squared = 0.0;
    let mut sad = 0.0;
    let mut inter = 0.0;
    let mut union = 0.0;
    let mut count = 0usize;
    let mut band = vec![false; candidate.len()];
    for (i, value) in candidate.iter().enumerate() {
        let x = i as u32 % width;
        let y = i as u32 / width;
        let in_band = ref_boundary
            .iter()
            .any(|(bx, by)| x.abs_diff(*bx).max(y.abs_diff(*by)) <= tolerance);
        if in_band {
            band[i] = true;
            let d = f64::from((value - reference[i]).abs());
            error += d;
            squared += d * d;
            sad += d;
            inter += f64::from(value.min(reference[i]));
            union += f64::from(value.max(reference[i]));
            count += 1;
        }
    }
    let candidate_boundary = boundary_points(candidate, width, height)
        .into_iter()
        .filter(|(x, y)| band[(*y * width + *x) as usize])
        .collect::<Vec<_>>();
    let f1 = boundary_f1_sets(&candidate_boundary, &ref_boundary, tolerance);
    let mae = if count == 0 {
        0.0
    } else {
        error / count as f64
    };
    let rmse = if count == 0 {
        0.0
    } else {
        (squared / count as f64).sqrt()
    };
    let soft = if union == 0.0 { 1.0 } else { inter / union };
    let iou = |threshold: f32| {
        let mut i = 0usize;
        let mut u = 0usize;
        for (idx, is_band) in band.iter().enumerate() {
            if *is_band {
                let a = candidate[idx] >= threshold;
                let b = reference[idx] >= threshold;
                if a && b {
                    i += 1
                };
                if a || b {
                    u += 1
                }
            }
        }
        if u == 0 {
            1.0
        } else {
            i as f64 / u as f64
        }
    };
    (
        f1,
        mae,
        rmse,
        if count == 0 { 0.0 } else { sad / count as f64 },
        soft,
        iou(0.1),
        iou(0.5),
        iou(0.9),
    )
}

fn boundary_f1_sets(from: &[(u32, u32)], to: &[(u32, u32)], tolerance: u32) -> f64 {
    if from.is_empty() && to.is_empty() {
        return 1.0;
    }
    if from.is_empty() || to.is_empty() {
        return 0.0;
    }
    let m = |a: &[(u32, u32)], b: &[(u32, u32)]| {
        a.iter()
            .filter(|(x, y)| {
                b.iter()
                    .any(|(tx, ty)| x.abs_diff(*tx).max(y.abs_diff(*ty)) <= tolerance)
            })
            .count()
    };
    let p = m(from, to) as f64 / from.len() as f64;
    let r = m(to, from) as f64 / to.len() as f64;
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

fn topology_counts(values: &[f32], width: u32, height: u32) -> (usize, usize) {
    let w = width as usize;
    let h = height as usize;
    let n = w.saturating_mul(h);
    let mut seen = vec![false; n];
    let mut components = 0usize;
    for i in 0..n {
        if values[i] >= 0.5 && !seen[i] {
            components += 1;
            flood_binary(values, w, h, i, true, &mut seen);
        }
    }
    seen.fill(false);
    let mut background_components = 0usize;
    for i in 0..n {
        if values[i] < 0.5 && !seen[i] {
            background_components += 1;
            flood_binary(values, w, h, i, false, &mut seen);
        }
    }
    let touches = |start: usize| {
        start < w
            || start >= n.saturating_sub(w)
            || start.is_multiple_of(w)
            || start % w == w.saturating_sub(1)
    };
    let mut boundary_background = 0usize;
    seen.fill(false);
    for i in 0..n {
        if values[i] < 0.5 && !seen[i] && touches(i) {
            boundary_background += 1;
            flood_binary(values, w, h, i, false, &mut seen);
        }
    }
    (
        components,
        background_components.saturating_sub(boundary_background),
    )
}

fn flood_binary(
    values: &[f32],
    width: usize,
    height: usize,
    start: usize,
    foreground: bool,
    seen: &mut [bool],
) {
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(i) = stack.pop() {
        let x = i % width;
        let y = i / width;
        let neighbors = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for (nx, ny) in neighbors {
            if nx < width && ny < height {
                let j = ny * width + nx;
                if !seen[j] && (values[j] >= 0.5) == foreground {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
    }
}

fn connectivity_similarity(candidate: &[f32], reference: &[f32], width: u32, height: u32) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for threshold in [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let c: Vec<f32> = candidate
            .iter()
            .map(|v| if *v >= threshold { 1.0 } else { 0.0 })
            .collect();
        let r: Vec<f32> = reference
            .iter()
            .map(|v| if *v >= threshold { 1.0 } else { 0.0 })
            .collect();
        let (cc, ch) = topology_counts(&c, width, height);
        let (rc, rh) = topology_counts(&r, width, height);
        total += 1.0 / (1.0 + (cc as f64 - rc as f64).abs() + (ch as f64 - rh as f64).abs());
        count += 1;
    }
    total / count as f64
}

fn gradient_error_metric(candidate: &[f32], reference: &[f32], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;
    let mut total = 0.0;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let xr = (x + 1).min(w - 1);
            let yd = (y + 1).min(h - 1);
            let cg = ((candidate[y * w + xr] - candidate[i]).powi(2)
                + (candidate[yd * w + x] - candidate[i]).powi(2))
            .sqrt();
            let rg = ((reference[y * w + xr] - reference[i]).powi(2)
                + (reference[yd * w + x] - reference[i]).powi(2))
            .sqrt();
            total += f64::from((cg - rg).abs());
        }
    }
    total / (candidate.len().max(1) as f64)
}

fn fractional_alpha_error_metric(candidate: &[f32], reference: &[f32]) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for (c, r) in candidate.iter().zip(reference) {
        if *r > 0.02 && *r < 0.98 {
            total += f64::from((c - r).abs());
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn color_band_error(
    candidate: &[[f32; 3]],
    reference: &[[f32; 3]],
    truth: &[f32],
    width: u32,
    height: u32,
    tolerance: u32,
) -> f64 {
    let band = boundary_points(truth, width, height);
    let mut total = 0.0;
    let mut count = 0usize;
    for (i, (c, r)) in candidate.iter().zip(reference).enumerate() {
        let x = i as u32 % width;
        let y = i as u32 / width;
        if band
            .iter()
            .any(|(bx, by)| x.abs_diff(*bx).max(y.abs_diff(*by)) <= tolerance)
        {
            total += (0..3)
                .map(|ch| {
                    (srgb_to_linear(f64::from(c[ch])) - srgb_to_linear(f64::from(r[ch]))).abs()
                })
                .sum::<f64>()
                / 3.0;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn foreground_color_error_metric(
    candidate: &[[f32; 3]],
    reference: &[[f32; 3]],
    alpha: &[f32],
    truth: &[f32],
) -> f64 {
    let mut total = 0.0;
    let mut weights = 0.0;
    for ((c, r), (a, t)) in candidate.iter().zip(reference).zip(alpha.iter().zip(truth)) {
        let w = f64::from((*a).max(*t));
        if w > 0.02 {
            total += w
                * (0..3)
                    .map(|ch| {
                        (srgb_to_linear(f64::from(c[ch])) - srgb_to_linear(f64::from(r[ch]))).abs()
                    })
                    .sum::<f64>()
                / 3.0;
            weights += w;
        }
    }
    if weights == 0.0 {
        0.0
    } else {
        total / weights
    }
}

fn estimate_foreground_rgb(
    p: &RankedPipeline,
    rgb: &[[f32; 3]],
    alpha: &[f32],
    width: u32,
    height: u32,
) -> Result<Vec<[f32; 3]>> {
    let image = bgremove_core::CanonicalImage::new(width, height, rgb.to_vec())?;
    let mask = bgremove_core::AlphaMask::new(width, height, alpha.to_vec())?;
    let matte = bgremove_core::RefinedMatte::new(mask, None, None)?;
    let estimated = match p.foreground.as_str() {
        "fast" => FastForegroundEstimator::default().estimate(&image, &matte),
        "multilevel" => MultilevelForegroundEstimator::default().estimate(&image, &matte),
        "fba" => FbaForegroundEstimator.estimate(&image, &matte),
        "original" => OriginalRgbEstimator::default().estimate(&image, &matte),
        other => anyhow::bail!("unknown foreground estimator {other}"),
    }?;
    Ok(estimated.data().to_vec())
}

#[allow(dead_code)]
fn composite_ssim(rgb: &[[f32; 3]], candidate: &[f32], reference: &[f32]) -> f64 {
    let width = (rgb.len() as f64).sqrt().max(1.0) as u32;
    composite_ssim_dims(
        rgb,
        rgb,
        candidate,
        reference,
        width,
        (rgb.len() as u32).div_ceil(width),
    )
}

fn composite_ssim_dims(
    rgb: &[[f32; 3]],
    reference_rgb: &[[f32; 3]],
    candidate: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
) -> f64 {
    let mut total = 0.0;
    for background_index in 0..7 {
        let mut channels = 0.0;
        for channel in 0..3 {
            let a: Vec<f64> = rgb
                .iter()
                .zip(candidate)
                .enumerate()
                .map(|(index, (p, alpha))| {
                    srgb_to_linear(f64::from(p[channel])) * f64::from(*alpha)
                        + background_channel(index, channel, width, height, background_index)
                            * (1.0 - f64::from(*alpha))
                })
                .collect();
            let b: Vec<f64> = reference_rgb
                .iter()
                .zip(reference)
                .enumerate()
                .map(|(index, (p, alpha))| {
                    srgb_to_linear(f64::from(p[channel])) * f64::from(*alpha)
                        + background_channel(index, channel, width, height, background_index)
                            * (1.0 - f64::from(*alpha))
                })
                .collect();
            channels += global_ssim_2d(&a, &b, width, height);
        }
        total += channels / 3.0;
    }
    total / 7.0
}

fn global_ssim_2d(a: &[f64], b: &[f64], width: u32, height: u32) -> f64 {
    if a.len() != b.len() || a.len() != (width as usize).saturating_mul(height as usize) {
        return 0.0;
    }
    let n = a.len();
    let kernel: Vec<f64> = (-5..=5)
        .map(|x| (-(f64::from(x * x)) / (2.0 * 1.5f64.powi(2))).exp())
        .collect();
    let blur = |values: &[f64]| {
        let mut horizontal = vec![0.0; n];
        let mut output = vec![0.0; n];
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut norm = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    let xx =
                        (x as i32 + k as i32 - 5).clamp(0, width.saturating_sub(1) as i32) as u32;
                    sum += *weight * values[(y * width + xx) as usize];
                    norm += *weight;
                }
                horizontal[(y * width + x) as usize] = sum / norm;
            }
        }
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut norm = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    let yy =
                        (y as i32 + k as i32 - 5).clamp(0, height.saturating_sub(1) as i32) as u32;
                    sum += *weight * horizontal[(yy * width + x) as usize];
                    norm += *weight;
                }
                output[(y * width + x) as usize] = sum / norm;
            }
        }
        output
    };
    let mean_a = blur(a);
    let mean_b = blur(b);
    let a2: Vec<f64> = a.iter().map(|v| v * v).collect();
    let b2: Vec<f64> = b.iter().map(|v| v * v).collect();
    let ab: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    let var_a = blur(&a2);
    let var_b = blur(&b2);
    let covariance = blur(&ab);
    let mut total = 0.0;
    for i in 0..n {
        let va = (var_a[i] - mean_a[i] * mean_a[i]).max(0.0);
        let vb = (var_b[i] - mean_b[i] * mean_b[i]).max(0.0);
        let cov = covariance[i] - mean_a[i] * mean_b[i];
        let c1 = 0.01f64.powi(2);
        let c2 = 0.03f64.powi(2);
        total += ((2.0 * mean_a[i] * mean_b[i] + c1) * (2.0 * cov + c2)
            / ((mean_a[i].powi(2) + mean_b[i].powi(2) + c1) * (va + vb + c2)))
            .clamp(0.0, 1.0);
    }
    total / n.max(1) as f64
}

fn binary_iou(a: &[f32], b: &[f32], threshold: f32) -> f64 {
    let (mut inter, mut union) = (0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        let ax = *x >= threshold;
        let by = *y >= threshold;
        inter += usize::from(ax && by);
        union += usize::from(ax || by);
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

fn binary_precision_recall(a: &[f32], b: &[f32], threshold: f32) -> (f64, f64) {
    let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        match (*x >= threshold, *y >= threshold) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => {}
        }
    }
    (
        tp as f64 / (tp + fp).max(1) as f64,
        tp as f64 / (tp + fn_).max(1) as f64,
    )
}

fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn declared_backgrounds() -> [[f32; 3]; 7] {
    [
        [0.0; 3],
        [1.0; 3],
        [0.5; 3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.23, 0.41, 0.67],
    ]
}

fn background_channel(
    index: usize,
    channel: usize,
    width: u32,
    height: u32,
    background: usize,
) -> f64 {
    let _ = height;
    if background != 6 {
        return srgb_to_linear(f64::from(declared_backgrounds()[background][channel]));
    }
    let x = (index as u32) % width.max(1);
    let y = (index as u32) / width.max(1);
    let seed = u64::from(x)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(u64::from(y).wrapping_mul(0x85eb_ca6b))
        .wrapping_add(15);
    let texture =
        (((seed.wrapping_mul(6364136223846793005) >> 33) as f64) / u32::MAX as f64) * 0.35 + 0.325;
    srgb_to_linear(texture)
}

fn composite_mae(
    rgb: &[[f32; 3]],
    reference_rgb: &[[f32; 3]],
    candidate: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
) -> f64 {
    let mut sum = 0.0;
    for background in 0..7 {
        for (index, (((pixel, ref_pixel), a), b)) in rgb
            .iter()
            .zip(reference_rgb)
            .zip(candidate)
            .zip(reference)
            .enumerate()
        {
            for channel in 0..3 {
                let p = srgb_to_linear(f64::from(pixel[channel])) * f64::from(*a)
                    + background_channel(index, channel, width, height, background)
                        * (1.0 - f64::from(*a));
                let q = srgb_to_linear(f64::from(ref_pixel[channel])) * f64::from(*b)
                    + background_channel(index, channel, width, height, background)
                        * (1.0 - f64::from(*b));
                sum += (p - q).abs();
            }
        }
    }
    sum / (rgb.len().max(1) * 3 * 7) as f64
}

fn composite_psnr(
    rgb: &[[f32; 3]],
    reference_rgb: &[[f32; 3]],
    candidate: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
) -> f64 {
    let mse = composite_mse(rgb, reference_rgb, candidate, reference, width, height);
    if mse == 0.0 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

fn composite_mse(
    rgb: &[[f32; 3]],
    reference_rgb: &[[f32; 3]],
    candidate: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
) -> f64 {
    let mut sum = 0.0;
    for background in 0..7 {
        for (index, (((pixel, ref_pixel), a), b)) in rgb
            .iter()
            .zip(reference_rgb)
            .zip(candidate)
            .zip(reference)
            .enumerate()
        {
            for channel in 0..3 {
                let p = srgb_to_linear(f64::from(pixel[channel])) * f64::from(*a)
                    + background_channel(index, channel, width, height, background)
                        * (1.0 - f64::from(*a));
                let q = srgb_to_linear(f64::from(ref_pixel[channel])) * f64::from(*b)
                    + background_channel(index, channel, width, height, background)
                        * (1.0 - f64::from(*b));
                let delta = p - q;
                sum += delta * delta;
            }
        }
    }
    sum / (rgb.len().max(1) * 3 * 7) as f64
}

#[allow(clippy::too_many_arguments)]
fn composite_ssim_roi(
    rgb: &[[f32; 3]],
    reference_rgb: &[[f32; 3]],
    candidate: &[f32],
    reference: &[f32],
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    width: u32,
) -> f64 {
    let mut r = Vec::new();
    let mut c = Vec::new();
    let mut rr = Vec::new();
    let mut t = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = (y * width + x) as usize;
            r.push(rgb[i]);
            rr.push(reference_rgb[i]);
            c.push(candidate[i]);
            t.push(reference[i]);
        }
    }
    composite_ssim_dims(&r, &rr, &c, &t, x1 - x0 + 1, y1 - y0 + 1)
}

#[allow(dead_code)]
fn global_ssim(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut windows = 0usize;
    for start in (0..a.len()).step_by(11) {
        let end = (start + 11).min(a.len());
        let aa = &a[start..end];
        let bb = &b[start..end];
        let n = aa.len() as f64;
        let mean_a = aa.iter().sum::<f64>() / n;
        let mean_b = bb.iter().sum::<f64>() / n;
        let var_a = aa.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / n;
        let var_b = bb.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / n;
        let covariance = aa
            .iter()
            .zip(bb)
            .map(|(x, y)| (x - mean_a) * (y - mean_b))
            .sum::<f64>()
            / n;
        let c1 = 0.01f64.powi(2);
        let c2 = 0.03f64.powi(2);
        total += ((2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)
            / ((mean_a.powi(2) + mean_b.powi(2) + c1) * (var_a + var_b + c2)))
            .clamp(0.0, 1.0);
        windows += 1;
    }
    total / windows.max(1) as f64
}

fn summarize(scores: Vec<PerImageScore>) -> ScoreSummary {
    if scores.is_empty() {
        return empty_score();
    }
    let n = scores.len() as f64;
    let mean = scores.iter().map(|s| s.agreement).sum::<f64>() / n;
    let delta: Vec<f64> = scores.iter().map(|s| s.agreement).collect();
    let (lo, hi) = bootstrap_ci(&delta, 0x4d3135).unwrap_or((f64::NAN, f64::NAN));
    ScoreSummary {
        mean_agreement: mean,
        min_image_agreement: scores.iter().map(|s| s.agreement).fold(1.0, f64::min),
        mean_alpha_mae: scores.iter().map(|s| s.alpha_mae).sum::<f64>() / n,
        mean_boundary_f1: scores.iter().map(|s| s.boundary_f1).sum::<f64>() / n,
        mean_soft_iou: scores.iter().map(|s| s.soft_iou).sum::<f64>() / n,
        mean_composite_ssim: scores.iter().map(|s| s.composite_ssim).sum::<f64>() / n,
        mean_alpha_rmse: scores.iter().map(|s| s.alpha_rmse).sum::<f64>() / n,
        mean_alpha_sad: scores.iter().map(|s| s.alpha_sad).sum::<f64>() / n,
        mean_boundary_band_f1: scores
            .iter()
            .map(|s| s.boundary_band_f1.unwrap_or(0.0))
            .sum::<f64>()
            / n,
        mean_boundary_band_mae: scores.iter().map(|s| s.boundary_band_mae).sum::<f64>() / n,
        mean_boundary_band_rmse: scores.iter().map(|s| s.boundary_band_rmse).sum::<f64>() / n,
        mean_boundary_band_sad: scores.iter().map(|s| s.boundary_band_sad).sum::<f64>() / n,
        mean_boundary_band_soft_iou: scores.iter().map(|s| s.boundary_band_soft_iou).sum::<f64>()
            / n,
        mean_boundary_band_iou_01: scores.iter().map(|s| s.boundary_band_iou_01).sum::<f64>() / n,
        mean_boundary_band_iou_05: scores.iter().map(|s| s.boundary_band_iou_05).sum::<f64>() / n,
        mean_boundary_band_iou_09: scores.iter().map(|s| s.boundary_band_iou_09).sum::<f64>() / n,
        mean_boundary_f1_tol1: scores.iter().map(|s| s.boundary_f1_tol1).sum::<f64>() / n,
        mean_boundary_f1_tol2: scores.iter().map(|s| s.boundary_f1_tol2).sum::<f64>() / n,
        mean_boundary_f1_tol4: scores.iter().map(|s| s.boundary_f1_tol4).sum::<f64>() / n,
        mean_binary_iou_01: scores.iter().map(|s| s.binary_iou_01).sum::<f64>() / n,
        mean_binary_iou_05: scores.iter().map(|s| s.binary_iou_05).sum::<f64>() / n,
        mean_binary_iou_09: scores.iter().map(|s| s.binary_iou_09).sum::<f64>() / n,
        mean_precision: scores.iter().map(|s| s.precision).sum::<f64>() / n,
        mean_recall: scores.iter().map(|s| s.recall).sum::<f64>() / n,
        mean_composite_mae: scores.iter().map(|s| s.composite_mae).sum::<f64>() / n,
        mean_composite_psnr: scores.iter().map(|s| s.composite_psnr).sum::<f64>() / n,
        mean_topology: scores
            .iter()
            .map(|s| s.topology_score.unwrap_or(0.0))
            .sum::<f64>()
            / n,
        mean_connectivity: scores
            .iter()
            .map(|s| s.connectivity_score.unwrap_or(0.0))
            .sum::<f64>()
            / n,
        mean_edge_color: scores
            .iter()
            .map(|s| s.edge_color_error.unwrap_or(0.0))
            .sum::<f64>()
            / n,
        mean_foreground_color: scores
            .iter()
            .map(|s| s.foreground_color_error.unwrap_or(0.0))
            .sum::<f64>()
            / n,
        mean_gradient: scores
            .iter()
            .map(|s| s.gradient_error.unwrap_or(0.0))
            .sum::<f64>()
            / n,
        mean_fractional_alpha: scores
            .iter()
            .map(|s| s.fractional_alpha_error.unwrap_or(0.0))
            .sum::<f64>()
            / n,
        bootstrap_95_low: lo,
        bootstrap_95_high: hi,
        warm_p95_ms: None,
        peak_rss_bytes: None,
        cold_start_ms: None,
        warm_median_ms: None,
        throughput_images_per_sec: None,
        model_bytes: None,
        provider: None,
        hardware: None,
        per_image: scores,
    }
}
fn empty_score() -> ScoreSummary {
    ScoreSummary {
        mean_agreement: f64::NAN,
        min_image_agreement: f64::NAN,
        mean_alpha_mae: f64::NAN,
        mean_boundary_f1: f64::NAN,
        mean_soft_iou: f64::NAN,
        mean_composite_ssim: f64::NAN,
        mean_alpha_rmse: f64::NAN,
        mean_alpha_sad: f64::NAN,
        mean_boundary_band_f1: f64::NAN,
        mean_boundary_band_mae: f64::NAN,
        mean_boundary_band_rmse: f64::NAN,
        mean_boundary_band_sad: f64::NAN,
        mean_boundary_band_soft_iou: f64::NAN,
        mean_boundary_band_iou_01: f64::NAN,
        mean_boundary_band_iou_05: f64::NAN,
        mean_boundary_band_iou_09: f64::NAN,
        mean_boundary_f1_tol1: f64::NAN,
        mean_boundary_f1_tol2: f64::NAN,
        mean_boundary_f1_tol4: f64::NAN,
        mean_binary_iou_01: f64::NAN,
        mean_binary_iou_05: f64::NAN,
        mean_binary_iou_09: f64::NAN,
        mean_precision: f64::NAN,
        mean_recall: f64::NAN,
        mean_composite_mae: f64::NAN,
        mean_composite_psnr: f64::NAN,
        mean_topology: f64::NAN,
        mean_connectivity: f64::NAN,
        mean_edge_color: f64::NAN,
        mean_foreground_color: f64::NAN,
        mean_gradient: f64::NAN,
        mean_fractional_alpha: f64::NAN,
        bootstrap_95_low: f64::NAN,
        bootstrap_95_high: f64::NAN,
        warm_p95_ms: None,
        peak_rss_bytes: None,
        cold_start_ms: None,
        warm_median_ms: None,
        throughput_images_per_sec: None,
        model_bytes: None,
        provider: None,
        hardware: None,
        per_image: Vec::new(),
    }
}
fn compare_score(a: &ScoreSummary, b: &ScoreSummary) -> Ordering {
    if !a.mean_agreement.is_finite() || !b.mean_agreement.is_finite() {
        return if a.mean_agreement.is_finite() {
            Ordering::Greater
        } else if b.mean_agreement.is_finite() {
            Ordering::Less
        } else {
            Ordering::Equal
        };
    }
    let difference = a.mean_agreement - b.mean_agreement;
    if difference.abs() > 0.0005 {
        return a
            .mean_agreement
            .partial_cmp(&b.mean_agreement)
            .unwrap_or(Ordering::Equal);
    }
    a.min_image_agreement
        .partial_cmp(&b.min_image_agreement)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            a.mean_boundary_f1
                .partial_cmp(&b.mean_boundary_f1)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| match (a.warm_p95_ms, b.warm_p95_ms) {
            (Some(av), Some(bv)) => bv.partial_cmp(&av).unwrap_or(Ordering::Equal),
            _ => Ordering::Equal,
        })
        .then_with(|| match (a.peak_rss_bytes, b.peak_rss_bytes) {
            (Some(av), Some(bv)) => bv.cmp(&av),
            _ => Ordering::Equal,
        })
}
fn parse_pipeline(base: &str) -> (String, String, String, String) {
    let parts: Vec<&str> = base.split('+').collect();
    (
        parts.first().unwrap_or(&"P1-isnet-fast").to_string(),
        parts.get(1).unwrap_or(&"none").to_string(),
        parts.get(2).unwrap_or(&"none").to_string(),
        parts.get(3).unwrap_or(&"original").to_string(),
    )
}

fn parse_pipeline_strict(base: &str) -> Result<(String, String, String, String)> {
    let parts: Vec<&str> = base.split('+').collect();
    ensure!(parts.len() == 4, "malformed M15 pipeline id {base}");
    ensure!(
        parts.iter().all(|part| !part.trim().is_empty()),
        "empty M15 pipeline component in {base}"
    );
    ensure!(
        matches!(
            parts[0],
            "P0-u2-cf" | "P1-isnet-fast" | "P2-tracer-fba" | "P3-biref-vit" | "P4-bria-vit"
        ),
        "M15 pipeline segmenter component is invalid: {base}"
    );
    ensure!(
        matches!(parts[1], "none" | "symmetric"),
        "M15 pipeline trimap component is invalid: {base}"
    );
    ensure!(
        matches!(parts[2], "none" | "closed-form" | "vitmatte"),
        "M15 pipeline refiner component is invalid: {base}"
    );
    ensure!(
        matches!(parts[3], "original" | "multilevel" | "fast" | "fba"),
        "M15 pipeline foreground component is invalid: {base}"
    );
    Ok((
        parts[0].into(),
        parts[1].into(),
        parts[2].into(),
        parts[3].into(),
    ))
}
fn paired_comparison(base: &ScoreSummary, challenger: &ScoreSummary) -> Result<PairedComparison> {
    paired_comparison_named(
        base,
        challenger,
        "P1-isnet-fast+none+none+original@0.500",
        "selected-synthetic-tune-config",
    )
}

fn paired_comparison_named(
    base: &ScoreSummary,
    challenger: &ScoreSummary,
    baseline: &str,
    challenger_id: &str,
) -> Result<PairedComparison> {
    let base_ids: std::collections::BTreeSet<&str> =
        base.per_image.iter().map(|p| p.id.as_str()).collect();
    let challenger_ids: std::collections::BTreeSet<&str> =
        challenger.per_image.iter().map(|p| p.id.as_str()).collect();
    ensure!(
        !base_ids.is_empty() && base_ids == challenger_ids,
        "paired comparison requires identical nonempty image-ID sets"
    );
    ensure!(
        base.per_image.len() == base_ids.len()
            && challenger.per_image.len() == challenger_ids.len(),
        "paired comparison rejects duplicate image IDs"
    );
    ensure!(
        base.per_image.len() >= 2,
        "paired comparison requires at least two images"
    );
    let base_by_id: BTreeMap<&str, &PerImageScore> =
        base.per_image.iter().map(|p| (p.id.as_str(), p)).collect();
    let challenger_by_id: BTreeMap<&str, &PerImageScore> = challenger
        .per_image
        .iter()
        .map(|p| (p.id.as_str(), p))
        .collect();
    let deltas = base_ids
        .iter()
        .map(|id| {
            let a = base_by_id[id];
            let b = challenger_by_id[id];
            ensure!(
                a.agreement.is_finite() && b.agreement.is_finite(),
                "paired scores must be finite"
            );
            Ok::<f64, anyhow::Error>(b.agreement - a.agreement)
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        deltas.iter().all(|value| value.is_finite()),
        "paired deltas must be finite"
    );
    let mean = deltas.iter().sum::<f64>() / deltas.len().max(1) as f64;
    let (lo, hi) = bootstrap_ci(&deltas, 0x50414952)?;
    Ok(PairedComparison {
        baseline: baseline.into(),
        challenger: challenger_id.into(),
        per_image_deltas: deltas,
        mean_delta: mean,
        bootstrap_delta_95_low: lo,
        bootstrap_delta_95_high: hi,
        method: "paired image-ID bootstrap percentile CI",
        seed: 0x50414952,
        resamples: BOOTSTRAP_RESAMPLES,
    })
}
fn bootstrap_ci(values: &[f64], seed: u64) -> Result<(f64, f64)> {
    ensure!(!values.is_empty(), "bootstrap requires nonempty values");
    ensure!(seed != 0, "bootstrap seed must be nonzero");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "bootstrap values must be finite"
    );
    let mut rng = seed;
    let mut means = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let mut sum = 0.0;
        for _ in values {
            rng = xorshift(rng);
            sum += values[(rng as usize) % values.len()];
        }
        means.push(sum / values.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok((
        means[BOOTSTRAP_RESAMPLES * 25 / 1000],
        means[BOOTSTRAP_RESAMPLES * 975 / 1000],
    ))
}
fn sweep_report() -> SweepReport {
    let random_values = lcg_values(0x15, 5);
    let latin_hypercube_values =
        latin_hypercube_checked(0x1515, 5).expect("M15 checked sweep seed/domain are valid");
    SweepReport {
        grid_values: vec![0.25, 0.5, 0.75],
        random_seed: 0x15,
        random_values,
        latin_hypercube_seed: 0x1515,
        latin_hypercube_values,
        deterministic: true,
        blind_used_for_selection: false,
    }
}
fn latin_hypercube_checked(seed: u64, dimensions: usize) -> Result<Vec<f32>> {
    ensure!(seed != 0, "M15 sweep RNG seed must be nonzero");
    ensure!(
        dimensions > 0,
        "M15 LHS domain must have at least one stratum"
    );
    Ok(latin_hypercube(seed, dimensions))
}
fn latin_hypercube(mut seed: u64, dimensions: usize) -> Vec<f32> {
    if dimensions == 0 {
        return Vec::new();
    }
    let mut strata: Vec<usize> = (0..dimensions).collect();
    for i in (1..dimensions).rev() {
        seed = xorshift(seed);
        let j = (seed as usize) % (i + 1);
        strata.swap(i, j);
    }
    strata
        .into_iter()
        .map(|stratum| {
            seed = xorshift(seed);
            let jitter = ((seed >> 11) as f64 / (u64::MAX >> 11) as f64) as f32;
            (stratum as f32 + jitter) / dimensions as f32
        })
        .collect()
}
fn lcg_values(mut state: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            0.2 + ((state >> 32) % 600) as f32 / 1000.0
        })
        .collect()
}
fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}
fn arena_rules(
    corpus_available: bool,
    loo_complete: bool,
    real_one_frozen: bool,
    synthetic_contract_passed: bool,
) -> ArenaRules {
    ArenaRules {
        one_frozen_config_across_inputs: real_one_frozen,
        failures_score_zero: synthetic_contract_passed,
        filename_and_reference_leakage_blocked: corpus_available,
        assisted_sam_separate: synthetic_contract_passed,
        equal_image_weight: synthetic_contract_passed,
        roi_definition: "union(candidate/reference alpha bounding boxes) padded by 5% image diagonal",
        maximize_score_semantics: synthetic_contract_passed,
        deterministic_tie_breaks: vec![
            "mean agreement descending",
            "minimum per-image descending",
            "boundary F1 descending",
            "warm p95 ascending",
            "peak RSS ascending",
            "candidate id ascending",
        ],
        track_level_reporting: synthetic_contract_passed,
        leave_one_image_out_tuning: loo_complete,
        leave_one_image_out_folds: if loo_complete { 6 } else { 0 },
        each_arena_image_held_out_once: loo_complete,
        real_one_frozen_config_across_inputs: real_one_frozen,
        six_image_gate: "mean >= 0.99; every image >= 0.97; no failures (synthetic fixture is not a PhotoRoom claim)",
    }
}
fn sha_text(value: &str) -> String {
    sha_bytes(value.as_bytes())
}

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}
fn sha_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn write_string(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    ensure!(value.len() <= u16::MAX as usize, "M15 cache hash too long");
    bytes.extend_from_slice(&(value.len() as u16).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}
fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16> {
    let end = (*cursor)
        .checked_add(2)
        .context("M15 cache cursor overflow")?;
    ensure!(end <= bytes.len(), "M15 cache truncated u16");
    let v = u16::from_le_bytes([bytes[*cursor], bytes[*cursor + 1]]);
    *cursor = end;
    Ok(v)
}
fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = (*cursor)
        .checked_add(4)
        .context("M15 cache cursor overflow")?;
    ensure!(end <= bytes.len(), "M15 cache truncated u32");
    let v = u32::from_le_bytes(
        bytes[*cursor..*cursor + 4]
            .try_into()
            .context("M15 cache u32 conversion")?,
    );
    *cursor = end;
    Ok(v)
}
fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = (*cursor)
        .checked_add(8)
        .context("M15 cache cursor overflow")?;
    ensure!(end <= bytes.len(), "M15 cache truncated u64");
    let v = u64::from_le_bytes(
        bytes[*cursor..*cursor + 8]
            .try_into()
            .context("M15 cache u64 conversion")?,
    );
    *cursor = end;
    Ok(v)
}
fn read_string(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<String> {
    let end = (*cursor)
        .checked_add(len)
        .context("M15 cache string cursor overflow")?;
    ensure!(end <= bytes.len(), "M15 cache truncated string");
    let value = String::from_utf8(bytes[*cursor..end].to_vec())?;
    *cursor = end;
    Ok(value)
}

fn cache_payload_range(bytes: &[u8]) -> Result<std::ops::Range<usize>> {
    ensure!(bytes.starts_with(CACHE_MAGIC), "M15 cache magic missing");
    let mut cursor = CACHE_MAGIC.len();
    let input_len = usize::from(read_u16(bytes, &mut cursor)?);
    let _ = read_string(bytes, &mut cursor, input_len)?;
    let manifest_len = usize::from(read_u16(bytes, &mut cursor)?);
    let _ = read_string(bytes, &mut cursor, manifest_len)?;
    let _ = read_u32(bytes, &mut cursor)?;
    let _ = read_u32(bytes, &mut cursor)?;
    let count = usize::try_from(read_u64(bytes, &mut cursor)?)
        .context("M15 cache payload count overflow")?;
    let payload_bytes = count
        .checked_mul(4)
        .context("M15 cache payload size overflow")?;
    let end = cursor
        .checked_add(payload_bytes)
        .context("M15 cache payload range overflow")?;
    ensure!(
        end.checked_add(64) == Some(bytes.len()),
        "M15 cache payload range does not match digest"
    );
    Ok(cursor..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(label: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("m15-{label}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn m15_fixture_is_deterministic() {
        let path = test_dir("deterministic");
        run(&path).unwrap();
        let first = fs::read(path.join("report.json")).unwrap();
        let foreground = image::ImageReader::open(
            path.join("synthetic-artifacts/synthetic-validation-1/selected/foreground.png"),
        )
        .unwrap()
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();
        let cutout = image::ImageReader::open(
            path.join("synthetic-artifacts/synthetic-validation-1/selected/cutout.png"),
        )
        .unwrap()
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();
        assert_eq!(foreground.color(), image::ColorType::Rgb8);
        assert_eq!(cutout.color(), image::ColorType::Rgba8);
        run(&path).unwrap();
        let second = fs::read(path.join("report.json")).unwrap();
        assert_eq!(first, second);
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn m15_cache_recovers_corruption_and_rejects_bad_key() {
        let path = test_dir("cache");
        let c = RawMaskCache::new(&path).unwrap();
        let input = sha_text("input");
        let model = sha_text("model");
        let (v, hit) = c
            .load_or_compute(&input, &model, 2, 2, || Ok(vec![0.1, 0.2, 0.3, 0.4]))
            .unwrap();
        assert!(!hit);
        assert_eq!(v.len(), 4);
        let (_, hit) = c
            .load_or_compute(&input, &model, 2, 2, || Ok(vec![0.0; 4]))
            .unwrap();
        assert!(hit);
        let cache_file = c.path(&RawMaskCache::key(&input, &model)).unwrap();
        let mut tampered = fs::read(&cache_file).unwrap();
        let payload_index = tampered.len() - 64 - 1;
        tampered[payload_index] ^= 1;
        fs::write(&cache_file, tampered).unwrap();
        let (_, hit) = c
            .load_or_compute(&input, &model, 2, 2, || Ok(vec![0.2; 4]))
            .unwrap();
        assert!(!hit);
        assert!(c.path("../escape").is_err());
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn m15_bootstrap_and_tie_breaks_are_deterministic() {
        assert_eq!(
            bootstrap_ci(&[0.1, 0.2, 0.3], 5).unwrap(),
            bootstrap_ci(&[0.1, 0.2, 0.3], 5).unwrap()
        );
        let mut a = empty_score();
        let mut b = empty_score();
        a.mean_agreement = 0.9;
        b.mean_agreement = 0.8;
        assert_eq!(compare_score(&a, &b), Ordering::Greater);
        b.mean_agreement = 0.8998;
        b.min_image_agreement = 0.95;
        a.min_image_agreement = 0.96;
        assert_eq!(compare_score(&a, &b), Ordering::Greater);
        b.mean_agreement = 0.991;
        a.mean_agreement = 0.999;
        assert_eq!(compare_score(&a, &b), Ordering::Greater);
        assert!(bootstrap_ci(&[], 5).is_err());
        assert!(bootstrap_ci(&[f64::NAN], 5).is_err());
        assert!(bootstrap_ci(&[0.1], 0).is_err());
    }

    #[test]
    fn m15_evaluation_is_shared_by_score_and_artifact_stages() {
        let image = synthetic_images().into_iter().next().unwrap();
        let mut raw = BTreeMap::new();
        raw.insert(
            ("P1-isnet-fast".into(), image.id.clone()),
            synthetic_raw_mask(&image, "P1-isnet-fast"),
        );
        let pipeline = RankedPipeline {
            id: "P1-isnet-fast+none+none+fast".into(),
            segmenter: "P1-isnet-fast".into(),
            trimap: "none".into(),
            refiner: "none".into(),
            foreground: "fast".into(),
            strength: 0.5,
            tune: empty_score(),
            validation: None,
            selected_on_tune_only: false,
        };
        let evaluation =
            evaluate_pipeline(&pipeline, &image, &raw, EvaluationMode::SyntheticContract).unwrap();
        let score = score_one(&pipeline, &image, &raw);
        let expected = evaluation
            .refined_alpha
            .iter()
            .zip(image.truth.iter())
            .map(|(a, b)| f64::from((a - b).abs()))
            .sum::<f64>()
            / image.truth.len() as f64;
        assert!((score.alpha_mae - expected).abs() < 1e-9);
        assert!(evaluation.refined_alpha != evaluation.cleaned);
    }

    #[test]
    fn m15_blind_lifecycle_transitions_reject_retired_promotion() {
        let initial = blind_lifecycle();
        let evaluated = initial
            .transition(BlindEvent::EvaluateReleaseCandidate)
            .unwrap();
        let retired = evaluated
            .transition(BlindEvent::InfluencePromotion)
            .unwrap();
        assert!(!retired.promotion_allowed());
        assert!(retired.transition(BlindEvent::InfluencePromotion).is_err());
        assert!(retired
            .transition(BlindEvent::RegisterNewUntouchedSet)
            .unwrap()
            .promotion_allowed());
    }

    #[test]
    fn m15_html_escapes_hostile_status_strings() {
        let escaped = escape_html("<script>alert(\"x\")</script>");
        assert_eq!(escaped, "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;");
    }

    #[test]
    fn m15_lhs_has_one_sample_per_stratum_and_seed_changes_order() {
        let a = latin_hypercube(17, 8);
        let b = latin_hypercube(17, 8);
        let c = latin_hypercube(18, 8);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let strata: std::collections::BTreeSet<usize> =
            a.iter().map(|v| (*v * 8.0).floor() as usize).collect();
        assert_eq!(strata, (0..8).collect());
    }

    #[test]
    fn m15_boundary_and_composite_metrics_are_not_perfect_for_mismatch() {
        let rgb = vec![[0.8, 0.2, 0.1]; 9];
        let candidate = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let reference = vec![0.0; 9];
        assert!(boundary_f1_at_tolerance(&candidate, &reference, 3, 3, 1) < 1.0);
        assert!(composite_ssim(&rgb, &candidate, &reference) < 1.0);
        assert_eq!(composite_mae(&rgb, &rgb, &reference, &reference, 3, 3), 0.0);
        assert_eq!(
            composite_psnr(&rgb, &rgb, &reference, &reference, 3, 3),
            100.0
        );
        assert!(composite_mae(&rgb, &rgb, &candidate, &reference, 3, 3) > 0.0);
    }

    #[test]
    fn m15_section_62_metrics_are_finite_and_distinguish_fractional_alpha() {
        let image = synthetic_images().into_iter().next().unwrap();
        let (components, holes) = topology_counts(&image.truth, image.width, image.height);
        assert!(components > 0);
        assert!(holes <= image.truth.len());
        let mut perturbed = image.truth.clone();
        if let Some(value) = perturbed
            .iter_mut()
            .find(|value| **value > 0.02 && **value < 0.98)
        {
            *value = (*value + 0.2).min(1.0);
        }
        let gradient = gradient_error_metric(&perturbed, &image.truth, image.width, image.height);
        let fractional = fractional_alpha_error_metric(&perturbed, &image.truth);
        assert!(gradient.is_finite() && fractional.is_finite());
        assert!(fractional > 0.0);
        assert_eq!(
            boundary_band_metrics(&image.truth, &image.truth, image.width, image.height, 1).1,
            0.0
        );
    }

    #[test]
    fn m15_real_arena_preserves_declared_splits_and_excludes_blind() {
        let arena = inspect_real_arena(&repo_path("test_images")).unwrap();
        assert_eq!(
            arena
                .records
                .iter()
                .filter(|r| r.declared_split == "blind")
                .count(),
            2
        );
        assert!(arena.records.iter().any(|r| r.declared_split == "tune"));
        assert!(arena
            .records
            .iter()
            .any(|r| r.declared_split == "validation"));
        assert_eq!(
            arena
                .records
                .iter()
                .filter(|r| r.tags.iter().any(|tag| tag == "portrait-character"))
                .count(),
            4
        );
        assert_eq!(
            arena
                .records
                .iter()
                .filter(|r| r.tags.iter().any(|tag| tag == "emissive-artwork"))
                .count(),
            2
        );
    }

    #[test]
    fn m15_loo_requires_one_identical_resolved_config_for_champion() {
        let same = vec!["P1@0.500".to_owned(); 6];
        assert!(same_resolved_config(&same));
        let mut differing = same;
        differing[5] = "P1@0.750".to_owned();
        assert!(!same_resolved_config(&differing));
    }

    #[test]
    fn m15_fixed_arena_loo_retires_legacy_declared_blind_labels() {
        let arena = inspect_real_arena(&repo_path("test_images")).unwrap();
        let not_run = blind_lifecycle_for_arena(&arena, false);
        assert_eq!(not_run.state, "untouched");
        assert!(not_run.promotion_allowed());
        let lifecycle = blind_lifecycle_for_arena(&arena, true);
        assert!(lifecycle.fixed_arena_loo_override);
        assert_eq!(lifecycle.state, "retired_after_fixed_arena_loo_override");
        assert!(lifecycle.blind_used_for_sweeps);
        assert!(!lifecycle.blind_used_for_promotion);
        assert!(lifecycle.new_untouched_set_required);
        assert_eq!(lifecycle.legacy_declared_blind_ids.len(), 2);
        assert!(!lifecycle.promotion_allowed());
    }

    #[test]
    fn m15_fixed_arena_loo_does_not_claim_legacy_blind_ids_excluded() {
        let arena = inspect_real_arena(&repo_path("test_images")).unwrap();
        assert_eq!(
            arena
                .records
                .iter()
                .filter(|record| record.declared_split == "blind")
                .count(),
            2
        );
        // The completed real tournament constructor uses an empty exclusion
        // list for the fixed-arena override; legacy IDs live in lifecycle
        // provenance instead because they are actually consumed by folds.
        let lifecycle = blind_lifecycle_for_arena(&arena, true);
        assert_eq!(lifecycle.legacy_declared_blind_ids.len(), 2);
    }

    #[test]
    fn m15_fixed_arena_loo_holds_each_id_once_and_trains_on_the_other_five() {
        let ids = (0..6).map(|i| format!("arena-{i}")).collect::<Vec<_>>();
        let folds = loo_partitions(&ids).unwrap();
        assert_eq!(folds.len(), 6);
        let held = folds
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(held, ids.iter().cloned().collect());
        for (held_id, training) in folds {
            assert_eq!(training.len(), 5);
            assert!(!training.contains(&held_id));
            assert_eq!(
                training
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                5
            );
        }
        assert!(loo_partitions(&ids[..5]).is_err());
        let mut duplicate = ids.clone();
        duplicate[5] = duplicate[0].clone();
        assert!(loo_partitions(&duplicate).is_err());
    }

    #[test]
    fn m15_cache_rejects_uppercase_hashes_and_failure_scores_zero() {
        let path = test_dir("hash");
        let cache = RawMaskCache::new(&path).unwrap();
        assert!(cache
            .load_or_compute(&"A".repeat(64), &"a".repeat(64), 1, 1, || Ok(vec![0.0]))
            .is_err());
        let image = synthetic_images().into_iter().next().unwrap();
        let pipeline = RankedPipeline {
            id: "P1+none+none+original@1".into(),
            segmenter: "P1-isnet-fast".into(),
            trimap: "none".into(),
            refiner: "none".into(),
            foreground: "original".into(),
            strength: 1.0,
            tune: empty_score(),
            validation: None,
            selected_on_tune_only: false,
        };
        let score = score_one(&pipeline, &image, &BTreeMap::new());
        assert!(score.failure && score.agreement == 0.0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn m15_blind_lifecycle_does_not_require_new_set_before_influence() {
        let lifecycle = blind_lifecycle();
        assert_eq!(lifecycle.state, "untouched");
        assert!(!lifecycle.new_untouched_set_required);
        assert!(!lifecycle.blind_used_for_promotion);
    }

    #[test]
    fn m15_negative_contracts_reject_invalid_ids_and_zero_lhs_seed() {
        assert!(parse_pipeline_strict("P9-unknown+none+none+original").is_err());
        assert!(latin_hypercube_checked(0, 4).is_err());
        assert!(latin_hypercube_checked(7, 0).is_err());
    }

    #[test]
    fn m15_real_mode_does_not_apply_synthetic_affine_transform() {
        let image = synthetic_images().into_iter().next().unwrap();
        let pipeline = RankedPipeline {
            id: "P1-isnet-fast+none+none+original@0.250".into(),
            segmenter: "P1-isnet-fast".into(),
            trimap: "symmetric".into(),
            refiner: "closed-form".into(),
            foreground: "original".into(),
            strength: 0.25,
            tune: empty_score(),
            validation: None,
            selected_on_tune_only: true,
        };
        let raw = BTreeMap::from([(
            (pipeline.segmenter.clone(), image.id.clone()),
            vec![0.5; image.truth.len()],
        )]);
        let score = score_one_mode(&pipeline, &image, &raw, EvaluationMode::RealMechanisms);
        assert!(score.agreement.is_finite());
        assert!(!score.failure);
    }
}
