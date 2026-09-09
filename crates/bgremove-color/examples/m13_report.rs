//! Deterministic M13 synthetic recovery/ranking report.
use anyhow::{bail, Result};
use bgremove_color::{
    FastForegroundConfig, FastForegroundEstimator, FbaForegroundEstimator,
    MultilevelForegroundConfig, MultilevelForegroundEstimator, OriginalRgbEstimator,
};
use bgremove_core::{CanonicalImage, ForegroundEstimator, RefinedMatte, RgbImageF32};
use serde_json::json;
use std::fs;
use std::path::Path;

fn mae_visible(candidate: &RgbImageF32, truth: &RgbImageF32, alpha: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for ((a, t), alpha) in candidate.data().iter().zip(truth.data()).zip(alpha) {
        if *alpha > 0.0 {
            for c in 0..3 {
                sum += (a[c] - t[c]).abs();
                count += 1;
            }
        }
    }
    sum / count as f32
}

fn composite_mae(
    candidate: &RgbImageF32,
    truth: &RgbImageF32,
    alpha: &[f32],
    background: [f32; 3],
) -> f32 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (((a, t), alpha), _) in candidate
        .data()
        .iter()
        .zip(truth.data())
        .zip(alpha)
        .zip(0..)
    {
        if *alpha > 0.0 {
            for c in 0..3 {
                let expected = *alpha * t[c] + (1.0 - *alpha) * background[c];
                let actual = *alpha * a[c] + (1.0 - *alpha) * background[c];
                sum += (actual - expected).abs();
                count += 1;
            }
        }
    }
    sum / count as f32
}

fn edge_mae(candidate: &RgbImageF32, truth: &RgbImageF32, alpha: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for ((a, t), alpha) in candidate.data().iter().zip(truth.data()).zip(alpha) {
        if (0.05..=0.95).contains(alpha) {
            for c in 0..3 {
                sum += (a[c] - t[c]).abs();
                count += 1;
            }
        }
    }
    sum / count as f32
}

fn edge_composite_mae(
    candidate: &RgbImageF32,
    truth: &RgbImageF32,
    alpha: &[f32],
    background: [f32; 3],
) -> f32 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for ((a, t), alpha) in candidate.data().iter().zip(truth.data()).zip(alpha) {
        if (0.05..=0.95).contains(alpha) {
            for c in 0..3 {
                let expected = *alpha * t[c] + (1.0 - *alpha) * background[c];
                let actual = *alpha * a[c] + (1.0 - *alpha) * background[c];
                sum += (actual - expected).abs();
                count += 1;
            }
        }
    }
    sum / count as f32
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

struct CandidateMetric {
    candidate: &'static str,
    working_space: &'static str,
    visible_foreground_mae: f32,
    edge_band_foreground_mae: f32,
    composites: Vec<f32>,
    edge_composites: Vec<f32>,
    all_finite: bool,
}

fn main() -> Result<()> {
    let width = 6;
    let height = 4;
    let f = [0.8, 0.2, 0.1];
    let b = [0.1, 0.3, 0.9];
    let alpha: Vec<f32> = (0..width * height)
        .map(|i| match i % 6 {
            0 => 0.0,
            1 => 0.05,
            2 => 0.25,
            3 => 0.5,
            4 => 0.95,
            _ => 1.0,
        })
        .collect();
    let truth_pixels = vec![f; (width * height) as usize];
    let input_pixels = alpha
        .iter()
        .map(|a| std::array::from_fn(|c| *a * f[c] + (1.0 - *a) * b[c]))
        .collect::<Vec<[f32; 3]>>();
    let image = CanonicalImage::new(width, height, input_pixels)?;
    let truth = RgbImageF32::new(width, height, truth_pixels)?;
    let matte = RefinedMatte::new(
        bgremove_core::AlphaMask::new(width, height, alpha.clone())?,
        None,
        None,
    )?;
    let full_matte = RefinedMatte::new(
        matte.alpha().clone(),
        Some(truth.clone()),
        Some(RgbImageF32::constant(width, height, b)?),
    )?;
    let candidates: [(&str, &str, RgbImageF32); 5] = [
        (
            "original-rgb",
            "encoded-srgb",
            OriginalRgbEstimator::default().estimate(&image, &matte)?,
        ),
        (
            "pymatting-multilevel",
            "encoded-srgb",
            MultilevelForegroundEstimator::new(MultilevelForegroundConfig::default())?
                .estimate(&image, &matte)?,
        ),
        (
            "fast-coarse-fine",
            "encoded-srgb",
            FastForegroundEstimator::new(FastForegroundConfig::photoroom_source())?
                .estimate(&image, &matte)?,
        ),
        (
            "fast-coarse-fine",
            "linear-light",
            FastForegroundEstimator::new(FastForegroundConfig {
                working_space: bgremove_color::ForegroundWorkingSpace::LinearLight,
                ..FastForegroundConfig::photoroom_source()
            })?
            .estimate(&image, &matte)?,
        ),
        (
            "fba-provided",
            "encoded-srgb",
            FbaForegroundEstimator.estimate(&image, &full_matte)?,
        ),
    ];
    let backgrounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.5, 0.8]];
    let mut metric_values = candidates
        .iter()
        .map(|(name, working_space, candidate)| {
            let composites = backgrounds
                .iter()
                .map(|background| composite_mae(candidate, &truth, &alpha, *background))
                .collect::<Vec<_>>();
            let edge_composites = backgrounds
                .iter()
                .map(|background| edge_composite_mae(candidate, &truth, &alpha, *background))
                .collect::<Vec<_>>();
            CandidateMetric {
                candidate: name,
                working_space,
                visible_foreground_mae: mae_visible(candidate, &truth, &alpha),
                edge_band_foreground_mae: edge_mae(candidate, &truth, &alpha),
                composites,
                edge_composites,
                all_finite: candidate
                    .data()
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite()),
            }
        })
        .collect::<Vec<_>>();
    metric_values.sort_by(|left, right| {
        mean(&left.edge_composites)
            .total_cmp(&mean(&right.edge_composites))
            .then_with(|| mean(&left.composites).total_cmp(&mean(&right.composites)))
            .then_with(|| left.candidate.cmp(right.candidate))
            .then_with(|| left.working_space.cmp(right.working_space))
    });
    let original_edge = metric_values
        .iter()
        .find(|metric| metric.candidate == "original-rgb")
        .map(|metric| mean(&metric.edge_composites))
        .expect("original control metric");
    let encoded_fast_edge = metric_values
        .iter()
        .find(|metric| {
            metric.candidate == "fast-coarse-fine" && metric.working_space == "encoded-srgb"
        })
        .map(|metric| mean(&metric.edge_composites))
        .expect("encoded fast metric");
    let pymatting = metric_values
        .iter()
        .find(|metric| metric.candidate == "pymatting-multilevel")
        .expect("PyMatting metric");
    let fba = metric_values
        .iter()
        .find(|metric| metric.candidate == "fba-provided")
        .expect("FBA metric");
    let all_finite = metric_values.iter().all(|metric| {
        metric.all_finite
            && metric.visible_foreground_mae.is_finite()
            && metric.edge_band_foreground_mae.is_finite()
            && metric.composites.iter().all(|value| value.is_finite())
            && metric.edge_composites.iter().all(|value| value.is_finite())
    });
    let gates = json!({
        "pymatting_near_truth": pymatting.visible_foreground_mae <= 1e-4,
        "pymatting_threshold": 1e-4,
        "fba_exact": fba.visible_foreground_mae <= 1e-7 && fba.edge_band_foreground_mae <= 1e-7,
        "fba_threshold": 1e-7,
        "encoded_fast_improves_original_edge_composite": encoded_fast_edge < original_edge,
        "all_finite": all_finite,
    });
    let status = if gates["pymatting_near_truth"].as_bool().unwrap()
        && gates["fba_exact"].as_bool().unwrap()
        && gates["encoded_fast_improves_original_edge_composite"]
            .as_bool()
            .unwrap()
        && all_finite
    {
        "pass"
    } else {
        "fail"
    };
    let metrics = metric_values
        .iter()
        .enumerate()
        .map(|(index, metric)| {
            json!({
                "rank": index + 1,
                "candidate": metric.candidate,
                "working_space": metric.working_space,
                "visible_foreground_mae": metric.visible_foreground_mae,
                "edge_band_foreground_mae": metric.edge_band_foreground_mae,
                "multi_background_composite_mae": metric.composites,
                "edge_band_multi_background_composite_mae": metric.edge_composites,
                "all_finite": metric.all_finite,
                "hidden_rgb_alpha_zero_excluded": true,
            })
        })
        .collect::<Vec<_>>();
    let report = json!({
        "schema": "m13.foreground-recovery.v1",
        "status": status,
        "input": {"width": width, "height": height, "alpha_pattern": "0,.05,.25,.5,.95,1 repeated"},
        "formula": "Ab=cv2.blur(A,(k,k)); Fb=cv2.blur(F*A,(k,k))/(Ab+epsilon); Bb=cv2.blur(B*(1-A),(k,k))/(1-Ab+epsilon); F=clip(Fb+A*(I-A*Fb-(1-A)*Bb),0,1); return (F,Bb)",
        "fast_config": {"coarse_kernel_width": 90, "fine_kernel_width": 6, "coarse_iterations": 1, "fine_iterations": 1, "reference_size": 1024, "epsilon": 1e-5, "edge_policy": "opencv-reflect101", "scale_kernel_widths": false, "complexity": "O(width*height) per pass and channel"},
        "working_space_ablation": ["encoded-srgb", "linear-light"],
        "multi_backgrounds": backgrounds,
        "metrics": metrics,
        "gates": gates,
        "ranking_key": "mean edge_band_multi_background_composite_mae then mean visible multi_background_composite_mae",
        "metric_note": "All candidates share the same known alpha, so changing the composite background scales foreground error by alpha; three backgrounds are retained and this limitation is reported rather than claiming independent background discrimination.",
        "source_authority": {
            "pymatting": "tests/fixtures/m13/reference/report.json",
            "photoroom": "tests/fixtures/m13/fast-reference/report.json",
        },
    });
    let output = Path::new("runs/m13-foreground");
    fs::create_dir_all(output)?;
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if status != "pass" {
        bail!(
            "M13 foreground recovery acceptance gates failed; see runs/m13-foreground/report.json"
        );
    }
    Ok(())
}
