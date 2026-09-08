#!/usr/bin/env python3
"""Execute the pinned CarveKit FBA wrapper stages on a small synthetic graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, __version__ as PIL_VERSION

ROOT = Path(__file__).resolve().parents[3]
CARVEKIT = ROOT / "projects/python/image-background-remove-tool"
COMMIT = "f141a311af67fb1da64269c508a6d1f786420801"
CONFIGURED_SIZE = (11, 7)
CANONICAL_SIZE = (13, 9)
EXPECTED_MODEL_SHA256 = "57183a7d2adbd274aea3a69bce1d997ece2c1ae36a51ac8fc7566d102be4a20c"
FILES = [
    "carvekit/ml/wrap/fba_matting.py",
    "carvekit/ml/arch/fba_matting/transforms.py",
    "carvekit/ml/arch/fba_matting/models.py",
]
LICENSE = ROOT / "models/M11_FBA_SYNTHETIC_LICENSE.txt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_relevant_files_sha256() -> str:
    digest = hashlib.sha256()
    for relative in FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((CARVEKIT / relative).read_bytes())
    return digest.hexdigest()


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype=np.float32).tobytes(order="C"))


class SyntheticFba(torch.nn.Module):
    def forward(self, image, trimap, image_normalized, trimap_transformed):
        # This is a decoder-shaped synthetic graph followed by the exact
        # pinned FBA.forward clamp/sigmoid/fusion contract. Its ONNX output is
        # final [alpha, foreground BGR, background BGR], never pre-fusion.
        live = 1.0e-6 * image_normalized[:, :1] + 1.0e-6 * trimap_transformed[:, :1]
        alpha = torch.clamp(trimap[:, 1:2] + live, 0.0, 1.0)
        foreground = torch.sigmoid(image + live)
        background = torch.sigmoid(image + live)
        foreground = alpha * image + (1 - alpha**2) * foreground - alpha * (1 - alpha) * background
        background = (1 - alpha) * image + (2 * alpha - alpha**2) * background - alpha * (1 - alpha) * foreground
        foreground = torch.clamp(foreground, 0, 1)
        background = torch.clamp(background, 0, 1)
        numerator = alpha * 0.1 + torch.sum((image - background) * (foreground - background), 1, keepdim=True)
        denominator = torch.sum((foreground - background) * (foreground - background), 1, keepdim=True) + 0.1
        alpha = torch.clamp(numerator / denominator, 0, 1)
        return torch.cat((alpha, foreground, background), dim=1)


def ensure_model(path: Path) -> None:
    if path.exists():
        if sha(path) != EXPECTED_MODEL_SHA256:
            raise SystemExit("existing synthetic graph hash does not match pinned export")
        return
    model = SyntheticFba().eval()
    zeros = (
        torch.zeros((1, 3, 8, 16)),
        torch.zeros((1, 2, 8, 16)),
        torch.zeros((1, 3, 8, 16)),
        torch.zeros((1, 6, 8, 16)),
    )
    torch.onnx.export(
        model,
        zeros,
        path,
        input_names=["image", "trimap", "image_normalized", "trimap_transformed"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
    )
    if sha(path) != EXPECTED_MODEL_SHA256:
        raise SystemExit("deterministic synthetic graph export hash drifted")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(CARVEKIT), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/fixtures/m11_fba_synthetic.onnx")
    args = parser.parse_args()
    if git("rev-parse", "HEAD") != COMMIT:
        raise SystemExit("CarveKit checkout is not pinned")
    if git("status", "--short", "--untracked-files=no"):
        raise SystemExit("CarveKit tracked source is dirty")
    runtime = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL_VERSION,
        "onnxruntime": ort.__version__,
    }
    expected_runtime = {
        "python": "3.12.11",
        "torch": "2.9.1",
        "numpy": "2.3.2",
        "opencv": "4.10.0",
        "pillow": "12.2.0",
        "onnxruntime": "1.23.2",
    }
    if runtime != expected_runtime:
        raise SystemExit(f"runtime drift: expected {expected_runtime}, got {runtime}")
    for relative in FILES:
        if not (CARVEKIT / relative).is_file():
            raise SystemExit(f"missing pinned source file: {relative}")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    ensure_model(args.model)
    Path.home = staticmethod(lambda: Path("/tmp/m11-carvekit-home"))
    sys.path.insert(0, str(CARVEKIT))
    from carvekit.ml.wrap.fba_matting import FBAMatting
    from carvekit.ml.arch.fba_matting.models import fba_fusion

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    image_values = np.zeros((CANONICAL_SIZE[1], CANONICAL_SIZE[0], 3), dtype=np.uint8)
    for y in range(CANONICAL_SIZE[1]):
        for x in range(CANONICAL_SIZE[0]):
            image_values[y, x] = [(x * 13 + y * 3) % 256, (x * 5 + y * 17) % 256, (x * 19 + y * 7) % 256]
    # Use wide, class-stable bands.  A one-pixel band is legitimately softened
    # away by the configured bicubic resize; the authority must exercise both
    # one-hot channels and both distance-transform planes after resizing.
    trimap_values = np.full((CANONICAL_SIZE[1], CANONICAL_SIZE[0]), 127, dtype=np.uint8)
    trimap_values[:, :4] = 0
    trimap_values[:, -4:] = 255
    image = Image.fromarray(image_values, mode="RGB")
    trimap = Image.fromarray(trimap_values, mode="L")
    wrapper = FBAMatting(input_tensor_size=list(CONFIGURED_SIZE), batch_size=2, load_pretrained=False)
    image_raw, image_normalized = wrapper.data_preprocessing(image)
    trimap_raw, trimap_transformed = wrapper.data_preprocessing(trimap)
    image_raw_np = image_raw.numpy().astype(np.float32)
    image_normalized_np = image_normalized.numpy().astype(np.float32)
    trimap_raw_np = trimap_raw.numpy().astype(np.float32)
    trimap_transformed_np = trimap_transformed.numpy().astype(np.float32)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    expected_inputs = {
        "image": [1, 3, 8, 16],
        "trimap": [1, 2, 8, 16],
        "image_normalized": [1, 3, 8, 16],
        "trimap_transformed": [1, 6, 8, 16],
    }
    actual_inputs = {item.name: item.shape for item in session.get_inputs()}
    if actual_inputs != expected_inputs:
        raise SystemExit(f"synthetic graph input contract drift: {actual_inputs}")
    if [item.name for item in session.get_outputs()] != ["output"] or session.get_outputs()[0].shape != [1, 7, 8, 16]:
        raise SystemExit("synthetic graph final seven-channel output contract drift")
    live = 1.0e-6 * torch.from_numpy(image_normalized_np[:, :1]) + 1.0e-6 * torch.from_numpy(trimap_transformed_np[:, :1])
    decoder_alpha = torch.clamp(torch.from_numpy(trimap_raw_np[:, 1:2]) + live, 0, 1)
    decoder_foreground = torch.sigmoid(torch.from_numpy(image_raw_np) + live)
    decoder_background = torch.sigmoid(torch.from_numpy(image_raw_np) + live)
    decoder = torch.cat((decoder_alpha, torch.from_numpy(image_raw_np) + live, torch.from_numpy(image_raw_np) + live), 1).numpy().astype(np.float32)
    raw = session.run(
        ["output"],
        {
            "image": image_raw_np,
            "trimap": trimap_raw_np,
            "image_normalized": image_normalized_np,
            "trimap_transformed": trimap_transformed_np,
        },
    )[0].astype(np.float32)
    fused_alpha, fused_foreground, fused_background = fba_fusion(
        decoder_alpha,
        torch.from_numpy(image_raw_np),
        decoder_foreground,
        decoder_background,
    )
    fused = torch.cat((fused_alpha, fused_foreground, fused_background), 1).detach().numpy().astype(np.float32)
    # The source's decoder calls fba_fusion internally; the synthetic graph
    # supplies decoder-like seven channels, so this explicitly executes the
    # same pinned fusion equations before both candidate reports.
    # The connected final-output chain is produced by the synthetic graph;
    # invoke the pinned wrapper postprocess on that exact graph output.  The
    # independently executed source fusion remains in fused-output evidence.
    alpha_only = np.asarray(wrapper.data_postprocessing(torch.from_numpy(raw[0]), trimap), dtype=np.uint8)
    restored_fused = np.stack(
        [cv2.resize(raw[0, channel], CANONICAL_SIZE, interpolation=cv2.INTER_LANCZOS4) for channel in range(7)],
        axis=0,
    ).astype(np.float32)
    full_alpha = fused[:, 0:1]
    full_foreground = fused[:, 1:4]
    full_background = fused[:, 4:7]
    reconstructed = full_alpha * full_foreground + (1.0 - full_alpha) * full_background
    composite_error = np.abs(reconstructed - image_raw_np)
    trimap_plane_counts = [int(np.count_nonzero(trimap_raw_np[0, channel])) for channel in range(2)]
    transformed_plane_counts = [int(np.count_nonzero(trimap_transformed_np[0, channel])) for channel in range(6)]
    if min(trimap_plane_counts) == 0 or min(transformed_plane_counts) == 0:
        raise SystemExit("trimap authority fixture lost a one-hot or transformed class")
    if not np.isfinite(trimap_transformed_np).all():
        raise SystemExit("trimap transformed authority contains NaN/Inf")
    padded_pixels = 16 * 8
    adapter_peak = (
        CANONICAL_SIZE[0] * CANONICAL_SIZE[1] * 4
        + CONFIGURED_SIZE[0] * CONFIGURED_SIZE[1] * 4
        + CONFIGURED_SIZE[0] * CONFIGURED_SIZE[1] * 5 * 4
        + 16 * CONFIGURED_SIZE[1] * 5 * 8
        + padded_pixels * 5 * 4
        + padded_pixels * 14 * 4
        + padded_pixels * 2 * 8
        + padded_pixels * 2 * 4
    )
    adapter_restore = (
        (16 * 8 * 7 * 4) * 2
        + 13 * 8 * 7 * 8
        + (13 * 9 * 7 * 4) * 2
    )
    for name, values in {
        "decoded-rgb.f32le": image_values.astype(np.float32) / 255.0,
        "image-input.f32le": image_raw_np,
        "trimap-input.f32le": trimap_raw_np,
        "image-normalized.f32le": image_normalized_np,
        "trimap-transformed.f32le": trimap_transformed_np,
        "decoder-output.f32le": decoder,
        "raw-output.f32le": raw,
        "fused-output.f32le": fused,
        "full-alpha.f32le": restored_fused[0:1],
        "full-foreground.f32le": restored_fused[1:4],
        "full-background.f32le": restored_fused[4:7],
    }.items():
        write_f32(output / name, values)
    (output / "alpha-only.u8").write_bytes(alpha_only.tobytes())
    (output / "input-trimap.u8").write_bytes(trimap_values.tobytes())
    source_hashes = {relative: sha(CARVEKIT / relative) for relative in FILES}
    report = {
        "schema": "m11.carvekit-fba-authoritative.v1",
        "authoritative_sources_executed": True,
        "synthetic_graph": True,
        "real_checkpoint_executed": False,
        "source": {
            "repository": "projects/python/image-background-remove-tool",
            "commit": COMMIT,
            "tracked_source_clean": True,
            "source_file_sha256": source_hashes,
            "source_relevant_files_sha256": source_relevant_files_sha256(),
            "license": {"path": "projects/python/image-background-remove-tool/LICENSE", "sha256": sha(CARVEKIT / "LICENSE")},
            "fixture_license": {"path": "models/M11_FBA_SYNTHETIC_LICENSE.txt", "sha256": sha(LICENSE)},
            "runtime": runtime,
            "network_downloads": False,
        },
        "model": {"path": "models/fixtures/m11_fba_synthetic.onnx", "sha256": sha(args.model), "bytes": args.model.stat().st_size, "license_path": "models/M11_FBA_SYNTHETIC_LICENSE.txt", "license_sha256": sha(LICENSE)},
        "contract": {"canonical_size": list(CANONICAL_SIZE), "configured_size": list(CONFIGURED_SIZE), "padded_size": [16, 8], "inputs": {"image": [1, 3, 8, 16], "trimap": [1, 2, 8, 16], "image_normalized": [1, 3, 8, 16], "trimap_transformed": [1, 6, 8, 16]}, "output": [1, 7, 8, 16], "output_semantics": "final fused alpha, foreground BGR, background BGR", "channel_order": "BGR", "fusion": "pinned carvekit fba_fusion embedded in graph", "alpha_only": "CarveKit data_postprocessing after pinned cv2.resize binding-default interpolation, background enforcement, threshold 0.3, and uint8 conversion", "trimap_evidence": {"one_hot_nonzero_counts": trimap_plane_counts, "transformed_nonzero_counts": transformed_plane_counts, "transformed_min": float(trimap_transformed_np.min()), "transformed_max": float(trimap_transformed_np.max())}},
        "candidates": {"alpha_only": {"status": "executed", "background_alpha_zero": bool(np.all(alpha_only[:, 0] == 0)), "foreground_columns_preserved": bool(np.all(alpha_only[:, -1] > 0))}, "full_fba": {"status": "executed", "foreground_channels": 3, "background_channels": 3}},
        "metrics": {"alpha_only": {"background_zero_fraction": float(np.mean(alpha_only[:, 0] == 0)), "foreground_column_positive_fraction": float(np.mean(alpha_only[:, -1] > 0))}, "full_fba": {"reconstructed_composite_max_abs": float(composite_error.max()), "reconstructed_composite_mean_abs": float(composite_error.mean()), "visible_alpha_weighted_foreground_mean_abs": float(np.mean(full_alpha * np.abs(full_foreground - image_raw_np)))}},
        "artifacts": {name: {"path": name, "sha256": sha(output / name), "bytes": (output / name).stat().st_size} for name in ["decoded-rgb.f32le", "image-input.f32le", "trimap-input.f32le", "image-normalized.f32le", "trimap-transformed.f32le", "decoder-output.f32le", "raw-output.f32le", "fused-output.f32le", "full-alpha.f32le", "full-foreground.f32le", "full-background.f32le", "alpha-only.u8", "input-trimap.u8"]},
        "resource_limits": {"fixture_max_pixels": 1024 * 1024, "edt": "Felzenszwalb exact separable Euclidean transform; O(width*height)", "peak_memory_bytes_cap": 512 * 1024 * 1024, "adapter_preprocessing_estimated_peak_bytes": adapter_peak, "adapter_restoration_estimated_peak_bytes": adapter_restore, "adapter_peak_estimated_bytes": max(adapter_peak, adapter_restore), "adapter_peak_semantics": "checked source/configured/padded f32/f64 resize, EDT, transformed tensors, and seven-plane restoration; excludes external model parameters/activations", "ort_model_peak_bytes": None, "ort_model_peak_status": "unknown until approved checkpoint runtime measurement", "real_1024_status": "skipped: approved checkpoint and model activation measurement unavailable", "real_2048_status": "closed: approved checkpoint and model activation measurement unavailable", "network_downloads": False},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    parity = {
        "schema": "bgremove.m11.parity.v1",
        "source_execution": True,
        "synthetic_graph": True,
        "real_checkpoint_executed": False,
        "authoritative_report": {"path": "report.json", "sha256": sha(output / "report.json")},
        "source": report["source"],
        "model": report["model"],
        "contract": report["contract"],
        "tolerances": {
            "preprocess_f32_abs": 1.0e-6,
            "fusion_f32_abs": 3.0e-6,
            "graph_final_f32_abs": 3.0e-6,
            "alpha_only_u8_abs": 1,
        },
        "artifacts": report["artifacts"],
        "candidates": report["candidates"],
        "resource_limits": report["resource_limits"],
    }
    (output / "parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
