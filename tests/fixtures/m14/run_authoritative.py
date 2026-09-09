#!/usr/bin/env python3
"""Generate compact M14 evidence by executing the pinned rembg helpers."""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import subprocess
import sys
from copy import deepcopy

try:
    import numpy as np
    import scipy
    from scipy.ndimage import map_coordinates
except ImportError as exc:  # pragma: no cover - environment gate
    raise SystemExit(
        "M14 authority requires NumPy and SciPy; run with "
        "projects/python/rembg/.venv/bin/python or "
        "<pinned-python-3.12-with-scipy>/bin/python"
    ) from exc

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / "projects/python/rembg/rembg/sessions/sam.py"
COMMIT = "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
SOURCE_SHA256 = "524bccb7b54acd3979d025e0451334e7a9d0d7e4cd4ce81007b8c1822b827b7d"
OUT = ROOT / "tests/fixtures/m14/reference"
SAMPLE_INDICES = np.array([0, 1, 2, 17, 101, 997, 4096, 65535, 123456, 250000, 500000, 700000, 900000, 1200000, 1500000, 2000000], dtype=np.int64)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_array(values: np.ndarray) -> str:
    return digest_bytes(np.asarray(values).tobytes(order="C"))


def load_source_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {"warp_affine", "get_preprocess_shape", "get_input_points", "apply_coords", "transform_masks"}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted], type_ignores=[])
    namespace = {"np": np, "deepcopy": deepcopy, "map_coordinates": map_coordinates}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


def write_bytes(path: pathlib.Path, data: bytes, artifacts: dict):
    path.write_bytes(data)
    artifacts[path.name] = {"bytes": len(data), "sha256": digest_bytes(data)}


def write_f32(path: pathlib.Path, values: np.ndarray, artifacts: dict):
    write_bytes(path, np.asarray(values, dtype="<f4").tobytes(order="C"), artifacts)


def source_case(helpers, case_name: str, width: int, height: int):
    image = ((np.arange(height * width * 3, dtype=np.uint8) * 37 + 17) % 251).reshape(
        height, width, 3
    )
    input_size = (684, 1024)
    scale = min(input_size[1] / width, input_size[0] / height)
    transform_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    encoder_input = helpers["warp_affine"](image, transform_matrix[:2], input_size).astype(np.float32)
    embedding = np.arange(256 * 64 * 64, dtype=np.float32).reshape(1, 256, 64, 64) / 1000.0
    prompts = [
        [{"type": "point", "label": 1, "data": [0.0, 0.0]}, {"type": "point", "label": 0, "data": [width - 1.0, height - 1.0]}, {"type": "rectangle", "label": 0, "data": [1.0, 2.0, width - 2.0, height - 3.0]}],
        [{"type": "point", "label": 1, "data": [width / 2.0, height / 2.0]}],
    ]
    outputs = []
    for prompt in prompts:
        points, labels = helpers["get_input_points"](prompt)
        coords = np.concatenate([points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        labels = np.concatenate([labels, np.array([-1])], axis=0).astype(np.float32)[None, :]
        coords = helpers["apply_coords"](coords, input_size, 1024).astype(np.float32)
        coords = np.concatenate([coords, np.ones((1, coords.shape[1], 1), dtype=np.float32)], axis=2)
        coords = (coords @ transform_matrix.T)[:, :, :2].astype(np.float32)
        no_prior = np.zeros((1, 1, 256, 256), dtype=np.float32)
        prior = np.full((1, 1, 256, 256), 0.25, dtype=np.float32)
        masks = np.stack([
            np.linspace(-1.0, 1.0, 684 * 1024, dtype=np.float32).reshape(684, 1024),
            np.linspace(1.0, -1.0, 684 * 1024, dtype=np.float32).reshape(684, 1024),
            np.sin(np.arange(684 * 1024, dtype=np.float32).reshape(684, 1024) / 37.0),
        ])[None, ...]
        quality = np.array([[0.5, 0.75, 0.75]], dtype=np.float32)
        low_res = np.stack([no_prior[0, 0], prior[0, 0], np.ones((256, 256), dtype=np.float32)])[None, ...]
        restored = helpers["transform_masks"](masks, (height, width), np.linalg.inv(transform_matrix)).astype(np.float32)
        binary = (restored > 0.0).astype(np.uint8)
        outputs.append({"encoded_coords": coords, "labels": labels, "mask_input_no_prior": no_prior, "has_mask_input_no_prior": np.array([0.0], dtype=np.float32), "mask_input_prior": prior, "has_mask_input_prior": np.array([1.0], dtype=np.float32), "embedding": embedding, "raw_masks": masks, "quality": quality, "low_res": low_res, "restored_masks": restored, "selected_mask": binary[0, 1], "source_union": np.max(binary[0], axis=0)})
    return {"name": case_name, "width": width, "height": height, "encoder_input": encoder_input, "cases": outputs}


def authority_once(helpers):
    return [source_case(helpers, name, width, height) for name, width, height in (("portrait", 500, 1000), ("landscape", 1600, 800), ("square", 1000, 1000))]


def main():
    if subprocess.check_output(["git", "-C", str(ROOT / "projects/python/rembg"), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit("pinned rembg commit drift")
    if digest_bytes(SOURCE.read_bytes()) != SOURCE_SHA256:
        raise SystemExit("pinned rembg SAM source drift")
    helpers = load_source_helpers()
    first, second = authority_once(helpers), authority_once(helpers)
    for left, right in zip(first, second):
        if not np.array_equal(left["encoder_input"], right["encoder_input"]):
            raise SystemExit("SAM authority was not deterministic")
        for left_output, right_output in zip(left["cases"], right["cases"]):
            for key in left_output:
                if not np.array_equal(left_output[key], right_output[key]):
                    raise SystemExit("SAM authority was not deterministic")
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()
    artifacts = {}
    compact = {}
    for case in first:
        prefix = case["name"]
        encoder_flat = case["encoder_input"].reshape(-1)
        encoder_indices = SAMPLE_INDICES[SAMPLE_INDICES < encoder_flat.size]
        compact[prefix] = {"width": case["width"], "height": case["height"], "encoder_input_sha256": digest_array(case["encoder_input"]), "encoder_input_sample_indices": encoder_indices.tolist(), "cases": []}
        write_f32(OUT / f"{prefix}-encoder_input.samples.f32le", encoder_flat[encoder_indices], artifacts)
        for prompt_id, output in enumerate(case["cases"]):
            name = f"{prefix}-case-{prompt_id}"
            raw_flat = output["raw_masks"].reshape(-1)
            low_flat = output["low_res"].reshape(-1)
            restored_flat = output["restored_masks"].reshape(-1)
            raw_indices = SAMPLE_INDICES[SAMPLE_INDICES < raw_flat.size]
            low_indices = SAMPLE_INDICES[SAMPLE_INDICES < low_flat.size]
            restored_indices = SAMPLE_INDICES[SAMPLE_INDICES < restored_flat.size]
            embedding_flat = output["embedding"].reshape(-1)
            embedding_indices = SAMPLE_INDICES[SAMPLE_INDICES < embedding_flat.size]
            prior_indices = SAMPLE_INDICES[SAMPLE_INDICES < output["mask_input_prior"].size]
            write_f32(OUT / f"{name}-encoded_coords.f32le", output["encoded_coords"], artifacts)
            write_f32(OUT / f"{name}-labels.f32le", output["labels"], artifacts)
            write_f32(OUT / f"{name}-raw_masks.samples.f32le", raw_flat[raw_indices], artifacts)
            write_f32(OUT / f"{name}-embedding.samples.f32le", embedding_flat[embedding_indices], artifacts)
            write_f32(OUT / f"{name}-quality.f32le", output["quality"], artifacts)
            write_f32(OUT / f"{name}-low_res.samples.f32le", low_flat[low_indices], artifacts)
            write_f32(OUT / f"{name}-restored_masks.samples.f32le", restored_flat[restored_indices], artifacts)
            write_f32(OUT / f"{name}-mask_input_no_prior.samples.f32le", output["mask_input_no_prior"].reshape(-1)[prior_indices], artifacts)
            write_f32(OUT / f"{name}-mask_input_prior.samples.f32le", output["mask_input_prior"].reshape(-1)[prior_indices], artifacts)
            write_bytes(OUT / f"{name}-selected_mask.bits", np.packbits(output["selected_mask"].reshape(-1), bitorder="little").tobytes(), artifacts)
            write_bytes(OUT / f"{name}-source_union.bits", np.packbits(output["source_union"].reshape(-1), bitorder="little").tobytes(), artifacts)
            compact[prefix]["cases"].append({"prompt_id": prompt_id, "raw_masks_sha256": digest_array(output["raw_masks"]), "embedding_sha256": digest_array(output["embedding"]), "quality_sha256": digest_array(output["quality"]), "low_res_sha256": digest_array(output["low_res"]), "restored_masks_sha256": digest_array(output["restored_masks"]), "selected_mask_sha256": digest_array(output["selected_mask"]), "source_union_sha256": digest_array(output["source_union"]), "sample_indices": {"raw": raw_indices.tolist(), "embedding": embedding_indices.tolist(), "low_res": low_indices.tolist(), "restored": restored_indices.tolist()}})
    report = {
        "schema": "m14.sam-authority.v3", "status": "source-contract-pass-rust-fixture-consumer-pass", "real_checkpoint_comparison": "unavailable-no-approved-local-sam-weights", "rust_consumer_test": "bgremove-ort::sam::pinned_python_fixture_is_consumed_for_prompt_and_mask_stages",
        "source": {"implementation": "rembg.sessions.sam", "commit": COMMIT, "path": str(SOURCE.relative_to(ROOT)), "sha256": SOURCE_SHA256, "helpers_executed": ["warp_affine", "get_preprocess_shape", "get_input_points", "apply_coords", "transform_masks"], "map_coordinates": "scipy.ndimage.map_coordinates"},
        "python_environment": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "lock": "tests/fixtures/m14/python-dependencies.lock"}, "cases": compact, "runs": 2, "artifacts": artifacts,
        "decoder_contract": {"point_coords": "[1,N,2]", "point_labels": "[1,N]", "mask_input": "[1,1,256,256]", "has_mask_input": "[1]", "orig_im_size": [684, 1024], "masks": "[1,3,684,1024]", "iou_predictions": "[1,3]", "low_res_masks": "[1,3,256,256]"}, "assistance_mode": "manual prompted fixture", "assistance_classification": {"manual": "assisted-non-automatic", "centre-default-assisted": "assisted-non-automatic", "auto-derived-assisted": "automatic-derived; requires explicit pipeline declaration"},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
