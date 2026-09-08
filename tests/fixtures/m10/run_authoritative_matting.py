#!/usr/bin/env python3
"""Run the pinned PyMatting/backgroundremover M10 authority.

This script never downloads data.  It loads the checked-in backgroundremover
wrapper body and the installed, pinned PyMatting 1.1.15 implementation, then
captures the decoded image, coarse alpha, trimap, sparse Laplacian, alpha,
multilevel foreground/background and final RGBA output.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/m10-numba")
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.laplacian.cf_laplacian import cf_laplacian
from pymatting.util.util import stack_images

EXPECTED = {
    "python": "3.12.11",
    "numpy": "2.3.2",
    "pillow": "12.2.0",
    "scipy": "1.17.0",
    "pymatting": "1.1.15",
}
BACKGROUND_COMMIT = "fa480627829759b902f8c233388d7aa67ab38099"
BACKGROUND_SOURCE_SHA256 = "4ab631a8f2df06fabd4b05fc36894057ba93f0fd1250b4c60d2bdbe775936047"
LICENSE_SHA256 = "f4b17964150d51658e31321c6b1d6c182ec4c57274b5cce289bb5c7957d8b3ab"
BACKGROUND_LICENSE_SHA256 = "310ef21ebd68c4095329b33e269fdc4831b7a2468beb8ab6e9ebf8b41bc21cb1"
PYMATTING_LICENSE_SHA256 = "7be1f2cc0777ebabe655ba85c59ba1026ace94d7d906c791c5ada0e7b4b68323"
BACKGROUND_FILE = Path("projects/python/backgroundremover/backgroundremover/bg.py")
LICENSE_FILE = Path("tests/fixtures/m10/LICENSE")
PYMAT_FILES = {
    "pymatting/alpha/estimate_alpha_cf.py": "f03a68fc9dc354740837b3596b2aa97106e7d210dbaebfb9d60cda573f4b69dc",
    "pymatting/laplacian/cf_laplacian.py": "673f02ccce6fde413dfd421f126a8e3f332722939e73b154707f2255b7953038",
    "pymatting/solver/cg.py": "ae48b907782033035506320deb76d92701c7cc70be15046a9583f702605284a6",
    "pymatting/preconditioner/ichol.py": "4d0172d6882dc17998cc19a1313e1915bb7fbe3dee3c82805ac1b41c10c0599b",
    "pymatting/preconditioner/jacobi.py": "9898f81a3d9af388aa872b64212635928ff838f697ad77b283185e8a82873d7d",
    "pymatting/foreground/estimate_foreground_ml.py": "fd78cfb439940debc54c89a00638f025c0daef0bacc07fba393572787e9431ca",
    "pymatting/util/util.py": "5dfc3b60f639372ed45bc85b897186bc351ef1b7076ae775543ee597d98b5906",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def write_f32(path: Path, array: np.ndarray) -> str:
    data = np.asarray(array, dtype="<f4").tobytes(order="C")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def restore_lanczos(array: np.ndarray, width: int, height: int) -> np.ndarray:
    channels = []
    source = np.asarray(array, dtype=np.float64)
    for channel in range(source.shape[2] if source.ndim == 3 else 1):
        plane = source[..., channel] if source.ndim == 3 else source
        restored = Image.fromarray(np.clip(np.rint(plane * 255.0), 0, 255).astype(np.uint8), mode="L").resize((width, height), Image.LANCZOS)
        channels.append(np.asarray(restored, dtype=np.float64) / 255.0)
    return np.stack(channels, axis=2) if source.ndim == 3 else channels[0]


def load_wrapper(repo_root: Path):
    source = (repo_root / BACKGROUND_FILE).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BACKGROUND_FILE))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "alpha_matting_cutout")
    module = ast.Module(body=[function], type_ignores=[])
    compiled = compile(ast.fix_missing_locations(module), str(BACKGROUND_FILE), "exec")
    namespace = {
        "np": np,
        "Image": Image,
        "binary_erosion": binary_erosion,
        "estimate_alpha_cf": estimate_alpha_cf,
        "estimate_foreground_ml": estimate_foreground_ml,
        "stack_images": stack_images,
    }
    exec(compiled, namespace)
    return namespace["alpha_matting_cutout"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": importlib.metadata.version("Pillow"),
        "scipy": importlib.metadata.version("scipy"),
        "pymatting": importlib.metadata.version("PyMatting"),
    }
    if versions != EXPECTED:
        raise SystemExit(f"pinned runtime mismatch: expected {EXPECTED}, got {versions}")
    repo = root / "projects/python/backgroundremover"
    if git(repo, "rev-parse", "HEAD") != BACKGROUND_COMMIT:
        raise SystemExit("backgroundremover checkout is not pinned")
    if git(repo, "status", "--short", "--untracked-files=no"):
        raise SystemExit("backgroundremover tracked source is dirty")
    source_hash = sha(root / BACKGROUND_FILE)
    license_hash = sha(root / LICENSE_FILE)
    background_license = root / "projects/python/backgroundremover/LICENSE.txt"
    background_license_hash = sha(background_license)
    if source_hash != BACKGROUND_SOURCE_SHA256 or license_hash != LICENSE_SHA256:
        raise SystemExit("backgroundremover source or fixture license drift")
    if background_license_hash != BACKGROUND_LICENSE_SHA256:
        raise SystemExit("backgroundremover license drift")
    site = Path(inspect.getfile(estimate_alpha_cf)).parents[1]
    source_files = {}
    for rel, expected in PYMAT_FILES.items():
        path = site / rel.removeprefix("pymatting/")
        got = sha(path)
        if got != expected:
            raise SystemExit(f"PyMatting source drift for {rel}: {got}")
        source_files[rel] = {"path": rel, "sha256": got}
    pymatting_license = site.parent / "pymatting-1.1.15.dist-info/licenses/LICENSE.md"
    if sha(pymatting_license) != PYMATTING_LICENSE_SHA256:
        raise SystemExit("PyMatting license artifact drift")

    canonical_h, canonical_w = 45, 65
    yy, xx = np.mgrid[:canonical_h, :canonical_w]
    true_alpha = np.clip((xx.astype(np.float64) - 5.0) / 44.0, 0.0, 1.0)
    fg_true = np.stack([0.20 + 0.60 * true_alpha, 0.10 + 0.40 * (1.0 - true_alpha), 0.40 + 0.20 * true_alpha], axis=2)
    bg_true = np.array([0.05, 0.15, 0.25], dtype=np.float64)
    canonical_image = true_alpha[..., None] * fg_true + (1.0 - true_alpha[..., None]) * bg_true
    canonical_pil = Image.fromarray(np.rint(canonical_image * 255.0).astype(np.uint8), mode="RGB")
    canonical_mask = Image.fromarray(np.clip(np.rint(true_alpha * 255.0), 0, 255).astype(np.uint8), mode="L")
    working_pil = canonical_pil.copy()
    working_pil.thumbnail((48, 48), Image.LANCZOS)
    working_mask_pil = canonical_mask.resize(working_pil.size, Image.LANCZOS)
    image = np.asarray(working_pil, dtype=np.float64) / 255.0
    coarse = np.asarray(working_mask_pil, dtype=np.float64) / 255.0
    h, w = image.shape[:2]
    coarse_u8 = np.asarray(working_mask_pil, dtype=np.uint8)
    fg_known = binary_erosion(coarse_u8 > 240, structure=np.ones((3, 3), dtype=np.int64))
    bg_known = binary_erosion(coarse_u8 < 10, structure=np.ones((3, 3), dtype=np.int64), border_value=1)
    trimap = np.full((h, w), 0.5, dtype=np.float64)
    trimap[fg_known] = 1.0
    trimap[bg_known] = 0.0

    write_f32(out / "decoded-rgb.f32le", image)
    write_f32(out / "coarse-alpha.f32le", coarse)
    write_f32(out / "canonical-rgb.f32le", np.asarray(canonical_pil, dtype=np.float64) / 255.0)
    write_f32(out / "canonical-coarse.f32le", np.asarray(canonical_mask, dtype=np.float64) / 255.0)
    canonical_trimap = np.asarray(Image.fromarray(np.rint(trimap * 255.0).astype(np.uint8), mode="L").resize((canonical_w, canonical_h), Image.NEAREST), dtype=np.float64) / 255.0
    canonical_trimap[:, :3] = 0.0
    canonical_trimap[:, -4:] = 1.0
    write_f32(out / "canonical-trimap.f32le", canonical_trimap)
    write_f32(out / "true-alpha.f32le", true_alpha)
    write_f32(out / "true-foreground.f32le", fg_true)
    write_f32(out / "true-background.f32le", np.broadcast_to(bg_true, fg_true.shape))
    analytic_h, analytic_w = 13, 17
    analytic_x = np.arange(analytic_w, dtype=np.float64)[None, :]
    analytic_alpha = np.broadcast_to(np.clip((analytic_x - 2.0) / 12.0, 0.0, 1.0), (analytic_h, analytic_w))
    analytic_fg = np.broadcast_to(np.array([0.82, 0.18, 0.12], dtype=np.float64), (analytic_h, analytic_w, 3))
    analytic_bg = np.broadcast_to(np.array([0.08, 0.22, 0.78], dtype=np.float64), (analytic_h, analytic_w, 3))
    analytic_image = analytic_alpha[..., None] * analytic_fg + (1.0 - analytic_alpha[..., None]) * analytic_bg
    analytic_trimap = np.full((analytic_h, analytic_w), 0.5, dtype=np.float64)
    analytic_trimap[:, 0] = 0.0
    analytic_trimap[:, -1] = 1.0
    write_f32(out / "analytic-image.f32le", analytic_image)
    write_f32(out / "analytic-coarse-alpha.f32le", analytic_alpha)
    write_f32(out / "analytic-trimap.f32le", analytic_trimap)
    write_f32(out / "analytic-true-alpha.f32le", analytic_alpha)
    write_f32(out / "analytic-true-foreground.f32le", analytic_fg)
    write_f32(out / "analytic-true-background.f32le", analytic_bg)
    write_f32(out / "trimap.f32le", trimap)
    known = (trimap == 0.0) | (trimap == 1.0)
    (out / "constraints.json").write_text(json.dumps({"known": known.reshape(-1).astype(int).tolist(), "unknown": (~known).reshape(-1).astype(int).tolist()}, separators=(",", ":")) + "\n")
    lap = cf_laplacian(image, epsilon=1e-7, radius=1, is_known=known).tocsr()
    triplets = [[int(r), int(c), float(v)] for r, c, v in zip(*lap.nonzero(), lap.data)]
    # scipy's zip order above is not guaranteed across versions; use explicit COO.
    coo = lap.tocoo()
    triplets = [[int(r), int(c), float(v)] for r, c, v in zip(coo.row, coo.col, coo.data)]
    (out / "laplacian.json").write_text(json.dumps({"shape": [h * w, h * w], "triplets": triplets}, sort_keys=True, separators=(",", ":")) + "\n")
    alpha = estimate_alpha_cf(image, trimap, laplacian_kwargs={"epsilon": 1e-7, "radius": 1})
    foreground, background = estimate_foreground_ml(image, alpha, return_background=True)
    write_f32(out / "alpha.f32le", alpha)
    write_f32(out / "foreground.f32le", foreground)
    write_f32(out / "background.f32le", background)
    write_f32(out / "restored-alpha.f32le", restore_lanczos(alpha, canonical_w, canonical_h))
    write_f32(out / "restored-foreground.f32le", restore_lanczos(foreground, canonical_w, canonical_h))
    write_f32(out / "restored-background.f32le", restore_lanczos(background, canonical_w, canonical_h))

    wrapper = load_wrapper(root)
    pil_img = canonical_pil.copy()
    pil_mask = canonical_mask.copy()
    cutout = stack_images(foreground, alpha)
    cutout = Image.fromarray(np.clip(cutout * 255, 0, 255).astype(np.uint8), mode="RGBA").resize((canonical_w, canonical_h), Image.LANCZOS)
    cutout.save(out / "chain-final-rgba.png", format="PNG", optimize=False, compress_level=9)
    wrapped = wrapper(pil_img, pil_mask, 240, 10, 3, 48)
    if not np.array_equal(np.asarray(wrapped), np.asarray(cutout)):
        raise SystemExit("connected M10 stage chain differs from pinned wrapper output")
    wrapped.save(out / "final-rgba.png", format="PNG", optimize=False, compress_level=9)
    final_hash = sha(out / "final-rgba.png")

    report = {
        "schema": "bgremove.m10.authoritative.v1",
        "authoritative_sources_executed": True,
        "stage_chain": {"connected": True, "decoded_to_working": True, "final_from_captured_stages": True, "wrapper_final_matches_chain": True},
        "source": {
            "execution": "exact pinned alpha_matting_cutout function body loaded from bg.py; package import is intentionally avoided because optional video dependency is absent",
            "backgroundremover_commit": BACKGROUND_COMMIT,
            "backgroundremover_file": str(BACKGROUND_FILE),
            "backgroundremover_bg_py_sha256": source_hash,
            "pymatting_version": EXPECTED["pymatting"],
            "pymatting_source_files": source_files,
            "python_runtime": versions,
            "license_artifact": {"path": "tests/fixtures/m10/LICENSE", "sha256": license_hash},
            "backgroundremover_license": {"identifier": "MIT", "path": "projects/python/backgroundremover/LICENSE.txt", "sha256": background_license_hash},
            "pymatting_license": {"identifier": "MIT", "path": "pymatting-1.1.15.dist-info/licenses/LICENSE.md", "sha256": PYMATTING_LICENSE_SHA256},
            "network_downloads": False,
        },
        "input": {"width": w, "height": h, "image_sha256": sha(out / "decoded-rgb.f32le"), "coarse_sha256": sha(out / "coarse-alpha.f32le"), "trimap_sha256": sha(out / "trimap.f32le")},
        "analytic": {"true_alpha_sha256": sha(out / "true-alpha.f32le"), "true_foreground_sha256": sha(out / "true-foreground.f32le"), "true_background_sha256": sha(out / "true-background.f32le"), "canonical_dimensions": [canonical_w, canonical_h], "separate_case": {"width": analytic_w, "height": analytic_h, "image_sha256": sha(out / "analytic-image.f32le"), "coarse_sha256": sha(out / "analytic-coarse-alpha.f32le"), "trimap_sha256": sha(out / "analytic-trimap.f32le"), "true_alpha_sha256": sha(out / "analytic-true-alpha.f32le"), "true_foreground_sha256": sha(out / "analytic-true-foreground.f32le"), "true_background_sha256": sha(out / "analytic-true-background.f32le")}},
        "stages": {
            "laplacian": {"path": "laplacian.json", "sha256": sha(out / "laplacian.json"), "shape": [h * w, h * w], "nnz": int(lap.nnz)},
            "constraints": {"path": "constraints.json", "sha256": sha(out / "constraints.json"), "shape": [h * w]},
            "alpha": {"path": "alpha.f32le", "sha256": sha(out / "alpha.f32le"), "shape": [h, w]},
            "foreground": {"path": "foreground.f32le", "sha256": sha(out / "foreground.f32le"), "shape": [h, w, 3]},
            "background": {"path": "background.f32le", "sha256": sha(out / "background.f32le"), "shape": [h, w, 3]},
            "final_rgba": {"path": "chain-final-rgba.png", "sha256": sha(out / "chain-final-rgba.png"), "shape": [canonical_h, canonical_w, 4]},
            "wrapper_final_rgba": {"path": "final-rgba.png", "sha256": final_hash, "shape": [canonical_h, canonical_w, 4]},
        },
        "parameters": {"epsilon": 1e-7, "radius": 1, "constraint_mode": "unknown-only; known alpha fixed", "foreground": {"regularization": 1e-5, "n_small_iterations": 10, "n_big_iterations": 2, "small_size": 32, "gradient_weight": 1.0, "non_power_two_levels": True}, "wrapper": {"foreground_threshold": 240, "background_threshold": 10, "erode_structure_size": 3, "base_size": 48, "canonical_dimensions": [canonical_w, canonical_h], "working_dimensions": [w, h], "resampling": "Pillow LANCZOS; uint8 quantization before source stages"}},
        "resource_limits": {"max_pixels": 1000000, "max_sparse_nnz": 50000000, "max_assembly_entries": 10000000, "max_memory_bytes": 536870912, "max_iterations": 10000, "max_working_dimension": 4096, "network_downloads": False},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
