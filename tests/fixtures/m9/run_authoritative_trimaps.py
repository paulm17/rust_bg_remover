#!/usr/bin/env python3
"""Execute the pinned rembg and CarveKit trimap source bodies.

This runner deliberately supplies no model/download path.  rembg's matting
module is loaded from its exact pinned source with a no-op session base because
the package initializer imports optional model runtimes; ``post_process`` is
compiled from its exact pinned function body.  CarveKit's trimap modules are
loaded from their exact pinned source files.  The resulting PNGs and the
provenance report are the Level-2 authority for the Rust implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.morphology import disk, opening


REMBG_COMMIT = "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
CARVEKIT_COMMIT = "f141a311af67fb1da64269c508a6d1f786420801"
REMBG_MATTING = "projects/python/rembg/rembg/matting.py"
REMBG_BG = "projects/python/rembg/rembg/bg.py"
REMBG_FILES = (REMBG_MATTING, REMBG_BG)
CARVEKIT_FILES = (
    "projects/python/image-background-remove-tool/carvekit/trimap/cv_gen.py",
    "projects/python/image-background-remove-tool/carvekit/trimap/add_ops.py",
    "projects/python/image-background-remove-tool/carvekit/trimap/generator.py",
)
EXPECTED_RUNTIME = {
    "python": "3.12.11",
    "numpy": "2.3.2",
    "pillow": "12.2.0",
    "scipy": "1.17.0",
    "opencv": "4.10.0",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_hash(root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def tracked_clean(root: Path, paths: list[str]) -> bool:
    return subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *paths],
        check=False,
    ).returncode == 0


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_function(path: Path, name: str, namespace: dict[str, object]) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise RuntimeError(f"expected one {name} function in {path}")
    module = ast.Module(body=[functions[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def load_rembg_matting(path: Path) -> types.ModuleType:
    """Load the complete pinned module without importing rembg's model graph."""
    package = types.ModuleType("rembg")
    package.__path__ = []  # type: ignore[attr-defined]
    sessions = types.ModuleType("rembg.sessions")
    sessions.__path__ = []  # type: ignore[attr-defined]
    base = types.ModuleType("rembg.sessions.base")

    class BaseSession:  # pragma: no cover - source module only needs its type
        pass

    base.BaseSession = BaseSession
    sys.modules["rembg"] = package
    sys.modules["rembg.sessions"] = sessions
    sys.modules["rembg.sessions.base"] = base
    setattr(sessions, "base", base)
    return load_module("rembg.matting", path)


def load_carvekit(root: Path) -> type:
    # Force imports in generator.py to resolve only against the pinned checkout.
    package = types.ModuleType("carvekit")
    package.__path__ = []  # type: ignore[attr-defined]
    trimap = types.ModuleType("carvekit.trimap")
    trimap.__path__ = []  # type: ignore[attr-defined]
    sys.modules["carvekit"] = package
    sys.modules["carvekit.trimap"] = trimap
    cv_gen = load_module("carvekit.trimap.cv_gen", root / CARVEKIT_FILES[0])
    add_ops = load_module("carvekit.trimap.add_ops", root / CARVEKIT_FILES[1])
    setattr(trimap, "cv_gen", cv_gen)
    setattr(trimap, "add_ops", add_ops)
    generator = load_module("carvekit.trimap.generator", root / CARVEKIT_FILES[2])
    return generator.TrimapGenerator


def write_png(path: Path, values: np.ndarray) -> str:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rembg_path = root / REMBG_MATTING
    carvekit_paths = [root / name for name in CARVEKIT_FILES]
    rembg_repo = root / "projects/python/rembg"
    carvekit_repo = root / "projects/python/image-background-remove-tool"
    if git(rembg_repo, "rev-parse", "HEAD") != REMBG_COMMIT:
        raise SystemExit("rembg checkout is not the pinned commit")
    if git(carvekit_repo, "rev-parse", "HEAD") != CARVEKIT_COMMIT:
        raise SystemExit("CarveKit checkout is not the pinned commit")
    rembg_clean = tracked_clean(rembg_repo, ["rembg/matting.py", "rembg/bg.py"])
    carvekit_clean = tracked_clean(
        carvekit_repo,
        ["carvekit/trimap/cv_gen.py", "carvekit/trimap/add_ops.py", "carvekit/trimap/generator.py"],
    )
    if not rembg_clean:
        raise SystemExit("tracked rembg source is dirty")
    if not carvekit_clean:
        raise SystemExit("tracked CarveKit source is dirty")
    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "scipy": __import__("scipy").__version__,
        "opencv": cv2.__version__,
    }
    if runtime != EXPECTED_RUNTIME:
        raise SystemExit(f"pinned M9 runtime mismatch: expected {EXPECTED_RUNTIME}, got {runtime}")

    # Small deterministic mask: thresholds, corners, and a one-pixel island all
    # exercise strict comparisons and explicit outside-image morphology.
    mask = np.array(
        [
            [0, 1, 9, 10, 11, 80, 199, 200],
            [1, 5, 10, 11, 50, 201, 240, 255],
            [9, 10, 11, 100, 200, 201, 254, 255],
            [0, 2, 12, 180, 199, 230, 250, 255],
            [0, 0, 1, 20, 210, 220, 250, 255],
            [0, 3, 4, 30, 200, 205, 245, 255],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(mask, mode="L").save(output / "input-mask.png")

    # A larger border-sensitive mask exercises rembg's literal even kernel and
    # its structure=None default without any model or downloaded asset.
    erode_mask = np.zeros((21, 21), dtype=np.uint8)
    erode_mask[2:19, 2:19] = 255
    Image.fromarray(erode_mask, mode="L").save(output / "erode-mask.png")

    rembg_module = load_rembg_matting(rembg_path)
    rembg_result = rembg_module.trimap_from_mask(
        Image.fromarray(mask, mode="L"), foreground_threshold=200, background_threshold=10, erode_size=3
    )
    rembg_values = np.asarray(rembg_result, dtype=np.uint8)
    rembg_e10 = np.asarray(
        rembg_module.trimap_from_mask(
            Image.fromarray(erode_mask, mode="L"), foreground_threshold=200, background_threshold=10, erode_size=10
        ),
        dtype=np.uint8,
    )
    rembg_e0 = np.asarray(
        rembg_module.trimap_from_mask(
            Image.fromarray(erode_mask, mode="L"), foreground_threshold=200, background_threshold=10, erode_size=0
        ),
        dtype=np.uint8,
    )
    post_process = load_function(
        root / REMBG_BG,
        "post_process",
        {"np": np, "opening": opening, "disk": disk, "gaussian_filter": gaussian_filter, "kernel": disk(1)},
    )
    post_input = np.full((3, 3), 255, dtype=np.uint8)
    Image.fromarray(post_input, mode="L").save(output / "post-process-input.png")
    post_values = np.asarray(post_process(post_input), dtype=np.uint8)
    gaussian_input = np.array([0, 64, 128, 255, 0], dtype=np.uint8)
    gaussian_values = gaussian_filter(gaussian_input.astype(np.float64), sigma=1.3)
    oversized_input = np.array([[1, 1, 1, 1, 0]], dtype=np.uint8)
    oversized_values = binary_erosion(
        oversized_input,
        structure=np.ones((6, 6), dtype=np.uint8),
        border_value=1,
    ).astype(np.uint8)

    TrimapGenerator = load_carvekit(root)
    carvekit_result = TrimapGenerator(prob_threshold=200, kernel_size=1, erosion_iters=1)(
        Image.new("RGB", (mask.shape[1], mask.shape[0])), Image.fromarray(mask, mode="L")
    )
    carvekit_values = np.asarray(carvekit_result, dtype=np.uint8)

    rembg_out = output / "rembg-symmetric-trimap.png"
    rembg_e10_out = output / "rembg-symmetric-trimap-e10.png"
    rembg_e0_out = output / "rembg-symmetric-trimap-e0.png"
    post_out = output / "rembg-post-process.png"
    carvekit_out = output / "carvekit-probability-trimap.png"
    (output / "gaussian-values.json").write_text(
        json.dumps({"input": gaussian_input.tolist(), "sigma": 1.3, "values": gaussian_values.tolist()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "oversized-erosion.json").write_text(
        json.dumps(
            {
                "input": oversized_input.tolist(),
                "shape": [1, 5],
                "kernel_size": 6,
                "border_value": 1,
                "values": oversized_values.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "m9.trimap-authoritative.v1",
        "authoritative_sources_executed": True,
        "input": {
            "path": "input-mask.png",
            "sha256": sha256(output / "input-mask.png"),
            "shape": list(mask.shape),
            "dtype": "uint8",
        },
        "profiles": [
            {
                "name": "rembg-symmetric",
                "source": {
                    "repository": "projects/python/rembg",
                    "commit": REMBG_COMMIT,
                    "source_files": list(REMBG_FILES),
                    "source_file_sha256": {name: sha256(root / name) for name in REMBG_FILES},
                    "source_tree_sha256": source_tree_hash(root, list(REMBG_FILES)),
                    "tracked_source_clean": rembg_clean,
                    "execution": "complete pinned rembg.matting module and exact post_process body; model session never constructed",
                    "license_path": "projects/python/rembg/LICENSE.txt",
                    "license_sha256": sha256(rembg_repo / "LICENSE.txt"),
                },
                "settings": {"foreground_threshold": 200, "background_threshold": 10, "erode_size": 3},
                "output": {"path": rembg_out.name, "sha256": write_png(rembg_out, rembg_values), "shape": list(rembg_values.shape)},
            },
            {
                "name": "carvekit-probability-aware",
                "source": {
                    "repository": "projects/python/image-background-remove-tool",
                    "commit": CARVEKIT_COMMIT,
                    "source_files": CARVEKIT_FILES,
                    "source_file_sha256": {name: sha256(root / name) for name in CARVEKIT_FILES},
                    "source_tree_sha256": source_tree_hash(root, CARVEKIT_FILES),
                    "tracked_source_clean": carvekit_clean,
                    "execution": "exact pinned TrimapGenerator and imported trimap operations",
                    "license_path": "projects/python/image-background-remove-tool/LICENSE",
                    "license_sha256": sha256(carvekit_repo / "LICENSE"),
                },
                "settings": {"probability_threshold": 200, "kernel_size": 1, "erosion_iters": 1},
                "output": {"path": carvekit_out.name, "sha256": write_png(carvekit_out, carvekit_values), "shape": list(carvekit_values.shape)},
            },
        ],
        "rembg_cases": [
            {"name": "erode_size_10", "input": "erode-mask.png", "settings": {"foreground_threshold": 200, "background_threshold": 10, "erode_size": 10}, "path": rembg_e10_out.name, "sha256": write_png(rembg_e10_out, rembg_e10), "shape": list(rembg_e10.shape)},
            {"name": "erode_size_0_default_cross", "input": "erode-mask.png", "settings": {"foreground_threshold": 200, "background_threshold": 10, "erode_size": 0}, "path": rembg_e0_out.name, "sha256": write_png(rembg_e0_out, rembg_e0), "shape": list(rembg_e0.shape)},
            {"name": "post_process_reflect", "input": "post-process-input", "settings": {"sigma": 2.0, "threshold": 127}, "path": post_out.name, "sha256": write_png(post_out, post_values), "shape": list(post_values.shape)},
        ],
        "gaussian_reference": {"input": gaussian_input.tolist(), "sigma": 1.3, "path": "gaussian-values.json", "sha256": sha256(output / "gaussian-values.json"), "values": gaussian_values.tolist()},
        "oversized_erosion_reference": {"path": "oversized-erosion.json", "sha256": sha256(output / "oversized-erosion.json"), "input": oversized_input.tolist(), "kernel_size": 6, "border_value": 1, "values": oversized_values.tolist()},
        "runtime": runtime,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
