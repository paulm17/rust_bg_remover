#!/usr/bin/env python3
"""Execute the pinned PyMatting foreground estimator twice.

The installed pymatting package cannot be imported on this host because its
Numba cache locator rejects the read-only site-packages path. Loading this
single pinned source module directly, with JIT/cache decorators replaced by
identity decorators, executes the published Python algorithm without changing
its arithmetic or iteration order.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[3]
PYTHON_ROOT = pathlib.Path(
    "/Volumes/Data/Users/paul/scratch/comfyUI/.venv/lib/python3.12/site-packages"
)
SOURCE = PYTHON_ROOT / "pymatting/foreground/estimate_foreground_ml.py"
OUT = ROOT / "tests/fixtures/m13/reference"
EXPECTED_SOURCE_SHA256 = "fd78cfb439940debc54c89a00638f025c0daef0bacc07fba393572787e9431ca"
EXPECTED_LICENSE_SHA256 = "7be1f2cc0777ebabe655ba85c59ba1026ace94d7d906c791c5ada0e7b4b68323"
EXPECTED_PYMATTING_VERSION = "1.1.15"
BACKGROUND_COMMIT = "fa480627829759b902f8c233388d7aa67ab38099"
BACKGROUND_SOURCE_SHA256 = "4ab631a8f2df06fabd4b05fc36894057ba93f0fd1250b4c60d2bdbe775936047"
REMBG_COMMIT = "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
REMBG_SOURCE_SHA256 = "2e1977b6d16d5369e2c20c5ca5d9ba2e2c0228f8beef39c5d2296df1879df62b"


def load_authority():
    import numba

    def identity_njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1:
            return args[0]
        return lambda function: function

    numba.njit = identity_njit
    numba.prange = range
    spec = importlib.util.spec_from_file_location("pymatting_m13_authority", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def fixture():
    width, height = 6, 4
    foreground = np.array([0.8, 0.2, 0.1], dtype=np.float32)
    background = np.array([0.1, 0.3, 0.9], dtype=np.float32)
    alpha = np.array(
        [[0.0, 0.05, 0.25, 0.5, 0.95, 1.0]] * height, dtype=np.float32
    )
    image = alpha[..., None] * foreground + (1.0 - alpha[..., None]) * background
    return image.astype(np.float32), alpha


def write_f32(path: pathlib.Path, values: np.ndarray):
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def sha256(path: pathlib.Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(path: pathlib.Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    if importlib.metadata.version("PyMatting") != EXPECTED_PYMATTING_VERSION:
        raise SystemExit("PyMatting package version drift")
    if not SOURCE.is_file():
        raise SystemExit(f"pinned PyMatting source missing: {SOURCE}")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("PyMatting foreground source drift")
    license_path = PYTHON_ROOT / "pymatting-1.1.15.dist-info/licenses/LICENSE.md"
    if sha256(license_path) != EXPECTED_LICENSE_SHA256:
        raise SystemExit("PyMatting license drift")
    background = ROOT / "projects/python/backgroundremover"
    rembg = ROOT / "projects/python/rembg"
    if git_head(background) != BACKGROUND_COMMIT or sha256(background / "backgroundremover/bg.py") != BACKGROUND_SOURCE_SHA256:
        raise SystemExit("backgroundremover foreground import contract drift")
    if git_head(rembg) != REMBG_COMMIT or sha256(rembg / "rembg/bg.py") != REMBG_SOURCE_SHA256:
        raise SystemExit("rembg foreground import contract drift")
    for path in (background / "backgroundremover/bg.py", rembg / "rembg/bg.py"):
        if "estimate_foreground_ml" not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"foreground import missing from {path}")
    authority = load_authority()
    image, alpha = fixture()
    outputs = []
    for _ in range(2):
        foreground, background = authority.estimate_foreground_ml(
            image,
            alpha,
            regularization=1e-5,
            n_small_iterations=10,
            n_big_iterations=2,
            small_size=32,
            return_background=True,
            gradient_weight=1.0,
        )
        outputs.append((foreground.copy(), background.copy()))
    if not np.array_equal(outputs[0][0], outputs[1][0]) or not np.array_equal(
        outputs[0][1], outputs[1][1]
    ):
        raise SystemExit("PyMatting authority was not deterministic")
    OUT.mkdir(parents=True, exist_ok=True)
    write_f32(OUT / "image.f32le", image)
    write_f32(OUT / "alpha.f32le", alpha)
    write_f32(OUT / "foreground.f32le", outputs[0][0])
    write_f32(OUT / "background.f32le", outputs[0][1])
    source_hash = sha256(SOURCE)
    artifacts = {
        name: {"bytes": (OUT / name).stat().st_size, "sha256": sha256(OUT / name)}
        for name in ("image.f32le", "alpha.f32le", "foreground.f32le", "background.f32le")
    }
    report = {
        "schema": "m13.foreground-authority.v1",
        "status": "source-executed-deterministic",
        "source": {
            "implementation": "pymatting.foreground.estimate_foreground_ml",
            "version": "1.1.15-installed-pinned-source",
            "path": "pymatting/foreground/estimate_foreground_ml.py",
            "sha256": source_hash,
            "license_sha256": EXPECTED_LICENSE_SHA256,
            "jit": "disabled-for-source-execution-only",
        },
        "consumer_import_contracts": {
            "backgroundremover": {"commit": BACKGROUND_COMMIT, "source": "backgroundremover/bg.py", "source_sha256": BACKGROUND_SOURCE_SHA256, "calls": "pymatting.foreground.estimate_foreground_ml"},
            "rembg": {"commit": REMBG_COMMIT, "source": "rembg/bg.py", "source_sha256": REMBG_SOURCE_SHA256, "calls": "pymatting.foreground.estimate_foreground_ml"},
        },
        "input": {"width": 6, "height": 4, "channels": 3, "working_space": "encoded-srgb"},
        "parameters": {
            "regularization": 1e-5,
            "n_small_iterations": 10,
            "n_big_iterations": 2,
            "small_size": 32,
            "gradient_weight": 1.0,
        },
        "runs": 2,
        "artifacts": artifacts,
        "hidden_rgb_under_alpha_zero_excluded": True,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
