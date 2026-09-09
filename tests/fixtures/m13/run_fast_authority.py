#!/usr/bin/env python3
"""Execute externally provisioned, pinned PhotoRoom blur-fusion source twice.

The upstream repository has no checked-in license file, so source is never
vendored in this repository.  The caller provisions the exact file and passes
its path and commit through the environment.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import platform
import subprocess
import os

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = pathlib.Path(os.environ.get("M13_PHOTOROOM_SOURCE", ""))
REPO = pathlib.Path(os.environ.get("M13_PHOTOROOM_REPO", ""))
OUT = ROOT / "tests/fixtures/m13/fast-reference"
EXPECTED_COMMIT = "a98fe5dfe4b61521de1a92dcaeb6d138f95bc97d"
EXPECTED_SOURCE_SHA256 = "b9cbb83ddfa93c4fb2686fe99f9bc61769d9b6001df363660c664de8ad9ce94c"
EXPECTED = {"python": "3.12.11", "numpy": "2.3.2", "opencv-python": "4.10.0.84"}


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("photoroom_m13_authority", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_f32(path: pathlib.Path, value: np.ndarray):
    data = np.asarray(value, dtype="<f4").tobytes(order="C")
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def cases():
    for case_id, (height, width) in enumerate(((3, 5), (4, 7), (6, 2))):
        yy, xx = np.mgrid[:height, :width]
        image = np.stack(
            [
                ((xx * 17 + yy * 13 + 3) % 101) / 100.0,
                ((xx * 7 + yy * 29 + 11) % 97) / 96.0,
                ((xx * 31 + yy * 5 + 19) % 89) / 88.0,
            ],
            axis=2,
        ).astype(np.float32)
        alpha = np.clip(
            0.07 + 0.11 * xx + 0.13 * yy + 0.03 * ((xx + 2 * yy) % 3),
            0.0,
            1.0,
        ).astype(np.float32)
        yield case_id, image, alpha


def main():
    if not SOURCE.is_file():
        raise SystemExit("set M13_PHOTOROOM_SOURCE to the externally provisioned exact source file")
    if not REPO.is_dir():
        raise SystemExit("set M13_PHOTOROOM_REPO to the externally provisioned checkout")
    if os.environ.get("M13_PHOTOROOM_COMMIT") != EXPECTED_COMMIT:
        raise SystemExit("set M13_PHOTOROOM_COMMIT to the pinned PhotoRoom commit")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != EXPECTED_COMMIT:
        raise SystemExit(f"PhotoRoom checkout drift: expected {EXPECTED_COMMIT}, got {actual_commit}")
    if SOURCE.resolve() != (REPO / "blurfusion_foreground_estimation.py").resolve():
        raise SystemExit("PhotoRoom source must be the pinned checkout's exact authority file")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv-python": importlib.metadata.version("opencv-python"),
    }
    if versions != EXPECTED:
        raise SystemExit(f"pinned runtime mismatch: expected {EXPECTED}, got {versions}")
    if sha(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("externally provisioned PhotoRoom source hash drift")
    authority = load_source()
    OUT.mkdir(parents=True, exist_ok=True)
    cases_report = []
    for case_id, image, alpha in cases():
        outputs = []
        for _ in range(2):
            a = alpha[:, :, None]
            pass1_f, pass1_b = authority.FB_blur_fusion_foreground_estimator(
                image, image, image, a, r=90
            )
            pass2_f, pass2_b = authority.FB_blur_fusion_foreground_estimator(
                image, pass1_f, pass1_b, a, r=6
            )
            outputs.append((pass1_f.copy(), pass1_b.copy(), pass2_f.copy(), pass2_b.copy()))
        if not all(np.array_equal(outputs[0][i], outputs[1][i]) for i in range(4)):
            raise SystemExit(f"authority case {case_id} was not deterministic")
        case_out = OUT / f"case-{case_id}"
        case_out.mkdir(parents=True, exist_ok=True)
        values = {
            "image.f32le": image,
            "alpha.f32le": alpha,
            "pass1-foreground.f32le": outputs[0][0],
            "pass1-blurred-background.f32le": outputs[0][1],
            "pass2-foreground.f32le": outputs[0][2],
            "pass2-blurred-background.f32le": outputs[0][3],
        }
        artifacts = {name: write_f32(case_out / name, value) for name, value in values.items()}
        cases_report.append(
            {
                "id": case_id,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "kernel_widths": [90, 6],
                "artifacts": artifacts,
            }
        )
    report = {
        "schema": "m13.photoroom-fast-authority.v1",
        "status": "source-executed-deterministic",
        "source": {
            "repository": "https://github.com/Photoroom/fast-foreground-estimation",
            "file": "blurfusion_foreground_estimation.py",
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "commit": EXPECTED_COMMIT,
            "source_url": "https://raw.githubusercontent.com/Photoroom/fast-foreground-estimation/a98fe5dfe4b61521de1a92dcaeb6d138f95bc97d/blurfusion_foreground_estimation.py",
            "license_status": "upstream repository has no checked-in license file; source is not retained, only numerical parity artifacts are retained",
        },
        "runtime": versions,
        "opencv": {"blur": "cv2.blur", "border": "BORDER_DEFAULT/BORDER_REFLECT_101", "kernel": "(r,r), even anchor preserved"},
        "parameters": {"coarse_kernel_width": 90, "fine_kernel_width": 6, "coarse_iterations": 1, "fine_iterations": 1, "epsilon": 1e-5, "passes": 2, "background_state": "pass1 blurred_B is pass2 B"},
        "runs": 2,
        "cases": cases_report,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
