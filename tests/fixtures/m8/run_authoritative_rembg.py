#!/usr/bin/env python3
"""Capture the pinned rembg Bria profile without downloading a checkpoint.

The caller supplies the permissively licensed M3 identity graph as a synthetic
stand-in.  This still executes the checked-in rembg ``BriaRmBgSession`` and
``BaseSession.normalize`` source, including its resize, normalization, output-0
and channel-0 selection, min/max restoration, and mask resize.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


RGBA = bytes(
    value
    for rgb, alpha in zip(
        (
            (0, 64, 255),
            (128, 191, 26),
            (51, 102, 204),
            (255, 26, 0),
            (77, 153, 230),
            (204, 51, 102),
        ),
        (0, 64, 180, 255, 128, 220),
    )
    for value in (*rgb, alpha)
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("base.py", "bria_rmbg.py"):
        digest.update((source_root / "sessions" / name).read_bytes())
    return digest.hexdigest()


class RecordingSession:
    def __init__(self, session: object) -> None:
        self.session = session
        self.inputs: dict[str, np.ndarray] = {}
        self.raw: np.ndarray | None = None

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def run(self, output_names, inputs):
        self.inputs = dict(inputs)
        outputs = self.session.run(output_names, inputs)
        self.raw = outputs[0]
        return outputs


def load_session(source_root: Path, model: Path) -> tuple[object, RecordingSession]:
    """Load only the pinned session module, bypassing rembg package discovery."""
    package = types.ModuleType("rembg")
    package.__path__ = [str(source_root)]
    sys.modules["rembg"] = package
    sessions = types.ModuleType("rembg.sessions")
    sessions.__path__ = [str(source_root / "sessions")]
    sys.modules["rembg.sessions"] = sessions
    base_spec = importlib.util.spec_from_file_location(
        "rembg.sessions.base", source_root / "sessions" / "base.py"
    )
    assert base_spec and base_spec.loader
    base = importlib.util.module_from_spec(base_spec)
    sys.modules["rembg.sessions.base"] = base
    base_spec.loader.exec_module(base)
    spec = importlib.util.spec_from_file_location(
        "rembg.sessions.bria_rmbg", source_root / "sessions" / "bria_rmbg.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["rembg.sessions.bria_rmbg"] = module
    spec.loader.exec_module(module)
    def forbidden_download(*_args, **_kwargs):
        raise RuntimeError("M8 authoritative runner forbids checkpoint downloads")
    if hasattr(module.BriaRmBgSession, "download_models"):
        module.BriaRmBgSession.download_models = forbidden_download
    pooch = sys.modules.get("pooch")
    if pooch is not None and hasattr(pooch, "retrieve"):
        pooch.retrieve = forbidden_download
    session = object.__new__(module.BriaRmBgSession)
    recorder = RecordingSession(ort.InferenceSession(
        str(model), providers=["CPUExecutionProvider"]
    ))
    session.inner_session = recorder
    return session, recorder


def write_f32(path: Path, values: np.ndarray) -> str:
    data = np.asarray(values, dtype="<f4").tobytes(order="C")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("projects/python/rembg/rembg"))
    parser.add_argument("--model", type=Path, default=Path("models/fixtures/m3_identity.onnx"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.resolve()
    model = args.model.resolve()
    repository = source_root.parent
    expected_commit = "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
    commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    tracked_clean = subprocess.run(["git", "-C", str(repository), "diff", "--quiet"]).returncode == 0 and subprocess.run(
        ["git", "-C", str(repository), "diff", "--cached", "--quiet"]).returncode == 0
    if commit != expected_commit or not tracked_clean:
        raise SystemExit("rembg checkout is not the pinned clean tracked source")
    expected_bria = "e3c3747be5af3db15597796a83e73cfa5c464bb5df9c2b047c4113c4bfc3f811"
    expected_base = "ec4c58b33dd47ad6f03883ee375353314b19d5688fffd5c1ddb57bb21e9846a3"
    if sha256(source_root / "sessions" / "bria_rmbg.py") != expected_bria or sha256(source_root / "sessions" / "base.py") != expected_base:
        raise SystemExit("rembg source file hash mismatch")
    if sys.version.split()[0] != "3.12.11" or np.__version__ != "2.3.2" or ort.__version__ != "1.23.2" or Image.__version__ != "12.2.0":
        raise SystemExit("Python/NumPy/Pillow/ONNX Runtime version pin mismatch")
    if not model.is_file():
        raise SystemExit("synthetic model must be preseeded; downloads are disabled")
    expected_model = "988a46eefe32b72ba884f552c88088ba267d28cf46253b086af9b07c58ff50a9"
    if sha256(model) != expected_model:
        raise SystemExit("unexpected synthetic model; refusing to run")
    license_path = source_root.parents[3] / "models/M3_FIXTURE_LICENSE.txt"
    license_sha = "cfed44a701bec837a8ae43d9e6baf69fa5b7fd88aeed383c5ad630b8f430b610"
    if not license_path.is_file() or sha256(license_path) != license_sha:
        raise SystemExit("M3 synthetic license artifact is missing or tampered")
    image = Image.frombytes("RGBA", (3, 2), RGBA)
    session, recorder = load_session(source_root, model)
    # This single call executes the authoritative source's complete predict
    # path. The recording proxy captures its normalized tensor and raw output.
    mask = session.predict(image)[0]
    input_name = recorder.get_inputs()[0].name
    tensor = recorder.inputs[input_name]
    assert recorder.raw is not None
    raw = recorder.raw
    final = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    final[:, :, 3] = np.asarray(mask, dtype=np.uint8)
    profile = args.output / "rembg-bria"
    profile.mkdir(exist_ok=True)
    stages = {
        "decoded-rgb.f32le": write_f32(
            profile / "decoded-rgb.f32le",
            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0,
        ),
        "preprocessed-tensor.f32le": write_f32(profile / "preprocessed-tensor.f32le", tensor),
        "raw-onnx-output.f32le": write_f32(profile / "raw-onnx-output.f32le", raw),
        "restored-alpha.f32le": write_f32(
            profile / "restored-alpha.f32le",
            np.asarray(mask, dtype=np.float32) / 255.0,
        ),
    }
    final_path = profile / "final-straight-alpha-cutout.rgba"
    final_path.write_bytes(final.tobytes(order="C"))
    stages[final_path.name] = sha256(final_path)
    report = {
        "schema": "m8.rmbg-authoritative-profile.v2",
        "authoritative_execution": True,
        "profile": "rembg-bria",
        "source": {
            "commit": commit,
            "tracked_source_clean": tracked_clean,
            "source_file": "projects/python/rembg/rembg/sessions/bria_rmbg.py",
            "source_file_sha256": sha256(source_root / "sessions" / "bria_rmbg.py"),
            "base_source_file_sha256": sha256(source_root / "sessions" / "base.py"),
            "source_tree_sha256": source_tree_sha256(source_root),
        },
        "model": {
            "path": "models/fixtures/m3_identity.onnx",
            "sha256": sha256(model),
            "license": "MIT OR Apache-2.0 (M3 synthetic identity fixture)",
            "license_path": "models/M3_FIXTURE_LICENSE.txt",
            "license_sha256": license_sha,
        },
        "runtime": {
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "python": sys.version.split()[0],
            "public_predict_executed": True,
            "capture": "RecordingSession around BriaRmBgSession.predict",
            "input_name": input_name,
            "output_name": recorder.get_outputs()[0].name,
            "raw_shape": list(raw.shape),
        },
        "input_dimensions": [3, 2],
        "model_output_dimensions": [1024, 1024],
        "stages": stages,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
