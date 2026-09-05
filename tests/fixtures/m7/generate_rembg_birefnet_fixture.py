#!/usr/bin/env python3
"""Generate pinned rembg BiRefNet level-2 fixtures; never downloads weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import types
from pathlib import Path


REMBG_COMMIT = "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
PYTHON_VERSION = "3.12.12"
NUMPY_VERSION = "1.26.4"
ONNXRUNTIME_VERSION = "1.23.2"
PILLOW_VERSION = "10.4.0"
WEIGHT_LICENSE_PATH = "models/M7_BIREFNET_WEIGHT_LICENSE.txt"
WEIGHT_LICENSE_IDENTIFIER = "MIT (BiRefNet upstream)"
WEIGHT_LICENSE_SHA256 = "92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb"
WEIGHT_LICENSE_SOURCE_URL = "https://raw.githubusercontent.com/ZhengPeng7/BiRefNet/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/LICENSE"
WEIGHT_LICENSE_SOURCE_COMMIT = "ebcc0bc8ec7fe919cec829f2dea656b3078acddc"
WEIGHT_SHA256 = {
    "general": "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
    "general-lite": "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
}
WEIGHT_FILES = {
    "general": "BiRefNet-general-epoch_244.onnx",
    "general-lite": "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, values) -> str:
    import numpy as np

    data = np.asarray(values, dtype="<f4").tobytes()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("tests/fixtures/m5/landscape-3x2.png"))
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/m7/reference"))
    args = parser.parse_args()

    if platform.python_version() != PYTHON_VERSION:
        raise SystemExit(f"unsupported Python {platform.python_version()}; expected {PYTHON_VERSION}")

    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "-C", str(repo / "projects/python/rembg"), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REMBG_COMMIT:
        raise SystemExit(f"unexpected rembg checkout {commit}; expected {REMBG_COMMIT}")
    source_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo / "projects/python/rembg"),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "rembg",
        ],
        text=True,
    ).strip()
    if source_status:
        raise SystemExit(f"pinned rembg source tree is dirty:\n{source_status}")
    sys.path.insert(0, str(repo / "projects/python/rembg"))
    # Load only the pinned session modules. Importing the top-level rembg
    # package would eagerly import optional SAM/matting dependencies.
    rembg_package = types.ModuleType("rembg")
    rembg_package.__path__ = [str(repo / "projects/python/rembg/rembg")]
    sessions_package = types.ModuleType("rembg.sessions")
    sessions_package.__path__ = [str(repo / "projects/python/rembg/rembg/sessions")]
    sys.modules["rembg"] = rembg_package
    sys.modules["rembg.sessions"] = sessions_package

    # The generator only exercises segmentation sessions. The pinned rembg
    # package imports optional matting modules from its package initializer;
    # provide fail-loud stubs for those unused optional imports rather than
    # installing unpinned dependencies or changing the source checkout.
    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("optional rembg matting dependency was not used by this fixture")

    pymatting = types.ModuleType("pymatting")
    pymatting_alpha = types.ModuleType("pymatting.alpha")
    pymatting_alpha_cf = types.ModuleType("pymatting.alpha.estimate_alpha_cf")
    pymatting_alpha_cf.estimate_alpha_cf = _unavailable
    pymatting_foreground = types.ModuleType("pymatting.foreground")
    pymatting_foreground_ml = types.ModuleType("pymatting.foreground.estimate_foreground_ml")
    pymatting_foreground_ml.estimate_foreground_ml = _unavailable
    pymatting_util = types.ModuleType("pymatting.util")
    pymatting_util_util = types.ModuleType("pymatting.util.util")
    pymatting_util_util.stack_images = _unavailable
    scipy = types.ModuleType("scipy")
    scipy_ndimage = types.ModuleType("scipy.ndimage")
    scipy_ndimage.binary_erosion = _unavailable
    scipy_ndimage.gaussian_filter = _unavailable
    skimage = types.ModuleType("skimage")
    skimage_morphology = types.ModuleType("skimage.morphology")
    skimage_morphology.disk = _unavailable
    skimage_morphology.opening = _unavailable
    pooch = types.ModuleType("pooch")
    pooch.retrieve = _unavailable
    sys.modules.update(
        {
            "pymatting": pymatting,
            "pymatting.alpha": pymatting_alpha,
            "pymatting.alpha.estimate_alpha_cf": pymatting_alpha_cf,
            "pymatting.foreground": pymatting_foreground,
            "pymatting.foreground.estimate_foreground_ml": pymatting_foreground_ml,
            "pymatting.util": pymatting_util,
            "pymatting.util.util": pymatting_util_util,
            "scipy": scipy,
            "scipy.ndimage": scipy_ndimage,
            "skimage": skimage,
            "skimage.morphology": skimage_morphology,
            "pooch": pooch,
        }
    )
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from rembg.sessions.birefnet_general import BiRefNetSessionGeneral
    sessions_package.BiRefNetSessionGeneral = BiRefNetSessionGeneral
    from rembg.sessions.birefnet_general_lite import BiRefNetSessionGeneralLite

    if (
        np.__version__ != NUMPY_VERSION
        or ort.__version__ != ONNXRUNTIME_VERSION
        or Image.__version__ != PILLOW_VERSION
    ):
        raise SystemExit(
            f"unsupported fixture dependencies: numpy={np.__version__}, ort={ort.__version__}, Pillow={Image.__version__}"
        )

    image = Image.open(repo / args.input).convert("RGB")
    source = np.asarray(image, dtype=np.float32) / 255.0

    provenance = {
        "repository": "projects/python/rembg",
        "commit": REMBG_COMMIT,
        "source_files": {
            "rembg/rembg/sessions/base.py": sha256(repo / "projects/python/rembg/rembg/sessions/base.py"),
            "rembg/rembg/sessions/birefnet_general.py": sha256(repo / "projects/python/rembg/rembg/sessions/birefnet_general.py"),
        },
        "dependencies": {"onnxruntime": ort.__version__, "Pillow": Image.__version__, "numpy": np.__version__},
        "weight_license": {
            "path": WEIGHT_LICENSE_PATH,
            "identifier": WEIGHT_LICENSE_IDENTIFIER,
            "sha256": WEIGHT_LICENSE_SHA256,
            "upstream_repository": "https://github.com/ZhengPeng7/BiRefNet",
            "upstream_commit": WEIGHT_LICENSE_SOURCE_COMMIT,
            "upstream_url": WEIGHT_LICENSE_SOURCE_URL,
        },
        "input": {"path": str(args.input), "sha256": sha256(repo / args.input), "dimensions": list(image.size)},
    }
    if sha256(repo / WEIGHT_LICENSE_PATH) != WEIGHT_LICENSE_SHA256:
        raise SystemExit("BiRefNet upstream weight license artifact hash mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    cases = {}
    with tempfile.TemporaryDirectory(prefix="m7-rembg-model-home-") as home:
        os.environ["U2NET_HOME"] = home
        for variant, filename in WEIGHT_FILES.items():
            weight = args.weights_dir / filename
            if not weight.is_file() or sha256(weight) != WEIGHT_SHA256[variant]:
                raise SystemExit(f"missing or mismatched external weight {weight}")
            name = "birefnet-general" if variant == "general" else "birefnet-general-lite"
            model_dir = Path(home) / "models" / name
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / f"{name}.onnx").symlink_to(weight.resolve())
            session_type = BiRefNetSessionGeneral if variant == "general" else BiRefNetSessionGeneralLite
            session = session_type(name, ort.SessionOptions(), providers=["CPUExecutionProvider"])
            feed = session.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (1024, 1024))
            input_name = next(iter(feed))
            raw = session.inner_session.run(None, feed)[0][:, 0, :, :]
            logits = np.asarray(raw, dtype=np.float32).squeeze()
            mask = session.predict(image)[0]
            restored_u8 = np.asarray(mask, dtype=np.uint8)
            cutout = np.concatenate([np.asarray(image, dtype=np.uint8), restored_u8[:, :, None]], axis=2)
            case_dir = args.output / variant / "landscape-3x2"
            case_dir.mkdir(parents=True, exist_ok=True)
            files = {
                "decoded_rgb": write_f32(case_dir / "decoded-rgb.f32le", source),
                "preprocessed_tensor": write_f32(case_dir / "preprocessed-tensor.f32le", feed[input_name]),
                "raw_output": write_f32(case_dir / "raw-output.f32le", logits),
                "restored_alpha": write_f32(case_dir / "restored-alpha.f32le", restored_u8.astype(np.float32) / 255.0),
                "final_cutout": hashlib.sha256(cutout.tobytes()).hexdigest(),
            }
            (case_dir / "final-straight-alpha-cutout.rgba").write_bytes(cutout.tobytes())
            files["final_cutout_file"] = sha256(case_dir / "final-straight-alpha-cutout.rgba")
            cases[variant] = {
                "weights": {"path": f"external/{filename}", "sha256": WEIGHT_SHA256[variant], "bytes": weight.stat().st_size},
                "onnx": {"input_name": input_name, "output_name": session.inner_session.get_outputs()[0].name, "input_shape": list(feed[input_name].shape), "raw_shape": list(logits.shape)},
                "files": files,
                "normalization": "pinned rembg BaseSession.normalize: global resized-image max, then ImageNet mean/std",
                "postprocess": "pinned rembg BiRefNetSession*.predict: output0/channel0 logits -> sigmoid -> per-image minmax -> uint8 floor -> Pillow LANCZOS",
            }

    report = {
        "schema": "m7.rembg-birefnet-python-level2.v1",
        "provenance": provenance,
        "tolerances": {"preprocessed_tensor_max_abs": 2e-6, "raw_output_max_abs": 5e-4, "restored_alpha_max_abs": 1e-6},
        "cases": cases,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
