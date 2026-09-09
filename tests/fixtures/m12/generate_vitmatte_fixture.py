#!/usr/bin/env python3
"""Generate and execute the deterministic offline M12 ViTMatte contract graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import PIL
import scipy
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
MODEL_SHA256 = "114b5870dd444275958dadf739b60ad7aa452084830559d0021f757e41e43d5b"
REMBG_COMMIT = "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"
REMBG_MATTING = ROOT / "projects/python/rembg/rembg/matting.py"
REMBG_LICENSE = ROOT / "projects/python/rembg/LICENSE.txt"
EXPECTED_RUNTIME = {
    "python": "3.12.11",
    "onnxruntime": "1.23.2",
    "numpy": "2.3.2",
    "pillow": "12.2.0",
    "scipy": "1.17.0",
    "torch": "2.9.1",
}


class SyntheticViTMatte(torch.nn.Module):
    def forward(self, pixel_values):
        # The graph is deliberately tiny but has the production four-channel
        # input/output shapes. It exposes RGB and trimap planes so preprocessing
        # and alpha restoration are exercised without claiming real-model quality.
        rgb = pixel_values[:, :3].mean(dim=1, keepdim=True)
        trimap = pixel_values[:, 3:4]
        return torch.clamp(0.25 * rgb + 0.75 * trimap, 0.0, 1.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def relevant_files_sha256() -> str:
    digest = hashlib.sha256()
    relative = "projects/python/rembg/rembg/matting.py"
    digest.update(relative.encode("utf-8"))
    digest.update(REMBG_MATTING.read_bytes())
    return digest.hexdigest()


def rembg_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "projects/python/rembg"), *args], text=True
    ).strip()


class CaptureSession:
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input = None
        self.raw = None

    def run(self, _output_names, inputs):
        self.input = np.asarray(inputs["pixel_values"], dtype=np.float32).copy()
        self.raw = self.session.run(["alphas"], {"pixel_values": self.input})[0].astype(np.float32)
        return [self.raw]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/fixtures/vitmatte_synthetic.onnx")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    if not args.model.exists():
        torch.onnx.export(
            SyntheticViTMatte().eval(),
            (torch.zeros((1, 4, 1024, 1024)),),
            args.model,
            input_names=["pixel_values"],
            output_names=["alphas"],
            opset_version=17,
            do_constant_folding=False,
            dynamo=False,
        )
    if sha(args.model) != MODEL_SHA256:
        raise SystemExit(f"fixture hash drift: {sha(args.model)}")
    if rembg_git("rev-parse", "HEAD") != REMBG_COMMIT:
        raise SystemExit("pinned rembg commit drifted")
    if rembg_git("status", "--short", "--untracked-files=no"):
        raise SystemExit("pinned rembg tracked source is dirty")
    runtime = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    if runtime != EXPECTED_RUNTIME:
        raise SystemExit(f"runtime drift: expected {EXPECTED_RUNTIME}, got {runtime}")
    if sha(REMBG_LICENSE) != "90a3215072968fd304669c5389f04f1274a587abdd0507d99dead0f5511f8999":
        raise SystemExit("rembg licence drifted")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inputs = {item.name: item.shape for item in session.get_inputs()}
    outputs = {item.name: item.shape for item in session.get_outputs()}
    if inputs != {"pixel_values": [1, 4, 1024, 1024]}:
        raise SystemExit(f"unexpected input contract: {inputs}")
    if outputs != {"alphas": [1, 1, 1024, 1024]}:
        raise SystemExit(f"unexpected output contract: {outputs}")
    # A synthetic source grid gives every RGB channel and each trimap class a
    # deterministic non-constant signal for the Python authority report.
    width, height = 13, 9
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = [(13 * x + 3 * y) % 256, (5 * x + 17 * y) % 256, (19 * x + 7 * y) % 256]
    mask = np.full((height, width), 128, dtype=np.uint8)
    mask[:, :4] = 0
    mask[:, -4:] = 255
    image_pil = Image.fromarray(image, mode="RGB")
    mask_pil = Image.fromarray(mask, mode="L")

    # Import the pinned source and patch only its session lookup. The actual
    # vitmatte_alpha function owns trimap construction, Pillow interpolation,
    # tensor assembly, uint8 conversion and restoration.
    sys.path.insert(0, str(ROOT / "projects/python/rembg"))
    import rembg.matting as matting

    capture = CaptureSession(session)
    matting._get_session = lambda _variant: capture
    matting._sessions.clear()
    restored = matting.vitmatte_alpha(
        image_pil,
        mask_pil,
        variant="small-distinctions-646",
        foreground_threshold=240,
        background_threshold=10,
        erode_size=1,
    )
    if capture.input is None or capture.raw is None:
        raise SystemExit("pinned rembg function did not invoke the capture session")
    tensor = capture.input
    raw = capture.raw
    source_trimap = np.asarray(
        matting.trimap_from_mask(mask_pil, foreground_threshold=240, background_threshold=10, erode_size=1),
        dtype=np.uint8,
    )
    restored_u8 = np.asarray(restored, dtype=np.uint8)
    if not np.isfinite(tensor).all() or not np.isfinite(raw).all() or restored_u8.shape != (height, width):
        raise SystemExit("synthetic source execution produced invalid outputs")
    if set(np.unique(source_trimap).tolist()) != {0, 128, 255}:
        raise SystemExit("source trimap did not exercise all three classes")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "image.u8").write_bytes(image.tobytes())
    (args.output / "mask.u8").write_bytes(mask.tobytes())
    (args.output / "trimap.u8").write_bytes(source_trimap.tobytes())
    (args.output / "authority-input.f32le").write_bytes(tensor.tobytes())
    (args.output / "authority-output.f32le").write_bytes(raw.tobytes())
    (args.output / "restored-alpha.u8").write_bytes(restored_u8.tobytes())
    artifact_paths = ["image.u8", "mask.u8", "trimap.u8", "authority-input.f32le", "authority-output.f32le", "restored-alpha.u8"]
    try:
        model_report_path = str(args.model.relative_to(ROOT))
    except ValueError:
        model_report_path = str(args.model)
    report = {
        "schema": "m12.vitmatte-authoritative.v1",
        "source_execution": True,
        "real_checkpoint_executed": False,
        "network_downloads": False,
        "model": {"path": model_report_path, "sha256": sha(args.model), "bytes": args.model.stat().st_size},
        "variants": matting.VARIANTS,
        "contract": {"input": [1, 4, 1024, 1024], "output": [1, 1, 1024, 1024], "rgb": "Pillow bilinear / 255", "trimap": "Pillow nearest / 255", "normalization": "none"},
        "authority_artifacts": {name: {"bytes": (args.output / name).stat().st_size, "sha256": sha(args.output / name)} for name in artifact_paths},
        "source": {"repository": "projects/python/rembg", "commit": REMBG_COMMIT, "tracked_source_clean": True, "file": "projects/python/rembg/rembg/matting.py", "sha256": sha(REMBG_MATTING), "relevant_files_sha256": relevant_files_sha256(), "license": {"path": "projects/python/rembg/LICENSE.txt", "sha256": sha(REMBG_LICENSE)}, "network_downloads": False},
        "runtime": runtime,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
