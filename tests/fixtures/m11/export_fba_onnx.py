#!/usr/bin/env python3
"""Offline, fail-closed export of the pinned CarveKit FBA checkpoint.

This script is deliberately not a downloader.  It only accepts a caller-
supplied, already licensed checkpoint whose size and both pinned digests match;
the exported graph is the four-input final-fused seven-channel contract used
by the Rust adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

EXPECTED_BYTES = 138_813_960
EXPECTED_SHA256 = "a0eb7132f70f4d69d0bcb53763ec4e5ff1d1c65237ef9a9d69f9c94da8b85dee"
EXPECTED_SHA512 = "890906ec94c1bfd2ad08707a63e4ccb0955d7f5d25e32853950c24c784cbad2e59be277999defc3754905d0f15aa75702cdead3cfe669ff72f08811c52971613"
COMMIT = "f141a311af67fb1da64269c508a6d1f786420801"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_export_model(size: int):
    """Build the architecture whose four-input ``forward`` is exportable.

    ``FBAMatting`` is a convenience wrapper around this architecture, but it
    overrides ``__call__`` for filesystem/PIL batches.  Export must target the
    underlying pinned ``FBA`` module directly so Torch invokes its four-input
    forward contract and preserves the final fused seven-channel output.
    """
    from carvekit.ml.arch.fba_matting.models import FBA

    model = FBA(encoder="resnet50_GN_WS")
    model.eval()
    parameters = list(inspect.signature(model.forward).parameters.values())
    if [parameter.name for parameter in parameters] != [
        "image",
        "two_chan_trimap",
        "image_n",
        "trimap_transformed",
    ]:
        raise SystemExit("pinned FBA forward contract drifted from four tensor inputs")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--size", type=int, choices=(1024, 2048), required=True)
    parser.add_argument(
        "--check-callable",
        action="store_true",
        help="instantiate the export architecture and validate its four-input forward without weights",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if args.check_callable:
        sha256 = sha512 = None
    else:
        if checkpoint is None:
            raise SystemExit("--checkpoint is required unless --check-callable is used")
        if not checkpoint.is_file():
            raise SystemExit("checkpoint is absent; this offline exporter never downloads weights")
        if checkpoint.stat().st_size != EXPECTED_BYTES:
            raise SystemExit("checkpoint byte length does not match pinned Carve/fba artifact")
        sha256 = digest(checkpoint, "sha256")
        sha512 = digest(checkpoint, "sha512")
        if sha256 != EXPECTED_SHA256 or sha512 != EXPECTED_SHA512:
            raise SystemExit("checkpoint digest does not match both pinned Carve/fba digests")

    import numpy as np
    import onnxruntime as ort
    import torch

    runtime = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
    }
    expected_runtime = {
        "python": "3.12.11",
        "numpy": "2.3.2",
        "onnxruntime": "1.23.2",
        "torch": "2.9.1",
    }
    if runtime != expected_runtime:
        raise SystemExit(f"runtime drift: expected {expected_runtime}, got {runtime}")

    # Import only the pinned local checkout.  No model downloader is imported.
    root = Path(__file__).resolve().parents[3]
    carvekit = root / "projects/python/image-background-remove-tool"
    if not carvekit.is_dir():
        raise SystemExit("pinned CarveKit checkout is absent")
    if subprocess.check_output(["git", "-C", str(carvekit), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit("CarveKit checkout is not pinned to the required revision")
    if subprocess.check_output(["git", "-C", str(carvekit), "status", "--short", "--untracked-files=no"], text=True).strip():
        raise SystemExit("CarveKit tracked source is dirty")
    sys.path.insert(0, str(carvekit))
    size = args.size
    if args.check_callable:
        # This path intentionally does not inspect or load a checkpoint.  It
        # is the deterministic regression check for the exporter entrypoint.
        build_export_model(size)
        return
    model = build_export_model(size)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    shape = (1, size, size)
    image = torch.zeros((1, 3, *shape[1:]), dtype=torch.float32)
    trimap = torch.zeros((1, 2, *shape[1:]), dtype=torch.float32)
    normalized = torch.zeros_like(image)
    transformed = torch.zeros((1, 6, *shape[1:]), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (image, trimap, normalized, transformed),
        args.output,
        input_names=["image", "trimap", "image_normalized", "trimap_transformed"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    expected_inputs = {
        "image": [1, 3, size, size],
        "trimap": [1, 2, size, size],
        "image_normalized": [1, 3, size, size],
        "trimap_transformed": [1, 6, size, size],
    }
    actual_inputs = {item.name: item.shape for item in session.get_inputs()}
    if actual_inputs != expected_inputs:
        raise SystemExit(f"exported input contract mismatch: {actual_inputs}")
    outputs = session.get_outputs()
    if len(outputs) != 1 or outputs[0].name != "output" or outputs[0].shape != [1, 7, size, size]:
        raise SystemExit(f"exported graph is not final fused [1,7,{size},{size}]")
    report = {
        "source_commit": COMMIT,
        "checkpoint": {"bytes": EXPECTED_BYTES, "sha256": sha256, "sha512": sha512},
        "graph": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": digest(args.output, "sha256"),
            "inputs": expected_inputs,
            "output": [1, 7, size, size],
            "semantics": "final fused alpha, foreground BGR, background BGR",
        },
        "runtime": runtime,
        "network_downloads": False,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
