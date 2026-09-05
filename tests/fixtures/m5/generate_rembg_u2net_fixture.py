"""Generate M5 level-2 rembg preprocessing fixtures with pinned Pillow.

This intentionally performs only the authoritative Python preprocessing from
projects/python/rembg/rembg/sessions/base.py. ONNX inference is performed by
the Rust m5-smoke command against the same pinned checkpoint; Python ORT is an
optional external gate because no pinned wheel is available for this host.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import PIL

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("M5_FIXTURE_DIR", str(Path(__file__).parent)))
RUST_ROOT = Path(os.environ["M5_RUST_ROOT"]) if os.environ.get("M5_RUST_ROOT") else None
OUT.mkdir(parents=True, exist_ok=True)
if PIL.__version__ != "10.4.0":
    raise RuntimeError(f"M5 fixture generation requires Pillow 10.4.0, got {PIL.__version__}")

FIXTURES = {
    "landscape-3x2": (3, 2, 11),
    "portrait-2x3": (2, 3, 29),
    "odd-5x3": (5, 3, 47),
}

def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def make(width: int, height: int, seed: int) -> np.ndarray:
    out = np.empty((height, width, 3), dtype=np.uint8)
    for i in range(width * height):
        out.reshape(-1, 3)[i] = ((i * 53 + seed) % 256, (i * 97 + 64 + seed) % 256, (255 - i * 29 + seed + 256) % 256)
    return out

records = []
for name, (width, height, seed) in FIXTURES.items():
    rgb = make(width, height, seed)
    Image.fromarray(rgb, "RGB").save(OUT / f"{name}.png")
    resized = np.asarray(Image.fromarray(rgb, "RGB").resize((320, 320), Image.Resampling.LANCZOS), dtype=np.float64)
    normalized = resized / max(float(np.max(resized)), 1e-6)
    tensor = np.stack([(normalized[:, :, c] - mean) / std for c, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225)))], axis=0).astype(np.float32)
    tensor_bytes = tensor.tobytes(order="C")
    rgb_f32 = (rgb.astype(np.float32) / 255.0).tobytes(order="C")
    (OUT / f"{name}.tensor.f32le").write_bytes(tensor_bytes)
    (OUT / f"{name}.decoded-rgb.f32le").write_bytes(rgb_f32)
    record = {"id": name, "dimensions": [width, height], "decoded_rgb_sha256": hash_bytes(rgb_f32), "tensor_sha256": hash_bytes(tensor_bytes), "tensor_shape": [1, 3, 320, 320], "resize": "Pillow-10.4.0-LANCZOS", "source": "rembg/BaseSession.normalize@030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"}
    if RUST_ROOT:
        rust = RUST_ROOT / name
        rust_tensor = np.fromfile(rust / "preprocessed-tensor.f32le", dtype="<f4")
        difference = np.abs(tensor.reshape(-1) - rust_tensor)
        record["rust_tensor_sha256"] = hash_bytes((rust / "preprocessed-tensor.f32le").read_bytes())
        max_abs = float(difference.max())
        mean_abs = float(difference.mean())
        # Rust performs the same uint8 Pillow path but normalizes with f32
        # arithmetic, so the accepted bound is one f32 ulp at this scale.
        if max_abs > 1e-6 or mean_abs > 1e-7:
            raise AssertionError(f"{name}: Pillow/Rust tensor parity failed: max={max_abs} mean={mean_abs}")
        record["tensor_parity"] = {"max_abs": max_abs, "mean_abs": mean_abs, "max_abs_tolerance": 1e-6, "mean_abs_tolerance": 1e-7, "status": "pass"}
        record["rust_raw_output_sha256"] = hash_bytes((rust / "raw-output.f32le").read_bytes())
        record["rust_restored_alpha_sha256"] = hash_bytes((rust / "restored-alpha.f32le").read_bytes())
        record["rust_cutout_sha256"] = hash_bytes((rust / "final-straight-alpha-cutout.rgba").read_bytes())
    records.append(record)

result = {"schema": "m5.rembg-python-preprocessing.v1", "records": records}
if RUST_ROOT:
    (OUT / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
