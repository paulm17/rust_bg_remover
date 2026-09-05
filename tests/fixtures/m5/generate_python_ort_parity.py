"""Generate authoritative Python rembg/U2-Net level-2 artifacts.

This is development-only: it requires an explicitly provisioned Python 3.12
environment with numpy, Pillow and onnxruntime. No model is downloaded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("M5_PYTHON_ORT_OUT", "/private/tmp/m5-python-ort"))
INPUTS = {
    "landscape-3x2": ROOT / "tests/fixtures/m5/landscape-3x2.png",
    "portrait-2x3": ROOT / "tests/fixtures/m5/portrait-2x3.png",
    "odd-5x3": ROOT / "tests/fixtures/m5/odd-5x3.png",
}
MODELS = {
    "general": ROOT / "projects/python/rembg/u2net.onnx",
    "light": ROOT / "projects/python/rembg/u2netp.onnx",
    "human": ROOT / "projects/python/rembg/u2net_human_seg.onnx",
    "silueta": ROOT / "projects/python/rembg/silueta.onnx",
    "cloth": ROOT / "projects/python/rembg/u2net_cloth_seg.onnx",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32_bytes(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype="<f4", order="C").tobytes()


def rembg_input(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    rgb = np.asarray(resized)
    normalized = rgb / max(float(np.max(rgb)), 1e-6)
    tmp = np.zeros((size[1], size[0], 3), dtype=np.float64)
    for channel, (mean, std) in enumerate(
        zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ):
        tmp[:, :, channel] = (normalized[:, :, channel] - mean) / std
    return np.expand_dims(tmp.transpose((2, 0, 1)), 0).astype(np.float32)


def general_artifacts(session: ort.InferenceSession, image: Image.Image, out: Path):
    tensor = rembg_input(image, (320, 320))
    raw = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0])
    raw_first = raw[:, 0, :, :] if raw.ndim == 4 else raw
    lo, hi = float(np.min(raw_first)), float(np.max(raw_first))
    safe = np.zeros_like(raw_first, dtype=np.float32) if hi == lo else np.clip((raw_first - lo) / (hi - lo), 0, 1).astype(np.float32)
    mask_u8 = (safe.squeeze(0) * 255).astype(np.uint8)
    restored_u8 = np.asarray(Image.fromarray(mask_u8, "L").resize(image.size, Image.Resampling.LANCZOS), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    cutout = np.dstack((rgb, restored_u8))
    for name, data in {
        "decoded-rgb.f32le": np.asarray(rgb, dtype=np.float32).ravel() / 255,
        "preprocessed-tensor.f32le": tensor,
        "raw-output.f32le": raw_first,
        "restored-alpha.f32le": restored_u8.astype(np.float32) / 255,
        "final-straight-alpha-cutout.rgba": cutout,
    }.items():
        (out / name).write_bytes(data if isinstance(data, bytes) else f32_bytes(data))
    (out / "final-straight-alpha-cutout.rgba").write_bytes(cutout.tobytes())
    return {"raw_min": lo, "raw_max": hi, "raw_shape": list(raw_first.shape)}


def cloth_artifacts(session: ort.InferenceSession, image: Image.Image, out: Path):
    tensor = rembg_input(image, (768, 768))
    raw = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0])
    classes = np.argmax(raw[0], axis=0).astype(np.uint8)
    restored = np.asarray(Image.fromarray(classes, "L").resize(image.size, Image.Resampling.LANCZOS), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    (out / "decoded-rgb.f32le").write_bytes(f32_bytes(rgb.astype(np.float32).ravel() / 255))
    (out / "preprocessed-tensor.f32le").write_bytes(f32_bytes(tensor))
    (out / "raw-output.f32le").write_bytes(f32_bytes(raw))
    (out / "restored-class-map.u8").write_bytes(restored.tobytes())
    for category, class_id in (("upper", 1), ("lower", 2), ("full", 3)):
        # rembg's binary mask convention is an 8-bit image domain: 0 or 255.
        (out / f"{category}-mask.u8").write_bytes(((restored == class_id).astype(np.uint8) * 255).tobytes())
    cutout = np.dstack((rgb, (restored == 3).astype(np.uint8) * 255))
    (out / "final-straight-alpha-cutout.rgba").write_bytes(cutout.tobytes())
    return {"raw_shape": list(raw.shape), "classes": [int(x) for x in np.unique(restored)]}


def main() -> None:
    if __import__("PIL").__version__ != "10.4.0":
        raise RuntimeError(f"M5 level-2 parity requires Pillow 10.4.0, got {__import__('PIL').__version__}")
    if ort.__version__ != "1.23.2":
        raise RuntimeError(f"M5 level-2 parity requires ONNX Runtime 1.23.2, got {ort.__version__}")
    if np.__version__ != "1.26.4":
        raise RuntimeError(f"M5 level-2 parity requires NumPy 1.26.4, got {np.__version__}")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "m5.python-ort-level2.v1", "python": {"numpy": np.__version__, "pillow": __import__("PIL").__version__, "onnxruntime": ort.__version__, "providers": ["CPUExecutionProvider"]}, "source": "rembg/BaseSession.normalize and session.predict@030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709", "tolerances": {"preprocessed_tensor_max_abs": 1e-6, "preprocessed_tensor_mean_abs": 1e-7, "single_mask_raw_output_max_abs": 1e-5, "cloth_raw_logits_max_abs": 1e-4, "cloth_raw_logits_mean_abs": 3e-6, "raw_output_mean_abs": 1e-6, "restored_alpha_u8_max_abs": 0, "cutout_byte_mismatches": 0}, "models": {}}
    domains = MODELS
    if os.environ.get("M5_ONLY_DOMAIN"):
        domains = {os.environ["M5_ONLY_DOMAIN"]: MODELS[os.environ["M5_ONLY_DOMAIN"]]}
    inputs = INPUTS
    if os.environ.get("M5_ONLY_INPUT"):
        inputs = {os.environ["M5_ONLY_INPUT"]: INPUTS[os.environ["M5_ONLY_INPUT"]]}
    for domain, model in domains.items():
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        report["models"][domain] = {"path": str(model.relative_to(ROOT)), "sha256": digest(model.read_bytes()), "provider": session.get_providers(), "inputs": [x.name for x in session.get_inputs()], "outputs": [x.name for x in session.get_outputs()], "records": {}}
        for name, path in inputs.items():
            target = OUT / domain / name
            target.mkdir(parents=True, exist_ok=True)
            image = Image.open(path).convert("RGB")
            result = cloth_artifacts(session, image, target) if domain == "cloth" else general_artifacts(session, image, target)
            report["models"][domain]["records"][name] = result
            rust_root = os.environ.get("M5_RUST_ROOT")
            rust_domain = os.environ.get("M5_RUST_DOMAIN", "light")
            if rust_root and domain == rust_domain:
                rust = Path(rust_root) / name
                python = OUT / domain / name
                metrics = {}
                for artifact in ("preprocessed-tensor.f32le", "raw-output.f32le", "restored-alpha.f32le"):
                    left = np.fromfile(python / artifact, dtype="<f4")
                    right = np.fromfile(rust / artifact, dtype="<f4")
                    if left.shape != right.shape:
                        raise RuntimeError(f"Rust/Python {domain}/{name}/{artifact} length mismatch")
                    difference = np.abs(left - right)
                    metrics[artifact] = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}
                left = (python / "final-straight-alpha-cutout.rgba").read_bytes()
                right = (rust / "final-straight-alpha-cutout.rgba").read_bytes()
                metrics["final-straight-alpha-cutout.rgba"] = {"byte_mismatch_count": sum(a != b for a, b in zip(left, right)), "byte_length_equal": len(left) == len(right)}
                metrics["verdict"] = "pass" if metrics["preprocessed-tensor.f32le"]["max_abs"] <= 1e-6 and metrics["preprocessed-tensor.f32le"]["mean_abs"] <= 1e-7 and metrics["raw-output.f32le"]["max_abs"] <= 1e-5 and metrics["raw-output.f32le"]["mean_abs"] <= 1e-6 and metrics["restored-alpha.f32le"]["max_abs"] == 0.0 and metrics["final-straight-alpha-cutout.rgba"]["byte_mismatch_count"] == 0 and metrics["final-straight-alpha-cutout.rgba"]["byte_length_equal"] else "fail"
                report["models"][domain]["records"][name]["rust_parity"] = metrics
            if rust_root and domain == "cloth":
                rust = Path(rust_root) / "cloth" / name
                python = OUT / domain / name
                metrics = {}
                for artifact in ("preprocessed-tensor.f32le", "raw-output.f32le"):
                    left = np.fromfile(python / artifact, dtype="<f4")
                    right = np.fromfile(rust / artifact, dtype="<f4")
                    if left.shape != right.shape:
                        raise RuntimeError(f"Rust/Python cloth/{name}/{artifact} length mismatch")
                    difference = np.abs(left - right)
                    metrics[artifact] = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}
                for artifact in ("restored-class-map.u8", "upper-mask.u8", "lower-mask.u8", "full-mask.u8"):
                    left = np.fromfile(python / artifact, dtype=np.uint8)
                    right = np.fromfile(rust / artifact, dtype=np.uint8)
                    difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
                    metrics[artifact] = {"max_abs": int(difference.max()), "mismatch_count": int(np.count_nonzero(difference))}
                metrics["verdict"] = "pass" if (
                    metrics["preprocessed-tensor.f32le"]["max_abs"] <= 1e-6
                    and metrics["preprocessed-tensor.f32le"]["mean_abs"] <= 1e-7
                    and metrics["raw-output.f32le"]["max_abs"] <= 1e-4
                    and metrics["raw-output.f32le"]["mean_abs"] <= 3e-6
                    and all(metrics[x]["mismatch_count"] == 0 for x in ("restored-class-map.u8", "upper-mask.u8", "lower-mask.u8", "full-mask.u8"))
                ) else "fail"
                report["models"][domain]["records"][name]["rust_parity"] = metrics
    report_bytes = json.dumps(report, indent=2) + "\n"
    (OUT / "report.json").write_text(report_bytes)
    if os.environ.get("M5_PARITY_REPORT"):
        Path(os.environ["M5_PARITY_REPORT"]).write_text(report_bytes)
    failures = [
        f"{domain}/{name}"
        for domain, domain_data in report["models"].items()
        for name, record in domain_data["records"].items()
        if "rust_parity" in record and record["rust_parity"].get("verdict") != "pass"
    ]
    if failures and os.environ.get("M5_RUST_ROOT"):
        raise SystemExit("Rust parity failed for " + ", ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
