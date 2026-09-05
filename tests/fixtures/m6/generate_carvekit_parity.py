"""Generate M6 Level-2 evidence from pinned CarveKit and exported ONNX.

The checkpoint files are external and ignored.  This generator intentionally
keeps every stage: decoded RGB, complete model tensor, raw ONNX output, hard or
soft restored alpha, and straight-alpha cutout.  It covers three geometries
and two disjoint encoded colour ranges.
"""
from pathlib import Path
import json
import os
import sys
import hashlib
import subprocess
import numpy as np
import PIL
from PIL import Image
import torch
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "projects/python/image-background-remove-tool"
sys.path.insert(0, str(SRC))
# CarveKit's package initializer creates ~/.cache/carvekit even when all
# checkpoints are supplied explicitly. Keep that cache in the temporary,
# task-scoped workspace rather than mutating the user's cache.
Path.home = staticmethod(lambda: Path("/private/tmp/m6-carvekit-home"))
OUT = ROOT / "tests/fixtures/m6"
MODELS = ROOT / "projects/python/image-background-remove-tool/m6-onnx"
CHECKPOINT_ROOT = Path(os.environ.get("M6_CHECKPOINT_ROOT", "/private/tmp/m6-checkpoints"))
SOURCE_COMMIT = "f141a311af67fb1da64269c508a6d1f786420801"
PYTHON = {"source": "CarveKit f141a311", "torch": torch.__version__, "onnxruntime": ort.__version__}
PTH_SHA512 = {
    "basnet": "e409cb709f4abca87cb11bd44a9ad3f909044a917977ab65244b4c94dd338b1a37755c4253d7cb54526b7763622a094d7b676d34b5e6886689256754e5a5e6ad",
    "deeplabv3": "9c5a1795bc8baa267200a44b49ac544a1ba2687d210f63777e4bd715387324469a59b072f8a289cc471c637b367932177e5b312e8ea6351c1763d9ff44b4857c",
    "tracer-b7": "c439c5c12d4d43d5f9be9ec61e68b2e54658a541bccac2577ef5a54fb252b6e8415d41f7ec2487033d0c02b4dd08367958e4e62091318111c519f93e2632be7b",
}
ONNX_SHA256 = {
    "basnet": "f20b2de9a108b92574cd6902ecade1721ee0607be7b46683794930392f49d838",
    "deeplabv3": "4683b7f27832c0d5576cc7b81c8522e029549002861267077df0f19e16a62cb3",
    "tracer-b7": "cab33cedb809050612a107a5a6b929dabeb04445a95b988be3f3203b0662fbad",
}
PTH_PATHS = {
    "basnet": CHECKPOINT_ROOT / "basnet.pth",
    "deeplabv3": CHECKPOINT_ROOT / "deeplab.pth",
    "tracer-b7": CHECKPOINT_ROOT / "tracer_b7.pth",
}

def digest(path):
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def verify_provenance():
    head = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if head != SOURCE_COMMIT:
        raise RuntimeError(f"CarveKit HEAD mismatch: {head} != {SOURCE_COMMIT}")
    if subprocess.run(["git", "-C", str(SRC), "diff", "--quiet"], check=False).returncode != 0:
        raise RuntimeError("CarveKit tracked worktree differs from pinned HEAD")
    if subprocess.run(["git", "-C", str(SRC), "diff", "--cached", "--quiet"], check=False).returncode != 0:
        raise RuntimeError("CarveKit index differs from pinned HEAD")
    source_files = ["carvekit/ml/wrap/basnet.py", "carvekit/ml/wrap/deeplab_v3.py", "carvekit/ml/wrap/tracer_b7.py"]
    for relative in source_files:
        if not (SRC / relative).is_file():
            raise RuntimeError(f"missing pinned CarveKit source file: {relative}")
    versions = {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__, "Pillow": PIL.__version__, "numpy": np.__version__, "onnxruntime": ort.__version__}
    expected_versions = {"torch": "2.6.0", "torchvision": "0.21.0", "Pillow": "12.3.0", "numpy": "2.5.2", "onnxruntime": "1.23.2"}
    if versions != expected_versions:
        raise RuntimeError(f"dependency version mismatch: {versions} != {expected_versions}")
    wrapper_hashes = {relative: digest(SRC / relative) for relative in source_files}
    provenance = {"carvekit_commit": head, "tracked_source_clean": True, "source_files": source_files, "source_file_sha256": wrapper_hashes, "dependencies": versions, "checkpoints": {}, "onnx": {}}
    for family, path in PTH_PATHS.items():
        if not path.is_file():
            raise RuntimeError(f"missing pinned checkpoint: {path}")
        actual = hashlib.sha512(path.read_bytes()).hexdigest()
        if actual != PTH_SHA512[family]:
            raise RuntimeError(f"{family} PTH SHA-512 mismatch: {actual}")
        provenance["checkpoints"][family] = {"path": f"checkpoints/{path.name}", "sha512": actual}
        onnx_name = {"basnet": "basnet.onnx", "deeplabv3": "deeplab.onnx", "tracer-b7": "tracer_b7.onnx"}[family]
        onnx = MODELS / onnx_name
        actual_onnx = digest(onnx)
        if actual_onnx != ONNX_SHA256[family]:
            raise RuntimeError(f"{family} ONNX SHA-256 mismatch: {actual_onnx}")
        provenance["onnx"][family] = {"path": f"projects/python/image-background-remove-tool/m6-onnx/{onnx_name}", "sha256": actual_onnx}
    return provenance

def f32(path, values):
    np.asarray(values, dtype="<f4").tofile(path)

def build_image(width, height, low):
    lo, hi = ((0, 64) if low else (192, 255))
    y, x = np.indices((height, width))
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[..., 0] = lo + ((x * 17 + y * 3) % (hi - lo + 1))
    arr[..., 1] = lo + ((x * 5 + y * 19 + 7) % (hi - lo + 1))
    arr[..., 2] = lo + ((x * 11 + y * 13 + 23) % (hi - lo + 1))
    return Image.fromarray(arr, "RGB")

def save_case(family, name, image, tensor, raw, alpha):
    case = OUT / "reference" / family / name
    case.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=np.uint8)
    f32(case / "decoded-rgb.f32le", arr.astype(np.float32).reshape(-1) / 255.0)
    f32(case / "preprocessed-tensor.f32le", tensor)
    f32(case / "raw-output.f32le", raw)
    f32(case / "restored-alpha.f32le", alpha)
    rgba = np.concatenate([arr, np.rint(np.clip(alpha, 0, 1)[..., None] * 255).astype(np.uint8)], axis=2)
    (case / "final-straight-alpha-cutout.rgba").write_bytes(rgba.tobytes())
    Image.fromarray(rgba, "RGBA").save(case / "final-straight-alpha-cutout.png")
    return {p.name: digest(p) for p in case.iterdir() if p.is_file()}

def main():
    # Verify the pinned tracked source before importing any CarveKit wrapper.
    # The external m6-onnx export directory is intentionally untracked, but a
    # tracked wrapper edit must never be hidden by the imported module cache.
    provenance = verify_provenance()
    from carvekit.ml.wrap.basnet import BASNET
    from carvekit.ml.wrap.deeplab_v3 import DeepLabV3
    from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
    from carvekit.ml.arch.basnet.basnet import BASNet
    from carvekit.ml.arch.tracerb7.tracer import TracerDecoder
    from carvekit.ml.arch.tracerb7.efficientnet import EfficientEncoderB7
    from torchvision import transforms

    bas = BASNET(load_pretrained=False); bas.load_state_dict(torch.load(PTH_PATHS["basnet"], map_location='cpu', weights_only=True)); bas.eval()
    deep = DeepLabV3(load_pretrained=False); deep.network.load_state_dict(torch.load(PTH_PATHS["deeplabv3"], map_location='cpu', weights_only=True)); deep.network.eval()
    tracer = TracerDecoder(encoder=EfficientEncoderB7(), rfb_channel=[32,64,128], features_channels=[48,80,224,640]); tracer.load_state_dict(torch.load(PTH_PATHS["tracer-b7"], map_location='cpu', weights_only=True), strict=False); tracer.eval()
    tracer_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640)), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    sessions = {
        "basnet": ort.InferenceSession(str(MODELS / "basnet.onnx"), providers=["CPUExecutionProvider"]),
        "deeplabv3": ort.InferenceSession(str(MODELS / "deeplab.onnx"), providers=["CPUExecutionProvider"]),
        "tracer-b7": ort.InferenceSession(str(MODELS / "tracer_b7.onnx"), providers=["CPUExecutionProvider"]),
    }
    onnx_max_tolerance = 1e-3
    onnx_mean_tolerance = 2e-6
    records = []
    for width, height in ((3, 2), (2, 3), (1025, 3)):
        for low in (True, False):
            name = f"{width}x{height}-{'low' if low else 'high'}"
            image = build_image(width, height, low)
            with torch.no_grad():
                b = bas.data_preprocessing(image); braw = torch.nn.Module.__call__(bas, b)[0].numpy()[0]
                d = deep.data_preprocessing(image); draw = deep.network(d.unsqueeze(0))["out"].numpy()[0]
                t = tracer_transform(image).unsqueeze(0).float(); traw = tracer(t).numpy()[0]
            outputs = {"basnet": (b, braw), "deeplabv3": (d.unsqueeze(0), draw), "tracer-b7": (t, traw)}
            for family, (tensor, raw) in outputs.items():
                if family == "basnet":
                    restored = np.asarray(BASNET.data_postprocessing(torch.from_numpy(raw), image), dtype=np.float32) / 255.0
                elif family == "tracer-b7":
                    restored = np.asarray(TracerUniversalB7.data_postprocessing(torch.from_numpy(raw), image), dtype=np.float32) / 255.0
                else:
                    classes = np.argmax(draw, axis=0)
                    restored = np.asarray(Image.fromarray((classes != 0).astype(np.uint8) * 255, "L").resize(image.size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0
                rust_raw = sessions[family].run(None, {"input": tensor.numpy().astype(np.float32)})[0]
                err = np.abs(rust_raw.astype(np.float32) - raw.astype(np.float32))
                artifacts = save_case(family, name, image, tensor.numpy(), raw, restored)
                records.append({"family": family, "case": name, "geometry": [width, height], "colour_range": "low" if low else "high", "input_tensor_shape": list(tensor.shape), "raw_shape": list(raw.shape), "onnx_raw_max_abs": float(err.max()), "onnx_raw_mean_abs": float(err.mean()), "artifacts": artifacts, "verdict": bool(err.max() <= onnx_max_tolerance and err.mean() <= onnx_mean_tolerance)})
    tolerances = {
        "onnx_raw_max_abs": {"value": onnx_max_tolerance, "units": "float32 alpha/logit units", "justification": "Pinned PyTorch and ONNX Runtime CPU execution must agree within one-thousandth."},
        "onnx_raw_mean_abs": {"value": onnx_mean_tolerance, "units": "float32 alpha/logit units", "justification": "Mean error gate catches broad drift while allowing CPU kernel rounding."},
        "rust_tensor_max_abs": {"basnet": {"value": 1e-4, "units": "normalized ImageNet tensor units"}, "deeplabv3": {"value": 1e-4, "units": "normalized ImageNet tensor units"}, "tracer-b7": {"value": 1e-5, "units": "normalized ImageNet tensor units (0.00057 input-code values at std=0.224)", "justification": "TRACER follows torchvision tensor bilinear half-pixel Resize with antialiasing; the observed 1.67e-6 maximum is sub-one-code and this tight bound guards resize semantics rather than hiding a mismatch."}},
        "rust_tensor_mean_abs": {"value": 1e-3, "units": "normalized ImageNet tensor units", "justification": "Mean bound is applied in addition to the family max bound."},
        "rust_raw_max_abs": {"value": 1e-3, "units": "float32 alpha/logit units"},
        "rust_raw_mean_abs": {"value": 1e-4, "units": "float32 alpha/logit units"},
        "rust_restored_max_abs": {"value": 1.0 / 255.0 + 1e-6, "units": "alpha code values in [0,1]"},
        "rust_restored_mean_abs": {"value": 1.0 / 255.0 + 1e-6, "units": "alpha code values in [0,1]"},
        "final_cutout": {"mode": "rgba-byte-tolerance", "max_abs": 1, "mean_abs": 0.1, "units": "RGBA uint8 code values", "justification": "RGB remains exact; Pillow-version rounding can move restored alpha by at most one encoded code."},
    }
    report = {"schema": "m6.carvekit-python-level2.v1", "provenance": provenance, "python": PYTHON, "source": {"repository": "image-background-remove-tool", "commit": SOURCE_COMMIT}, "coverage": {"geometries": [[3,2],[2,3],[1025,3]], "colour_ranges": ["low", "high"]}, "tolerances": tolerances, "records": records, "verdict": all(r["verdict"] for r in records)}
    (OUT / "python-onnx-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verdict": report["verdict"], "records": len(records)}))
    if not report["verdict"]:
        raise SystemExit("M6 parity verdict failed closed")

if __name__ == "__main__": main()
