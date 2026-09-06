#!/usr/bin/env python3
"""Assemble M8 references from two authoritative source runs."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

STAGES = ("decoded-rgb.f32le", "preprocessed-tensor.f32le", "raw-onnx-output.f32le", "restored-alpha.f32le", "final-straight-alpha-cutout.rgba")
COMMITS = {"rmbg-rust": "8ce479cac1f2940502da1a55e19d19183f4862f7", "rembg-bria": "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709"}
SOURCE_TREES = {"rmbg-rust": "f9fc3538d1e167bc30268dae85d664fa59a97897eab65024fcb04d5eca248417", "rembg-bria": "a9c2584b47370c5f7f71e0049c9130a311b028150e444ed979ffafbccdd6b058"}
LICENSES = {
    "rmbg-rust": ("models/M8_SYNTHETIC_ONNX_LICENSE.txt", "11762333d44173f00c5bbe7e7e805105f1d75ab38c93b079807e33d23136d8a6"),
    "rembg-bria": ("models/M3_FIXTURE_LICENSE.txt", "cfed44a701bec837a8ae43d9e6baf69fa5b7fd88aeed383c5ad630b8f430b610"),
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def profile(root: Path, name: str, report: dict) -> dict:
    if report.get("authoritative_execution") is not True or report.get("profile") != name:
        raise SystemExit(f"{name}: authoritative report is invalid")
    if report["source"]["commit"] != COMMITS[name]:
        raise SystemExit(f"{name}: source commit mismatch")
    if report["source"].get("source_tree_sha256") != SOURCE_TREES[name]:
        raise SystemExit(f"{name}: source tree hash mismatch")
    license_path, license_hash = LICENSES[name]
    if report["model"]["license_path"] != license_path or report["model"]["license_sha256"] != license_hash:
        raise SystemExit(f"{name}: synthetic license provenance mismatch")
    source = root if name == "rmbg-rust" else root / name
    stages = {}
    for stage in STAGES:
        path = source / stage
        if not path.is_file() or sha256(path) != report["stages"].get(stage):
            raise SystemExit(f"{name}: missing or tampered {stage}")
        stages[stage] = sha256(path)
    return {"profile": name, "input_dimensions": report["input_dimensions"], "model_output_shape": [1, 3, 1024, 1024], "stages": stages}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rust-source-output", type=Path, required=True)
    parser.add_argument("--python-source-output", type=Path, required=True)
    args = parser.parse_args()
    rr = args.rust_source_output.resolve(); pr = args.python_source_output.resolve()
    rust_report = json.loads((rr / "report.json").read_text())
    python_report = json.loads((pr / "report.json").read_text())
    profiles = [profile(rr, "rmbg-rust", rust_report), profile(pr, "rembg-bria", python_report)]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    for item in profiles:
        src = rr if item["profile"] == "rmbg-rust" else pr / item["profile"]
        dst = out / item["profile"]; dst.mkdir(parents=True, exist_ok=True)
        for stage in STAGES: shutil.copyfile(src / stage, dst / stage)
    parity = {"schema": "m8.rmbg-profile-fixture.v2", "authoritative_sources_executed": True,
      "weights_status": "excluded-by-license-and-not-present", "source": {
        "rust_rmbg_commit": rust_report["source"]["commit"], "rembg_commit": python_report["source"]["commit"],
        "rust_instrumentation_patch_sha256": rust_report["source"].get("instrumentation_patch_sha256"),
        "rust_tracked_source_clean": rust_report["source"].get("tracked_source_clean"),
        "authoritative_reports": ["tests/fixtures/m8/authoritative/rust-rmbg/report.json", "tests/fixtures/m8/authoritative/report.json"]},
      "tolerances": {"decoded_rgb": 0.0, "preprocessed_tensor": 0.00002, "raw_output": 0.0, "restored_alpha": 0.0, "final_cutout": 0.0}, "profiles": profiles}
    (out / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")

if __name__ == "__main__": main()
