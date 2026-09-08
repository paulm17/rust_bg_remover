#!/usr/bin/env python3
"""Assemble a portable M9 reference tree from a fresh authority run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_output.resolve()
    output = args.output.resolve()
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if report.get("schema") != "m9.trimap-authoritative.v1" or not report.get("authoritative_sources_executed"):
        raise SystemExit("authoritative M9 report is missing or not executed")
    profiles = report.get("profiles", [])
    if {profile.get("name") for profile in profiles} != {"rembg-symmetric", "carvekit-probability-aware"}:
        raise SystemExit("M9 report does not contain both required source profiles")
    expected_commits = {
        "rembg-symmetric": "030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709",
        "carvekit-probability-aware": "f141a311af67fb1da64269c508a6d1f786420801",
    }
    for profile in profiles:
        source_info = profile.get("source", {})
        if source_info.get("commit") != expected_commits[profile["name"]]:
            raise SystemExit(f"unexpected source commit for {profile['name']}")
        if source_info.get("tracked_source_clean") is not True:
            raise SystemExit(f"source is not clean for {profile['name']}")
        if len(source_info.get("source_tree_sha256", "")) != 64:
            raise SystemExit(f"missing source tree hash for {profile['name']}")
        if len(source_info.get("license_sha256", "")) != 64:
            raise SystemExit(f"missing license hash for {profile['name']}")
        if not source_info.get("source_file_sha256"):
            raise SystemExit(f"missing source file hashes for {profile['name']}")
    output.mkdir(parents=True, exist_ok=True)
    names = [
        "input-mask.png",
        "erode-mask.png",
        "post-process-input.png",
        "rembg-symmetric-trimap.png",
        "rembg-symmetric-trimap-e10.png",
        "rembg-symmetric-trimap-e0.png",
        "rembg-post-process.png",
        "carvekit-probability-trimap.png",
        "gaussian-values.json",
        "oversized-erosion.json",
    ]
    for name in names:
        path = source / name
        if not path.is_file():
            raise SystemExit(f"missing authoritative artifact: {name}")
        shutil.copyfile(path, output / name)
    if digest(output / report["input"]["path"]) != report["input"]["sha256"]:
        raise SystemExit("input artifact hash mismatch")
    for profile in profiles:
        artifact = output / profile["output"]["path"]
        if digest(artifact) != profile["output"]["sha256"]:
            raise SystemExit(f"artifact hash mismatch: {artifact.name}")
    for case in report.get("rembg_cases", []):
        artifact = output / case["path"]
        if digest(artifact) != case["sha256"]:
            raise SystemExit(f"case artifact hash mismatch: {artifact.name}")
    if {case.get("name") for case in report.get("rembg_cases", [])} != {
        "erode_size_10",
        "erode_size_0_default_cross",
        "post_process_reflect",
    }:
        raise SystemExit("required rembg source cases are missing")
    gaussian = report.get("gaussian_reference", {})
    if digest(output / gaussian["path"]) != gaussian["sha256"]:
        raise SystemExit("Gaussian reference hash mismatch")
    oversized = report.get("oversized_erosion_reference", {})
    if digest(output / oversized["path"]) != oversized["sha256"]:
        raise SystemExit("oversized erosion reference hash mismatch")
    parity = {
        "schema": "m9.trimap-parity.v1",
        "authoritative_sources_executed": True,
        "input": report["input"],
        "profiles": profiles,
        "rembg_cases": report.get("rembg_cases", []),
        "gaussian_reference": gaussian,
        "oversized_erosion_reference": oversized,
        "semantics": {
            "binary_threshold": "strict greater-than",
            "square_morphology": "full (2r+1)^2 kernel; constant false outside",
            "disk_opening": "disk offsets dx^2+dy^2 <= r^2; erosion then dilation; SciPy/skimage reflect outside",
            "gaussian": "separable normalized Gaussian, radius floor(4*sigma + 0.5) (SciPy truncate=4), half-sample reflect border, f64 accumulation",
            "rembg_trimap": "foreground > threshold and background < threshold; literal erode_size square (including even sizes), erode_size=0 uses SciPy default connectivity-1 cross; foreground false/background true border",
            "carvekit_trimap": "prob_filter > threshold, (2r+1)^2 dilation, unknown 127, foreground 255, one 3x3 erosion iteration per configured iteration",
            "relative_radius": "round(fraction * min(width,height))",
            "oversized_literal_kernel": "constant-border anchored windows are evaluated exactly without allocating the full structure",
        },
        "artifacts": {name: {"path": name, "sha256": digest(output / name)} for name in names},
    }
    (output / "parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
