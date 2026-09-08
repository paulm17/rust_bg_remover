#!/usr/bin/env python3
"""Assemble a checked-in M10 reference from a fresh authority run."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

REQUIRED = ["decoded-rgb.f32le", "coarse-alpha.f32le", "trimap.f32le", "canonical-rgb.f32le", "canonical-coarse.f32le", "canonical-trimap.f32le", "true-alpha.f32le", "true-foreground.f32le", "true-background.f32le", "analytic-image.f32le", "analytic-coarse-alpha.f32le", "analytic-trimap.f32le", "analytic-true-alpha.f32le", "analytic-true-foreground.f32le", "analytic-true-background.f32le", "constraints.json", "laplacian.json", "alpha.f32le", "foreground.f32le", "background.f32le", "restored-alpha.f32le", "restored-foreground.f32le", "restored-background.f32le", "chain-final-rgba.png", "final-rgba.png"]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.source / "report.json").read_text())
    if report.get("schema") != "bgremove.m10.authoritative.v1" or report.get("authoritative_sources_executed") is not True:
        raise SystemExit("source is not a complete authoritative M10 report")
    if report["source"].get("network_downloads") is not False:
        raise SystemExit("M10 authority must be offline")
    if report.get("stage_chain") != {"connected": True, "decoded_to_working": True, "final_from_captured_stages": True, "wrapper_final_matches_chain": True}:
        raise SystemExit("M10 authority stage chain is not connected")
    for name in REQUIRED:
        path = args.source / name
        if not path.is_file():
            raise SystemExit(f"missing authoritative artifact: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        shutil.copyfile(args.source / name, args.output / name)
    shutil.copyfile(args.source / "report.json", args.output / "authoritative-report.json")
    artifacts = {name: {"path": name, "sha256": sha(args.output / name), "bytes": (args.output / name).stat().st_size} for name in REQUIRED}
    # The parity contract compares all five stages, with f32 numerical error
    # bounded tightly enough to expose ordering and indexing mistakes.
    parity = {
        "schema": "bgremove.m10.parity.v1",
        "authority_report": {"path": "authoritative-report.json", "sha256": sha(args.output / "authoritative-report.json")},
        "source_execution": True,
        "provenance": report["source"],
        "analytic_thresholds": {"true_alpha": 0.02, "visible_foreground": 0.02, "composite": 0.02},
        "profiles": [{"name": "pymatting-1.1.15-backgroundremover", "authority": "pinned PyMatting and backgroundremover", "input": "decoded-rgb.f32le", "stages": ["decoded-rgb", "coarse-alpha", "trimap", "constraints", "laplacian", "alpha", "foreground", "background", "chain-final-rgba", "wrapper-final-rgba"], "tolerances": {"decoded-rgb": 0.0, "coarse-alpha": 0.0, "trimap": 0.0, "constraints": 0.0, "laplacian": 3e-7, "alpha": 1e-5, "foreground": 1e-5, "background": 1e-5, "final-rgba": 1, "base-size-resample": 1e-7}}],
        "artifacts": artifacts,
        "earliest_divergence": "not measured until Rust smoke consumes this reference",
    }
    (args.output / "parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
    (args.output.parent / "authoritative-report.json").write_text((args.output / "authoritative-report.json").read_text())

if __name__ == "__main__":
    main()
