#!/usr/bin/env python3
"""Validate and assemble one deterministic M11 authority run."""
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.source / "report.json").read_text())
    report_bytes = (args.source / "report.json").read_bytes()
    parity = json.loads((args.source / "parity.json").read_text())
    if report.get("authoritative_sources_executed") is not True:
        raise SystemExit("authority source execution is not true")
    if report.get("real_checkpoint_executed") is not False:
        raise SystemExit("real checkpoint status is not explicitly unavailable")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SystemExit("authority has no artifacts")
    for metadata in artifacts.values():
        relative = Path(metadata["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("authority artifact path escapes output")
        path = args.source / relative
        if path.stat().st_size != metadata["bytes"] or digest(path) != metadata["sha256"]:
            raise SystemExit(f"authority artifact hash mismatch: {relative}")
    if parity.get("authoritative_report") != {
        "path": "report.json",
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
    }:
        raise SystemExit("parity authoritative-report hash is stale")
    parity_artifacts = parity.get("artifacts")
    if parity_artifacts != artifacts:
        raise SystemExit("parity and authority artifact metadata differ")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in [*artifacts, "report.json", "parity.json"]:
        shutil.copyfile(args.source / name, args.output / name)
    shutil.copyfile(args.source / "report.json", args.output.parent / "authoritative-report.json")


if __name__ == "__main__":
    main()
