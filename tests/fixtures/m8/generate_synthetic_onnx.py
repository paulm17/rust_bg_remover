#!/usr/bin/env python3
"""Write the deterministic permissively licensed M8 Identity ONNX graph."""
from __future__ import annotations
import argparse, base64, hashlib
from pathlib import Path

# Hand-encoded protobuf for an opset-13 Identity graph with dynamic NCHW I/O.
GRAPH = base64.b64decode("CAgSDG04LXN5bnRoZXRpYzqfAQoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRIRbThfcm1iZ19zeW50aGV0aWNaNgoFaW5wdXQSLQorCAESJwoHEgViYXRjaAoJEgdjaGFubmVsCggSBmhlaWdodAoHEgV3aWR0aGI3CgZvdXRwdXQSLQorCAESJwoHEgViYXRjaAoJEgdjaGFubmVsCggSBmhlaWdodAoHEgV3aWR0aEIECgAQDQ==")
EXPECTED = "270f3af536551a7ca1a4834b987b3da9c0a5c8f55ccd30cf89a1a3eeeadd18b3"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/fixtures/m8_rmbg_identity_output.onnx"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(GRAPH)
    actual = hashlib.sha256(GRAPH).hexdigest()
    if actual != EXPECTED:
        raise SystemExit(f"synthetic graph hash mismatch: {actual}")
    print(f"{args.output}: {actual}")

if __name__ == "__main__":
    main()
