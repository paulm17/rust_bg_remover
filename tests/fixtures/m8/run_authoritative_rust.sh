#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
SOURCE="$ROOT/projects/rust/rmbg"
PATCH="$ROOT/tests/fixtures/m8/rust_authoritative_instrumentation.patch"
MODEL="$ROOT/models/fixtures/m8_rmbg_identity_output.onnx"
OUT=${1:?output directory is required; use a temporary directory}
ORT=${ORT_DYLIB:?ORT_DYLIB must point to the preseeded external ORT dylib}
test -f "$ORT"
ORT_SHA=$(shasum -a 256 "$ORT" | awk '{print $1}')
test "$ORT_SHA" = 92500010659b052368797e30c1841956b0efd699a8061ef8b3b25f27449c86e7
ORT_NAME=$(basename "$ORT")
case "$ORT_NAME" in *1.23.2*) ;; *) echo "ORT dylib filename must pin 1.23.2" >&2; exit 1 ;; esac
test "$(git -C "$SOURCE" rev-parse HEAD)" = 8ce479cac1f2940502da1a55e19d19183f4862f7
git -C "$SOURCE" diff --quiet
git -C "$SOURCE" diff --cached --quiet
test "$(shasum -a 256 "$SOURCE/src/lib.rs" | awk '{print $1}')" = f9fc3538d1e167bc30268dae85d664fa59a97897eab65024fcb04d5eca248417
test "$(shasum -a 256 "$SOURCE/Cargo.lock" | awk '{print $1}')" = e0aab181c261ca6935d283d25969d6a8f0783724472bf464a1861ebc4f5c469f
test "$(rustc --version)" = "rustc 1.94.1 (e408947bf 2026-03-25)"
test "$(cargo --version)" = "cargo 1.94.1 (29ea6fb6a 2026-03-24)"
test "$(shasum -a 256 "$MODEL" | awk '{print $1}')" = 270f3af536551a7ca1a4834b987b3da9c0a5c8f55ccd30cf89a1a3eeeadd18b3
test "$(shasum -a 256 "$ROOT/models/M8_SYNTHETIC_ONNX_LICENSE.txt" | awk '{print $1}')" = 11762333d44173f00c5bbe7e7e805105f1d75ab38c93b079807e33d23136d8a6
TMP=$(mktemp -d "${TMPDIR:-/tmp}/m8-rmbg.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
cp -R "$SOURCE" "$TMP/source"
git -C "$TMP/source" apply --unidiff-zero --check "$PATCH"
git -C "$TMP/source" apply --unidiff-zero "$PATCH"
mkdir -p "$OUT"
(cd "$TMP/source" && ORT_DYLIB="$ORT" cargo run --offline --locked --bin m8_capture -- "$MODEL" "$OUT")
M8_ORT_SHA="$ORT_SHA" M8_ORT_NAME="$ORT_NAME" python3 - "$OUT" "$MODEL" "$PATCH" <<'PY'
import hashlib, json, pathlib, subprocess, sys
import os
out, model, patch = map(pathlib.Path, sys.argv[1:])
stages = ["decoded-rgb.f32le", "preprocessed-tensor.f32le", "raw-onnx-output.f32le", "restored-alpha.f32le", "final-straight-alpha-cutout.rgba"]
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
report = {"schema":"m8.rmbg-authoritative-profile.v2", "authoritative_execution":True, "profile":"rmbg-rust", "source":{"repository":"projects/rust/rmbg", "commit":"8ce479cac1f2940502da1a55e19d19183f4862f7", "source_file":"projects/rust/rmbg/src/lib.rs", "source_file_sha256":"f9fc3538d1e167bc30268dae85d664fa59a97897eab65024fcb04d5eca248417", "source_tree_sha256":"f9fc3538d1e167bc30268dae85d664fa59a97897eab65024fcb04d5eca248417", "tracked_source_clean":True, "instrumentation_patch_sha256":h(patch), "instrumentation":"checked-in patch applied to temporary clean copy"}, "model":{"path":"models/fixtures/m8_rmbg_identity_output.onnx", "sha256":h(model), "license":"MIT OR Apache-2.0; locally generated Identity graph", "license_path":"models/M8_SYNTHETIC_ONNX_LICENSE.txt", "license_sha256":"11762333d44173f00c5bbe7e7e805105f1d75ab38c93b079807e33d23136d8a6"}, "runtime":{"rustc":subprocess.check_output(["rustc","--version"], text=True).strip(), "cargo":subprocess.check_output(["cargo","--version"], text=True).strip(), "cargo_lock_sha256":"e0aab181c261ca6935d283d25969d6a8f0783724472bf464a1861ebc4f5c469f", "onnxruntime":{"dylib":os.environ["M8_ORT_NAME"], "sha256":os.environ["M8_ORT_SHA"], "version":"1.23.2"}, "ort_crate":"2.0.0-rc.1", "fast_image_resize":"3.0.4", "public_result_equal_to_instrumented_result":True, "raw_shape":[1,3,1024,1024]}, "input_dimensions":[3,2], "model_output_dimensions":[1024,1024], "stages":{s:h(out/s) for s in stages}}
(out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
PY
