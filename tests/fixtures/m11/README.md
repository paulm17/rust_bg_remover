# M11 FBA parity fixture

This fixture executes the pinned CarveKit FBA preprocessing, transformed
trimap, fusion, and alpha-only postprocessing code against a small
permissively licensed synthetic ONNX graph. The graph takes four inputs and
returns final fused `[alpha, foreground BGR, background BGR]` seven-channel
output; Rust does not fuse graph output a second time. The fixture uses a
13x9 canonical image, configured 11x7 resize, and padded 16x8 working grid,
exercising Pillow bicubic and OpenCV Lanczos4 separately. No checkpoint is
downloaded or included. The real CarveKit checkpoint remains an external, unapproved
candidate and is reported as unavailable; therefore the 1024/2048 real-model
gates are not claimed as executed.

The authority interpreter is Python 3.12.11 from the provisioned environment
with torch 2.9.1, NumPy 2.3.2, OpenCV 4.10.0, Pillow, and ONNX Runtime 1.23.2.
Set `M11_PYTHON` to that pinned interpreter and keep the CarveKit checkout at
commit `f141a311af67fb1da64269c508a6d1f786420801`:

```sh
export M11_PYTHON=/path/to/pinned/python3.12
export NUMBA_CACHE_DIR=/tmp/m11-numba
PYTHONPATH=projects/python/image-background-remove-tool \
  "$M11_PYTHON" tests/fixtures/m11/generate_fba_fixture.py \
  --output /tmp/m11-authority
```

The command is offline and fail-loud on source drift. Assemble the tracked
reference with the validating deterministic assembler:

```sh
M11_PYTHON=... # as above
"$M11_PYTHON" tests/fixtures/m11/assemble_fba_fixture.py \
  --source /tmp/m11-authority \
  --output tests/fixtures/m11/reference
```

Re-running the authority in two clean temporary directories and assembling
both outputs must be byte-identical.

The Rust parity smoke runs the actual preprocessing and fusion helpers against
canonical fixture inputs and emits deterministic artifacts:

```sh
cargo run -p bgremove-bench --offline --locked -- \
  m11-smoke --output runs/m11-fba
```

The ignored end-to-end ORT gate runs the checked-in synthetic graph when an
external, provisioned runtime is supplied:

```sh
ORT_DYLIB=/path/to/onnxruntime.1.23.2.dylib \
  cargo test -p bgremove-ort --offline --locked \
  real_m11_synthetic_fba_matches_all_runtime_stages -- --ignored
```

An approved external checkpoint is never fetched by this repository.  To
export one, provide the already-supplied 138,813,960-byte Carve/fba artifact;
the offline exporter checks the pinned SHA-256 and SHA-512, loads the pinned
checkout, and refuses any mismatch or absent file:

```sh
export M11_PYTHON=/path/to/pinned/python3.12
"$M11_PYTHON" tests/fixtures/m11/export_fba_onnx.py \
  --checkpoint /path/to/fba_matting.pth \
  --size 1024 \
  --output /tmp/fba_matting-final-fused.onnx \
  --report /tmp/fba_matting-final-fused.json
```

The exporter accepts `--size 1024` and `--size 2048`, producing a static graph
and matching four input shapes for that size. The export contract is four
named inputs (`image`, `trimap`, `image_normalized`, `trimap_transformed`) and
one final fused `[1,7,H,W]` output (`alpha`, foreground BGR, background BGR).
The external manifests stay
unapproved and runtime-gated until the checkpoint's license is approved and
the exported graph has been independently measured.

Before supplying a checkpoint, the export entrypoint's callable regression can
be run offline; it instantiates the pinned `FBA.forward` architecture directly
and proves the four tensor inputs without invoking the `FBAMatting.__call__`
PIL batch wrapper:

```sh
"$M11_PYTHON" tests/fixtures/m11/export_fba_onnx.py \
  --check-callable --size 1024 --output /tmp/unused.onnx
```
