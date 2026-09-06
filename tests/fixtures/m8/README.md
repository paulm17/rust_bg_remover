# M8 BRIA RMBG profile fixtures

`generate_synthetic_onnx.py` writes the pinned 184-byte Identity graph; it
must be run before either authoritative source. `generate_rmbg_fixture.py`
then assembles deterministic contract fixtures from the two source runs. The checked-in `reference/`
stages were regenerated from the authoritative source runs recorded under
`authoritative/`, using permissively licensed synthetic Identity ONNX graphs;
no BRIA weights are bundled. Each profile stores decoded RGB, the complete
preprocessed tensor, complete raw ONNX output, restored alpha and final
straight-alpha cutout.

The pinned Python `BriaRmBgSession` has additionally been executed once using
the permissively licensed `models/fixtures/m3_identity.onnx` graph (no BRIA
weights or downloads). Complete source-produced stages are in
temporary authoritative output directories, with provenance reports retained
under `authoritative/`; canonical stage binaries are retained only under
`reference/`.
The Rust adapter has a separate ignored synthetic-ORT gate. The Rust source
run used a temporary instrumentation copy only; the original checkout was
clean and remains untouched. BRIA 1.4/2.0 checkpoint tournament rows remain
excluded because their weights are absent and unapproved.

The rust/rmbg source is pinned to `8ce479cac1f2940502da1a55e19d19183f4862f7`;
the rembg source is pinned to `030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709`.
Their source-code licences and the BRIA checkpoint licence are recorded
separately in the model manifests.

End-to-end regeneration (using the pinned Python 3.12.11 environment with
NumPy 2.3.2, Pillow 12.2.0 and ONNX Runtime 1.23.2, and the exact external
ORT dylib SHA recorded in the Rust report) is:

```text
python3 tests/fixtures/m8/generate_synthetic_onnx.py
ORT_DYLIB=/path/to/libonnxruntime.1.23.2.dylib tests/fixtures/m8/run_authoritative_rust.sh /tmp/m8-authoritative-rust
cp /tmp/m8-authoritative-rust/report.json tests/fixtures/m8/authoritative/rust-rmbg/report.json
M8_PYTHON=/path/to/pinned/python3.12
"$M8_PYTHON" tests/fixtures/m8/run_authoritative_rembg.py --output /tmp/m8-authoritative-python --source-root projects/python/rembg/rembg --model models/fixtures/m3_identity.onnx
cp /tmp/m8-authoritative-python/report.json tests/fixtures/m8/authoritative/report.json
python3 tests/fixtures/m8/generate_rmbg_fixture.py --output /tmp/m8-reference --rust-source-output /tmp/m8-authoritative-rust --python-source-output /tmp/m8-authoritative-python
cargo run -p bgremove-bench --offline --locked -- m8-smoke --output /tmp/m8-smoke
```

Run assembly twice and compare the actual output trees to verify reproducibility:

```text
python3 tests/fixtures/m8/generate_rmbg_fixture.py --output /tmp/m8-a --rust-source-output /tmp/m8-authoritative-rust --python-source-output /tmp/m8-authoritative-python
python3 tests/fixtures/m8/generate_rmbg_fixture.py --output /tmp/m8-b --rust-source-output /tmp/m8-authoritative-rust --python-source-output /tmp/m8-authoritative-python
diff -r /tmp/m8-a /tmp/m8-b
```
