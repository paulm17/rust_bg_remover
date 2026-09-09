# M12 ViTMatte fixture and provenance

The checked-in ONNX is a deterministic, synthetic four-channel graph. It is
not a ViTMatte checkpoint and is never used to claim model quality. The Python
authority script executes the graph with the pinned ORT runtime and records the
input/output contract; the Rust `m12-smoke` command executes the same graph
when `ORT_DYLIB` is explicitly supplied.

Authority source:

- rembg commit `030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709`
- `projects/python/rembg/rembg/matting.py` SHA-256
  `e86b7e608354abd24499f64ef98edfd0e29d66b65dbc3995e46220bdcfd833e1`
- rembg license `projects/python/rembg/LICENSE.txt` SHA-256
  `90a3215072968fd304669c5389f04f1274a587abdd0507d99dead0f5511f8999`
- no network access and no checkpoint download

The four real manifests are registered with the exact upstream checkpoint
SHA-256 values and `intended_use_approved = false`; the absent files therefore
fail closed before ORT session creation. No Rust default is selected. A default may only be selected after
all four real variants have edge-category quality and resource evidence, and
the decision must not be based on aggregate alpha IoU alone.

Run the offline authority and Rust smoke:

```text
NUMBA_CACHE_DIR=/tmp/m12-numba python tests/fixtures/m12/generate_vitmatte_fixture.py \
  --output runs/m12-vitmatte/authority
ORT_DYLIB=/path/to/libonnxruntime.dylib \
  cargo run -p bgremove-bench -- m12-smoke --output runs/m12-vitmatte
```
