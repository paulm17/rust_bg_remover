# M6 CarveKit Level-2 parity fixtures

`generate_carvekit_parity.py` uses the pinned CarveKit source at
`f141a311af67fb1da64269c508a6d1f786420801`, the externally acquired and
exported checkpoints in `projects/python/image-background-remove-tool/m6-onnx`,
and CPU ONNX Runtime. It generates six cases per adapter: 3×2, 2×3 and
1025×3 geometries, each in low `[0,64]` and high `[192,255]` encoded RGB
ranges. The 1025×3 case exercises DeepLabV3's aspect-preserving thumbnail and
Pillow-compatible bicubic resize instead of merely testing no-op tiny inputs.

Each case retains decoded RGB, the complete NCHW tensor, raw output, restored
alpha, and straight-alpha RGBA cutout. DeepLabV3 records the explicit VOC class
argmax (class 0 background; classes 1–20 foreground) and restores with nearest
neighbour so the adapter remains hard until an explicit refiner.

The report declares max and mean float tolerances in normalized tensor and raw
output units. Generation exits nonzero on any failed max/mean gate; the Rust
fixture test reads and enforces the same declarations. TRACER uses the pinned
torchvision tensor bilinear half-pixel antialiased resize, with a measured
maximum tensor delta of 1.67e-6 normalized units (well below one input code).
The generator also fail-closes on the exact CarveKit commit, torch/torchvision/
Pillow/numpy/ONNX Runtime versions, source-checkpoint SHA-512 values, and
exported-ONNX SHA-256 values; these are retained in the report provenance.
Restored alpha uses declared max/mean tolerances, exact RGB cutout bytes, and
at-most-one-code alpha cutout tolerance.

Regenerate with the pinned temporary environment:

```sh
PYTHONPATH=/private/tmp/m6-py312-torch:/private/tmp/m5-py312-pinned:$PWD/projects/python/image-background-remove-tool \
  /opt/homebrew/bin/python3.12 tests/fixtures/m6/generate_carvekit_parity.py
```
