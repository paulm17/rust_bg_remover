# M7 BiRefNet Level-2 parity fixture

The authoritative Level-2 artifacts are the source-generated files under
`reference/`. The pinned Python generator imports and calls rembg's official
`BiRefNetSessionGeneral` and `BiRefNetSessionGeneralLite` classes at commit
`030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709`, with preseeded external weights and
no download path. Each case contains complete decoded RGB, preprocessed NCHW
tensor, raw ONNX output, restored alpha, and final straight-alpha cutout
artifacts. `reference/report.json` records the source/dependency revisions,
weight and license hashes, stage hashes, and portable paths; the ignored Rust
parity test verifies this provenance and every artifact hash before comparing
the stages. The separate weight-license artifact is the MIT text from the
upstream BiRefNet repository at commit
`ebcc0bc8ec7fe919cec829f2dea656b3078acddc`; the report records its raw URL and
hash. The exact rembg-exported weight URLs and hashes remain recorded per case.

Weights remain external and ignored. The small pure Rust tests separately pin
the sigmoid-before-normalization order, ImageNet/global-max normalization,
shape, and finite-value behavior. Regenerate with
`/opt/homebrew/bin/python3.12 tests/fixtures/m7/generate_rembg_birefnet_fixture.py
--weights-dir projects/python/rembg` only from the clean pinned rembg checkout
and a task-scoped model home with the pinned NumPy, ONNX Runtime, and Pillow
versions recorded in `reference/report.json`.
