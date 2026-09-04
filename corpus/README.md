# M0 frozen corpus

`manifest.jsonl` is the source of truth for the six supplied arena pairs. Paths
are relative to this directory: the inputs are `../test_images/reference/*`
and the PhotoRoom targets are `../test_images/photoroom/*`.

The target policy is `shadow_policy: preserve-target-effects`: alpha, shadows,
glows, and translucency encoded by the supplied PhotoRoom RGBA target are part
of parity. This is fixed for the corpus and must not change between milestones.
`subject_policy: primary-subject` selects the intended primary subject while
retaining target-encoded effects around it. All prompt fields are null because
these are automatic-arena records, not assisted SAM evaluations.

PhotoRoom creation date, tool, and version are unknown. They are represented as
JSON `null`; no provenance is guessed.

The six images are not statistically sufficient for the full taxonomy in
`plan.txt`. Coverage gaps are listed in validator and baseline output and are
not treated as malformed records.

Splits are disjoint and duplicate groups are kept within one split:

- `tune`: arena 1 and 2 (portraits)
- `validation`: arena 3 and 4 (emissive/translucent artwork)
- `blind`: arena 5 and 6 (character/low-resolution challenges)

The Section 6 leave-one-image-out arena remains available for fixed ranking;
the M0 baseline marks blind as evaluation-only and performs no parameter sweep.

Run from the repository root:

```text
cargo run --release -- check
cargo test
```
