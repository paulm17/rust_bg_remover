# M9 trimap parity fixture

The fixture executes the pinned source implementations for rembg's symmetric
trimap and CarveKit's probability-aware trimap. No checkpoint, model, or
network access is used. The source runners fail closed if either checkout is
not at its pinned commit or its tracked trimap files are dirty.

The expected interpreter is Python 3.12.11 with NumPy 2.3.2, Pillow 12.2.0,
SciPy 1.17.0, and OpenCV 4.10.0. Set `M9_PYTHON` to that interpreter:

```sh
M9_PYTHON=/path/to/pinned/python3.12
"$M9_PYTHON" tests/fixtures/m9/run_authoritative_trimaps.py \
  --output /tmp/m9-authoritative --repo-root .
cp /tmp/m9-authoritative/report.json tests/fixtures/m9/authoritative-report.json
"$M9_PYTHON" tests/fixtures/m9/generate_m9_fixture.py \
  --source-output /tmp/m9-authoritative --output /tmp/m9-reference
rm -rf tests/fixtures/m9/reference && mkdir -p tests/fixtures/m9/reference
cp /tmp/m9-reference/* tests/fixtures/m9/reference/
cargo run -p bgremove-bench --offline --locked -- m9-smoke \
  --output /tmp/m9-smoke
```

The two source output directories can be generated independently and compared
recursively. Assembly copies only the complete input and two trimap PNGs plus
`parity.json`; all provenance is recorded in `authoritative-report.json`.

Rust uses strict `>` thresholding, full square `(2r+1)^2` kernels, explicit
constant borders, disk offsets `dx²+dy² <= r²`, separable f64 Gaussian blur
with SciPy's `floor(4*sigma + 0.5)` (`truncate=4`) radius and half-sample
reflect borders, and `round(fraction * min(width,height))` for relative
radii. rembg uses literal square `erode_size` kernels (including even `10`),
SciPy's default connectivity-1 cross for `erode_size=0`, and
foreground-false/background-true erosion borders. Its post-process uses the
skimage disk opening with reflect borders. CarveKit uses OpenCV's neutral
erosion border, false dilation border, 127 unknown, and 255 foreground output
levels. The fixture also includes the source-backed 1x5, 6x6, border-value-1
oversized-kernel case; Rust evaluates its anchored in-bounds window exactly
without allocating a full oversized footprint per pixel.
