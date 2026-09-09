# M13 foreground-colour parity fixture

This fixture is a small synthetic, redistributable Level-2/Level-3 check for
the foreground-colour stage. `run_authoritative.py` loads the pinned
PyMatting 1.1.15 source file directly and executes `estimate_foreground_ml`
twice (the installed package cannot be imported normally on this host because
Numba attempts to create a cache beside the read-only site-package file).
The script disables only Numba's JIT/cache decorator; the algorithm body and
its nearest-neighbour multilevel arithmetic are the pinned source.

The Rust implementation compares against the resulting `foreground.f32le`
and `background.f32le` artifacts. All runs use the same known synthetic
foreground/background/alpha composite, so recovery is measured independently
of segmentation. Hidden RGB at alpha zero is not scored.

`run_fast_authority.py` source-executes an externally provisioned snapshot of
the official PhotoRoom `blurfusion_foreground_estimation.py` at commit
`a98fe5dfe4b61521de1a92dcaeb6d138f95bc97d`, SHA-256
`b9cbb83ddfa93c4fb2686fe99f9bc61769d9b6001df363660c664de8ad9ce94c`. The
source is intentionally not vendored because the upstream repository has no
checked-in license file. The parity cases compare both passes and both
returned background fields across asymmetric geometries. Source mode uses
literal even kernel widths 90 and 6 with OpenCV `BORDER_REFLECT_101`;
reference-size scaling and clamp borders are separate explicit extensions.

Source:

- PyMatting 1.1.15, installed in the pinned ComfyUI Python environment.
- `pymatting/foreground/estimate_foreground_ml.py`

Run from the repository root:

```text
NUMBA_DISABLE_CACHING=1 /Volumes/Data/Users/paul/scratch/comfyUI/.venv/bin/python \
  tests/fixtures/m13/run_authoritative.py
git clone https://github.com/Photoroom/fast-foreground-estimation.git /tmp/fast-foreground-estimation
git -C /tmp/fast-foreground-estimation checkout a98fe5dfe4b61521de1a92dcaeb6d138f95bc97d
M13_PHOTOROOM_SOURCE=/tmp/fast-foreground-estimation/blurfusion_foreground_estimation.py \
M13_PHOTOROOM_REPO=/tmp/fast-foreground-estimation \
M13_PHOTOROOM_COMMIT=a98fe5dfe4b61521de1a92dcaeb6d138f95bc97d \
/Volumes/Data/Users/paul/scratch/comfyUI/.venv/bin/python tests/fixtures/m13/run_fast_authority.py
cargo test -p bgremove-color
```
