# bg_remover_rust

`bg_remover_rust` is a typed Rust workspace for reproducible background-removal
experiments. It contains image decoding and alpha contracts, model-manifest
validation, ONNX Runtime adapters, classical matting/foreground stages, and
benchmark evidence for milestones M0–M17.

## Important first fact

The user-facing `bgremove run` command is currently a deterministic **no-op
pipeline**. It is useful for checking decoding, geometry, alpha handling, PNG
encoding, and artifact layout; it is not a production AI background remover and
does not claim model quality. `bgremove mask` uses the same no-op stages to
write a mask.

Model-backed inference is available only through the explicitly wired
benchmark/model paths, with a locally installed compatible ONNX Runtime,
hash-verified model files, and the required licence metadata. The project never
downloads weights or runtimes during a normal command.

## Start here, in order

1. **Enter the repository.**

   ```sh
   cd /path/to/bg_remover_rust
   rustc --version
   cargo --version
   ```

2. **Build offline, one command at a time.** The first compilation can take a
   while because Cargo builds all workspace crates.

   ```sh
   CARGO_BUILD_JOBS=1 cargo build --workspace --offline --locked
   ```

3. **Run the fast focused tests first.**

   ```sh
   CARGO_BUILD_JOBS=1 cargo test -p bgremove-cli --offline --locked -- --test-threads=1
   CARGO_BUILD_JOBS=1 cargo test -p bgremove-bench --offline --locked -- --test-threads=1
   ```

4. **Run the full serialized suite.** This is substantially longer; wait for
   each command to finish and do not run another Cargo or benchmark process at
   the same time.

   ```sh
   CARGO_BUILD_JOBS=1 cargo test --workspace --offline --locked -- --test-threads=1
   ```

5. **Run one image through the honest user-facing path.** No current single
   `bgremove` command produces a production AI cutout; `run` is intentionally
   the deterministic no-op pipeline described above.

   ```sh
   CARGO_BUILD_JOBS=1 cargo run -p bgremove-cli --offline --locked --bin bgremove -- \
     run --input test_images/reference/1.png --output runs/example-run
   ```

6. **Inspect what it produced.**

   ```sh
   find runs/example-run -maxdepth 1 -type f -print | sort
   jq . runs/example-run/run.json
   ```

7. **Optionally attempt one verified model-backed run.** Set `ORT_DYLIB` and
   provision the exact hash-verified model files first. The checked-in Edge ORT
   attempt is currently blocked by an unsupported `MaxPool(11)` operator, so a
   blocked report is expected and must not be described as AI success.

   ```sh
   ORT_DYLIB=/path/to/libonnxruntime.dylib \
   CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
   cargo run -p bgremove-bench --offline --locked -- m17-smoke \
     --input test_images/reference/1.png --workers 1 \
     --output runs/m17-performance
   ```

## Prerequisites and safe operating rules

- Rust and Cargo with the toolchain needed by the workspace.
- A checked-out repository and its tracked `Cargo.lock`.
- For model-backed runs: a compatible ONNX Runtime dynamic library and the
  external checkpoint files named by the selected TOML manifest.
- Run commands from the repository root.
- Run tests and benchmarks sequentially. Use `CARGO_BUILD_JOBS=1` and
  `--test-threads=1`; do not launch multiple `bgremove-bench` processes.

The normal build is offline and locked:

```sh
CARGO_BUILD_JOBS=1 cargo build --workspace --offline --locked
```

There is no application-level “clean” or “release” mode. `cargo run` uses a
development build; adding `--release` only changes Cargo’s compiler profile.
The evidence commands define their own output directories. M17 additionally
creates a reproducible, ownership-marked release bundle in a dedicated
`m17*` directory.

## First run: the genuinely usable no-op paths

Use a tracked image such as `test_images/reference/1.png`.

```sh
# User-facing deterministic cutout (not AI removal)
CARGO_BUILD_JOBS=1 cargo run -p bgremove-cli --offline --locked --bin bgremove -- \
  run --input test_images/reference/1.png --output runs/example-run

# User-facing deterministic mask
CARGO_BUILD_JOBS=1 cargo run -p bgremove-cli --offline --locked --bin bgremove -- \
  mask --input test_images/reference/1.png --output runs/example-mask

# Equivalent benchmark harness path
CARGO_BUILD_JOBS=1 cargo run -p bgremove-bench --offline --locked -- \
  run --input test_images/reference/1.png --output runs/example-bench

# Compare two deterministic artifact files byte-for-byte
CARGO_BUILD_JOBS=1 cargo run -p bgremove-bench --offline --locked -- \
  compare --left runs/example-bench/run.json --right runs/example-bench/run.json
```

The user CLI’s `run` writes `resolved-config.json`, `cutout.png`, and `run.json`;
its `mask` subcommand writes `resolved-config.json`, `mask.png`, and `mask.json`.
The benchmark `run` additionally writes both `cutout.png` and `mask.png`.
PNGs are straight-alpha RGBA cutouts or single-channel masks at canonical
source dimensions; source RGB is not silently premultiplied.

The default transparent-input policy is `multiply-predicted`. Use
`--transparent-input-policy replace-source-alpha` when the predicted mask should
replace the source alpha instead.

## Inspect help and arguments

The two binaries are separate:

```sh
CARGO_BUILD_JOBS=1 cargo run -p bgremove-cli --offline --locked --bin bgremove -- --help
CARGO_BUILD_JOBS=1 cargo run -p bgremove-cli --offline --locked --bin bgremove -- run --help
CARGO_BUILD_JOBS=1 cargo run -p bgremove-cli --offline --locked --bin bgremove -- mask --help
CARGO_BUILD_JOBS=1 cargo run -p bgremove-cli --offline --locked --bin bgremove -- inspect-model --help

CARGO_BUILD_JOBS=1 cargo run -p bgremove-bench --offline --locked -- --help
CARGO_BUILD_JOBS=1 cargo run -p bgremove-bench --offline --locked -- m17-smoke --help
```

### `bgremove` options

| Subcommand | Required | Options and defaults |
|---|---|---|
| `run` | `--input PATH` | `--output runs/m2-run`; `--transparent-input-policy multiply-predicted` or `replace-source-alpha` |
| `mask` | `--input PATH` | `--output runs/m2-mask`; same transparent-input policy options |
| `inspect-model` | none | `--manifest models/m3_identity.toml`; `--provider cpu`; `--allow-provider-fallback` |

`inspect-model` validates the TOML manifest, licence hash, model hash, and ONNX
graph before opening a session. Its accepted providers are `cpu`, `coreml`, and
`cuda`; an `ORT_DYLIB` environment variable naming an existing compatible
runtime is required.

### `bgremove-bench` everyday commands

| Command | Purpose | Main defaults |
|---|---|---|
| `validate` | Validate every corpus record, image/hash, split, and leakage rule | `--manifest corpus/manifest.jsonl` |
| `baseline` | Write the deterministic zero/one-alpha baseline report | output `runs/m0-baseline/report.json` |
| `check` | Validate, then write the baseline report | same manifest/output defaults |
| `run` | Run the deterministic M2 no-op pipeline | `--input` required; output `runs/m2-bench` |
| `compare` | Compare two files byte-for-byte | `--left` and `--right` required |
| `m3-smoke` | Run the checked-in identity ONNX fixture twice | manifest `models/m3_identity.toml`; output `runs/m3-ort`; `--workers 2` |
| `m4-smoke` | IS-Net tensor/profile report | optional `--input`; manifest `models/m4_isnet_fp32.toml`; provider `cpu` |
| `m5-smoke` | U2-Net family report | optional `--input`; manifest `models/m5_u2net.toml`; provider `cpu` |
| `m6-smoke` | CarveKit registry/raw-alpha report | optional `--input` and paired `--reference` |
| `m7-smoke` | BiRefNet report | optional `--input` and paired `--reference` |
| `m8-smoke`–`m12-smoke` | Registry, trimap, matting, FBA, and ViTMatte evidence | each has an output default under `runs/` |
| `m15-smoke` | Staged tournament over the six-image arena | output `runs/m15-tournament` |
| `m16-smoke` | P5 hybrid contract and fail-closed real status | output `runs/m16-hybrid` |
| `m17-smoke` | Provider/performance/batch/release evidence | output `runs/m17-performance` |

Most M4–M7 runtime reports require an explicit `ORT_DYLIB`, an input, and
external hash-verified weights. M8 and M12 deliberately report unavailable
external checkpoints where those files are not supplied. M9–M12 also have
deterministic fixture paths that do not download anything.

## Corpus and model manifests

The six checked-in arena pairs are described by
[`corpus/manifest.jsonl`](corpus/manifest.jsonl), with inputs in
[`test_images/reference`](test_images/reference) and PhotoRoom targets in
[`test_images/photoroom`](test_images/photoroom). The compact arena view is
[`test_images/arena.jsonl`](test_images/arena.jsonl). The corpus README explains
the fixed target policy and tune/validation/blind labels.

Manifests under [`models/`](models) are declarations, not downloaded weights.
They pin the model id, tensor contract, preprocessing, source revision, model
SHA-256, licence identifier/hash, intended-use approval, and opset. Examples:

- [`models/m3_identity.toml`](models/m3_identity.toml) points to the tracked
  local identity fixture.
- [`models/m5_u2net.toml`](models/m5_u2net.toml) describes the external rembg
  U2-Net checkpoint.
- [`models/m4_isnet_fp32.toml`](models/m4_isnet_fp32.toml),
  [`models/m6_tracer_b7.toml`](models/m6_tracer_b7.toml), and
  [`models/m7_birefnet_general.toml`](models/m7_birefnet_general.toml) describe
  external model families.

To inspect the local fixture, provide a compatible runtime:

```sh
ORT_DYLIB=/path/to/libonnxruntime.dylib \
  CARGO_BUILD_JOBS=1 cargo run -p bgremove-cli --offline --locked --bin bgremove -- \
  inspect-model --manifest models/m3_identity.toml --provider cpu
```

For M3/M4/M5 and later ORT-backed smoke reports, the same `ORT_DYLIB` convention
applies. A model file must exist at the manifest’s `file` path and match its
declared SHA-256; normal commands do not fetch it.

### One-image model-backed recipes

When the external files named by the manifests are actually provisioned, these
are the concrete single-image entry points. They fail closed before inference
if a file, hash, licence, or compatible runtime is missing:

```sh
# M4 IS-Net
ORT_DYLIB=/path/to/libonnxruntime.dylib \
  CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  cargo run -p bgremove-bench --offline --locked -- m4-smoke \
  --input test_images/reference/1.png --manifest models/m4_isnet_fp32.toml \
  --output runs/m4-isnet --workers 1

# M5 U2-Net
ORT_DYLIB=/path/to/libonnxruntime.dylib \
  CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  cargo run -p bgremove-bench --offline --locked -- m5-smoke \
  --input test_images/reference/1.png --manifest models/m5_u2net.toml \
  --output runs/m5-u2net --workers 1

# M6 CarveKit and M7 BiRefNet (both need the paired target for scoring)
ORT_DYLIB=/path/to/libonnxruntime.dylib \
  CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  cargo run -p bgremove-bench --offline --locked -- m6-smoke \
  --input test_images/reference/1.png --reference test_images/photoroom/1.png \
  --output runs/m6-carvekit --workers 1
ORT_DYLIB=/path/to/libonnxruntime.dylib \
  CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  cargo run -p bgremove-bench --offline --locked -- m7-smoke \
  --input test_images/reference/1.png --reference test_images/photoroom/1.png \
  --output runs/m7-birefnet --workers 1
```

`m3-smoke` is the checked-in identity fixture and takes no image; it exercises
the runtime/session contract. M8, M9, M10, M11, and M12 are registry or fixture
smokes with their own output defaults rather than one-image production commands.

## M17: one-image provider/performance attempt

M17 preflights encoded dimensions before pixel decode, bounds each image at
16,777,216 pixels, bounds a batch at 33,554,432 pixels, caps workers at eight,
and rejects unsafe or malformed inputs. Its defaults are:

| Option | Default/meaning |
|---|---|
| `--input PATH` | Repeatable image input; no input produces no inference work |
| `--output PATH` | `runs/m17-performance`; must be a dedicated `m17*` directory |
| `--workers N` | `1`, with a hard maximum of `8` |
| `--max-pixels N` | `16,777,216` per image; cannot exceed the hard maximum |
| `--provider NAME` | Repeatable; on macOS/aarch64 omission means CPU then CoreML; otherwise CPU |
| `--allow-provider-fallback` | Off by default; records an explicitly allowed accelerated-to-CPU fallback |

On Apple Silicon, omit `--provider` to select the CPU→CoreML default. CUDA and
OpenVINO are optional explicit additions (`--provider cuda` and
`--provider openvino`); OpenVINO is reported unavailable in this build. CPU is
always attempted first. The batch runner emits per-image output PNGs and a
release/evidence bundle; it does not implement video because no temporal
consistency design has been accepted.

Example with the known local Edge runtime:

```sh
ORT_DYLIB='/Applications/Microsoft Edge.app/Contents/Frameworks/Microsoft Edge Framework.framework/Versions/152.0.4191.66/Libraries/libonnxruntime.dylib' \
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
cargo run -p bgremove-bench --offline --locked -- m17-smoke \
  --input test_images/reference/1.png --workers 1 \
  --output runs/m17-performance
```

This is a real runtime attempt, not a promise of success. The checked-in Edge
evidence currently fails session creation with `MaxPool(11)`; it therefore does
not claim CPU inference, accelerated parity, or a promoted release. M17 records
model-load, first-inference, warm median/p95, throughput, output encoding,
model-byte, provider, and memory fields as unavailable when they cannot be
measured.

## M15 and M16 evidence runs

These commands are deterministic and do not download weights:

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  cargo run -p bgremove-bench --offline --locked -- \
  m15-smoke --output runs/m15-tournament
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  cargo run -p bgremove-bench --offline --locked -- \
  m16-smoke --output runs/m16-hybrid
```

M15 runs the synthetic contract tournament and inventories/attempts the real
six-image arena. M16 runs its independent synthetic P5 mechanics evidence and
keeps real arena promotion fail-closed when approved runtime/weights are not
available. Neither synthetic report is a claim of real PhotoRoom quality.

## Inspecting outputs and checksums

Every command prints its output directory. Useful inspection commands are:

```sh
find runs/example-run -maxdepth 1 -type f -print | sort
find runs/m17-performance -maxdepth 2 -type f -print | sort
sed -n '1,220p' runs/m2-correctness/run.json
shasum -a 256 runs/m17-performance/report.json
shasum -a 256 -c runs/m17-performance/report.sha256
shasum -a 256 -c runs/m17-performance/artifacts.sha256
```

M17’s `report.sha256` covers `report.json`; its deterministic artifact manifest
covers the final bundle’s declared files and `artifacts.sha256` covers that
manifest. The bundle also contains `model-bom.json`,
`code-dependencies.json`, copied model manifests, `Cargo.lock`, and
`THIRD-PARTY-NOTICES.txt`. Raw M15/M16 cache probes are intentionally ignored
from committed deterministic artifact sets.

For visual inspection, open the PNGs with any image viewer. For structural
inspection, use `jq` if installed, for example:

```sh
jq . runs/m17-performance/report.json
jq . runs/m15-tournament/report.json
```

## Common failures and what they mean

| Symptom | Meaning and next check |
|---|---|
| `ORT_DYLIB must name an existing ... dylib` | Set `ORT_DYLIB` to an installed compatible ONNX Runtime; the project does not download one. |
| Edge `Could not find an implementation for MaxPool(11)` | The local Edge runtime is incompatible with the verified graph; this is an honest blocked runtime result, not a quality score. |
| CoreML is disabled/unavailable | The current build/runtime cannot execute CoreML; strict fallback does not turn CPU into accelerated success. |
| model/checkpoint not found or SHA-256 mismatch | Provision the exact file named by the manifest, or choose a tracked fixture. Do not edit the hash to make an unknown file pass. |
| missing licence file/hash or unapproved checkpoint | The manifest is intentionally fail-closed; inspect the model’s source/licence metadata and approval state. |
| unsupported GIF or malformed image | The decoder supports JPEG, PNG, and WebP in this build; convert the input or use a supported file. |
| M17 pixel/worker/output ownership error | Lower `--max-pixels`/input count or workers; use a dedicated owned `m17*` output directory and preserve prior bundles. |
| command succeeds but output is not an AI cutout | `bgremove run` and benchmark `run` are the deterministic no-op path by design. Use a verified model-backed smoke command when its prerequisites exist. |

Do not delete broad `runs/` or model directories to “fix” a failure. Choose a
new output path or remove only a directory you have verified is yours.

## Sequential verification commands

Run each command to completion before starting the next one:

```sh
CARGO_BUILD_JOBS=1 cargo fmt --all -- --check
CARGO_BUILD_JOBS=1 cargo test --workspace --offline --locked -- --test-threads=1
CARGO_BUILD_JOBS=1 cargo test --workspace --no-default-features --offline --locked -- --test-threads=1
CARGO_BUILD_JOBS=1 cargo clippy --workspace --all-targets --all-features --offline --locked -- -D warnings
CARGO_BUILD_JOBS=1 cargo clippy --workspace --all-targets --no-default-features --offline --locked -- -D warnings
git diff --check
```

If a dependency is not already cached, `--offline` will fail rather than
silently downloading it. Run a network-enabled dependency provisioning step only
when you explicitly intend to do that, then repeat the locked offline checks.

## Milestone and evidence index

The repository does not contain a standalone markdown file for every milestone;
the links below point to the checked-in documentation or its authoritative
evidence report.

| Milestone | Documentation/evidence |
|---|---|
| M0 | [`corpus/README.md`](corpus/README.md), [`corpus/manifest.jsonl`](corpus/manifest.jsonl), and `runs/m0-baseline/` |
| M1 | [`M1.md`](M1.md), `runs/m1-skeleton/` |
| M2 | [`M2.md`](M2.md), `runs/m2-correctness/` |
| M3 | [`M3.md`](M3.md), `runs/m3-ort/` |
| M4 | [`M4.md`](M4.md), `runs/m4-isnet/` |
| M5 | [`M5.md`](M5.md), `runs/m5-u2net/` |
| M6 | [`M6.md`](M6.md), `runs/m6-carvekit/` |
| M7 | `runs/m7-birefnet/` and the `m7-smoke --help` contract |
| M8 | `runs/m8-rmbg/` and the `m8-smoke --help` contract |
| M9 | `runs/m9-trimap/` |
| M10 | `runs/m10-matting/` |
| M11 | `runs/m11-fba/` |
| M12 | [`M12.md`](M12.md), `runs/m12-vitmatte/` |
| M13 | [`M13.md`](M13.md), `runs/m13-foreground/` |
| M14 | [`M14.md`](M14.md), `runs/m14-sam/` |
| M15 | [`M15.md`](M15.md), `runs/m15-tournament/` |
| M16 | [`M16.md`](M16.md), `runs/m16-hybrid/` |
| M17 | [`M17.md`](M17.md), `runs/m17-performance/` |

The complete design, scoring, safety, licensing, and reproducibility contract
is [`plan.txt`](plan.txt).
