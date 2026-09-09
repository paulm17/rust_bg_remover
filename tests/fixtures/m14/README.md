# M14 prompted SAM authority fixture

This Level-2 fixture source-executes the coordinate and prompt encoding and
SciPy-backed restoration functions from the pinned
`projects/python/rembg/rembg/sessions/sam.py` at commit
`030a9ed79dbfcf8c58a1dc15a8dca3ccd2355709`. Run it with the pinned Python
environment recorded in `python-dependencies.lock`; it fails closed when SciPy
is unavailable. The release SAM weights are not locally available, so the
checked-in synthetic encoder/decoder pair and deterministic embedding, decoder
logits, quality scores and low-resolution prior are explicitly not evidence of
real-weight quality.

The reference directory is intentionally compact: the full encoder-input
digest and packed final masks are retained alongside bounded source samples;
deterministic synthetic arrays are reconstructed in the Rust consumer. Both
prompt cases are checked for portrait, landscape and square images, including
no-prior/iterative-prior inputs, candidate raw/restored logits,
quality/low-resolution outputs, final selected mask, and explicit source-union
parity. The raw/embedding/low-resolution/restored tensor arrays are
sample-checked rather than claimed as full byte-identical parity.

Run twice from the repository root:

```text
python3 tests/fixtures/m14/run_authoritative.py
python3 tests/fixtures/m14/run_authoritative.py
```

The script fails closed if the pinned source commit or file hash changes.
