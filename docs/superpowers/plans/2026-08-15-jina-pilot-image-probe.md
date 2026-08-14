# Jina Pilot Image Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rejected 1×1 generated pilot image with a deterministic 224×224 RGB image so the generated-only live pilot can truthfully verify Jina v4 image and mixed-modality readiness.

**Architecture:** Keep the existing `_encoded_png()` seam and request flow. Change only its deterministic pixel payload and update the structural fixture checks; exact-size ancillary padding remains unchanged, and a live pilot remains the final hosted-contract gate.

**Tech Stack:** Python 3.13, stdlib `base64`/`struct`/`zlib`, pytest, Ruff, Jina Embeddings v4 hosted API.

## Global Constraints

- The pilot remains generated-only and must not read or write licensed corpus state.
- Normal image dimensions are exactly 224×224 RGB with PNG color type 2 and 8-bit channels.
- The 5,000,000-byte and 8,000,000-byte boundary images remain exact-size valid PNGs.
- Adapter request shape, quota policy, catalog schema, and corpus image transformations do not change.
- Do not run a real article canary or full-corpus embedding job in this plan.

---

### Task 1: Generate a realistic pilot PNG and verify hosted readiness

**Files:**
- Modify: `tests/test_jina_pilot.py:1033-1096`
- Modify: `src/wsj_embeddings/pilot.py:730-753`

**Interfaces:**
- Consumes: `_encoded_png(target_size: int | None = None) -> str`
- Produces: the same function signature, returning base64 for a deterministic 224×224 RGB PNG; when `target_size` is supplied, decoded bytes equal it exactly.

- [ ] **Step 1: Write the failing structural test**

Add a normal-image test and generalize the existing helper so it verifies the
224×224 IHDR, decompressed row count, filter byte, RGB payload size, CRCs, and
optional padding:

```python
def test_normal_generated_png_has_realistic_static_dimensions() -> None:
    image = base64.b64decode(_encoded_png(), validate=True)

    _assert_valid_png(image, padded=False)


def _assert_valid_png(image: bytes, *, padded: bool) -> None:
    # Preserve the existing chunk/CRC parser.
    expected_kinds = [b"IHDR", b"IDAT", *([b"tEXt"] if padded else []), b"IEND"]
    assert [kind for kind, _ in chunks] == expected_kinds
    assert chunks[0][1] == struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0)
    pixels = zlib.decompress(chunks[1][1])
    assert len(pixels) == 224 * (1 + 224 * 3)
    assert all(pixels[row * (1 + 224 * 3)] == 0 for row in range(224))
```

Update `test_near_limit_generated_pngs_are_structurally_valid` to call
`_assert_valid_png(image, padded=True)` and preserve its exact-size assertions.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_jina_pilot.py::test_normal_generated_png_has_realistic_static_dimensions \
  tests/test_jina_pilot.py::test_near_limit_generated_pngs_are_structurally_valid
```

Expected: the normal-image test fails because the current IHDR is 1×1.

- [ ] **Step 3: Implement the minimal deterministic image correction**

In `_encoded_png`, set width and height to 224 and create a deterministic
non-uniform checkerboard without Pillow:

```python
width = height = 224
dark = b"\x1e\x5a\xb4"
light = b"\xdc\xb4\x28"
rows = []
for y in range(height):
    row = bytearray(b"\x00")
    for x in range(width):
        row.extend(dark if ((x // 16) + (y // 16)) % 2 == 0 else light)
    rows.append(bytes(row))
pixels = b"".join(rows)
png = _png_chunk(
    b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
)
png += _png_chunk(b"IDAT", zlib.compress(pixels))
```

Retain the existing signature, PNG signature, IEND chunk, exact-size padding,
and base64 return.

- [ ] **Step 4: Run focused GREEN and adjacent regression checks**

Run:

```bash
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_jina_pilot.py tests/test_jina_hosted_adapter.py
/home/willis/projects/finance_wsj/.venv/bin/ruff check \
  src/wsj_embeddings/pilot.py tests/test_jina_pilot.py
git diff --check
```

Expected: all selected tests pass, Ruff reports no errors, and diff check is
clean.

- [ ] **Step 5: Rerun the live generated-only pilot**

Load the ignored root `.env` without printing it and write the content-free
result to a mode-600 temporary file:

```bash
set -a
. /home/willis/projects/finance_wsj/.env
set +a
umask 077
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/wsj-embeddings pilot \
  > /tmp/wsj-jina-pilot.json
```

Parse and report only readiness, probe status, returned safe model labels,
dimensions, request/retry counts, and safe HTTP classifications. Expected:
`text_normal`, `image_normal`, and `mixed_normal` succeed at 2,048 dimensions,
and `readiness.outcome` is `ready`. If not, stop and diagnose rather than
starting a corpus canary.

- [ ] **Step 6: Commit the pilot correction**

```bash
git add src/wsj_embeddings/pilot.py tests/test_jina_pilot.py
git commit -m "fix: use realistic generated pilot image"
```

Do not include generated pilot output, `.env`, catalogs, vectors, or article
artifacts in the commit.
