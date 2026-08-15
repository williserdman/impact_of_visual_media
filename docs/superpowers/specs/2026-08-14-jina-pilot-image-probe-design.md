# Jina Pilot Image Probe Correction

## Context

The generated-only live pilot currently uses a valid but unrealistic 1×1 RGB
PNG for its normal image and mixed-modality readiness probes. Jina's hosted
`jina-embeddings-v4` endpoint rejects that image with HTTP 400, while the same
request using Jina's published 28×56 sample succeeds with a 2,048-dimensional
embedding. The API documents base64 image input and an image byte limit, but it
does not document an exact required width or height.

This is a pilot-fixture defect. It does not demonstrate a defect in canonical
header-image preparation or publication, and no licensed article or image was
used to diagnose it.

## Considered approaches

1. Generate a realistic 224×224 PNG locally. This keeps the pilot deterministic,
   generated-only, dependency-free, and independent of a third-party fixture.
2. Embed Jina's published 28×56 base64 sample in the repository. This is known
   to work today, but unnecessarily couples the pilot to external sample bytes.
3. Accept the failed image probe. This would make readiness untruthful and could
   allow a corpus run without proving the image modality.

Approach 1 is selected.

## Design

`_encoded_png()` will generate a deterministic, valid, static 224×224 RGB PNG.
The optional exact-size behavior used by the 5 MB and 8 MB boundary probes will
remain intact by padding the generated PNG with one valid ancillary chunk.

The readiness contract is unchanged:

- `text_normal`, `image_normal`, and `mixed_normal` must all succeed;
- successful vectors must have the configured 2,048 dimensions;
- the pilot remains generated-only and writes no corpus/catalog state; and
- boundary-probe rejection remains an observation, not a readiness failure.

No adapter request schema, quota setting, article-image transform, catalog
schema, or embedding identity changes.

## Verification

Before changing production code, a focused test will assert that the normal
generated image is a valid 224×224 RGB PNG and that exact-size variants remain
valid and byte-exact. The test must fail against the existing 1×1 generator.

After the minimal generator correction:

1. run the focused pilot tests and hosted-adapter tests;
2. run Ruff and diff hygiene;
3. rerun the live generated-only pilot using the configured API key; and
4. require `readiness.outcome == "ready_for_operator_review"` before proposing
   a licensed one-article canary.

No real article canary or full-corpus run is authorized by this design.
