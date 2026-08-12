# Task 10 implementation report: adapt oversized header images safely

## Delivered

Header images now cross a bounded local preparation seam before any hosted
image request. Static JPEG, PNG, and WebP inputs at or below 5,000,000 bytes and
20,000,000 pixels retain their exact immutable source bytes. Larger eligible
inputs are decoded only through a 40,000,000-pixel safety ceiling, EXIF
transposed, alpha-composited on white, converted to RGB, aspect-scaled only
when above the hosted pixel ceiling, and encoded as metadata-free JPEG with
Pillow 11.3.0, quality 85, 4:2:0 subsampling, Lanczos resizing, and
`optimize=false`/`progressive=false`.

Those limits and encoding settings are a conservative implementation policy
pending confirmation by the credentialed synthetic pilot. The available pilot
evidence did not contain a completed live result choosing an image transform or
format matrix. Both the complete input-rule description and the versioned
transform description participate in immutable configuration identity, so a
future pilot-confirmed change creates new vector meaning instead of overwriting
existing meaning.

Derived bytes are installed beneath
`renditions/<sha256-of-transform-id>/<derived-sha256>.jpg`. Publication uses an
exclusive descriptor-relative temporary leaf, fsync, reopen, byte/hash/decode
verification, atomic no-overwrite installation, directory fsync, and final
reverification. A pre-state failpoint proves that the rendition exists before
any run or work row can commit. Existing content-addressed leaves must verify
exactly before reuse. Source bytes are never renamed, rewritten, moved, or
deleted.

Unsafe, animated, unsupported, corrupt, encode-failed, and still-oversized
inputs become explicit terminal zero-attempt image work. They make no image
adapter call, store no provenance row, publish no image or multimodal vector,
and create no placeholder.

## Schema, provenance, and refusal contract

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from `9` to `10`.
- Added append-only `image_input_provenance`, keyed by article,
  configuration, and immutable generation run.
- Each successful image generation records source and exact embedded-input
  SHA-256, formats, byte counts, dimensions, transform identity, creation time,
  and nullable output-relative rendition path. No database column contains
  image bytes.
- Pass-through rows use `exact-source-bytes-v1`, identical source/embedded
  facts, and no rendition path. Derived rows use the configuration's transform
  and name the exact content-addressed rendition supplied to the adapter.
- Work and research-vector `input_sha256` remain the immutable source hash for
  change detection and supersession. The new provenance separately preserves
  the exact derived input identity when those bytes differ.
- The catalog has exactly eleven base tables and no indexes. An exact physical
  schema-v9 fixture is refused without mutation; earlier versions and malformed
  current schemas retain the existing fail-closed behavior.
- Added the exact reproducible runtime dependency `pillow==11.3.0`. No shared
  environment or machine configuration was changed.

## Red-green evidence

The exact schema fixture first observed RED because
`image_input_provenance` was absent: `1 failed in 2.70s`. After the schema bump,
table, constraints, and creation path, it observed GREEN: `1 passed in 0.78s`.

At the public coordinator seam:

- exact-byte pass-through/provenance observed RED with the missing codec seam
  (`1 failed in 1.71s`), then GREEN (`1 passed in 4.81s`);
- deterministic rendition observed RED with
  `rendition_installation_unavailable` (`1 failed in 1.80s`), then GREEN
  (`1 passed in 5.13s`);
- unsupported, corrupt, and still-oversized inputs observed RED as escaping
  preparation errors (`3 failed, 130 deselected in 3.62s`), then GREEN
  (`3 passed, 130 deselected in 5.69s`); the strengthened terminal validator
  selection remained GREEN (`3 passed, 134 deselected in 9.77s`);
- the install/state interruption fixture observed RED because the failpoint was
  absent (`1 failed in 1.79s`), then GREEN (`1 passed in 4.34s`); and
- missing/tampered rendition validation observed RED because changed bytes were
  invisible (`1 failed in 4.73s`), then GREEN (`1 passed in 10.90s`).

The exact v10/v9/v8 schema/refusal selection passed
`3 passed, 133 deselected in 4.00s`. Transform-change isolation passed in
`8.66s`; codec/profile ambiguity refusal passed in `1.95s`; and complete
configuration-identity coverage passed in `0.47s`.

## Focused verification

No full repository suite was run. Completed affected behavior evidence before
commit included:

- pass-through, successful rendition, unsupported/corrupt/still-oversized,
  interruption, validator, transform-isolation, codec/profile, and schema
  selections listed above;
- prior absent/changed/interrupted image behavior: `3 passed in 17.26s`;
- prior missing/retryable/deterministic-terminal image behavior:
  `3 passed in 12.63s`;
- prior terminal registration/removal/history behavior:
  `3 passed in 19.10s`.

Final non-suite verification after the complete configuration labels and docs:

- changed-file Ruff: `All checks passed!`;
- worktree-source embedding smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- preprocessing smoke: two canonical articles, three discovered inputs, one
  duplicate, three successful candidates, zero failures, validation true;
- worktree-source `run --help` retained all three explicit roots, required
  positive `--limit`, bounded `--reprocess`, and explicit hosted authorization;
- `git diff --check` produced no output;
- the tracked `data|artifacts|outputs` audit produced no matches;
- the schema-v9/ten-table stale-claim audit produced no matches; and
- all relative links in the two current operator documents resolve.

No licensed archive, credential, network request, real embedding output,
shared-environment mutation, or full-corpus operation was used.

## Self-review

- The source reader completes its stable no-follow read and hash before output
  mutation. Preparation recomputes the same hash and never opens the source for
  writing.
- Pillow performs a complete verified decode before pass-through. Pixel,
  format, frame-count, and positive-dimension checks fail closed. The codec and
  profile must declare byte-for-byte identical input and transform identities;
  ambiguity aborts before run/work state.
- The rendition's namespace includes transform identity and its filename
  includes exact derived identity. Neither a transform change nor different
  bytes can overwrite prior meaning.
- Installation precedes `begin_run`; publication of the image vector and its
  provenance shares one transaction. An interruption between those boundaries
  leaves only a harmless deterministic derived file, and replay verifies and
  reuses it.
- Adapter input is base64 of `PreparedImageInput.data`. Provenance hashes those
  exact bytes, while source identity remains available independently for
  change detection and source validation.
- Validator coverage requires one provenance row for every active or archived
  header-image generation, rejects orphan/malformed rows, proves pass-through
  equivalence, checks derived configuration/path identity, and reopens the
  rendition safely to verify exact length and hash.
- Local preparation terminals are restricted to five stable content-free
  codes with zero attempts. Existing provider/transport retry and deterministic
  request-attempt behavior remains separate and unchanged.
- Generated fixtures snapshot the source/preprocessing inputs around successful
  rendition, terminal preparation, and interrupted installation, making source
  mutation observable.

## Concerns

The recorded pilot artifacts do not prove the 5,000,000-byte/20,000,000-pixel
contract or the chosen JPEG rendition against a credentialed live endpoint.
The policy is therefore deliberately versioned and documented as pending pilot
confirmation. The unchanged shared virtual environment does not contain
Pillow 11.3.0, so the production Pillow codec was not executed here; generated
coordinator tests exercised the same policy and publication seam with a
deterministic injectable codec. Ticket 15 owns a clean temporary dependency
install and production-codec test with synthetic Pillow fixtures before real
corpus authorization.
