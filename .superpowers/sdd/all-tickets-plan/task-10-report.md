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

## Review-fix round 1

Pillow decompression-bomb exceptions and warnings promoted during header open,
verify, decode, orientation, or render now map to the stable local terminal
`unsafe_image`. They do not abort the complete run, purchase an image request,
or publish a placeholder. A synthetic production-import seam covers both the
exception and warning paths without installing Pillow into the shared
environment.

The runtime transform identity now extends the pinned algorithm with sanitized
linked JPEG, libjpeg-turbo, zlib, and WebP build versions reported by Pillow.
The production adapter binds this codec before calculating or publishing its
profile/configuration identity. Rebinding to a different build or supplying a
codec that does not exactly match the declared profile fails closed before run
state. Two injected build matrices produce distinct runtime profiles and
configuration IDs. This verifies binding logic, not the real encoder; Task
14/ticket 15 still owns clean-install execution of Pillow 11.3.0 with generated
images.

Rendition directory creation is now crash durable: each newly created
descriptor-relative child entry is followed by fsync of its parent before the
next nested publication. Final verification holds the no-follow descriptor
through read/hash/decode, proves regular/single-link/size and pre/post descriptor
stability, then re-resolves the leaf without following links and requires the
same device/inode before returning. Replacement, added-hardlink, and symlink
races all abort before run or work state.

Read-only validation now uses the exact matching profile codec to decode current
source and embedded bytes. It independently checks hashes, formats, byte counts,
dimensions, exact pass-through eligibility, and the derived JPEG/hosted limits.
It fails closed when no matching decoder is available, detects image provenance
for unknown configurations globally, and descriptor-scans all rendition
namespaces without following symlinks. Unsafe entries, unreferenced final files,
and interrupted temporary files are reported without cleanup.

Review-fix RED/GREEN evidence:

- bomb and build identity initially observed `2 failed, 138 deselected`; after
  adding the warning case the complete RED was `3 failed, 138 deselected`, then
  GREEN was `3 passed, 138 deselected in 5.07s`;
- parent-fsync and final re-anchor fixtures observed
  `4 failed, 141 deselected in 18.80s`, then
  `4 passed, 141 deselected in 8.52s`;
- decoded source facts observed `1 failed, 145 deselected`, then
  `1 passed, 145 deselected in 4.94s`;
- unknown-configuration and descriptor-orphan scanning observed
  `2 failed, 148 deselected in 9.51s`; the combined decoded-facts,
  derived-contract, global-reference, and orphan selection then passed
  `5 passed, 145 deselected in 26.64s`;
- fail-closed missing decoder observed `1 failed, 150 deselected`, then
  `1 passed, 150 deselected in 5.32s`; and
- production-adapter runtime binding observed `1 failed, 149 deselected`, then
  `1 passed, 149 deselected in 1.82s`.

The combined affected implementation selection passed
`18 passed, 133 deselected in 48.86s`; changed-file Ruff and `git diff --check`
were clean at that checkpoint.

An established pass-through/oversized/fresh-validation/transform selection
then exposed one stricter-validator fixture wiring omission: three passed and
the synthetic three-modality case reported `image_codec_unavailable`. Supplying
that fixture's already injected matching codec produced GREEN:
`1 passed, 150 deselected in 4.81s`.

Review-fix self-review confirmed that build identity is captured before
`configuration_id` and `begin_run`; local bomb dispositions remain distinct
from hosted-attempt failures; every successful derived row declares and decodes
as JPEG; directory-entry fsync ordering precedes deeper publication; final leaf
verification compares descriptor and path identity; and orphan traversal never
follows a rendition symlink. No schema shape changed, so schema version 10
remains correct.

Final fix-round verification (no full suite): changed-file Ruff passed; embedding
smoke returned `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
preprocessing smoke returned two articles, three successful inputs, one
duplicate, zero failures, and validation true; `run --help` retained its three
explicit roots, positive limit, bounded reprocess, and hosted authorization;
diff whitespace, tracked generated artifacts, stale current schema claims, and
relative operator-document links had no findings.

## Review-fix round 2

Directory-entry durability is now proven on every invocation, including replay
after a crash that occurred between `mkdir` and the first parent fsync. Each
descriptor-relative child is opened and identity-checked, then its parent is
fsynced whether the child was created now or already existed. Replay cannot
begin deeper publication or commit state before both namespace parents have
crossed that durability boundary.

Final publication now reanchors the complete namespace chain from the held
embedding-output-root descriptor. It reopens `renditions`, compares its held and
reopened directory device/inode, reopens the transform namespace and compares
again, then performs the full regular/single-link/stable-read/path-inode leaf
verification through that reanchored chain. Rename-and-replace races at either
ancestor level fail `unsafe_rendition_output` before run or work state.

Image provenance now records source and embedded frame counts, so the complete
policy remains independently checkable for archived generations whose source
bytes are intentionally not retained. Validation requires static source and
embedded inputs; supported within-limit sources must be exact pass-through;
derived input is permitted only when source bytes or pixels exceed the hosted
threshold; derived format is JPEG; and derived dimensions must exactly equal
the deterministic aspect scale. Coordinator and validator use the same pure
`scaled_image_dimensions` helper. Hash, byte, limit, transform, and path checks
remain mandatory. This physical shape change bumps schema 10 to 11 and adds an
exact prior-v10 refusal fixture without migration.

Review-fix round-2 RED/GREEN evidence:

- replay parent-fsync and `renditions`/transform ancestor replacement races:
  `3 failed, 151 deselected in 14.67s`, then
  `3 passed, 151 deselected in 7.87s`;
- exact schema v11 frame provenance first failed on the absent columns, then
  passed `1 passed, 153 deselected in 0.87s`;
- animated source/embedded, arbitrary rendition dimensions, and archived
  within-limit-derived policy: `4 failed, 154 deselected in 19.75s`, then
  `4 passed, 154 deselected in 18.49s`;
- schema v11/prior-v10 plus established pass-through/rendition/global-reference
  wiring: `5 passed, 154 deselected in 13.60s`.

Round-2 self-review verified fsync happens after safe child identity resolution,
the complete reanchor runs before descriptors close and before `begin_run`, and
ancestor races cannot be hidden by a still-valid held descriptor. Frame counts
are content-free and necessary for archived policy audit. The threshold and
dimension decision uses stored source facts for every generation, while current
generations additionally decode source and embedded bytes with the exact codec.

Final round-2 affected verification passed
`22 passed, 137 deselected in 68.88s`. Changed-file Ruff, embedding smoke,
preprocessing smoke, live `run --help`, diff whitespace, tracked generated
artifacts, stale current-schema claims, and both operator-document relative-link
sets were clean. No full suite, shared dependency change, network, credential,
licensed archive, or corpus operation was used.
