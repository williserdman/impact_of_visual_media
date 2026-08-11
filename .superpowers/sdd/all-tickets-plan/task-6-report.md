# Task 6 implementation report: local header-image embeddings

## Delivered

The bounded embedding coordinator now reads each canonical local
`header_image_path` below the immutable source root through descriptor-relative,
no-follow, replacement-detecting access. It hashes the exact accepted bytes and
base64-encodes those same bytes for the adapter. The production Jina adapter
uses a single image request item and never receives or constructs a remote image
URL. Remote `inline_image_urls` are not selected by the coordinator.

Text and header images have separate `(article_id, modality,
configuration_id)` work checkpoints and active vectors. A successful
`header_image` row retains the source-relative path, exact source hash,
configuration identity, normalized 2,048-dimensional vector, and float32 vector
hash. An article without a local header-image path gets a durable
`not_applicable` image checkpoint with null path/hash and no image attempt or
placeholder vector. Adding a later local image reuses the unchanged successful
article-text vector and generates only the new image vector. Existing explicit
`--reprocess` remains article-text-only and reuses unchanged image work.

The generated coordinator fixture includes one local image plus a forbidden
remote URL. Through a recording fake adapter it proves one text vector, one
image vector, exact-byte base64 input, source/vector hashes, independent work
rows, and read-only validation. Traversal, absolute, and symlinked image paths
are rejected before embedding-output creation with content-free errors.

## Schema and refusal contract

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from `3` to `4`.
- Added nullable `source_relative_path` to `embedding_work_items`, `embeddings`,
  and `embedding_generation_history`.
- Made work-item `input_sha256` nullable for explicit `not_applicable` image
  work; successful vector/history input hashes remain non-null.
- Added exact schema-v4 tests and explicit version-3 refusal without migration,
  overwrite, repair, or deletion. Versions 1 and 2 remain refused.
- Updated catalog creation/validation, coordinator and catalog APIs, vector
  validation, test fixture SQL, README, and architecture crash course together.
- No table stores Markdown, article excerpts, image bytes, remote URLs,
  credentials, request bodies, or adapter responses.

## Red-green evidence

### Exact schema, coordinator, containment, and real adapter seam

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'exact_version_four or publishes_text_and_exact_local_header_image or absent_then_added_header_image or rejects_uncontained_header_image' \
  tests/test_jina_hosted_adapter.py \
  -k 'embeds_one_already_encoded_image or exact_version_four or publishes_text_and_exact_local_header_image or absent_then_added_header_image or rejects_uncontained_header_image'
```

Observed RED: `6 failed, 81 deselected in 5.73s`. The coordinator published
only article text, image work/path columns and `not_applicable` did not exist,
unsafe image paths were ignored, the catalog remained schema v3, and the public
Jina adapter had no `embed_image` seam.

After the smallest schema/coordinator/adapter implementation, the expanded
focused command including version-3 refusal observed GREEN:
`7 passed, 80 deselected in 10.86s`.

The added no-follow symlink fixture and the image/absence/containment cases then
observed `5 passed, 61 deselected in 6.54s`.

Self-review found that the meaning-bearing profile still labeled image
transformation `not-implemented-v1`. A focused identity assertion observed RED:
`1 failed, 65 deselected in 0.56s`. Production, fake, and default profiles now
pin `source-bytes-no-transform-v1`; GREEN was
`1 passed, 65 deselected in 0.49s`.

### Existing lifecycle regression follow-up

The first coordinator/hosted-adapter focused run exposed only legacy fixture
assumptions about one work row and direct insertion over the new durable
`not_applicable` key: `82 passed, 5 failed in 81.66s`. Queries were narrowed to
their intended article-text modality, generated seed SQL became an exact-key
upsert, and the dependent-invalidation fixture attached a real generated header
image. The affected checks then passed, including
`2 passed, 63 deselected in 18.47s` for both text-supersession cases.

The initial CLI run similarly exposed a deliberately catalogued but unwritten
fixture header path: `7 passed, 10 failed in 5.59s`, all with the expected new
`missing_header_image` classification. The generated fixture now creates its
declared local image and asserts two first-run modalities plus image reuse on
article-text reprocess. GREEN: `17 passed in 12.25s`.

## Final focused verification

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py tests/test_jina_hosted_adapter.py
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_cli.py
/home/willis/projects/finance_wsj/.venv/bin/ruff check \
  src/wsj_embeddings tests/test_embeddings_pipeline.py \
  tests/test_jina_hosted_adapter.py tests/test_embeddings_cli.py
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings smoke
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings run --help
git diff --check
git ls-files | rg '^(data|artifacts|outputs)/'
```

Observed:

- focused coordinator/catalog/hosted-adapter suite: `88 passed in 95.07s`;
- focused public CLI suite: `17 passed in 12.25s`;
- changed Python lint: `All checks passed!`;
- generated smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- live module help lists the required positive `--limit`, optional
  article-text `--reprocess`, three explicit roots, and explicit hosted
  authorization;
- `git diff --check` produced no output; and
- the generated-artifact tracked-path audit produced no matches (expected
  `rg` exit 1).

No command used the licensed archive, a live credential, network access, a real
embedding output, or full-corpus scope. Per task direction, the full suite was
not run.

## Self-review

- The coordinator reads selected image bytes before output mutation, and path
  containment/no-follow/replacement checks fail closed without exposing bytes.
- Hashing and base64 encoding use the same in-memory byte object; Jina receives
  only the base64 request item, never a URL.
- Text and image registration, in-progress, success, failure, reuse, and
  generation keys are independent. Image addition/reuse cannot reset or call
  the successful text work item.
- `not_applicable` has no input identity, attempt, failure metadata, or vector;
  validation accepts it only for header-image work with those fields absent.
- Active image vectors and superseded provenance retain only relative paths and
  hashes, not source bytes. Errors and validation issues remain content-free.
- The coordinator query selects `header_image_path` but never
  `inline_image_urls`; the acceptance fixture leaves a remote URL in the
  preprocessing row and observes exactly one local image call.
- Schema v4 is exact and index-free. Existing output is inspected read-only
  before writable open; v1-v3 output is not migrated or overwritten.
- Complete missing/retryable/terminal/stale-image transition scenarios remain
  intentionally scoped to Task 7 / Ticket 08, as directed. This ticket supplies
  the absent state and independent checkpoint/generation foundation they need.

## Concerns

None. The shared virtual environment's installed console script is not linked
to this worktree, so smoke and live help were verified against worktree source
with `PYTHONPATH=src`. No machine configuration changed.

## Fix round 1: immutable generation provenance and validation hardening

### Review findings addressed

- `embedding_work_items.generation_run_id` now records the run that actually
  generated an active successful vector. `last_run_id` remains mutable
  observation/attempt bookkeeping. Reuse updates only `last_run_id`; success
  sets both fields; invalidation, attempts, failures, interruption, and
  `not_applicable` clear generation identity. Supersession history reads the
  immutable generation field.
- Schema version is now 5. Exact schema tests and explicit schema-v4 refusal
  extend the no-migration contract; versions 1 through 4 are refused.
- Read-only validation requires a valid SHA-256 for every applicable work row,
  permits `source_relative_path` only for `header_image`, and syntactically
  rejects absolute, traversal, and HTTP(S) paths. `not_applicable` requires a
  header-image key with null path/hash/generation, zero attempts, no failure
  metadata, and no active vector. Successful work requires a valid generating
  run, while non-successful work may not retain one.
- Generation history now validates supported modality plus its conditional
  source-path contract: header images require a safe archive-relative path and
  other modalities require null.
- Direct generated fixtures prove replacement-after-open detection and
  hard-link rejection through the existing stable-byte image reader, retaining
  the domain-specific `unsafe_header_image_path` coordinator error.

### RED and GREEN

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'version_five or refuses_version_four or image_supersession_after_reuse or header_image_replacement_after_safe_open or hard_linked_header_image or missing_or_malformed_active_work_hash or invalid_work_and_vector_source_paths or vector_for_not_applicable or malformed_generation_modality_and_path'
```

Observed RED: `11 failed, 4 passed, 65 deselected in 26.34s`. The replacement
and hard-link cases already failed closed through the stable reader, while the
new schema/provenance field and malformed work/history contracts were absent.
One article-text path case also passed because that narrower invariant already
existed.

After schema v5, immutable generation transitions, shared syntactic path
recognition, and conditional validation, the same focused selection observed
GREEN: `15 passed, 65 deselected in 29.62s`.

Additional malformed fixtures for forbidden `not_applicable` generation/error
metadata and missing/unknown successful generation runs observed
`3 passed, 80 deselected in 5.57s`.

The provenance fixture generates an image, reuses it unchanged in a later run,
then changes the exact bytes. It proves `last_run_id` advances on reuse while
`generation_run_id` does not, and the later history row attributes the archived
image vector to its true first generating run rather than the replay.

### Fix-round focused verification

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'reports_run_without_configuration_reference or image_supersession_after_reuse or version_five or malformed_generation_modality_and_path'
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_cli.py
/home/willis/projects/finance_wsj/.venv/bin/ruff check \
  src/wsj_embeddings tests/test_embeddings_pipeline.py \
  tests/test_embeddings_cli.py
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings smoke
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings run --help
git diff --check
git ls-files | rg '^(data|artifacts|outputs)/'
```

Observed:

- the focused coordinator/catalog/validation module reached
  `82 passed, 1 failed in 137.97s`; the sole failure was an existing exact
  assertion that now correctly receives both the missing run-configuration
  issue and the consequential invalid successful generation checkpoint;
- after updating that expected additive diagnosis, the affected provenance,
  schema, history, and run-reference selection was
  `6 passed, 77 deselected in 15.62s`;
- focused CLI: `17 passed in 14.42s`;
- changed Python Ruff: `All checks passed!`;
- generated smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- live `run --help` retained the required roots, positive limit, optional
  article-text reprocess, and explicit hosted authorization;
- `git diff --check` produced no output; current-facing README/crash-course
  searches found no stale schema-v4 claims; and the tracked generated-artifact
  audit produced no matches (expected `rg` exit 1).

No full repository suite, licensed archive, live credential, network request,
or full-corpus operation was used.

### Fix-round self-review

- Generation identity is written only in the same transaction that publishes
  vector success. Reuse updates observation metadata without changing it.
- Every transition that removes or cannot prove an active success clears the
  generation field. History reads only that field, never mutable
  `last_run_id`.
- Validator run-identity checks are configuration-scoped and diagnose both a
  broken run reference and any success that consequently lacks valid
  generation provenance.
- Applicable work hashes are syntactically exact lowercase SHA-256 values.
  Absent work is the only nullable-hash state and cannot retain a vector,
  attempt, failure, source path, or generation run.
- The shared syntactic image-path predicate performs no I/O and rejects
  absolute, traversal, and HTTP(S) identities. The stable-byte reader still
  owns descriptor-relative no-follow/replacement checks and domain-specific
  errors.
- Generation history accepts only the domain modality vocabulary, requires a
  safe local path for header images, and forbids paths for text/multimodal
  generations. It still stores no bodies, bytes, vectors, URLs, or exception
  text.

Fix-round concerns: none.
