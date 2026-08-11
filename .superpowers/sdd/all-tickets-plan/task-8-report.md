# Task 8 implementation report: multimodal article embeddings

## Delivered

Eligible articles now publish three independently queryable 2,048-dimensional
modalities. `multimodal_article` is created locally only after current
`article_text` and `header_image` generations have both succeeded under the
same immutable configuration. The published vector is the float32,
L2-normalized equal-weight midpoint of the two verified normalized source
vectors. No hosted request is made for this derivation.

Schema version 7 adds append-only, content-free
`multimodal_embedding_provenance`. Each composite generation records its two
source generation run IDs, both stored-vector SHA-256 values, the immutable
formula version, and creation time. Source identity changes archive and remove
only the dependent same-article/configuration composite before replacement;
the unchanged source, sibling articles, and other configurations remain
unchanged. A replay with two proven sources and a missing composite purchases
only the composite.

Read-only validation resolves provenance against active or superseded
generations, checks formula/source linkage, and recomputes every active
composite. It detects a wrong source reference and a self-consistent vector/hash
replacement. Run coverage now counts active plus properly archived generations,
so a legitimate applicable-to-absent transition validates while a deleted
claimed generation remains visible.

## Schema and refusal contract

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from `6` to `7`.
- Added exactly one table with primary key
  `(article_id, configuration_id, generation_run_id)` and no indexes.
- Updated exact table/column/constraint tests, catalog creation/inspection,
  pipeline publication, read-only validation, README, and architecture docs.
- Added an exact prior-v6 refusal fixture by dropping only the new table and
  pinning metadata to version 6. The catalog is refused without migration or
  mutation. Versions 1 through 5 remain refused.
- No provenance or failure row stores Markdown, excerpts, image bytes, vectors,
  remote URLs, credentials, request bodies, or adapter responses.

## Red-green evidence

The initial focused selection was:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'all_three_modalities or composite_only_recovery or recomputes_composite or exact_version_seven or refuses_version_six'
```

Observed RED: `6 failed, 96 deselected in 12.95s`. The catalog lacked the v7
table/refusal boundary, eligible sources produced only two vectors, the
composite failpoint was never reached, and validation neither recomputed nor
linked composites. After the smallest schema/coordinator/validator change, the
same selection observed GREEN: `6 passed, 96 deselected in 21.89s`.

A valid applicable-to-absent fixture then observed RED:
`1 failed, 101 deselected in 7.30s`; run coverage treated archived image and
composite generations as missing. Counting active plus superseded generations,
while retaining the deleted-vector control, observed GREEN:
`2 passed, 100 deselected in 8.55s`.

The full fixture-only embedding module was used once to locate affected legacy
expectations and observed `100 passed, 2 failed in 196.89s`. Both failures were
test setup that changed all history rows after real composite history began
sharing a generation run. Scoping the intended corruption to its source row
observed `3 passed, 99 deselected in 13.49s`. Per direction, no broad rerun was
performed afterward.

## Final focused verification

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'all_three_modalities or composite_only_recovery or recomputes_composite or exact_version_seven or refuses_version_six or changed_header_bytes or removed_canonical_header or text_supersession or malformed_generation_modality_and_path or missing_vector_claimed or absent_then_added or missing_header_is_retryable or retryable_header_failure or interrupted_header_attempt'
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_cli.py \
  -k 'authorized_limited_run or authorized_reprocess_flag'
/home/willis/projects/finance_wsj/.venv/bin/ruff check \
  src/wsj_embeddings/catalog.py src/wsj_embeddings/pipeline.py \
  src/wsj_embeddings/validate.py tests/test_embeddings_pipeline.py \
  tests/test_embeddings_cli.py
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings smoke
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings run --help
git diff --check
git ls-files | rg '^(data|artifacts|outputs)/'
```

Observed:

- affected multimodal/lifecycle/schema/validation fixtures:
  `18 passed, 84 deselected in 84.46s`;
- affected public CLI fixtures: `2 passed, 17 deselected in 9.37s`;
- changed Python Ruff: `All checks passed!`;
- generated smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- live module help retained the three explicit roots, required positive
  `--limit`, bounded `--reprocess`, and hosted-processing authorization;
- `git diff --check` produced no output; and
- the tracked generated-artifact audit produced no matches (expected `rg`
  exit 1).

No licensed archive, network request, credential, real embedding output, or
full-corpus operation was used. No full repository test suite was run.

## Self-review

- Eligibility is proven by exact-configuration joins from both successful work
  checkpoints to their matching active embeddings. Each source vector is
  finite, unit length, dimension 2,048, and hash-matching before derivation.
- The formula independently normalizes each persisted float32 source, averages
  it at equal weight, normalizes the midpoint, and float32-quantizes the result.
  The high-level fixture hand-checks the orthogonal result as two leading
  `0.7071067690849304` values.
- Composite input identity contains only formula and source generation/hash
  facts. Provenance is inserted in the same transaction as the composite
  vector/checkpoint and is never updated or deleted during supersession.
- Missing, absent, retryable, terminal, interrupted, stale, or unmatched source
  work cannot enter the successful-source join. Normal source invalidation
  archives/removes its dependent before any replacement adapter request.
- Composite-only recovery reuses both immutable source generation IDs, performs
  zero text/image adapter calls, and publishes one new composite generation.
- Validation is read-only, resolves active and historical links, verifies
  content-free hashes/formula, and compares exact recomputed float32 values and
  hash. Existing intrinsic vector validation remains independent.
- Run summaries count locally derived composite publications as successful
  embeddings but not hosted attempts. Reuse counts each independently proven
  source or composite generation.

## Concerns

None. The shared virtual environment's installed console script is not linked
to this worktree, so smoke and live help were verified against worktree source
with `PYTHONPATH=src`. No machine configuration changed.
