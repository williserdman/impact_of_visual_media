# Task 7 implementation report: header-image eligibility and retries

## Delivered

Header-image work now has an explicit, content-free disposition for every
selected article and configuration. A true no-header article is
`not_applicable`; a readable header progresses through `queued`, `in_progress`,
and `succeeded`; hosted/transport failures remain `retryable`; deterministic
image-request rejection becomes `terminal` on its third attempt; and an
interrupted request remains recoverable. Changed exact bytes archive the prior
image generation with `input_changed`, invalidate only that header vector and
its same-article/configuration multimodal dependent, and preserve successful
text, sibling articles, and prior configurations.

A declared but unavailable local header is the only applicable work allowed to
have a null input hash. It is a narrowly validated `retryable` checkpoint with
the safe canonical relative path, `missing_header_image`, zero adapter attempts,
no status/retry metadata, no generation identity, and no active vector. It does
not fail or regenerate independently successful text. Creating the file later
queues and publishes only the image vector.

Image adapter failures no longer abort a completed text publication. Public
coordinator and CLI summaries include `header_absent` and `header_failed`, so a
valid no-header article is not conflated with missing, in-progress, retryable,
or terminal image work. Read-only validation emits stable content-free issue
codes for each incomplete/failed image disposition and remains silent for a
valid `not_applicable` image.

## Schema and refusal contract

- Bumped `EMBEDDING_CATALOG_SCHEMA_VERSION` from `5` to `6`.
- Added non-null integer `header_absent` and `header_failed` columns to `runs`.
- Updated exact schema creation/inspection, run initialization, counter
  whitelisting, result reads, coordinator writes, CLI JSON, tests, README, and
  architecture documentation together.
- Added an exact prior-v5 refusal fixture by generating a catalog, removing only
  the v6 run columns, and pinning metadata to version 5. It is refused read-only
  without migration, overwrite, repair, or deletion. Versions 1 through 4
  remain refused.
- No table stores Markdown, article excerpts, image bytes, vectors in failure
  rows, remote URLs, credentials, request bodies, or adapter responses.

## Red-green evidence

The initial generated lifecycle/schema selection was:

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'exact_version_six or refuses_version_five or missing_header_is_retryable or retryable_header_failure or deterministic_header_rejection or changed_header_bytes or interrupted_header_attempt'
```

Observed RED: `6 failed, 1 passed, 82 deselected in 15.72s`. Missing input
aborted before output, image adapter failures escaped the coordinator,
deterministic rejection became terminal on the first attempt, image-byte
changes left the same-configuration multimodal vector active, interrupted image
work was invisible to validation, and the catalog remained schema v5. The one
pass was the first weak metadata-only refusal fixture.

That refusal fixture was strengthened to transform a generated current catalog
into the exact prior shape. Before the bump it observed the intended RED:
`1 failed, 88 deselected in 2.88s`, because version 5 remained accepted.

After the smallest schema/catalog/coordinator/validation implementation, the
same seven-case selection observed GREEN:
`7 passed, 82 deselected in 21.85s`.

The broader coordinator run exposed eight existing exact result assertions that
needed the new `header_absent` field. After updating those generated-fixture
expectations, the affected rerun was `8 passed, 81 deselected in 17.49s`.
The public CLI module, including generated absent-vs-missing nonzero summary
fixtures, then passed `19 passed in 16.22s`.

## Final focused verification

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py tests/test_embeddings_cli.py
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

- focused coordinator/catalog/validation/CLI tests:
  `108 passed in 193.46s`;
- changed Python Ruff: `All checks passed!`;
- generated smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- live module help retained the three explicit roots, required positive
  `--limit`, article-text-only `--reprocess`, and explicit hosted authorization;
- `git diff --check` produced no output;
- the initial current-facing documentation search produced no matches because
  its pattern missed the hyphenated `schema-version-5` wording later corrected
  in fix round 1; and
- the tracked generated-artifact audit produced no matches (expected `rg`
  exit 1).

No full repository suite, licensed archive, live credential, network request,
real embedding output, or full-corpus operation was used.

## Self-review

- Missing input is classified before output mutation but becomes durable only
  under the coordinator lock. Unsafe, symlinked, hard-linked, and replaced
  source paths retain their existing fail-closed behavior and never become the
  missing exception.
- The missing exception is modality-, state-, error-, path-, hash-, attempt-,
  generation-, and vector-specific. General applicable work still requires an
  exact lowercase SHA-256 identity.
- Text success commits before image work. Image failure commits only stable
  codes and numeric metadata, increments a failed-header summary, and continues
  without a placeholder vector. Text adapter failures retain their existing
  fail-fast behavior.
- Provider-declared retryability is kept separate from the explicit
  deterministic image policy. Only `deterministic_request` receives the
  bounded three-attempt transition; other nonretryable failures remain
  terminal immediately.
- Exact-byte changes reset the affected image attempt lifecycle, archive the
  active image and multimodal generations, and never touch article text or a
  different configuration. Configuration changes retain independent old and
  new work keys.
- A success commit writes the vector, immutable `generation_run_id`, work state,
  and run counters together. Recovery reuses unchanged text and image successes
  and purchases only incomplete image work.
- Run summaries account for every selected header as absent, successful, or
  failed without storing identities or content in the summary.

## Concerns

None. The shared virtual environment's installed console script is not linked
to this worktree, so smoke and live help were verified against worktree source
with `PYTHONPATH=src`. No machine configuration changed.

## Fix round 1: archive-root and disposition integrity

### Review findings addressed

- The coordinator now opens the configured archive root as a no-follow
  directory before reading the preprocessing catalog, preparing items, calling
  the adapter, or creating embedding output. Absent, non-directory, and
  inaccessible roots raise content-free `source_root`; only a missing optional
  leaf below a proven root becomes `missing_header_image`.
- When a canonical header becomes absent, one existing registration transaction
  archives/removes both its same-configuration header generation and multimodal
  dependent, resets their active checkpoints, then records the header as
  `not_applicable`. Successful text, other articles, and other configurations
  remain untouched.
- Read-only validation derives required header disposition keys from canonical
  articles selected by article-text work. A deleted header checkpoint is now
  visible even when it had no vector. The latest configuration run's
  `header_absent` and `header_failed` claims are reconciled against durable
  header states; invalid lifecycle states retain their primary diagnosis rather
  than causing a derivative counter finding.
- Schema remains version 6 because no table, column, type, constraint, or index
  changed. The stale README `schema-version-5` label is corrected, and the
  original report now records why its initial narrow search missed that
  hyphenated wording.

### RED and GREEN

The invalid-root fixture observed RED:
`3 failed, 89 deselected in 3.08s`. An absent root completed as a retryable
missing leaf, while non-directory and inaccessible roots surfaced as unsafe
item paths. After the root-open gate it observed GREEN:
`3 passed, 89 deselected in 1.43s`.

The applicable-to-absent fixture observed RED:
`1 failed, 92 deselected in 7.70s`, because the current-configuration
multimodal vector remained active. After dependent invalidation it observed
GREEN: `1 passed, 92 deselected in 10.35s`.

The deleted-checkpoint and run-counter fixtures observed RED:
`4 failed, 93 deselected in 7.78s`. After explicit required-key coverage and
latest-run count reconciliation they observed GREEN:
`4 passed, 93 deselected in 6.42s`.

An affected lifecycle selection then exposed one parameter-list defect in the
new terminal-observation update. The last-failed rerun observed
`2 passed, 95 deselected in 4.25s` after correcting that call. No unrelated
production behavior changed.

### Fix-round focused verification

```bash
/home/willis/projects/finance_wsj/.venv/bin/python -m pytest -q \
  tests/test_embeddings_pipeline.py \
  -k 'invalid_archive_root or removed_canonical_header or missing_header_disposition_checkpoint or header_disposition_counter_mismatch or absent_then_added_header_image or missing_header_is_retryable or retryable_header_failure or deterministic_header_rejection or changed_header_bytes or interrupted_header_attempt or validator_accepts_fresh or validator_reports_invalid_durable_work_state or terminal_failure_is_visible'
/home/willis/projects/finance_wsj/.venv/bin/ruff check \
  src/wsj_embeddings/catalog.py src/wsj_embeddings/pipeline.py \
  src/wsj_embeddings/validate.py tests/test_embeddings_pipeline.py
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings smoke
PYTHONPATH=src /home/willis/projects/finance_wsj/.venv/bin/python \
  -m wsj_embeddings run --help
git diff --check
git ls-files | rg '^(data|artifacts|outputs)/'
```

Observed:

- affected coordinator/catalog/validation selection:
  `17 passed, 80 deselected in 39.36s`;
- changed Python Ruff: `All checks passed!`;
- generated smoke:
  `{"articles": 1, "embeddings": 1, "validation_ok": true}`;
- live help retained its required bounded/authorized interface;
- `git diff --check` and the stale current-doc claim search produced no output;
  and
- the tracked generated-artifact audit produced no matches (expected `rg`
  exit 1).

Per review direction, no broad or full suite was run. No licensed archive, live
credential, network request, real embedding output, or full-corpus operation
was used.

### Fix-round self-review

- Root validation uses the same no-follow directory-open flags as the output
  anchor and closes the descriptor immediately. The check precedes both
  preprocessing reads and all output mutation.
- Missing-leaf classification remains unchanged after the root is proven;
  unsafe relative paths, symlinks, hard links, and replacement races retain
  their fail-closed classifications.
- Applicable-to-absent invalidation is bounded by the existing registration
  transaction and configuration key. Both superseded generations keep their
  immutable provenance before active rows are removed.
- Required header coverage follows actual bounded selection through
  article-text work keys rather than incorrectly requiring the whole canonical
  corpus after a limited run.
- Counter reconciliation is limited to the latest run for the chosen
  configuration because older work rows legitimately move their `last_run_id`
  on replay. Terminal replay now records that observation without changing its
  immutable generation identity or retry state.
- Corrupt/unknown work states are diagnosed by the lifecycle validator and are
  excluded from derivative counter reconciliation.

Fix-round concerns: none.
